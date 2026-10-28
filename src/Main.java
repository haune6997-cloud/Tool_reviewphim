import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.TableModelEvent;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.dnd.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.prefs.Preferences;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Auto Movie Review Studio Pro v3.1
 * Yêu cầu: Java 11+, ffmpeg, Python AI Server (ai_service.py).
 */
public class Main extends JFrame {

    private static final String APP_TITLE = "Auto Movie Review Studio Pro - AI Edition v3.1";
    private static final String AI_SERVER_URL = System.getProperty("ai.url", "http://127.0.0.1:8000");
    private static final String[] EDGE_VOICES = {"vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"};
    private static final Pattern TIME_PAT = Pattern.compile("^(?:(\\d{1,2}):)?([0-5]?\\d):([0-5]\\d(?:\\.\\d{1,3})?)$");
    private static final int COL_META = 6;

    // ---------- Thành phần giao diện (chỉ thao tác trên EDT) ----------
    private JTextField txtVideoPath, txtBgmPath, txtOutputPath;
    private JRadioButton rbAuto, rbTimestamp;
    private JComboBox<String> cbAspectMode, cbMirrorMode, cbSubColor, cbSubSize, cbTtsEngine, cbTtsVoice, cbOrderStrict;
    private JCheckBox cbGpuNvenc, cbBurnSubtitles, cbKeepOriginalAudio, cbKeepTempFiles, cbExportSrt, cbDucking, cbUseOllama,
            cbAllowNoMatch, cbAllowFallback, cbRefine;
    private JSpinner spOrigAudioVol, spBgmVol, spTtsRate, spMaxScenes, spRematchWindow;
    private JTextArea txtScript, txtLog;
    private JTable timelineTable;
    private DefaultTableModel tableModel;
    private JTabbedPane tabbedPane;
    private JProgressBar progressBar;
    private JLabel lblStatus;
    private JButton btnRun, btnCancel, btnParseTimeline, btnAiMatch, btnRematch;
    private boolean updatingTable = false;

    // ---------- Trạng thái đồng bộ ----------
    private final AtomicBoolean cancelRequested = new AtomicBoolean(false);
    private final AtomicBoolean busy = new AtomicBoolean(false);
    private final AtomicBoolean edgeTtsBroken = new AtomicBoolean(false);
    private final Set<Process> activeProcesses = ConcurrentHashMap.newKeySet();
    private final Preferences prefs = Preferences.userNodeForPackage(Main.class);
    private volatile String tableScriptHash = "";
    private volatile String currentJobId = "";
    private static final String[] ORDER_MODE_NAMES = {"loose", "medium", "strict", "free"};

    // ---------- Mô hình dữ liệu ----------
    private static class JobConfig {
        String videoPath, bgmPath, outputPath, rawScript;
        boolean isAutoMode, useGpuNvenc, burnSubtitles, keepOriginalAudio, keepTempFiles, exportSrt, ducking, useOllama;
        int aspectMode, mirrorMode, subSizeIdx, ttsEngine, ttsVoice, ttsRatePercent, maxScenes, orderStrict;
        String subColorStyle;
        double origAudioVolume, bgmVolume;
        List<SceneItem> scenes;
        AiOpts ai;
    }

    private static class AiOpts {
        int maxScenes, strict;
        boolean ollama, allowNoMatch, allowFallback, refine;
    }

    /** Một ứng viên cảnh do AI đề xuất cho một câu. */
    public static class Cand {
        String sceneId = "", conf = "", caption = "", status = "review";
        double start, end, focus = Double.NaN, actionStart = Double.NaN, actionEnd = Double.NaN, score, raw;
    }

    public static class SceneItem {
        int index;
        double startSec, endSec, boundaryStart, boundaryEnd;
        double focusSec = Double.NaN, actionStart = Double.NaN, actionEnd = Double.NaN;
        String text, aiInfo;
        /** ok | review | low | abstract (câu trừu tượng, cảnh chọn theo mạch) | user (người dùng chọn lại) | manual (sửa tay) */
        String status = "ok", sceneId = "";
        List<Cand> cands = new ArrayList<>();          // ứng viên thay thế do AI trả về
        List<double[]> tried = new ArrayList<>();      // các đoạn đã thử/loại, không đề xuất lại

        SceneItem(int index, double s, double e, double bs, double be, String text, String aiInfo) {
            this.index = index; this.startSec = s; this.endSec = e;
            this.boundaryStart = bs; this.boundaryEnd = be; this.text = text; this.aiInfo = aiInfo;
        }
    }

    private static class MediaInfo {
        final double duration; final boolean hasVideo, hasAudio;
        MediaInfo(double d, boolean v, boolean a) { duration = d; hasVideo = v; hasAudio = a; }
    }

    private static class SubEntry {
        final double start, end; final String text;
        SubEntry(double s, double e, String t) { start = s; end = e; text = t; }
    }

    private static class HttpResult {
        final int code; final byte[] body;
        HttpResult(int c, byte[] b) { code = c; body = b; }
        String text() { return new String(body, StandardCharsets.UTF_8); }
    }

    // =====================================================================
    //  JSON tối giản (parse + escape) - thay cho regex mong manh
    // =====================================================================
    private static final class Json {
        private final String s;
        private int p;
        private Json(String s) { this.s = s; }

        static Object parse(String text) {
            Json j = new Json(text);
            j.ws();
            Object v = j.value();
            j.ws();
            if (j.p != j.s.length()) throw new IllegalArgumentException("JSON thừa dữ liệu tại " + j.p);
            return v;
        }

        static String quote(String str) {
            StringBuilder sb = new StringBuilder("\"");
            for (int i = 0; i < str.length(); i++) {
                char c = str.charAt(i);
                switch (c) {
                    case '"': sb.append("\\\""); break;
                    case '\\': sb.append("\\\\"); break;
                    case '\n': sb.append("\\n"); break;
                    case '\r': sb.append("\\r"); break;
                    case '\t': sb.append("\\t"); break;
                    default:
                        if (c < 0x20) sb.append(String.format("\\u%04x", (int) c)); else sb.append(c);
                }
            }
            return sb.append('"').toString();
        }

        @SuppressWarnings("unchecked")
        static Map<String, Object> map(Object o) { return o instanceof Map ? (Map<String, Object>) o : new LinkedHashMap<>(); }
        @SuppressWarnings("unchecked")
        static List<Object> list(Object o) { return o instanceof List ? (List<Object>) o : new ArrayList<>(); }
        static double num(Map<String, Object> m, String k, double def) { Object v = m.get(k); return v instanceof Number ? ((Number) v).doubleValue() : def; }
        static String str(Map<String, Object> m, String k, String def) { Object v = m.get(k); return v instanceof String ? (String) v : def; }

        private void ws() { while (p < s.length() && Character.isWhitespace(s.charAt(p))) p++; }
        private char peek() { if (p >= s.length()) throw new IllegalArgumentException("JSON kết thúc đột ngột"); return s.charAt(p); }

        private Object value() {
            ws();
            char c = peek();
            if (c == '{') return obj();
            if (c == '[') return arr();
            if (c == '"') return str();
            if (s.startsWith("true", p)) { p += 4; return Boolean.TRUE; }
            if (s.startsWith("false", p)) { p += 5; return Boolean.FALSE; }
            if (s.startsWith("null", p)) { p += 4; return null; }
            return number();
        }

        private Map<String, Object> obj() {
            Map<String, Object> m = new LinkedHashMap<>();
            p++; ws();
            if (peek() == '}') { p++; return m; }
            while (true) {
                ws();
                String k = str();
                ws();
                if (peek() != ':') throw new IllegalArgumentException("Thiếu ':' tại " + p);
                p++;
                m.put(k, value());
                ws();
                char c = peek(); p++;
                if (c == '}') return m;
                if (c != ',') throw new IllegalArgumentException("Thiếu ',' tại " + p);
            }
        }

        private List<Object> arr() {
            List<Object> l = new ArrayList<>();
            p++; ws();
            if (peek() == ']') { p++; return l; }
            while (true) {
                l.add(value());
                ws();
                char c = peek(); p++;
                if (c == ']') return l;
                if (c != ',') throw new IllegalArgumentException("Thiếu ',' trong mảng tại " + p);
            }
        }

        private String str() {
            if (peek() != '"') throw new IllegalArgumentException("Cần chuỗi tại " + p);
            p++;
            StringBuilder sb = new StringBuilder();
            while (true) {
                char c = peek(); p++;
                if (c == '"') return sb.toString();
                if (c != '\\') { sb.append(c); continue; }
                char e = peek(); p++;
                switch (e) {
                    case 'n': sb.append('\n'); break;
                    case 't': sb.append('\t'); break;
                    case 'r': sb.append('\r'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'u': sb.append((char) Integer.parseInt(s.substring(p, p + 4), 16)); p += 4; break;
                    default: sb.append(e);
                }
            }
        }

        private Number number() {
            int st = p;
            while (p < s.length() && "+-0123456789.eE".indexOf(s.charAt(p)) >= 0) p++;
            if (st == p) throw new IllegalArgumentException("Ký tự lạ '" + s.charAt(p) + "' tại " + p);
            return Double.parseDouble(s.substring(st, p));
        }
    }

    // =====================================================================
    //  KHỞI TẠO GIAO DIỆN
    // =====================================================================
    public Main() {
        setTitle(APP_TITLE);
        setSize(1120, 780);
        setMinimumSize(new Dimension(980, 700));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        initUI();
        loadSettings();
        addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent e) {
                saveSettings();
                cancelRequested.set(true);
                for (Process p : activeProcesses) killProcessTree(p);
            }
        });
        checkSystemStartup();
    }

    private void initUI() {
        JPanel mainPanel = new JPanel(new BorderLayout(8, 8));
        mainPanel.setBorder(new EmptyBorder(8, 10, 8, 10));

        // ----- 1. File -----
        JPanel filePanel = new JPanel(new GridBagLayout());
        filePanel.setBorder(BorderFactory.createTitledBorder("1. Nguồn Video, Nhạc Nền & Nơi Lưu (có thể kéo-thả file vào ô)"));
        txtVideoPath = new JTextField();
        txtBgmPath = new JTextField();
        txtOutputPath = new JTextField(new File("output_review.mp4").getAbsolutePath());
        addFileRow(filePanel, 0, "Video phim gốc (*):", txtVideoPath, "Chọn Video...", false);
        addFileRow(filePanel, 1, "Nhạc nền BGM (tùy chọn):", txtBgmPath, "Chọn Nhạc...", false);
        addFileRow(filePanel, 2, "Lưu video đầu ra (*):", txtOutputPath, "Chọn Nơi Lưu...", true);
        mainPanel.add(filePanel, BorderLayout.NORTH);

        tabbedPane = new JTabbedPane();
        tabbedPane.addTab("1. Kịch Bản", buildScriptTab());
        tabbedPane.addTab("2. Timeline", buildTimelineTab());
        tabbedPane.addTab("3. Cài Đặt", new JScrollPane(buildSettingsTab()));
        tabbedPane.addTab("4. Nhật Ký", buildLogTab());
        mainPanel.add(tabbedPane, BorderLayout.CENTER);

        // ----- Thanh dưới -----
        JPanel bottom = new JPanel(new BorderLayout(10, 5));
        lblStatus = new JLabel("Sẵn sàng.");
        lblStatus.setForeground(Color.BLUE);
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        JPanel sp = new JPanel(new BorderLayout(5, 3));
        sp.add(lblStatus, BorderLayout.NORTH);
        sp.add(progressBar, BorderLayout.CENTER);

        btnCancel = new JButton("⏹ HỦY");
        btnCancel.setEnabled(false);
        btnCancel.addActionListener(e -> requestCancel());
        btnRun = new JButton("🚀 BƯỚC 2: TẠO VIDEO");
        btnRun.setFont(btnRun.getFont().deriveFont(Font.BOLD, 13f));
        btnRun.setForeground(new Color(0, 102, 51));
        btnRun.addActionListener(e -> startWorkflowFromEDT());
        JPanel ap = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        ap.add(btnCancel);
        ap.add(btnRun);
        bottom.add(sp, BorderLayout.CENTER);
        bottom.add(ap, BorderLayout.EAST);
        mainPanel.add(bottom, BorderLayout.SOUTH);
        add(mainPanel);
    }

    private void addFileRow(JPanel p, int row, String label, JTextField field, String btnText, boolean save) {
        GridBagConstraints g = new GridBagConstraints();
        g.fill = GridBagConstraints.HORIZONTAL;
        g.insets = new Insets(2, 5, 2, 5);
        g.gridy = row;
        g.gridx = 0; g.weightx = 0;
        p.add(new JLabel(label), g);
        g.gridx = 1; g.weightx = 1;
        p.add(field, g);
        g.gridx = 2; g.weightx = 0;
        JButton b = new JButton(btnText);
        b.addActionListener(e -> selectFile(field, save));
        p.add(b, g);
        enableFileDrop(field);
    }

    @SuppressWarnings("unchecked")
    private void enableFileDrop(JTextField field) {
        field.setDropTarget(new DropTarget(field, new DropTargetAdapter() {
            @Override public void drop(DropTargetDropEvent e) {
                try {
                    e.acceptDrop(DnDConstants.ACTION_COPY);
                    List<File> files = (List<File>) e.getTransferable().getTransferData(DataFlavor.javaFileListFlavor);
                    if (!files.isEmpty()) field.setText(files.get(0).getAbsolutePath());
                    e.dropComplete(true);
                } catch (Exception ex) { e.dropComplete(false); }
            }
        }));
    }

    private JPanel buildScriptTab() {
        JPanel tab = new JPanel(new BorderLayout(5, 5));
        tab.setBorder(new EmptyBorder(5, 5, 5, 5));

        JPanel mode = new JPanel(new FlowLayout(FlowLayout.LEFT, 12, 0));
        rbAuto = new JRadioButton("Auto Cut bằng AI (nhập văn bản thường, AI tự chọn cảnh)", true);
        rbTimestamp = new JRadioButton("Timestamp Cut (mỗi dòng: MM:SS - MM:SS | Lời bình)");
        ButtonGroup bg = new ButtonGroup();
        bg.add(rbAuto); bg.add(rbTimestamp);
        mode.add(rbAuto); mode.add(rbTimestamp);
        tab.add(mode, BorderLayout.NORTH);

        txtScript = new JTextArea(8, 60);
        txtScript.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        txtScript.setLineWrap(true);
        txtScript.setWrapStyleWord(true);
        txtScript.setText("Bạn có tin một món đồ chơi nhỏ bé có thể thay đổi cả thế giới? "
                + "Trong tương lai, những robot chiến đấu mang tên LBX trở thành trò chơi nổi tiếng trên toàn cầu. "
                + "Các robot này chiến đấu bên trong những chiếc hộp đặc biệt, được gọi là chiến trường LBX.");
        tab.add(new JScrollPane(txtScript), BorderLayout.CENTER);

        JPanel actions = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 2));
        btnAiMatch = new JButton("🤖 BƯỚC 1: AI TỰ ĐỘNG CHỌN CẢNH");
        btnAiMatch.setFont(btnAiMatch.getFont().deriveFont(Font.BOLD, 12f));
        btnAiMatch.setForeground(new Color(102, 0, 153));
        btnAiMatch.addActionListener(e -> runAiSceneMatchingUI());
        btnParseTimeline = new JButton("📋 Nạp Timeline Thủ Công");
        btnParseTimeline.addActionListener(e -> parseScriptToTimeline());
        JButton btnClear = new JButton("🗑️ Xóa Timeline");
        btnClear.addActionListener(e -> { tableModel.setRowCount(0); tableScriptHash = ""; });
        actions.add(btnAiMatch); actions.add(btnParseTimeline); actions.add(btnClear);
        tab.add(actions, BorderLayout.SOUTH);
        return tab;
    }

    private JPanel buildTimelineTab() {
        JPanel tab = new JPanel(new BorderLayout(5, 5));
        tab.setBorder(new EmptyBorder(5, 5, 5, 5));
        String[] cols = {"Cảnh #", "Bắt Đầu", "Kết Thúc", "Độ Dài (s)", "Lời Bình Luận (TTS)", "Độ Tin Cậy AI & Mô Tả", "meta"};
        tableModel = new DefaultTableModel(cols, 0) {
            @Override public boolean isCellEditable(int r, int c) { return c == 1 || c == 2 || c == 4; }
        };
        timelineTable = new JTable(tableModel);
        timelineTable.setRowHeight(26);
        int[] widths = {50, 80, 80, 75, 430, 300};
        for (int i = 0; i < widths.length; i++) timelineTable.getColumnModel().getColumn(i).setPreferredWidth(widths[i]);
        timelineTable.removeColumn(timelineTable.getColumnModel().getColumn(COL_META));
        DefaultTableCellRenderer conf = new DefaultTableCellRenderer() {
            @Override public Component getTableCellRendererComponent(JTable t, Object v, boolean sel, boolean foc, int r, int c) {
                Component comp = super.getTableCellRendererComponent(t, v, sel, foc, r, c);
                String s = v == null ? "" : v.toString();
                if (!sel) {
                    comp.setBackground(s.contains("Đã chọn lại") ? new Color(214, 234, 255)
                            : s.contains("Câu trừu tượng") ? new Color(238, 214, 250)
                            : (s.contains("Khớp thấp") || s.contains("Độ khớp thấp")) ? new Color(255, 214, 214)
                            : s.contains("Cần kiểm tra") ? new Color(255, 243, 205)
                            : (s.contains("Khớp tốt") || s.contains("Phù hợp cao")) ? new Color(220, 245, 222) : Color.WHITE);
                    comp.setForeground(Color.BLACK);
                }
                return comp;
            }
        };
        timelineTable.getColumnModel().getColumn(5).setCellRenderer(conf);
        tableModel.addTableModelListener(e -> {
            if (updatingTable || e.getType() != TableModelEvent.UPDATE) return;
            if (e.getColumn() == 1 || e.getColumn() == 2) recomputeDuration(e.getFirstRow());
        });
        tab.add(new JScrollPane(timelineTable), BorderLayout.CENTER);

        JPanel tools = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 2));
        JButton bPrev = new JButton("🖼 Xem trước cảnh");
        bPrev.addActionListener(e -> previewSelectedScene());
        JButton bUp = new JButton("⬆ Lên");
        bUp.addActionListener(e -> moveRow(-1));
        JButton bDown = new JButton("⬇ Xuống");
        bDown.addActionListener(e -> moveRow(1));
        JButton bDel = new JButton("✖ Xóa dòng");
        bDel.addActionListener(e -> deleteSelectedRows());
        btnRematch = new JButton("🔁 AI tìm cảnh khác");
        btnRematch.setToolTipText("Chọn một dòng sai cảnh, AI sẽ đề xuất các cảnh khác (có thể gần đoạn hiện tại) để thay thế");
        btnRematch.addActionListener(e -> findAlternativeScene());
        tools.add(bPrev); tools.add(btnRematch); tools.add(bUp); tools.add(bDown); tools.add(bDel);
        JLabel hint = new JLabel("  Giây (83.5) hoặc MM:SS. Xanh lá = khớp tốt · Vàng = cần kiểm tra · Đỏ = khớp thấp · Tím = câu trừu tượng (cảnh theo mạch phim) · Xanh dương = đã chọn lại");
        hint.setBorder(new EmptyBorder(0, 6, 2, 0));
        JPanel south = new JPanel();
        south.setLayout(new BoxLayout(south, BoxLayout.Y_AXIS));
        tools.setAlignmentX(Component.LEFT_ALIGNMENT);
        hint.setAlignmentX(Component.LEFT_ALIGNMENT);
        south.add(tools); south.add(hint);
        tab.add(south, BorderLayout.SOUTH);
        return tab;
    }

    private JPanel buildSettingsTab() {
        JPanel root = new JPanel();
        root.setLayout(new BoxLayout(root, BoxLayout.Y_AXIS));
        root.setBorder(new EmptyBorder(6, 6, 6, 6));

        cbAspectMode = new JComboBox<>(new String[]{
                "16:9 Ngang (1920x1080)", "9:16 Dọc - Smart Fill (phóng to lấp đầy)",
                "9:16 Dọc - Nền mờ (Blur Background)", "9:16 Dọc - Fit (viền đen)"});
        cbMirrorMode = new JComboBox<>(new String[]{"Không lật hình", "Lật toàn bộ nhất quán", "Lật xen kẽ từng cảnh"});
        cbGpuNvenc = new JCheckBox("Render bằng GPU NVIDIA (h264_nvenc, tự quay về CPU nếu không hỗ trợ)", true);
        root.gadd(group("Khung hình & Render", new String[]{"Tỷ lệ khung hình:", "Lật hình:", ""},
                new JComponent[]{cbAspectMode, cbMirrorMode, cbGpuNvenc}));

        cbTtsEngine = new JComboBox<>(new String[]{"Microsoft Edge Neural (qua AI Server) - khuyên dùng", "Google Translate (dự phòng)"});
        cbTtsVoice = new JComboBox<>(new String[]{"Hoài My (Nữ)", "Nam Minh (Nam)"});
        spTtsRate = new JSpinner(new SpinnerNumberModel(0, -40, 60, 5));
        root.add(group("Giọng đọc (TTS)", new String[]{"Công cụ:", "Giọng (Edge):", "Tốc độ (%):"},
                new JComponent[]{cbTtsEngine, cbTtsVoice, spTtsRate}));

        cbBurnSubtitles = new JCheckBox("Ghi phụ đề lên video", true);
        cbSubColor = new JComboBox<>(new String[]{"Vàng (Review)", "Trắng (Cơ bản)"});
        cbSubSize = new JComboBox<>(new String[]{"Nhỏ", "Vừa", "Lớn", "Rất lớn"});
        cbSubSize.setSelectedIndex(1);
        cbExportSrt = new JCheckBox("Xuất thêm file .srt cạnh video", false);
        root.add(group("Phụ đề", new String[]{"", "Màu chữ:", "Cỡ chữ:", ""},
                new JComponent[]{cbBurnSubtitles, cbSubColor, cbSubSize, cbExportSrt}));

        cbKeepOriginalAudio = new JCheckBox("Giữ tiếng gốc phim", true);
        spOrigAudioVol = new JSpinner(new SpinnerNumberModel(12, 0, 100, 2));
        spBgmVol = new JSpinner(new SpinnerNumberModel(14, 0, 100, 2));
        cbDucking = new JCheckBox("Tự hạ nhạc nền khi có giọng đọc (ducking)", true);
        root.add(group("Âm thanh", new String[]{"", "Âm lượng tiếng gốc (%):", "Âm lượng nhạc nền (%):", ""},
                new JComponent[]{cbKeepOriginalAudio, spOrigAudioVol, spBgmVol, cbDucking}));

        spMaxScenes = new JSpinner(new SpinnerNumberModel(600, 20, 1500, 20));
        cbOrderStrict = new JComboBox<>(new String[]{"Lỏng (nhảy cảnh tự do)", "Vừa (khuyên dùng)", "Chặt (bám sát thứ tự phim)",
                "Tự do hoàn toàn (hồi tưởng / kể không theo thứ tự)"});
        cbOrderStrict.setSelectedIndex(1);
        cbUseOllama = new JCheckBox("Dùng Ollama Vision mô tả cảnh (chậm hơn, cần Ollama chạy)", false);
        cbKeepTempFiles = new JCheckBox("Giữ file tạm để debug", false);
        spRematchWindow = new JSpinner(new SpinnerNumberModel(120, 0, 3600, 30));
        cbAllowNoMatch = new JCheckBox("Đánh dấu câu trừu tượng (tên riêng/câu dẫn không có hình ảnh) - cảnh vẫn chọn theo mạch phim", true);
        cbAllowFallback = new JCheckBox("Cho phép chế độ dự phòng khi AI lỗi (chia đều/đoán mò - dễ SAI cảnh, mặc định TẮT)", false);
        cbRefine = new JCheckBox("Dò kỹ thời điểm khớp nhất trong cảnh (lấy mẫu dày, chậm hơn một chút)", true);
        root.add(group("AI & Khác", new String[]{"Số cảnh phân tích tối đa:", "Bám theo thứ tự phim:",
                        "'AI tìm cảnh khác' - tìm trong ±giây quanh đoạn cũ (0 = cả phim):", "", "", "", "", ""},
                new JComponent[]{spMaxScenes, cbOrderStrict, spRematchWindow, cbAllowNoMatch, cbAllowFallback, cbRefine, cbUseOllama, cbKeepTempFiles}));
        root.add(Box.createVerticalGlue());
        return root;
    }

    private JPanel group(String title, String[] labels, JComponent[] comps) {
        JPanel p = new JPanel(new GridBagLayout());
        p.setBorder(BorderFactory.createTitledBorder(title));
        p.setAlignmentX(Component.LEFT_ALIGNMENT);
        GridBagConstraints g = new GridBagConstraints();
        g.insets = new Insets(3, 6, 3, 6);
        g.anchor = GridBagConstraints.WEST;
        for (int i = 0; i < comps.length; i++) {
            g.gridy = i;
            if (labels[i].isEmpty()) {
                g.gridx = 0; g.gridwidth = 2;
                p.add(comps[i], g);
            } else {
                g.gridx = 0; g.gridwidth = 1;
                p.add(new JLabel(labels[i]), g);
                g.gridx = 1;
                p.add(comps[i], g);
            }
        }
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, p.getPreferredSize().height));
        return p;
    }

    private JPanel buildLogTab() {
        JPanel tab = new JPanel(new BorderLayout());
        txtLog = new JTextArea(8, 60);
        txtLog.setEditable(false);
        txtLog.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        txtLog.setBackground(new Color(248, 248, 248));
        tab.add(new JScrollPane(txtLog), BorderLayout.CENTER);
        return tab;
    }

    // =====================================================================
    //  CÀI ĐẶT (ghi nhớ giữa các lần chạy)
    // =====================================================================
    private void loadSettings() {
        try {
            txtBgmPath.setText(prefs.get("bgm", ""));
            String out = prefs.get("out", "");
            if (!out.isEmpty()) txtOutputPath.setText(out);
            cbAspectMode.setSelectedIndex(clampIdx(prefs.getInt("aspect", 0), cbAspectMode));
            cbMirrorMode.setSelectedIndex(clampIdx(prefs.getInt("mirror", 0), cbMirrorMode));
            cbGpuNvenc.setSelected(prefs.getBoolean("gpu", true));
            cbTtsEngine.setSelectedIndex(clampIdx(prefs.getInt("ttsEngine", 0), cbTtsEngine));
            cbTtsVoice.setSelectedIndex(clampIdx(prefs.getInt("ttsVoice", 0), cbTtsVoice));
            spTtsRate.setValue(prefs.getInt("ttsRate", 0));
            cbBurnSubtitles.setSelected(prefs.getBoolean("burn", true));
            cbSubColor.setSelectedIndex(clampIdx(prefs.getInt("subColor", 0), cbSubColor));
            cbSubSize.setSelectedIndex(clampIdx(prefs.getInt("subSize", 1), cbSubSize));
            cbExportSrt.setSelected(prefs.getBoolean("srt", false));
            cbKeepOriginalAudio.setSelected(prefs.getBoolean("keepAudio", true));
            spOrigAudioVol.setValue(prefs.getInt("origVol", 12));
            spBgmVol.setValue(prefs.getInt("bgmVol", 14));
            cbDucking.setSelected(prefs.getBoolean("duck", true));
            spMaxScenes.setValue(prefs.getInt("maxScenes2", 600));   // khóa mới: bỏ giá trị 240 cũ làm gộp mất ranh giới cảnh
            cbOrderStrict.setSelectedIndex(clampIdx(prefs.getInt("strict", 1), cbOrderStrict));
            cbUseOllama.setSelected(prefs.getBoolean("ollama", false));
            spRematchWindow.setValue(prefs.getInt("rematchWin", 120));
            cbAllowNoMatch.setSelected(prefs.getBoolean("noMatch", true));
            cbRefine.setSelected(prefs.getBoolean("refine", true));
            cbAllowFallback.setSelected(prefs.getBoolean("fallback", false));
        } catch (Exception ignored) { }
    }

    private int clampIdx(int i, JComboBox<?> cb) { return Math.max(0, Math.min(cb.getItemCount() - 1, i)); }

    private void saveSettings() {
        try {
            prefs.put("bgm", txtBgmPath.getText().trim());
            prefs.put("out", txtOutputPath.getText().trim());
            prefs.putInt("aspect", cbAspectMode.getSelectedIndex());
            prefs.putInt("mirror", cbMirrorMode.getSelectedIndex());
            prefs.putBoolean("gpu", cbGpuNvenc.isSelected());
            prefs.putInt("ttsEngine", cbTtsEngine.getSelectedIndex());
            prefs.putInt("ttsVoice", cbTtsVoice.getSelectedIndex());
            prefs.putInt("ttsRate", ((Number) spTtsRate.getValue()).intValue());
            prefs.putBoolean("burn", cbBurnSubtitles.isSelected());
            prefs.putInt("subColor", cbSubColor.getSelectedIndex());
            prefs.putInt("subSize", cbSubSize.getSelectedIndex());
            prefs.putBoolean("srt", cbExportSrt.isSelected());
            prefs.putBoolean("keepAudio", cbKeepOriginalAudio.isSelected());
            prefs.putInt("origVol", ((Number) spOrigAudioVol.getValue()).intValue());
            prefs.putInt("bgmVol", ((Number) spBgmVol.getValue()).intValue());
            prefs.putBoolean("duck", cbDucking.isSelected());
            prefs.putInt("maxScenes2", ((Number) spMaxScenes.getValue()).intValue());
            prefs.putInt("strict", cbOrderStrict.getSelectedIndex());
            prefs.putBoolean("ollama", cbUseOllama.isSelected());
            prefs.putInt("rematchWin", ((Number) spRematchWindow.getValue()).intValue());
            prefs.putBoolean("noMatch", cbAllowNoMatch.isSelected());
            prefs.putBoolean("refine", cbRefine.isSelected());
            prefs.putBoolean("fallback", cbAllowFallback.isSelected());
        } catch (Exception ignored) { }
    }

    // =====================================================================
    //  KHỞI ĐỘNG & KẾT NỐI AI SERVER
    // =====================================================================
    private void checkSystemStartup() {
        String ff = findFFmpegExecutable();
        if (ff == null) {
            appendLog("[CẢNH BÁO] Không tìm thấy ffmpeg (thư mục dự án hoặc PATH)!");
            lblStatus.setText("Chưa tìm thấy ffmpeg!");
            lblStatus.setForeground(Color.RED);
        } else {
            appendLog("[HỆ THỐNG] FFmpeg khả dụng: " + ff);
        }
        Thread t = new Thread(() -> {
            try {
                HttpResult r = http("GET", "/health", null, 1500, 1500);
                if (r.code == 200) {
                    Map<String, Object> h = Json.map(Json.parse(r.text()));
                    appendLog("[AI SERVER] Đã kết nối " + AI_SERVER_URL + " | CLIP: " + Json.str(h, "clip", "?")
                            + " | edge-tts: " + h.get("edge_tts") + " | ngưỡng đã hiệu chỉnh: " + h.get("calibrated") + " | thuật toán: " + h.get("match_version"));
                    if ("failed".equals(Json.str(h, "clip", "")))
                        appendLog("[AI SERVER] CLIP nạp lỗi: " + Json.str(h, "clip_error", "?") + " (server sẽ thử lại sau ít phút, kết quả tạm thời kém chính xác).");
                    if (!Boolean.TRUE.equals(h.get("edge_tts")))
                        appendLog("[AI SERVER] Chưa có edge-tts, giọng đọc sẽ dùng Google dự phòng. Cài: pip install edge-tts");
                    return;
                }
            } catch (Exception ignored) { }
            appendLog("[AI SERVER] Chưa phát hiện AI Server. Hãy chạy: python ai_service.py");
        }, "startup-check");
        t.setDaemon(true);
        t.start();
    }

    private boolean isAiServerAlive() {
        try { return http("GET", "/health", null, 1500, 1500).code == 200; } catch (Exception e) { return false; }
    }

    private HttpResult http(String method, String path, String jsonBody, int connectMs, int readMs) throws IOException {
        HttpURLConnection c = (HttpURLConnection) new URL(AI_SERVER_URL + path).openConnection();
        try {
            c.setRequestMethod(method);
            c.setConnectTimeout(connectMs);
            c.setReadTimeout(readMs);
            String key = System.getProperty("ai.key", System.getenv("AI_API_KEY") == null ? "" : System.getenv("AI_API_KEY"));
            if (!key.isEmpty()) c.setRequestProperty("X-API-Key", key);
            if (jsonBody != null) {
                c.setDoOutput(true);
                c.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
                try (OutputStream os = c.getOutputStream()) { os.write(jsonBody.getBytes(StandardCharsets.UTF_8)); }
            }
            int code = c.getResponseCode();
            InputStream is = code < 400 ? c.getInputStream() : c.getErrorStream();
            byte[] data = new byte[0];
            if (is != null) try (InputStream in = is) { data = in.readAllBytes(); }
            return new HttpResult(code, data);
        } finally {
            c.disconnect();
        }
    }

    private String serverError(HttpResult r) {
        try {
            Object d = Json.map(Json.parse(r.text())).get("detail");
            if (d instanceof Map) { Object msg = ((Map<?, ?>) d).get("message"); if (msg != null) return msg.toString(); }
            if (d != null) return d.toString();
        } catch (Exception ignored) { }
        return r.text();
    }

    // =====================================================================
    //  AI CHỌN CẢNH
    // =====================================================================
    private void runAiSceneMatchingUI() {
        String videoPath = txtVideoPath.getText().trim();
        String script = txtScript.getText().trim();
        if (videoPath.isEmpty() || !new File(videoPath).isFile()) {
            JOptionPane.showMessageDialog(this, "Vui lòng chọn file video phim gốc trước!", "Thiếu Video", JOptionPane.WARNING_MESSAGE);
            return;
        }
        if (script.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Vui lòng nhập kịch bản trước!", "Thiếu Kịch Bản", JOptionPane.WARNING_MESSAGE);
            return;
        }
        final AiOpts opts = readAiOpts();

        cancelRequested.set(false);
        setBusy(true);
        progressBar.setValue(3);
        lblStatus.setText("🤖 Đang gửi dữ liệu cho Python AI...");
        lblStatus.setForeground(new Color(102, 0, 153));

        new SwingWorker<List<SceneItem>, Void>() {
            private String err;

            @Override protected List<SceneItem> doInBackground() {
                try {
                    return requestAiMatchedScenes(videoPath, script, opts);
                } catch (Exception ex) {
                    err = ex.getMessage();
                    appendLog("[LỖI AI] " + err);
                    return null;
                }
            }

            @Override protected void done() {
                setBusy(false);
                if (cancelRequested.get()) {
                    lblStatus.setText("Đã hủy phân tích AI.");
                    lblStatus.setForeground(Color.RED);
                    progressBar.setValue(0);
                    return;
                }
                if (err != null) {
                    lblStatus.setText("Lỗi AI!");
                    lblStatus.setForeground(Color.RED);
                    progressBar.setValue(0);
                    JOptionPane.showMessageDialog(Main.this, err, "Lỗi AI Chọn Cảnh", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                try {
                    List<SceneItem> list = get();
                    if (list == null || list.isEmpty()) return;
                    populateTimelineTable(list);
                    tableScriptHash = scriptHash(script);
                    tabbedPane.setSelectedIndex(1);
                    progressBar.setValue(100);
                    long none = list.stream().filter(s -> "abstract".equals(s.status)).count();
                    long low = list.stream().filter(s -> "low".equals(s.status)).count();
                    long rev = list.stream().filter(s -> "review".equals(s.status)).count();
                    lblStatus.setText("🤖 AI đã xử lý " + list.size() + " phân cảnh (" + none + " câu trừu tượng)");
                    lblStatus.setForeground(none > 0 ? new Color(153, 51, 153) : new Color(0, 153, 51));
                    JOptionPane.showMessageDialog(Main.this,
                            "AI đã gán " + list.size() + " cảnh vào Timeline.\n"
                                    + (none > 0 ? "🟣 " + none + " câu trừu tượng (tên riêng/câu dẫn, không có hình ảnh đặc trưng) - cảnh được chọn theo mạch phim, nên xem lại nếu cần.\n" : "")
                                    + (low > 0 ? "🔴 " + low + " câu khớp thấp (tô đỏ).\n" : "")
                                    + (rev > 0 ? "🟡 " + rev + " câu cần kiểm tra (tô vàng).\n" : "")
                                    + ((none + low + rev) > 0 ? "→ Chọn dòng sai rồi bấm '🔁 AI tìm cảnh khác' để thay thế, hoặc 'Xem trước cảnh' để kiểm tra.\n" : "")
                                    + "Khi ổn, bấm '🚀 BƯỚC 2: TẠO VIDEO'.",
                            "Hoàn Tất", JOptionPane.INFORMATION_MESSAGE);
                } catch (Exception ignored) { }
            }
        }.execute();
    }

    private AiOpts readAiOpts() {
        AiOpts o = new AiOpts();
        o.maxScenes = ((Number) spMaxScenes.getValue()).intValue();
        o.strict = cbOrderStrict.getSelectedIndex();
        o.ollama = cbUseOllama.isSelected();
        o.allowNoMatch = cbAllowNoMatch.isSelected();
        o.allowFallback = cbAllowFallback.isSelected();
        o.refine = cbRefine.isSelected();
        return o;
    }

    private List<SceneItem> requestAiMatchedScenes(String videoPath, String rawScript, AiOpts o) throws Exception {
        if (!isAiServerAlive()) {
            throw new RuntimeException("Không kết nối được Python AI Server tại " + AI_SERVER_URL
                    + "\nHãy chạy: python ai_service.py");
        }
        List<String> sentences = extractCleanSentencesForAI(rawScript);
        if (sentences.isEmpty()) throw new RuntimeException("Không tách được câu hợp lệ nào từ kịch bản!");
        appendLog("[AI CLIENT] Gửi " + sentences.size() + " câu tới AI Engine...");

        final String jobId = UUID.randomUUID().toString().replace("-", "").substring(0, 12);
        currentJobId = jobId;
        AtomicBoolean polling = new AtomicBoolean(true);
        Thread poll = new Thread(() -> {
            while (polling.get() && !cancelRequested.get()) {
                try {
                    HttpResult r = http("GET", "/progress?job_id=" + jobId, null, 1000, 1000);
                    if (r.code == 200) {
                        Map<String, Object> m = Json.map(Json.parse(r.text()));
                        String msg = Json.str(m, "message", null);
                        if (msg != null) updateProgress("🤖 AI: " + msg, (int) Json.num(m, "percent", -1));
                    }
                    Thread.sleep(800);
                } catch (InterruptedException ie) {
                    return;
                } catch (Exception ignored) { }
            }
        }, "ai-poll");
        poll.setDaemon(true);
        poll.start();

        try {
            StringBuilder sb = new StringBuilder("{\"video_path\":").append(Json.quote(videoPath))
                    .append(",\"job_id\":").append(Json.quote(jobId))
                    .append(",\"max_scenes\":").append(o.maxScenes)
                    .append(",\"use_ollama\":").append(o.ollama)
                    .append(",\"allow_no_match\":").append(o.allowNoMatch)
                    .append(",\"allow_fallback\":").append(o.allowFallback)
                    .append(",\"refine\":").append(o.refine)
                    .append(",\"order_mode\":").append(Json.quote(ORDER_MODE_NAMES[Math.max(0, Math.min(ORDER_MODE_NAMES.length - 1, o.strict))]))
                    .append(",\"sentences\":[");
            for (int i = 0; i < sentences.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(Json.quote(sentences.get(i)));
            }
            sb.append("]}");
            HttpResult r = http("POST", "/auto_match_all", sb.toString(), 10000, 1_800_000);
            if (r.code != 200) {
                if (cancelRequested.get() || r.code == 499) throw new RuntimeException("Đã hủy.");
                throw new RuntimeException((r.code == 409 ? "AI Server đang bận: " : "AI Server lỗi HTTP " + r.code + ": ") + serverError(r));
            }
            Map<String, Object> root = Json.map(Json.parse(r.text()));
            for (Object w : Json.list(root.get("warnings"))) appendLog("[AI CẢNH BÁO] " + w);
            if (!Boolean.TRUE.equals(root.get("calibrated")))
                appendLog("[AI] Ngưỡng tin cậy CHƯA hiệu chỉnh bằng dữ liệu thật: nhãn 'Khớp tốt' chỉ mang tính tương đối, hãy xem trước các cảnh quan trọng.");
            List<SceneItem> res = parseAiResponse(root, sentences);
            appendLog("[AI CLIENT] Nhận " + res.size() + " cảnh (matcher: " + Json.str(root, "matcher", "?")
                    + ", cache: " + root.get("from_cache") + ", thời gian: " + root.get("timings") + ").");
            return res;
        } finally {
            polling.set(false);
            poll.interrupt();
            currentJobId = "";
        }
    }

    private static Cand parseCand(Map<String, Object> m) {
        Cand c = new Cand();
        c.sceneId = Json.str(m, "scene_id", "");
        c.start = Json.num(m, "start_sec", 0);
        c.end = Json.num(m, "end_sec", c.start + 3.5);
        if (c.end <= c.start) c.end = c.start + 3.5;
        c.focus = Json.num(m, "focus_sec", Double.NaN);
        c.actionStart = Json.num(m, "action_start_sec", Double.NaN);
        c.actionEnd = Json.num(m, "action_end_sec", Double.NaN);
        c.score = Json.num(m, "score", 0);
        c.raw = Json.num(m, "raw_sim", 0);
        c.conf = Json.str(m, "confidence", "");
        c.caption = Json.str(m, "caption", "");
        c.status = Json.str(m, "status", "review");
        return c;
    }

    private static String mmss(double v) {
        if (Double.isNaN(v)) return "--";
        int m = (int) (v / 60);
        return String.format(Locale.US, "%d:%04.1f", m, v - m * 60.0);
    }

    private static String aiInfo(String conf, double z, double raw, double focus, String status, String caption) {
        String f = Double.isNaN(focus) ? "" : String.format(Locale.US, " | focus %.1fs", focus);
        String tag = "abstract".equals(status) ? "theo mạch, " : "";
        return String.format(Locale.US, "[%s | %sz=%.2f raw=%.2f%s] %s", conf, tag, z, raw, f, caption);
    }

    private List<SceneItem> parseAiResponse(Map<String, Object> root, List<String> sentences) {
        List<Object> results = Json.list(root.get("results"));
        if (results.isEmpty()) throw new RuntimeException("AI Server không trả về cảnh nào!");
        List<SceneItem> items = new ArrayList<>();
        for (int i = 0; i < results.size(); i++) {
            Map<String, Object> m = Json.map(results.get(i));
            double s = Json.num(m, "start_sec", 0), e = Json.num(m, "end_sec", s + 3.5);
            if (e <= s) e = s + 3.5;
            String text = i < sentences.size() ? sentences.get(i) : Json.str(m, "text", "");
            String status = Json.str(m, "status", "ok");
            double focus = Json.num(m, "focus_sec", Double.NaN);
            String info = aiInfo(Json.str(m, "confidence", "AI Match"), Json.num(m, "score", 0), Json.num(m, "raw_sim", 0),
                    focus, status, Json.str(m, "caption", ""));
            SceneItem it = new SceneItem(i + 1, s, e, Json.num(m, "boundary_start", s), Json.num(m, "boundary_end", e), text, info);
            it.focusSec = focus;
            it.actionStart = Json.num(m, "action_start_sec", Double.NaN);
            it.actionEnd = Json.num(m, "action_end_sec", Double.NaN);
            it.status = status;
            it.sceneId = Json.str(m, "scene_id", "");
            for (Object c : Json.list(m.get("candidates"))) it.cands.add(parseCand(Json.map(c)));
            items.add(it);
        }
        return items;
    }

    /** Tách câu, gộp câu quá ngắn để tránh cảnh cắt vụn. */
    private List<String> extractCleanSentencesForAI(String rawScript) {
        List<String> lines = new ArrayList<>();
        for (String line : rawScript.split("\\r?\\n")) {
            String t = line.trim();
            if (t.isEmpty()) continue;
            int bar = t.indexOf('|');
            if (bar >= 0 && !t.substring(bar + 1).trim().isEmpty() && TIME_PAT_LOOSE.matcher(t.substring(0, bar)).find())
                t = t.substring(bar + 1).trim();
            lines.add(t);
        }
        List<String> raw = splitText(String.join(" ", lines), 150);
        List<String> merged = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        for (String s : raw) {
            if (cur.length() > 0 && cur.length() + 1 + s.length() <= 170) cur.append(' ');
            else if (cur.length() > 0) { merged.add(cur.toString()); cur.setLength(0); }
            cur.append(s);
            if (cur.length() >= 40) { merged.add(cur.toString()); cur.setLength(0); }
        }
        if (cur.length() > 0) {
            if (!merged.isEmpty() && cur.length() < 25 && merged.get(merged.size() - 1).length() + cur.length() < 170)
                merged.set(merged.size() - 1, merged.get(merged.size() - 1) + " " + cur);
            else merged.add(cur.toString());
        }
        return merged;
    }

    private static final Pattern TIME_PAT_LOOSE = Pattern.compile("\\d{1,2}:\\d{2}");

    private String scriptHash(String s) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-1").digest(String.join(" ", extractCleanSentencesForAI(s)).getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : d) sb.append(String.format("%02x", b));
            return sb.toString();
        } catch (Exception e) { return ""; }
    }

    // =====================================================================
    //  BẢNG TIMELINE
    // =====================================================================
    private void populateTimelineTable(List<SceneItem> list) {
        Runnable r = () -> {
            updatingTable = true;
            try {
                tableModel.setRowCount(0);
                for (SceneItem it : list) {
                    tableModel.addRow(new Object[]{it.index, fmt(it.startSec), fmt(it.endSec), fmt(it.endSec - it.startSec),
                            it.text, it.aiInfo != null ? it.aiInfo : "Thủ công", it});
                }
            } finally { updatingTable = false; }
        };
        if (SwingUtilities.isEventDispatchThread()) r.run(); else SwingUtilities.invokeLater(r);
    }

    private static String fmt(double v) { return String.format(Locale.US, "%.2f", v); }

    private void recomputeDuration(int row) {
        if (row < 0 || row >= tableModel.getRowCount()) return;
        updatingTable = true;
        try {
            double s = parseFlexibleTime(tableModel.getValueAt(row, 1).toString());
            double e = parseFlexibleTime(tableModel.getValueAt(row, 2).toString());
            tableModel.setValueAt(e > s ? fmt(e - s) : "Lỗi", row, 3);
        } catch (Exception ex) {
            tableModel.setValueAt("Lỗi", row, 3);
        } finally { updatingTable = false; }
    }

    private double parseFlexibleTime(String s) {
        s = s.trim().replace(',', '.');
        if (s.contains(":")) return parseTimeSec(s, TIME_PAT, 0);
        return Double.parseDouble(s);
    }

    private void renumberRows() {
        updatingTable = true;
        try { for (int i = 0; i < tableModel.getRowCount(); i++) tableModel.setValueAt(i + 1, i, 0); }
        finally { updatingTable = false; }
    }

    private void moveRow(int d) {
        int r = timelineTable.getSelectedRow(), t = r + d;
        if (r < 0 || t < 0 || t >= tableModel.getRowCount()) return;
        if (timelineTable.isEditing()) timelineTable.getCellEditor().stopCellEditing();
        updatingTable = true;
        try { tableModel.moveRow(r, r, t); } finally { updatingTable = false; }
        renumberRows();
        timelineTable.setRowSelectionInterval(t, t);
    }

    private void deleteSelectedRows() {
        int[] rows = timelineTable.getSelectedRows();
        if (timelineTable.isEditing()) timelineTable.getCellEditor().stopCellEditing();
        updatingTable = true;
        try { for (int i = rows.length - 1; i >= 0; i--) tableModel.removeRow(rows[i]); }
        finally { updatingTable = false; }
        renumberRows();
    }

    private SceneItem rowMeta(int row) {
        Object m = tableModel.getValueAt(row, COL_META);
        return m instanceof SceneItem ? (SceneItem) m : null;
    }

    private List<BufferedImage> extractFrames(String ff, String video, double[] ts, int width) throws Exception {
        List<BufferedImage> imgs = new ArrayList<>();
        File tmp = Files.createTempDirectory("preview_").toFile();
        try {
            for (int i = 0; i < ts.length; i++) {
                File out = new File(tmp, "f" + i + ".png");
                Process p = new ProcessBuilder(ff, "-y", "-nostdin", "-v", "error", "-ss", String.format(Locale.US, "%.3f", Math.max(0, ts[i])),
                        "-i", video, "-frames:v", "1", "-vf", "scale=" + width + ":-2", out.getAbsolutePath())
                        .redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.DISCARD).start();
                if (!p.waitFor(15, TimeUnit.SECONDS)) p.destroyForcibly();
                BufferedImage img = out.exists() ? ImageIO.read(out) : null;
                imgs.add(img != null ? img : new BufferedImage(width, width * 9 / 16, BufferedImage.TYPE_INT_RGB));
            }
        } finally { deleteDirectory(tmp); }
        return imgs;
    }

    private void previewSelectedScene() {
        int r = timelineTable.getSelectedRow();
        String ff = findFFmpegExecutable();
        String video = txtVideoPath.getText().trim();
        if (r < 0) { JOptionPane.showMessageDialog(this, "Hãy chọn một dòng trong bảng trước."); return; }
        if (ff == null || !new File(video).isFile()) { JOptionPane.showMessageDialog(this, "Cần ffmpeg và file video hợp lệ."); return; }
        final double s, e;
        try {
            s = parseFlexibleTime(tableModel.getValueAt(r, 1).toString());
            e = parseFlexibleTime(tableModel.getValueAt(r, 2).toString());
        } catch (Exception ex) { JOptionPane.showMessageDialog(this, "Thời gian không hợp lệ."); return; }
        final String caption = tableModel.getValueAt(r, 4).toString();
        SceneItem meta = rowMeta(r);
        final double focus = (meta != null && !Double.isNaN(meta.focusSec) && meta.focusSec >= s && meta.focusSec <= e)
                ? meta.focusSec : (s + e) / 2.0;

        new SwingWorker<List<BufferedImage>, Void>() {
            @Override protected List<BufferedImage> doInBackground() throws Exception {
                return extractFrames(ff, video, new double[]{s + 0.1, focus, Math.max(s, e - 0.2)}, 360);
            }

            @Override protected void done() {
                try {
                    List<BufferedImage> imgs = get();
                    JPanel row = new JPanel(new FlowLayout(FlowLayout.CENTER, 6, 6));
                    String[] names = {"Đầu", "Khoảnh khắc khớp " + mmss(focus), "Cuối"};
                    for (int i = 0; i < imgs.size(); i++) {
                        JLabel l = new JLabel(names[i], new ImageIcon(imgs.get(i)), SwingConstants.CENTER);
                        l.setVerticalTextPosition(SwingConstants.BOTTOM);
                        l.setHorizontalTextPosition(SwingConstants.CENTER);
                        row.add(l);
                    }
                    JTextArea ta = new JTextArea("Lời bình: " + caption, 3, 40);
                    ta.setLineWrap(true); ta.setWrapStyleWord(true); ta.setEditable(false);
                    JPanel p = new JPanel(new BorderLayout(4, 4));
                    p.add(row, BorderLayout.CENTER);
                    p.add(new JScrollPane(ta), BorderLayout.SOUTH);
                    JOptionPane.showMessageDialog(Main.this, p, String.format(Locale.US, "Xem trước %.2fs - %.2fs", s, e), JOptionPane.PLAIN_MESSAGE);
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(Main.this, "Không trích xuất được khung hình: " + ex.getMessage());
                }
            }
        }.execute();
    }

    // =====================================================================
    //  AI TÌM LẠI CẢNH KHÁC (thay thế cảnh sai, có thể gần đoạn cũ)
    // =====================================================================
    private static class RematchResult {
        List<Cand> cands = new ArrayList<>();
        boolean expanded, calibrated;
        double window;
    }

    private RematchResult requestRematch(String sentence, double nearSec, double windowSec, List<double[]> exclude, int topK, AiOpts o, boolean abstractHint) throws Exception {
        StringBuilder sb = new StringBuilder("{\"video_path\":").append(Json.quote(txtVideoPath.getText().trim()))
                .append(",\"sentence\":").append(Json.quote(sentence))
                .append(",\"max_scenes\":").append(o.maxScenes)
                .append(",\"use_ollama\":").append(o.ollama)
                .append(",\"allow_fallback\":").append(o.allowFallback)
                .append(",\"abstract\":").append(abstractHint)
                .append(",\"refine\":").append(o.refine)
                .append(",\"top_k\":").append(topK)
                .append(",\"near_sec\":").append(fmt(nearSec))
                .append(",\"window_sec\":").append(fmt(windowSec))
                .append(",\"exclude\":[");
        for (int i = 0; i < exclude.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append('[').append(fmt(exclude.get(i)[0])).append(',').append(fmt(exclude.get(i)[1])).append(']');
        }
        sb.append("]}");
        HttpResult r = http("POST", "/rematch", sb.toString(), 10000, 1_800_000);
        if (r.code != 200) throw new RuntimeException((r.code == 409 ? "AI Server đang bận: " : "AI Server lỗi HTTP " + r.code + ": ") + serverError(r));
        Map<String, Object> root = Json.map(Json.parse(r.text()));
        RematchResult rr = new RematchResult();
        rr.expanded = Boolean.TRUE.equals(root.get("window_expanded"));
        rr.calibrated = Boolean.TRUE.equals(root.get("calibrated"));
        rr.window = Json.num(root, "window_sec", 0);
        for (Object c : Json.list(root.get("candidates"))) rr.cands.add(parseCand(Json.map(c)));
        return rr;
    }

    private void applyCandidate(int row, Cand c) {
        if (row < 0 || row >= tableModel.getRowCount()) return;
        SceneItem old = rowMeta(row);
        String text = tableModel.getValueAt(row, 4).toString();
        SceneItem it = new SceneItem(row + 1, c.start, c.end, c.start, c.end, text, null);
        it.focusSec = c.focus; it.actionStart = c.actionStart; it.actionEnd = c.actionEnd;
        it.sceneId = c.sceneId; it.status = "user";
        if (old != null) { it.cands = old.cands; it.tried = new ArrayList<>(old.tried); }
        try {
            double os = parseFlexibleTime(tableModel.getValueAt(row, 1).toString());
            double oe = parseFlexibleTime(tableModel.getValueAt(row, 2).toString());
            it.tried.add(new double[]{os, oe});
        } catch (Exception ignored) { }
        it.aiInfo = String.format(Locale.US, "[Đã chọn lại | AI: %s z=%.2f raw=%.2f%s] %s", c.conf, c.score, c.raw,
                Double.isNaN(c.focus) ? "" : String.format(Locale.US, " | focus %.1fs", c.focus), c.caption);
        updatingTable = true;
        try {
            tableModel.setValueAt(fmt(c.start), row, 1);
            tableModel.setValueAt(fmt(c.end), row, 2);
            tableModel.setValueAt(fmt(c.end - c.start), row, 3);
            tableModel.setValueAt(it.aiInfo, row, 5);
            tableModel.setValueAt(it, row, COL_META);
        } finally { updatingTable = false; }
        appendLog(String.format(Locale.US, "[CHỌN LẠI] Câu #%d -> %s - %s", row + 1, mmss(c.start), mmss(c.end)));
    }

    private void findAlternativeScene() {
        int r = timelineTable.getSelectedRow();
        if (r < 0) { JOptionPane.showMessageDialog(this, "Hãy chọn dòng (câu) có cảnh bị sai trong bảng Timeline trước."); return; }
        if (timelineTable.isEditing()) timelineTable.getCellEditor().stopCellEditing();
        String ff = findFFmpegExecutable();
        String video = txtVideoPath.getText().trim();
        if (ff == null || !new File(video).isFile()) { JOptionPane.showMessageDialog(this, "Cần ffmpeg và file video hợp lệ."); return; }
        if (busy.get()) { JOptionPane.showMessageDialog(this, "Đang có tác vụ chạy, hãy đợi xong rồi thử lại."); return; }
        if (!isAiServerAlive()) { JOptionPane.showMessageDialog(this, "Không kết nối được Python AI Server.\nHãy chạy: python ai_service.py"); return; }
        double s, e;
        try {
            s = parseFlexibleTime(tableModel.getValueAt(r, 1).toString());
            e = parseFlexibleTime(tableModel.getValueAt(r, 2).toString());
        } catch (Exception ex) { JOptionPane.showMessageDialog(this, "Thời gian của dòng này không hợp lệ."); return; }
        SceneItem cur = new SceneItem(r + 1, s, e, s, e, tableModel.getValueAt(r, 4).toString().trim(), "");
        SceneItem meta = rowMeta(r);
        if (meta != null) { cur.focusSec = meta.focusSec; cur.cands = meta.cands; cur.tried = meta.tried; cur.status = meta.status; }
        if (cur.text.isEmpty()) { JOptionPane.showMessageDialog(this, "Lời bình của dòng này đang trống."); return; }
        double near = (!Double.isNaN(cur.focusSec) && cur.focusSec >= s && cur.focusSec <= e) ? cur.focusSec : (s + e) / 2.0;
        double window = ((Number) spRematchWindow.getValue()).doubleValue();
        new AltDialog(r, cur, ff, video, near, window, readAiOpts()).setVisible(true);
    }

    private final class AltDialog extends JDialog {
        private final int row;
        private final SceneItem cur;
        private final String ff, video;
        private final double near, windowSec;
        private final AiOpts opts;
        private final DefaultListModel<Cand> model = new DefaultListModel<>();
        private final JList<Cand> list = new JList<>(model);
        private final JLabel[] pics = {new JLabel("", SwingConstants.CENTER), new JLabel("", SwingConstants.CENTER), new JLabel("", SwingConstants.CENTER)};
        private final JLabel lblInfo = new JLabel(" ");
        private final JButton btnUse = new JButton("✔ Dùng cảnh này"), btnMore = new JButton("🔁 Tìm thêm cảnh khác"), btnClose = new JButton("Đóng");
        private final List<double[]> exclude = new ArrayList<>();
        private int previewSeq = 0;
        private boolean fetching = false;

        AltDialog(int row, SceneItem cur, String ff, String video, double near, double windowSec, AiOpts opts) {
            super(Main.this, "AI tìm cảnh khác cho câu #" + (row + 1), true);
            this.row = row; this.cur = cur; this.ff = ff; this.video = video;
            this.near = near; this.windowSec = windowSec; this.opts = opts;
            exclude.add(new double[]{cur.startSec, cur.endSec});
            exclude.addAll(cur.tried);
            for (Cand c : cur.cands) { model.addElement(c); exclude.add(new double[]{c.start, c.end}); }

            setLayout(new BorderLayout(8, 8));
            ((JComponent) getContentPane()).setBorder(new EmptyBorder(8, 10, 8, 10));
            JTextArea ta = new JTextArea("Lời bình: " + cur.text
                    + String.format(Locale.US, "\nCảnh hiện tại: %s - %s | Tìm quanh mốc %s%s", mmss(cur.startSec), mmss(cur.endSec), mmss(near),
                    windowSec > 0 ? String.format(Locale.US, " (±%.0fs, tự nới rộng nếu thiếu ứng viên)", windowSec) : " (toàn phim)"), 3, 60);
            ta.setLineWrap(true); ta.setWrapStyleWord(true); ta.setEditable(false);
            add(new JScrollPane(ta), BorderLayout.NORTH);

            list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
            list.setCellRenderer(new DefaultListCellRenderer() {
                @Override public Component getListCellRendererComponent(JList<?> l, Object v, int i, boolean sel, boolean foc) {
                    Cand c = (Cand) v;
                    String t = String.format(Locale.US, "%d) %s – %s | %s | z=%.2f raw=%.2f | khớp tại %s", i + 1, mmss(c.start), mmss(c.end),
                            c.conf.isEmpty() ? "ứng viên" : c.conf, c.score, c.raw, mmss(c.focus));
                    return super.getListCellRendererComponent(l, t, i, sel, foc);
                }
            });
            list.addListSelectionListener(ev -> { if (!ev.getValueIsAdjusting()) loadPreview(list.getSelectedValue()); });
            JScrollPane left = new JScrollPane(list);
            left.setPreferredSize(new Dimension(430, 300));
            JPanel right = new JPanel(new GridLayout(3, 1, 4, 4));
            for (JLabel p : pics) {
                p.setVerticalTextPosition(SwingConstants.BOTTOM);
                p.setHorizontalTextPosition(SwingConstants.CENTER);
                right.add(p);
            }
            JSplitPane split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, left, new JScrollPane(right));
            split.setResizeWeight(0.5);
            add(split, BorderLayout.CENTER);

            JPanel south = new JPanel(new BorderLayout(6, 4));
            south.add(lblInfo, BorderLayout.CENTER);
            JPanel btns = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
            btns.add(btnMore); btns.add(btnUse); btns.add(btnClose);
            south.add(btns, BorderLayout.EAST);
            add(south, BorderLayout.SOUTH);

            btnUse.addActionListener(ev -> {
                Cand c = list.getSelectedValue();
                if (c == null) { lblInfo.setText("Hãy chọn một ứng viên trong danh sách."); return; }
                applyCandidate(row, c);
                dispose();
            });
            btnMore.addActionListener(ev -> fetchMore());
            btnClose.addActionListener(ev -> dispose());
            setSize(1000, 560);
            setLocationRelativeTo(Main.this);
            SwingUtilities.invokeLater(() -> { if (model.isEmpty()) fetchMore(); else list.setSelectedIndex(0); });
        }

        private void fetchMore() {
            if (fetching) return;
            fetching = true;
            btnMore.setEnabled(false);
            lblInfo.setText("Đang nhờ AI tìm thêm ứng viên (có thể mất vài chục giây nếu video chưa được phân tích)...");
            final List<double[]> ex = new ArrayList<>(exclude);
            new SwingWorker<RematchResult, Void>() {
                private String err;
                @Override protected RematchResult doInBackground() {
                    try { return requestRematch(cur.text, near, windowSec, ex, 5, opts, "abstract".equals(cur.status)); }
                    catch (Exception e2) { err = e2.getMessage(); return null; }
                }
                @Override protected void done() {
                    fetching = false;
                    btnMore.setEnabled(true);
                    try {
                        RematchResult rr = get();
                        if (rr == null) { lblInfo.setText("Lỗi: " + err); return; }
                        int before = model.size();
                        for (Cand c : rr.cands) { model.addElement(c); exclude.add(new double[]{c.start, c.end}); }
                        int added = rr.cands.size();
                        lblInfo.setText(added == 0 ? "AI không tìm thấy thêm ứng viên nào." : "Đã thêm " + added + " ứng viên"
                                + (rr.expanded ? String.format(Locale.US, " (đã nới vùng tìm ra ±%.0fs)", rr.window) : "")
                                + (rr.calibrated ? "" : " | ngưỡng chưa hiệu chỉnh, hãy xem ảnh để xác nhận"));
                        if (added > 0 && list.getSelectedIndex() < 0) list.setSelectedIndex(before);
                    } catch (Exception e2) { lblInfo.setText("Lỗi: " + e2.getMessage()); }
                }
            }.execute();
        }

        private void loadPreview(Cand c) {
            if (c == null) return;
            final int seq = ++previewSeq;
            for (JLabel p : pics) { p.setIcon(null); p.setText("..."); }
            new SwingWorker<List<BufferedImage>, Void>() {
                @Override protected List<BufferedImage> doInBackground() throws Exception {
                    double f = Double.isNaN(c.focus) ? (c.start + c.end) / 2.0 : c.focus;
                    return extractFrames(ff, video, new double[]{c.start + 0.1, f, Math.max(c.start, c.end - 0.2)}, 280);
                }
                @Override protected void done() {
                    if (seq != previewSeq) return;
                    try {
                        List<BufferedImage> imgs = get();
                        String[] names = {"Đầu " + mmss(c.start), "Khoảnh khắc khớp " + mmss(c.focus), "Cuối " + mmss(c.end)};
                        for (int i = 0; i < pics.length && i < imgs.size(); i++) {
                            pics[i].setIcon(new ImageIcon(imgs.get(i)));
                            pics[i].setText(names[i]);
                        }
                    } catch (Exception e2) { pics[0].setText("Không trích xuất được khung hình"); }
                }
            }.execute();
        }
    }

    private void parseScriptToTimeline() {
        String script = txtScript.getText().trim();
        if (script.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Vui lòng nhập kịch bản trước!", "Thông Báo", JOptionPane.WARNING_MESSAGE);
            return;
        }
        try {
            List<SceneItem> list = parseTimestampScript(script);
            populateTimelineTable(list);
            tableScriptHash = "";
            tabbedPane.setSelectedIndex(1);
            JOptionPane.showMessageDialog(this, "Đã nạp " + list.size() + " phân cảnh!", "Thành Công", JOptionPane.INFORMATION_MESSAGE);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, ex.getMessage(), "Lỗi Cú Pháp Kịch Bản", JOptionPane.ERROR_MESSAGE);
        }
    }

    // =====================================================================
    //  TIỆN ÍCH UI
    // =====================================================================
    private void setBusy(boolean b) {
        busy.set(b);
        btnRun.setEnabled(!b);
        btnAiMatch.setEnabled(!b);
        btnParseTimeline.setEnabled(!b);
        if (btnRematch != null) btnRematch.setEnabled(!b);
        btnCancel.setEnabled(b);
    }

    private void selectFile(JTextField field, boolean isSave) {
        String cur = field.getText().trim();
        File start = cur.isEmpty() ? new File(prefs.get("lastDir", System.getProperty("user.home"))) : new File(cur).getParentFile();
        JFileChooser ch = new JFileChooser(start != null && start.isDirectory() ? start : null);
        if (isSave && !cur.isEmpty()) ch.setSelectedFile(new File(cur));
        int res = isSave ? ch.showSaveDialog(this) : ch.showOpenDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            File f = ch.getSelectedFile();
            field.setText(f.getAbsolutePath());
            if (f.getParentFile() != null) prefs.put("lastDir", f.getParentFile().getAbsolutePath());
        }
    }

    private void appendLog(String msg) {
        SwingUtilities.invokeLater(() -> {
            txtLog.append(msg + "\n");
            txtLog.setCaretPosition(txtLog.getDocument().getLength());
        });
    }

    private void updateProgress(String status, int progress) {
        SwingUtilities.invokeLater(() -> {
            lblStatus.setText(status);
            if (progress >= 0) progressBar.setValue(Math.min(100, progress));
        });
    }

    private void requestCancel() {
        cancelRequested.set(true);
        appendLog("[HỦY] Đang dừng toàn bộ tiến trình...");
        lblStatus.setText("Đang hủy tác vụ...");
        for (Process p : activeProcesses) killProcessTree(p);
        activeProcesses.clear();
        final String jid = currentJobId;
        Thread t = new Thread(() -> { try { http("POST", "/cancel", jid.isEmpty() ? "{}" : "{\"job_id\":" + Json.quote(jid) + "}", 1500, 1500); } catch (Exception ignored) { } });
        t.setDaemon(true);
        t.start();
    }

    private void killProcessTree(Process proc) {
        if (proc == null) return;
        try {
            proc.toHandle().descendants().forEach(ProcessHandle::destroyForcibly);
            proc.destroyForcibly();
        } catch (Exception ignored) { }
    }

    // =====================================================================
    //  BẮT ĐẦU DỰNG VIDEO
    // =====================================================================
    private void startWorkflowFromEDT() {
        String ff = findFFmpegExecutable();
        if (ff == null) {
            JOptionPane.showMessageDialog(this, "Không tìm thấy ffmpeg!", "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        if (timelineTable.isEditing()) timelineTable.getCellEditor().stopCellEditing();

        JobConfig cfg = new JobConfig();
        cfg.videoPath = txtVideoPath.getText().trim();
        cfg.bgmPath = txtBgmPath.getText().trim();
        cfg.outputPath = txtOutputPath.getText().trim();
        cfg.isAutoMode = rbAuto.isSelected();
        cfg.aspectMode = cbAspectMode.getSelectedIndex();
        cfg.mirrorMode = cbMirrorMode.getSelectedIndex();
        cfg.useGpuNvenc = cbGpuNvenc.isSelected();
        cfg.burnSubtitles = cbBurnSubtitles.isSelected();
        cfg.subColorStyle = cbSubColor.getSelectedIndex() == 0 ? "&H0000FFFF" : "&H00FFFFFF";
        cfg.subSizeIdx = cbSubSize.getSelectedIndex();
        cfg.exportSrt = cbExportSrt.isSelected();
        cfg.keepOriginalAudio = cbKeepOriginalAudio.isSelected();
        cfg.keepTempFiles = cbKeepTempFiles.isSelected();
        cfg.origAudioVolume = ((Number) spOrigAudioVol.getValue()).doubleValue() / 100.0;
        cfg.bgmVolume = ((Number) spBgmVol.getValue()).doubleValue() / 100.0;
        cfg.ducking = cbDucking.isSelected();
        cfg.ttsEngine = cbTtsEngine.getSelectedIndex();
        cfg.ttsVoice = cbTtsVoice.getSelectedIndex();
        cfg.ttsRatePercent = ((Number) spTtsRate.getValue()).intValue();
        cfg.maxScenes = ((Number) spMaxScenes.getValue()).intValue();
        cfg.orderStrict = cbOrderStrict.getSelectedIndex();
        cfg.useOllama = cbUseOllama.isSelected();
        cfg.ai = readAiOpts();
        cfg.rawScript = txtScript.getText().trim();

        File inVid = new File(cfg.videoPath);
        if (cfg.videoPath.isEmpty() || !inVid.isFile()) {
            JOptionPane.showMessageDialog(this, "File video phim gốc không tồn tại!", "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        if (cfg.rawScript.isEmpty() && tableModel.getRowCount() == 0) {
            JOptionPane.showMessageDialog(this, "Kịch bản không được để trống!", "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        if (cfg.outputPath.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Hãy chọn nơi lưu video!", "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }
        if (!cfg.outputPath.toLowerCase(Locale.ROOT).endsWith(".mp4")) cfg.outputPath += ".mp4";
        File outFile = new File(cfg.outputPath);
        try {
            if (inVid.getCanonicalPath().equalsIgnoreCase(outFile.getCanonicalPath())) {
                JOptionPane.showMessageDialog(this, "File đầu ra không được trùng video gốc!", "Lỗi", JOptionPane.ERROR_MESSAGE);
                return;
            }
        } catch (IOException ignored) { }
        if (outFile.exists() && JOptionPane.showConfirmDialog(this,
                "File \"" + outFile.getName() + "\" đã tồn tại. Ghi đè?", "Cảnh Báo Ghi Đè",
                JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE) != JOptionPane.YES_OPTION) return;

        // Bảng timeline cũ không khớp kịch bản hiện tại -> hỏi người dùng (tránh dựng nhầm cảnh cũ)
        boolean useTable = tableModel.getRowCount() > 0;
        if (useTable && cfg.isAutoMode && !tableScriptHash.isEmpty() && !tableScriptHash.equals(scriptHash(cfg.rawScript))) {
            Object[] opts = {"Dùng bảng Timeline hiện có", "Xóa bảng, chạy lại AI", "Hủy"};
            int c = JOptionPane.showOptionDialog(this, "Kịch bản đã thay đổi so với lúc AI chọn cảnh.\nBạn muốn làm gì?",
                    "Timeline lỗi thời", JOptionPane.DEFAULT_OPTION, JOptionPane.WARNING_MESSAGE, null, opts, opts[1]);
            if (c == 1) { tableModel.setRowCount(0); tableScriptHash = ""; useTable = false; }
            else if (c != 0) return;
        }

        try {
            if (useTable) {
                cfg.scenes = readScenesFromTable();
            } else if (!cfg.isAutoMode) {
                cfg.scenes = parseTimestampScript(cfg.rawScript);
            }
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Lỗi đọc bảng/kịch bản: " + ex.getMessage(), "Lỗi", JOptionPane.ERROR_MESSAGE);
            return;
        }

        saveSettings();
        cancelRequested.set(false);
        edgeTtsBroken.set(false);
        setBusy(true);
        txtLog.setText("");
        progressBar.setValue(0);
        lblStatus.setForeground(Color.BLUE);

        new SwingWorker<Void, Void>() {
            private String error;

            @Override protected Void doInBackground() {
                try { executeJob(ff, cfg); }
                catch (Exception ex) { error = ex.getMessage() != null ? ex.getMessage() : ex.toString(); appendLog("[LỖI] " + error); }
                return null;
            }

            @Override protected void done() {
                setBusy(false);
                activeProcesses.clear();
                if (cancelRequested.get()) {
                    lblStatus.setText("Đã dừng tác vụ.");
                    lblStatus.setForeground(Color.RED);
                    JOptionPane.showMessageDialog(Main.this, "Đã hủy tác vụ và dọn dẹp tiến trình.", "Thông Báo", JOptionPane.WARNING_MESSAGE);
                } else if (error != null) {
                    lblStatus.setText("Thất bại!");
                    lblStatus.setForeground(Color.RED);
                    JOptionPane.showMessageDialog(Main.this, error, "Lỗi Dựng Video", JOptionPane.ERROR_MESSAGE);
                } else {
                    lblStatus.setText("Hoàn tất thành công!");
                    lblStatus.setForeground(new Color(0, 153, 51));
                    progressBar.setValue(100);
                    JOptionPane.showMessageDialog(Main.this, "Đã xuất video thành công:\n" + cfg.outputPath, "Thành Công", JOptionPane.INFORMATION_MESSAGE);
                }
            }
        }.execute();
    }

    private List<SceneItem> readScenesFromTable() {
        List<SceneItem> scenes = new ArrayList<>();
        for (int r = 0; r < tableModel.getRowCount(); r++) {
            try {
                double s = parseFlexibleTime(tableModel.getValueAt(r, 1).toString());
                double e = parseFlexibleTime(tableModel.getValueAt(r, 2).toString());
                if (e <= s) throw new IllegalArgumentException("kết thúc <= bắt đầu");
                String t = tableModel.getValueAt(r, 4).toString().trim();
                if (t.isEmpty()) throw new IllegalArgumentException("lời bình trống");
                String info = String.valueOf(tableModel.getValueAt(r, 5));
                SceneItem it = new SceneItem(r + 1, s, e, s, e, t, info);
                Object meta = tableModel.getValueAt(r, COL_META);
                // Giữ ranh giới/tiêu điểm của AI nếu người dùng không sửa mốc thời gian
                if (meta instanceof SceneItem) {
                    SceneItem m = (SceneItem) meta;
                    if (Math.abs(m.startSec - s) < 0.011 && Math.abs(m.endSec - e) < 0.011) {
                        it.boundaryStart = m.boundaryStart; it.boundaryEnd = m.boundaryEnd; it.focusSec = m.focusSec;
                        it.actionStart = m.actionStart; it.actionEnd = m.actionEnd; it.status = m.status; it.sceneId = m.sceneId;
                    } else {
                        it.status = "manual";      // người dùng đã sửa mốc thời gian -> coi là đã xác nhận
                    }
                }
                scenes.add(it);
            } catch (Exception ex) {
                throw new RuntimeException("Dòng " + (r + 1) + ": " + ex.getMessage());
            }
        }
        return scenes;
    }

    // =====================================================================
    //  PIPELINE DỰNG PHIM (luồng nền)
    // =====================================================================
    private void executeJob(String ff, JobConfig cfg) throws Exception {
        File tempDir = Files.createTempDirectory("review_studio_v3_").toFile();
        appendLog("[THƯ MỤC TẠM] " + tempDir.getAbsolutePath());
        try {
            updateProgress("Đang kiểm tra video gốc...", 2);
            MediaInfo vi = getMediaInfo(ff, cfg.videoPath);
            if (!vi.hasVideo || vi.duration <= 1.0) throw new RuntimeException("Video gốc không hợp lệ hoặc quá ngắn!");
            appendLog(String.format(Locale.US, "[VIDEO GỐC] %.2fs | Âm thanh: %s", vi.duration, vi.hasAudio ? "Có" : "Không"));

            if (cfg.scenes == null || cfg.scenes.isEmpty()) {
                updateProgress("🤖 Đang nhờ Python AI chọn cảnh...", 4);
                cfg.scenes = buildAutoScenes(cfg, vi.duration);
                populateTimelineTable(cfg.scenes);
                tableScriptHash = scriptHash(cfg.rawScript);
            }
            for (SceneItem sc : cfg.scenes) {
                if (sc.startSec >= vi.duration) sc.startSec = Math.max(0, vi.duration - 4.0);
                if (sc.endSec > vi.duration || sc.endSec <= sc.startSec) sc.endSec = Math.min(vi.duration, sc.startSec + 4.0);
            }
            final int total = cfg.scenes.size();

            String vCodec = "libx264";
            if (cfg.useGpuNvenc) {
                if (verifyNvencRealEncode(ff, tempDir)) { vCodec = "h264_nvenc"; appendLog("[GPU] NVENC hoạt động tốt."); }
                else appendLog("[GPU FALLBACK] NVENC không khả dụng, dùng libx264 (CPU).");
            }

            // ---- Giai đoạn A: tạo toàn bộ giọng đọc song song ----
            double[] voiceDur = new double[total];
            File[] voices = new File[total];
            generateAllVoices(ff, cfg, tempDir, voices, voiceDur);

            // ---- Giai đoạn B: dựng từng cảnh ----
            boolean portrait = cfg.aspectMode != 0;
            List<File> clips = new ArrayList<>();
            List<SubEntry> subs = new ArrayList<>();
            double timeline = 0.0;
            int lineChars = subtitleLineChars(cfg, portrait);

            for (int i = 0; i < total; i++) {
                checkCancelled();
                SceneItem item = cfg.scenes.get(i);
                updateProgress("Đang dựng cảnh " + (i + 1) + "/" + total + "...", 32 + (int) (i * 50.0 / total));

                double vd = voiceDur[i];
                double center = !Double.isNaN(item.focusSec) ? item.focusSec : (item.startSec + item.endSec) / 2.0;
                double bStart = item.boundaryStart > 0 ? item.boundaryStart : item.startSec;
                double bEnd = item.boundaryEnd > bStart ? item.boundaryEnd : item.endSec;
                // Không bao giờ lấn ra ngoài ranh giới cảnh AI đã chọn (tránh dính đuôi cảnh trước / đầu cảnh sau).
                // Lời thoại dài hơn cảnh -> chậm nhẹ (tối đa 1.42x) rồi giữ khung cuối, KHÔNG kéo thêm hình của cảnh khác.
                double minStart = Math.max(0.0, bStart), maxEnd = Math.min(vi.duration, bEnd);
                double sceneLen = Math.max(0.4, maxEnd - minStart);
                double clipDur = Math.min(vd, sceneLen);
                double clipStart = Math.max(minStart, Math.min(center - clipDur / 2.0, maxEnd - clipDur));
                clipStart = Math.max(0.0, Math.min(clipStart, Math.max(0.0, vi.duration - 0.5)));
                if (clipStart + clipDur > vi.duration) clipDur = Math.max(0.3, vi.duration - clipStart);
                double pts = Math.max(0.68, Math.min(1.42, vd / Math.max(clipDur, 0.1)));

                boolean mirror = (cfg.mirrorMode == 1) || (cfg.mirrorMode == 2 && i % 2 == 1);
                String graph = buildVideoFilterGraph(pts, mirror, cfg.aspectMode, Math.max(2.0, vd));
                File clip = new File(tempDir, String.format("clip_%03d.mp4", i));
                renderSingleScene(ff, cfg, vCodec, vi.hasAudio, clipStart, clipDur, vd, pts, voices[i], graph, clip, tempDir);
                clips.add(clip);

                // Đo thời lượng thật của clip để phụ đề không bị lệch tích lũy
                double real = getMediaInfo(ff, clip.getAbsolutePath()).duration;
                if (real <= 0.1) real = vd;
                addSubtitleEntries(subs, item.text, timeline, real, lineChars);
                timeline += real;
            }

            checkCancelled();
            updateProgress("Đang ghép các phân cảnh...", 84);
            File concatList = new File(tempDir, "concat.txt");
            try (PrintWriter pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(concatList), StandardCharsets.UTF_8))) {
                for (File c : clips) pw.println("file '" + c.getAbsolutePath().replace("\\", "/").replace("'", "'\\''") + "'");
            }
            File merged = new File(tempDir, "merged.mp4");
            runCmd(tempDir, 900, args("-y", "-nostdin", "-f", "concat", "-safe", "0", "-i", concatList.getAbsolutePath(),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", merged.getAbsolutePath()));
            double totalDur = getMediaInfo(ff, merged.getAbsolutePath()).duration;
            if (totalDur <= 0.5) totalDur = timeline;

            checkCancelled();
            updateProgress("Đang gắn phụ đề, trộn nhạc nền...", 90);
            File ass = new File(tempDir, "subs.ass");
            writeText(ass, buildAss(subs, cfg, portrait));
            File srt = new File(tempDir, "subs.srt");
            writeText(srt, buildSrt(subs));

            File finalTmp = new File(tempDir, "final_verified.mp4");
            finalizeOutput(ff, cfg, vCodec, merged, totalDur, finalTmp, tempDir);
            verifyRenderedOutput(ff, finalTmp, totalDur, tempDir);

            File outFile = new File(cfg.outputPath);
            File parent = outFile.getAbsoluteFile().getParentFile();
            if (parent != null) parent.mkdirs();
            Files.move(finalTmp.toPath(), outFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            if (cfg.exportSrt) {
                String base = outFile.getAbsolutePath().replaceAll("(?i)\\.mp4$", "");
                Files.copy(srt.toPath(), new File(base + ".srt").toPath(), StandardCopyOption.REPLACE_EXISTING);
            }
            appendLog(String.format(Locale.US, "[XUẤT BẢN] %s (%.2f MB, %.1fs)", outFile.getAbsolutePath(),
                    outFile.length() / (1024.0 * 1024.0), totalDur));
        } finally {
            if (!cfg.keepTempFiles) deleteDirectory(tempDir);
            else appendLog("[DEBUG] Giữ file tạm tại: " + tempDir.getAbsolutePath());
        }
    }

    // ---------- Bộ lọc video ----------
    private String buildVideoFilterGraph(double pts, boolean mirror, int aspect, double padSec) {
        // tpad giữ khung cuối nếu hình ngắn hơn giọng đọc (tránh hình đứt, âm còn)
        String base = String.format(Locale.US, "[0:v]setpts=%.4f*PTS,fps=30%s,tpad=stop_mode=clone:stop_duration=%.2f", pts, mirror ? ",hflip" : "", padSec);
        switch (aspect) {
            case 0:
                return base + ",scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p[vout]";
            case 1:
                return base + ",scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,format=yuv420p[vout]";
            case 2:
                return base + ",split=2[fg][bg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:10[blurred];"
                        + "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[scaled];"
                        + "[blurred][scaled]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[vout]";
            default:
                return base + ",scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p[vout]";
        }
    }

    private List<String> encArgs(String vCodec) {
        if ("h264_nvenc".equals(vCodec))
            return args("-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0", "-maxrate", "16M", "-bufsize", "32M", "-pix_fmt", "yuv420p");
        return args("-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p");
    }

    private void renderSingleScene(String ff, JobConfig cfg, String vCodec, boolean hasAudio, double start, double dur,
                                   double targetDur, double pts, File voice, String videoGraph, File out, File tempDir) throws Exception {
        List<String> a = args("-y", "-nostdin", "-ss", String.format(Locale.US, "%.3f", start), "-t", String.format(Locale.US, "%.3f", dur),
                "-i", cfg.videoPath, "-i", voice.getAbsolutePath());
        if (cfg.keepOriginalAudio && hasAudio && cfg.origAudioVolume > 0.01) {
            double tempo = Math.max(0.5, Math.min(2.0, 1.0 / pts)); // đồng bộ tiếng gốc với hình đã co giãn
            String fc = String.format(Locale.US, "%s;[0:a]atempo=%.4f,volume=%.2f,afade=t=in:st=0:d=0.15,apad[origA];[1:a]volume=1.0[ttsA];"
                    + "[origA][ttsA]amix=inputs=2:duration=shortest:dropout_transition=1:normalize=0[aout]", videoGraph, tempo, cfg.origAudioVolume);
            a.addAll(args("-filter_complex", fc, "-map", "[vout]", "-map", "[aout]"));
        } else {
            a.addAll(args("-filter_complex", videoGraph, "-map", "[vout]", "-map", "1:a:0"));
        }
        a.addAll(encArgs(vCodec));
        a.addAll(args("-r", "30", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                "-t", String.format(Locale.US, "%.3f", targetDur), out.getAbsolutePath()));
        runCmd(tempDir, 300, a);
    }

    private void finalizeOutput(String ff, JobConfig cfg, String vCodec, File merged, double totalDur, File target, File tempDir) throws Exception {
        boolean hasBgm = !cfg.bgmPath.isEmpty() && new File(cfg.bgmPath).isFile() && cfg.bgmVolume > 0.01;
        boolean burn = cfg.burnSubtitles;
        List<String> a = args("-y", "-nostdin", "-i", merged.getAbsolutePath());
        if (hasBgm) a.addAll(args("-stream_loop", "-1", "-i", cfg.bgmPath));

        StringBuilder fc = new StringBuilder();
        if (burn) fc.append("[0:v]ass=subs.ass[vout]");
        if (hasBgm) {
            if (fc.length() > 0) fc.append(';');
            double fadeStart = Math.max(0.5, totalDur - 1.5);
            fc.append(String.format(Locale.US, "[1:a]volume=%.2f,afade=t=in:st=0:d=1.0,afade=t=out:st=%.2f:d=1.5[bgm0];", cfg.bgmVolume, fadeStart));
            if (cfg.ducking) {
                fc.append("[0:a]asplit=2[mixa][sc];[bgm0][sc]sidechaincompress=threshold=0.02:ratio=4:attack=20:release=400[bgm];")
                        .append("[mixa][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]");
            } else {
                fc.append("[0:a][bgm0]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]");
            }
        }
        if (fc.length() > 0) a.addAll(args("-filter_complex", fc.toString()));
        a.addAll(args("-map", burn ? "[vout]" : "0:v:0", "-map", hasBgm ? "[aout]" : "0:a:0"));
        a.addAll(burn ? encArgs(vCodec) : args("-c:v", "copy"));
        a.addAll(hasBgm ? args("-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2") : args("-c:a", "copy"));
        a.addAll(args("-t", String.format(Locale.US, "%.3f", totalDur), "-movflags", "+faststart", target.getAbsolutePath()));
        runCmd(tempDir, (int) Math.max(900, totalDur * 4), a);
    }

    private void verifyRenderedOutput(String ff, File f, double expectedDur, File tempDir) throws Exception {
        if (!f.exists() || f.length() < 2048) throw new RuntimeException("File xuất ra rỗng hoặc hỏng!");
        MediaInfo info = getMediaInfo(ff, f.getAbsolutePath());
        if (!info.hasVideo || !info.hasAudio || info.duration <= 0.5) throw new RuntimeException("Kiểm định thất bại: thiếu luồng hình hoặc tiếng!");
        if (Math.abs(info.duration - expectedDur) > 3.0)
            appendLog(String.format(Locale.US, "[CẢNH BÁO] Thời lượng xuất %.1fs lệch dự kiến %.1fs.", info.duration, expectedDur));
        runCmd(tempDir, 60, args(ff, "-v", "error", "-t", "2", "-i", f.getAbsolutePath(), "-f", "null", "-"));
        if (info.duration > 6)
            runCmd(tempDir, 60, args(ff, "-v", "error", "-sseof", "-2", "-i", f.getAbsolutePath(), "-f", "null", "-"));
    }

    // ---------- Phụ đề ----------
    private int subtitleFontPx(JobConfig cfg, boolean portrait) {
        int[] land = {44, 52, 60, 70}, port = {50, 60, 70, 82};
        return (portrait ? port : land)[Math.max(0, Math.min(3, cfg.subSizeIdx))];
    }

    private int subtitleLineChars(JobConfig cfg, boolean portrait) {
        double usable = (portrait ? 1080 : 1920) * 0.86;
        return Math.max(20, Math.min(56, (int) (usable / (subtitleFontPx(cfg, portrait) * 0.52))));
    }

    /** Chia lời bình thành 1-3 phụ đề theo tỉ lệ số ký tự để người xem dễ đọc. */
    private void addSubtitleEntries(List<SubEntry> subs, String text, double t0, double dur, int lineChars) {
        List<String> parts = splitText(text, lineChars * 2);
        if (parts.isEmpty()) return;
        int totalChars = 0;
        for (String p : parts) totalChars += p.length();
        double cur = t0 + 0.04, usable = Math.max(0.3, dur - 0.14);
        for (String p : parts) {
            double d = usable * p.length() / Math.max(1, totalChars);
            subs.add(new SubEntry(cur, cur + Math.max(0.2, d - 0.02), wrapSubtitleText(p, lineChars)));
            cur += d;
        }
    }

    private String buildAss(List<SubEntry> subs, JobConfig cfg, boolean portrait) {
        int w = portrait ? 1080 : 1920, h = portrait ? 1920 : 1080;
        int px = subtitleFontPx(cfg, portrait), marginV = portrait ? 280 : 60, marginLR = (int) (w * 0.07);
        StringBuilder sb = new StringBuilder();
        sb.append("[Script Info]\nScriptType: v4.00+\nPlayResX: ").append(w).append("\nPlayResY: ").append(h)
                .append("\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n")
                .append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
                .append("Style: Default,Arial,").append(px).append(',').append(cfg.subColorStyle)
                .append(",&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,1,2,")
                .append(marginLR).append(',').append(marginLR).append(',').append(marginV).append(",1\n\n")
                .append("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");
        for (SubEntry s : subs)
            sb.append("Dialogue: 0,").append(formatAss(s.start)).append(',').append(formatAss(s.end))
                    .append(",Default,,0,0,0,,").append(s.text.replace("{", "(").replace("}", ")")).append('\n');
        return sb.toString();
    }

    private String buildSrt(List<SubEntry> subs) {
        StringBuilder sb = new StringBuilder();
        int i = 1;
        for (SubEntry s : subs)
            sb.append(i++).append('\n').append(formatSrt(s.start)).append(" --> ").append(formatSrt(s.end)).append('\n')
                    .append(s.text.replace("\\N", "\n")).append("\n\n");
        return sb.toString();
    }

    private String formatAss(double sec) {
        long cs = Math.round(sec * 100);
        return String.format(Locale.US, "%d:%02d:%02d.%02d", cs / 360000, (cs / 6000) % 60, (cs / 100) % 60, cs % 100);
    }

    private String formatSrt(double sec) {
        long ms = Math.round(sec * 1000);
        return String.format(Locale.US, "%02d:%02d:%02d,%03d", ms / 3600000, (ms / 60000) % 60, (ms / 1000) % 60, ms % 1000);
    }

    private String wrapSubtitleText(String text, int maxLine) {
        text = text.trim();
        int lines = (int) Math.ceil(text.length() / (double) maxLine);
        if (lines > 1) maxLine = Math.min(maxLine, (int) Math.ceil(text.length() / (double) lines) + 4); // cân đối các dòng
        StringBuilder sb = new StringBuilder();
        int len = 0;
        for (String w : text.split("\\s+")) {
            if (len > 0 && len + 1 + w.length() > maxLine) { sb.append("\\N"); len = 0; }
            else if (len > 0) { sb.append(' '); len++; }
            sb.append(w);
            len += w.length();
        }
        return sb.toString();
    }

    private void writeText(File f, String content) throws IOException {
        try (Writer w = new OutputStreamWriter(new FileOutputStream(f), StandardCharsets.UTF_8)) { w.write(content); }
    }

    // =====================================================================
    //  KỊCH BẢN & GIỌNG ĐỌC
    // =====================================================================
    private List<SceneItem> parseTimestampScript(String script) {
        List<SceneItem> list = new ArrayList<>();
        String[] lines = script.split("\\r?\\n");
        Pattern linePat = Pattern.compile("^\\s*([^|]+?)\\s*\\|\\s*(.+)$");
        int idx = 0;
        for (int i = 0; i < lines.length; i++) {
            String raw = lines[i].trim();
            if (raw.isEmpty()) continue;
            Matcher m = linePat.matcher(raw);
            if (!m.matches()) throw new RuntimeException("Dòng " + (i + 1) + " sai định dạng. Cần: MM:SS - MM:SS | Lời thoại");
            String[] tp = m.group(1).trim().split("\\s*[-–—−]\\s*");
            if (tp.length != 2) throw new RuntimeException("Dòng " + (i + 1) + ": mốc thời gian phải dạng 'bắt đầu - kết thúc'.");
            double s = parseTimeSec(tp[0], TIME_PAT, i + 1), e = parseTimeSec(tp[1], TIME_PAT, i + 1);
            if (e <= s) throw new RuntimeException("Dòng " + (i + 1) + ": thời gian kết thúc phải lớn hơn bắt đầu.");
            list.add(new SceneItem(++idx, s, e, s, e, m.group(2).trim(), "Timestamp thủ công"));
        }
        if (list.isEmpty()) throw new RuntimeException("Không có dòng kịch bản hợp lệ!");
        return list;
    }

    private double parseTimeSec(String s, Pattern pat, int line) {
        Matcher m = pat.matcher(s.trim());
        if (!m.matches()) throw new RuntimeException((line > 0 ? "Dòng " + line + ": " : "") + "Thời gian \"" + s + "\" không hợp lệ!");
        double h = m.group(1) != null ? Double.parseDouble(m.group(1)) : 0;
        return h * 3600 + Double.parseDouble(m.group(2)) * 60 + Double.parseDouble(m.group(3));
    }

    private AiOpts readAiOptsSafe(JobConfig cfg) {
        AiOpts o = new AiOpts();
        o.maxScenes = cfg.maxScenes; o.strict = cfg.orderStrict; o.ollama = cfg.useOllama; o.allowNoMatch = true; o.allowFallback = false; o.refine = true;
        return o;
    }

    private List<SceneItem> buildAutoScenes(JobConfig cfg, double totalDur) {
        AiOpts o = cfg.ai != null ? cfg.ai : readAiOptsSafe(cfg);
        try {
            if (!isAiServerAlive()) throw new RuntimeException("Không kết nối được Python AI Server tại " + AI_SERVER_URL);
            appendLog("[AUTO CUT AI] Gọi Python AI Engine...");
            return requestAiMatchedScenes(cfg.videoPath, cfg.rawScript, o);
        } catch (Exception ex) {
            if (cancelRequested.get()) throw new RuntimeException("Tác vụ đã bị dừng bởi người dùng.");
            if (!o.allowFallback) {
                throw new RuntimeException("AI chọn cảnh thất bại: " + ex.getMessage()
                        + "\nChương trình KHÔNG tự chia đều phim vì sẽ ra video sai cảnh. Hãy sửa lỗi AI rồi chạy lại,"
                        + " hoặc bật 'Cho phép chế độ dự phòng' trong Cài đặt nếu vẫn muốn.", ex);
            }
            appendLog("[CẢNH BÁO AI] " + ex.getMessage() + " -> chia cảnh đều (chất lượng thấp, theo cài đặt của bạn).");
        }
        List<String> sentences = extractCleanSentencesForAI(cfg.rawScript);
        List<SceneItem> res = new ArrayList<>();
        double st = totalDur * 0.04, span = Math.max(1.0, totalDur * 0.94 - st);
        int n = Math.max(1, sentences.size());
        for (int i = 0; i < sentences.size(); i++) {
            double p = st + i * span / n, e = Math.min(totalDur, p + 5.0);
            res.add(new SceneItem(i + 1, p, e, p, e, sentences.get(i), "[Khớp thấp] Auto Fallback (AI tắt, chia đều)"));
        }
        return res;
    }

    private void generateAllVoices(String ff, JobConfig cfg, File tempDir, File[] voices, double[] durs) throws Exception {
        int total = cfg.scenes.size();
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, Math.min(4, total)));
        AtomicInteger done = new AtomicInteger();
        List<Future<Double>> fs = new ArrayList<>();
        try {
            for (int i = 0; i < total; i++) {
                final int idx = i;
                voices[i] = new File(tempDir, String.format("voice_%03d.wav", i));
                fs.add(pool.submit(() -> {
                    checkCancelled();
                    generateSceneTTS(ff, cfg, cfg.scenes.get(idx).text, voices[idx], tempDir, idx);
                    double d = getMediaInfo(ff, voices[idx].getAbsolutePath()).duration;
                    if (d <= 0.1) throw new RuntimeException("Giọng đọc cảnh " + (idx + 1) + " bị lỗi thời lượng!");
                    int n = done.incrementAndGet();
                    updateProgress("Đang tạo giọng đọc " + n + "/" + total + "...", 8 + (int) (n * 22.0 / total));
                    return d;
                }));
            }
            for (int i = 0; i < total; i++) durs[i] = fs.get(i).get();
        } catch (ExecutionException ee) {
            Throwable c = ee.getCause();
            if (c instanceof Exception) throw (Exception) c;
            throw ee;
        } finally {
            pool.shutdownNow();
        }
    }

    private void generateSceneTTS(String ff, JobConfig cfg, String text, File outWav, File tempDir, int idx) throws Exception {
        List<String> chunks = splitText(text, cfg.ttsEngine == 0 ? 400 : 110);
        if (chunks.isEmpty()) throw new RuntimeException("Lời bình cảnh " + (idx + 1) + " trống!");
        List<File> parts = new ArrayList<>();
        for (int c = 0; c < chunks.size(); c++) {
            checkCancelled();
            File mp3 = new File(tempDir, String.format("tts_%d_%d.mp3", idx, c));
            synthChunk(cfg, chunks.get(c), mp3);
            parts.add(mp3);
        }
        File list = new File(tempDir, "tts_list_" + idx + ".txt");
        try (PrintWriter pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(list), StandardCharsets.UTF_8))) {
            for (File p : parts) pw.println("file '" + p.getAbsolutePath().replace("\\", "/").replace("'", "'\\''") + "'");
        }
        runCmd(tempDir, 90, args(ff, "-y", "-nostdin", "-f", "concat", "-safe", "0", "-i", list.getAbsolutePath(),
                "-af", "dynaudnorm=f=150:g=11,apad=pad_dur=0.22", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", outWav.getAbsolutePath()));
    }

    private void synthChunk(JobConfig cfg, String chunk, File out) throws Exception {
        if (cfg.ttsEngine == 0 && !edgeTtsBroken.get()) {
            try { synthCached("edge", cfg, chunk, out); return; }
            catch (Exception ex) {
                if (cancelRequested.get()) throw ex;
                if (edgeTtsBroken.compareAndSet(false, true))
                    appendLog("[TTS] Edge TTS lỗi (" + ex.getMessage() + ") -> chuyển sang Google Translate TTS.");
            }
        }
        synthCached("google", cfg, chunk, out);
    }

    private void synthCached(String engine, JobConfig cfg, String chunk, File out) throws Exception {
        String voice = EDGE_VOICES[Math.max(0, Math.min(EDGE_VOICES.length - 1, cfg.ttsVoice))];
        String rate = String.format(Locale.US, "%+d%%", cfg.ttsRatePercent);
        String key = sha1(engine + "|" + ("edge".equals(engine) ? voice + "|" + rate : "") + "|" + chunk);
        File dir = new File(System.getProperty("user.home"), ".review_studio" + File.separator + "tts_cache");
        dir.mkdirs();
        File cached = new File(dir, key + ".mp3");
        if (!(cached.isFile() && cached.length() > 500)) {
            File tmp = new File(dir, key + "." + Thread.currentThread().getId() + ".tmp");
            try {
                if ("edge".equals(engine)) downloadEdgeTts(chunk, voice, rate, tmp);
                else downloadGoogleTts(chunk, tmp, 3);
                Files.move(tmp.toPath(), cached.toPath(), StandardCopyOption.REPLACE_EXISTING);
            } finally { tmp.delete(); }
        }
        Files.copy(cached.toPath(), out.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }

    private void downloadEdgeTts(String text, String voice, String rate, File out) throws Exception {
        String body = "{\"text\":" + Json.quote(text) + ",\"voice\":" + Json.quote(voice) + ",\"rate\":" + Json.quote(rate) + "}";
        HttpResult r = http("POST", "/tts", body, 5000, 60000);
        if (r.code != 200 || r.body.length < 500) throw new IOException("HTTP " + r.code + " " + (r.code == 200 ? "" : serverError(r)));
        Files.write(out.toPath(), r.body);
    }

    private void downloadGoogleTts(String chunk, File out, int retries) throws Exception {
        for (int attempt = 1; attempt <= retries; attempt++) {
            checkCancelled();
            HttpURLConnection conn = null;
            try {
                String u = "https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=vi&q=" + URLEncoder.encode(chunk, "UTF-8");
                conn = (HttpURLConnection) new URL(u).openConnection();
                conn.setRequestProperty("User-Agent", "Mozilla/5.0");
                conn.setConnectTimeout(7000);
                conn.setReadTimeout(10000);
                if (conn.getResponseCode() != 200) throw new IOException("HTTP " + conn.getResponseCode());
                try (InputStream in = conn.getInputStream()) { Files.copy(in, out.toPath(), StandardCopyOption.REPLACE_EXISTING); }
                if (out.length() > 200) return;
                throw new IOException("Dữ liệu âm thanh rỗng");
            } catch (Exception ex) {
                if (cancelRequested.get() || attempt == retries) throw ex;
                Thread.sleep(700L * attempt);
            } finally {
                if (conn != null) conn.disconnect();
            }
        }
    }

    private String sha1(String s) throws Exception {
        StringBuilder sb = new StringBuilder();
        for (byte b : MessageDigest.getInstance("SHA-1").digest(s.getBytes(StandardCharsets.UTF_8))) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    /** Tách câu theo dấu câu; mảnh nào vẫn quá dài thì cắt theo từ (không bao giờ vượt max). */
    private List<String> splitText(String text, int max) {
        List<String> out = new ArrayList<>();
        for (String s : text.replaceAll("\\s+", " ").trim().split("(?<=[.!?…;])\\s+")) {
            if (s.trim().isEmpty()) continue;
            if (s.length() <= max) { out.add(s.trim()); continue; }
            for (String c : s.split("(?<=[,:])\\s+")) {
                if (c.trim().isEmpty()) continue;
                if (c.length() <= max) { out.add(c.trim()); continue; }
                StringBuilder cur = new StringBuilder();
                for (String w : c.trim().split(" ")) {
                    if (cur.length() > 0 && cur.length() + 1 + w.length() > max) { out.add(cur.toString()); cur.setLength(0); }
                    if (cur.length() > 0) cur.append(' ');
                    cur.append(w);
                }
                if (cur.length() > 0) out.add(cur.toString());
            }
        }
        return out;
    }

    // =====================================================================
    //  FFMPEG & TIẾN TRÌNH
    // =====================================================================
    private static List<String> args(Object... parts) {
        List<String> l = new ArrayList<>();
        for (Object o : parts) {
            if (o instanceof Collection) for (Object x : (Collection<?>) o) l.add(String.valueOf(x));
            else l.add(String.valueOf(o));
        }
        return l;
    }

    /** Lệnh ffmpeg: phần tử đầu tiên là đường dẫn ffmpeg, tự chèn nếu chưa có. */
    private void runCmd(File dir, int timeoutSec, List<String> cmd) throws Exception {
        checkCancelled();
        List<String> full = new ArrayList<>(cmd);
        if (!full.isEmpty() && full.get(0).startsWith("-")) full.add(0, currentFfmpeg);
        full.addAll(1, Arrays.asList("-hide_banner"));
        ProcessBuilder pb = new ProcessBuilder(full);
        if (dir != null) pb.directory(dir);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        activeProcesses.add(p);

        List<String> tail = new ArrayList<>();
        Thread t = new Thread(() -> {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                String l;
                while ((l = r.readLine()) != null) synchronized (tail) { if (tail.size() > 30) tail.remove(0); tail.add(l); }
            } catch (Exception ignored) { }
        });
        t.setDaemon(true);
        t.start();
        try {
            if (!p.waitFor(timeoutSec, TimeUnit.SECONDS)) {
                killProcessTree(p);
                throw new RuntimeException("Quá thời gian xử lý (" + timeoutSec + "s)!");
            }
            checkCancelled();
            if (p.exitValue() != 0) {
                t.join(1000);
                StringBuilder sb = new StringBuilder();
                synchronized (tail) { for (String s : tail) sb.append(s).append('\n'); }
                appendLog("[CHI TIẾT LỖI FFMPEG]\n" + sb);
                throw new RuntimeException("FFmpeg lỗi (mã " + p.exitValue() + "). Xem tab Nhật Ký.");
            }
        } finally {
            activeProcesses.remove(p);
        }
    }

    private volatile String currentFfmpeg = "ffmpeg";

    private boolean verifyNvencRealEncode(String ff, File tempDir) {
        Process p = null;
        try {
            List<String> cmd = args(ff, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2", encArgs("h264_nvenc"), "-f", "null", "-");
            p = new ProcessBuilder(cmd).directory(tempDir).redirectErrorStream(true).start();
            activeProcesses.add(p);
            Process fp = p;
            Thread drain = new Thread(() -> { try { fp.getInputStream().readAllBytes(); } catch (IOException ignored) { } });
            drain.setDaemon(true);
            drain.start();
            return p.waitFor(10, TimeUnit.SECONDS) && p.exitValue() == 0;
        } catch (Exception e) {
            return false;
        } finally {
            if (p != null) { activeProcesses.remove(p); if (p.isAlive()) p.destroyForcibly(); }
        }
    }

    private MediaInfo getMediaInfo(String ff, String path) throws Exception {
        checkCancelled();
        Process p = new ProcessBuilder(ff, "-hide_banner", "-nostdin", "-i", path).redirectErrorStream(true).start();
        activeProcesses.add(p);
        Pattern dur = Pattern.compile("Duration:\\s*(\\d+):(\\d+):([\\d.]+)");
        double d = 0;
        boolean v = false, a = false;
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                Matcher m = dur.matcher(line);
                if (m.find()) d = Double.parseDouble(m.group(1)) * 3600 + Double.parseDouble(m.group(2)) * 60 + Double.parseDouble(m.group(3));
                if (line.contains("Video:") && !line.contains("attached pic")) v = true;
                if (line.contains("Audio:")) a = true;
            }
        } finally {
            p.waitFor(8, TimeUnit.SECONDS);
            if (p.isAlive()) p.destroyForcibly();
            activeProcesses.remove(p);
        }
        return new MediaInfo(d, v, a);
    }

    private void checkCancelled() {
        if (cancelRequested.get()) throw new RuntimeException("Tác vụ đã bị dừng bởi người dùng.");
    }

    private String findFFmpegExecutable() {
        String found = null;
        boolean win = System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win");
        String exe = win ? "ffmpeg.exe" : "ffmpeg";
        for (File f : new File[]{new File(exe), new File("..", exe), new File("bin", exe)}) {
            if (f.isFile()) { found = f.getAbsolutePath(); break; }
        }
        if (found == null) {
            try {
                Process p = new ProcessBuilder("ffmpeg", "-version").redirectErrorStream(true).start();
                p.getInputStream().readAllBytes();
                if (p.waitFor(3, TimeUnit.SECONDS) && p.exitValue() == 0) found = "ffmpeg";
            } catch (Exception ignored) { }
        }
        if (found != null) currentFfmpeg = found;
        return found;
    }

    private void deleteDirectory(File dir) {
        if (dir != null && dir.isDirectory()) {
            File[] fs = dir.listFiles();
            if (fs != null) for (File f : fs) deleteDirectory(f);
            dir.delete();
        } else if (dir != null) dir.delete();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try { UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); } catch (Exception ignored) { }
            new Main().setVisible(true);
        });
    }
}
