# v3.2 – Chọn cảnh theo chuỗi liền mạch (sửa các lỗi trong tài liệu phân tích v3.1 vs v3.0)

## Nguyên nhân v3.1 chọn sai và cách sửa
| Lỗi trong v3.1 | Cách sửa v3.2 |
|---|---|
| Điểm z nhân hệ số bằng chứng thô (0.35+0.65e) rồi so với `null_score`=1.25 → hầu hết câu "rơi" vào null | **Bỏ**. Điểm chọn = z-score (+ prior nhẹ + điểm cộng) như v3.0; bằng chứng thô chỉ dùng để gắn nhãn |
| Trạng thái null chi phí 0 → mất trí nhớ thứ tự; `if is_null: j = ranked[0]` lấy cảnh cao nhất toàn phim | **Bỏ trạng thái null**; chuỗi Viterbi luôn liền mạch |
| Phạt cảnh giống nhau (DUP_SIM 0.93) + cửa sổ cấm lặp 6 câu → bị đẩy sang cảnh khác | **Tắt** (`REUSE_WINDOW=0`, không truyền `dup_sim`). Chỉ phạt dùng lại *đúng một* cảnh liên tiếp (2.0) |
| Phạt nhảy tiến xa | **Bỏ** (hệ số 0; chỉ còn chi phí 0.003 để phá hòa) |
| Prior vị trí 0.45 ép cảnh trượt đều theo tỷ lệ câu | Giảm còn 0.15 (medium), 0 (loose/free), 0.35 (strict) |
| Tách mệnh đề tạo vector nhiễu | **Tắt mặc định** (`USE_CLAUSES=1` để bật lại) |
| Điểm cộng hành động/khuôn mặt bị khóa sau `gate` gần như luôn = 0 | Khôi phục như v3.0 (+0.45 theo chuyển động, +0.2 khuôn mặt) |
| z-score phóng đại nhiễu của câu trừu tượng/tên riêng | Câu bị phát hiện là trừu tượng → hàng điểm làm **phẳng**, cảnh chọn **theo mạch** (liền sau câu trước), gắn nhãn "Câu trừu tượng"; không còn hộp thoại chặn |
| `max_scenes=240` gộp mất ranh giới cảnh | Mặc định **600** (tối đa 1500), khóa cài đặt mới nên không dính giá trị 240 cũ |
| Java cắt lấn ±1.5s và kéo clip vượt ranh giới khi lời thoại dài | Không bao giờ vượt ranh giới cảnh; lời thoại dài → chậm tối đa 1.42× rồi giữ khung cuối |
| Rơi âm thầm sang chia đều / caption heuristic khi AI lỗi | Server trả **503** `clip_unavailable`, Java **dừng và báo lỗi**; chỉ chạy dự phòng nếu bật "Cho phép chế độ dự phòng" (mặc định tắt) |

Giữ nguyên từ v3.1: giao diện, xem trước, **🔁 AI tìm cảnh khác** (`/rematch`), `job_id`, cache an toàn, dò kỹ thời điểm.
Nhãn "Khớp tốt / Cần kiểm tra / Khớp thấp" chỉ để thông tin, **không** ảnh hưởng việc chọn cảnh.

## Kiểm chứng
- `pytest -q test_ai_service.py` → 29 test (gồm: góc máy liền kề cùng bối cảnh không bị phạt, nhảy tiến xa miễn phí, câu trừu tượng đi theo mạch, 503 khi CLIP lỗi).
- `python simulate.py`: mô phỏng 60 kịch bản × 240 cảnh (cosine thô 0.14–0.22, góc máy kề giống nhau, bỏ qua đoạn giữa, ~20% câu trừu tượng):
  v3.1 đúng 80.2% (±2 cảnh) / 68.1% (chính xác) → v3.2 **94.1% / 84.6%**; v3.1 rơi vào null trung bình 17 lần/kịch bản.

## Lưu ý thật thà
- Mô phỏng dùng giả định số liệu của mình, **không phải phim thật**: nó chứng minh logic chọn cảnh không còn tự phá chuỗi, không chứng minh độ chính xác tuyệt đối của CLIP trên phim của bạn.
- Bộ phát hiện câu trừu tượng cố ý thận trọng (trong mô phỏng bắt ~22%, nhầm câu thật 0%). Nó chỉ ảnh hưởng nhãn/tín hiệu nhiễu, không phải lý do chính của việc chọn đúng.
- `max_scenes` cao → phân tích lần đầu lâu hơn (mã hóa nhiều khung hình), lần sau dùng cache.
- Vẫn chưa chạy CLIP thật và chưa xem trực quan giao diện Swing.
