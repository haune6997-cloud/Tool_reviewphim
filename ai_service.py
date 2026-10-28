"""
Movie Review AI Engine v3.2 (Hybrid Core v3.0 + TTS Fix)
========================================================
- Giữ nguyên thuật toán khớp cảnh CLIP + Viterbi DP của v3.0 (độ chính xác thực tế cao nhất).
- Nâng cấp Viterbi chống lặp cảnh xen kẽ (A -> B -> A).
- Sửa triệt để lỗi đổi giọng lồng tiếng (/tts): tự động map tên giọng từ Java UI,
  tự chuẩn hóa rate/pitch không gây lỗi 400, làm sạch ký tự lỗi và chống cache âm thanh cũ.
- Tương thích 100% cấu trúc dữ liệu JSON với Main.java (v3.0 -> v3.2).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("MovieReviewAI")

ALGO_VERSION = "3.2_clip_dp_v3core"
CLIP_IMG_MODEL = os.getenv("CLIP_IMG_MODEL", "clip-ViT-B-32")
CLIP_TXT_MODEL = os.getenv("CLIP_TXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
TEXT_MODEL = os.getenv("TEXT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")

# Thông số chuẩn từ v3.0 giúp chọn cảnh mượt nhất
FRAME_FRACTIONS = (0.15, 0.38, 0.62, 0.85)
MIN_SCENE_SEC = 1.5
MAX_SCENE_SEC = 14.0
SPLIT_CHUNK_SEC = 7.0
PRIOR_WEIGHT = 0.45
REUSE_PENALTY = 3.0          # Phạt nặng khi 2 câu liên tiếp dùng cùng 1 cảnh
GLOBAL_REUSE_PENALTY = 0.85  # Phạt bổ sung khi dùng lại cảnh đã xuất hiện ở các câu trước đó
DEFAULT_BACK_PENALTY = 1.2   # Phạt khi nhảy lùi về cảnh trước đó trong phim

DATA_DIR = Path("ai_cache")
DATA_DIR.mkdir(exist_ok=True)

# Danh sách giọng chuẩn hỗ trợ đổi giọng từ Java UI
SUPPORTED_VOICES = {
    "hoaimy": "vi-VN-HoaiMyNeural",
    "namminh": "vi-VN-NamMinhNeural",
    "female": "vi-VN-HoaiMyNeural",
    "male": "vi-VN-NamMinhNeural",
    "nu": "vi-VN-HoaiMyNeural",
    "nữ": "vi-VN-HoaiMyNeural",
    "nam": "vi-VN-NamMinhNeural",
    "jenny": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",
    "yunxi": "zh-CN-YunxiNeural",
}


# ----------------------------------------------------------------------------
# Trạng thái dùng chung (an toàn đa luồng)
# ----------------------------------------------------------------------------
class Progress:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = {
            "status": "idle",
            "percent": 0,
            "message": "Sẵn sàng",
            "vision_mode": "unknown",
        }

    def set(self, status: str, percent: int, message: str, vision_mode: Optional[str] = None) -> None:
        with self._lock:
            self._data["status"] = status
            self._data["percent"] = int(max(0, min(100, percent)))
            self._data["message"] = message
            if vision_mode:
                self._data["vision_mode"] = vision_mode

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)


PROGRESS = Progress()
CANCEL = threading.Event()
JOB_LOCK = threading.Lock()


class JobCancelled(Exception):
    pass


def check_cancel() -> None:
    if CANCEL.is_set():
        raise JobCancelled()


# ----------------------------------------------------------------------------
# Quản lý mô hình (nạp lười + dự phòng)
# ----------------------------------------------------------------------------
class Models:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clip_img = None
        self.clip_txt = None
        self.text = None
        self.clip_error: Optional[str] = None
        self.state = "not_loaded"  # not_loaded | loading | ready | fallback

    def load_clip(self) -> bool:
        with self.lock:
            if self.clip_img is not None and self.clip_txt is not None:
                return True
            if self.clip_error:
                return False
            self.state = "loading"
            try:
                from sentence_transformers import SentenceTransformer
                log.info("Đang nạp CLIP đa ngôn ngữ (lần đầu sẽ tải ~600MB)...")
                self.clip_img = SentenceTransformer(CLIP_IMG_MODEL)
                self.clip_txt = SentenceTransformer(CLIP_TXT_MODEL)
                self.state = "ready"
                log.info("CLIP đã sẵn sàng.")
                return True
            except Exception as ex:  # noqa: BLE001
                self.clip_error = str(ex)
                self.clip_img = self.clip_txt = None
                self.state = "fallback"
                log.warning("Không nạp được CLIP (%s). Dùng chế độ dự phòng caption.", ex)
                return False

    def get_text_model(self):
        with self.lock:
            if self.text is None:
                from sentence_transformers import SentenceTransformer
                log.info("Đang nạp mô hình văn bản %s...", TEXT_MODEL)
                self.text = SentenceTransformer(TEXT_MODEL)
            return self.text


MODELS = Models()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=MODELS.load_clip, daemon=True).start()
    yield


app = FastAPI(title="Movie Review AI Engine v3.2", lifespan=lifespan)

face_cascade = None
try:
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception as _ex:
    log.warning("Bỏ qua Haar Cascade: %s", _ex)
    face_cascade = None


# ----------------------------------------------------------------------------
# Schema Request (Cho phép trường mở rộng từ Main.java mà không báo lỗi 422)
# ----------------------------------------------------------------------------
class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video_path: str
    sentences: list[str] = Field(..., min_length=1)
    max_scenes: int = 240
    use_ollama: bool = False
    order_penalty: float = DEFAULT_BACK_PENALTY


class TTSRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=1)
    voice: Optional[str] = "vi-VN-HoaiMyNeural"
    voice_id: Optional[str] = None      # Dự phòng nếu Java gửi key voice_id
    voiceName: Optional[str] = None     # Dự phòng nếu Java gửi key voiceName
    rate: Any = "+0%"
    speed: Any = None                   # Dự phòng nếu Java gửi key speed (vd: 1.1)
    pitch: Any = "+0Hz"
    output_path: Optional[str] = None   # Dự phòng nếu Java truyền đường dẫn lưu file


# ----------------------------------------------------------------------------
# Bộ chuẩn hóa tham số TTS (Chống lỗi đổi giọng từ Main.java)
# ----------------------------------------------------------------------------
def resolve_voice(raw_voice: Optional[str]) -> str:
    """Chuyển đổi mọi định dạng tên giọng từ Java ComboBox về chuẩn Microsoft Edge TTS."""
    if not raw_voice or not str(raw_voice).strip():
        return "vi-VN-HoaiMyNeural"

    v = str(raw_voice).strip()

    # 1. Nếu chuỗi đã chứa mã chuẩn kiểu xx-XX-NameNeural (kể cả có mô tả phía sau như "vi-VN-NamMinhNeural (Nam)")
    m = re.search(r"([a-z]{2}-[A-Z]{2}-[A-Za-z0-9]+Neural)", v)
    if m:
        return m.group(1)

    # 2. Kiểm tra theo từ khóa tiếng Việt / tên rút gọn
    low = v.lower().replace("_", "").replace("-", "").replace(" ", "")
    if "namminh" in low or "giọngnam" in low or "giongnam" in low:
        return "vi-VN-NamMinhNeural"
    if "hoaimy" in low or "hoàimy" in low or "giọngnữ" in low or "giongnu" in low:
        return "vi-VN-HoaiMyNeural"

    for key, target in SUPPORTED_VOICES.items():
        if key in v.lower():
            return target

    return "vi-VN-HoaiMyNeural"


def normalize_rate(rate_val: Any, speed_val: Any = None) -> str:
    """Chuyển đổi tốc độ đọc từ bất kỳ định dạng nào (1.2, '10%', '+10%', -5) sang '+10%'."""
    val = speed_val if speed_val is not None else rate_val
    if val is None:
        return "+0%"

    s = str(val).strip().replace(" ", "")
    if not s or s in ("0", "0%", "+0%", "-0%", "1.0", "1"):
        return "+0%"

    # Nếu đã đúng chuẩn [+-]dd%
    if re.fullmatch(r"[+-]\d{1,3}%", s):
        return s

    # Nếu thiếu dấu + ở đầu (vd: "15%")
    if re.fullmatch(r"\d{1,3}%", s):
        return f"+{s}"

    # Nếu là số thực kiểu hệ số tốc độ (vd: 1.2 -> +20%, 0.85 -> -15%) hoặc phần trăm nguyên (vd: 15 -> +15%)
    try:
        num = float(s.rstrip("%"))
        if not s.endswith("%") and 0.25 <= num <= 3.0 and ("." in s or num <= 2.0):
            pct = int(round((num - 1.0) * 100))
        else:
            pct = int(round(num))
        pct = max(-80, min(150, pct))
        return f"{pct:+d}%"
    except ValueError:
        return "+0%"


def normalize_pitch(pitch_val: Any) -> str:
    """Chuyển đổi cao độ từ bất kỳ định dạng nào ('0Hz', 5, '-2Hz', '+0%') sang '+0Hz'."""
    if pitch_val is None:
        return "+0Hz"

    s = str(pitch_val).strip().replace(" ", "")
    if not s or s in ("0", "0Hz", "+0Hz", "-0Hz", "0%", "+0%"):
        return "+0Hz"

    if re.fullmatch(r"[+-]\d{1,3}Hz", s, re.IGNORECASE):
        return s[:-2] + "Hz"

    if re.fullmatch(r"\d{1,3}Hz", s, re.IGNORECASE):
        return f"+{s[:-2]}Hz"

    try:
        num = int(round(float(re.sub(r"[^0-9.+-]", "", s))))
        num = max(-100, min(100, num))
        return f"{num:+d}Hz"
    except ValueError:
        return "+0Hz"


def clean_tts_text(text: str) -> str:
    """Làm sạch văn bản để tránh ngắt kết nối WebSocket của Edge-TTS."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ----------------------------------------------------------------------------
# Tiện ích video
# ----------------------------------------------------------------------------
def validate_video_file(video_path: str) -> dict:
    if not video_path or not os.path.isfile(video_path):
        raise HTTPException(status_code=400, detail="File video không tồn tại!")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail="Không thể mở hoặc giải mã file video!")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None or frame_count <= 10:
        raise HTTPException(status_code=400, detail="Video bị hỏng hoặc không chứa khung hình hợp lệ!")
    return {"fps": float(fps), "frame_count": frame_count, "duration": frame_count / fps}


def cache_key(video_path: str, max_scenes: int, mode: str, use_ollama: bool) -> str:
    p = Path(video_path).resolve()
    st = p.stat()
    raw = (
        f"{str(p).lower()}|{st.st_size}|{st.st_mtime_ns}|{ALGO_VERSION}|"
        f"{CLIP_IMG_MODEL}|{CLIP_TXT_MODEL}|{max_scenes}|{mode}|{int(use_ollama)}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def check_ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.5)
        if r.status_code == 200:
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(k in n for n in names for k in ("qwen", "llava", "moondream", "minicpm", "gemma3"))
    except Exception:  # noqa: BLE001
        pass
    return False


def build_intervals(raw_scenes: list[tuple[float, float]], total_dur: float, max_scenes: int) -> list[tuple[float, float]]:
    """Gộp cảnh quá ngắn, chia cảnh quá dài, gộp bớt nếu vượt max_scenes (không bỏ rơi nội dung)."""
    if not raw_scenes:
        step = max(4.0, total_dur / max_scenes)
        raw_scenes = []
        t = 0.0
        while t < total_dur - 1.0:
            raw_scenes.append((t, min(total_dur, t + step)))
            t += step

    merged: list[list[float]] = []
    for s, e in raw_scenes:
        if merged and (e - s) < MIN_SCENE_SEC:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < MIN_SCENE_SEC:
        merged[1][0] = merged[0][0]
        merged.pop(0)

    out: list[tuple[float, float]] = []
    for s, e in merged:
        if e - s > MAX_SCENE_SEC:
            n = int(np.ceil((e - s) / SPLIT_CHUNK_SEC))
            edges = np.linspace(s, e, n + 1)
            out.extend((float(edges[k]), float(edges[k + 1])) for k in range(n))
        else:
            out.append((float(s), float(e)))

    while len(out) > max_scenes:
        durs = np.array([e - s for s, e in out])
        pair = durs[:-1] + durs[1:]
        i = int(np.argmin(pair))
        out[i] = (out[i][0], out[i + 1][1])
        out.pop(i + 1)
    return out


def grab_frames(cap: cv2.VideoCapture, s: float, e: float, fps: float) -> tuple[list[np.ndarray], list[float]]:
    dur = max(0.5, e - s)
    frames, times = [], []
    for f in FRAME_FRACTIONS:
        t = s + dur * f
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if ok and fr is not None:
            h, w = fr.shape[:2]
            frames.append(cv2.resize(fr, (384, max(2, int(h * 384.0 / w)))))
            times.append(t)
    return frames, times


def visual_features(frames: list[np.ndarray]) -> dict:
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [float(np.mean(cv2.absdiff(grays[i], grays[i + 1]))) for i in range(len(grays) - 1)]
    mid = grays[len(grays) // 2]
    face_cnt = 0
    if face_cascade is not None:
        try:
            faces = face_cascade.detectMultiScale(mid, 1.2, 4)
            face_cnt = len(faces)
        except Exception:
            face_cnt = 0
    return {
        "motion": round(float(np.mean(diffs)) if diffs else 0.0, 2),
        "brightness": round(float(np.mean(mid)), 1),
        "edges": float(np.mean(cv2.Canny(mid, 80, 160))),
        "faces": int(face_cnt),
    }


def heuristic_caption(feat: dict, rel_pos: float) -> str:
    arc = (
        "mở đầu giới thiệu bối cảnh và nhân vật" if rel_pos < 0.18 else
        "diễn biến phát triển tình tiết" if rel_pos < 0.55 else
        "cao trào xung đột căng thẳng" if rel_pos < 0.85 else
        "đoạn kết giải quyết và hé lộ"
    )
    act = (
        "hành động nhanh, giao chiến, tấn công, né tránh" if feat["motion"] > 28 else
        "nhân vật di chuyển, truy đuổi, chuẩn bị đối phó" if feat["motion"] > 14 else
        "cảnh tĩnh, đối thoại, quan sát, suy nghĩ"
    )
    shot = (
        "cận cảnh nhiều nhân vật" if feat["faces"] >= 2 else
        "cận cảnh gương mặt nhân vật" if feat["faces"] == 1 else
        "khung hình nhiều chi tiết máy móc, vật thể" if feat["edges"] > 25 else
        "toàn cảnh không gian"
    )
    tone = "bối cảnh tối, kịch tính" if feat["brightness"] < 75 else "khung hình sáng rõ"
    return f"{arc}; {act}; {shot}; {tone}."


def describe_with_ollama(frames: list[np.ndarray]) -> Optional[str]:
    imgs = []
    for f in frames[:2]:
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            imgs.append(base64.b64encode(buf.tobytes()).decode())
    if not imgs:
        return None
    for _ in range(2):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "stream": False,
                    "images": imgs,
                    "prompt": "Mô tả ngắn gọn bằng tiếng Việt (1-2 câu): nhân vật/vật thể nào, đang làm gì, ở đâu, không khí ra sao?",
                },
                timeout=40,
            )
            if r.status_code == 200:
                txt = r.json().get("response", "").strip()
                if len(txt) > 10:
                    return txt
        except Exception as ex:  # noqa: BLE001
            log.warning("Ollama lỗi: %s", ex)
    return None


# ----------------------------------------------------------------------------
# Phân tích phim -> danh sách cảnh + embedding
# ----------------------------------------------------------------------------
def analyze_video(video_path: str, meta: dict, max_scenes: int, use_clip: bool, use_ollama: bool):
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    fps, total_dur = meta["fps"], meta["duration"]
    PROGRESS.set("analyzing", 6, "Đang dò điểm chuyển cảnh (PySceneDetect)...")
    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=max(12, int(fps * 0.8))))
    sm.detect_scenes(video, show_progress=False)
    raw = [(a.get_seconds(), b.get_seconds()) for a, b in sm.get_scene_list()]
    check_cancel()

    intervals = build_intervals(raw, total_dur, max_scenes)
    ollama_ready = use_ollama and check_ollama_available()
    vision_mode = (
        ("clip+ollama" if ollama_ready else "clip")
        if use_clip
        else ("ollama_vision" if ollama_ready else "opencv_visual_heuristics")
    )
    log.info("Chế độ: %s | số cảnh: %d", vision_mode, len(intervals))

    cap = cv2.VideoCapture(video_path)
    scenes: list[dict] = []
    emb_chunks: list[np.ndarray] = []
    pending_imgs, pending_idx = [], []
    n_frames = len(FRAME_FRACTIONS)
    img_emb = None
    n = len(intervals)

    def flush():
        nonlocal pending_imgs, pending_idx
        if not pending_imgs:
            return
        vecs = MODELS.clip_img.encode(
            pending_imgs,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        k = 0
        for cnt in pending_idx:
            block = vecs[k : k + cnt]
            k += cnt
            if cnt < n_frames:
                block = np.concatenate([block, np.repeat(block[-1:], n_frames - cnt, axis=0)], axis=0)
            emb_chunks.append(block)
        pending_imgs.clear()
        pending_idx.clear()

    from PIL import Image

    for i, (s, e) in enumerate(intervals):
        check_cancel()
        PROGRESS.set("analyzing", 10 + int(55 * i / max(1, n)), f"Đang phân tích hình ảnh cảnh {i + 1}/{n}...", vision_mode)
        frames, times = grab_frames(cap, s, e, fps)
        if not frames:
            continue
        while len(times) < n_frames:
            times.append(times[-1])
        feat = visual_features(frames)
        rel = ((s + e) / 2.0) / max(total_dur, 1.0)
        caption = heuristic_caption(feat, rel)
        status = "heuristic"
        if ollama_ready:
            d = describe_with_ollama(frames)
            if d:
                caption, status = d, "ollama_ok"
        if use_clip:
            pending_imgs.extend(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames)
            pending_idx.append(len(frames))
            status = "clip" if status == "heuristic" else "clip+" + status
            if len(pending_idx) >= 16:
                flush()
        scenes.append({
            "scene_id": f"sc_{i:04d}",
            "start_time": round(s, 2),
            "end_time": round(e, 2),
            "frame_times": [round(t, 2) for t in times[:n_frames]],
            "motion_score": feat["motion"],
            "face_count": feat["faces"],
            "caption": caption,
            "vision_status": status,
        })
    cap.release()
    if use_clip:
        PROGRESS.set("embedding", 68, "Đang mã hóa hình ảnh bằng CLIP...", vision_mode)
        flush()
        img_emb = np.stack(emb_chunks).astype(np.float32) if emb_chunks else None
    return scenes, img_emb, vision_mode


# ----------------------------------------------------------------------------
# Khớp câu <-> cảnh (Lõi v3.0 + Nâng cấp chống lặp cảnh xa)
# ----------------------------------------------------------------------------
def zscore_rows(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)


ACTION_WORDS = (
    "chiến đấu", "tấn công", "phản công", "truy đuổi", "chạy", "đánh", "tốc độ", "đòn",
    "lao vào", "kích hoạt", "nổ", "va chạm", "đuổi theo", "chiến thắng", "bắn", "chém",
    "né tránh", "giao tranh", "đối đầu", "tung chiêu", "hạ gục", "vũ khí", "cứu",
)
CALM_WORDS = (
    "nhớ lại", "suy nghĩ", "bí mật", "quá khứ", "cha", "mẹ", "ước mơ", "quan sát",
    "nhận ra", "trò chuyện", "tâm sự", "khóc", "buồn", "hồi tưởng", "lặng lẽ", "nói",
)


def visual_bonus_matrix(sentences: list[str], scenes: list[dict]) -> np.ndarray:
    motion = np.array([s["motion_score"] for s in scenes], dtype=np.float32)
    faces = np.array([s["face_count"] for s in scenes], dtype=np.float32)
    out = np.zeros((len(sentences), len(scenes)), dtype=np.float32)
    for i, sent in enumerate(sentences):
        low = sent.lower()
        if any(w in low for w in ACTION_WORDS):
            out[i] += np.minimum(0.45, motion / 60.0)
        if any(w in low for w in CALM_WORDS):
            out[i] += 0.2 * (faces >= 1)
    return out


def monotonic_viterbi(score: np.ndarray, back_penalty: float) -> list[int]:
    """
    Quy hoạch động Viterbi của v3.0 kết hợp bộ nhớ lịch sử đường đi:
    - Giữ mạch thời gian tự nhiên của phim.
    - Phạt nặng khi lặp cảnh sát nhau (REUSE_PENALTY) và phạt tích lũy nếu dùng lại cảnh cũ (GLOBAL_REUSE_PENALTY).
    """
    n, m = score.shape
    idx = np.arange(m)
    trans = np.zeros((m, m), dtype=np.float32)  # trans[k, j]: chi phí chuyển từ cảnh k sang cảnh j
    back = idx[None, :] < idx[:, None]
    trans[back] = back_penalty + 0.004 * (idx[:, None] - idx[None, :])[back]
    trans[idx, idx] = REUSE_PENALTY

    dp = score[0].copy()
    ptr = np.zeros((n, m), dtype=np.int32)
    # Lưu lịch sử số lần mỗi cảnh đã xuất hiện trên nhánh tối ưu dẫn tới trạng thái k
    usage_hist = np.zeros((m, m), dtype=np.float32)
    usage_hist[idx, idx] = 1.0

    for i in range(1, n):
        # Cộng thêm phạt nếu cảnh đích j đã từng xuất hiện trong lịch sử của nhánh k
        repeat_cost = GLOBAL_REUSE_PENALTY * usage_hist
        cand = dp[:, None] - trans - repeat_cost
        best_prev = np.argmax(cand, axis=0)
        ptr[i] = best_prev
        dp = score[i] + cand[best_prev, idx]

        new_usage = usage_hist[best_prev].copy()
        new_usage[idx, idx] += 1.0
        usage_hist = new_usage

    path = [int(np.argmax(dp))]
    for i in range(n - 1, 0, -1):
        path.append(int(ptr[i][path[-1]]))
    path.reverse()
    return path


def match_sentences(
    req: MatchRequest,
    scenes: list[dict],
    img_emb: Optional[np.ndarray],
    cap_emb: Optional[np.ndarray],
    total_dur: float,
    vision_mode: str,
) -> list[dict]:
    sentences = req.sentences
    n, m = len(sentences), len(scenes)
    focus_idx = np.full((n, m), 1, dtype=np.int32)
    parts = []

    if img_emb is not None and MODELS.clip_txt is not None:
        txt = MODELS.clip_txt.encode(
            sentences,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = np.einsum("nd,mfd->nmf", txt, img_emb)
        focus_idx = np.argmax(sims, axis=2)
        clip_scene = 0.6 * sims.max(axis=2) + 0.4 * sims.mean(axis=2)
        parts.append((zscore_rows(clip_scene), 0.65 if cap_emb is not None else 1.0))

    if cap_emb is not None:
        tm = MODELS.get_text_model()
        sent_emb = tm.encode(
            sentences,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        parts.append((zscore_rows(sent_emb @ cap_emb.T), 0.35 if img_emb is not None else 1.0))

    if not parts:
        raise HTTPException(status_code=500, detail="Không có dữ liệu embedding để khớp cảnh!")

    base = sum(w * z for z, w in parts) / sum(w for _, w in parts)
    base = zscore_rows(base)

    mids = np.array([(s["start_time"] + s["end_time"]) / 2.0 for s in scenes]) / max(total_dur, 1.0)
    expected = (np.arange(n) + 0.5) / n
    prior = PRIOR_WEIGHT * np.exp(-0.5 * ((mids[None, :] - expected[:, None]) / 0.30) ** 2)
    adj = base + prior + visual_bonus_matrix(sentences, scenes)

    order_pen = float(max(0.0, min(6.0, req.order_penalty)))
    path = monotonic_viterbi(adj.astype(np.float32), order_pen)

    results = []
    for i, j in enumerate(path):
        sc = scenes[j]
        z = float(base[i, j])
        others = np.delete(base[i], j)
        margin = z - float(others.max()) if others.size else 0.0
        if z >= 2.4 and margin >= -0.3:
            conf = "Phù hợp cao"
        elif z >= 1.5:
            conf = "Cần kiểm tra"
        else:
            conf = "Độ khớp thấp (Nên duyệt lại)"

        focus = (
            sc["frame_times"][int(focus_idx[i, j])]
            if img_emb is not None
            else round((sc["start_time"] + sc["end_time"]) / 2.0, 2)
        )
        results.append({
            "index": i + 1,
            "scene_id": sc["scene_id"],
            "start_sec": sc["start_time"],
            "end_sec": sc["end_time"],
            "start_time": sc["start_time"],
            "end_time": sc["end_time"],
            "boundary_start": sc["start_time"],
            "boundary_end": sc["end_time"],
            "focus_sec": focus,
            "duration": round(sc["end_time"] - sc["start_time"], 2),
            "text": sentences[i],
            "caption": sc["caption"],
            "score": round(z, 3),
            "confidence": conf,
            "vision_status": sc["vision_status"],
            "motion_score": sc["motion_score"],
            "face_count": sc["face_count"],
        })
    return results


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------
def load_cache(key: str):
    meta_f, npz_f = DATA_DIR / f"{key}.json", DATA_DIR / f"{key}.npz"
    if not (meta_f.exists() and npz_f.exists()):
        return None
    try:
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if meta.get("algo_version") != ALGO_VERSION:
            return None
        with np.load(npz_f) as z:
            img = z["img"] if "img" in z.files and z["img"].size else None
            cap = z["cap"] if "cap" in z.files and z["cap"].size else None
        return meta["scenes"], img, cap, meta.get("vision_mode", "cached")
    except Exception as ex:  # noqa: BLE001
        log.warning("Cache hỏng, sẽ phân tích lại: %s", ex)
        return None


def save_cache(key: str, scenes, img, cap, vision_mode) -> None:
    try:
        (DATA_DIR / f"{key}.json").write_text(
            json.dumps(
                {"algo_version": ALGO_VERSION, "vision_mode": vision_mode, "scenes": scenes},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            DATA_DIR / f"{key}.npz",
            img=img if img is not None else np.zeros(0),
            cap=cap if cap is not None else np.zeros(0),
        )
    except Exception as ex:  # noqa: BLE001
        log.warning("Không ghi được cache: %s", ex)


# ----------------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        import edge_tts  # noqa: F401
        edge = True
    except ImportError:
        edge = False
    return {
        "status": "ok",
        "version": ALGO_VERSION,
        "clip": MODELS.state,
        "edge_tts": edge,
        "busy": JOB_LOCK.locked(),
        "voices": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
    }


@app.get("/voices")
def list_voices():
    return {
        "voices": [
            {"id": "vi-VN-HoaiMyNeural", "name": "Hoài My (Nữ - Miền Nam)", "gender": "Female"},
            {"id": "vi-VN-NamMinhNeural", "name": "Nam Minh (Nam - Miền Bắc)", "gender": "Male"},
        ]
    }


@app.get("/progress")
def get_progress():
    return PROGRESS.snapshot()


@app.post("/cancel")
def cancel():
    CANCEL.set()
    return {"status": "cancelling"}


@app.post("/auto_match_all")
def auto_match_all(req: MatchRequest):
    if not JOB_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="AI Server đang bận xử lý tác vụ khác, vui lòng đợi.")
    CANCEL.clear()
    try:
        max_scenes = int(max(10, min(800, req.max_scenes or 240)))
        meta = validate_video_file(req.video_path)
        total_dur = meta["duration"]

        PROGRESS.set("loading", 2, "Đang nạp mô hình CLIP (lần đầu có thể mất vài phút)...")
        use_clip = MODELS.load_clip()
        mode_tag = "clip" if use_clip else "heur"
        key = cache_key(req.video_path, max_scenes, mode_tag, req.use_ollama)

        cached = load_cache(key)
        if cached:
            scenes, img_emb, cap_emb, vision_mode = cached
            log.info("Nạp %d cảnh từ cache (%s).", len(scenes), key)
        else:
            scenes, img_emb, vision_mode = analyze_video(
                req.video_path, meta, max_scenes, use_clip, req.use_ollama
            )
            if not scenes:
                raise HTTPException(status_code=500, detail="Không trích xuất được cảnh hợp lệ từ video!")
            cap_emb = None
            if (not use_clip) or req.use_ollama:
                check_cancel()
                PROGRESS.set("embedding", 75, "Đang mã hóa mô tả cảnh...", vision_mode)
                cap_emb = MODELS.get_text_model().encode(
                    [s["caption"] for s in scenes],
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            save_cache(key, scenes, img_emb, cap_emb, vision_mode)

        check_cancel()
        PROGRESS.set("matching", 90, "Đang tối ưu khớp cảnh toàn cục cho kịch bản...", vision_mode)
        results = match_sentences(req, scenes, img_emb, cap_emb, total_dur, vision_mode)
        PROGRESS.set("done", 100, f"Hoàn tất khớp {len(results)} phân cảnh!", vision_mode)
        return {
            "vision_mode": vision_mode,
            "matcher": "clip_viterbi" if img_emb is not None else "caption_viterbi",
            "total_scenes": len(scenes),
            "results": results,
        }
    except JobCancelled:
        PROGRESS.set("cancelled", 0, "Đã hủy")
        raise HTTPException(status_code=409, detail="Đã hủy bởi người dùng.")
    finally:
        JOB_LOCK.release()


async def _synthesize_edge_tts(text: str, voice: str, rate: str, pitch: str) -> bytes:
    import edge_tts

    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            # Ở lần thử cuối nếu lỗi do tham số rate/pitch, thử gọi mặc định không truyền rate/pitch
            if attempt == 3:
                comm = edge_tts.Communicate(text, voice)
            else:
                comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

            buf = bytearray()
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    buf.extend(ch["data"])
            if len(buf) > 200:
                return bytes(buf)
        except Exception as ex:  # noqa: BLE001
            last_err = ex
            log.warning("Thử lại TTS lần %d (voice=%s, rate=%s, pitch=%s): %s", attempt + 1, voice, rate, pitch, ex)
        await asyncio.sleep(0.6 * (attempt + 1))

    raise HTTPException(status_code=502, detail=f"edge-tts thất bại (voice={voice}): {last_err}")


@app.post("/tts")
@app.get("/tts")
async def tts(
    req: Optional[TTSRequest] = None,
    text: Optional[str] = None,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
):
    """Hỗ trợ cả POST (JSON từ Main.java) và GET (query params) với bộ tự sửa lỗi tham số."""
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        raise HTTPException(status_code=501, detail="Chưa cài edge-tts: pip install edge-tts")

    raw_text = req.text if req else (text or "")
    cleaned_text = clean_tts_text(raw_text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Nội dung văn bản lồng tiếng trống!")

    raw_voice = (
        (req.voice or req.voice_id or req.voiceName)
        if req
        else (voice or "vi-VN-HoaiMyNeural")
    )
    resolved_voice = resolve_voice(raw_voice)
    resolved_rate = normalize_rate(req.rate if req else rate, req.speed if req else None)
    resolved_pitch = normalize_pitch(req.pitch if req else pitch)

    log.info("TTS -> voice: %s (gốc: %s) | rate: %s | pitch: %s", resolved_voice, raw_voice, resolved_rate, resolved_pitch)
    audio_bytes = await _synthesize_edge_tts(cleaned_text, resolved_voice, resolved_rate, resolved_pitch)

    if req and req.output_path:
        try:
            out_p = Path(req.output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(audio_bytes)
        except Exception as ex:  # noqa: BLE001
            log.warning("Không thể ghi trực tiếp file TTS ra %s: %s", req.output_path, ex)

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Resolved-Voice": resolved_voice,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("AI_PORT", "8000")), log_level="info")