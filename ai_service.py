"""
Movie Review AI Engine v3.2 (Full Upgraded Edition)
===================================================
Tích hợp đầy đủ:
1. Lõi chọn cảnh CLIP + Viterbi DP từ v3.0 (chính xác nhất thực tế).
2. Chống lặp cảnh toàn cục (kể cả lặp cách quãng A -> B -> A).
3. Chống đứng hình (Frozen Frame): tăng MIN_SCENE_SEC = 3.8s + phạt cảnh tĩnh/đen.
4. Chống bẫy chữ Intro/Title Card: giảm điểm cảnh tĩnh hiện chữ giới thiệu đầu/cuối phim.
5. Nhận diện vùng Highlight hành động để khớp ưu tiên với các câu kịch bản cao trào.
6. Khóa giọng TTS cố định suốt clip + Tích hợp giọng "Cô gái hoạt ngôn" (Review nhanh).
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

ALGO_VERSION = "3.2_clip_dp_full_upgrade"
CLIP_IMG_MODEL = os.getenv("CLIP_IMG_MODEL", "clip-ViT-B-32")
CLIP_TXT_MODEL = os.getenv("CLIP_TXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
TEXT_MODEL = os.getenv("TEXT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")

# Cấu hình cắt cảnh & Viterbi tối ưu chống đứng hình và lặp cảnh
FRAME_FRACTIONS = (0.15, 0.38, 0.62, 0.85)
MIN_SCENE_SEC = 3.8           # Tăng lên 3.8s để cảnh luôn đủ dài cho câu đọc, chống đứng khung hình cuối câu
MAX_SCENE_SEC = 14.0
SPLIT_CHUNK_SEC = 6.5
PRIOR_WEIGHT = 0.45           # Độ ưu tiên vị trí tương đối theo mạch phim
REUSE_PENALTY = 5.0           # Phạt cực nặng khi 2 câu liên tiếp dùng cùng 1 cảnh
GLOBAL_REUSE_PENALTY = 2.2    # Phạt tích lũy khi dùng lại bất kỳ cảnh nào đã dùng trước đó
DEFAULT_BACK_PENALTY = 1.35   # Phạt khi nhảy lùi về cảnh trước đó trong phim
STATIC_FRAME_PENALTY = 0.65   # Phạt các cảnh đứng yên (tránh màn hình chữ giới thiệu / cảnh chết)

DATA_DIR = Path("ai_cache")
DATA_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Cấu hình Giọng đọc (Bao gồm "Cô gái hoạt ngôn" & Khóa giọng xuyên suốt clip)
# ----------------------------------------------------------------------------
VOICE_PRESETS = {
    "hoatngon": {
        "key": "hoatngon",
        "base_voice": "vi-VN-HoaiMyNeural",
        "base_rate": 18,    # Đọc nhanh +18% chuẩn review phim cuốn hút
        "base_pitch": 4,    # Tông giọng tươi sáng +4Hz
        "label": "Cô gái hoạt ngôn (Review nhanh)",
    },
    "hoaimy": {
        "key": "hoaimy",
        "base_voice": "vi-VN-HoaiMyNeural",
        "base_rate": 0,
        "base_pitch": 0,
        "label": "Hoài My (Nữ chuẩn)",
    },
    "namminh": {
        "key": "namminh",
        "base_voice": "vi-VN-NamMinhNeural",
        "base_rate": 0,
        "base_pitch": 0,
        "label": "Nam Minh (Nam chuẩn)",
    },
    "namtram": {
        "key": "namtram",
        "base_voice": "vi-VN-NamMinhNeural",
        "base_rate": 8,
        "base_pitch": -4,
        "label": "Nam trầm (Review hành động)",
    },
}

TTS_THREAD_LOCK = threading.Lock()
SESSION_VOICE_STATE = {
    "active_preset": "hoatngon",  # Mặc định ưu tiên giọng Cô gái hoạt ngôn
    "locked_rate": None,
    "locked_pitch": None,
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
# Schema Request
# ----------------------------------------------------------------------------
class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video_path: str
    sentences: list[str] = Field(..., min_length=1)
    max_scenes: int = 240
    use_ollama: bool = False
    order_penalty: float = DEFAULT_BACK_PENALTY
    voice: Optional[str] = None  # Nếu Java gửi kèm giọng ngay lúc match, khóa giọng ngay từ đầu


class TTSRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=1)
    voice: Optional[str] = None
    voice_id: Optional[str] = None
    voiceName: Optional[str] = None
    rate: Any = "+0%"
    speed: Any = None
    pitch: Any = "+0Hz"
    output_path: Optional[str] = None


# ----------------------------------------------------------------------------
# Bộ xử lý Giọng đọc TTS (Khóa giọng cố định, không nhảy giọng giữa clip)
# ----------------------------------------------------------------------------
def strip_vietnamese_accents(s: str) -> str:
    s = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", s)
    s = re.sub(r"[èéẹẻẽêềếệểễ]", "e", s)
    s = re.sub(r"[ìíịỉĩ]", "i", s)
    s = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", s)
    s = re.sub(r"[ùúụủũưừứựửữ]", "u", s)
    s = re.sub(r"[ỳýỵỷỹ]", "y", s)
    s = re.sub(r"[đ]", "d", s)
    return s


def resolve_voice_preset(raw_voice: Optional[str]) -> dict:
    """
    Nhận diện chính xác giọng từ UI Java và lưu vào SESSION_VOICE_STATE.
    Nếu một câu trong kịch bản gửi thiếu tham số voice, tự động lấy đúng giọng của câu trước đó.
    """
    if not raw_voice or not str(raw_voice).strip():
        return VOICE_PRESETS[SESSION_VOICE_STATE["active_preset"]]

    v_clean = strip_vietnamese_accents(str(raw_voice).strip().lower())
    v_compact = re.sub(r"[^a-z0-9]", "", v_clean)

    # 1. Nhận diện giọng "Cô gái hoạt ngôn"
    if any(k in v_compact for k in ("hoatngon", "cogai", "tiktok", "reviewnhanh", "nhanh", "capcut", "stral", "viral")):
        SESSION_VOICE_STATE["active_preset"] = "hoatngon"
        return VOICE_PRESETS["hoatngon"]

    # 2. Nhận diện giọng "Nam trầm"
    if any(k in v_compact for k in ("namtram", "tramnam", "tram")):
        SESSION_VOICE_STATE["active_preset"] = "namtram"
        return VOICE_PRESETS["namtram"]

    # 3. Nhận diện giọng "Nam Minh"
    if any(k in v_compact for k in ("namminh", "giongnam", "male", "nam")):
        SESSION_VOICE_STATE["active_preset"] = "namminh"
        return VOICE_PRESETS["namminh"]

    # 4. Nhận diện giọng "Hoài My"
    if any(k in v_compact for k in ("hoaimy", "giongnu", "female", "nu")):
        SESSION_VOICE_STATE["active_preset"] = "hoaimy"
        return VOICE_PRESETS["hoaimy"]

    return VOICE_PRESETS[SESSION_VOICE_STATE["active_preset"]]


def parse_offset_int(val: Any, is_rate: bool = True) -> int:
    if val is None:
        return 0
    s = str(val).strip().replace(" ", "")
    if not s or s in ("0", "0%", "+0%", "-0%", "0Hz", "+0Hz", "-0Hz", "1.0", "1"):
        return 0
    try:
        clean = re.sub(r"[^0-9.+-]", "", s)
        num = float(clean)
        if is_rate and ("." in s and not s.endswith("%") and 0.3 <= num <= 2.5):
            return int(round((num - 1.0) * 100))
        return int(round(num))
    except ValueError:
        return 0


def build_prosody(preset: dict, user_rate: Any, user_speed: Any, user_pitch: Any) -> tuple[str, str]:
    raw_r = user_speed if user_speed is not None else user_rate
    r_offset = parse_offset_int(raw_r, is_rate=True)
    p_offset = parse_offset_int(user_pitch, is_rate=False)

    final_rate = max(-50, min(100, preset["base_rate"] + r_offset))
    final_pitch = max(-50, min(50, preset["base_pitch"] + p_offset))
    return f"{final_rate:+d}%", f"{final_pitch:+d}Hz"


def clean_tts_text(text: str) -> str:
    """Làm sạch văn bản để Edge-TTS đọc liền mạch, không bị ngắt hoặc nhảy tông giọng."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f*#_~`<>|\"'“”‘’]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ----------------------------------------------------------------------------
# Tiện ích phân tích Video & Chống đứng hình / Bẫy chữ
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
    """
    Gộp cảnh quá ngắn (< MIN_SCENE_SEC = 3.8s) để tránh lỗi khung hình đứng yên khi câu đọc dài,
    và chia các cảnh quá dài thành các đoạn vừa phải.
    """
    if not raw_scenes:
        step = max(5.0, total_dur / max_scenes)
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
    min_diff = float(np.min(diffs)) if diffs else 0.0
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    mid = grays[len(grays) // 2]
    edges_val = float(np.mean(cv2.Canny(mid, 80, 160)))

    face_cnt = 0
    if face_cascade is not None:
        try:
            faces = face_cascade.detectMultiScale(mid, 1.2, 4)
            face_cnt = len(faces)
        except Exception:
            face_cnt = 0

    # Phát hiện cảnh bị đứng hình (frozen) hoặc màn hình hiện chữ tĩnh (title card)
    is_static = bool(mean_diff < 1.1 or min_diff < 0.25)
    is_title_like = bool(mean_diff < 2.2 and edges_val > 28.0 and face_cnt == 0)

    return {
        "motion": round(mean_diff, 2),
        "brightness": round(float(np.mean(mid)), 1),
        "edges": round(edges_val, 2),
        "faces": int(face_cnt),
        "is_static": is_static,
        "is_title_like": is_title_like,
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
            "is_static": feat["is_static"],
            "is_title_like": feat["is_title_like"],
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
# Khớp câu <-> cảnh (Chống bẫy chữ Intro, chống đứng hình & chống lặp cảnh)
# ----------------------------------------------------------------------------
def zscore_rows(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)


ACTION_WORDS = (
    "chiến đấu", "tấn công", "phản công", "truy đuổi", "chạy", "đánh", "tốc độ", "đòn",
    "lao vào", "kích hoạt", "nổ", "va chạm", "đuổi theo", "chiến thắng", "bắn", "chém",
    "né tránh", "giao tranh", "đối đầu", "tung chiêu", "hạ gục", "vũ khí", "cứu", "quyết định",
)
CALM_WORDS = (
    "nhớ lại", "suy nghĩ", "bí mật", "quá khứ", "cha", "mẹ", "ước mơ", "quan sát",
    "nhận ra", "trò chuyện", "tâm sự", "khóc", "buồn", "hồi tưởng", "lặng lẽ", "nói",
)


def enrich_sentences_for_clip(sentences: list[str]) -> list[str]:
    """
    Thêm ngữ cảnh thị giác vào câu ngắn chứa tên riêng để tránh CLIP bị 'bẫy chữ' (OCR)
    chọn nhầm màn hình giới thiệu tên nhân vật ở phần mở đầu.
    """
    enriched = []
    for s in sentences:
        low = s.lower()
        if any(w in low for w in ACTION_WORDS):
            enriched.append(f"{s} (cảnh hành động, nhân vật đang chiến đấu chuyển động trong phim)")
        elif len(s.split()) <= 10:
            enriched.append(f"{s} (cảnh quay nhân vật và bối cảnh hoạt động trong phim)")
        else:
            enriched.append(s)
    return enriched


def visual_bonus_matrix(sentences: list[str], scenes: list[dict], total_dur: float) -> np.ndarray:
    motion = np.array([s["motion_score"] for s in scenes], dtype=np.float32)
    faces = np.array([s["face_count"] for s in scenes], dtype=np.float32)
    is_static = np.array([bool(s.get("is_static", False)) for s in scenes], dtype=np.float32)
    is_title = np.array([bool(s.get("is_title_like", False)) for s in scenes], dtype=np.float32)
    mids = np.array([(s["start_time"] + s["end_time"]) / 2.0 for s in scenes], dtype=np.float32)
    rel_pos = mids / max(total_dur, 1.0)

    out = np.zeros((len(sentences), len(scenes)), dtype=np.float32)

    # 1. Phạt các cảnh bị đứng hình (tránh lỗi khung hình đứng yên trong video xuất ra)
    scene_penalty = -STATIC_FRAME_PENALTY * is_static - 0.55 * is_title

    # 2. Phạt cảnh Opening / Title Card (90s đầu nếu phim dài > 8 phút) và Ending Credits (cuối phim)
    if total_dur > 480.0:
        intro_zone = (mids < 85.0) & ((is_title > 0) | (motion < 4.0))
        outro_zone = (rel_pos > 0.93) & ((is_title > 0) | (motion < 4.0))
        scene_penalty -= 1.15 * intro_zone.astype(np.float32)
        scene_penalty -= 1.15 * outro_zone.astype(np.float32)

    out += scene_penalty[None, :]

    # 3. Cộng điểm thưởng Highlight cho các câu hành động / tâm lý
    for i, sent in enumerate(sentences):
        low = sent.lower()
        if any(w in low for w in ACTION_WORDS):
            out[i] += np.minimum(0.60, motion / 48.0)
            # Phạt thêm nếu câu hành động mà chọn trúng cảnh tĩnh
            out[i] -= 0.5 * (motion < 3.0).astype(np.float32)
        elif any(w in low for w in CALM_WORDS):
            out[i] += 0.25 * (faces >= 1)

    return out


def monotonic_viterbi(score: np.ndarray, back_penalty: float) -> list[int]:
    """
    Quy hoạch động Viterbi có bộ nhớ toàn cục:
    - Giữ mạch thời gian tự nhiên của phim.
    - Chặn đứng việc lặp lại cùng 1 cảnh (kể cả lặp liên tiếp lẫn lặp xen kẽ A -> B -> A).
    """
    n, m = score.shape
    idx = np.arange(m)
    trans = np.zeros((m, m), dtype=np.float32)
    back = idx[None, :] < idx[:, None]
    trans[back] = back_penalty + 0.005 * (idx[:, None] - idx[None, :])[back]
    trans[idx, idx] = REUSE_PENALTY

    dp = score[0].copy()
    ptr = np.zeros((n, m), dtype=np.int32)
    usage_hist = np.zeros((m, m), dtype=np.float32)
    usage_hist[idx, idx] = 1.0

    for i in range(1, n):
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
        clip_queries = enrich_sentences_for_clip(sentences)
        txt = MODELS.clip_txt.encode(
            clip_queries,
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
    adj = base + prior + visual_bonus_matrix(sentences, scenes, total_dur)

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

        s_start = float(sc["start_time"])
        s_end = float(sc["end_time"])
        raw_focus = (
            float(sc["frame_times"][int(focus_idx[i, j])])
            if img_emb is not None
            else round((s_start + s_end) / 2.0, 2)
        )
        # Giữ focus_sec cách rìa cảnh ít nhất 1.2s để khi Java cắt quanh focus_sec không bị hụt hình gây đứng frame
        if (s_end - s_start) > 2.6:
            focus = round(max(s_start + 1.2, min(s_end - 1.2, raw_focus)), 2)
        else:
            focus = round((s_start + s_end) / 2.0, 2)

        results.append({
            "index": i + 1,
            "scene_id": sc["scene_id"],
            "start_sec": s_start,
            "end_sec": s_end,
            "start_time": s_start,
            "end_time": s_end,
            "boundary_start": s_start,
            "boundary_end": s_end,
            "focus_sec": focus,
            "duration": round(s_end - s_start, 2),
            "text": sentences[i],
            "caption": sc["caption"],
            "score": round(z, 3),
            "confidence": conf,
            "vision_status": sc["vision_status"],
            "motion_score": sc["motion_score"],
            "face_count": sc["face_count"],
            "is_highlight": bool(sc["motion_score"] >= 22.0),
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
        "active_voice": VOICE_PRESETS[SESSION_VOICE_STATE["active_preset"]]["label"],
        "voices": [
            "vi-VN-HoaiMy-HoatNgon",
            "vi-VN-HoaiMyNeural",
            "vi-VN-NamMinhNeural",
            "vi-VN-NamMinh-Tram",
        ],
    }


@app.get("/voices")
def list_voices():
    return {
        "voices": [
            {"id": "vi-VN-HoaiMy-HoatNgon", "name": "Cô gái hoạt ngôn (Review nhanh)", "gender": "Female"},
            {"id": "vi-VN-HoaiMyNeural", "name": "Hoài My (Nữ chuẩn)", "gender": "Female"},
            {"id": "vi-VN-NamMinhNeural", "name": "Nam Minh (Nam chuẩn)", "gender": "Male"},
            {"id": "vi-VN-NamMinh-Tram", "name": "Nam trầm (Review hành động)", "gender": "Male"},
        ]
    }


@app.post("/set_voice")
def set_voice(voice: str):
    """Endpoint tùy chọn để khóa cứng giọng đọc trước khi render."""
    preset = resolve_voice_preset(voice)
    return {"status": "ok", "locked_voice": preset["base_voice"], "label": preset["label"]}


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
        if req.voice:
            resolve_voice_preset(req.voice)

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


async def _synthesize_edge_tts_strict(text: str, base_voice: str, rate: str, pitch: str) -> bytes:
    """
    Chỉ kết xuất đúng 1 giọng `base_voice` và giữ nguyên `rate`/`pitch` trong mọi lần thử lại.
    Đảm bảo không bao giờ bị biến đổi giọng hay tốc độ giữa các câu trong cùng 1 clip.
    """
    import edge_tts

    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            comm = edge_tts.Communicate(text, base_voice, rate=rate, pitch=pitch)
            buf = bytearray()
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    buf.extend(ch["data"])
            if len(buf) > 200:
                return bytes(buf)
        except Exception as ex:  # noqa: BLE001
            last_err = ex
            log.warning("Thử lại TTS lần %d (voice=%s, rate=%s, pitch=%s): %s", attempt + 1, base_voice, rate, pitch, ex)
        await asyncio.sleep(0.5 * (attempt + 1))

    raise HTTPException(status_code=502, detail=f"edge-tts thất bại (voice={base_voice}): {last_err}")


@app.post("/tts")
@app.get("/tts")
async def tts(
    req: Optional[TTSRequest] = None,
    text: Optional[str] = None,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
):
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        raise HTTPException(status_code=501, detail="Chưa cài edge-tts: pip install edge-tts")

    raw_text = req.text if req else (text or "")
    cleaned_text = clean_tts_text(raw_text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Nội dung văn bản lồng tiếng trống!")

    with TTS_THREAD_LOCK:
        raw_voice = (req.voice or req.voice_id or req.voiceName) if req else voice
        preset = resolve_voice_preset(raw_voice)
        base_voice = preset["base_voice"]
        final_rate, final_pitch = build_prosody(
            preset,
            req.rate if req else rate,
            req.speed if req else None,
            req.pitch if req else pitch,
        )

    log.info(
        "TTS [%s] -> model: %s | rate: %s | pitch: %s | text: %s...",
        preset["label"], base_voice, final_rate, final_pitch, cleaned_text[:32],
    )
    audio_bytes = await _synthesize_edge_tts_strict(cleaned_text, base_voice, final_rate, final_pitch)

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
            "X-Resolved-Voice": base_voice,
            "X-Voice-Preset": preset["label"],
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("AI_PORT", "8000")), log_level="info")