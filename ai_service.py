"""
Movie Review AI Engine v3.2 (chọn cảnh theo chuỗi liền mạch)
===========================================================
v3.2: đưa việc chọn cảnh về đúng tinh thần v3.0 (một chuỗi Viterbi liền mạch, điểm z-score là điểm chọn).
  - Bỏ nhân bằng chứng thô vào điểm chọn, bỏ trạng thái "null" làm đứt chuỗi, bỏ phạt trùng góc máy / cửa sổ lặp,
    bỏ phạt nhảy tiến, tắt tách mệnh đề mặc định, khôi phục điểm cộng hành động/khuôn mặt như v3.0.
  - Câu trừu tượng (không có hình ảnh đặc trưng): hàng điểm bị làm phẳng để nhiễu z-score không chi phối,
    cảnh được chọn THEO MẠCH (liền sau câu trước) và chỉ bị gắn nhãn "Câu trừu tượng".
  - Không âm thầm dùng chế độ dự phòng khi CLIP lỗi (trả 503 trừ khi allow_fallback=true).

Từ v3.1:
Kiến trúc: CLIP Retrieval -> Candidate Verification (dò kỹ thời điểm) -> Sequence Optimization.

Cải tiến so với v3.0 (theo tài liệu phân tích khuyết điểm):
  P0  * Cho phép TỪ CHỐI cảnh (no_match): trạng thái "không chọn cảnh" nằm trong hàm mục tiêu Viterbi,
        dựa trên bằng chứng thô (cosine CLIP gốc) chứ không chỉ z-score tương đối.
      * Tách rõ 3 chế độ: CLIP thuần / CLIP + Ollama thật / dự phòng caption (hạ độ tin cậy, báo rõ).
        Caption heuristic KHÔNG còn bị trộn vào nhánh CLIP chỉ vì cờ use_ollama bật.
      * Cache: khóa theo toàn bộ cấu hình ảnh hưởng dữ liệu, ghi nguyên tử 1 file .npz,
        kiểm tra shape/schema/hữu hạn khi nạp, có dọn dung lượng.
      * Ngưỡng tin cậy: trạng thái "chưa hiệu chỉnh" cho đến khi có calibration.json.
  P1  * Top-K ứng viên mỗi câu (đa dạng, loại cảnh gần trùng).
      * Lấy mẫu dày quanh cảnh được chọn (mỗi ~0.4s) -> focus_sec + action_start/end chính xác hơn.
      * Lọc khung hình đen/mờ; tách câu thành mệnh đề để tăng bằng chứng.
  P2  * Viterbi: phạt nhảy lùi + nhảy tiến xa + trùng/gần trùng cảnh + vòng lặp phạt tái sử dụng
        trong cửa sổ N câu; 4 chế độ thứ tự (free/loose/medium/strict), tiên nghiệm vị trí là ràng buộc mềm.
  *   /rematch: tìm lại cảnh khác (có thể gần một mốc thời gian) để thay thế khi người dùng duyệt thấy sai.
  *   Job theo job_id (hủy/tiến độ riêng từng job), HTTP 409 = bận, 499 = đã hủy.
  *   Model: trạng thái not_loaded/loading/ready/failed, không giữ khóa khi tải, tự thử lại sau cooldown,
        kiểm tra chiều embedding của 2 encoder CLIP.
  *   Bảo mật: API key tùy chọn (AI_API_KEY), giới hạn thư mục video (ALLOWED_VIDEO_ROOTS),
        Ollama chỉ loopback, TTS có allow-list giọng + giới hạn rate/pitch.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("MovieReviewAI")

# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------
ALGO_VERSION = "3.1_retrieve_verify_sequence"   # khóa cache dữ liệu (embedding) - giữ nguyên để không phải phân tích lại
MATCH_VERSION = "3.2_chain_first"                # phiên bản thuật toán chọn cảnh
CACHE_SCHEMA = 2
CLIP_IMG_MODEL = os.getenv("CLIP_IMG_MODEL", "clip-ViT-B-32")
CLIP_TXT_MODEL = os.getenv("CLIP_TXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
TEXT_MODEL = os.getenv("TEXT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_PROMPT = "Mô tả ngắn gọn bằng tiếng Việt (1-2 câu): nhân vật/vật thể nào, đang làm gì, ở đâu, không khí ra sao?"
API_KEY = os.getenv("AI_API_KEY", "")
ALLOWED_VIDEO_ROOTS = [Path(p).resolve() for p in os.getenv("ALLOWED_VIDEO_ROOTS", "").split(os.pathsep) if p.strip()]

FRAME_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)   # lấy mẫu thô trong mỗi cảnh
FRAME_WIDTH = 384
DETECT_THRESHOLD = float(os.getenv("DETECT_THRESHOLD", "27"))
MIN_SCENE_SEC = 1.0
MAX_SCENE_SEC = 14.0
SPLIT_CHUNK_SEC = 7.0
DENSE_STEP_SEC = 0.4        # bước lấy mẫu dày khi dò thời điểm
DENSE_MAX_FRAMES = 24
TOP_K = 5
DUP_SIM = 0.93              # hai cảnh có cosine embedding >= ngưỡng này coi là gần trùng hình
CAND_DUP_SIM = 0.96         # ngưỡng loại ứng viên gần trùng khi liệt kê top-K
REUSE_WINDOW = int(os.getenv("REUSE_WINDOW", "0"))   # 0 = TẮT (cho phép nhiều góc máy liên tiếp trong cùng bối cảnh)
REUSE_STEP = 1.0            # mức phạt mỗi vòng lặp Lagrange (chỉ khi REUSE_WINDOW > 0)
SAME_SCENE_PENALTY = float(os.getenv("SAME_SCENE_PENALTY", "2.0"))   # chỉ phạt dùng lại ĐÚNG MỘT cảnh liên tiếp
DUP_SCENE_PENALTY = 1.2     # chỉ dùng khi bật REUSE_WINDOW / gọi build_transitions với dup_sim
FWD_TIE_EPS = 0.003         # chi phí cực nhỏ chỉ để phá hòa (ưu tiên cảnh gần nhất phía sau); KHÔNG phạt nhảy tiến thật
USE_CLAUSES = os.getenv("USE_CLAUSES", "0") == "1"   # tách mệnh đề dễ tạo vector nhiễu -> mặc định tắt
MAX_SCENES_DEFAULT = 600
CLIP_RETRY_SEC = float(os.getenv("CLIP_RETRY_SEC", "60"))
CLIP_WAIT_SEC = float(os.getenv("CLIP_WAIT_SEC", "900"))
CACHE_MAX_MB = float(os.getenv("CACHE_MAX_MB", "2048"))
DEFAULT_BACK_PENALTY = 1.2

# chế độ thứ tự: (phạt nhảy lùi, hệ số phạt nhảy tiến xa, trọng số tiên nghiệm vị trí)
ORDER_MODES = {
    "free":   (0.0, 0.0, 0.00),    # kể không theo thứ tự phim / hồi tưởng
    "loose":  (0.6, 0.0, 0.00),
    "medium": (1.2, 0.0, 0.15),    # prior vị trí chỉ là gợi ý rất nhẹ (v3.1 0.45 làm cảnh trượt đều theo tỷ lệ câu)
    "strict": (2.5, 0.0, 0.35),
}

EDGE_VOICES_ALLOWED = set(filter(None, os.getenv(
    "EDGE_VOICES", "vi-VN-HoaiMyNeural,vi-VN-NamMinhNeural").split(",")))

DATA_DIR = Path(os.getenv("AI_CACHE_DIR", "ai_cache"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Calibration:
    raw_low: float = 0.14     # cosine thô dưới mức này = không có bằng chứng (CLIP đa ngôn ngữ: khớp tốt thường 0.17-0.24)
    raw_high: float = 0.24    # cosine thô từ mức này = bằng chứng mạnh
    z_ok: float = 2.4
    z_review: float = 1.5
    null_score: float = 1.25  # điểm của trạng thái "không chọn cảnh" trong Viterbi
    calibrated: bool = False


def load_calibration() -> Calibration:
    cal = Calibration()
    path = os.getenv("CALIBRATION_FILE", "calibration.json")
    try:
        if os.path.isfile(path):
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for k in ("raw_low", "raw_high", "z_ok", "z_review", "null_score"):
                if k in data:
                    setattr(cal, k, float(data[k]))
            cal.calibrated = bool(data.get("calibrated", True))
            log.info("Đã nạp hiệu chỉnh từ %s", path)
    except Exception as ex:  # noqa: BLE001
        log.warning("calibration.json lỗi (%s) -> dùng mặc định chưa hiệu chỉnh.", ex)
        cal = Calibration()
    if cal.raw_high <= cal.raw_low + 1e-3:
        cal.raw_low, cal.raw_high = 0.14, 0.24
    return cal


CAL = load_calibration()


# ----------------------------------------------------------------------------
# Job (hủy + tiến độ riêng từng job)
# ----------------------------------------------------------------------------
class JobCancelled(Exception):
    pass


class Progress:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = {"status": "idle", "percent": 0, "message": "Sẵn sàng", "vision_mode": "unknown"}

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


class Job:
    def __init__(self, job_id: Optional[str] = None) -> None:
        self.id = job_id or uuid.uuid4().hex[:12]
        self.cancel = threading.Event()
        self.progress = Progress()
        self.timings: dict[str, float] = {}
        self._t = time.time()

    def check(self) -> None:
        if self.cancel.is_set():
            raise JobCancelled()

    def lap(self, name: str) -> None:
        now = time.time()
        self.timings[name] = round(self.timings.get(name, 0.0) + now - self._t, 3)
        self._t = now


JOB_LOCK = threading.Lock()
_JOBS: "OrderedDict[str, Job]" = OrderedDict()
_JOBS_LOCK = threading.Lock()
ACTIVE_JOB: Optional[Job] = None
LAST_JOB: Optional[Job] = None


def register_job(job: Job) -> None:
    with _JOBS_LOCK:
        _JOBS[job.id] = job
        while len(_JOBS) > 8:
            _JOBS.popitem(last=False)


def get_job(job_id: str) -> Optional[Job]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def busy_error() -> HTTPException:
    return HTTPException(status_code=409, detail={"code": "busy", "message": "AI Server đang bận xử lý tác vụ khác, vui lòng đợi."})


# ----------------------------------------------------------------------------
# Quản lý mô hình
# ----------------------------------------------------------------------------
class Models:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()   # chỉ bảo vệ biến trạng thái (không giữ khi tải)
        self._text_lock = threading.Lock()
        self.clip_img = None
        self.clip_txt = None
        self.text = None
        self.dim: Optional[int] = None
        self.error: Optional[str] = None
        self.failed_at = 0.0
        self.state = "not_loaded"  # not_loaded | loading | ready | failed

    def _begin_load(self) -> bool:
        with self._state_lock:
            if self.state in ("loading", "ready"):
                return False
            if self.state == "failed" and time.time() - self.failed_at < CLIP_RETRY_SEC:
                return False
            self.state, self.error = "loading", None
            return True

    def _worker(self) -> None:
        try:
            from PIL import Image
            from sentence_transformers import SentenceTransformer
            log.info("Đang nạp CLIP đa ngôn ngữ (lần đầu sẽ tải ~600MB)...")
            img = SentenceTransformer(CLIP_IMG_MODEL)
            txt = SentenceTransformer(CLIP_TXT_MODEL)
            vi = img.encode([Image.new("RGB", (64, 64), (128, 128, 128))], convert_to_numpy=True, normalize_embeddings=True)
            vt = txt.encode(["một cảnh phim"], convert_to_numpy=True, normalize_embeddings=True)
            if vi.shape[1] != vt.shape[1]:
                raise RuntimeError(f"Hai encoder CLIP không cùng chiều embedding ({vi.shape[1]} != {vt.shape[1]}).")
            if not (np.isfinite(vi).all() and np.isfinite(vt).all()):
                raise RuntimeError("Embedding thử nghiệm chứa giá trị không hợp lệ.")
            self.clip_img, self.clip_txt, self.dim = img, txt, int(vi.shape[1])
            with self._state_lock:
                self.state = "ready"
            log.info("CLIP đã sẵn sàng (dim=%d).", self.dim)
        except Exception as ex:  # noqa: BLE001
            with self._state_lock:
                self.state, self.error, self.failed_at = "failed", str(ex), time.time()
            log.warning("Không nạp được CLIP (%s). Dùng chế độ dự phòng, thử lại sau %.0fs.", ex, CLIP_RETRY_SEC)

    def start_loading(self) -> None:
        if self._begin_load():
            threading.Thread(target=self._worker, daemon=True, name="clip-loader").start()

    def ensure_clip(self, job: Optional[Job] = None, timeout: float = CLIP_WAIT_SEC) -> bool:
        deadline = time.time() + timeout
        while True:
            self.start_loading()
            st = self.state
            if st == "ready":
                return True
            if st == "failed":
                return False
            if job:
                job.check()
                job.progress.set("loading", 2, "Đang nạp mô hình CLIP (lần đầu có thể mất vài phút)...")
            if time.time() > deadline:
                self.error = "Quá thời gian chờ nạp CLIP."
                return False
            time.sleep(0.4)

    def get_text_model(self):
        if self.text is None:
            with self._text_lock:
                if self.text is None:
                    from sentence_transformers import SentenceTransformer
                    log.info("Đang nạp mô hình văn bản %s...", TEXT_MODEL)
                    self.text = SentenceTransformer(TEXT_MODEL)
        return self.text


MODELS = Models()


@asynccontextmanager
async def lifespan(_: FastAPI):
    MODELS.start_loading()
    yield


app = FastAPI(title="Movie Review AI Engine v3.1", lifespan=lifespan)

face_cascade = None
try:
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if face_cascade.empty():
            face_cascade = None
except Exception as _ex:  # noqa: BLE001
    log.warning("Bỏ qua Haar Cascade: %s", _ex)
    face_cascade = None


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Thiếu hoặc sai X-API-Key."})


class MatchRequest(BaseModel):
    video_path: str
    sentences: list[str] = Field(..., min_length=1, max_length=1000)
    max_scenes: int = Field(default=MAX_SCENES_DEFAULT, ge=10, le=1500)
    allow_fallback: bool = False      # False: CLIP lỗi -> trả 503 thay vì âm thầm đoán mò bằng caption heuristic
    use_ollama: bool = False
    order_penalty: Optional[float] = Field(default=None, ge=0.0, le=6.0)
    order_mode: str = Field(default="medium", pattern="^(free|loose|medium|strict)$")
    allow_no_match: bool = True
    refine: bool = True
    top_k: int = Field(default=TOP_K, ge=1, le=10)
    detect_threshold: Optional[float] = Field(default=None, ge=8.0, le=60.0)
    job_id: Optional[str] = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class RematchRequest(BaseModel):
    video_path: str
    sentence: str = Field(..., min_length=1, max_length=800)
    max_scenes: int = Field(default=MAX_SCENES_DEFAULT, ge=10, le=1500)
    allow_fallback: bool = False
    abstract: bool = False            # True: câu đã được gắn "trừu tượng" ở lần khớp trước -> xếp theo độ gần mốc near_sec
    use_ollama: bool = False
    detect_threshold: Optional[float] = Field(default=None, ge=8.0, le=60.0)
    near_sec: Optional[float] = Field(default=None, ge=0.0)
    window_sec: float = Field(default=0.0, ge=0.0, le=36000.0)    # 0 = tìm toàn phim
    exclude: list[list[float]] = Field(default_factory=list, max_length=200)  # [[start, end], ...]
    top_k: int = Field(default=TOP_K, ge=1, le=10)
    refine: bool = True
    job_id: Optional[str] = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class CancelRequest(BaseModel):
    job_id: Optional[str] = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)
    voice: str = "vi-VN-HoaiMyNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


# ----------------------------------------------------------------------------
# Tiện ích video
# ----------------------------------------------------------------------------
def validate_video_file(video_path: str) -> dict:
    if not video_path:
        raise HTTPException(status_code=400, detail="File video không tồn tại!")
    p = Path(video_path).expanduser().resolve()
    if ALLOWED_VIDEO_ROOTS and not any(p == r or r in p.parents for r in ALLOWED_VIDEO_ROOTS):
        raise HTTPException(status_code=403, detail="Video nằm ngoài thư mục được phép (ALLOWED_VIDEO_ROOTS).")
    if not p.is_file():
        raise HTTPException(status_code=400, detail="File video không tồn tại!")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Không thể mở hoặc giải mã file video!")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps < 1:
            fps = 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret or frame is None or frame_count <= 10:
        raise HTTPException(status_code=400, detail="Video bị hỏng hoặc không chứa khung hình hợp lệ!")
    return {"path": str(p), "fps": float(fps), "frame_count": frame_count, "duration": frame_count / fps}


def ollama_url_safe() -> bool:
    host = (urlparse(OLLAMA_URL).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1") or os.getenv("OLLAMA_ALLOW_REMOTE") == "1"


def check_ollama_available() -> bool:
    if not ollama_url_safe():
        log.warning("OLLAMA_URL không phải loopback -> bị chặn (đặt OLLAMA_ALLOW_REMOTE=1 nếu thật sự tin cậy).")
        return False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.5, allow_redirects=False)
        if r.status_code == 200:
            names = [m.get("name", "") for m in r.json().get("models", [])]
            if any(OLLAMA_MODEL == n or n.startswith(OLLAMA_MODEL.split(":")[0] + ":") and OLLAMA_MODEL in n for n in names):
                return True
            return any(k in n for n in names for k in ("qwen", "llava", "moondream", "minicpm", "gemma3"))
    except Exception:  # noqa: BLE001
        pass
    return False


def read_frame_at(cap: cv2.VideoCapture, t: float, fps: float, total_dur: float):
    """Đọc khung tại thời điểm t, ưu tiên timestamp của decoder; trả (frame|None, thời điểm thực)."""
    t = float(min(max(0.0, t), max(0.0, total_dur - 0.1)))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if ok else t
    if (not ok) or fr is None or abs(actual - t) > 1.5:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        actual = t
    if not ok or fr is None:
        return None, t
    if not (0.0 <= actual <= total_dur + 1.0):
        actual = t
    return fr, float(actual)


def small_frame(fr: np.ndarray) -> np.ndarray:
    h, w = fr.shape[:2]
    return cv2.resize(fr, (FRAME_WIDTH, max(2, int(h * FRAME_WIDTH / max(w, 1)))))


def frame_ok(fr_small: np.ndarray) -> bool:
    """Loại khung hình gần đen hoặc quá mờ (không đáng tin để so khớp)."""
    g = cv2.cvtColor(fr_small, cv2.COLOR_BGR2GRAY)
    return float(g.mean()) >= 12.0 and float(cv2.Laplacian(g, cv2.CV_64F).var()) >= 12.0


def hist_similarity(cap: cv2.VideoCapture, t: float, fps: float, total_dur: float) -> float:
    """Độ giống màu giữa hai phía của điểm cắt t (1 = cùng bối cảnh, <=0 = khác hẳn)."""
    a, _ = read_frame_at(cap, max(0.0, t - 0.15), fps, total_dur)
    b, _ = read_frame_at(cap, t + 0.15, fps, total_dur)
    if a is None or b is None:
        return 0.5

    def h(f):
        hsv = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    return float(cv2.compareHist(h(a), h(b), cv2.HISTCMP_CORREL))


def build_intervals(raw_scenes: list[tuple[float, float]], total_dur: float, max_scenes: int,
                    boundary_sim: Optional[Callable[[float], float]] = None) -> tuple[list[tuple[float, float]], bool]:
    """Gộp cảnh quá ngắn (theo độ giống nội dung nếu có), chia cảnh quá dài, giảm xuống max_scenes
    bằng cách gộp cặp kề nhau GIỐNG nhau nhất. Trả (khoảng, có_phải_chia_đều_dự_phòng)."""
    used_fallback = False
    if not raw_scenes:
        used_fallback = True
        step = max(4.0, total_dur / max_scenes)
        raw_scenes, t = [], 0.0
        while t < total_dur - 1.0:
            raw_scenes.append((t, min(total_dur, t + step)))
            t += step
        if not raw_scenes:
            raw_scenes = [(0.0, total_dur)]

    cache: dict[float, Optional[float]] = {}

    def bs(t: float) -> Optional[float]:
        if boundary_sim is None:
            return None
        k = round(t, 2)
        if k not in cache:
            cache[k] = boundary_sim(t)
        return cache[k]

    sc = [[float(s), float(e)] for s, e in raw_scenes]
    i = 0
    while i < len(sc):
        s, e = sc[i]
        if (e - s) < MIN_SCENE_SEC and len(sc) > 1:
            if i == 0:
                target = "right"
            elif i == len(sc) - 1:
                target = "left"
            else:
                left, right = bs(s), bs(e)
                target = "left" if (left is None or right is None or left >= right) else "right"
            if target == "left":
                sc[i - 1][1] = e
            else:
                sc[i + 1][0] = s
            sc.pop(i)
            continue
        i += 1

    out: list[tuple[float, float]] = []
    kinds: list[str] = []   # kinds[k] = loại ranh giới giữa out[k] và out[k+1]
    for s, e in sc:
        if e - s > MAX_SCENE_SEC:
            n = int(np.ceil((e - s) / SPLIT_CHUNK_SEC))
            edges = np.linspace(s, e, n + 1)
            for k in range(n):
                out.append((float(edges[k]), float(edges[k + 1])))
                kinds.append("chunk" if k < n - 1 else "cut")
        else:
            out.append((float(s), float(e)))
            kinds.append("cut")
    kinds = kinds[:-1] if kinds else kinds

    if len(out) > max_scenes:
        bsim = []
        for k in range(len(out) - 1):
            if kinds[k] == "chunk":
                bsim.append(0.85)
            else:
                v = bs(out[k][1])
                bsim.append(0.5 if v is None else float(min(1.0, max(0.0, v))))
        while len(out) > max_scenes:
            durs = np.array([e - s for s, e in out])
            cost = (durs[:-1] + durs[1:]) * (2.0 - np.array(bsim))
            k = int(np.argmin(cost))
            out[k] = (out[k][0], out[k + 1][1])
            out.pop(k + 1)
            bsim.pop(k)
    return out, used_fallback


def detect_scene_list(video_path: str, fps: float, threshold: float, total_dur: float) -> list[tuple[float, float]]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    def run(th: float):
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=th, min_scene_len=max(12, int(fps * 0.8))))
        sm.detect_scenes(video, show_progress=False)
        sec = (lambda t: t.seconds if hasattr(t, "seconds") else t.get_seconds())
        return [(sec(a), sec(b)) for a, b in sm.get_scene_list()]

    raw = run(threshold)
    if len(raw) < max(2, int(total_dur / 40)):   # quá ít điểm cắt: phim tối/chuyển cảnh mềm -> thử nhạy hơn
        raw2 = run(max(12.0, threshold * 0.65))
        if len(raw2) > len(raw):
            log.info("Ngưỡng %.1f cho %d cảnh, hạ xuống %.1f cho %d cảnh.", threshold, len(raw), threshold * 0.65, len(raw2))
            raw = raw2
    return raw


def grab_frames(cap: cv2.VideoCapture, s: float, e: float, fps: float, total_dur: float):
    dur = max(0.5, e - s)
    frames, times, ok_q = [], [], []
    for f in FRAME_FRACTIONS:
        fr, t = read_frame_at(cap, s + dur * f, fps, total_dur)
        if fr is None:
            continue
        sm = small_frame(fr)
        frames.append(sm)
        times.append(t)
        ok_q.append(frame_ok(sm))
    if frames and not any(ok_q):   # cảnh tối toàn bộ: không phạt, dùng tất cả
        ok_q = [True] * len(frames)
    return frames, times, ok_q


def visual_features(frames: list[np.ndarray]) -> dict:
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [float(np.mean(cv2.absdiff(grays[i], grays[i + 1]))) for i in range(len(grays) - 1)]
    mid = grays[len(grays) // 2]
    face_cnt = 0
    if face_cascade is not None:
        try:
            face_cnt = len(face_cascade.detectMultiScale(mid, 1.2, 4))
        except Exception:  # noqa: BLE001
            face_cnt = 0
    return {"motion": round(float(np.mean(diffs)) if diffs else 0.0, 2), "brightness": round(float(np.mean(mid)), 1),
            "edges": float(np.mean(cv2.Canny(mid, 80, 160))), "faces": int(face_cnt)}


def heuristic_caption(feat: dict, rel_pos: float) -> str:
    """Mô tả tổng quát, CHẤT LƯỢNG THẤP - chỉ là phương án dự phòng khi không có CLIP."""
    arc = ("mở đầu giới thiệu bối cảnh và nhân vật" if rel_pos < 0.18 else
           "diễn biến phát triển tình tiết" if rel_pos < 0.55 else
           "cao trào xung đột căng thẳng" if rel_pos < 0.85 else "đoạn kết giải quyết và hé lộ")
    act = ("hành động nhanh, giao chiến, tấn công, né tránh" if feat["motion"] > 28 else
           "nhân vật di chuyển, truy đuổi, chuẩn bị đối phó" if feat["motion"] > 14 else
           "cảnh tĩnh, đối thoại, quan sát, suy nghĩ")
    shot = ("cận cảnh nhiều nhân vật" if feat["faces"] >= 2 else "cận cảnh gương mặt nhân vật" if feat["faces"] == 1 else
            "khung hình nhiều chi tiết máy móc, vật thể" if feat["edges"] > 25 else "toàn cảnh không gian")
    tone = "bối cảnh tối, kịch tính" if feat["brightness"] < 75 else "khung hình sáng rõ"
    return f"{arc}; {act}; {shot}; {tone}."


def describe_with_ollama(frames: list[np.ndarray]) -> Optional[str]:
    if not ollama_url_safe():
        return None
    pick = [frames[len(frames) // 4], frames[(3 * len(frames)) // 4]] if len(frames) >= 4 else frames[:2]
    imgs = []
    for f in pick:
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            imgs.append(base64.b64encode(buf.tobytes()).decode())
    if not imgs:
        return None
    for _ in range(2):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": OLLAMA_MODEL, "stream": False, "images": imgs, "prompt": OLLAMA_PROMPT},
                timeout=40, allow_redirects=False)
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
@dataclass
class Analysis:
    key: str
    scenes: list
    img_emb: Optional[np.ndarray]        # (m, F, d)
    valid: Optional[np.ndarray]          # (m, F) bool
    cap_emb: Optional[np.ndarray]        # (m, dt)
    cap_mask: Optional[np.ndarray]       # (m,) bool: cảnh nào có caption đáng tin
    vision_mode: str
    total_dur: float
    fps: float
    use_clip: bool
    ollama_ready: bool
    scene_sim: Optional[np.ndarray] = None
    from_cache: bool = False
    used_fallback_split: bool = False
    warnings: list = field(default_factory=list)


def analyze_video(job: Job, video_path: str, meta: dict, max_scenes: int, use_clip: bool,
                  ollama_ready: bool, detect_threshold: float):
    fps, total_dur = meta["fps"], meta["duration"]
    job.progress.set("analyzing", 6, "Đang dò điểm chuyển cảnh (PySceneDetect)...")
    raw = detect_scene_list(video_path, fps, detect_threshold, total_dur)
    job.check()
    job.lap("detect_scenes")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Không mở lại được video để phân tích.")
    try:
        intervals, used_fallback = build_intervals(raw, total_dur, max_scenes,
                                                   lambda t: hist_similarity(cap, t, fps, total_dur))
        vision_mode = ("clip+ollama" if ollama_ready else "clip") if use_clip else \
                      ("ollama_vision" if ollama_ready else "opencv_visual_heuristics")
        log.info("Chế độ: %s | số cảnh: %d%s", vision_mode, len(intervals), " (chia đều dự phòng)" if used_fallback else "")

        scenes: list[dict] = []
        emb_chunks: list[np.ndarray] = []
        valid_chunks: list[np.ndarray] = []
        pend_imgs, pend_meta = [], []
        F = len(FRAME_FRACTIONS)
        n = len(intervals)
        from PIL import Image

        def flush():
            nonlocal pend_imgs, pend_meta
            if not pend_imgs:
                return
            vecs = MODELS.clip_img.encode(pend_imgs, batch_size=32, convert_to_numpy=True,
                                          normalize_embeddings=True, show_progress_bar=False)
            if len(vecs) != len(pend_imgs):
                raise RuntimeError(f"Encoder trả {len(vecs)} vector cho {len(pend_imgs)} ảnh.")
            k = 0
            for cnt, vflags in pend_meta:
                part = vecs[k:k + cnt]
                k += cnt
                block = np.repeat(part[-1:], F, axis=0)      # cnt >= 1 được đảm bảo
                block[:cnt] = part
                vm = np.zeros(F, dtype=bool)
                vm[:cnt] = vflags                              # khung đệm = không hợp lệ (không tính 2 lần)
                emb_chunks.append(block)
                valid_chunks.append(vm)
            pend_imgs, pend_meta = [], []

        for i, (s, e) in enumerate(intervals):
            job.check()
            job.progress.set("analyzing", 10 + int(55 * i / max(1, n)), f"Đang phân tích hình ảnh cảnh {i + 1}/{n}...", vision_mode)
            frames, times, ok_q = grab_frames(cap, s, e, fps, total_dur)
            if not frames:
                continue
            feat = visual_features(frames)
            rel = ((s + e) / 2.0) / max(total_dur, 1.0)
            caption, source, status = heuristic_caption(feat, rel), "heuristic", "heuristic"
            if ollama_ready:
                d = describe_with_ollama(frames)
                if d:
                    caption, source, status = d, "ollama", "ollama_ok"
            if use_clip:
                pend_imgs.extend(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames)
                pend_meta.append((len(frames), ok_q))
                status = "clip" if status == "heuristic" else "clip+" + status
                if len(pend_meta) >= 16:
                    flush()
            while len(times) < F:
                times.append(times[-1])
            scenes.append({"scene_id": f"sc_{i:04d}", "start_time": round(s, 2), "end_time": round(e, 2),
                           "frame_times": [round(t, 2) for t in times[:F]], "motion_score": feat["motion"],
                           "face_count": feat["faces"], "brightness": feat["brightness"],
                           "caption": caption, "caption_source": source, "vision_status": status})
    finally:
        cap.release()

    img_emb = valid = None
    if use_clip:
        job.progress.set("embedding", 68, "Đang mã hóa hình ảnh bằng CLIP...", vision_mode)
        flush()
        if emb_chunks:
            shapes = {c.shape for c in emb_chunks}
            if len(shapes) != 1:
                raise RuntimeError(f"Embedding không đồng nhất: {shapes}")
            img_emb = np.stack(emb_chunks).astype(np.float32)
            valid = np.stack(valid_chunks)
    job.lap("frames_embed")
    return scenes, img_emb, valid, vision_mode, used_fallback


def analysis_config(max_scenes: int, use_clip: bool, ollama_ready: bool, detect_threshold: float) -> dict:
    try:
        import scenedetect
        sd = getattr(scenedetect, "__version__", "?")
    except ImportError:
        sd = "n/a"
    return {
        "algo": ALGO_VERSION, "schema": CACHE_SCHEMA, "max_scenes": max_scenes, "use_clip": use_clip,
        "ollama": ollama_ready, "clip_img": CLIP_IMG_MODEL if use_clip else None,
        "clip_txt": CLIP_TXT_MODEL if use_clip else None,
        "text_model": TEXT_MODEL if (ollama_ready or not use_clip) else None,
        "ollama_model": OLLAMA_MODEL if ollama_ready else None,
        "ollama_prompt": hashlib.sha1(OLLAMA_PROMPT.encode()).hexdigest()[:8] if ollama_ready else None,
        "fractions": list(FRAME_FRACTIONS), "frame_w": FRAME_WIDTH, "detect": round(detect_threshold, 2),
        "min_scene": MIN_SCENE_SEC, "max_scene": MAX_SCENE_SEC, "split": SPLIT_CHUNK_SEC,
        "scenedetect": sd, "cv2": ".".join(cv2.__version__.split(".")[:2]),
    }


def cache_paths(video_path: str, cfg: dict) -> tuple[Path, str]:
    p = Path(video_path).resolve()
    st = p.stat()
    vid = hashlib.sha256(f"{str(p).lower()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    return DATA_DIR / f"{vid}_{cfg_hash[:12]}.npz", cfg_hash


def save_cache(path: Path, cfg_hash: str, an: Analysis) -> None:
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:6]}.tmp.npz")
    try:
        meta = {"schema": CACHE_SCHEMA, "algo": ALGO_VERSION, "cfg_hash": cfg_hash, "scenes": an.scenes,
                "vision_mode": an.vision_mode, "total_dur": an.total_dur, "fps": an.fps,
                "used_fallback_split": an.used_fallback_split, "n": len(an.scenes)}
        np.savez_compressed(
            tmp, meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
            img=an.img_emb if an.img_emb is not None else np.zeros(0, np.float32),
            valid=an.valid if an.valid is not None else np.zeros(0, bool),
            cap=an.cap_emb if an.cap_emb is not None else np.zeros(0, np.float32),
            capmask=an.cap_mask if an.cap_mask is not None else np.zeros(0, bool))
        os.replace(tmp, path)      # ghi nguyên tử: không bao giờ để lại cache nửa vời
        prune_cache()
    except Exception as ex:  # noqa: BLE001
        log.warning("Không ghi được cache: %s", ex)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def load_cache(path: Path, cfg_hash: str, use_clip: bool) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            if meta.get("schema") != CACHE_SCHEMA or meta.get("algo") != ALGO_VERSION or meta.get("cfg_hash") != cfg_hash:
                return None
            scenes = meta["scenes"]
            m = len(scenes)
            img = z["img"] if z["img"].size else None
            valid = z["valid"] if z["valid"].size else None
            cap = z["cap"] if z["cap"].size else None
            capmask = z["capmask"] if z["capmask"].size else None
            if m == 0:
                return None
            if use_clip:
                if img is None or valid is None or img.ndim != 3 or img.shape[0] != m or img.shape[1] != len(FRAME_FRACTIONS):
                    return None
                if MODELS.dim and img.shape[2] != MODELS.dim:
                    return None
                if valid.shape != img.shape[:2] or not np.isfinite(img).all():
                    return None
            if cap is not None and (cap.shape[0] != m or capmask is None or capmask.shape[0] != m or not np.isfinite(cap).all()):
                return None
            for s in scenes:
                if "start_time" not in s or "end_time" not in s or len(s.get("frame_times", [])) != len(FRAME_FRACTIONS):
                    return None
            return {"scenes": scenes, "img": img, "valid": valid, "cap": cap, "capmask": capmask,
                    "vision_mode": meta.get("vision_mode", "cached"), "total_dur": float(meta.get("total_dur", 0)),
                    "fps": float(meta.get("fps", 24)), "fallback_split": bool(meta.get("used_fallback_split"))}
    except Exception as ex:  # noqa: BLE001
        log.warning("Cache hỏng, sẽ phân tích lại: %s", ex)
        return None


def prune_cache() -> None:
    try:
        files = [(f.stat().st_mtime, f.stat().st_size, f) for f in DATA_DIR.glob("*.npz")]
        total = sum(s for _, s, _ in files)
        limit = CACHE_MAX_MB * 1024 * 1024
        if total <= limit:
            return
        for _, size, f in sorted(files, key=lambda x: x[0]):
            f.unlink(missing_ok=True)
            total -= size
            if total <= limit * 0.8:
                break
    except Exception as ex:  # noqa: BLE001
        log.warning("Dọn cache lỗi: %s", ex)


def build_scene_sim(img_emb: Optional[np.ndarray], valid: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img_emb is None or valid is None:
        return None
    w = valid.astype(np.float32)[:, :, None]
    mean = (img_emb * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1.0)
    mean /= (np.linalg.norm(mean, axis=1, keepdims=True) + 1e-8)
    return (mean @ mean.T).astype(np.float32)


def get_analysis(job: Job, meta: dict, max_scenes: int, use_ollama: bool, detect_threshold: Optional[float],
                 allow_fallback: bool = False) -> Analysis:
    th = float(detect_threshold or DETECT_THRESHOLD)
    use_clip = MODELS.ensure_clip(job)
    if not use_clip and not allow_fallback:
        raise HTTPException(status_code=503, detail={"code": "clip_unavailable", "message":
            "Mô hình CLIP chưa sẵn sàng (%s). Kiểm tra mạng / thư viện sentence-transformers rồi thử lại "
            "(server tự thử nạp lại sau ~%.0f giây). Không tự dùng chế độ đoán mò vì sẽ chọn sai cảnh; "
            "nếu vẫn muốn, bật 'Cho phép chế độ dự phòng'." % (MODELS.error or "chưa nạp xong", CLIP_RETRY_SEC)})
    ollama_ready = bool(use_ollama and check_ollama_available())
    warnings = []
    if use_ollama and not ollama_ready:
        warnings.append("Đã bật Ollama nhưng dịch vụ/model không khả dụng: KHÔNG dùng caption heuristic để trộn với CLIP.")
    if not use_clip:
        warnings.append("CLIP không khả dụng (%s): đang dùng chế độ dự phòng caption chất lượng thấp, "
                        "không có kết quả nào được xếp 'Khớp tốt'." % (MODELS.error or "chưa sẵn sàng"))
    cfg = analysis_config(max_scenes, use_clip, ollama_ready, th)
    path, cfg_hash = cache_paths(meta["path"], cfg)
    cached = load_cache(path, cfg_hash, use_clip)
    job.lap("prepare")
    if cached:
        log.info("Nạp %d cảnh từ cache.", len(cached["scenes"]))
        an = Analysis(path.stem, cached["scenes"], cached["img"], cached["valid"], cached["cap"], cached["capmask"],
                      cached["vision_mode"], meta["duration"], meta["fps"], use_clip, ollama_ready,
                      from_cache=True, used_fallback_split=cached["fallback_split"], warnings=warnings)
    else:
        scenes, img, valid, vmode, fb = analyze_video(job, meta["path"], meta, max_scenes, use_clip, ollama_ready, th)
        if not scenes:
            raise HTTPException(status_code=500, detail="Không trích xuất được cảnh hợp lệ từ video!")
        if use_clip and img is None:
            raise HTTPException(status_code=500, detail="Không tạo được embedding hình ảnh!")
        cap_emb = cap_mask = None
        if (not use_clip) or ollama_ready:
            job.check()
            job.progress.set("embedding", 75, "Đang mã hóa mô tả cảnh...", vmode)
            mask = np.array([(s["caption_source"] == "ollama") or (not use_clip) for s in scenes], dtype=bool)
            if mask.any():
                idx = np.where(mask)[0]
                tv = MODELS.get_text_model().encode([scenes[k]["caption"] for k in idx], batch_size=32, convert_to_numpy=True,
                                                    normalize_embeddings=True, show_progress_bar=False)
                cap_emb = np.zeros((len(scenes), tv.shape[1]), np.float32)
                cap_emb[idx] = tv
                cap_mask = mask
            if ollama_ready and use_clip and (cap_mask is None or cap_mask.mean() < 0.5):
                warnings.append("Ollama chỉ tạo được mô tả cho %d/%d cảnh; các cảnh còn lại chỉ dùng CLIP." %
                                (0 if cap_mask is None else int(cap_mask.sum()), len(scenes)))
        job.lap("captions")
        an = Analysis(path.stem, scenes, img, valid, cap_emb, cap_mask, vmode, meta["duration"], meta["fps"],
                      use_clip, ollama_ready, used_fallback_split=fb, warnings=warnings)
        save_cache(path, cfg_hash, an)
    if an.used_fallback_split:
        an.warnings.append("Không dò được điểm cắt cảnh: video bị chia đều theo thời gian (chất lượng thấp).")
    an.scene_sim = build_scene_sim(an.img_emb, an.valid)
    return an


# ----------------------------------------------------------------------------
# Hiểu câu review (nhẹ): mệnh đề, hồi tưởng, phủ định
# ----------------------------------------------------------------------------
_CLAUSE_SPLIT = re.compile(r"[,;:]|\s(?:và|rồi|nhưng|sau đó|trong khi|khiến|để)\s", re.I)
_FLASHBACK = re.compile(r"nhớ lại|ngày xưa|quá khứ|trước đó|hồi ấy|hồi nhỏ|ký ức|kể lại|năm xưa", re.I)
_NEGATION = re.compile(r"(?<!\w)(không|chưa|chẳng|chả|đừng)(?!\w)", re.I)
_ACTION = re.compile(r"chiến đấu|tấn công|phản công|truy đuổi|(?<!\w)chạy(?!\w)|(?<!\w)đánh(?!\w)|tốc độ|(?<!\w)đòn(?!\w)|"
                     r"lao vào|kích hoạt|(?<!\w)nổ(?!\w)|va chạm|đuổi theo|chiến thắng", re.I)
_CALM = re.compile(r"nhớ lại|suy nghĩ|quan sát|nhận ra|trò chuyện|bí mật|quá khứ|ước mơ|(?<!\w)cha(?!\w)|(?<!\w)mẹ(?!\w)", re.I)


def split_clauses(sent: str) -> list[str]:
    parts = [p.strip(" .!?…\t") for p in _CLAUSE_SPLIT.split(sent) if p]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts[:4] if len(parts) >= 2 else []


def visual_bonus_matrix(sentences: list[str], scenes: list[dict]) -> np.ndarray:
    """Điểm cộng như v3.0: câu hành động -> cảnh nhiều chuyển động (tối đa +0.45); câu tâm lý/đối thoại -> cảnh có mặt người (+0.2).
    Không khóa sau bằng chứng CLIP (CLIP đa ngôn ngữ hiếm khi vượt ngưỡng cao). Bỏ qua câu phủ định."""
    motion = np.array([s["motion_score"] for s in scenes], dtype=np.float32)
    faces = np.array([s["face_count"] for s in scenes], dtype=np.float32)
    out = np.zeros((len(sentences), len(scenes)), dtype=np.float32)
    for i, sent in enumerate(sentences):
        if _NEGATION.search(sent):
            continue
        if _ACTION.search(sent):
            out[i] += np.minimum(0.45, motion / 60.0)
        if _CALM.search(sent):
            out[i] += 0.2 * (faces >= 1)
    return out


# ----------------------------------------------------------------------------
# Chấm điểm câu <-> cảnh
# ----------------------------------------------------------------------------
def zscore_rows(m: np.ndarray) -> np.ndarray:
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)


def aggregate_frames(sims: np.ndarray, valid: np.ndarray):
    """sims (T,m,F), valid (m,F) -> điểm cảnh (T,m) = 0.6*max + 0.4*mean (chỉ khung hợp lệ) và chỉ số khung tốt nhất."""
    v = valid[None, :, :]
    masked = np.where(v, sims, -np.inf)
    mx = masked.max(axis=2)
    cnt = valid.sum(axis=1)
    mean = np.where(v, sims, 0.0).sum(axis=2) / np.maximum(cnt, 1)[None, :]
    raw = 0.6 * mx + 0.4 * mean
    bad = cnt == 0
    if bad.any():
        good_min = raw[:, ~bad].min(axis=1, keepdims=True) if (~bad).any() else np.zeros((raw.shape[0], 1), np.float32)
        raw[:, bad] = good_min - 0.02
    fidx = np.argmax(masked, axis=2).astype(np.int32)
    return raw.astype(np.float32), fidx


def encode_queries(sentences: list[str]):
    clauses = [split_clauses(s) if USE_CLAUSES else [] for s in sentences]
    flat = list(sentences) + [c for cl in clauses for c in cl]
    vecs = MODELS.clip_txt.encode(flat, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    n = len(sentences)
    sent_vecs, off, cvecs = vecs[:n], n, []
    for cl in clauses:
        k = len(cl)
        cvecs.append(vecs[off:off + k] if k else None)
        off += k
    return sent_vecs, cvecs


def detect_abstract_rows(raw: np.ndarray, force: bool = False) -> np.ndarray:
    """Câu KHÔNG có hình ảnh đặc trưng (tên riêng, câu dẫn trừu tượng).
    Tiêu chí: (1) cosine thô cao nhất của cả phim vẫn dưới raw_low; hoặc (2) với kịch bản >= 8 câu, cosine cao nhất của câu này
    thấp bất thường so với chính các câu khác của kịch bản (tự thích nghi theo từng phim, không cần ngưỡng cứng).
    Không dùng (max - median): với hàng trăm cảnh, nhiễu thuần túy cũng đã cách median ~2.7 độ lệch chuẩn."""
    n = raw.shape[0]
    if force:
        return np.ones(n, dtype=bool)
    mx = raw.max(axis=1)
    out = mx < CAL.raw_low
    if n >= 8:
        med = float(np.median(mx))
        upper = mx[mx >= med] - med        # độ lệch chỉ tính trên nửa TRÊN: không bị chính nhóm câu trừu tượng làm phồng
        sigma = 1.4826 * float(np.median(upper)) if upper.size else 0.0
        out |= mx < med - max(2.0 * sigma, 0.02)
    return out


def score_matrix(sentences: list[str], an: Analysis, flag_abstract: bool = True, force_abstract: bool = False) -> dict:
    n, m = len(sentences), len(an.scenes)
    clip_ok = an.img_emb is not None and MODELS.clip_txt is not None
    cap_ok = an.cap_emb is not None and an.cap_mask is not None and bool(an.cap_mask.any())
    raw = None
    fidx = np.ones((n, m), dtype=np.int32)
    sent_vecs, cvecs = None, None
    z_clip = z_cap = None
    if clip_ok:
        sent_vecs, cvecs = encode_queries(sentences)
        sims = np.einsum("nd,mfd->nmf", sent_vecs, an.img_emb)
        raw, fidx = aggregate_frames(sims, an.valid)
        for i, cv in enumerate(cvecs):
            if cv is not None:
                r2, _ = aggregate_frames(np.einsum("cd,mfd->cmf", cv, an.img_emb), an.valid)
                raw[i] = 0.65 * raw[i] + 0.35 * r2.max(axis=0)
        z_clip = zscore_rows(raw)
    if cap_ok:
        se = MODELS.get_text_model().encode(sentences, batch_size=32, convert_to_numpy=True,
                                            normalize_embeddings=True, show_progress_bar=False)
        cs = np.where(an.cap_mask[None, :], se @ an.cap_emb.T, np.nan)
        cs = np.where(np.isnan(cs), np.nanmean(cs, axis=1, keepdims=True), cs)
        z_cap = zscore_rows(cs)
    if z_clip is not None and z_cap is not None:
        base = zscore_rows(0.65 * z_clip + 0.35 * z_cap)
    elif z_clip is not None:
        base = z_clip
    elif z_cap is not None:
        base = z_cap
    else:
        raise HTTPException(status_code=500, detail="Không có dữ liệu embedding để khớp cảnh!")
    base = base.astype(np.float32)
    abstract = np.zeros(n, dtype=bool)
    if clip_ok:
        evid = np.clip((raw - CAL.raw_low) / (CAL.raw_high - CAL.raw_low), 0.0, 1.0).astype(np.float32)
        if (flag_abstract or force_abstract) and m > 1:
            # z-score phóng đại nhiễu của câu trừu tượng thành điểm cao -> làm phẳng hàng để chuỗi Viterbi (thứ tự phim) quyết định.
            abstract = detect_abstract_rows(raw, force_abstract)
            base[abstract] = 0.0
    else:
        raw = np.zeros((n, m), np.float32)
        evid = np.full((n, m), 0.5, np.float32)
    return {"base": base, "raw": raw, "e": evid, "fidx": fidx, "fallback": not clip_ok, "abstract": abstract,
            "sent_vecs": sent_vecs, "cvecs": cvecs}


# ----------------------------------------------------------------------------
# Tối ưu chuỗi (Viterbi có trạng thái "không chọn cảnh")
# ----------------------------------------------------------------------------
def build_transitions(m: int, back_pen: float, fwd_coef: float, dup_sim: Optional[np.ndarray]) -> np.ndarray:
    idx = np.arange(m)
    d = idx[None, :] - idx[:, None]            # j - k
    trans = np.zeros((m, m), dtype=np.float32)
    back = d < 0
    trans[back] = back_pen + 0.004 * (-d[back])
    fwd = d > 1
    trans[fwd] = fwd_coef * np.log1p((d[fwd] - 1) / 10.0)
    fwd1 = d >= 1
    trans[fwd1] += FWD_TIE_EPS * np.log1p(d[fwd1])      # phá hòa: câu "phẳng" đi tiếp sang cảnh gần nhất phía sau
    if dup_sim is not None:
        trans += np.where(dup_sim >= DUP_SIM, DUP_SCENE_PENALTY, 0.0).astype(np.float32)
    trans[idx, idx] = SAME_SCENE_PENALTY
    return trans


def sequence_optimize(score: np.ndarray, trans: np.ndarray, null_score: Optional[float]) -> list[int]:
    """Trả chỉ số cảnh cho mỗi câu; giá trị m (= số cảnh) nghĩa là 'không chọn cảnh phù hợp'."""
    n, m = score.shape
    if null_score is not None:
        S = np.concatenate([score, np.full((n, 1), null_score, np.float32)], axis=1)
        T = np.zeros((m + 1, m + 1), np.float32)
        T[:m, :m] = trans
    else:
        S, T = score.astype(np.float32), trans
    k = S.shape[1]
    idx = np.arange(k)
    dp = S[0].copy()
    ptr = np.zeros((n, k), dtype=np.int32)
    for i in range(1, n):
        cand = dp[:, None] - T
        ptr[i] = np.argmax(cand, axis=0)
        dp = S[i] + cand[ptr[i], idx]
    path = [int(np.argmax(dp))]
    for i in range(n - 1, 0, -1):
        path.append(int(ptr[i][path[-1]]))
    path.reverse()
    return path


def find_reuse_violations(path: list[int], m: int, dup_sim: Optional[np.ndarray], window: int):
    out = []
    for i, j in enumerate(path):
        if j >= m:
            continue
        for i2 in range(i + 1, min(len(path), i + 1 + window)):
            j2 = path[i2]
            if j2 >= m:
                continue
            if j2 == j or (dup_sim is not None and dup_sim[j, j2] >= DUP_SIM):
                out.append((i, i2))
    return out


def optimize_with_diversity(score: np.ndarray, trans: np.ndarray, null_score: Optional[float],
                            dup_sim: Optional[np.ndarray], window: int = REUSE_WINDOW, iters: int = 4) -> list[int]:
    """Viterbi + vòng lặp phạt tái sử dụng theo lịch sử (không chỉ hai câu liền kề)."""
    m = score.shape[1]
    sc = score.copy()
    path = sequence_optimize(sc, trans, null_score)
    for _ in range(iters):
        viol = find_reuse_violations(path, m, dup_sim, window)
        if not viol:
            break
        for i, i2 in viol:
            loser = i2 if sc[i2, path[i2]] <= sc[i, path[i]] else i
            sc[loser, path[loser]] -= REUSE_STEP
        path = sequence_optimize(sc, trans, null_score)
    return path


def top_candidates(rank_row: np.ndarray, sim: Optional[np.ndarray], k: int, excluded: Optional[np.ndarray] = None) -> list[int]:
    chosen: list[int] = []
    for j in np.argsort(-rank_row):
        j = int(j)
        if excluded is not None and excluded[j]:
            continue
        if sim is not None and any(sim[j, c] >= CAND_DUP_SIM for c in chosen):
            continue
        chosen.append(j)
        if len(chosen) >= k:
            break
    return chosen


def z_cap(m: int) -> float:
    """z-score tối đa có thể đạt khi chỉ có m cảnh (video ngắn/ít cảnh thì ngưỡng cố định là bất khả thi)."""
    return max(0.0, (m - 1) / math.sqrt(max(m, 1)))


def eff_threshold(th: float, m: int, frac: float = 0.8) -> float:
    return min(th, frac * z_cap(m))


def classify(z: float, raw: float, gap: float, fallback: bool, abstract: bool, m: int = 10_000) -> tuple[str, str]:
    """(nhãn hiển thị, trạng thái). Chỉ dùng để GẮN NHÃN, không ảnh hưởng việc chọn cảnh. Ngưỡng chưa hiệu chỉnh trừ khi có calibration.json."""
    if abstract:
        return "Câu trừu tượng (theo mạch)", "abstract"
    z_ok, z_review = eff_threshold(CAL.z_ok, m), eff_threshold(CAL.z_review, m)
    if fallback:
        return ("Cần kiểm tra", "review") if z >= z_review else ("Khớp thấp", "low")
    if raw < CAL.raw_low:
        return "Khớp thấp", "low"
    strong_raw = raw >= CAL.raw_low + 0.4 * (CAL.raw_high - CAL.raw_low)
    if z >= z_ok and gap <= 0.3 and strong_raw:
        return "Khớp tốt", "ok"
    if z >= z_review or (raw >= CAL.raw_high and gap <= 0.3):   # bằng chứng tuyệt đối mạnh: ít nhất đáng xem
        return "Cần kiểm tra", "review"
    return "Khớp thấp", "low"


# ----------------------------------------------------------------------------
# Dò kỹ thời điểm (lấy mẫu dày quanh cảnh ứng viên)
# ----------------------------------------------------------------------------
class DenseCache:
    def __init__(self, cap: int = 600) -> None:
        self._d: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self._cap = cap

    def get(self, k):
        with self._lock:
            v = self._d.get(k)
            if v is not None:
                self._d.move_to_end(k)
            return v

    def put(self, k, v) -> None:
        with self._lock:
            self._d[k] = v
            while len(self._d) > self._cap:
                self._d.popitem(last=False)


DENSE = DenseCache()


class VideoReader:
    """Mở video lười, luôn đóng đúng cách."""
    def __init__(self, path: str, fps: float, total_dur: float) -> None:
        self.path, self.fps, self.total_dur, self.cap = path, fps, total_dur, None

    def __enter__(self):
        return self

    def get(self) -> cv2.VideoCapture:
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise HTTPException(status_code=500, detail="Không mở được video để dò thời điểm.")
        return self.cap

    def __exit__(self, *_):
        if self.cap is not None:
            self.cap.release()


def dense_embed(reader: VideoReader, key: str, scene: dict):
    ck = (key, scene["scene_id"])
    hit = DENSE.get(ck)
    if hit is not None:
        return hit
    from PIL import Image
    s, e = scene["start_time"], scene["end_time"]
    dur = max(0.5, e - s)
    n = int(min(DENSE_MAX_FRAMES, max(5, math.ceil(dur / DENSE_STEP_SEC))))
    pad = min(0.1, dur * 0.05)
    imgs, ts, oks = [], [], []
    for t in np.linspace(s + pad, max(s + pad, e - pad), n):
        fr, ta = read_frame_at(reader.get(), float(t), reader.fps, reader.total_dur)
        if fr is None:
            continue
        sm = small_frame(fr)
        imgs.append(Image.fromarray(cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)))
        ts.append(ta)
        oks.append(frame_ok(sm))
    if not imgs:
        return None
    if not any(oks):
        oks = [True] * len(imgs)
    vecs = MODELS.clip_img.encode(imgs, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    out = (np.array(ts, dtype=np.float64), vecs.astype(np.float32), np.array(oks, dtype=bool))
    DENSE.put(ck, out)
    return out


def localize_action(times: np.ndarray, vecs: np.ndarray, valid: np.ndarray, q_full: np.ndarray, q_clauses) -> dict:
    sims = vecs @ q_full
    if q_clauses is not None and len(q_clauses):
        sims = 0.65 * sims + 0.35 * (vecs @ q_clauses.T).max(axis=1)
    med = float(np.median(sims[valid])) if valid.any() else float(np.median(sims))
    sims = np.where(valid, sims, med)
    padded = np.concatenate([[sims[0]], sims, [sims[-1]]])
    smooth = np.convolve(padded, [0.25, 0.5, 0.25], mode="valid")
    k = int(np.argmax(smooth))
    peak = float(smooth[k])
    floor = float(np.median(smooth))
    thr = peak - 0.4 * max(peak - floor, 1e-6)
    a = b = k
    while a > 0 and smooth[a - 1] >= thr:
        a -= 1
    while b < len(smooth) - 1 and smooth[b + 1] >= thr:
        b += 1
    prom = peak - floor
    return {"focus_sec": round(float(times[k]), 2), "action_start_sec": round(float(times[a]), 2),
            "action_end_sec": round(float(times[b]), 2), "peak_prominence": round(prom, 4),
            "time_confidence": "cao" if (prom >= 0.02 and len(times) >= 6) else "thấp"}


def refine_scene(reader: VideoReader, an: Analysis, j: int, q_full: np.ndarray, q_clauses) -> Optional[dict]:
    sc = an.scenes[j]
    d = dense_embed(reader, an.key, sc)
    if d is None:
        return None
    return localize_action(d[0], d[1], d[2], q_full, q_clauses)


def coarse_focus(an: Analysis, j: int, fidx_ij: int) -> dict:
    sc = an.scenes[j]
    mid = round((sc["start_time"] + sc["end_time"]) / 2.0, 2)
    focus = sc["frame_times"][int(fidx_ij)] if an.img_emb is not None else mid
    return {"focus_sec": focus, "action_start_sec": sc["start_time"], "action_end_sec": sc["end_time"],
            "peak_prominence": 0.0, "time_confidence": "thô"}


def cand_dict(an: Analysis, j: int, z: float, raw: float, e: float, timing: dict) -> dict:
    sc = an.scenes[j]
    return {"scene_id": sc["scene_id"], "start_sec": sc["start_time"], "end_sec": sc["end_time"],
            "score": round(float(z), 3), "raw_sim": round(float(raw), 4), "evidence": round(float(e), 3),
            "caption": sc["caption"], "vision_status": sc["vision_status"], **timing}


# ----------------------------------------------------------------------------
# Khớp toàn bộ kịch bản
# ----------------------------------------------------------------------------
def match_sentences(job: Job, req: MatchRequest, an: Analysis, meta: dict) -> tuple[list[dict], dict]:
    sentences = req.sentences
    n, m = len(sentences), len(an.scenes)
    S = score_matrix(sentences, an, flag_abstract=req.allow_no_match)
    base, e, raw, fallback, abstract = S["base"], S["e"], S["raw"], S["fallback"], S["abstract"]
    job.lap("score")

    back, fwd, prior_w = ORDER_MODES[req.order_mode]
    if req.order_penalty is not None:
        back = req.order_penalty
    mids = np.array([(s["start_time"] + s["end_time"]) / 2.0 for s in an.scenes]) / max(an.total_dur, 1.0)
    prior = np.zeros((n, m), np.float32)
    if prior_w > 0:
        expected = (np.arange(n) + 0.5) / n
        mult = np.array([0.3 if _FLASHBACK.search(s) else 1.0 for s in sentences])
        prior = (prior_w * mult[:, None] * np.exp(-0.5 * ((mids[None, :] - expected[:, None]) / 0.30) ** 2)).astype(np.float32)
    bonus = visual_bonus_matrix(sentences, an.scenes)
    bonus[abstract] = 0.0
    adj = (base + prior + bonus).astype(np.float32)        # ĐIỂM CHỌN = z-score (+ prior nhẹ + điểm cộng), như v3.0

    trans = build_transitions(m, back, fwd, None)           # chỉ phạt nhảy lùi + dùng lại đúng 1 cảnh liên tiếp
    if REUSE_WINDOW > 0:
        path = optimize_with_diversity(adj, trans, None, an.scene_sim, window=REUSE_WINDOW)
    else:
        path = sequence_optimize(adj, trans, None)          # một chuỗi liền mạch, không trạng thái "null"
    job.lap("sequence")

    reader = VideoReader(meta["path"], an.fps, an.total_dur)
    results = []
    with reader:
        for i, j in enumerate(path):
            job.check()
            is_abs = bool(abstract[i])
            order = np.argsort(-base[i])
            top_z = float(base[i, order[0]])
            gap = top_z - float(base[i, j])
            second = float(base[i, order[1]]) if m > 1 else top_z
            z, rw, ev = float(base[i, j]), float(raw[i, j]), float(e[i, j])
            conf, status = classify(z, rw, gap, fallback, is_abs, m)

            do_refine = req.refine and an.img_emb is not None
            if do_refine:
                job.progress.set("verifying", 90 + int(9 * i / max(1, n)), f"Đang dò kỹ thời điểm câu {i + 1}/{n}...", an.vision_mode)
            timing = (refine_scene(reader, an, j, S["sent_vecs"][i], S["cvecs"][i]) if do_refine and not is_abs else None) \
                or coarse_focus(an, j, S["fidx"][i, j])

            if is_abs:      # ứng viên thay thế = các cảnh lân cận theo mạch phim
                ranked = [c for c in (j + 1, j - 1, j + 2, j - 2) if 0 <= c < m][:req.top_k]
            else:
                ranked = [c for c in top_candidates(adj[i], an.scene_sim, req.top_k + 1) if c != j][:req.top_k]
            cands = [cand_dict(an, c, base[i, c], raw[i, c], e[i, c], coarse_focus(an, c, S["fidx"][i, c])) for c in ranked]
            sc = an.scenes[j]
            res = {"index": i + 1, "text": sentences[i], "scene_id": sc["scene_id"],
                   "start_sec": sc["start_time"], "end_sec": sc["end_time"],
                   "boundary_start": sc["start_time"], "boundary_end": sc["end_time"],
                   "caption": sc["caption"], "vision_status": sc["vision_status"],
                   "score": round(z, 3), "raw_sim": round(rw, 4), "evidence": round(ev, 3),
                   "margin": round(top_z - second, 3), "gap_to_best": round(gap, 3),
                   "confidence": conf, "status": status, "candidates": cands, **timing}
            if is_abs:
                res["reason"] = "Câu không có hình ảnh đặc trưng (tên riêng/câu dẫn trừu tượng): cảnh được chọn theo mạch phim, không theo điểm hình ảnh."
            results.append(res)
    job.lap("verify")
    info = {"fallback": fallback, "ranking": "clip_chain_viterbi" if not fallback else "caption_viterbi_fallback"}
    return results, info


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        import edge_tts  # noqa: F401
        edge = True
    except ImportError:
        edge = False
    return {"status": "ok", "version": ALGO_VERSION, "match_version": MATCH_VERSION, "clip": MODELS.state, "clip_error": MODELS.error,
            "embed_dim": MODELS.dim, "edge_tts": edge, "busy": JOB_LOCK.locked(),
            "auth_required": bool(API_KEY), "calibrated": CAL.calibrated}


@app.get("/progress", dependencies=[Depends(require_key)])
def get_progress(job_id: Optional[str] = None):
    job = get_job(job_id) if job_id else (ACTIVE_JOB or LAST_JOB)
    if job is None:
        return {"status": "idle", "percent": 0, "message": "Sẵn sàng", "vision_mode": "unknown", "job_id": job_id}
    d = job.progress.snapshot()
    d["job_id"] = job.id
    return d


@app.post("/cancel", dependencies=[Depends(require_key)])
def cancel(req: Optional[CancelRequest] = None):
    job = get_job(req.job_id) if (req and req.job_id) else ACTIVE_JOB
    if job is None or job is not ACTIVE_JOB:
        return {"status": "not_running"}      # job đã xong/không tồn tại -> không ảnh hưởng job khác
    job.cancel.set()
    return {"status": "cancelling", "job_id": job.id}


@app.get("/cache/stats", dependencies=[Depends(require_key)])
def cache_stats():
    files = list(DATA_DIR.glob("*.npz"))
    return {"files": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1048576, 2), "limit_mb": CACHE_MAX_MB}


@app.post("/cache/clear", dependencies=[Depends(require_key)])
def cache_clear():
    k = 0
    for f in DATA_DIR.glob("*.npz"):
        f.unlink(missing_ok=True)
        k += 1
    return {"deleted": k}


def _run_job(job: Job, fn):
    global ACTIVE_JOB, LAST_JOB
    if not JOB_LOCK.acquire(blocking=False):
        raise busy_error()
    register_job(job)
    ACTIVE_JOB = job
    try:
        return fn()
    except JobCancelled:
        job.progress.set("cancelled", 0, "Đã hủy")
        raise HTTPException(status_code=499, detail={"code": "cancelled", "message": "Đã hủy bởi người dùng."})
    finally:
        LAST_JOB, ACTIVE_JOB = job, None
        JOB_LOCK.release()


@app.post("/auto_match_all", dependencies=[Depends(require_key)])
def auto_match_all(req: MatchRequest):
    job = Job(req.job_id)

    def work():
        meta = validate_video_file(req.video_path)
        an = get_analysis(job, meta, req.max_scenes, req.use_ollama, req.detect_threshold, req.allow_fallback)
        job.check()
        job.progress.set("matching", 88, "Đang chọn ứng viên và tối ưu chuỗi cảnh...", an.vision_mode)
        results, info = match_sentences(job, req, an, meta)
        n_nm = sum(1 for r in results if r["status"] == "no_match")
        job.progress.set("done", 100, f"Hoàn tất khớp {len(results)} phân cảnh ({n_nm} chưa tìm thấy)!", an.vision_mode)
        return {"job_id": job.id, "vision_mode": an.vision_mode, "matcher": info["ranking"], "calibrated": CAL.calibrated,
                "from_cache": an.from_cache, "warnings": an.warnings, "timings": job.timings, "results": results}

    return _run_job(job, work)


@app.post("/rematch", dependencies=[Depends(require_key)])
def rematch(req: RematchRequest):
    """Tìm lại cảnh khác cho MỘT câu (loại các đoạn đã thử, có thể giới hạn gần một mốc thời gian)."""
    job = Job(req.job_id)

    def work():
        meta = validate_video_file(req.video_path)
        an = get_analysis(job, meta, req.max_scenes, req.use_ollama, req.detect_threshold, req.allow_fallback)
        job.check()
        m = len(an.scenes)
        S = score_matrix([req.sentence], an, force_abstract=req.abstract)
        base, e, raw = S["base"][0], S["e"][0], S["raw"][0]
        is_abs = bool(S["abstract"][0])
        adj = base if is_abs else base + visual_bonus_matrix([req.sentence], an.scenes)[0]
        starts = np.array([s["start_time"] for s in an.scenes])
        ends = np.array([s["end_time"] for s in an.scenes])
        mids = (starts + ends) / 2.0
        durs = np.maximum(ends - starts, 1e-3)
        if is_abs and req.near_sec is not None:       # câu trừu tượng: không có điểm hình ảnh -> ưu tiên cảnh gần mốc hiện tại
            adj = -np.abs(mids - req.near_sec) / max(an.total_dur, 1.0)

        excluded = np.zeros(m, dtype=bool)
        seed = np.zeros(m, dtype=bool)
        for rng in req.exclude:
            if len(rng) < 2:
                continue
            ov = np.minimum(ends, rng[1]) - np.maximum(starts, rng[0])
            seed |= (ov / durs) > 0.4
        excluded |= seed
        if an.scene_sim is not None and seed.any():
            excluded |= an.scene_sim[:, seed].max(axis=1) >= CAND_DUP_SIM

        window_expanded = False
        window = req.window_sec
        pool = excluded.copy()
        if req.near_sec is not None and window > 0:
            while True:
                far = np.abs(mids - req.near_sec) > window
                avail = int((~(excluded | far)).sum())
                if avail >= req.top_k or window >= an.total_dur:
                    pool = excluded | far
                    break
                window *= 2.0
                window_expanded = True
        ranked = top_candidates(adj, an.scene_sim, req.top_k, pool)
        order = np.argsort(-base)
        top_z = float(base[order[0]])
        out = []
        with VideoReader(meta["path"], an.fps, an.total_dur) as reader:
            for r, j in enumerate(ranked):
                job.check()
                tm = None
                if req.refine and an.img_emb is not None and r < 3:
                    tm = refine_scene(reader, an, j, S["sent_vecs"][0], S["cvecs"][0])
                tm = tm or coarse_focus(an, j, S["fidx"][0, j])
                conf, status = classify(float(base[j]), float(raw[j]), top_z - float(base[j]), S["fallback"], is_abs, m)
                d = cand_dict(an, j, base[j], raw[j], e[j], tm)
                d.update({"confidence": conf, "status": status, "distance_sec": None if req.near_sec is None else round(float(abs(mids[j] - req.near_sec)), 2)})
                out.append(d)
        return {"job_id": job.id, "candidates": out, "window_sec": window, "window_expanded": window_expanded,
                "calibrated": CAL.calibrated, "warnings": an.warnings, "considered": int((~pool).sum())}

    return _run_job(job, work)


_TTS_SEM = asyncio.Semaphore(4)


@app.post("/tts", dependencies=[Depends(require_key)])
async def tts(req: TTSRequest):
    rate_m, pitch_m = re.fullmatch(r"([+-])(\d{1,3})%", req.rate), re.fullmatch(r"([+-])(\d{1,3})Hz", req.pitch)
    if not rate_m or not pitch_m:
        raise HTTPException(status_code=400, detail="rate/pitch không hợp lệ (ví dụ +10% / -2Hz).")
    if int(rate_m.group(2)) > 100 or int(pitch_m.group(2)) > 50:
        raise HTTPException(status_code=400, detail="rate tối đa ±100%, pitch tối đa ±50Hz.")
    if req.voice not in EDGE_VOICES_ALLOWED:
        raise HTTPException(status_code=400, detail="Giọng đọc không nằm trong danh sách cho phép.")
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(status_code=501, detail="Chưa cài edge-tts: pip install edge-tts")
    last: Optional[Exception] = None
    async with _TTS_SEM:
        for attempt in range(3):
            try:
                async def run():
                    buf = bytearray()
                    comm = edge_tts.Communicate(req.text, req.voice, rate=req.rate, pitch=req.pitch)
                    async for ch in comm.stream():
                        if ch["type"] == "audio":
                            buf.extend(ch["data"])
                    return bytes(buf)
                data = await asyncio.wait_for(run(), timeout=45)
                if len(data) > 500:
                    return Response(content=data, media_type="audio/mpeg")
            except Exception as ex:  # noqa: BLE001
                last = ex
            await asyncio.sleep(0.8 * (attempt + 1))
    log.warning("edge-tts thất bại: %s", last)
    raise HTTPException(status_code=502, detail="Dịch vụ giọng đọc (edge-tts) tạm thời không khả dụng.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("AI_HOST", "127.0.0.1"), port=int(os.getenv("AI_PORT", "8000")), log_level="info")

def match_sentences(job, req, an, meta):
    sentences = req.sentences
    n, m = len(sentences), len(an.scenes)

    if n == 0 or m == 0:
        return [], {"fallback": False, "ranking": "clip_chain_viterbi"}

    S = score_matrix(
        sentences,
        an,
        flag_abstract=True,
        force_abstract=False,
    )

    base = S["base"].copy().astype(np.float32)
    raw = S["raw"].copy().astype(np.float32)
    evidence = S["e"].copy().astype(np.float32)
    fidx = S["fidx"]
    fallback = S["fallback"]
    abstract = S["abstract"]
    job.lap("score")

    back, fwd, prior_w = ORDER_MODES.get(
        req.order_mode, ORDER_MODES["free"]
    )

    if req.order_penalty is not None:
        back = req.order_penalty

    mids = np.array([
        (s["start_time"] + s["end_time"]) / 2.0
        for s in an.scenes
    ], dtype=np.float32) / max(an.total_dur, 1.0)

    prior = np.zeros((n, m), dtype=np.float32)

    if prior_w > 0:
        expected = (np.arange(n, dtype=np.float32) + 0.5) / n
        mult = np.array([
            0.2 if _FLASHBACK.search(s) else 1.0
            for s in sentences
        ], dtype=np.float32)

        prior = (
            prior_w
            * mult[:, None]
            * np.exp(
                -0.5
                * ((mids[None, :] - expected[:, None]) / 0.30) ** 2
            )
        ).astype(np.float32)

    bonus = visual_bonus_matrix(sentences, an.scenes)
    bonus[abstract] = 0.0

    adj = base + prior + bonus

    no_match = np.zeros(n, dtype=bool)

    if req.allow_no_match and not fallback:
        max_raw = raw.max(axis=1)
        no_match = (~abstract) & (max_raw < CAL.raw_low)

    trans = build_transitions(m, back, fwd, None)

    if REUSE_WINDOW > 0:
        path = optimize_with_diversity(
            adj.copy(),
            trans,
            -0.25,
            an.scene_sim,
            window=REUSE_WINDOW,
        )
    else:
        path = sequence_optimize(adj, trans, -0.25)

    path = np.asarray(path, dtype=np.int32)

    for i in range(n):
        if no_match[i]:
            path[i] = m

    job.lap("sequence")

    results = []

    with VideoReader(meta["path"], an.fps, an.total_dur) as reader:
        for i, j in enumerate(path):
            job.check()

            if j >= m:
                results.append({
                    "index": i + 1,
                    "text": sentences[i],
                    "scene_id": None,
                    "start_sec": None,
                    "end_sec": None,
                    "boundary_start": None,
                    "boundary_end": None,
                    "focus_sec": None,
                    "action_start_sec": None,
                    "action_end_sec": None,
                    "caption": "",
                    "vision_status": an.vision_mode,
                    "score": 0.0,
                    "raw_sim": round(float(raw[i].max()), 4),
                    "evidence": round(float(evidence[i].max()), 3),
                    "margin": 0.0,
                    "gap_to_best": 0.0,
                    "confidence": "Không tìm thấy cảnh phù hợp",
                    "status": "no_match",
                    "candidates": [],
                    "reason": "Độ tương đồng hình ảnh thấp; cần duyệt thủ công.",
                })
                continue

            sc = an.scenes[j]
            order = np.argsort(-adj[i])
            best_score = float(adj[i, order[0]])
            second_score = (
                float(adj[i, order[1]]) if m > 1 else best_score
            )

            z = float(base[i, j])
            rw = float(raw[i, j])
            ev = float(evidence[i, j])
            gap = best_score - float(adj[i, j])

            conf, status = classify(
                z, rw, gap, fallback, bool(abstract[i]), m
            )

            timing = None

            if req.refine and an.img_emb is not None and not abstract[i]:
                job.progress.set(
                    "verifying",
                    90 + int(9 * i / max(1, n)),
                    f"Đang dò kỹ thời điểm câu {i + 1}/{n}...",
                    an.vision_mode,
                )

                timing = refine_scene(
                    reader,
                    an,
                    j,
                    S["sent_vecs"][i],
                    S["cvecs"][i],
                )

            if timing is None:
                timing = coarse_focus(an, j, fidx[i, j])

            if abstract[i]:
                ranked = [
                    c for c in (j + 1, j - 1, j + 2, j - 2)
                    if 0 <= c < m
                ][:req.top_k]
            else:
                ranked = [
                    c for c in top_candidates(
                        adj[i],
                        an.scene_sim,
                        req.top_k + 1,
                    )
                    if c != j
                ][:req.top_k]

            candidates = [
                cand_dict(
                    an,
                    c,
                    base[i, c],
                    raw[i, c],
                    evidence[i, c],
                    coarse_focus(an, c, fidx[i, c]),
                )
                for c in ranked
            ]

            results.append({
                "index": i + 1,
                "text": sentences[i],
                "scene_id": sc["scene_id"],
                "start_sec": sc["start_time"],
                "end_sec": sc["end_time"],
                "boundary_start": sc["start_time"],
                "boundary_end": sc["end_time"],
                "caption": sc["caption"],
                "vision_status": sc["vision_status"],
                "score": round(z, 3),
                "raw_sim": round(rw, 4),
                "evidence": round(ev, 3),
                "margin": round(best_score - second_score, 3),
                "gap_to_best": round(gap, 3),
                "confidence": conf,
                "status": status,
                "candidates": candidates,
                **timing,
                **({
                    "reason": "Câu trừu tượng; chọn cảnh theo mạch phim."
                } if abstract[i] else {}),
            })

    job.lap("verify")

    return results, {
        "fallback": fallback,
        "ranking": (
            "clip_chain_viterbi"
            if not fallback
            else "caption_viterbi_fallback"
        ),
    }
