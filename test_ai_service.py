"""Kiểm thử ai_service v3.1 (không cần tải CLIP: dùng encoder giả theo màu).

Chạy:  pytest -q test_ai_service.py
"""
import os
import tempfile

os.environ["AI_CACHE_DIR"] = tempfile.mkdtemp(prefix="aicache_")

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import ai_service as ai

COLORS = {"đỏ": (0, 0, 255), "xanh lá": (0, 255, 0), "xanh dương": (255, 0, 0), "vàng": (0, 255, 255)}  # BGR
DIM = 8


def _vec(bgr):
    b, g, r = [c / 255.0 for c in bgr]
    v = np.array([r, g, b, 0.05, 0.02, 0.01, 0.0, 0.03], np.float32)
    return v / np.linalg.norm(v)


class FakeImg:
    def encode(self, imgs, **kw):
        out = []
        for im in imgs:
            a = np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(axis=0)  # RGB
            out.append(_vec((a[2], a[1], a[0])))
        return np.array(out)


class FakeTxt:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            v = np.array([0, 0, 0, 0, 0, 0, 1.0, 0], np.float32)   # câu trừu tượng: không có bằng chứng hình ảnh
            for name, bgr in COLORS.items():
                if name in t:
                    v = _vec(bgr)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    ai.MODELS.clip_img, ai.MODELS.clip_txt, ai.MODELS.dim, ai.MODELS.state = FakeImg(), FakeTxt(), DIM, "ready"
    monkeypatch.setattr(ai.CAL, "raw_low", 0.2)
    monkeypatch.setattr(ai.CAL, "raw_high", 0.9)
    yield


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """12 giây: đỏ(0-3) xanh lá(3-6) xanh dương(6-9) vàng(9-12) với nhiễu để khỏi trùng tuyệt đối."""
    path = str(tmp_path_factory.mktemp("v") / "t.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 90))
    rng = np.random.default_rng(1)
    seq = [COLORS["đỏ"], COLORS["xanh lá"], COLORS["xanh dương"], COLORS["vàng"]]
    for k in range(120):
        base = np.array(seq[k // 30], np.uint8)
        fr = np.tile(base, (90, 160, 1)).astype(np.int16) + rng.integers(-6, 7, (90, 160, 3))
        w.write(np.clip(fr, 0, 255).astype(np.uint8))
    w.release()
    return path


# ---------- Viterbi / tối ưu chuỗi ----------
def test_null_state_rejects_weak_rows():
    score = np.array([[3.0, 0.2, 0.1], [0.4, 0.3, 0.2], [0.1, 0.2, 3.0]], np.float32)
    trans = ai.build_transitions(3, 1.2, 0.1, None)
    path = ai.sequence_optimize(score, trans, null_score=1.25)
    assert path == [0, 3, 2]          # hàng giữa yếu -> 3 = null


def test_no_null_forces_choice():
    score = np.array([[3.0, 0.2], [0.1, 0.2]], np.float32)
    path = ai.sequence_optimize(score, ai.build_transitions(2, 1.2, 0.1, None), None)
    assert max(path) < 2


def test_back_penalty_keeps_order_but_allows_jump_when_strong():
    score = np.array([[0, 3, 0], [2.0, 0, 2.1], [0, 0, 3]], np.float32)
    mild = ai.sequence_optimize(score, ai.build_transitions(3, 0.6, 0.0, None), None)
    strict = ai.sequence_optimize(score, ai.build_transitions(3, 5.0, 0.0, None), None)
    assert strict[1] == 2 and mild[0] == 1


def test_reuse_window_penalizes_non_adjacent_repeat():
    # câu 0 và 2 cùng thích cảnh 0; câu 1 thích cảnh 1. Có cảnh 2 gần bằng cho câu 2.
    score = np.array([[3.0, 0, 0], [0, 3.0, 0], [2.9, 0, 2.6]], np.float32)
    trans = ai.build_transitions(3, 0.0, 0.0, None)
    plain = ai.sequence_optimize(score, trans, None)
    div = ai.optimize_with_diversity(score, trans, None, None, window=3)
    assert plain[0] == plain[2] == 0
    assert div[0] != div[2]


def test_dup_scenes_detected():
    sim = np.array([[1, .97, 0.1], [.97, 1, .1], [.1, .1, 1]], np.float32)
    viol = ai.find_reuse_violations([0, 2, 1], 3, sim, 3)
    assert (0, 2) in viol


def test_top_candidates_skip_near_duplicates():
    sim = np.array([[1, .99, 0.1], [.99, 1, .1], [.1, .1, 1]], np.float32)
    assert ai.top_candidates(np.array([3.0, 2.9, 1.0]), sim, 2) == [0, 2]


# ---------- Phân đoạn ----------
def test_build_intervals_merge_split_and_reduce():
    raw = [(0, 0.4), (0.4, 3), (3, 30), (30, 31)]
    out, fb = ai.build_intervals(raw, 31, 100)
    assert not fb and out[0][0] == 0 and out[-1][1] == 31
    assert all(e - s <= ai.MAX_SCENE_SEC + 1e-6 for s, e in out)
    assert all(abs(out[i][1] - out[i + 1][0]) < 1e-6 for i in range(len(out) - 1))
    small, _ = ai.build_intervals(raw, 31, 3)
    assert len(small) == 3


def test_build_intervals_fallback_when_no_cut():
    out, fb = ai.build_intervals([], 40, 10)
    assert fb and len(out) >= 4


def test_merge_prefers_similar_neighbor():
    # cảnh ngắn (3-3.3) giống cảnh sau (sim cao tại 3.3) hơn cảnh trước (sim thấp tại 3)
    raw = [(0, 3), (3, 3.3), (3.3, 9)]
    sims = {3.0: 0.0, 3.3: 0.95}
    out, _ = ai.build_intervals(raw, 9, 100, boundary_sim=lambda t: sims.get(round(t, 2), 0.5))
    assert out == [(0, 3), (3, 9)] or [tuple(map(float, o)) for o in out] == [(0.0, 3.0), (3.0, 9.0)]


# ---------- Định vị thời điểm ----------
def test_localize_action_finds_peak_between_coarse_samples():
    times = np.linspace(8, 14, 16)
    q = np.array([1.0, 0, 0], np.float32)
    vecs = np.tile(np.array([0, 1.0, 0], np.float32), (16, 1))
    vecs[8] = [0.95, 0.3, 0]      # khoảnh khắc quyết định ở giữa
    vecs[8] /= np.linalg.norm(vecs[8])
    r = ai.localize_action(times, vecs, np.ones(16, bool), q, None)
    assert abs(r["focus_sec"] - times[8]) <= 0.4 + 1e-6
    assert r["action_start_sec"] <= r["focus_sec"] <= r["action_end_sec"]


def test_aggregate_ignores_invalid_frames():
    sims = np.array([[[0.9, 0.1, 0.1]]], np.float32)         # (1,1,3)
    valid = np.array([[False, True, True]])
    raw, fidx = ai.aggregate_frames(sims, valid)
    assert raw[0, 0] < 0.2 and fidx[0, 0] in (1, 2)


def test_thresholds_scale_with_scene_count():
    # ít cảnh: ngưỡng z cố định bất khả thi -> phải co lại, câu khớp hoàn hảo vẫn đạt "Khớp tốt"
    assert ai.eff_threshold(2.4, 4) < 2.4 <= ai.eff_threshold(2.4, 500)
    assert ai.classify(1.3, 0.95, 0.0, False, False, m=4)[1] == "ok"
    assert ai.classify(1.3, 0.5, 0.0, False, False, m=500)[1] == "low"        # z thấp + bằng chứng tuyệt đối trung bình
    assert ai.classify(1.3, 0.95, 0.0, False, False, m=500)[1] == "review"    # bằng chứng tuyệt đối mạnh: ít nhất đáng xem
    assert ai.classify(3.0, 0.1, 0.0, False, False, m=500)[1] == "low"        # z cao nhưng không có bằng chứng thô
    assert ai.classify(0.0, 0.05, 0.0, False, True, m=500)[1] == "abstract"


# ---------- Mệnh đề / phủ định ----------
def test_clauses_and_flags():
    assert len(ai.split_clauses("Nhân vật rút kiếm, né đòn rồi phản công đối thủ")) >= 2
    assert ai.split_clauses("Ngắn thôi") == []
    assert ai._FLASHBACK.search("Cô nhớ lại lời hứa")
    assert ai._NEGATION.search("Anh không hề sợ")


# ---------- Cache ----------
def test_cache_roundtrip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
    F = len(ai.FRAME_FRACTIONS)
    scenes = [{"scene_id": f"sc_{i}", "start_time": i, "end_time": i + 1, "frame_times": [0.0] * F,
               "motion_score": 1, "face_count": 0, "caption": "x", "caption_source": "heuristic", "vision_status": "clip"}
              for i in range(3)]
    an = ai.Analysis("k", scenes, np.random.rand(3, F, DIM).astype(np.float32), np.ones((3, F), bool), None, None,
                     "clip", 3.0, 10.0, True, False)
    p = tmp_path / "a.npz"
    ai.save_cache(p, "H", an)
    assert not list(tmp_path.glob("*.tmp.npz"))
    assert ai.load_cache(p, "H", True)["scenes"][1]["scene_id"] == "sc_1"
    assert ai.load_cache(p, "OTHER", True) is None                 # cấu hình khác -> bỏ cache
    p.write_bytes(b"corrupt")
    assert ai.load_cache(p, "H", True) is None                     # file hỏng -> không nổ


def test_config_changes_cache_hash():
    a = ai.analysis_config(240, True, False, 27)
    assert a != ai.analysis_config(240, True, True, 27)            # Ollama thật sự bật/tắt
    assert a != ai.analysis_config(240, True, False, 20)           # ngưỡng cảnh khác


# ---------- Tích hợp end-to-end (video tổng hợp + encoder giả) ----------
@pytest.fixture()
def client():
    return TestClient(ai.app)


def test_end_to_end_match_and_no_match_and_rematch(client, video):
    body = {"video_path": video, "max_scenes": 50, "refine": True, "order_mode": "free", "detect_threshold": 10,
            "sentences": ["Cảnh đỏ xuất hiện", "Rồi đến màu xanh dương", "Cuối cùng là màu vàng rực"]}
    r = client.post("/auto_match_all", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    res = data["results"]
    assert len(res) == 3 and data["calibrated"] is False
    assert res[0]["start_sec"] < 3 <= res[0]["end_sec"] + 0.5
    assert 5.5 < res[1]["start_sec"] + 0.1 and res[1]["start_sec"] < 9
    assert res[2]["start_sec"] >= 8.5
    assert res[0]["status"] == "ok" and res[1]["status"] == "ok"
    for r_ in res:
        assert r_["start_sec"] <= r_["focus_sec"] <= r_["end_sec"] + 0.5
        assert r_["status"] in ("ok", "review", "low", "no_match")
    # lần 2: dùng cache
    assert client.post("/auto_match_all", json=body).json()["from_cache"] is True

    # câu không liên quan hình ảnh -> bị từ chối
    nm = client.post("/auto_match_all", json={**body, "sentences": ["Một câu hoàn toàn trừu tượng về triết lý"]}).json()
    assert nm["results"][0]["status"] == "abstract"
    assert nm["results"][0]["candidates"] is not None
    off = client.post("/auto_match_all", json={**body, "allow_no_match": False,
                                               "sentences": ["Một câu hoàn toàn trừu tượng về triết lý"]}).json()
    assert off["results"][0]["status"] != "abstract"

    # rematch: loại cảnh đỏ cũ -> không trả lại đoạn đó
    first = res[0]
    rm = client.post("/rematch", json={"video_path": video, "max_scenes": 50, "detect_threshold": 10, "sentence": "Cảnh đỏ xuất hiện",
                                       "exclude": [[first["start_sec"], first["end_sec"]]], "top_k": 4}).json()
    assert rm["candidates"]
    for c in rm["candidates"]:
        assert not (c["start_sec"] >= first["start_sec"] - 0.01 and c["end_sec"] <= first["end_sec"] + 0.01)


def test_rematch_window_expands(client, video):
    r = client.post("/rematch", json={"video_path": video, "max_scenes": 50, "sentence": "Cảnh vàng", "detect_threshold": 10, "near_sec": 1.0,
                                      "window_sec": 1.0, "top_k": 3}).json()
    assert r["window_expanded"] is True and len(r["candidates"]) >= 1


def test_job_cancel_isolated(client, video):
    assert client.post("/cancel", json={}).json()["status"] == "not_running"
    assert client.post("/cancel", json={"job_id": "old_job"}).json()["status"] == "not_running"
    body = {"video_path": video, "sentences": ["Cảnh đỏ"], "job_id": "abc123", "detect_threshold": 10}
    assert client.post("/auto_match_all", json=body).status_code == 200
    assert client.get("/progress", params={"job_id": "abc123"}).json()["status"] == "done"


def test_busy_returns_409_and_bad_video_400(client, video):
    assert ai.JOB_LOCK.acquire(blocking=False)
    try:
        r = client.post("/auto_match_all", json={"video_path": video, "sentences": ["x"]})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "busy"
    finally:
        ai.JOB_LOCK.release()
    assert client.post("/auto_match_all", json={"video_path": "/nope.mp4", "sentences": ["x"]}).status_code == 400


def test_auth_and_tts_validation(client, monkeypatch):
    monkeypatch.setattr(ai, "API_KEY", "secret")
    assert client.get("/progress").status_code == 401
    assert client.get("/progress", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200
    monkeypatch.setattr(ai, "API_KEY", "")
    assert client.post("/tts", json={"text": "a", "voice": "evil"}).status_code == 400
    assert client.post("/tts", json={"text": "a", "rate": "+999%"}).status_code == 400


def test_fallback_mode_never_reports_ok(client, video, monkeypatch):
    monkeypatch.setattr(ai.MODELS, "state", "failed")
    monkeypatch.setattr(ai.MODELS, "failed_at", ai.time.time())
    monkeypatch.setattr(ai.MODELS, "error", "offline")

    class FakeText:
        def encode(self, t, **kw):
            return np.random.default_rng(0).random((len(t), 6)).astype(np.float32)
    monkeypatch.setattr(ai.MODELS, "text", FakeText())
    r = client.post("/auto_match_all", json={"video_path": video, "max_scenes": 50, "detect_threshold": 10, "allow_fallback": True,
                                                     "sentences": ["Cảnh đỏ", "Cảnh vàng"]})
    assert r.status_code == 200
    d = r.json()
    assert d["matcher"].endswith("fallback") and any("CLIP không khả dụng" in w for w in d["warnings"])
    assert all(x["status"] in ("review", "low") for x in d["results"])


# ====================== v3.2: các lỗi chọn cảnh nêu trong tài liệu ======================
def test_adjacent_angles_in_same_setting_are_not_penalized():
    """Nhiều góc máy gần giống nhau liên tiếp (cùng bối cảnh) phải được dùng liên tiếp, không bị đẩy sang cảnh khác."""
    m = 6
    sim = np.full((m, m), 0.99, np.float32)                       # mọi cảnh trong trận đấu giống hệt nhau
    np.fill_diagonal(sim, 1.0)
    score = np.zeros((3, m), np.float32)
    score[0, 1], score[1, 2], score[2, 3] = 3.0, 3.0, 3.0         # 3 câu liên tiếp -> sc1, sc2, sc3
    trans = ai.build_transitions(m, 1.2, 0.0, None)               # đường chạy mặc định: KHÔNG truyền dup_sim
    assert ai.sequence_optimize(score, trans, None) == [1, 2, 3]
    assert ai.REUSE_WINDOW == 0                                   # cửa sổ cấm lặp mặc định tắt
    assert trans[1, 2] < 0.01                                     # đi sang cảnh kề sau gần như miễn phí


def test_skipping_far_ahead_costs_nothing_real():
    """Review tóm tắt, bỏ qua 10 phút tình tiết phụ: nhảy xa về phía trước không bị phạt."""
    m = 400
    trans = ai.build_transitions(m, 1.2, 0.0, None)
    assert trans[10, 300] < 0.02 and trans[300, 10] > 1.2
    score = np.zeros((2, m), np.float32)
    score[0, 10], score[1, 300] = 2.0, 2.0
    assert ai.sequence_optimize(score, trans, None) == [10, 300]


def test_prior_is_only_a_gentle_hint():
    assert ai.ORDER_MODES["medium"][2] <= 0.15 and ai.ORDER_MODES["loose"][2] == 0.0
    assert all(v[1] == 0.0 for v in ai.ORDER_MODES.values())      # không phạt nhảy tiến


def test_abstract_sentence_is_flattened_not_noise_amplified():
    an = ai.Analysis("k", [{"scene_id": f"s{i}", "start_time": i, "end_time": i + 1, "frame_times": [0.0] * 5,
                            "motion_score": 0, "face_count": 0, "caption": "", "caption_source": "heuristic",
                            "vision_status": "clip"} for i in range(4)],
                     np.stack([np.tile(FakeImg().encode([__import__("PIL.Image", fromlist=["x"]).new("RGB", (8, 8), c)])[0], (5, 1))
                               for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]]).astype(np.float32),
                     np.ones((4, 5), bool), None, None, "clip", 4.0, 1.0, True, False)
    S = ai.score_matrix(["Một câu hoàn toàn trừu tượng về triết lý", "Cảnh đỏ"], an)
    assert S["abstract"].tolist() == [True, False]
    assert np.allclose(S["base"][0], 0.0)                         # không còn z-score phóng đại nhiễu
    assert S["base"][1].max() > 1.0


def test_abstract_sentence_follows_the_chain_not_global_top(client, video):
    """Câu trừu tượng nằm giữa hai câu rõ ràng phải rơi vào GIỮA (theo mạch), không phải một cảnh bất kỳ toàn phim."""
    body = {"video_path": video, "max_scenes": 50, "detect_threshold": 10, "order_mode": "loose",
            "sentences": ["Cảnh đỏ xuất hiện", "Bạn có tin một món đồ chơi nhỏ bé thay đổi cả thế giới?", "Rồi đến màu vàng rực"]}
    r = client.post("/auto_match_all", json=body).json()["results"]
    assert r[1]["status"] == "abstract"
    assert r[0]["start_sec"] <= r[1]["start_sec"] <= r[2]["start_sec"]      # giữ thứ tự thời gian
    assert r[1]["start_sec"] != r[0]["start_sec"] or r[0]["end_sec"] - r[0]["start_sec"] > 5   # không dùng lại đúng cảnh vừa rồi
    assert r[1]["candidates"]                                            # có cảnh lân cận để người dùng chọn


def test_clip_failure_returns_503_instead_of_guessing(client, video, monkeypatch):
    monkeypatch.setattr(ai.MODELS, "state", "failed")
    monkeypatch.setattr(ai.MODELS, "failed_at", ai.time.time())
    monkeypatch.setattr(ai.MODELS, "error", "offline")
    r = client.post("/auto_match_all", json={"video_path": video, "sentences": ["Cảnh đỏ"], "detect_threshold": 10})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "clip_unavailable"
    r2 = client.post("/rematch", json={"video_path": video, "sentence": "Cảnh đỏ", "detect_threshold": 10})
    assert r2.status_code == 503


def test_no_clause_mixing_by_default_and_bonus_restored():
    assert ai.USE_CLAUSES is False
    sc = [{"motion_score": 60, "face_count": 0}, {"motion_score": 5, "face_count": 1}]
    b = ai.visual_bonus_matrix(["Hai bên chiến đấu dữ dội", "Cô nhớ lại bí mật"], sc)
    assert abs(b[0, 0] - 0.45) < 1e-6 and b[0, 0] > b[0, 1]       # như v3.0: tối đa +0.45, không bị khóa sau gate
    assert abs(b[1, 1] - 0.2) < 1e-6


def test_detect_abstract_rows_adapts_to_the_script():
    rng = np.random.default_rng(0)
    truth_rows = np.arange(30) % 5 != 0                       # 24 câu có cảnh khớp, 6 câu trừu tượng
    raw = np.where(truth_rows[:, None], rng.normal(0.14, 0.018, (30, 240)), rng.normal(0.125, 0.015, (30, 240))).astype(np.float32)
    for i in np.where(truth_rows)[0]:
        raw[i, rng.integers(0, 240)] += 0.08
    flag = ai.detect_abstract_rows(raw)
    assert flag[~truth_rows].mean() >= 0.8 and flag[truth_rows].mean() <= 0.1
    assert ai.detect_abstract_rows(raw[:1], force=True).all()
    assert not ai.detect_abstract_rows(np.full((1, 50), 0.3, np.float32)).any()   # 1 câu rõ ràng không bị gắn nhầm
