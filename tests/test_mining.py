import pytest

from sotif_miner.io.video import ArraySource
from sotif_miner.mining.runner import run_perception
from sotif_miner.mining.scorer import mine_log
from sotif_miner.perception.detector import ScriptedDetector
from sotif_miner.perception.embed import TinyEmbedder
from sotif_miner.perception.tracking import CMCSort
from sotif_miner.synth.clips import render, scenarios

SC = scenarios()


def _mine(name, cfg):
    sc = SC[name]
    frames, dets = render(sc)
    src = ArraySource(frames, 10.0, name)
    t = cfg["tracker"]
    tracker = CMCSort(t["high_thresh"], t["low_thresh"], t["match_iou"], t["lost_buffer"], t["min_hits"], True)
    log = run_perception(src, ScriptedDetector(dets), tracker, TinyEmbedder(), cfg["mining"]["step_s"])
    return sc, mine_log(log, cfg)


@pytest.mark.parametrize("name", [n for n, s in SC.items() if s.event is not None])
def test_event_recalled(name, cfg):
    sc, (samples, windows) = _mine(name, cfg)
    ev0, ev1 = sc.event
    hits = [w for w in windows if w.start_frame / w.fps <= ev1 and w.end_frame / w.fps >= ev0]
    assert hits, f"{name}: no window over event {sc.event}; peak score {max(s.combined for s in samples):.2f}"
    for w in windows:
        length = (w.end_frame - w.start_frame) / w.fps
        assert 3.9 <= length <= 8.1


def test_negative_has_no_window(cfg):
    _, (samples, windows) = _mine("steady_negative", cfg)
    assert windows == [], [(w.dominant_signal, w.peak_score) for w in windows]


def test_dominant_signals(cfg):
    expect = {"tunnel_collapse": "optical_density", "looming_vehicle": "continuity",
              "occlusion_emergence": "continuity"}
    for name, sig in expect.items():
        _, (_, windows) = _mine(name, cfg)
        assert any(w.dominant_signal == sig for w in windows), (name, [w.dominant_signal for w in windows])


def test_dense_crowd_density_peaks(cfg):
    _, (_, windows) = _mine("dense_crowd", cfg)
    assert any(w.signal_peaks["optical_density"] >= 0.9 for w in windows)


def test_busy_but_normal_clip_has_no_window(cfg):
    """Real dashcams always show distant half-confident boxes and many cars. A clip that is busy
    from start to finish (constant uncertainty + density) must not trigger: the miner scores
    change relative to the clip's own baseline, not absolute levels (regression for the first
    Nexar run, where every clip fired at t=0)."""
    import numpy as np

    from sotif_miner.schemas import Detection

    rng = np.random.default_rng(1)
    frames, _ = render(SC["steady_negative"])
    dets = {}
    for i in range(len(frames)):
        d = [Detection(box=(20 + 25 * j, 150, 40 + 25 * j, 170), score=float(rng.uniform(0.3, 0.55)), cls="car")
             for j in range(22)]  # 22 distant, low-confidence cars, all the time
        dets[i] = d
    src = ArraySource(frames, 10.0, "busy")
    t = cfg["tracker"]
    log = run_perception(src, ScriptedDetector(dets),
                         CMCSort(t["high_thresh"], t["low_thresh"], t["match_iou"], t["lost_buffer"], t["min_hits"], True),
                         TinyEmbedder(), cfg["mining"]["step_s"])
    samples, windows = mine_log(log, cfg)
    assert windows == [], [(w.dominant_signal, w.onset_frame) for w in windows]
    assert all(s.combined == 0 for s in samples[:4]), "warm-up must not score"


def test_ego_corridor():
    from sotif_miner.mining.signals import in_ego_corridor

    assert in_ego_corridor((290, 250, 350, 330), 640, 360)       # car ahead, bottom-centre
    assert not in_ego_corridor((20, 250, 80, 330), 640, 360)     # car at the left edge
    assert not in_ego_corridor((310, 100, 330, 140), 640, 360)   # distant speck above the horizon band


def test_ego_looming_ignores_side_traffic(cfg):
    """A car growing fast at the frame edge (being overtaken) must not count as ego-path looming."""
    from sotif_miner.mining.signals import continuity_signal
    from sotif_miner.schemas import FrameRecord, TrackState

    def frames(box_fn):
        return [FrameRecord(frame=i, t=i / 10, detections=[],
                            tracks=[TrackState(track_id=1, frame=i, box=box_fn(i), score=0.9, cls="car")])
                for i in range(20)]
    side = frames(lambda i: (0.0, 200.0, 40.0 + 8 * i, 330.0))            # grows, but at the edge
    ahead = frames(lambda i: (300 - 4 * i, 230 - i, 340 + 4.0 * i, 300.0 + i))  # grows, straight ahead
    _, ds, _ = continuity_signal(side[10:], side[:10], 10.0, 640, 360)
    _, da, _ = continuity_signal(ahead[10:], ahead[:10], 10.0, 640, 360)
    assert ds["ego_looming"] == 0.0 and ds["looming_rate"] > 0.5
    assert da["ego_looming"] > 0.5 and da["ego_inv_ttc"] > 0.2


def test_ego_motion_jerk_detects_braking_pitch():
    from sotif_miner.mining.signals import ego_motion_features
    from sotif_miner.schemas import FrameRecord

    calm = [FrameRecord(frame=i, t=i / 10, detections=[], ego_dy=0.2) for i in range(10)]
    brake = [FrameRecord(frame=i, t=i / 10, detections=[], ego_dy=6.0) for i in range(10, 15)]
    e = ego_motion_features(brake, calm, 640, 360)
    assert e["pitch_jerk"] > 0.01 and e["yaw_jerk"] == 0.0


def test_perception_cache_roundtrip(tmp_path, cfg):
    from sotif_miner.mining.runner import load_log, save_log
    from sotif_miner.mining.scorer import compute_features

    sc = SC["occlusion_emergence"]
    frames, dets = render(sc)
    t = cfg["tracker"]
    log = run_perception(ArraySource(frames, 10.0, "x"), ScriptedDetector(dets),
                         CMCSort(t["high_thresh"], t["low_thresh"], t["match_iou"], t["lost_buffer"], t["min_hits"], True),
                         TinyEmbedder(), cfg["mining"]["step_s"])
    save_log(log, tmp_path / "x.json.gz")
    log2 = load_log(tmp_path / "x.json.gz")
    f1, _ = compute_features(log, cfg)
    f2, _ = compute_features(log2, cfg)
    assert len(f1) == len(f2)
    for a, b in zip(f1, f2, strict=True):
        for k in a:
            assert abs(a[k] - b[k]) < 0.05 + 0.02 * abs(a[k]), k


def _vanish_log(last_seen: int, n: int = 100, lost_buffer: int = 30):
    """A car mid-frame from frame 0 to `last_seen`, then gone. Like CMCSort, the tracker reports the
    termination only `lost_buffer + 1` updates later, when the box is long out of the scorer's history."""
    from sotif_miner.mining.runner import PerceptionLog
    from sotif_miner.schemas import FrameRecord, TrackState

    frames = []
    for i in range(n):
        tracks = [TrackState(track_id=7, frame=i, box=(300.0, 200.0, 360.0, 250.0), score=0.9, cls="car")] \
            if i <= last_seen else []
        term = [7] if i == last_seen + lost_buffer + 1 else []
        frames.append(FrameRecord(frame=i, t=i / 10, detections=[], tracks=tracks, terminated_ids=term))
    return PerceptionLog("v", 10.0, 640, 360, frames)


def test_mid_frame_vanish_counted_at_last_sighting(cfg):
    """Regression: the termination arrives ~3 s after the track was last seen, beyond the 2.5 s box
    history, so mid-frame terminations were never counted (c.mid_terminations was 0 on all of Nexar)."""
    from sotif_miner.mining.scorer import compute_features

    feats, impl = compute_features(_vanish_log(last_seen=40), cfg)
    hits = [f["t"] for f in feats if f["c.mid_terminations"] > 0]
    assert hits, "mid-frame vanish was never counted"
    assert all(3.9 <= t <= 4.1 for t in hits)  # attributed to the step where the car was last seen
    assert any(7 in s for s in impl)


def test_vanish_near_clip_end_still_counted(cfg):
    """A track that vanishes in the last seconds (before the tracker would report it) still counts,
    as long as the clip continues for at least 1 s after its last sighting."""
    from sotif_miner.mining.scorer import compute_features

    feats, _ = compute_features(_vanish_log(last_seen=85), cfg)
    assert any(f["c.mid_terminations"] > 0 for f in feats)
    feats, _ = compute_features(_vanish_log(last_seen=95), cfg)  # only 0.4 s left: not a vanish
    assert not any(f["c.mid_terminations"] > 0 for f in feats)


def _stretch_samples(peak_t: float = 12.0):
    """40 s clip, scored every 0.5 s: active from 10 s to 30 s, with the single highest score early."""
    from sotif_miner.schemas import SignalSample

    out = []
    for k in range(80):
        t = k * 0.5
        c = 1.0 if t == peak_t else (0.8 if 10.0 <= t <= 30.0 else 0.0)
        out.append(SignalSample(t=t, frame=k * 5, continuity=c, combined=c))
    return out


def test_long_active_stretch_is_tiled_not_clamped_to_its_peak(cfg):
    """Regression: a 20 s active stretch was cut to one 8 s window around its peak (12 s), so an event
    at 28 s inside the same stretch was missed. This is why recall *fell* at very low thresholds."""
    from sotif_miner.mining.scorer import slice_windows

    s = _stretch_samples()
    wins = slice_windows(s, [set() for _ in s], cfg, 10.0, 399, "x")
    spans = [(w.start_frame / 10.0, w.end_frame / 10.0) for w in wins]
    assert len(wins) >= 3
    assert all(e - b <= cfg["mining"]["max_window_s"] + 0.1 for b, e in spans)
    assert any(b <= 28.0 <= e for b, e in spans), spans   # the late part of the stretch is covered
    assert any(b <= 12.0 <= e for b, e in spans), spans   # and so is the peak


def test_long_stretch_split_can_be_disabled(cfg):
    """`split_long_stretches: false` reproduces the old single-window behaviour (for old numbers)."""
    import copy

    from sotif_miner.mining.scorer import slice_windows

    c = copy.deepcopy(cfg)
    c["mining"]["split_long_stretches"] = False
    s = _stretch_samples()
    wins = slice_windows(s, [set() for _ in s], c, 10.0, 399, "x")
    assert len(wins) == 1
