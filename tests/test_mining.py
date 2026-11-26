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
