"""Interaction cues (signals.scene_features): lead vehicle, lateral intrusion, VRU proximity, signed ego
motion; plus compatibility of models trained before these features existed."""

import numpy as np

from sotif_miner.mining.signals import scene_features
from sotif_miner.schemas import FrameRecord, TrackState

W, H, FPS = 640, 360, 10.0


def _frames(boxes_fn, n=35, cls="car", tid=1, **ego):
    out = []
    for i in range(n):
        b = boxes_fn(i)
        tracks = [] if b is None else [TrackState(track_id=tid, frame=i, box=b, score=0.9, cls=cls)]
        out.append(FrameRecord(frame=i, t=i / FPS, detections=[], tracks=tracks, **ego))
    return out


def _scene(fr):
    return scene_features(fr[-5:], fr[:-5], FPS, W, H)


def _grow(w0, rate_fn):
    """Lead car centred ahead; width follows w0 * exp(integral of rate_fn)."""
    def box(i):
        t = i / FPS
        w = w0 * np.exp(rate_fn(t))
        return (320 - w / 2, 300 - 0.6 * w, 320 + w / 2, 300.0)
    return box


def test_lead_braking_vs_steady_following():
    steady = _scene(_frames(_grow(80, lambda t: 0.0)))[0]
    closing = _scene(_frames(_grow(80, lambda t: 0.15 * t)))[0]           # constant closing
    braking = _scene(_frames(_grow(80, lambda t: 0.25 * max(0.0, t - 2.0) ** 2)))[0]  # closing speeds up
    assert steady["l.present"] == 1.0 and abs(steady["l.inv_ttc"]) < 0.02
    assert closing["l.inv_ttc"] > 0.1 and abs(closing["l.closing_accel"]) < 0.05
    assert braking["l.closing_accel"] > 0.1 and braking["l.danger"] > steady["l.danger"]


def test_no_lead_when_nothing_ahead():
    f, _ = _scene(_frames(lambda i: (10.0, 250.0, 60.0, 300.0)))  # parked car at the left edge
    assert f["l.present"] == 0.0 and f["l.inv_ttc"] == 0.0


def test_cut_in_toward_ego_path():
    toward = _scene(_frames(lambda i: (2.5 * i, 240.0, 60.0 + 2.5 * i, 300.0)))  # left lane, drifting in
    away = _scene(_frames(lambda i: (120.0 - 3 * i, 240.0, 200.0 - 3 * i, 300.0)))
    assert toward[0]["x.cut_in"] > 0.3 and toward[0]["x.entry_rate"] > 0.2 and 1 in toward[1]
    assert away[0]["x.cut_in"] == 0.0 and away[0]["x.entry_rate"] == 0.0


def test_camera_pan_is_not_a_cut_in():
    """Camera yawing: every static object shifts by ego_dx per frame; that is not object motion."""
    pan = _scene(_frames(lambda i: (20.0 + 6 * i, 240.0, 100.0 + 6 * i, 300.0), ego_dx=6.0))
    assert abs(pan[0]["x.cut_in"]) < 0.05 and pan[0]["x.entry_rate"] == 0.0


def test_vru_near_only_counts_people_ahead():
    near = _scene(_frames(lambda i: (300.0, 150.0, 340.0, 330.0), cls="person"))[0]
    car = _scene(_frames(lambda i: (300.0, 150.0, 340.0, 330.0), cls="car"))[0]
    assert near["x.vru_near"] > 0.4 and car["x.vru_near"] == 0.0


def test_signed_ego_deceleration():
    fr = _frames(lambda i: None, ego_scale=1.02)
    for f in fr[-5:]:
        f.ego_scale = 1.0          # expansion stops: the car braked
    braking = _scene(fr)[0]
    fr = _frames(lambda i: None, ego_scale=1.0)
    for f in fr[-5:]:
        f.ego_scale = 1.02         # expansion starts: accelerating
    accel = _scene(fr)[0]
    assert braking["e.decel"] > 0.01 and accel["e.decel"] < -0.01


def test_rule_scorer_ignores_learned_only_features():
    from sotif_miner.config import load_config
    from sotif_miner.mining.scorer import FEATURES, LEARNED_ONLY, score_features

    cfg = load_config(None)
    rng = np.random.default_rng(0)
    base = [{"t": k * 0.5, "frame": k * 5, "rarity": 0.0, **{n: float(rng.uniform()) for n in FEATURES}}
            for k in range(60)]
    noisy = [{**f, **{n: float(rng.uniform(0, 50)) for n in LEARNED_ONLY}} for f in base]
    a = [s.combined for s in score_features(base, cfg)]
    b = [s.combined for s in score_features(noisy, cfg)]
    assert a == b


def test_old_models_still_predict_on_new_features():
    """A model pickled before the interaction cues (no `names`) uses the legacy feature list."""
    from sotif_miner.config import load_config
    from sotif_miner.mining.learned import LEGACY_NAMES, NAMES, Stage1Model, design_matrix, fit

    cfg = load_config(None)
    rng = np.random.default_rng(1)

    def feats(spike):
        return [{"t": k * 0.5, "frame": k * 5, "rarity": 0.0,
                 **{n: float(rng.normal(3.0 if spike and k > 40 and n == "c.ego_looming" else 0.3, 0.1))
                    for n in NAMES}} for k in range(80)]
    recs = [{"clip_id": f"c{i}", "positive": i % 2 == 0, "event_s": 21.0 if i % 2 == 0 else None,
             "alert_s": 20.0 if i % 2 == 0 else None, "features": feats(i % 2 == 0)} for i in range(8)]
    legacy = fit(recs, cfg, names=LEGACY_NAMES)
    legacy.names = None  # as unpickled from a model saved before `names` existed
    old = Stage1Model(legacy.model, legacy.offsets, False, 0.5, legacy.n_features)
    p = old.predict_steps(recs[0]["features"], cfg)
    assert len(p) == 80 and old.n_features == design_matrix(recs[0]["features"], cfg, old.offsets, LEGACY_NAMES).shape[1]
    full = fit(recs, cfg)
    assert full.names == NAMES and full.n_features > old.n_features
