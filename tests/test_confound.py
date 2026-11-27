"""Confound check: a clip-level marker must vanish within clips; an event-timed cue must not."""

import numpy as np

from sotif_miner.eval.confound import confound_table, stratified_auc


def _rec(cid, positive, rng, marker, tracks, event=30.0):
    feats = []
    for k in range(80):
        t = k * 0.5
        in_band = positive and event - 2.5 <= t <= event + 0.5
        feats.append({"t": t, "frame": k * 5,
                      "clip_marker": marker + rng.normal(0, 0.1),           # constant per clip
                      "event_timer": (3.0 if in_band else 0.0) + rng.normal(0, 0.1),
                      "c.active_tracks": tracks + rng.normal(0, 0.1)})
    return {"clip_id": cid, "positive": positive, "event_s": event if positive else None,
            "alert_s": event - 1.0 if positive else None, "features": feats}


def _data():
    rng = np.random.default_rng(0)
    recs, meta = [], {}
    for i in range(30):
        pos = i % 2 == 0
        scene = ("day", "clear", "urban" if i % 4 < 2 else "highway")
        recs.append(_rec(f"c{i}", pos, rng, marker=1.0 if pos else 0.0, tracks=3 + (i % 6)))
        meta[f"c{i}"] = scene
    return recs, meta


def test_clip_marker_vanishes_within_clips_but_event_cue_does_not():
    rows, counts = confound_table(*_data())
    r = {d["feature"]: d for d in rows}
    assert r["clip_marker"]["pooled"] > 0.95 and abs(r["clip_marker"]["within_clip"] - 0.5) < 0.1
    assert r["event_timer"]["pooled"] > 0.95 and r["event_timer"]["within_clip"] > 0.95
    assert counts["within_clip_clips"] == 15


def test_marker_survives_scene_matching_when_unrelated_to_scene():
    """Scene matching only removes confounds carried by the labels; this marker is not one of them."""
    rows, counts = confound_table(*_data())
    r = {d["feature"]: d for d in rows}
    assert counts["scene_strata"] == 2 and r["clip_marker"]["scene_matched"] > 0.95


def test_traffic_matching_removes_a_pure_density_effect():
    rng = np.random.default_rng(1)
    recs = []
    for i in range(60):
        pos = i % 2 == 0
        tracks = float(rng.integers(1, 12))
        r = _rec(f"d{i}", pos, rng, marker=0.0, tracks=tracks)
        for f in r["features"]:          # the cue is just traffic density, in both classes alike
            f["density_cue"] = tracks + rng.normal(0, 0.1)
        recs.append(r)
    rows, _ = confound_table(recs, {})
    d = {x["feature"]: x for x in rows}["density_cue"]
    assert abs(d["traffic_matched"] - 0.5) < 0.12


def test_stratified_auc_skips_one_class_strata():
    a, n = stratified_auc([(np.array([1.0, 2.0]), np.array([])), (np.array([]), np.array([0.0]))])
    assert np.isnan(a) and n == 0
