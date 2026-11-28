"""The autoresearch referee: experimental-cue hook, keep/discard rules, guardrails, logging."""

import json

import numpy as np
import pytest

from sotif_miner.eval.autoresearch import add_experimental_features, decide, record, within_clip_auc

BASE = {"auc_mean": 0.917, "auc_std": 0.001, "fw_per_min": 1.83, "within_clip_auc": 0.94, "target_met": True}


def test_baseline_is_kept_then_only_real_gains():
    assert decide(None, BASE)[0] == "KEEP"
    assert decide(BASE, {**BASE, "auc_mean": 0.918})[0] == "DISCARD"            # +0.001: noise
    assert decide(BASE, {**BASE, "auc_mean": 0.920, "auc_std": 0.004})[0] == "DISCARD"  # within 2 x std
    assert decide(BASE, {**BASE, "auc_mean": 0.925})[0] == "KEEP"


def test_guardrails_reject_shortcut_and_false_alarm_regressions():
    v, why = decide(BASE, {**BASE, "auc_mean": 0.930, "within_clip_auc": 0.90})
    assert v == "DISCARD" and "within-clip" in why                               # gain from clip-level shortcuts
    v, why = decide(BASE, {**BASE, "auc_mean": 0.930, "fw_per_min": 2.3})
    assert v == "DISCARD" and "false windows" in why
    assert decide(BASE, {**BASE, "auc_mean": 0.95, "target_met": False})[0] == "DISCARD"


def test_record_survives_and_tracks_best(tmp_path):
    state = tmp_path / "state"
    record(state, "abc1234", BASE, "KEEP", "baseline", "baseline")
    record(state, "def5678", {**BASE, "auc_mean": 0.90}, "DISCARD", "noise", "try\tsomething")
    rows = (state / "results.tsv").read_text().splitlines()
    assert len(rows) == 3 and rows[2].endswith("try something")
    assert json.loads((state / "best.json").read_text())["commit"] == "abc1234"


def test_within_clip_auc_separates_event_from_quiet():
    feats = [{"t": k * 0.5} for k in range(60)]
    p = np.array([0.9 if 18.5 <= f["t"] <= 21.5 else 0.1 for f in feats])  # band: alert-1.5 .. event+0.5
    recs = [{"clip_id": "a", "positive": True, "event_s": 21.0, "alert_s": 20.0, "features": feats}]
    assert within_clip_auc(recs, {"a": p}) == 1.0


def test_experimental_cues_flow_into_stored_features(tmp_path, monkeypatch, cfg):
    """A cue defined in experimental.py is computed from the perception cache and merged per step."""
    from sotif_miner.mining import experimental, scorer
    from sotif_miner.mining.runner import PerceptionLog, save_log
    from sotif_miner.schemas import FrameRecord

    frames = [FrameRecord(frame=i, t=i / 10, detections=[]) for i in range(40)]
    save_log(PerceptionLog("c", 10.0, 640, 360, frames), tmp_path / "c.json.gz")
    spec = {"z.n_frames": ("continuity", +1, 0.5)}
    monkeypatch.setattr(experimental, "EXP_FEATURES", spec)
    monkeypatch.setattr(scorer, "EXP_FEATURES", spec)
    monkeypatch.setattr(experimental, "extra_features", lambda fr, prev, fps, w, h: {"z.n_frames": len(fr)})
    recs = [{"clip_id": "c", "features": [{"t": k * 0.5} for k in range(8)]}]
    assert add_experimental_features(recs, tmp_path, cfg, workers=1) == 1
    assert [f["z.n_frames"] for f in recs[0]["features"]] == [5.0] * 8
    recs = [{"clip_id": "c", "features": [{"t": 0.0}]}]                           # step count mismatch
    with pytest.raises(ValueError):
        add_experimental_features(recs, tmp_path, cfg, workers=1)


def test_referee_refuses_held_out_clips(tmp_path):
    """If any held-out clip appears in the training run, the experiment is a CRASH, logged, never scored."""
    import shutil
    import sys
    from pathlib import Path

    from typer.testing import CliRunner

    from sotif_miner.cli import app

    sys.path.insert(0, str(Path(__file__).parent))
    import test_learned as tl

    root, run = tl._fake_nexar(tmp_path, tl._recs(4, 4))
    held = tmp_path / "heldout" / "clips"
    held.mkdir(parents=True)
    shutil.copy(next((run / "clips").glob("*.json")), held)
    r = CliRunner().invoke(app, ["autoresearch-eval", "--run", str(run), "--perception", str(run), "--root", str(root),
                                 "--heldout", str(tmp_path / "heldout"), "--state", str(tmp_path / "state"),
                                 "--seeds", "2", "--no-tests", "--desc", "leak"])
    assert r.exit_code == 1 and "VERDICT: CRASH" in r.output and "held-out" in r.output
    assert "\tcrash\t" in (tmp_path / "state" / "results.tsv").read_text()
    assert not (tmp_path / "state" / "best.json").exists()


def test_experimental_names_are_isolated():
    from sotif_miner.mining.learned import LEGACY_NAMES
    from sotif_miner.mining.scorer import FEATURES, LEARNED_ONLY

    assert all(n in LEARNED_ONLY for n in FEATURES if n.startswith("z."))
    assert not any(n.startswith("z.") for n in LEGACY_NAMES)
