"""Learned Stage 1: labels, context offsets, grouped CV, and the train-stage1 / rescore --model CLI."""

import json

import cv2
import numpy as np

from sotif_miner.mining.learned import cross_validate, design_matrix, fit, offsets_for, step_labels
from sotif_miner.mining.scorer import FEATURES

HEADER = "file_name,time_of_event,time_of_alert,light_conditions,weather,scene,time_to_accident\n"


def _feats(n=80, spike_at=None, seed=0):
    """Synthetic per-step features: noise, plus a camera-shake + ego-looming burst at `spike_at` s."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        t = k * 0.5
        f = {"t": t, "frame": k * 5, "rarity": float(rng.uniform(0, 0.2))}
        for name in FEATURES:
            f[name] = float(abs(rng.normal(0.003 if name.startswith("e.") else 0.3, 0.001 if name.startswith("e.") else 0.1)))
        if spike_at is not None and spike_at - 1.0 <= t <= spike_at + 0.5:
            f["e.shake"] = 0.03
            f["c.ego_looming"] = 2.5
        out.append(f)
    return out


def _recs(n_pos=12, n_neg=12):
    recs = []
    for i in range(n_pos):
        ev = 20.0 + (i % 5)
        recs.append({"clip_id": f"nexar_train_{i:05d}", "positive": True, "event_s": ev, "alert_s": ev - 1.0,
                     "features": _feats(spike_at=ev - 0.5, seed=i), "fps": 10.0, "duration_s": 40.0,
                     "last_frame": 400, "implicated": [[] for _ in range(80)]})
    for i in range(n_neg):
        recs.append({"clip_id": f"nexar_train_{1000 + i:05d}", "positive": False, "event_s": None, "alert_s": None,
                     "features": _feats(seed=100 + i), "fps": 10.0, "duration_s": 40.0, "last_frame": 400,
                     "implicated": [[] for _ in range(80)]})
    return recs


def test_labels_bands():
    f = _feats()
    y = step_labels(f, True, 20.0, 19.0)
    t = np.array([x["t"] for x in f])
    assert (y[(t >= 17.5) & (t <= 20.5)] == 1).all()
    assert (y[(t > 15.0) & (t < 17.5)] == -1).all() and (y[t < 2.0] == -1).all()
    assert (y[t > 23.0] == 0).all()
    assert (step_labels(f, False, None, None)[t >= 2.0] == 0).all()


def test_causal_context_never_looks_ahead(cfg):
    f = _feats(spike_at=30.0)
    off = offsets_for(causal=True)
    X1 = design_matrix(f, cfg, off)
    f2 = [dict(x) for x in f]
    for x in f2:
        if x["t"] > 25.0:
            x["e.shake"] = 9.0  # change only the future
    X2 = design_matrix(f2, cfg, off)
    t = np.array([x["t"] for x in f])
    past = t <= 25.0
    assert np.allclose(np.nan_to_num(X1[past]), np.nan_to_num(X2[past]))


def test_cross_validated_model_learns_signal(cfg):
    recs = _recs()
    oof = cross_validate(recs, cfg, causal=False, folds=4)
    assert set(oof) == {r["clip_id"] for r in recs}
    pos_band = np.concatenate([oof[r["clip_id"]][step_labels(r["features"], True, r["event_s"], r["alert_s"]) == 1]
                               for r in recs if r["positive"]])
    neg = np.concatenate([oof[r["clip_id"]][20:] for r in recs if not r["positive"]])
    assert pos_band.mean() > 0.6 > neg.mean()
    m = fit(recs, cfg, causal=True, threshold=0.6)
    assert m.offsets == [-4, -2, 0] and m.predict_steps(recs[0]["features"], cfg).shape == (80,)


def _fake_nexar(tmp_path, recs):
    """A Nexar root (tiny videos + metadata.csv) and a run folder holding `recs` as mined clips."""
    root = tmp_path / "nexar"
    for lab in ("positive", "negative"):
        d = root / "train" / lab
        d.mkdir(parents=True)
        rows = []
        for r in recs:
            if r["positive"] != (lab == "positive"):
                continue
            fn = r["clip_id"].split("_")[-1] + ".mp4"
            vw = cv2.VideoWriter(str(d / fn), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            vw.write(np.zeros((24, 32, 3), np.uint8))
            vw.release()
            rows.append(f"{fn},{r['event_s'] or ''},{r['alert_s'] or ''},Normal,Clear,Urban,\n")
        (d / "metadata.csv").write_text(HEADER + "".join(rows))
    run = tmp_path / "run"
    (run / "clips").mkdir(parents=True)
    for r in recs:
        (run / "clips" / f"{r['clip_id']}.json").write_text(json.dumps({**r, "windows": [], "samples": []}))
    return root, run


def test_cli_drop_markers_zscores_only(tmp_path):
    """--drop markers --no-raw trains without the clip markers or absolute levels, saves to its own files,
    and the saved model re-applies with the same design (no shape mismatch)."""
    import joblib
    from typer.testing import CliRunner

    from sotif_miner.cli import app
    from sotif_miner.mining.learned import CLIP_MARKERS

    root, run = _fake_nexar(tmp_path, _recs(8, 8))
    res = CliRunner().invoke(app, ["train-stage1", "--run", str(run), "--root", str(root), "--folds", "4",
                                   "--drop", "markers", "--no-raw"])
    assert res.exit_code == 0, res.output
    assert "dropped: markers" in res.output and "z-scores only" in res.output
    model = joblib.load(run / "stage1_model_drop_zonly.joblib")
    assert model.raw is False and not set(CLIP_MARKERS) & set(model.names)
    meta = json.loads((run / "stage1_cv_metrics_drop_zonly.json").read_text())["_meta"]
    assert meta["raw_levels"] is False and meta["dropped"] == "markers"
    assert not (run / "stage1_model.joblib").exists()  # did not overwrite the main model
    res = CliRunner().invoke(app, ["rescore", "--run", str(run), "--model", str(run / "stage1_model_drop_zonly.joblib"),
                                   "--out", str(tmp_path / "applied")])
    assert res.exit_code == 0 and "Re-scored 16 clips" in res.output, res.output


def test_select_names():
    import pytest

    from sotif_miner.mining.learned import CLIP_MARKERS, NAMES, select_names

    assert set(select_names("markers")) == set(NAMES) - set(CLIP_MARKERS)
    assert not any(n.startswith("x.") for n in select_names("x.*"))
    assert "e.shake" not in select_names("e.shake, markers")
    with pytest.raises(ValueError):
        select_names("no.such_feature")


def test_cli_confound(tmp_path):
    from typer.testing import CliRunner

    from sotif_miner.cli import app

    root, run = _fake_nexar(tmp_path, _recs(8, 8))
    res = CliRunner().invoke(app, ["confound", "--run", str(run), "--root", str(root),
                                   "--out-json", str(tmp_path / "c.json")])
    assert res.exit_code == 0, res.output
    rows = {d["feature"]: d for d in json.loads((tmp_path / "c.json").read_text())["rows"]}
    assert rows["c.ego_looming"]["within_clip"] > 0.7      # the synthetic spike is event-timed (partial overlap)
    assert "1 light/weather/scene groups" in res.output


def test_cli_train_and_apply(tmp_path, cfg):
    from typer.testing import CliRunner

    from sotif_miner.cli import app

    recs = _recs(8, 8)
    root = tmp_path / "nexar"
    for lab in ("positive", "negative"):
        d = root / "train" / lab
        d.mkdir(parents=True)
        rows = []
        for r in recs:
            if r["positive"] != (lab == "positive"):
                continue
            fn = r["clip_id"].split("_")[-1] + ".mp4"
            vw = cv2.VideoWriter(str(d / fn), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            vw.write(np.zeros((24, 32, 3), np.uint8))
            vw.release()
            rows.append(f"{fn},{r['event_s'] or ''},{r['alert_s'] or ''},Normal,Clear,Urban,\n")
        (d / "metadata.csv").write_text(HEADER + "".join(rows))
    run = tmp_path / "run"
    (run / "clips").mkdir(parents=True)
    for r in recs:
        (run / "clips" / f"{r['clip_id']}.json").write_text(json.dumps({**r, "windows": [], "samples": []}))
    res = CliRunner().invoke(app, ["train-stage1", "--run", str(run), "--root", str(root), "--folds", "4"])
    assert res.exit_code == 0, res.output
    assert "Step-level AUC" in res.output and (run / "stage1_model.joblib").exists()
    res = CliRunner().invoke(app, ["rescore", "--run", str(run), "--model", str(run / "stage1_model.joblib"),
                                   "--out", str(tmp_path / "applied")])
    assert res.exit_code == 0 and "Re-scored 16 clips" in res.output, res.output
