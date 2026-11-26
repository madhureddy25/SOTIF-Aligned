import cv2
import numpy as np

from sotif_miner.datasets.nexar import load_nexar
from sotif_miner.eval.nexar_eval import ClipResult, average_precision, evaluate_test, evaluate_train
from sotif_miner.schemas import CandidateWindow, SignalSample

HEADER = "file_name,time_of_event,time_of_alert,light_conditions,weather,scene,time_to_accident\n"


def _video(path, seconds, fps=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(int(seconds * fps)):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()


def _fake_root(tmp_path):
    for split in ("train", "test-public"):
        for lab in ("positive", "negative"):
            (tmp_path / split / lab).mkdir(parents=True)
    tr = tmp_path / "train"
    (tr / "positive" / "metadata.csv").write_text(HEADER + "00001.mp4,19.5,18.6,Normal,Rain,Urban,\n"
                                                  "00002.mp4,20.0,19.0,Night,Clear,Highway,\n")
    (tr / "negative" / "metadata.csv").write_text(HEADER + "01001.mp4,,,Normal,Clear,Urban,\n")
    for f in ("positive/00001.mp4", "positive/00002.mp4", "negative/01001.mp4"):
        _video(tr / f, 4)
    tp = tmp_path / "test-public"
    (tp / "positive" / "metadata.csv").write_text(HEADER + "00009.mp4,18.385,15.446,Normal,Rain,Highway,1.000\n")
    (tp / "negative" / "metadata.csv").write_text(HEADER + "01044.mp4,,,Normal,Cloudy,Urban,0.500\n"
                                                  "09999.mp4,,,Normal,Cloudy,Urban,0.500\n")  # missing video
    _video(tp / "positive" / "00009.mp4", 10)
    _video(tp / "negative" / "01044.mp4", 10)
    return tmp_path


def test_loader_and_test_timeline(tmp_path):
    root = _fake_root(tmp_path)
    train = load_nexar(root, "train")
    assert [c.path.name for c in train] == ["00001.mp4", "00002.mp4", "01001.mp4"]
    assert train[0].event_in_clip_s() == 19.5 and train[0].alert_in_clip_s() == 18.6
    assert train[2].event_in_clip_s() is None
    test = load_nexar(root, "test-public")
    assert len(test) == 2  # missing video skipped
    pos = next(c for c in test if c.positive)
    assert abs(pos.duration_s - 10.0) < 0.2
    assert abs(pos.event_in_clip_s() - (pos.duration_s + 1.0)) < 1e-9
    assert abs((pos.event_in_clip_s() - pos.alert_in_clip_s()) - (18.385 - 15.446)) < 1e-9


def _win(s, e, on, fps=10.0):
    return CandidateWindow(clip_id="x", source="x", start_frame=int(s * fps), onset_frame=int(on * fps),
                           peak_frame=int(on * fps), end_frame=int(e * fps), fps=fps, peak_score=0.9,
                           dominant_signal="continuity", signal_peaks={})


def test_train_metrics(tmp_path):
    root = _fake_root(tmp_path)
    clips = {c.path.name: c for c in load_nexar(root, "train")}
    res = [
        ClipResult(clips["00001.mp4"], [], [_win(15.0, 21.0, 18.0)], 40.0),  # early hit, lead 1.5 s
        ClipResult(clips["00002.mp4"], [], [_win(19.5, 24.0, 19.6)], 40.0),  # hit after alert
        ClipResult(clips["01001.mp4"], [], [_win(5.0, 10.0, 7.0)], 40.0),  # false alarm
    ]
    m = evaluate_train(res)
    assert m["recall"] == 1.0 and m["early_recall_before_alert"] == 0.5
    assert abs(m["median_lead_time_s"] - (1.5 + 0.4) / 2) < 1e-9
    assert m["negative_clip_false_alarm_rate"] == 1.0
    assert abs(m["compression"] - (1 - (6 + 4.5 + 5) / 120)) < 1e-9
    assert m["by_group"]["weather=Rain"]["recall"] == 1.0


def test_test_metrics_ap(tmp_path):
    root = _fake_root(tmp_path)
    test = load_nexar(root, "test-public")
    pos = next(c for c in test if c.positive)
    neg = next(c for c in test if not c.positive)
    s_hi = [SignalSample(t=9.0, frame=90, combined=0.9)]
    s_lo = [SignalSample(t=9.0, frame=90, combined=0.2)]
    m = evaluate_test([ClipResult(pos, s_hi, [], 10.0), ClipResult(neg, s_lo, [], 10.0)])
    assert m["ap"] == 1.0 and m["by_time_to_accident"]["1.0s"] == 1.0


def test_average_precision():
    assert average_precision(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])) == (1 + 2 / 3) / 2


def test_cli_nexar_end_to_end(tmp_path, monkeypatch):
    """Runs the real `nexar` command on a fake Nexar folder with a scripted detector."""
    from typer.testing import CliRunner

    import sotif_miner.perception.detector as det_mod
    from sotif_miner.cli import app

    root = _fake_root(tmp_path / "nexar")
    monkeypatch.setattr(det_mod, "build_detector", lambda cfg: det_mod.ScriptedDetector({}))
    out = tmp_path / "run"
    r = CliRunner().invoke(app, ["nexar", "--root", str(root), "--split", "train", "--out", str(out), "--stride", "1"])
    assert r.exit_code == 0, r.output
    assert len(list((out / "clips").glob("*.json"))) == 3
    assert (out / "metrics.json").exists() and '"recall"' in r.output
    # resumable: second run does no work and still scores
    r2 = CliRunner().invoke(app, ["nexar", "--root", str(root), "--out", str(out)])
    assert r2.exit_code == 0 and "windows (" not in r2.output


def test_cli_rescore_and_diagnose(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import sotif_miner.perception.detector as det_mod
    from sotif_miner.cli import app

    root = _fake_root(tmp_path / "nexar")
    monkeypatch.setattr(det_mod, "build_detector", lambda cfg: det_mod.ScriptedDetector({}))
    out = tmp_path / "run"
    assert CliRunner().invoke(app, ["nexar", "--root", str(root), "--out", str(out), "--stride", "1"]).exit_code == 0
    cfg = tmp_path / "strict.yaml"
    cfg.write_text("mining: {t_on: 0.99}\n")
    r = CliRunner().invoke(app, ["rescore", "--run", str(out), "--config", str(cfg), "--out", str(tmp_path / "r2")])
    assert r.exit_code == 0 and "Re-scored 3 clips" in r.output, r.output
    r = CliRunner().invoke(app, ["nexar-diagnose", "--run", str(out)])
    assert r.exit_code == 0 and "AUC" in r.output, r.output
    assert len(list((out / "perception").glob("*.json.gz"))) == 3
    r = CliRunner().invoke(app, ["refeature", "--run", str(out), "--out", str(tmp_path / "r3")])
    assert r.exit_code == 0 and "Re-featured 3 clips" in r.output, r.output


def test_auc():
    from sotif_miner.eval.nexar_eval import _auc

    assert _auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert _auc(np.array([1.0]), np.array([1.0])) == 0.5
