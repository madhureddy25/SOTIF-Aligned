"""Blind review page: blinding, controls, and scoring."""

import json

import cv2
import numpy as np

from sotif_miner.eval.review import build_review, score_review, wilson


def _evidence(run, wid, positive, start_f, end_f, event_s=None, alert_s=None, fps=10.0):
    d = run / "evidence" / wid
    d.mkdir(parents=True)
    names = []
    for k in range(start_f, start_f + 6):
        n = f"video_only_f{k:05d}.jpg"
        cv2.imwrite(str(d / n), np.zeros((432, 768, 3), np.uint8))
        names.append(f"D:\\\\work\\\\runs\\\\evidence\\\\{wid}\\\\{n}")  # Windows paths, as written on the PC
    window = {"clip_id": wid, "start_frame": start_f, "end_frame": end_f, "fps": fps}
    (d / "evidence.json").write_text(json.dumps({"bundle": {"window": window}, "unmarked": names,
                                                 "positive": positive, "event_s": event_s, "alert_s": alert_s}))


def _run(tmp_path):
    run = tmp_path / "run"
    for i in range(10):                                    # normal clips: every window is a false alarm
        _evidence(run, f"nexar_train_01{i:03d}_w000", False, 100, 160)
    for i in range(6):                                     # near-miss clips: window over the event = control
        _evidence(run, f"nexar_train_00{i:03d}_w000", True, 200, 260, event_s=24.0, alert_s=23.0)
        _evidence(run, f"nexar_train_00{i:03d}_w001", True, 10, 70, event_s=24.0, alert_s=23.0)  # far from event
    return run


def test_page_is_blind_and_mixes_controls(tmp_path):
    run = _run(tmp_path)
    page, key, n_fa, n_ctrl = build_review(run, n_controls=4, seed=1)
    assert (n_fa, n_ctrl) == (10, 4)
    text = page.read_text()
    assert "nexar_train" not in text and "false_alarm" not in text and "control" not in text.lower().split("<script>")[1]
    assert "evidence/" not in text and "video_only" not in text   # frames are copied under anonymous names
    k = json.loads(key.read_text())
    assert len(k) == 14 and sum(v["kind"] == "control" for v in k.values()) == 4
    assert all("_w000" in v["window"] for v in k.values() if v["kind"] == "control")  # only event-overlapping
    items = json.loads(text.split("const ITEMS=")[1].split(", VERDICTS=")[0])
    for it in items:                                       # images resolve relative to the page
        assert all((run / p).exists() for p in it["frames"])


def test_score_review(tmp_path):
    run = _run(tmp_path)
    build_review(run, n_controls=4, seed=1)
    key = json.loads((run / "review_key.json").read_text())
    rows = ["item,verdict,causes,note"]
    for item, v in key.items():
        if v["kind"] == "control":
            rows.append(f"{item},close_call,lead_braking,")
        elif int(item) % 2:
            rows.append(f"{item},normal,adjacent_lane,")
        else:
            rows.append(f"{item},close_call,,")
    (tmp_path / "a.csv").write_text("\n".join(rows))
    s = score_review(tmp_path / "a.csv", run / "review_key.json")
    assert s["control"]["verdicts"]["close_call"]["share"] == 1.0
    fa = s["false_alarm"]
    assert fa["n"] == 10 and fa["verdicts"]["close_call"]["k"] + fa["verdicts"]["normal"]["k"] == 10
    assert fa["causes_of_true_errors"]["adjacent_lane"] == fa["verdicts"]["normal"]["k"]


def test_wilson():
    p, lo, hi = wilson(25, 100)
    assert abs(p - 0.25) < 1e-9 and abs(lo - 0.175) < 0.002 and abs(hi - 0.343) < 0.002
