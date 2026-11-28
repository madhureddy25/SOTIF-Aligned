"""Blind video review: three arms, blinding, clip cutting, the VLM judge, scoring and agreement."""

import csv
import json

import cv2
import numpy as np
import pytest

from sotif_miner.eval.review import (
    audit,
    audit_decision,
    build_review,
    cohen_kappa,
    collect_items,
    corrected_prevalence,
    fisher_greater,
    judge_review,
    score_review,
    wilson,
)

pytest.importorskip("imageio_ffmpeg")


def _video(path, seconds=20, fps=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for k in range(seconds * fps):
        img = np.full((240, 320, 3), 40, np.uint8)
        cv2.circle(img, (160, 120), 10 + k % 50, (200, 200, 200), -1)
        vw.write(img)
    vw.release()


def _run(tmp_path, n_neg=6, n_pos=6):
    run, vids = tmp_path / "run", tmp_path / "videos"
    (run / "clips").mkdir(parents=True)
    vids.mkdir()
    for i in range(n_neg):
        cid = f"nexar_train_01{i:03d}"
        _video(vids / f"{cid}.mp4")
        win = {"start_frame": 30, "end_frame": 80, "fps": 10.0}           # 3.0-8.0 s flagged
        rec = {"clip_id": cid, "source": str(vids / f"{cid}.mp4"), "duration_s": 20.0, "positive": False,
               "windows": [win], "event_s": None, "alert_s": None}
        (run / "clips" / f"{cid}.json").write_text(json.dumps(rec))
    for i in range(n_pos):
        cid = f"nexar_train_00{i:03d}"
        _video(vids / f"{cid}.mp4")
        rec = {"clip_id": cid, "source": str(vids / f"{cid}.mp4"), "duration_s": 20.0, "positive": True,
               "windows": [], "event_s": 9.0, "alert_s": 8.0}
        (run / "clips" / f"{cid}.json").write_text(json.dumps(rec))
    return run


def test_three_arms_and_unflagged_windows_avoid_model_windows(tmp_path):
    items = collect_items(_run(tmp_path), n_near_miss=4, n_unflagged=5, seed=0)
    kinds = [it["kind"] for it in items]
    assert kinds.count("flagged_normal") == 6 and kinds.count("unflagged_normal") == 5 and kinds.count("near_miss") == 4
    for it in items:
        if it["kind"] == "unflagged_normal":     # >= 1 s away from the 3-8 s model window, inside the clip
            assert it["start_s"] >= 9.0 and it["end_s"] <= 20.0
        if it["kind"] == "near_miss":            # from the labels: alert - 3 s .. event + 1.5 s
            assert it["start_s"] == 5.0 and it["end_s"] == 10.5


def test_page_is_blind_and_clips_play(tmp_path):
    run = _run(tmp_path)
    page, key, counts = build_review(run, n_near_miss=4, n_unflagged=4, seed=1)
    text = page.read_text()
    for leak in ("nexar_train", "flagged", "unflagged", "near_miss", "positive"):
        assert leak not in text
    k = json.loads(key.read_text())
    assert len(k) == sum(counts.values()) == 14
    for item in k:
        cap = cv2.VideoCapture(str(run / "review_clips" / f"{int(item):03d}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n > 0


def test_judge_and_score_with_agreement(tmp_path):
    from sotif_miner.vlm.scripted import ScriptedBackend

    run = _run(tmp_path)
    build_review(run, n_near_miss=4, n_unflagged=4, seed=1)
    backend = ScriptedBackend(lambda prompt, schema, n: json.dumps(
        {"verdict": "normal", "other_road_user": "", "reason": "steady traffic", "confidence": 0.7}))
    judge_csv = judge_review(run, backend, n_frames=6)
    assert all("Reply with JSON" in p for p in backend.prompts)        # telemetry-free prompt
    assert not any("TTC" in p or "looming" in p for p in backend.prompts)
    judge_review(run, backend, n_frames=6)                              # resumable: no duplicates
    assert len(list(csv.DictReader(judge_csv.open()))) == 14

    key = json.loads((run / "review_key.json").read_text())
    rows = [["item", "verdict", "causes", "note", "reviewer"]]
    for item, v in key.items():
        verdict = {"near_miss": "close_call", "flagged_normal": "close_call" if int(item) % 2 else "normal",
                   "unflagged_normal": "normal"}[v["kind"]]
        rows.append([item, verdict, "adjacent_lane" if verdict == "normal" else "", "", "alice"])
    with (tmp_path / "alice.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    s = score_review([tmp_path / "alice.csv", judge_csv], run / "review_key.json")
    alice = s["raters"]["alice"]
    assert alice["near_miss_recognised"]["share"] == 1.0
    assert alice["flagged_vs_unflagged_close_call"]["unflagged"] == 0.0
    assert alice["arms"]["flagged_normal"]["causes_of_non_conflicts"]["adjacent_lane"] == \
        alice["arms"]["flagged_normal"]["verdicts"]["normal"]["k"]
    g = s["agreement"]["alice vs vlm:scripted"]
    assert g["n"] == 14 and g["kappa_close_call"] <= 0.0                # judge says "normal" for everything


def test_statistics():
    p, lo, hi = wilson(25, 100)
    assert abs(p - 0.25) < 1e-9 and abs(lo - 0.175) < 0.002 and abs(hi - 0.343) < 0.002
    assert abs(fisher_greater(8, 10, 2, 10) - 0.01151) < 1e-4           # hypergeometric tail
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert abs(cohen_kappa(list("aabb"), list("abab"))) < 1e-9


def test_audit_end_to_end_flags_an_unreliable_judge(tmp_path):
    """A judge that calls everything a close call 'recognises' every near-miss but also every normal clip:
    the audit must refuse to use its verdicts instead of reporting 100 % unlabelled close calls."""
    from sotif_miner.vlm.scripted import ScriptedBackend

    run = _run(tmp_path)
    backend = ScriptedBackend(lambda prompt, schema, n: json.dumps(
        {"verdict": "close_call", "reason": "car ahead", "causes": ["adjacent_lane"], "confidence": 0.9}))
    r = audit(run, backend, n_near_miss=4, n_unflagged=4, n_frames=4)
    assert r["judge_sensitivity"]["share"] == 1.0 and r["judge_false_close_call_rate"]["share"] == 1.0
    assert r["judge_reliable"] is False and "NOT reliable" in r["decision"]
    assert r["flagged_close_call_corrected"] is None
    assert (run / "false_alarm_audit.json").exists()
    rows = list(csv.DictReader((run / "review_judge.csv").open()))
    assert rows and all(row["causes"] == "adjacent_lane" for row in rows)
    other = ScriptedBackend(lambda prompt, schema, n: json.dumps(
        {"verdict": "normal", "reason": "x", "causes": [], "confidence": 0.5}))
    other.name = "bigger"
    audit(run, other, n_near_miss=4, n_unflagged=4, n_frames=4)        # new judge: re-judged, clips reused
    rows = list(csv.DictReader((run / "review_judge.csv").open()))
    assert {row["reviewer"] for row in rows} == {"vlm:bigger"} and len(rows) == 14


def _rater(sens_k, n_near, fp_k, n_unfl, fl_k, n_fl, lane_k):
    def arm(k, n, extra=None):
        v = {x: {"k": 0, "share": 0.0, "ci95": [0.0, 0.0]} for x in ("close_call", "hard_brake", "normal", "cant_tell")}
        p, lo, hi = wilson(k, n)
        v["close_call"] = {"k": k, "share": p, "ci95": [lo, hi]}
        v["normal"] = {"k": n - k, "share": 1 - p, "ci95": [0.0, 1.0]}
        return {"n": n, "verdicts": v, "causes_of_non_conflicts": extra or {}}
    return {"arms": {"near_miss": arm(sens_k, n_near), "unflagged_normal": arm(fp_k, n_unfl),
                     "flagged_normal": arm(fl_k, n_fl, {"adjacent_lane": lane_k})},
            "flagged_vs_unflagged_close_call": {"difference": 0.0, "fisher_p_one_sided": 1.0}}


def test_audit_decision_reliable_paths():
    lane = audit_decision(_rater(45, 50, 3, 50, 40, 160, 60))          # 60 of 120 non-conflicts: adjacent lane
    assert lane["judge_reliable"] and "build lane" in lane["decision"]
    assert abs(lane["flagged_close_call_corrected"] - corrected_prevalence(0.25, 0.9, 0.06)) < 1e-9
    no_lane = audit_decision(_rater(45, 50, 3, 50, 40, 160, 10))
    assert no_lane["judge_reliable"] and "skip lane" in no_lane["decision"]
    weak = audit_decision(_rater(20, 50, 3, 50, 40, 160, 10))         # recognises only 40 % of near-misses
    assert not weak["judge_reliable"]


def test_corrected_prevalence():
    assert abs(corrected_prevalence(0.25, 0.9, 0.06) - (0.25 - 0.06) / 0.84) < 1e-12
    assert corrected_prevalence(0.05, 0.9, 0.1) == 0.0
