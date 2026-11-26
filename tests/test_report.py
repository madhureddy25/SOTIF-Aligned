"""Report + bootstrap: HTML builds and is well-formed; CIs bracket the point estimate."""

import json

import numpy as np

from sotif_miner.eval.bootstrap import _ci


def test_ci_brackets_median():
    lo, hi = _ci(list(np.linspace(0, 1, 101)))
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def test_report_builds_html(tmp_path):
    from sotif_miner.report import build_report

    run = tmp_path / "run"
    run.mkdir()
    (run / "diagnosis_summary.json").write_text(json.dumps({
        "cards": 10, "attribution": {"AGENT_BEHAVIOR": 6, "DYNAMIC_OCCLUSION": 3, "INCONCLUSIVE": 1},
        "status": {"accepted": 9, "needs_review": 1}, "repaired": 2,
        "vlm": {"calls": 30, "failures": 0, "json_validity": 1.0, "name": "ollama:qwen3-vl:4b-instruct"}}))
    tax = tmp_path / "tax"
    tax.mkdir()
    (tax / "clusters.json").write_text(json.dumps([
        {"cluster_id": "M000", "name": "Night closing conflict with car", "size": 6, "severity": 0.55,
         "rank_score": 6.3, "iso_21448_area": "AREA_2_KNOWN_HAZARDOUS", "noise": False,
         "attribution_hist": {"AGENT_BEHAVIOR": 6}, "representative_clip_ids": ["a", "b", "c"]},
        {"cluster_id": "M001", "name": "Glare optical miss with other", "size": 1, "severity": 0.2,
         "rank_score": 0.7, "iso_21448_area": "AREA_3_UNKNOWN_HAZARDOUS", "noise": True,
         "attribution_hist": {"ALEATORY_OPTICAL": 1}, "representative_clip_ids": ["z"]}]))
    test_metrics = {"ap": {"value": 0.48, "ci95": [0.43, 0.53]}, "n_clips": 667, "n_positive": 334, "prevalence": 0.5}
    heldout = {"_meta": {"kind": "held_out"}, "recall": {"value": 0.88, "ci95": [0.80, 0.94]},
               "negative_windows_per_minute": {"value": 1.9, "ci95": [1.5, 2.3]},
               "compression": {"value": 0.80, "ci95": [0.77, 0.83]}}
    h = build_report(run, tax, {"train": "100 pos / 100 neg"}, test_metrics=test_metrics, heldout_metrics=heldout)
    assert h.startswith("<!doctype html>") and h.rstrip().endswith("</html>")
    assert h.count("<body") == 1 and h.count("</body>") == 1
    # anticipation AP is shown as an aside with its chance baseline, not as the headline
    assert "anticipation benchmark" in h and "chance" in h and "0.480" in h
    # the held-out mining number is the locked headline
    assert "the locked number" in h and "0.880" in h
    assert "Night closing conflict with car" in h and "Area 2" in h
    assert "Limitations" in h and "Nexar" in h
    out = tmp_path / "r.html"
    out.write_text(h, encoding="utf-8")
    assert out.stat().st_size > 2000


def test_report_flags_in_sample_vs_cross_validated(tmp_path):
    from sotif_miner.report import build_report

    run = tmp_path / "run"
    run.mkdir()
    (run / "diagnosis_summary.json").write_text(
        '{"cards":1,"attribution":{"AGENT_BEHAVIOR":1},"vlm":{}}')
    cv = {"_meta": {"kind": "cross_validated_out_of_fold"}, "recall": {"value": 0.82, "ci95": [0.74, 0.90]}}
    insample = {"recall": {"value": 1.0, "ci95": [1.0, 1.0]}}
    h_cv = build_report(run, None, train_metrics=cv)
    assert "cross-validated (out-of-fold)" in h_cv and "IN-SAMPLE" not in h_cv
    h_in = build_report(run, None, train_metrics=insample)
    assert "IN-SAMPLE" in h_in and "overstate" in h_in
