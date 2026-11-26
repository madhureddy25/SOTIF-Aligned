"""Stage 5: COCO manifest export (pycocotools-valid), ratio, and the closed-loop comparison."""


import numpy as np
import pytest

from sotif_miner.eval.closed_loop import compare
from sotif_miner.eval.nexar_eval import ClipResult
from sotif_miner.remediation.manifest import CAT_ID, FrameLabels, ManifestPlan, write_coco
from sotif_miner.schemas import CandidateWindow, Detection


def _fl(clip, frame, role, boxes):
    return FrameLabels(clip, frame, [Detection(box=b, score=0.9, cls="car") for b in boxes], role)


def test_write_coco_is_valid_and_keeps_ratio(tmp_path):
    plan = ManifestPlan(
        corner=[_fl("c", i, "corner", [(10, 10, 50, 60)]) for i in range(4)],
        core=[_fl("c", 100 + i, "core", [(20, 20, 40, 40)]) for i in range(12)])
    imgs = {(fl.source_clip, fl.frame): np.zeros((80, 120, 3), np.uint8) for fl in plan.corner + plan.core}
    summary = write_coco(plan, imgs, tmp_path, val_frac=0.25, seed=0)
    assert abs(summary["ratio_corner_to_core"] - 4 / 12) < 2e-3
    # loads in pycocotools without error, with the expected category set
    from pycocotools.coco import COCO

    for split in ("train", "valid", "test"):
        ann = tmp_path / split / "_annotations.coco.json"
        assert ann.exists()
        coco = COCO(str(ann))
        assert len(coco.getCatIds()) == len(CAT_ID)
        for img_id in coco.getImgIds():
            for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                assert a["bbox"][2] > 0 and a["bbox"][3] > 0  # positive w/h


def test_coco_split_counts(tmp_path):
    plan = ManifestPlan(corner=[_fl("c", i, "corner", [(1, 1, 9, 9)]) for i in range(8)], core=[])
    summary = write_coco(plan, {(f.source_clip, f.frame): np.zeros((20, 20, 3), np.uint8) for f in plan.corner},
                         tmp_path, val_frac=0.25, seed=1)
    assert summary["splits"]["train"]["images"] + summary["splits"]["valid"]["images"] == 8


# --- closed-loop comparison ---------------------------------------------------

class _Clip:
    def __init__(self, cid, positive, dur=40.0, event=20.0, alert=19.0):
        self.clip_id, self.positive, self._d = cid, positive, dur
        self._ev, self._al = (event, alert) if positive else (None, None)
        self.weather = self.light = self.scene = "x"

    def event_in_clip_s(self):
        return self._ev

    def alert_in_clip_s(self):
        return self._al


def _win(s, e, on, fps=10.0):
    return CandidateWindow(clip_id="x", source="x", start_frame=int(s * fps), onset_frame=int(on * fps),
                           peak_frame=int(on * fps), end_frame=int(e * fps), fps=fps, peak_score=0.9,
                           dominant_signal="continuity", signal_peaks={})


def _results(hit_frac, fa_per_neg, n=20):
    """Build before/after ClipResults: positives get a covering window with prob hit_frac; negatives
    get `fa_per_neg` windows."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        c = _Clip(f"p{i}", True)
        wins = [_win(17.0, 23.0, 18.0)] if rng.random() < hit_frac else []
        out.append(ClipResult(c, [], wins, 40.0))
    for i in range(n):
        c = _Clip(f"n{i}", False)
        wins = [_win(5.0, 10.0, 6.0) for _ in range(fa_per_neg)]
        out.append(ClipResult(c, [], wins, 40.0))
    return out


def test_compare_reports_deltas_and_verdict():
    before = _results(hit_frac=0.7, fa_per_neg=2)
    after = _results(hit_frac=0.9, fa_per_neg=1)  # better recall, fewer false alarms
    res = compare(before, after, n_boot=300)
    assert res["recall"]["delta"] > 0
    assert res["negative_windows_per_minute"]["delta"] < 0
    assert res["_verdict"]["anti_forgetting_ok"] is True
    assert "delta_ci95" in res["recall"]


def test_compare_flags_forgetting():
    before = _results(hit_frac=0.8, fa_per_neg=1)
    after = _results(hit_frac=0.8, fa_per_neg=4)  # same recall, many more false alarms
    res = compare(before, after, n_boot=300, epsilon_fa=0.5)
    assert res["_verdict"]["anti_forgetting_ok"] is False


def test_compare_needs_overlap():
    with pytest.raises(ValueError, match="overlapping"):
        compare(_results(0.8, 1), [], n_boot=10)
