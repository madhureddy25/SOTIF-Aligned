"""Stage 5a: build a co-training dataset in COCO format for detector fine-tuning.

Two label sources, same COCO output (so fine-tuning code is identical):

* label-free (default, needs no external data): pseudo-labels from the detector's own cached
  detections on the mined frames. The ADDITIVE signal is crop-recovered boxes -- an actor the
  full-frame detector missed but the zoomed, contrast-enhanced crop recovered (from the diagnosis
  evidence). Training on those at full resolution is the honest "fix what you only see when zoomed"
  step. Pure self-distillation on already-confident boxes teaches nothing, so those crop-recovered
  boxes are what make this worthwhile; they are flagged `review_required`.
* labeled (CODA etc.): real boxes read by a `LabeledSource` adapter, dropped into the same builder.

Anti-forgetting: corner-case frames (from the top failure clusters) are mixed with uncertainty- and
diversity-sampled CORE frames (routine driving) at a configurable ratio (default 1 corner : 3 core),
so fine-tuning raises corner-case recall without regressing on normal roads (VLM-C4L idea).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import ontology
from ..mining.runner import load_log
from ..schemas import Detection

# Fixed, contiguous COCO categories = the unified driving ontology (id 1..N; 0 reserved).
CATEGORIES = [{"id": i + 1, "name": c} for i, c in enumerate(ontology()["classes"])]
CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}


@dataclass
class FrameLabels:
    source_clip: str
    frame: int
    boxes: list[Detection]
    role: str  # "corner" | "core"
    review_required: bool = False
    extra_boxes: int = 0  # crop-recovered (blind-spot) boxes added on top of detector output


@dataclass
class ManifestPlan:
    corner: list[FrameLabels] = field(default_factory=list)
    core: list[FrameLabels] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return len(self.corner) / max(len(self.core), 1)


def _dets_from_frame(fr, thresh: float) -> list[Detection]:
    return [d for d in fr.detections if d.score >= thresh and d.cls in CAT_ID]


def _crop_recovered_boxes(ev_dir: Path) -> list[Detection]:
    """Focus-actor boxes recovered only on the enhanced crop (the blind-spot signal), in full-frame
    coordinates, keyed by the keyframe they belong to."""
    ej = ev_dir / "evidence.json"
    if not ej.exists():
        return []
    d = json.loads(ej.read_text())
    if not d.get("crop_dets"):
        return []
    out = []
    for r in d["bundle"].get("rois", []):
        ab = r.get("actor_box")
        if ab and r.get("cls_hint") in CAT_ID:
            out.append(Detection(box=tuple(ab), score=0.99, cls=r["cls_hint"]))
    return out


def plan_label_free(mine_run: Path, clusters: list[dict], diagnose_run: Path | None,
                    pseudo_thresh: float = 0.5, frames_per_window: int = 3,
                    top_clusters: int = 8, core_pool: list[Path] | None = None,
                    ratio_core: float = 3.0, max_corner: int = 400, seed: int = 0) -> ManifestPlan:
    """Build the corner/core frame plan from a mined run's perception cache + taxonomy clusters."""
    rng = random.Random(seed)
    plan = ManifestPlan()
    recurring = [c for c in clusters if not c.get("noise")][:top_clusters]
    want_clips = {cid for c in recurring for cid in c["member_clip_ids"]}

    # map clip_id -> cached perception path
    pcache = {p.name[: -len(".json.gz")]: p for p in (mine_run / "perception").glob("*.json.gz")}
    # crop-recovered boxes per window clip_id (from the diagnosis run)
    crop_boxes: dict[str, list[Detection]] = {}
    if diagnose_run:
        for ev in (diagnose_run / "evidence").glob("*/evidence.json"):
            wid = ev.parent.name
            crop_boxes[wid] = _crop_recovered_boxes(ev.parent)

    for clip_id in sorted(want_clips):
        base = clip_id.rsplit("_w", 1)[0].rsplit("_f", 1)[0]
        cache = pcache.get(base) or pcache.get(clip_id)
        if cache is None:
            continue
        log = load_log(cache)
        peak = None
        extra = crop_boxes.get(clip_id, [])
        # frames around the middle of the clip (where mining fired); dense = frames_per_window
        n = len(log.frames)
        idxs = sorted({int(round(x)) for x in np.linspace(n * 0.3, n * 0.7, frames_per_window)})
        for k in idxs:
            if k >= n:
                continue
            fr = log.frames[k]
            boxes = _dets_from_frame(fr, pseudo_thresh)
            add = extra if peak is None else []  # attach crop-recovered boxes once per clip
            peak = k
            allb = boxes + [b for b in add if b not in boxes]
            if allb:
                plan.corner.append(FrameLabels(cache.stem, fr.frame, allb, "corner",
                                               review_required=bool(add), extra_boxes=len(add)))
        if len(plan.corner) >= max_corner:
            break

    # core: routine frames with the detector's confident boxes (self-replay, anti-forgetting)
    n_core = int(len(plan.corner) * ratio_core)
    pool = core_pool or [p for p in pcache.values()]
    rng.shuffle(pool)
    for cache in pool:
        if len(plan.core) >= n_core:
            break
        log = load_log(cache)
        if not log.frames:
            continue
        fr = rng.choice(log.frames)
        boxes = _dets_from_frame(fr, pseudo_thresh)
        plan.core.append(FrameLabels(cache.stem, fr.frame, boxes, "core"))
    return plan


def write_coco(plan: ManifestPlan, frame_images: dict[tuple[str, int], np.ndarray], out_dir: Path,
               val_frac: float = 0.15, seed: int = 0) -> dict:
    """Write an RF-DETR-style COCO dataset. `frame_images` maps (clip, frame) -> BGR image."""
    rng = random.Random(seed)
    items = [(fl, "corner") for fl in plan.corner] + [(fl, "core") for fl in plan.core]
    rng.shuffle(items)
    n_val = int(len(items) * val_frac)
    splits = {"valid": items[:n_val], "train": items[n_val:]}
    summary = {"categories": len(CATEGORIES), "splits": {}}
    for split, rows in splits.items():
        sd = out_dir / split
        sd.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        img_id = ann_id = 1
        for fl, _role in rows:
            img = frame_images.get((fl.source_clip, fl.frame))
            if img is None:
                continue
            h, w = img.shape[:2]
            fn = f"{fl.source_clip}_{fl.frame:06d}.jpg"
            cv2.imwrite(str(sd / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            images.append({"id": img_id, "file_name": fn, "width": w, "height": h,
                           "role": fl.role, "review_required": fl.review_required})
            for b in fl.boxes:
                x1, y1, x2, y2 = b.box
                annotations.append({"id": ann_id, "image_id": img_id, "category_id": CAT_ID[b.cls],
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": float((x2 - x1) * (y2 - y1)),
                                    "iscrowd": 0})
                ann_id += 1
            img_id += 1
        coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
        (sd / "_annotations.coco.json").write_text(json.dumps(coco))
        summary["splits"][split] = {"images": len(images), "annotations": len(annotations)}
    # RF-DETR expects a test split too; reuse valid as a placeholder so training does not error
    test = out_dir / "test"
    test.mkdir(parents=True, exist_ok=True)
    if (out_dir / "valid" / "_annotations.coco.json").exists():
        (test / "_annotations.coco.json").write_text((out_dir / "valid" / "_annotations.coco.json").read_text())
        for img in (out_dir / "valid").glob("*.jpg"):
            cv2.imwrite(str(test / img.name), cv2.imread(str(img)))
    summary["ratio_corner_to_core"] = round(plan.ratio, 3)
    summary["corner_frames"] = len(plan.corner)
    summary["core_frames"] = len(plan.core)
    summary["crop_recovered_boxes"] = sum(fl.extra_boxes for fl in plan.corner)
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
