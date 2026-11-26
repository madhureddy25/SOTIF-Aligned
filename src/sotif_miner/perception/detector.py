"""Detector interface + RF-DETR (Apache-2.0) implementation + scripted test detector."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from ..config import map_class
from ..schemas import Detection

# COCO-91 category id -> name as used by RF-DETR's COCO-pretrained checkpoints.
COCO_ID_TO_NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 7: "train", 8: "truck",
    10: "traffic light", 13: "stop sign", 16: "bird", 17: "cat", 18: "dog", 19: "horse",
    20: "sheep", 21: "cow", 22: "elephant",
}


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray, frame_index: int | None = None) -> list[Detection]: ...


class RFDETRDetector:
    """Wraps the `rfdetr` package. Imported lazily so CPU tests never load torch.

    `class_names` overrides the COCO mapping for fine-tuned checkpoints (index -> unified class).
    """

    SIZES = {"nano": "Nano", "small": "Small", "medium": "Medium", "large": "Large"}

    def __init__(self, size: str = "medium", threshold: float = 0.25, weights: str | None = None,
                 class_names: list[str] | None = None, segmentation: bool = False):
        import rfdetr  # noqa: PLC0415

        cls_name = ("RFDETRSeg" if segmentation else "RFDETR") + self.SIZES[size]
        model_cls = getattr(rfdetr, cls_name)
        self.model = model_cls(pretrain_weights=weights) if weights else model_cls()
        self.threshold = threshold
        self.class_names = class_names
        self.name = f"rfdetr-{size}{'-seg' if segmentation else ''}"
        try:  # rfdetr's own COCO-91 id -> name table, when available
            from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: PLC0415

            self.coco_names = dict(COCO_CLASSES)
        except ImportError:
            self.coco_names = COCO_ID_TO_NAME
        if hasattr(self.model, "optimize_for_inference"):
            try:
                self.model.optimize_for_inference()
            except Exception:  # noqa: BLE001 - optional speedup, not available on every build
                pass

    def detect(self, image: np.ndarray, frame_index: int | None = None) -> list[Detection]:
        rgb = image[:, :, ::-1].copy()
        dets = self.model.predict(rgb, threshold=self.threshold)
        out = []
        for box, score, cid in zip(dets.xyxy, dets.confidence, dets.class_id, strict=False):
            if self.class_names is not None:
                cls = self.class_names[int(cid)] if int(cid) < len(self.class_names) else "other"
            else:
                cls = map_class("coco", self.coco_names.get(int(cid), "other"))
            if cls == "other" and self.class_names is None:
                continue
            out.append(Detection(box=tuple(float(v) for v in box), score=float(score), cls=cls))
        return out


class ScriptedDetector:
    """Returns pre-computed detections per frame index (synthetic clips, replay of cached runs).

    `on_crop` lets tests define what the detector "sees" on an enhanced ROI crop.
    """

    name = "scripted"

    def __init__(self, per_frame: dict[int, list[Detection]],
                 on_crop: Callable[[np.ndarray], list[Detection]] | None = None):
        self.per_frame = per_frame
        self.on_crop = on_crop

    def detect(self, image: np.ndarray, frame_index: int | None = None) -> list[Detection]:
        if frame_index is None:  # crop / still-image query
            return self.on_crop(image) if self.on_crop else []
        return list(self.per_frame.get(frame_index, []))


def build_detector(cfg: dict, scripted: dict[int, list[Detection]] | None = None) -> Detector:
    d = cfg["detector"]
    if d["backend"] == "scripted" or scripted is not None:
        return ScriptedDetector(scripted or {})
    return RFDETRDetector(d["size"], d["threshold"], d.get("weights"), d.get("class_names"),
                          d.get("segmentation", False))
