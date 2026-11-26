"""Stage 1 pass 1: run detector + tracker + frame statistics over a source."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.video import FrameSource, frame_stats
from ..perception.detector import Detector
from ..perception.embed import ImageEmbedder
from ..perception.tracking import Tracker
from ..schemas import FrameRecord


@dataclass
class PerceptionLog:
    source_id: str
    fps: float
    width: int
    height: int
    frames: list[FrameRecord]
    step_frames: list[int] = field(default_factory=list)  # frame index of each step keyframe
    step_embeddings: np.ndarray | None = None
    single_frame: bool = False


def run_perception(source: FrameSource, detector: Detector, tracker: Tracker | None,
                   embedder: ImageEmbedder | None, step_s: float = 0.5) -> PerceptionLog:
    recs: list[FrameRecord] = []
    keyframes, key_idx = [], []
    step = max(1, int(round(step_s * source.fps)))
    w = h = 0
    for fr in source:
        h, w = fr.image.shape[:2]
        dets = detector.detect(fr.image, fr.index)
        tracks, term = tracker.update(dets, fr.image, fr.index) if tracker else ([], [])
        lum, sat, sharp = frame_stats(fr.image)
        A = getattr(tracker, "last_affine", None) if tracker else None
        ego = {}
        if A is not None:
            ego = {"ego_dx": float(A[0, 2]), "ego_dy": float(A[1, 2]),
                   "ego_scale": float(np.sqrt(abs(np.linalg.det(A[:, :2]))))}
        recs.append(FrameRecord(frame=fr.index, t=fr.t, detections=dets, tracks=tracks, luminance=lum,
                                saturation_frac=sat, sharpness=sharp, terminated_ids=term, **ego))
        if fr.index % step == 0:
            key_idx.append(fr.index)
            if embedder is not None:
                keyframes.append(fr.image)
    emb = embedder.embed(keyframes) if embedder is not None and keyframes else None
    return PerceptionLog(source.source_id, source.fps, w, h, recs, key_idx, emb,
                         single_frame=tracker is None)


def save_log(log: PerceptionLog, path) -> None:
    """Compact gzip JSON of the perception log, so features can be recomputed without detection."""
    import gzip
    import json
    from pathlib import Path

    def r(x):
        return round(float(x), 2)

    rec = {"source_id": log.source_id, "fps": log.fps, "width": log.width, "height": log.height,
           "single_frame": log.single_frame, "step_frames": log.step_frames,
           "step_embeddings": None if log.step_embeddings is None else np.round(log.step_embeddings, 4).tolist(),
           "frames": [{"f": fr.frame, "t": round(fr.t, 4), "lum": round(fr.luminance, 4), "sat": round(fr.saturation_frac, 5),
                       "shp": round(fr.sharpness, 4), "term": fr.terminated_ids,
                       "ego": [round(fr.ego_dx, 3), round(fr.ego_dy, 3), round(fr.ego_scale, 5)],
                       "det": [[*map(r, d.box), round(d.score, 3), d.cls] for d in fr.detections],
                       "trk": [[t.track_id, *map(r, t.box), round(t.score, 3), t.cls, int(t.is_new), t.reacquired_gap]
                               for t in fr.tracks]} for fr in log.frames]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(rec, fh, separators=(",", ":"))


def load_log(path) -> PerceptionLog:
    import gzip
    import json

    from ..schemas import Detection, TrackState

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rec = json.load(fh)
    frames = []
    for fr in rec["frames"]:
        frames.append(FrameRecord(
            frame=fr["f"], t=fr["t"], luminance=fr["lum"], saturation_frac=fr["sat"], sharpness=fr["shp"],
            terminated_ids=fr["term"], ego_dx=fr["ego"][0], ego_dy=fr["ego"][1], ego_scale=fr["ego"][2],
            detections=[Detection(box=tuple(d[:4]), score=d[4], cls=d[5]) for d in fr["det"]],
            tracks=[TrackState(track_id=t[0], frame=fr["f"], box=tuple(t[1:5]), score=t[5], cls=t[6],
                               is_new=bool(t[7]), reacquired_gap=t[8]) for t in fr["trk"]]))
    emb = rec.get("step_embeddings")
    return PerceptionLog(rec["source_id"], rec["fps"], rec["width"], rec["height"], frames, rec["step_frames"],
                         None if emb is None else np.asarray(emb, dtype=np.float32), rec["single_frame"])
