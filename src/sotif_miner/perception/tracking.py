"""Multi-object tracking for a moving windshield camera.

`CMCSort` is an in-repo BoT-SORT-style tracker: constant-velocity Kalman filter,
ByteTrack two-pass association (high- then low-confidence detections), and
global camera-motion compensation estimated with sparse optical flow + RANSAC.
It is dependency-light (numpy/scipy/OpenCV) and Apache-2.0 compatible, avoiding
AGPL tracker packages. Adapters for Roboflow `trackers` are provided when installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..schemas import Box, Detection, TrackState


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def box_iou(a: Box, b: Box) -> float:
    return float(iou_matrix(np.array([a]), np.array([b]))[0, 0])


class Tracker(Protocol):
    name: str

    def update(self, detections: list[Detection], image: np.ndarray | None, frame: int
               ) -> tuple[list[TrackState], list[int]]:
        """Returns (tracks matched this frame, ids terminated this frame)."""
        ...


# ---------------------------------------------------------------------------
# Kalman filter on [cx, cy, w, h, vcx, vcy, vw, vh]
# ---------------------------------------------------------------------------

_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.eye(4, 8)


def _xyxy_to_z(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w, h])


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    cx, cy, w, h = x[:4]
    w, h = max(w, 1.0), max(h, 1.0)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class _Track:
    tid: int
    x: np.ndarray
    P: np.ndarray
    cls: str
    score: float
    hits: int = 1
    lost: int = 0
    born_frame: int = 0
    confirmed: bool = False
    reported_new: bool = False
    history: list[int] = field(default_factory=list)

    @classmethod
    def create(cls, tid: int, det: Detection, frame: int) -> _Track:
        z = _xyxy_to_z(np.array(det.box))
        x = np.r_[z, np.zeros(4)]
        s = max(z[2], z[3])
        P = np.diag([s, s, s, s, 10 * s, 10 * s, 10 * s, 10 * s]) * 0.1
        return cls(tid, x, P, det.cls, det.score, born_frame=frame)

    def predict(self) -> None:
        s = max(self.x[2], self.x[3], 1.0)
        q = np.r_[np.full(4, (0.05 * s) ** 2), np.full(4, (0.0125 * s) ** 2)]
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + np.diag(q)

    def update(self, det: Detection) -> None:
        z = _xyxy_to_z(np.array(det.box))
        s = max(self.x[2], self.x[3], 1.0)
        R = np.diag(np.full(4, (0.05 * s) ** 2))
        S = _H @ self.P @ _H.T + R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - _H @ self.x)
        self.P = (np.eye(8) - K @ _H) @ self.P
        self.cls, self.score = det.cls, det.score

    def apply_affine(self, A: np.ndarray) -> None:
        """Warp the state by a 2x3 similarity transform (camera ego-motion)."""
        c = A[:, :2] @ self.x[:2] + A[:, 2]
        scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
        self.x[:2] = c
        self.x[2:4] *= scale
        self.x[4:6] = A[:, :2] @ self.x[4:6]
        self.x[6:8] *= scale

    @property
    def box(self) -> np.ndarray:
        return _x_to_xyxy(self.x)


class GlobalMotion:
    """Frame-to-frame camera motion (2x3 partial affine) from sparse optical flow."""

    def __init__(self, max_corners: int = 400, downscale: float = 0.5):
        self.prev: np.ndarray | None = None
        self.max_corners, self.downscale = max_corners, downscale

    def estimate(self, image: np.ndarray, mask_boxes: np.ndarray | None = None) -> np.ndarray:
        ident = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        g = cv2.resize(gray, None, fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, g
        if prev is None or prev.shape != g.shape:
            return ident
        mask = np.full(prev.shape, 255, np.uint8)
        if mask_boxes is not None:  # exclude moving foreground objects from ego-motion estimation
            for b in (mask_boxes * self.downscale).astype(int):
                mask[max(b[1], 0):b[3], max(b[0], 0):b[2]] = 0
        pts = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8, mask=mask)
        if pts is None or len(pts) < 8:
            return ident
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None)
        ok = st.ravel() == 1
        if ok.sum() < 8:
            return ident
        A, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if A is None:
            return ident
        A[:, 2] /= self.downscale
        return A


class CMCSort:
    name = "cmc_sort"

    def __init__(self, high_thresh: float = 0.6, low_thresh: float = 0.25, match_iou: float = 0.3,
                 lost_buffer: int = 30, min_hits: int = 2, motion_compensation: bool = True):
        self.high, self.low, self.match_iou = high_thresh, low_thresh, match_iou
        self.lost_buffer, self.min_hits = lost_buffer, min_hits
        self.gmc = GlobalMotion() if motion_compensation else None
        self.tracks: list[_Track] = []
        self._next = 1
        self.last_affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

    def _associate(self, tracks: list[_Track], dets: list[Detection], thr: float):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        ious = iou_matrix(np.array([t.box for t in tracks]), np.array([d.box for d in dets]))
        r, c = linear_sum_assignment(-ious)
        matches = [(i, j) for i, j in zip(r, c, strict=False) if ious[i, j] >= thr]
        mi = {i for i, _ in matches}
        mj = {j for _, j in matches}
        return matches, [i for i in range(len(tracks)) if i not in mi], [j for j in range(len(dets)) if j not in mj]

    def update(self, detections: list[Detection], image: np.ndarray | None, frame: int
               ) -> tuple[list[TrackState], list[int]]:
        for t in self.tracks:
            t.predict()
        if self.gmc is not None and image is not None:
            boxes = np.array([d.box for d in detections]) if detections else None
            self.last_affine = self.gmc.estimate(image, boxes)
            for t in self.tracks:
                t.apply_affine(self.last_affine)

        high = [d for d in detections if d.score >= self.high]
        low = [d for d in detections if self.low <= d.score < self.high]

        # Pass 1: all tracks vs high-confidence detections.
        m1, um_t, um_hi = self._associate(self.tracks, high, self.match_iou)
        matched: list[tuple[_Track, Detection]] = [(self.tracks[i], high[j]) for i, j in m1]
        # Pass 2: remaining recently-seen tracks vs low-confidence detections (recovers occluded/flickering boxes).
        rem = [self.tracks[i] for i in um_t]
        recent = [t for t in rem if t.lost == 0]
        m2, _, _ = self._associate(recent, low, 0.5)
        matched += [(recent[i], low[j]) for i, j in m2]
        matched_ids = {id(t) for t, _ in matched}

        out: list[TrackState] = []
        for t, d in matched:
            gap = t.lost
            t.update(d)
            t.hits += 1
            t.lost = 0
            if t.hits >= self.min_hits:
                t.confirmed = True
            if t.confirmed:
                is_new = not t.reported_new
                t.reported_new = True
                out.append(TrackState(track_id=t.tid, frame=frame, box=tuple(map(float, t.box)), score=d.score,
                                      cls=d.cls, is_new=is_new, reacquired_gap=gap if gap > 0 else 0))

        terminated: list[int] = []
        survivors: list[_Track] = []
        for t in self.tracks:
            if id(t) not in matched_ids:
                t.lost += 1
                if t.lost > self.lost_buffer or (not t.confirmed and t.lost > 1):
                    if t.confirmed:
                        terminated.append(t.tid)
                    continue
            survivors.append(t)
        self.tracks = survivors

        for j in um_hi:
            self.tracks.append(_Track.create(self._next, high[j], frame))
            self._next += 1
        return out, terminated

    def lost_track_boxes(self) -> dict[int, Box]:
        """Predicted boxes of confirmed tracks currently unmatched (for occlusion reasoning)."""
        return {t.tid: tuple(map(float, t.box)) for t in self.tracks if t.lost > 0 and t.confirmed}


class TrackersAdapter:  # pragma: no cover - exercised only when `trackers` is installed
    """Wraps Roboflow `trackers` (Apache-2.0) SORT-family trackers behind the same interface."""

    def __init__(self, kind: str = "bytetrack", **kw):
        import supervision as sv
        import trackers

        self._sv = sv
        cls = {"bytetrack": "ByteTrackTracker", "ocsort": "OCSORTTracker", "sort": "SORTTracker"}[kind]
        self.t = getattr(trackers, cls)(**kw)
        self.name = kind
        self.seen: set[int] = set()
        self.active: set[int] = set()

    def update(self, detections, image, frame):
        sv = self._sv
        if detections:
            xyxy = np.array([d.box for d in detections], dtype=float)
            dets = sv.Detections(xyxy=xyxy, confidence=np.array([d.score for d in detections]),
                                 class_id=np.arange(len(detections)))
        else:
            dets = sv.Detections.empty()
        res = self.t.update(dets)
        out, now = [], set()
        for box, tid, cid in zip(res.xyxy, res.tracker_id, res.class_id, strict=False):
            if tid is None or tid < 0:
                continue
            d = detections[int(cid)]
            now.add(int(tid))
            out.append(TrackState(track_id=int(tid), frame=frame, box=tuple(map(float, box)), score=d.score,
                                  cls=d.cls, is_new=int(tid) not in self.seen))
            self.seen.add(int(tid))
        terminated = sorted(self.active - now)  # approximation: unmatched == terminated for reporting
        self.active = now
        return out, terminated


def build_tracker(cfg: dict) -> Tracker:
    t = dict(cfg["tracker"])
    kind = t.pop("backend")
    if kind == "cmc_sort":
        return CMCSort(**t)
    return TrackersAdapter(kind)
