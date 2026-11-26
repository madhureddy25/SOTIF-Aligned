"""Stage 2: semantic keyframe compression + evidence packaging.

For one candidate window this builds what the diagnosis agent will see:

* evidence_gated (our method): 4-8 timestamped keyframes, dense around onset/peak; only the
  1-3 conflicting actors are marked with a small numeric ID; zoomed, contrast-normalized ROI
  crops of those actors; a 5-line telemetry header.
* video_only (ablation baseline): the same keyframes, unmarked, no crops, no telemetry.
* raw_cv_dump (ablation baseline): every detection drawn on every frame plus a text dump of
  all boxes, i.e. the "pipe detector output into the prompt" approach that the DeepAccident
  study found can add noise.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..mining.runner import PerceptionLog
from ..mining.signals import in_ego_corridor
from ..perception.tracking import box_iou
from ..schemas import CandidateWindow, EvidenceBundle, FrameRecord, RoiPatch, Telemetry

FOCUS_COLOURS = [(0, 220, 255), (255, 120, 0), (180, 0, 255)]  # BGR, high-contrast


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------


def select_keyframes(window: CandidateWindow, max_frames: int = 8, dense_fps: float = 5.0,
                     sparse_fps: float = 1.0) -> list[int]:
    """Frame indices (strided space): sparse over the approach and aftermath, dense from onset to
    shortly after the peak. Onset and peak are always included."""
    fps = window.fps
    s, e = window.start_frame, window.end_frame
    on, pk = window.onset_frame, window.peak_frame
    d0, d1 = max(s, on - int(0.5 * fps)), min(e, pk + int(1.0 * fps))
    dense = set(range(d0, d1 + 1, max(1, int(round(fps / dense_fps)))))
    sparse = set(range(s, e + 1, max(1, int(round(fps / sparse_fps)))))
    # the peak +-0.5 s must survive trimming (that is where the conflict is); a near-1 s gap over
    # the event, as seen on Nexar clip 00000, starves the VLM of the critical moment
    peak_win = {min(e, max(s, pk + off)) for off in range(-int(0.5 * fps), int(0.5 * fps) + 1)}
    must = {on, pk} | (peak_win & (dense | sparse))
    cand = sorted(dense | sparse | must)
    if len(cand) <= max_frames:
        return cand
    # keep onset/peak, then spread the rest evenly, preferring dense frames
    rest = [f for f in cand if f not in must]
    n_dense = max(0, max_frames - len(must) - 2)
    dense_rest = [f for f in rest if f in dense]
    pick = set(must)
    if dense_rest:
        idx = np.linspace(0, len(dense_rest) - 1, min(n_dense, len(dense_rest))).round().astype(int)
        pick |= {dense_rest[i] for i in idx}
    others = [f for f in rest if f not in pick]
    if others and len(pick) < max_frames:
        idx = np.linspace(0, len(others) - 1, min(max_frames - len(pick), len(others))).round().astype(int)
        pick |= {others[i] for i in idx}
    return sorted(pick)[:max_frames]


# ---------------------------------------------------------------------------
# Focus actors and telemetry
# ---------------------------------------------------------------------------


def _window_frames(log: PerceptionLog, w: CandidateWindow) -> list[FrameRecord]:
    return [f for f in log.frames if w.start_frame <= f.frame <= w.end_frame]


def pick_focus_tracks(log: PerceptionLog, w: CandidateWindow, k: int = 3) -> list[int]:
    """Rank candidate actors by how likely they are the conflict partner (ego-path closing,
    proximity, VRU), with perception instability only as a tie-breaker. Candidates = tracks
    Stage 1 implicated, plus the largest ego-corridor tracks near the peak."""
    from ..config import is_vru
    from ..mining.signals import _looming

    frames = _window_frames(log, w)
    cand = list(dict.fromkeys(w.focus_track_ids))
    near = [f for f in frames if abs(f.frame - w.peak_frame) <= int(w.fps)]
    area: dict[int, float] = defaultdict(float)
    for f in near:
        for t in f.tracks:
            if in_ego_corridor(t.box, log.width, log.height):
                area[t.track_id] = max(area[t.track_id], (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
    cand += [tid for tid, _ in sorted(area.items(), key=lambda x: -x[1]) if tid not in cand]
    if not cand:
        return []
    obs: dict[int, list] = defaultdict(list)
    for f in frames:
        for t in f.tracks:
            obs[t.track_id].append(t)
    _, _, loom = _looming(frames, log.width, log.height, ego_only=True, min_span_s=0.6)

    W, H = log.width, log.height

    def hazard(tid: int) -> float:
        """Who is the conflict partner? Ego-path closing, then proximity (box bottom low in the
        frame, near the centre, large), VRUs first on ties. Perception instability is NOT used
        here: ranking by instability and then attributing to instability would be circular."""
        o = obs.get(tid, [])
        if not o:
            return -1.0
        near_peak = min(o, key=lambda t: abs(t.frame - w.peak_frame))
        x1, y1, x2, y2 = near_peak.box
        proximity = (y2 / H) * (1 - abs((x1 + x2) / 2 - W / 2) / (W / 2)) + ((x2 - x1) * (y2 - y1)) / (W * H)
        in_path = 1.0 if in_ego_corridor(near_peak.box, W, H) else 0.0
        return 3 * max(0.0, loom.get(tid, 0.0)) + 2 * in_path + proximity + (0.5 if is_vru(o[0].cls) else 0.0)

    def instability(tid: int) -> float:  # tie-breaker only
        o = obs.get(tid, [])
        return sum(1 for t in o if t.reacquired_gap > 0) + (sum(t.score < 0.4 for t in o) / len(o) if o else 0)

    ranked = sorted(cand, key=lambda tid: (-round(hazard(tid), 2), -instability(tid)))
    return [tid for tid in ranked if tid in obs][:k]


def compute_telemetry(log: PerceptionLog, w: CandidateWindow, focus: list[int]) -> Telemetry:
    from ..mining.signals import _looming, ego_motion_features

    frames = _window_frames(log, w)
    if not frames:
        return Telemetry()
    t = Telemetry()
    by_id: dict[int, list] = defaultdict(list)
    for f in frames:
        for tr in f.tracks:
            by_id[tr.track_id].append((f.frame, tr))
    gaps = [tr.reacquired_gap for f in frames for tr in f.tracks if tr.track_id in focus]
    t.track_gap_frames = max(gaps, default=0)
    prim = [tr for f in frames for tr in f.tracks if focus and tr.track_id == focus[0]]
    t.focus_reacquired_gap = max((tr.reacquired_gap for tr in prim), default=0)
    t.focus_dropouts = sum(1 for tr in prim if tr.reacquired_gap > 0)
    t.focus_low_conf_frac = (sum(tr.score < 0.4 for tr in prim) / len(prim)) if prim else 0.0
    confs = [tr.score for tid in focus for _, tr in by_id.get(tid, [])]
    if confs:
        t.conf_min, t.conf_max = float(min(confs)), float(max(confs))
    if focus and by_id.get(focus[0]):
        obs = by_id[focus[0]]
        first_frame, first = obs[0]
        last_frame = obs[-1][0]
        t.focus_cls = first.cls
        # presence only over the actor's own lifespan in the window (appearing late is not "unstable")
        span = sum(1 for f in frames if first_frame <= f.frame <= last_frame)
        t.focus_presence = len(obs) / max(span, 1)
        if first.is_new and first_frame > w.start_frame:
            t.focus_born_mid = True
            others = [o.box for f in frames if f.frame == first_frame for o in f.tracks if o.track_id != focus[0]]
            others += [d.box for f in frames if f.frame == first_frame for d in f.detections]
            t.focus_born_adjacent = any(box_iou(first.box, b) > 0.0 and box_iou(first.box, b) < 0.9 for b in others)
    # ego-path looming over sliding ~1 s stretches (a 1.5 s closing event must not be averaged
    # with seconds of steady following)
    best, best_ttc = 0.0, None
    span = max(3, int(round(w.fps)))
    for i in range(0, max(1, len(frames) - span + 1), max(1, span // 4)):
        rate, ttc, _ = _looming(frames[i:i + span], log.width, log.height, ego_only=True, min_span_s=0.6)
        best = max(best, rate)
        if ttc is not None:
            best_ttc = ttc if best_ttc is None else min(best_ttc, ttc)
    t.ego_looming = 2 * best
    t.looming_rate = 2 * best
    t.min_ttc_s = best_ttc
    jerks = []
    step = max(1, int(round(w.fps / 2)))
    for i in range(0, len(frames), step):
        e = ego_motion_features(frames[i:i + step], frames[max(0, i - 2 * step):i], log.width, log.height)
        if e:
            jerks.append(max(e["pitch_jerk"], e["yaw_jerk"]))
    t.ego_jerk = float(max(jerks, default=0.0))
    # worst 1-second stretch inside the window (a 2 s tunnel must not be averaged away)
    k = max(1, int(round(w.fps)))
    def roll(vals, worst):
        v = np.asarray(vals, float)
        if len(v) <= k:
            return float(v.mean())
        m = np.convolve(v, np.ones(k) / k, mode="valid")
        return float(worst(m))
    t.luminance = roll([f.luminance for f in frames], np.min)
    t.saturation_frac = roll([f.saturation_frac for f in frames], np.max)
    t.sharpness = roll([f.sharpness for f in frames], np.min)
    t.active_tracks = int(max(len(f.tracks) for f in frames))
    return t


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def enhance_crop(img: np.ndarray, min_side: int = 384) -> np.ndarray:
    """Upscale so the short side is >= min_side, then CLAHE on lightness (contrast normalization)."""
    h, w = img.shape[:2]
    s = max(1.0, min_side / max(1, min(h, w)))
    if s > 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_box(box, w: int, h: int, pad: float = 0.35, min_px: int = 64):
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, min_px), max(y2 - y1, min_px)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = bw * (0.5 + pad), bh * (0.5 + pad)
    return (int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(w, cx + hw)), int(min(h, cy + hh)))


def _stamp(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (8 + 11 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def build_bundle(window: CandidateWindow, log: PerceptionLog, images: dict[int, np.ndarray], out_dir: Path,
                 cfg: dict, arm: str = "evidence_gated") -> EvidenceBundle:
    """`images` maps strided frame index -> BGR frame for at least the selected keyframes."""
    ev = cfg["evidence"]
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = select_keyframes(window, ev["max_frames"], ev["keyframe_fps_dense"], ev["keyframe_fps_sparse"])
    keys = [k for k in keys if k in images]
    rec_by_frame = {f.frame: f for f in log.frames}
    focus = pick_focus_tracks(log, window, ev["max_rois"])
    tele = compute_telemetry(log, window, focus)
    max_side = ev.get("frame_max_side", 768)

    paths, times, dump_lines = [], [], []
    for label, k in enumerate(keys, 1):
        img = images[k].copy()
        fr = rec_by_frame.get(k)
        t = k / window.fps
        if arm == "evidence_gated" and fr is not None:
            for i, tid in enumerate(focus):
                tr = next((x for x in fr.tracks if x.track_id == tid), None)
                if tr is None:
                    continue
                x1, y1, x2, y2 = map(int, tr.box)
                c = FOCUS_COLOURS[i % len(FOCUS_COLOURS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img, str(i + 1), (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        elif arm == "raw_cv_dump" and fr is not None:
            for d in fr.detections:
                x1, y1, x2, y2 = map(int, d.box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(img, f"{d.cls} {d.score:.2f}", (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 255, 0), 1)
            dump_lines.append(f"t={t:.2f}s: " + "; ".join(
                f"{d.cls}({d.score:.2f})[{d.box[0]:.0f},{d.box[1]:.0f},{d.box[2]:.0f},{d.box[3]:.0f}]"
                for d in fr.detections) if fr.detections else f"t={t:.2f}s: none")
        img = _stamp(_resize_max(img, max_side), f"F{label}  t={t:.2f}s")
        p = out_dir / f"{arm}_f{k:05d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(str(p))
        times.append(t)

    rois: list[RoiPatch] = []
    if arm == "evidence_gated":
        H, W = next(iter(images.values())).shape[:2]
        for i, tid in enumerate(focus):
            # Counterfactual crop: the image at the moment of failure (the peak), at the actor's
            # nearest known position, i.e. "was the actor physically visible when perception failed?"
            cands = [(abs(f.frame - window.peak_frame), tr) for f in log.frames
                     if window.start_frame <= f.frame <= window.end_frame
                     for tr in f.tracks if tr.track_id == tid]
            if not cands:
                continue
            _, tr = min(cands, key=lambda c: c[0])
            fidx = min(images, key=lambda fi: abs(fi - window.peak_frame))
            cb = crop_box(tr.box, W, H, ev["roi_pad"])
            crop = images[fidx][cb[1]:cb[3], cb[0]:cb[2]]
            if crop.size == 0:
                continue
            p = out_dir / f"roi{i + 1}_track{tid}_f{fidx:05d}.jpg"
            cv2.imwrite(str(p), enhance_crop(crop, ev["roi_upscale_min"]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            scale = max(1.0, ev["roi_upscale_min"] / max(1, min(crop.shape[:2])))
            rois.append(RoiPatch(track_id=tid, frame=fidx, crop_box=tuple(map(float, cb)), path=str(p), cls_hint=tr.cls,
                                 actor_box=tuple(map(float, tr.box)), scale=scale))

    return EvidenceBundle(
        clip_id=window.clip_id, arm=arm, frame_paths=paths, frame_ids=keys, frame_times=times, rois=rois,
        telemetry=tele if arm == "evidence_gated" else Telemetry(),
        text_dump="\n".join(dump_lines) if arm == "raw_cv_dump" else None, window=window)


def actor_box_in_crop(roi: RoiPatch) -> tuple[float, float, float, float] | None:
    """The focus actor's box in the coordinates of the enhanced (upscaled) crop image."""
    if roi.actor_box is None:
        return None
    cx, cy = roi.crop_box[0], roi.crop_box[1]
    x1, y1, x2, y2 = roi.actor_box
    s = roi.scale
    return ((x1 - cx) * s, (y1 - cy) * s, (x2 - cx) * s, (y2 - cy) * s)
