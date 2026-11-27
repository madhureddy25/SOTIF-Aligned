"""The four Stage 1 trigger signals, each normalized to [0, 1] per 0.5 s step.

Each scorer returns (score, detail, implicated_track_ids).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..schemas import FrameRecord


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _bin_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _near_border(box, w: int, h: int, margin: float = 0.03) -> bool:
    mx, my = w * margin, h * margin
    return box[0] <= mx or box[1] <= my or box[2] >= w - mx or box[3] >= h - my


def uncertainty_signal(frames: list[FrameRecord], band=(0.25, 0.60)):
    """Borderline-confidence entropy + per-track detection flicker + score variance."""
    lo, hi = band
    scores = np.array([d.score for f in frames for d in f.detections])
    band_mask = (scores >= lo) & (scores < hi) if len(scores) else np.array([], bool)
    band_part = float(_bin_entropy(scores[band_mask]).sum() / max(len(scores), 4)) if band_mask.any() else 0.0

    presence: dict[int, list[int]] = defaultdict(list)
    tscores: dict[int, list[float]] = defaultdict(list)
    for k, f in enumerate(frames):
        for t in f.tracks:
            presence[t.track_id].append(k)
            tscores[t.track_id].append(t.score)
    dropouts, implicated = 0, set()
    for tid, ks in presence.items():
        gaps = int(np.sum(np.diff(ks) > 1)) if len(ks) > 1 else 0
        if gaps:
            dropouts += gaps
            implicated.add(tid)
    for f in frames:
        for t in f.tracks:
            if t.reacquired_gap > 0:
                dropouts += 1
                implicated.add(t.track_id)
    std = max((float(np.std(v)) for v in tscores.values() if len(v) > 2), default=0.0)
    for tid, v in tscores.items():
        if len(v) > 2 and np.std(v) > 0.08:
            implicated.add(tid)
    flicker = _clip01(dropouts / 2.0)
    var_part = _clip01(std / 0.15)
    score = max(_clip01(band_part * 2.5), flicker, var_part)
    return score, {"band_entropy": band_part, "flicker": flicker, "dropouts": float(dropouts), "score_std": std}, implicated


def vanish_events(frames: list[FrameRecord], fps: float, min_gap_s: float = 1.0) -> dict[int, list[tuple[int, tuple]]]:
    """Tracks seen for the last time, keyed by the index of their last sighting: {index: [(id, last box)]}.

    The tracker only reports a termination after its lost buffer (~3 s), which is too late to place
    the event and lands outside the scorer's box history. A track that is never seen again, with at
    least `min_gap_s` of clip left after its last sighting, vanished at that sighting.
    """
    last: dict[int, tuple[int, tuple]] = {}
    for i, f in enumerate(frames):
        for t in f.tracks:
            last[t.track_id] = (i, t.box)
    gap = max(1, int(round(min_gap_s * fps)))
    out: dict[int, list[tuple[int, tuple]]] = {}
    for tid, (i, box) in last.items():
        if len(frames) - 1 - i >= gap:
            out.setdefault(i, []).append((tid, box))
    return out


def continuity_signal(frames: list[FrameRecord], prev_frames: list[FrameRecord], fps: float, w: int, h: int,
                      looming_rate: float = 0.8, vanished: list[tuple[int, tuple]] | None = None):
    """Mid-frame track terminations/births, reacquisition gaps, looming (TTC proxy).

    `vanished`: (track id, last box) of tracks last seen in `frames` (see `vanish_events`). Without it,
    the tracker's delayed termination reports are used, which only works for short lost buffers.
    """
    if vanished is None:
        last_box: dict[int, tuple] = {}
        for f in prev_frames + frames:
            for t in f.tracks:
                last_box[t.track_id] = t.box
        vanished = [(tid, last_box[tid]) for f in frames for tid in f.terminated_ids if tid in last_box]
    mid_term, mid_birth, implicated = 0, 0, set()
    for tid, b in vanished:
        if not _near_border(b, w, h):
            mid_term += 1
            implicated.add(tid)
    for f in frames:
        for t in f.tracks:
            if t.is_new and not _near_border(t.box, w, h) and f.frame > 2:
                area = (t.box[2] - t.box[0]) * (t.box[3] - t.box[1])
                if area > 0.002 * w * h:  # ignore distant specks appearing near the horizon
                    mid_birth += 1
                    implicated.add(t.track_id)
    reacq = [t.reacquired_gap for f in frames for t in f.tracks if t.reacquired_gap > 0]
    # Looming: per-track width expansion rate; TTC ~= w / (dw/dt).
    best_rate, min_ttc, loom_ids = _looming(prev_frames[-len(frames):] + frames, w, h, ego_only=False)
    implicated |= {t for t, r in loom_ids.items() if r * 2 > looming_rate}
    # Ego-path looming: only vehicles/VRUs in the corridor ahead of the camera, large enough to
    # matter, fitted over ~1 s. Jitter on distant or passing traffic is excluded.
    hist = prev_frames[-int(round(fps)):] + frames
    ego_rate, ego_ttc, ego_ids = _looming(hist, w, h, ego_only=True, min_span_s=0.6)
    implicated |= {t for t, r in ego_ids.items() if r * 2 > looming_rate}
    looming = _clip01(2 * max(best_rate, 0.0) / looming_rate)  # area rate ~= 2 * width rate
    # A track appearing or vanishing mid-frame is surprising in sparse traffic but routine in a
    # crowd, so normalize by how many tracks are already active.
    n_active = max((len(f.tracks) for f in frames), default=0)
    norm = max(1.0, 0.2 * n_active)
    score = max(_clip01(mid_term / norm), _clip01(mid_birth / norm), _clip01(len(reacq) / 2.0), looming)
    return score, {"active_tracks": n_active, "term_norm": mid_term / norm, "birth_norm": mid_birth / norm,
                   "mid_terminations": mid_term, "mid_births": mid_birth, "reacquisitions": len(reacq),
                   "max_gap": max(reacq, default=0), "looming_rate": 2 * best_rate,
                   "min_ttc_s": min_ttc if min_ttc is not None else 99.0,
                   "ego_looming": 2 * ego_rate, "ego_inv_ttc": 1.0 / ego_ttc if ego_ttc else 0.0}, implicated


def in_ego_corridor(box, w: int, h: int) -> bool:
    """Box bottom-centre inside a trapezoid ahead of the camera (lower 55% of the frame;
    central 30% of the width at the horizon widening to 70% at the bottom)."""
    cx, by = (box[0] + box[2]) / 2, box[3]
    if by < 0.45 * h:
        return False
    u = min(1.0, max(0.0, (by - 0.45 * h) / (0.55 * h)))
    half = (0.15 + 0.20 * u) * w
    return abs(cx - w / 2) <= half and (box[2] - box[0]) >= 0.03 * w


def _looming(frames: list[FrameRecord], w: int, h: int, ego_only: bool, min_span_s: float = 0.2
             ) -> tuple[float, float | None, dict[int, float]]:
    """Max log-width growth rate (1/s) over tracks, min TTC (s), and per-track rates."""
    series: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for f in frames:
        for t in f.tracks:
            if ego_only and not in_ego_corridor(t.box, w, h):
                continue
            series[t.track_id].append((f.t, max(t.box[2] - t.box[0], 1.0)))
    best, min_ttc, rates = 0.0, None, {}
    for tid, sr in series.items():
        if len(sr) < 3:
            continue
        ts, ws = np.array(sr).T
        if ts[-1] - ts[0] < min_span_s:
            continue
        slope = float(np.polyfit(ts, np.log(ws), 1)[0])  # d ln w / dt
        rates[tid] = slope
        best = max(best, slope)
        if slope > 0.05:
            min_ttc = 1.0 / slope if min_ttc is None else min(min_ttc, 1.0 / slope)
    return best, min_ttc, rates


def ego_motion_features(frames: list[FrameRecord], prev_frames: list[FrameRecord], w: int, h: int) -> dict:
    """Driver-reaction cues from the camera's own motion (estimated by the tracker's motion
    compensation): hard braking pitches the camera (vertical image shift), swerving shifts it
    sideways, and sudden deceleration changes the expansion rate (scale). Jerk = change of the
    step's mean motion vs. the previous ~1 s; shake = within-step spread."""
    def arr(fs):
        return np.array([[f.ego_dx / max(w, 1), f.ego_dy / max(h, 1), f.ego_scale - 1.0] for f in fs]) if fs else None
    cur, prev = arr(frames), arr(prev_frames[-2 * len(frames):] if prev_frames else [])
    if cur is None:
        return {}
    m = cur.mean(axis=0)
    pm = prev.mean(axis=0) if prev is not None and len(prev) else m
    return {"yaw_jerk": float(abs(m[0] - pm[0])), "pitch_jerk": float(abs(m[1] - pm[1])),
            "scale_jerk": float(abs(m[2] - pm[2])), "shake": float(cur[:, 1].std()),
            "lateral_speed": float(abs(m[0])), "pitch_speed": float(abs(m[1]))}


def optical_density_signal(frames: list[FrameRecord], prev_frames: list[FrameRecord], cfg: dict):
    lum = float(np.mean([f.luminance for f in frames]))
    sat = float(np.mean([f.saturation_frac for f in frames]))
    sharp = float(np.mean([f.sharpness for f in frames]))
    n_tr = max((len(f.tracks) or len(f.detections)) for f in frames)
    dark_t, glare_t, blur_t, dense = cfg["dark_luminance"], cfg["glare_saturation"], cfg["blur_sharpness"], cfg["dense_tracks"]
    dark = _clip01((0.35 - lum) / (0.35 - dark_t))
    glare = _clip01(sat / glare_t)
    blur = _clip01((0.45 - sharp) / (0.45 - blur_t))
    density = _clip01((n_tr - 10) / max(dense - 10, 1))
    collapse = 0.0
    if prev_frames:
        prev_lum = float(np.mean([f.luminance for f in prev_frames[-len(frames):]]))
        collapse = _clip01((prev_lum - lum) / 0.25)  # sudden luminance drop (tunnel entry, flare recovery)
    score = max(dark, glare, blur, density, collapse)
    return score, {"luminance": lum, "saturation": sat, "sharpness": sharp, "active_tracks": n_tr,
                   "dark": dark, "glare": glare, "blur": blur, "density": density, "collapse": collapse}, set()


def rarity_signal(step_emb: np.ndarray, corpus: np.ndarray | None, step_s: float, k: int = 5,
                  exclude_s: float = 2.0) -> np.ndarray:
    """Per-step kNN cosine distance, robust-z normalized to [0, 1].

    Reference set = external corpus index if given, else the clip's own other steps
    (excluding +-exclude_s, so a scene is not compared with itself)."""
    n = len(step_emb)
    if n == 0:
        return np.zeros(0)
    dists = np.zeros(n)
    excl = int(round(exclude_s / step_s))
    for i in range(n):
        if corpus is not None and len(corpus):
            ref = corpus
        else:
            idx = [j for j in range(n) if abs(j - i) > excl]
            if len(idx) < 2:
                return np.zeros(n)
            ref = step_emb[idx]
        d = 1.0 - ref @ step_emb[i]
        kk = min(k, len(d))
        dists[i] = float(np.sort(d)[:kk].mean())
    med = np.median(dists)
    mad = max(np.median(np.abs(dists - med)) * 1.4826, 0.03)
    return np.clip((dists - med) / (4 * mad), 0, 1)
