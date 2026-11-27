"""Stage 1 pass 2: rolling Edge-Case Candidate Score and window slicing.

Two steps, kept separate so thresholds can be retuned without re-running detection:

1. `compute_features`  (slow, needs the perception log)
   Every `step_s` (0.5 s) record raw, unclipped features: detector band entropy, dropouts,
   score spread, mid-frame births/terminations, reacquisitions, looming rate, luminance,
   specular saturation, sharpness, active tracks, luminance collapse, plus embedding rarity.

2. `score_features`    (instant, needs only the stored features)
   Real dashcam video is never "clean": distant half-confident boxes, bright skies and busy
   roads are the normal state of a drive. So each feature is scored as SURPRISE relative to
   the clip's own recent past: a robust z-score against the median/MAD of the previous
   `baseline_s` seconds (causal, no look-ahead), mapped to [0,1] between z_low and z_high.
   A 2 s warm-up yields no score. A few physically meaningful hazards also fire on their
   absolute value (very fast looming = short time-to-collision).
   Signal = max of its features; combined = 0.7*max(w*s) + 0.3*mean(w*s).

Windows: hysteresis (open at t_on, close below t_off), pre/post padding, 4-8 s clamp
around the peak, overlap merge.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..schemas import CandidateWindow, SignalSample
from .runner import PerceptionLog
from .signals import (
    continuity_signal,
    ego_motion_features,
    optical_density_signal,
    rarity_signal,
    scene_features,
    uncertainty_signal,
    vanish_events,
)

SIGNALS = ("uncertainty", "continuity", "optical_density", "rarity")

# feature name -> (signal it feeds, direction (+1: higher is worse, -1: lower is worse), MAD floor)
FEATURES: dict[str, tuple[str, int, float]] = {
    "u.band_entropy": ("uncertainty", +1, 0.05),
    "u.dropouts": ("uncertainty", +1, 0.5),
    "u.score_std": ("uncertainty", +1, 0.04),
    "c.term_norm": ("continuity", +1, 0.15),
    "c.birth_norm": ("continuity", +1, 0.15),
    "c.reacquisitions": ("continuity", +1, 0.5),
    "c.looming_rate": ("continuity", +1, 0.15),
    "c.ego_looming": ("continuity", +1, 0.1),
    "c.ego_inv_ttc": ("continuity", +1, 0.1),
    "e.pitch_jerk": ("continuity", +1, 0.002),
    "e.yaw_jerk": ("continuity", +1, 0.002),
    "e.scale_jerk": ("continuity", +1, 0.001),
    "e.shake": ("continuity", +1, 0.002),
    "o.luminance": ("optical_density", -1, 0.03),
    "o.saturation": ("optical_density", +1, 0.01),
    "o.sharpness": ("optical_density", -1, 0.04),
    "o.active_tracks": ("optical_density", +1, 2.0),
    "o.collapse": ("optical_density", +1, 0.1),
    # Interaction cues (signals.scene_features). Learned model only until they prove themselves.
    "l.present": ("continuity", +1, 0.5),
    "l.size": ("continuity", +1, 0.02),
    "l.inv_ttc": ("continuity", +1, 0.1),
    "l.closing_accel": ("continuity", +1, 0.15),
    "l.danger": ("continuity", +1, 0.02),
    "x.cut_in": ("continuity", +1, 0.2),
    "x.entry_rate": ("continuity", +1, 0.2),
    "x.vru_near": ("continuity", +1, 0.02),
    "x.n_corridor": ("continuity", +1, 0.5),
    "e.speed": ("continuity", -1, 0.001),
    "e.decel": ("continuity", +1, 0.001),
    "e.pitch_dir": ("continuity", +1, 0.002),
}
# Features the hand-weighted rule scorer ignores (the learned model reads all of FEATURES). A weak cue
# added to the rules as an equal max-pooled vote drags the combined score down (c.term_norm, journal #7).
LEARNED_ONLY = {n for n in FEATURES if n[:2] in ("l.", "x.")} | {"e.speed", "e.decel", "e.pitch_dir"}


def compute_features(log: PerceptionLog, cfg: dict, corpus: np.ndarray | None = None
                     ) -> tuple[list[dict], list[set[int]]]:
    """Raw per-step features. Each dict holds t, frame, rarity and the FEATURES keys."""
    m = cfg["mining"]
    if log.single_frame:
        step, n_steps = 1, len(log.frames)
    else:
        step = max(1, int(round(m["step_s"] * log.fps)))
        n_steps = int(np.ceil(len(log.frames) / step))
    rarity = np.zeros(n_steps)
    if log.step_embeddings is not None and len(log.step_embeddings):
        r = rarity_signal(log.step_embeddings, corpus, m["step_s"], k=m["rarity_k"])
        rarity[: min(len(r), n_steps)] = r[:n_steps]

    vanish = {} if log.single_frame else vanish_events(log.frames, log.fps)
    feats, implicated = [], []
    for k in range(n_steps):
        fr = log.frames[k * step:(k + 1) * step]
        if not fr:
            break
        prev = log.frames[max(0, (k - 4) * step):k * step]
        _, ud, ui = uncertainty_signal(fr, tuple(m["uncertainty_band"]))
        if log.single_frame:
            cd, ci = {}, set()
        else:
            gone = [v for i in range(k * step, k * step + len(fr)) for v in vanish.get(i, [])]
            _, cd, ci = continuity_signal(fr, prev, log.fps, log.width, log.height, m["looming_rate"], gone)
        _, od, _ = optical_density_signal(fr, [] if log.single_frame else prev, m)
        ed = {} if log.single_frame else ego_motion_features(fr, prev, log.width, log.height)
        sd, si = ({}, set()) if log.single_frame else scene_features(fr, prev, log.fps, log.width, log.height)
        f = {"t": float(fr[0].t), "frame": int(fr[0].frame), "rarity": float(rarity[k])}
        f.update({f"u.{a}": float(b) for a, b in ud.items()})
        f.update({f"c.{a}": float(b) for a, b in cd.items()})
        f.update({f"o.{a}": float(b) for a, b in od.items()})
        f.update({f"e.{a}": float(b) for a, b in ed.items()})
        f.update({k: float(v) for k, v in sd.items()})
        feats.append(f)
        implicated.append(ui | ci | si)
    return feats, implicated


def _surprise(x: float, hist: np.ndarray, direction: int, floor: float, z_low: float, z_high: float) -> float:
    med = float(np.median(hist))
    mad = float(np.median(np.abs(hist - med))) * 1.4826
    z = direction * (x - med) / max(mad, floor)
    return float(min(1.0, max(0.0, (z - z_low) / (z_high - z_low))))


def score_features(feats: list[dict], cfg: dict, single_frame: bool = False) -> list[SignalSample]:
    m = cfg["mining"]
    w = m["weights"]
    step_s = m["step_s"]
    base_n = max(2, int(round(m.get("baseline_s", 10.0) / step_s)))
    warm_n = int(round(m.get("warmup_s", 2.0) / step_s))
    gap = max(0, int(round(m.get("baseline_gap_s", 0.5) / step_s)))
    z_low, z_high = m.get("z_low", 2.5), m.get("z_high", 6.0)
    loom_abs = m.get("looming_abs", 1.2)  # relative area growth / s; ~TTC < 1.7 s
    names = [n for n in FEATURES if n not in LEARNED_ONLY]
    X = np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=float) if feats else np.zeros((0, len(names)))

    samples = []
    for k, f in enumerate(feats):
        vals = dict.fromkeys(SIGNALS, 0.0)
        detail: dict[str, float] = {}
        if single_frame:
            ref = np.delete(X, k, axis=0)  # stills: compare against the rest of the image set
            ok = len(ref) >= 4
        else:
            ref = X[max(0, k - gap - base_n):max(0, k - gap)]
            ok = k >= warm_n and len(ref) >= 2
        if ok:
            fw = m.get("feature_weights", {})
            for j, n in enumerate(names):
                sig, direction, floor = FEATURES[n]
                s = _surprise(X[k, j], ref[:, j], direction, floor, z_low, z_high) * fw.get(n, 1.0)
                detail[f"s.{n}"] = s
                vals[sig] = max(vals[sig], s)
        if not single_frame:  # absolute hazard: fast looming in the ego path regardless of baseline
            lr = f.get("c.ego_looming", 0.0)
            vals["continuity"] = max(vals["continuity"], min(1.0, max(0.0, (lr - loom_abs) / loom_abs)))
        vals["rarity"] = float(f.get("rarity", 0.0)) if ok or single_frame else 0.0
        weighted = [w[s] * vals[s] for s in SIGNALS]
        combined = float(min(1.0, 0.7 * max(weighted) + 0.3 * float(np.mean(weighted))))
        samples.append(SignalSample(t=f["t"], frame=f["frame"], combined=combined, detail=detail, **vals))
    return samples


def slice_windows(samples: list[SignalSample], implicated: list[set[int]], cfg: dict, fps: float,
                  last_frame: int, source_id: str, source: str = "", single_frame: bool = False
                  ) -> list[CandidateWindow]:
    m = cfg["mining"]
    if not samples:
        return []
    w = m["weights"]

    if single_frame:
        out = []
        for s, ids in zip(samples, implicated, strict=False):
            if s.combined >= m["t_on"]:
                dom = max(SIGNALS, key=lambda n: w[n] * getattr(s, n))
                out.append(CandidateWindow(
                    clip_id=f"{source_id}_f{s.frame:06d}", source=source or source_id,
                    start_frame=s.frame, onset_frame=s.frame, peak_frame=s.frame, end_frame=s.frame, fps=fps,
                    peak_score=s.combined, dominant_signal=dom,
                    signal_peaks={n: float(getattr(s, n)) for n in SIGNALS},
                    focus_track_ids=sorted(ids)[:3], single_frame=True))
        return out

    segs: list[tuple[int, int]] = []
    k, n = 0, len(samples)
    while k < n:
        if samples[k].combined >= m["t_on"]:
            k0 = k
            while k + 1 < n and samples[k + 1].combined >= m["t_off"]:
                k += 1
            segs.append((k0, k))
        k += 1

    def _raw(k0: int, k1: int, start_t: float, end_t: float, grp: int | None) -> dict:
        pk = max(range(k0, k1 + 1), key=lambda i: samples[i].combined)
        ids = Counter(i for j in range(k0, k1 + 1) for i in implicated[j]) if implicated else Counter()
        return {"start": start_t, "end": end_t, "onset": samples[k0].t, "peak": samples[pk].t, "pk": pk,
                "ids": ids, "k0": k0, "k1": k1, "grp": grp}

    raw = []
    for g, (k0, k1) in enumerate(segs):
        start_t = max(0.0, samples[k0].t - m["pre_s"])
        end_t = samples[k1].t + m["step_s"] + m["post_s"]
        tile_end = min(end_t, (last_frame + 1) / fps) if fps > 0 else end_t  # tiles stay inside the clip
        n_tiles = int(np.ceil((tile_end - start_t) / m["max_window_s"] - 1e-9))
        if n_tiles > 1 and m.get("split_long_stretches", True):
            end_t = tile_end
            # A stretch longer than one window is tiled into equal windows (each > min_window_s), each
            # around its own peak. Clamping it to one window around the global peak drops the rest of
            # the stretch, and with it events that are not at the peak.
            size = (end_t - start_t) / n_tiles
            for j in range(n_tiles):
                a, b = start_t + j * size, start_t + (j + 1) * size
                idx = [i for i in range(k0, k1 + 1) if a <= samples[i].t < b]
                if idx:
                    raw.append(_raw(idx[0], idx[-1], a, b, g))
            continue
        r = _raw(k0, k1, start_t, end_t, None)
        r["start"], r["end"] = _clamp(start_t, end_t, r["peak"], m["min_window_s"], m["max_window_s"])
        raw.append(r)

    merged: list[dict] = []
    for r in sorted(raw, key=lambda r: r["start"]):
        same_stretch = r["grp"] is not None and merged and merged[-1]["grp"] == r["grp"]
        if merged and r["start"] <= merged[-1]["end"] and not same_stretch:
            a = merged[-1]
            better = r if samples[r["pk"]].combined > samples[a["pk"]].combined else a
            a.update(start=min(a["start"], r["start"]), end=max(a["end"], r["end"]),
                     onset=min(a["onset"], r["onset"]), pk=better["pk"], peak=better["peak"],
                     ids=a["ids"] + r["ids"], k1=max(a["k1"], r["k1"]))
            a["start"], a["end"] = _clamp(a["start"], a["end"], a["peak"], m["min_window_s"], m["max_window_s"],
                                          keep_onset=a["onset"])
        else:
            merged.append(r)

    out = []
    to_f = lambda t: int(min(last_frame, max(0, round(t * fps))))  # noqa: E731
    for i, r in enumerate(merged):
        segment = samples[r["k0"]:r["k1"] + 1]
        peaks = {nme: float(max(getattr(s, nme) for s in segment)) for nme in SIGNALS}
        ps = samples[r["pk"]]
        dom = max(SIGNALS, key=lambda nme: w[nme] * getattr(ps, nme))
        out.append(CandidateWindow(
            clip_id=f"{source_id}_w{i:03d}", source=source or source_id,
            start_frame=to_f(r["start"]), onset_frame=to_f(r["onset"]), peak_frame=to_f(r["peak"]),
            end_frame=to_f(r["end"]), fps=fps, peak_score=ps.combined, dominant_signal=dom,
            signal_peaks=peaks, focus_track_ids=[t for t, _ in r["ids"].most_common(3)]))
    return out


def _clamp(start: float, end: float, peak: float, lo: float, hi: float, keep_onset: float | None = None
           ) -> tuple[float, float]:
    length = end - start
    if length < lo:
        pad = (lo - length) / 2
        start, end = start - pad, end + pad
    elif length > hi:
        start = max(start, peak - hi / 2)
        if keep_onset is not None:
            start = min(start, keep_onset)
        end = start + hi
    if start < 0:
        end, start = end - start, 0.0
    return start, end


def rescore(feats: list[dict], implicated: list[set[int]], cfg: dict, fps: float, last_frame: int,
            source_id: str, source: str = "", single_frame: bool = False
            ) -> tuple[list[SignalSample], list[CandidateWindow]]:
    samples = score_features(feats, cfg, single_frame)
    return samples, slice_windows(samples, implicated, cfg, fps, last_frame, source_id, source, single_frame)


def mine_log(log: PerceptionLog, cfg: dict, corpus: np.ndarray | None = None, source: str = ""
             ) -> tuple[list[SignalSample], list[CandidateWindow]]:
    feats, implicated = compute_features(log, cfg, corpus)
    last = log.frames[-1].frame if log.frames else 0
    return rescore(feats, implicated, cfg, log.fps, last, log.source_id, source, log.single_frame)
