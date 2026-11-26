"""Score Stage 1 mining against Nexar near-miss / collision ground truth.

Train split (full ~40 s clips, event visible):
  * hit           : a window overlaps [alert - tol, event + tol] for a positive clip
  * early_hit     : the window's onset is at or before the alert time (the miner anticipated it)
  * lead_time_s   : event - onset for hits with onset <= event
  * false alarm   : any window on a negative clip; windows/min on negatives
  * compression   : fraction of total video duration NOT covered by windows

Test splits (~10 s clips ending 0.5-1.5 s before the event, as in the Nexar challenge):
  * per-clip score = max combined score over the last `anticipation_s` seconds
  * average precision (positive vs negative), overall and per time_to_accident

All metrics are also broken down by weather / light / scene.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..datasets.nexar import NexarClip
from ..schemas import CandidateWindow, SignalSample


@dataclass
class ClipResult:
    clip: NexarClip
    samples: list[SignalSample]
    windows: list[CandidateWindow]
    duration_s: float


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under precision-recall (step-wise), no sklearn dependency."""
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def _win_span(w: CandidateWindow) -> tuple[float, float, float]:
    return w.start_frame / w.fps, w.end_frame / w.fps, w.onset_frame / w.fps


def evaluate_train(results: list[ClipResult], tol_s: float = 1.0) -> dict:
    pos = [r for r in results if r.clip.positive]
    neg = [r for r in results if not r.clip.positive]
    hits, early, leads = 0, 0, []
    per_group: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in pos:
        ev, al = r.clip.event_in_clip_s(), r.clip.alert_in_clip_s()
        if ev is None:
            continue
        al = al if al is not None else ev
        hit = early_hit = False
        best_lead = None
        for w in r.windows:
            s, e, on = _win_span(w)
            if s <= ev + tol_s and e >= al - tol_s:
                hit = True
                if on <= al + 1e-6:
                    early_hit = True
                if on <= ev:
                    best_lead = max(best_lead or 0.0, ev - on)
        hits += hit
        early += early_hit
        if best_lead is not None:
            leads.append(best_lead)
        for key in (f"weather={r.clip.weather}", f"light={r.clip.light}", f"scene={r.clip.scene}"):
            per_group[key]["recall"].append(float(hit))

    neg_with_window = sum(1 for r in neg if r.windows)
    neg_minutes = sum(r.duration_s for r in neg) / 60.0
    neg_windows = sum(len(r.windows) for r in neg)
    for r in neg:
        for key in (f"weather={r.clip.weather}", f"light={r.clip.light}", f"scene={r.clip.scene}"):
            per_group[key]["false_alarm"].append(float(bool(r.windows)))

    total = sum(r.duration_s for r in results)
    covered = sum(_covered_seconds(r) for r in results)
    n_pos = max(len(pos), 1)
    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "recall": hits / n_pos,
        "early_recall_before_alert": early / n_pos,
        "median_lead_time_s": float(np.median(leads)) if leads else None,
        "mean_lead_time_s": float(np.mean(leads)) if leads else None,
        "negative_clip_false_alarm_rate": neg_with_window / max(len(neg), 1),
        "negative_windows_per_minute": neg_windows / neg_minutes if neg_minutes else None,
        "compression": 1.0 - covered / total if total else None,
        "by_group": {k: {m: float(np.mean(v)) for m, v in d.items()} | {"n": max(len(v) for v in d.values())}
                     for k, d in sorted(per_group.items())},
    }


def _covered_seconds(r: ClipResult) -> float:
    spans = sorted((max(0.0, s), min(r.duration_s, e)) for s, e, _ in map(_win_span, r.windows))
    total, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def evaluate_test(results: list[ClipResult], anticipation_s: float = 4.0) -> dict:
    y, sc, tta = [], [], []
    for r in results:
        tail = [s.combined for s in r.samples if s.t >= r.duration_s - anticipation_s]
        y.append(1 if r.clip.positive else 0)
        sc.append(max(tail) if tail else 0.0)
        tta.append(r.clip.time_to_accident)
    y_a, s_a = np.array(y), np.array(sc)
    out = {"n": len(y), "ap": average_precision(y_a, s_a), "by_time_to_accident": {}}
    # Negatives are shared across horizons (as in the challenge); positives are split by horizon.
    for h in sorted({t for t, yy in zip(tta, y, strict=False) if yy == 1 and t is not None}):
        mask = np.array([(yy == 0) or (t == h) for t, yy in zip(tta, y, strict=False)])
        out["by_time_to_accident"][f"{h:.1f}s"] = average_precision(y_a[mask], s_a[mask])
    vals = [v for v in out["by_time_to_accident"].values() if not np.isnan(v)]
    out["mean_ap_over_horizons"] = float(np.mean(vals)) if vals else None
    return out


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score of an event step > score of a normal step)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort(kind="mergesort").argsort(kind="mergesort").astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def feature_separability(recs: list[dict], pre_s: float = 1.5, post_s: float = 0.5, warmup_s: float = 2.0
                         ) -> list[tuple[str, float, float, float]]:
    """Step-level AUC of every stored score and raw feature.

    'event' steps: positive clips, t in [alert - pre_s, event + post_s].
    'normal' steps: all steps of negative clips after warm-up.
    Returned sorted by how far AUC is from 0.5 (either direction is informative)."""
    ev_rows, nm_rows = [], []
    for r in recs:
        samples = {s["frame"]: s for s in r.get("samples", [])}
        rows = []
        for f in r.get("features") or []:
            row = {k: v for k, v in f.items() if k not in ("t", "frame")}
            s = samples.get(f["frame"])
            if s:
                row.update({"score.combined": s["combined"], "score.uncertainty": s["u"], "score.continuity": s["c"],
                            "score.optical": s["o"], "score.rarity": s["r"]})
            rows.append((f["t"], row))
        if r.get("positive") and r.get("event_s") is not None:
            ev = r["event_s"]
            al = r.get("alert_s") if r.get("alert_s") is not None else ev
            ev_rows += [row for t, row in rows if al - pre_s <= t <= ev + post_s]
        elif not r.get("positive"):
            nm_rows += [row for t, row in rows if t >= warmup_s]
    keys = sorted({k for row in ev_rows + nm_rows for k in row})
    out = []
    for k in keys:
        pe = np.array([row[k] for row in ev_rows if k in row], float)
        pn = np.array([row[k] for row in nm_rows if k in row], float)
        if len(pe) and len(pn):
            out.append((k, _auc(pe, pn), float(pe.mean()), float(pn.mean())))
    return sorted(out, key=lambda x: -abs(x[1] - 0.5) if not np.isnan(x[1]) else 0)
