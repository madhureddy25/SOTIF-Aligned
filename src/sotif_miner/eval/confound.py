"""Is a cue timing the event, or marking the kind of clip events happen in?

Pooled step AUC (event-band steps of positive clips vs. steps of negative clips) mixes two things: a cue
that rises *at* the event, and a cue that is simply higher throughout positive clips (denser traffic,
city scenes, a camera mount). Stratified AUCs separate them by only comparing steps within a stratum:

* within-clip: event band vs. the same positive clip's steps far from the event. Everything constant per
  clip (scene, traffic level, weather, camera) cancels; a pure clip marker scores 0.5 here.
* scene-matched: positive vs. negative clips with the same (light, weather, scene) labels.
* traffic-matched: positive vs. negative clips in the same tertile of median active-track count.

Stratified AUC = sum_s AUC_s * n_pos_s * n_neg_s / sum_s n_pos_s * n_neg_s (pairs compared within strata).
"""

from __future__ import annotations

import numpy as np

from .nexar_eval import _auc


def step_rows(rec: dict) -> list[tuple[float, dict]]:
    """(t, {feature: value}) per step, including the stored scores as score.*."""
    samples = {s["frame"]: s for s in rec.get("samples", [])}
    rows = []
    for f in rec.get("features") or []:
        row = {k: v for k, v in f.items() if k not in ("t", "frame")}
        s = samples.get(f["frame"])
        if s:
            row.update({"score.combined": s["combined"], "score.continuity": s["c"], "score.rarity": s["r"]})
        rows.append((f["t"], row))
    return rows


def stratified_auc(strata: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, int]:
    """Pair-weighted mean AUC over strata that hold both classes; also the number of such strata."""
    num = den = 0.0
    used = 0
    for pos, neg in strata:
        if len(pos) and len(neg):
            w = float(len(pos)) * len(neg)
            num += _auc(pos, neg) * w
            den += w
            used += 1
    return (num / den if den else float("nan")), used


def confound_table(recs: list[dict], meta: dict[str, tuple[str, str, str]], pre_s: float = 1.5,
                   post_s: float = 0.5, warmup_s: float = 2.0, margin_s: float = 2.5,
                   keys: list[str] | None = None) -> tuple[list[dict], dict]:
    """Per feature: pooled, within-clip, scene-matched and traffic-matched AUC. `meta`: clip_id ->
    (light, weather, scene). Returns (rows sorted by pooled AUC distance from 0.5, stratum counts)."""
    clips = []
    for r in recs:
        rows = [(t, row) for t, row in step_rows(r) if t >= warmup_s]
        if not rows:
            continue
        pos = bool(r.get("positive"))
        ev, al = r.get("event_s"), r.get("alert_s")
        if pos and ev is None:
            continue
        lo, hi = ((al if al is not None else ev) - pre_s, ev + post_s) if pos else (None, None)
        band = [row for t, row in rows if pos and lo <= t <= hi]
        far = [row for t, row in rows if pos and (t < lo - margin_s or t > hi + margin_s)]
        tracks = float(np.median([row.get("c.active_tracks", 0.0) for _, row in rows]))
        clips.append({"pos": pos, "band": band, "far": far, "all": [row for _, row in rows],
                      "scene": meta.get(r["clip_id"]), "tracks": tracks})
    if keys is None:
        keys = sorted({k for c in clips for row in c["all"] for k in row})
    edges = np.quantile([c["tracks"] for c in clips], [1 / 3, 2 / 3]) if clips else [0, 0]
    for c in clips:
        c["tertile"] = int(np.searchsorted(edges, c["tracks"], side="right"))

    def vals(rows, k):
        return np.array([row[k] for row in rows if k in row], float)

    def between(k, key):
        groups: dict = {}
        for c in clips:
            if key(c) is None:
                continue
            g = groups.setdefault(key(c), ([], []))
            if c["pos"]:
                g[0].append(vals(c["band"], k))
            else:
                g[1].append(vals(c["all"], k))
        return stratified_auc([(np.concatenate(p) if p else np.zeros(0), np.concatenate(n) if n else np.zeros(0))
                               for p, n in groups.values()])

    out = []
    counts = {}
    for k in keys:
        pooled = _auc(np.concatenate([vals(c["band"], k) for c in clips if c["pos"]] or [np.zeros(0)]),
                      np.concatenate([vals(c["all"], k) for c in clips if not c["pos"]] or [np.zeros(0)]))
        within, n_w = stratified_auc([(vals(c["band"], k), vals(c["far"], k)) for c in clips if c["pos"]])
        scene, n_s = between(k, lambda c: c["scene"])
        traffic, n_t = between(k, lambda c: c["tertile"])
        out.append({"feature": k, "pooled": pooled, "within_clip": within, "scene_matched": scene,
                    "traffic_matched": traffic})
        counts = {"within_clip_clips": n_w, "scene_strata": n_s, "traffic_strata": n_t}
    out.sort(key=lambda d: -abs(d["pooled"] - 0.5) if not np.isnan(d["pooled"]) else 0)
    return out, counts
