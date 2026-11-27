"""Learned Stage 1 scorer: gradient boosting over the stored per-step features.

Hand-set thresholds over ~25 weak signals cannot find a good operating point (the combined score
is too coarse). This model learns the combination from labelled clips (Nexar near-miss times).

Inputs per 0.5 s step
  * each feature's signed robust z-score vs. the clip's own previous 10 s (same baseline as the rule
    scorer, but unclipped), plus its raw value
  * temporal context: z-scores at step offsets. Offline mining (recorded video) may look ahead:
    offsets -4,-2,0,+2,+4 (+-2 s). `causal=True` uses only -4,-2,0 (real-time anticipation).

Labels
  * positive clips: 1 for t in [alert - pre_s, event + post_s]; steps just before/after that band are
    ambiguous and excluded; the rest of the clip is ordinary driving -> 0
  * negative clips: all steps after warm-up -> 0

Evaluation is grouped by clip (GroupKFold): no clip is ever both trained on and scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import SignalSample
from .scorer import FEATURES, score_features

NAMES = list(FEATURES)
RAW_EXTRA = ["rarity"]
# Feature list before the interaction cues were added: models saved without `names` were trained on it.
LEGACY_NAMES = [n for n in NAMES if not (n[:2] in ("l.", "x.") or n in ("e.speed", "e.decel", "e.pitch_dir"))]


@dataclass
class Stage1Model:
    model: object
    offsets: list[int]
    causal: bool
    threshold: float
    n_features: int
    cv_report: dict | None = None
    names: list[str] | None = None  # feature list it was trained on; None = LEGACY_NAMES (older models)

    def predict_steps(self, feats: list[dict], cfg: dict) -> np.ndarray:
        X = design_matrix(feats, cfg, self.offsets, self.names or LEGACY_NAMES)
        return self.model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)


def zscores(feats: list[dict], cfg: dict, names: list[str] | None = None) -> np.ndarray:
    """Signed robust z per feature vs. the previous `baseline_s` (NaN during warm-up)."""
    m = cfg["mining"]
    step_s = m["step_s"]
    base_n = max(2, int(round(m.get("baseline_s", 10.0) / step_s)))
    warm_n = int(round(m.get("warmup_s", 2.0) / step_s))
    gap = max(0, int(round(m.get("baseline_gap_s", 0.5) / step_s)))
    names = names or NAMES
    X = np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=float)
    Z = np.full_like(X, np.nan)
    for k in range(len(X)):
        ref = X[max(0, k - gap - base_n):max(0, k - gap)]
        if k < warm_n or len(ref) < 2:
            continue
        med = np.median(ref, axis=0)
        mad = np.median(np.abs(ref - med), axis=0) * 1.4826
        floors = np.array([FEATURES[n][2] for n in names])
        dirs = np.array([FEATURES[n][1] for n in names])
        Z[k] = dirs * (X[k] - med) / np.maximum(mad, floors)
    return np.clip(Z, -20, 20)


def design_matrix(feats: list[dict], cfg: dict, offsets: list[int], names: list[str] | None = None) -> np.ndarray:
    if not feats:
        return np.zeros((0, 0))
    names = names or NAMES
    Z = zscores(feats, cfg, names)
    raw = np.array([[f.get(n, 0.0) for n in names + RAW_EXTRA] for f in feats], dtype=float)
    n = len(Z)
    blocks = []
    for off in offsets:
        sh = np.full_like(Z, np.nan)
        if off >= 0:
            sh[: n - off] = Z[off:] if off else Z
        else:
            sh[-off:] = Z[: n + off]
        blocks.append(sh)
    return np.hstack(blocks + [raw])


def step_labels(feats: list[dict], positive: bool, event_s: float | None, alert_s: float | None,
                pre_s: float = 1.5, post_s: float = 0.5, warmup_s: float = 2.0, margin_s: float = 2.5) -> np.ndarray:
    """1 = event band, 0 = ordinary driving, -1 = excluded (warm-up or ambiguous margin)."""
    t = np.array([f["t"] for f in feats])
    y = np.zeros(len(t), dtype=int)
    y[t < warmup_s] = -1
    if positive and event_s is not None:
        al = alert_s if alert_s is not None else event_s
        lo, hi = al - pre_s, event_s + post_s
        y[(t >= lo - margin_s) & (t < lo)] = -1
        y[(t > hi) & (t <= hi + margin_s)] = -1
        y[(t >= lo) & (t <= hi)] = 1
    return y


def _new_model(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def _balanced_fit(model, X: np.ndarray, y: np.ndarray):
    """Balance classes by repeating minority-class rows. Same effect as class_weight='balanced'
    / sample weights, which take a ~60x slower code path in scikit-learn 1.9's histogram GBM."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) and len(neg):
        small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
        reps = max(1, int(round(len(big) / len(small))))
        idx = np.concatenate([big, np.tile(small, reps)])
        X, y = X[idx], y[idx]
    return model.fit(X, y)


def build_dataset(recs: list[dict], cfg: dict, offsets: list[int], names: list[str] | None = None):
    Xs, ys, groups = [], [], []
    for gi, r in enumerate(recs):
        feats = r.get("features") or []
        if not feats:
            continue
        X = design_matrix(feats, cfg, offsets, names)
        y = step_labels(feats, bool(r.get("positive")), r.get("event_s"), r.get("alert_s"))
        keep = y >= 0
        Xs.append(X[keep])
        ys.append(y[keep])
        groups.append(np.full(keep.sum(), gi))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(groups)


def offsets_for(causal: bool) -> list[int]:
    return [-4, -2, 0] if causal else [-4, -2, 0, 2, 4]


def cross_validate(recs: list[dict], cfg: dict, causal: bool = False, folds: int = 5, seed: int = 0,
                   names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Out-of-fold step probabilities per clip_id (every clip scored by a model that never saw it).
    Folds are assigned per clip (positives and negatives interleaved), so grouping is by clip."""
    offsets = offsets_for(causal)
    recs = [r for r in recs if r.get("features")]
    clip_idx = np.arange(len(recs))
    strata = np.array([1 if r.get("positive") else 0 for r in recs])
    # interleave positives/negatives so every fold has both
    order = np.concatenate([clip_idx[strata == 1], clip_idx[strata == 0]])
    fold_of = np.empty(len(recs), dtype=int)
    fold_of[order] = np.arange(len(order)) % folds
    oof: dict[str, np.ndarray] = {}
    for f in range(folds):
        train = [recs[i] for i in clip_idx if fold_of[i] != f]
        test = [recs[i] for i in clip_idx if fold_of[i] == f]
        X, y, _ = build_dataset(train, cfg, offsets, names)
        if len(np.unique(y)) < 2:
            raise ValueError("training fold has a single class; need positive and negative clips")
        m = _balanced_fit(_new_model(seed), X, y)
        for r in test:
            oof[r["clip_id"]] = m.predict_proba(design_matrix(r["features"], cfg, offsets, names))[:, 1]
    return oof


def fit(recs: list[dict], cfg: dict, causal: bool = False, threshold: float = 0.5, seed: int = 0,
        names: list[str] | None = None) -> Stage1Model:
    offsets = offsets_for(causal)
    names = list(names or NAMES)
    X, y, _ = build_dataset(recs, cfg, offsets, names)
    return Stage1Model(_balanced_fit(_new_model(seed), X, y), offsets, causal, threshold, X.shape[1], names=names)


def learned_samples(feats: list[dict], probs: np.ndarray, cfg: dict) -> list[SignalSample]:
    """Rule-based per-signal scores (for explanation) with the learned probability as `combined`."""
    samples = score_features(feats, cfg)
    for s, p in zip(samples, probs, strict=False):
        s.combined = float(p)
    return samples
