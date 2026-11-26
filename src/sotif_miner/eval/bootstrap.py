"""Clip-level bootstrap confidence intervals for the headline metrics.

Industry reporting needs an interval, not a point estimate. Resampling is over CLIPS (not steps or
windows), because clips are the independent unit; resampling steps would understate uncertainty.
"""

from __future__ import annotations

import numpy as np

from .nexar_eval import ClipResult, average_precision, evaluate_test, evaluate_train


def _ci(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    a = np.array([s for s in samples if s is not None and not np.isnan(s)])
    if len(a) == 0:
        return (float("nan"), float("nan"))
    return float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2))


def bootstrap_train(results: list[ClipResult], metric_keys: tuple[str, ...], n: int = 2000, seed: int = 0,
                    tol_s: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(results))
    draws: dict[str, list[float]] = {k: [] for k in metric_keys}
    for _ in range(n):
        sample = [results[i] for i in rng.choice(idx, len(idx), replace=True)]
        m = evaluate_train(sample, tol_s)
        for k in metric_keys:
            draws[k].append(m.get(k))
    point = evaluate_train(results, tol_s)
    return {k: {"value": point.get(k), "ci95": _ci(draws[k])} for k in metric_keys}


def bootstrap_test_ap(results: list[ClipResult], n: int = 2000, seed: int = 0, anticipation_s: float = 4.0) -> dict:
    rng = np.random.default_rng(seed)
    y, sc = [], []
    for r in results:
        tail = [s.combined for s in r.samples if s.t >= r.duration_s - anticipation_s]
        y.append(1 if r.clip.positive else 0)
        sc.append(max(tail) if tail else 0.0)
    y, sc = np.array(y), np.array(sc)
    idx = np.arange(len(y))
    aps = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0 or (1 - y[b]).sum() == 0:
            continue
        aps.append(average_precision(y[b], sc[b]))
    point = evaluate_test(results, anticipation_s)
    return {"ap": {"value": point["ap"], "ci95": _ci(aps)}, "n_clips": len(results),
            "n_positive": int(y.sum()), "prevalence": float(y.mean())}
