"""Stage 5 evaluation: did fine-tuning the detector help MINING, without forgetting normal driving?

Honest and label-free: we do NOT measure detection AP against pseudo-labels (circular). We re-mine
the SAME held-out clips with the before- and after- detectors, holding the Stage-1 scorer fixed, and
compare the mining metrics that have real ground truth (Nexar event times):

* recall of near-misses       (want up or equal)
* false windows / minute      (want down or equal)  -- the anti-forgetting signal on normal driving
* compression                 (want up or equal)

Deltas use a PAIRED clip bootstrap (same clips resampled for both detectors), so the CI is on the
change, not on each run separately. The anti-forgetting gate fails the run if false windows/min on
negatives rises by more than `epsilon`.
"""

from __future__ import annotations

import numpy as np

from .nexar_eval import ClipResult, evaluate_train


def _paired(before: list[ClipResult], after: list[ClipResult]) -> tuple[list[ClipResult], list[ClipResult]]:
    bi = {r.clip.clip_id: r for r in before}
    ai = {r.clip.clip_id: r for r in after}
    keys = [k for k in bi if k in ai]
    return [bi[k] for k in keys], [ai[k] for k in keys]


def compare(before: list[ClipResult], after: list[ClipResult], n_boot: int = 2000, seed: int = 0,
            epsilon_fa: float = 0.5) -> dict:
    b, a = _paired(before, after)
    if not b:
        raise ValueError("no overlapping clips between the before and after runs")
    mb, ma = evaluate_train(b), evaluate_train(a)
    keys = ["recall", "negative_windows_per_minute", "compression"]
    point = {k: {"before": mb.get(k), "after": ma.get(k),
                 "delta": (ma.get(k) or 0) - (mb.get(k) or 0)} for k in keys}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(b))
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        bs = [b[i] for i in s]
        as_ = [a[i] for i in s]
        db, da = evaluate_train(bs), evaluate_train(as_)
        for k in keys:
            draws[k].append((da.get(k) or 0) - (db.get(k) or 0))
    for k in keys:
        arr = np.array(draws[k])
        point[k]["delta_ci95"] = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]

    fa_delta = point["negative_windows_per_minute"]["delta"]
    recall_delta = point["recall"]["delta"]
    point["_verdict"] = {
        "n_clips": len(b),
        "recall_improved": recall_delta >= 0 and point["recall"]["delta_ci95"][0] >= -0.02,
        "anti_forgetting_ok": fa_delta <= epsilon_fa,
        "epsilon_fa": epsilon_fa,
        "summary": (f"recall {mb['recall']:.2f} -> {ma['recall']:.2f} ({recall_delta:+.2f}); "
                    f"false windows/min {mb['negative_windows_per_minute']:.2f} -> "
                    f"{ma['negative_windows_per_minute']:.2f} ({fa_delta:+.2f}); "
                    f"compression {mb['compression']:.2f} -> {ma['compression']:.2f}"),
    }
    return point
