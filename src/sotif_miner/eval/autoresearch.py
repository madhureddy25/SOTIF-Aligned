"""Fixed referee for the autoresearch (Karpathy) loop. The loop agent edits only `mining/experimental.py`.

One experiment = compute the experimental cues from cached perception logs, train the learned Stage 1 (drop-markers
feature set + experimental cues) with 5-fold CV over the training clips for several fold/seed draws, and report:

* auc_mean / auc_std  - step-level out-of-fold AUC across seeds (the primary metric; std = run-to-run noise);
* fw_per_min, recall, early_recall, compression at the threshold with the most video discarded and recall >= 0.9;
* within_clip_auc     - the confound guardrail (event moment vs. the same clip's quiet moments).

`decide` keeps a change only if AUC beats the best kept result by more than the noise, false windows do not rise
and the within-clip AUC does not fall. The 200 held-out clips are refused outright; they are scored once, at the end.
"""

from __future__ import annotations

import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _exp_for_clip(args: tuple[str, str, dict]) -> tuple[str, list[dict]]:
    """Worker: experimental cues per step of one clip (top-level so Windows' spawn can pickle it)."""
    from ..mining.runner import load_log
    from ..mining.scorer import experimental_step, iter_steps

    clip_id, path, cfg = args
    log = load_log(path)
    return clip_id, [experimental_step(fr, prev, log) for _, fr, prev in iter_steps(log, cfg)]


def add_experimental_features(recs: list[dict], perception: Path, cfg: dict, workers: int = 0) -> int:
    """Merge the current experimental cues into each record's stored features (in place). Returns #clips."""
    from ..mining.experimental import EXP_FEATURES

    if not EXP_FEATURES:
        return 0
    jobs = [(r["clip_id"], str(perception / f"{r['clip_id']}.json.gz"), cfg) for r in recs]
    missing = [j[1] for j in jobs if not Path(j[1]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} perception caches missing, e.g. {missing[0]}")
    if workers == 1:
        results = map(_exp_for_clip, jobs)
    else:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(workers or None)
        results = pool.imap_unordered(_exp_for_clip, jobs, chunksize=4)
    by_id = {r["clip_id"]: r for r in recs}
    n = 0
    for cid, steps in results:
        feats = by_id[cid]["features"]
        if len(steps) != len(feats):
            raise ValueError(f"{cid}: {len(steps)} experimental steps vs {len(feats)} stored steps")
        for f, x in zip(feats, steps, strict=True):
            f.update(x)
        n += 1
    if workers != 1:
        pool.close()
        pool.join()
    return n


def step_auc(recs: list[dict], oof: dict) -> float:
    from ..eval.nexar_eval import _auc
    from ..mining.learned import step_labels

    pe, pn = [], []
    for r in recs:
        y = step_labels(r["features"], bool(r["positive"]), r.get("event_s"), r.get("alert_s"))
        p = oof[r["clip_id"]]
        pe += list(p[y == 1])
        if not r["positive"]:
            pn += list(p[y == 0])
    return float(_auc(np.array(pe), np.array(pn)))


def within_clip_auc(recs: list[dict], oof: dict, pre_s: float = 1.5, post_s: float = 0.5, margin_s: float = 2.5,
                    warmup_s: float = 2.0) -> float:
    """Confound guardrail: model score in the event band vs. the same positive clip's steps far from it."""
    from .confound import stratified_auc

    strata = []
    for r in recs:
        if not r.get("positive") or r.get("event_s") is None:
            continue
        ev = r["event_s"]
        lo = (r["alert_s"] if r.get("alert_s") is not None else ev) - pre_s
        hi = ev + post_s
        t = np.array([f["t"] for f in r["features"]])
        p = oof[r["clip_id"]]
        band = (t >= lo) & (t <= hi) & (t >= warmup_s)
        far = ((t < lo - margin_s) | (t > hi + margin_s)) & (t >= warmup_s)
        strata.append((p[band], p[far]))
    return stratified_auc(strata)[0]


def operating_point(recs: list[dict], oof: dict, cfg: dict, clips: dict, target_recall: float = 0.9) -> dict:
    """Same threshold rule as train-stage1: most video discarded among thresholds with recall >= target."""
    from ..eval.nexar_eval import ClipResult, evaluate_train
    from ..mining.learned import learned_samples
    from ..mining.scorer import slice_windows

    rows = []
    for thr in THRESHOLDS:
        c2 = copy.deepcopy(cfg)
        c2["mining"]["t_on"], c2["mining"]["t_off"] = thr, thr * 0.7
        res = []
        for r in recs:
            s = learned_samples(r["features"], oof[r["clip_id"]], c2)
            w = slice_windows(s, [set(i) for i in r.get("implicated", [])], c2, r["fps"], r.get("last_frame", 0),
                              r["clip_id"])
            if r["clip_id"] in clips:
                res.append(ClipResult(clips[r["clip_id"]], s, w, r["duration_s"]))
        rows.append((thr, evaluate_train(res)))
    ok = [(t, m) for t, m in rows if m["recall"] >= target_recall]
    thr, m = max(ok, key=lambda x: x[1]["compression"] or 0) if ok else max(
        rows, key=lambda x: (x[1]["recall"], x[1]["compression"] or 0))
    return {"threshold": thr, "recall": m["recall"], "early_recall": m["early_recall_before_alert"],
            "fw_per_min": m["negative_windows_per_minute"], "compression": m["compression"],
            "normal_clips_flagged": m["negative_clip_false_alarm_rate"], "target_met": bool(ok)}


def evaluate(run: Path, perception: Path, root: Path, cfg: dict, seeds: tuple[int, ...] = (0, 1, 2),
             folds: int = 5, workers: int = 0, heldout: Path | None = None, log=print) -> dict:
    from ..datasets.nexar import load_nexar
    from ..mining.experimental import EXP_FEATURES
    from ..mining.learned import cross_validate, select_names

    t0 = time.time()
    recs = [json.loads(f.read_text()) for f in sorted((run / "clips").glob("*.json"))]
    recs = [r for r in recs if r.get("features") and "positive" in r]
    if heldout is not None and (heldout / "clips").exists():
        locked = {f.stem for f in (heldout / "clips").glob("*.json")}
        leak = locked & {r["clip_id"] for r in recs}
        if leak:
            raise RuntimeError(f"{len(leak)} held-out clips found in the training run (e.g. {sorted(leak)[0]})")
    n_exp = add_experimental_features(recs, perception, cfg, workers)
    log(f"{len(recs)} clips; experimental cues {sorted(EXP_FEATURES) or 'none'} on {n_exp} clips "
        f"({time.time() - t0:.0f} s)")
    names = select_names("markers")
    clips = {c.clip_id: c for c in load_nexar(root, "train")}
    aucs, oof0 = [], None
    for s in seeds:
        oof = cross_validate(recs, cfg, False, folds, seed=s, names=names)
        aucs.append(step_auc(recs, oof))
        oof0 = oof0 or oof
        log(f"  seed {s}: step AUC {aucs[-1]:.4f} ({time.time() - t0:.0f} s)")
    op = operating_point(recs, oof0, cfg, clips)
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "aucs": aucs, **op, "within_clip_auc": within_clip_auc(recs, oof0), "n_features": len(names),
            "experimental": sorted(EXP_FEATURES), "n_clips": len(recs), "seconds": round(time.time() - t0, 1)}


def decide(best: dict | None, new: dict, min_gain: float = 0.002, fw_slack: float = 0.15,
           within_slack: float = 0.01) -> tuple[str, str]:
    """KEEP or DISCARD with the reason. The first result is the baseline and is always kept."""
    if best is None:
        return "KEEP", "baseline"
    noise = max(min_gain, 2 * max(best.get("auc_std", 0.0), new.get("auc_std", 0.0)))
    gain = new["auc_mean"] - best["auc_mean"]
    if not new.get("target_met", True):
        return "DISCARD", "recall target 0.9 not reached at any threshold"
    if gain <= noise:
        return "DISCARD", f"AUC gain {gain:+.4f} within noise ({noise:.4f})"
    if new["fw_per_min"] > best["fw_per_min"] + fw_slack:
        return "DISCARD", f"false windows/min rose {best['fw_per_min']:.2f} -> {new['fw_per_min']:.2f}"
    if new["within_clip_auc"] < best["within_clip_auc"] - within_slack:
        return "DISCARD", (f"within-clip AUC fell {best['within_clip_auc']:.3f} -> {new['within_clip_auc']:.3f} "
                           "(gain likely from clip-level shortcuts)")
    return "KEEP", f"AUC {gain:+.4f} > noise {noise:.4f}; fw/min {new['fw_per_min']:.2f}; within {new['within_clip_auc']:.3f}"


def run_tests(repo: Path) -> tuple[bool, str]:
    """The fast guard suite; a failing experimental cue must not be kept."""
    import sys

    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "tests/test_mining.py", "tests/test_scene.py",
           "tests/test_learned.py", "tests/test_autoresearch.py"]
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-1] if (p.stdout + p.stderr).strip() else ""


def git_commit(repo: Path) -> str:
    p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True)
    return p.stdout.strip() or "nogit"


def record(state: Path, commit: str, result: dict | None, verdict: str, reason: str, desc: str) -> None:
    """Append one row to <state>/results.tsv and, on KEEP, update <state>/best.json (state is git-ignored)."""
    state.mkdir(parents=True, exist_ok=True)
    tsv = state / "results.tsv"
    if not tsv.exists():
        tsv.write_text("commit\tauc_mean\tauc_std\tfw_per_min\trecall\twithin_clip\tstatus\treason\tdescription\n")
    r = result or {}
    row = [commit, f"{r.get('auc_mean', 0):.4f}", f"{r.get('auc_std', 0):.4f}", f"{r.get('fw_per_min', 0):.2f}",
           f"{r.get('recall', 0):.2f}", f"{r.get('within_clip_auc', 0):.3f}", verdict.lower(), reason,
           desc.replace("\t", " ")]
    with tsv.open("a", encoding="utf-8") as fh:
        fh.write("\t".join(row) + "\n")
    if verdict == "KEEP" and result is not None:
        (state / "best.json").write_text(json.dumps({**result, "commit": commit, "description": desc}, indent=2))
