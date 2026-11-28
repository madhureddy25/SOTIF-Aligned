# autoresearch: the Karpathy loop for SOTIF-Aligned Stage 1

You run an autonomous research loop that tries to improve the learned Stage-1 hazard scorer. Each round you change
**one file**, run **one fixed experiment**, read **one verdict**, and **keep or discard** the change with git.
The referee (`sotif-miner autoresearch-eval`) is fixed: it measures, decides and logs. You never overrule it.

## Paths (Windows; forward slashes work in Git Bash and PowerShell)
- Training run (features):   `D:/work/sotif-miner/runs/nexar_v6`
- Perception caches:         `D:/work/sotif-miner/runs/nexar_v4`
- Nexar root:                `D:/data/nexar`
- Held-out run (LOCKED):     `D:/work/sotif-miner/runs/nexar_heldout` - never read, list, open or evaluate on it.
- Loop state (git-ignored):  `autoresearch/state/` - results.tsv, best.json, journal.md, run.log

The experiment command (call it EVAL):
```
sotif-miner autoresearch-eval --run D:/work/sotif-miner/runs/nexar_v6 --perception D:/work/sotif-miner/runs/nexar_v4 --root D:/data/nexar --heldout D:/work/sotif-miner/runs/nexar_heldout --desc "<one line>" > autoresearch/state/run.log 2>&1
```
Then read the last ~15 lines of `autoresearch/state/run.log`; the final line is `VERDICT: KEEP|DISCARD|CRASH (reason)`.

## What you may change
- **Only** `src/sotif_miner/mining/experimental.py`: `EXP_FEATURES` and `extra_features(...)`. Read its docstring.
- Everything else is fixed: the harness, tests, metrics, thresholds, seeds, data, configs, other source files.
- Cues must be causal-agnostic (offline mining may use the step's own frames and the ~2 s before it, as given),
  deterministic, fast, and use no I/O. Names start with `z.`.

## What is measured (the referee's rules)
- Primary: `auc_mean` (step-level out-of-fold AUC over 3 fold/seed draws), noise = `auc_std`.
- KEEP only if the gain beats max(0.002, 2 x std), false windows/min do not rise by > 0.15, within-clip AUC
  (the confound guardrail) does not fall by > 0.01, recall 0.9 is still reachable, and the guard tests pass.
- Simplicity: an equal score with less code is a win; removing a cue that does nothing is a valid experiment.

## Setup (once)
1. Ask the human for a run tag (e.g. `nov29`); `git checkout -b autoresearch/<tag>`; `mkdir -p autoresearch/state`.
2. Read: this file, `experimental.py`, `mining/signals.py` (read-only, for helpers), `docs/WHAT_WE_DID.md`
   (what was already tried: shortcuts, data scaling, the interaction cues), `autoresearch/state/results.tsv` if any.
3. Run EVAL once on the unmodified file with `--desc "baseline"`: the referee keeps it as the baseline.

## The loop (one round)
1. **analyst** (at the start and every 5 rounds): evidence about what the current model gets wrong.
2. **planner**: one hypothesis = one small change, not a repeat of anything in results.tsv.
3. **feature-engineer**: implements it in `experimental.py` only.
4. `git add src/sotif_miner/mining/experimental.py && git commit -m "exp: <desc>"`
5. **runner**: runs EVAL with the same `<desc>` and returns the metric block and the VERDICT line.
6. **auditor**: checks the round (diff touches only experimental.py, no held-out access, no I/O, verdict present).
   A veto counts as DISCARD.
7. **evaluator**: KEEP -> leave the commit; DISCARD/CRASH/veto -> `git reset --hard HEAD~1`.
8. **scribe**: one entry in `autoresearch/state/journal.md` (hypothesis, numbers, verdict, what it taught us).

A CRASH from your own bug (typo, shape error) may be fixed once and re-run; otherwise it is a discard.

## Stopping
- Stop after **30 rounds**, or after **8 discards in a row** (ask the planner for one pivot first; if the pivot
  also fails, stop). Do not stop early otherwise and do not ask the human whether to continue.
- At the end the **scribe** writes a summary (kept changes, best numbers vs. baseline, dead ends) to journal.md.
- The held-out evaluation is done **once, by the human**, with the commands in RUN_ON_WINDOWS.md
  ("autoresearch: final held-out check"). Never run it yourself.

## Disclosed information flow
The analyst may read the aggregated cause shares in `D:/work/sotif-miner/runs/nexar_heldout_v6_drop/false_alarm_audit.json`
(Step 1, computed on held-out clips). Only aggregate shares may be used, never clip-level rows; the scribe records
in the summary that this informed the hypotheses, so the final held-out result can be read with that in mind.
