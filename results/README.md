# Results snapshot

Small outputs of the full Nexar run that back the numbers in the main README, collected with
`scripts/collect_results.ps1`. Everything large or derived from the Nexar videos stays out of git
(`runs/` is ignored): perception caches, mined clip files, evidence keyframes and crops, videos.

| Folder | Contents |
|---|---|
| `models/` | Stage-1 models (scikit-learn, load with `joblib`). **`stage1_model_drop.joblib` is the recommended model** (interaction cues, no clip-marker inputs; held-out recall 0.92, 2.3 false windows/min). Also: the full-feature model, the legacy-feature ablation and the original 200-clip model. |
| `metrics/cv/` | 5-fold cross-validated metrics with bootstrap CIs, per experiment (`_legacy`, `_drop`, `_zonly`, `_drop_zonly`). |
| `metrics/heldout/` | Held-out and test metrics with 95% CIs (`heldout_*_ci.json`, `big_heldout_ci.json` = 1,100 clips). |
| `metrics/confound_v6.json` | Pooled vs. within-clip / scene- / traffic-matched AUC per feature. |
| `diagnosis/` | Root-cause cards (text, one JSON per line) for 1,142 near-miss windows and 123 held-out false-alarm windows, with run summaries. |
| `taxonomy/` | 56 failure modes from 856 events: ranked report (`clusters.md`), clusters and window-to-cluster assignments. |

Apply a model to a mined run: `sotif-miner rescore --run <run> --model results/models/stage1_model_drop.joblib`
(the run must be re-featured with the current code: `sotif-miner refeature`).
