# Run Stage 1 on your Nexar data (Windows PowerShell)

## One-time setup
```
conda create -n sotif python=3.11 -y
conda activate sotif
cd D:\work
Expand-Archive "$HOME\Downloads\sotif-miner-src.zip" -DestinationPath D:\work -Force
cd D:\work\sotif-miner
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"     # must print True
pip install -e ".[detect,dev]"
pytest -q                                                       # 18 passed
```

## 1. Synthetic demo (no GPU needed)
```
sotif-miner demo
```

## 2. Quick real test: 20 positive + 20 negative train clips
```
sotif-miner nexar --root D:\data\nexar --split train --limit 20 --out runs\nexar_train_small
```
First run downloads the RF-DETR weights into D:\cache\torch (TORCH_HOME).

## 3. Full train split (about 1,500 clips; resumable, press Ctrl+C any time)
```
sotif-miner nexar --root D:\data\nexar --split train --out runs\nexar_train
```

## 4. Anticipation test (clips end 0.5-1.5 s before the event)
```
sotif-miner nexar --root D:\data\nexar --split test-public --out runs\nexar_test_public
```

## New feature code without re-running detection
Each mined clip also caches its perception results in `<run>\perception\*.json.gz`:
```
sotif-miner refeature --run runs\nexar_v4 --config configs\nexar.yaml --out runs\nexar_v4b
```

## Tune thresholds in seconds (no detection re-run)
Every mined clip stores its raw per-step features, so new settings only need a re-score:
```
sotif-miner nexar-diagnose --run runs\nexar_train_small          # which signals separate near-misses (AUC)
sotif-miner rescore --run runs\nexar_train_small --config configs\tune.yaml --out runs\nexar_tune1
sotif-miner nexar-eval --run runs\nexar_tune1 --root D:\data\nexar --split train
```

## Re-score without recomputing
```
sotif-miner nexar-eval --run runs\nexar_train --root D:\data\nexar --split train
```

## Your own phone videos
```
sotif-miner mine --input D:\data\own_videos --out runs\own
```

## Faster / slower
* `--stride 3` (default) processes 10 fps from 30 fps video. `--stride 2` = 15 fps, slower, finer.
* Smaller detector: create `configs\fast.yaml` containing
  `detector: {size: small}` and pass `--config configs\fast.yaml`.


## Learned Stage 1 (train on stored features, seconds; no detection re-run)
```
sotif-miner train-stage1 --run runs\nexar_v4 --root D:\data\nexar                # offline mining (+-2 s context)
sotif-miner train-stage1 --run runs\nexar_v4 --root D:\data\nexar --causal       # real-time anticipation variant
sotif-miner rescore --run runs\nexar_v4 --model runs\nexar_v4\stage1_model.joblib  # apply (in place)
sotif-miner nexar-eval --run runs\nexar_v4 --root D:\data\nexar --split train
```
All numbers printed by train-stage1 are clip-grouped cross-validated (no clip scored by a model trained on it).

## Stage 2 + 3: diagnose the mined windows on your GPU (Ollama)
One-time:
```
winget install --id Ollama.Ollama -e
ollama pull qwen3-vl:4b-instruct
pytest -m gpu tests\test_ollama_live.py -s        # live check: valid JSON from the model
```
Run (pass A = evidence + RF-DETR on crops, pass B = VLM; resumable):
```
sotif-miner diagnose --run runs\nexar_v4 --only positive --limit 20
```
Outputs per window in `runs\nexar_v4\evidence\<window>\`: keyframes, zoomed crops, evidence.json, card.json.
All cards: `runs\nexar_v4\cards.jsonl`; totals: `runs\nexar_v4\diagnosis_summary.json`.

Bigger model (slower on 6 GB, partly runs from system RAM): create `configs\vlm8b.yaml` with
`vlm: {ollama: {model: "qwen3-vl:8b-instruct"}}` and add `--config configs\vlm8b.yaml`.

## Stage 4: fleet failure taxonomy (offline, CPU, no GPU)
Cluster the diagnosed cards into ranked, SOTIF-tagged failure modes:
```
sotif-miner taxonomy --runs runs\nexar_v4 --out runs\taxonomy
start runs\taxonomy\clusters.md
```
INCONCLUSIVE cards (Stage-1 false positives) are excluded by default; add --include-inconclusive to keep them.
Outputs: clusters.md (ranked report), clusters.json, assignments.json (clip -> cluster).

## Industry-grade: held-out test, confidence intervals, consolidated report
The train numbers above are cross-validated. The *locked* number is on the untouched test split.

1) Honest train metrics = out-of-fold cross-validation from train-stage1 (NOT in-sample).
   It writes stage1_cv_metrics.json automatically:
```
sotif-miner train-stage1 --run runs\nexar_v4 --root D:\data\nexar
```
   Do NOT report `nexar-eval` on a model-rescored run: that is in-sample and reads ~1.0 recall (overfit).
2) Mine the held-out test-public split once (GPU; smaller than train), then score it with the learned model:
```
sotif-miner nexar --root D:\data\nexar --split test-public --config configs\nexar.yaml --out runs\nexar_test
sotif-miner rescore --run runs\nexar_test --model runs\nexar_v4\stage1_model.joblib
sotif-miner nexar-eval --run runs\nexar_test --root D:\data\nexar --split test-public --bootstrap 2000 --out-metrics runs\test_ci.json
```
3) One consolidated HTML report tying Stage 1/3/4 together, with provenance and limitations:
```
sotif-miner report --run runs\nexar_v4 --taxonomy runs\taxonomy --data-root D:\data\nexar `
    --train-metrics runs\nexar_v4\stage1_cv_metrics.json --test-metrics runs\test_ci.json --out runs\report.html
start runs\report.html
```
`report.html` is self-contained (no internet, no assets) -- the file to present or send.

## Held-out MINING number (the honest locked figure for this system)
The model trained on the first 100+100 train clips. Mine a DISJOINT slice it never saw, score it:
```
sotif-miner nexar --root D:\data\nexar --split train --config configs\nexar.yaml --skip 100 --limit 100 --out runs\nexar_heldout
sotif-miner rescore --run runs\nexar_heldout --model runs\nexar_v4\stage1_model.joblib
sotif-miner nexar-eval --run runs\nexar_heldout --root D:\data\nexar --split train --bootstrap 2000 --metrics-kind held_out --out-metrics runs\heldout_ci.json
sotif-miner report --run runs\nexar_v4 --taxonomy runs\taxonomy --data-root D:\data\nexar `
    --train-metrics runs\nexar_v4\stage1_cv_metrics.json --heldout-metrics runs\heldout_ci.json `
    --test-metrics runs\test_ci.json --out runs\report.html
```
Note: the Nexar `test-public` split (`--test-metrics`) is an ANTICIPATION benchmark (clips end before the event);
near-chance AP there is expected and does not reflect mining quality. The held-out MINING recall is the real number.

## Large held-out test (mine the rest of train, score with the 200-clip model)
Mine every remaining train clip into the same run (about 14 h on a 3050; resumable):
```
sotif-miner nexar --root D:\data\nexar --split train --config configs\nexar.yaml --skip 200 --out runs\nexar_v4
```
Ignore the summary it prints: it mixes in-sample learned scores (first 200) with default-scorer scores.
Copy ONLY the newly mined clips into their own folder (by write time; must count 1100), then score:
```
New-Item -ItemType Directory runs\nexar_big_heldout\clips
Get-ChildItem runs\nexar_v4\clips\*.json | Where-Object { $_.LastWriteTime -gt [datetime]'<start of the --skip 200 run>' } | Copy-Item -Destination runs\nexar_big_heldout\clips
(Get-ChildItem runs\nexar_big_heldout\clips\*.json).Count
sotif-miner rescore --run runs\nexar_big_heldout --model runs\nexar_v4\stage1_model.joblib
sotif-miner nexar-eval --run runs\nexar_big_heldout --root D:\data\nexar --split train --bootstrap 2000 --metrics-kind held_out --out-metrics runs\big_heldout_ci.json
```
Do this BEFORE retraining train-stage1 on nexar_v4: a model retrained on these clips can no longer be tested on them.

## Retrain Stage 1 on all mined clips (after the large held-out test)
Back up the 200-clip model first (training overwrites it), retrain (minutes), then test on the 200
clips that were never used for training (`runs\nexar_heldout`, clips 100–199 per label):
```
mkdir runs\baseline_200
copy runs\nexar_v4\stage1_model.joblib runs\baseline_200\
copy runs\nexar_v4\stage1_cv_metrics.json runs\baseline_200\
sotif-miner train-stage1 --run runs\nexar_v4 --root D:\data\nexar
sotif-miner rescore --run runs\nexar_heldout --model runs\nexar_v4\stage1_model.joblib
sotif-miner nexar-eval --run runs\nexar_heldout --root D:\data\nexar --split train --bootstrap 2000 --metrics-kind held_out --out-metrics runs\heldout_1300_ci.json
```
Do not evaluate the retrained model on `runs\nexar_big_heldout`: it has now trained on those clips.

## Re-feature after a feature fix (no detection re-run; CPU)
Every mined clip caches its detections/tracks in `perception\*.json.gz`, so fixed or new feature code only
needs `refeature` (writes a NEW run folder; the old one stays untouched), then retraining:
```
mkdir runs\model_1300
copy runs\nexar_v4\stage1_model.joblib runs\model_1300\
copy runs\nexar_v4\stage1_cv_metrics.json runs\model_1300\
sotif-miner refeature --run runs\nexar_v4 --config configs\nexar.yaml --out runs\nexar_v5
sotif-miner nexar-diagnose --run runs\nexar_v5
sotif-miner train-stage1 --run runs\nexar_v5 --root D:\data\nexar
```
Compare the `train-stage1` output with `runs\model_1300\stage1_cv_metrics.json` (step AUC 0.851 before).
Set `mining: {split_long_stretches: false}` in a config to reproduce the old single-window slicer.

## Phase 4 — closed loop (fine-tune the detector on mined failures, prove it helps)
Label-free (runs with what you have). The honest proof is re-mining held-out clips, not pseudo-label AP.
```
# 1) build a COCO co-training dataset from the top failure clusters + core replay frames
sotif-miner export-manifest --mine-run runs\nexar_v4 --taxonomy runs\taxonomy --diagnose-run runs\nexar_v4 --out datasets\cotrain_v1

# 2) fine-tune RF-DETR-nano (fits ~6 GB; raise --size on a bigger GPU)
sotif-miner finetune --dataset datasets\cotrain_v1 --out checkpoints\ft_v1 --size nano --epochs 20 --batch-size 2 --grad-accum 8

# 3) re-mine the SAME held-out clips with the fine-tuned detector
#    (make configs\ft.yaml = nexar.yaml but detector.weights: "checkpoints\ft_v1\...best.pth",
#     detector.class_names: [the ontology class list])
sotif-miner nexar --root D:\data\nexar --split train --skip 100 --limit 100 --config configs\ft.yaml --out runs\nexar_heldout_ft

# 4) paired before/after comparison with CIs + anti-forgetting gate
sotif-miner eval-closed-loop --before runs\nexar_heldout --after runs\nexar_heldout_ft --root D:\data\nexar --out-metrics runs\closed_loop.json
```
For the REAL (labelled) version: download CODA, point export-manifest at it (CODA adapter), then steps 2–4 identical.
