# SOTIF-Aligned Dashcam Edge-Case Miner & Causal Root-Cause Agent

> Turn raw monocular dashcam video into a **ranked, explained, safety-standard-tagged** catalog of the
> driving situations where perception is most likely to fail â€” the input an ADAS/AV team needs to
> decide what to fix next.

![python](https://img.shields.io/badge/python-3.11-blue)
[![CI](https://github.com/madhureddy25/SOTIF-Aligned/actions/workflows/ci.yml/badge.svg)](https://github.com/madhureddy25/SOTIF-Aligned/actions/workflows/ci.yml)
![license](https://img.shields.io/badge/license-Apache--2.0-informational)
![detector](https://img.shields.io/badge/detector-RF--DETR%20(Apache--2.0)-success)
![standard](https://img.shields.io/badge/aligned-ISO%2021448%20SOTIF-orange)

A fleet records thousands of hours of dashcam video. The valuable 1% â€” the near-misses, the
occluded pedestrian, the car that braked hard ahead â€” is buried in it. This system **finds those
moments automatically, explains *why* each was dangerous, and groups them into recurring failure
modes** mapped to the ISO 21448 (SOTIF) safety standard. It runs on a single laptop GPU and every
headline number is reported with a confidence interval and an honest limitations section.

---

## Headline result (real Nexar footage, held-out clips the model never saw)

| Metric | Value | 95% CI |
|---|---|---|
| **Near-miss recall** | **0.91** | [0.85, 0.96] |
| Flagged **before** the official alert time | 0.73 | â€” |
| Median early-warning lead | 3.3 s | â€” |
| Video discarded (compression) | 0.67 | [0.65, 0.69] |
| False windows / minute (normal driving) | 3.1 | [2.8, 3.4] |
| VLM root-cause JSON validity | 99.8% | â€” |

Cross-validation predicted recall 0.90; the held-out test delivered 0.91 â€” **no overfitting.**
141 events were diagnosed and consolidated into **21 ranked failure modes**.

---

## High-level architecture

<p align="center"><img src="docs/architecture.png" alt="SOTIF miner architecture" width="100%"></p>

The five stages are **decoupled** and trade throughput for depth: cheap detection runs on every
frame; the expensive vision-language reasoning runs **only** on verified anomaly windows.

---

## Five things that make it different

1. **Zero-noise evidence gating.** The detector/tracker act as a *trigger and cropper*, not as prompt
   filler. The VLM sees semantically-downsampled frames + zoomed ROI crops â€” avoiding the documented
   "pipe raw detections into the prompt and inject noise" failure mode.
2. **Counterfactual optical-vs-model blame.** A second detection + VLM pass on the *enhanced crop*
   separates **aleatory** optical impossibility (glare/darkness â†’ an operating-domain limit) from
   **epistemic** model blind spots (fixable with training data).
3. **Deterministic, auditable attribution.** The VLM never chooses the root cause â€” fixed rules over
   measurements do, and a critic rejects any card that contradicts the evidence. Reproducible by design.
4. **Honest evaluation.** Cross-validated *and* held-out numbers, each with bootstrap CIs; in-sample
   results are flagged, not reported; the Nexar anticipation benchmark is shown only as a labelled
   aside (it measures a different task).
5. **Anti-forgetting closed loop.** Mined failures are exported as a co-training manifest (corner cases
   mixed with core-replay frames) so detector fine-tuning raises corner-case recall without regressing
   on normal driving.

---

## Pipeline at a glance

| Stage | Does | Key tech | Status |
|---|---|---|---|
| â‘  Mining | Flag 4â€“8 s edge-case windows, discard routine video | RF-DETR Â· CMC-SORT Â· gradient boosting | âœ… validated (held-out recall 0.91) |
| â‘¡ Evidence | Keyframes + zoomed ROI crops + telemetry | OpenCV Â· CLAHE | âœ… |
| â‘¢ Diagnosis | Describe + attribute root cause, critic-checked | Qwen3-VL (local) Â· deterministic rules | âœ… 99.8% valid JSON |
| â‘£ Taxonomy | Cluster into ranked SOTIF-tagged failure modes | UMAP/PCA Â· HDBSCAN | âœ… 21 modes |
| â‘¤ Remediation | Co-training manifest â†’ fine-tune â†’ prove gains | COCO export Â· RF-DETR train | âš™ï¸ machinery built (label-free + CODA-ready) |

---

## Quickstart

```bash
conda create -n sotif python=3.11 -y && conda activate sotif
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[detect,dev]"
pytest -q                     # 73 passing
sotif-miner demo              # synthetic end-to-end, CPU, no weights/data
```

Run on real video (needs the [Nexar dataset](https://huggingface.co/datasets/empiricearth/nexar_collision_prediction)):

```bash
sotif-miner nexar    --root <data> --split train --config configs/nexar.yaml --out runs/train
sotif-miner diagnose --run runs/train --only positive          # local VLM via Ollama
sotif-miner taxonomy --runs runs/train --out runs/taxonomy
sotif-miner report   --run runs/train --taxonomy runs/taxonomy --out runs/report.html
```

Full step-by-step (Windows, including the held-out protocol and Phase 4): [`RUN_ON_WINDOWS.md`](RUN_ON_WINDOWS.md).

- ðŸ“‹ **[What we built â€” project journal](docs/WHAT_WE_DID.md)** â€” everything done, and the honest failures we caught and fixed.
- ðŸ—‚ï¸ **[Full build plan](docs/PLAN.md)**

---

## Design notes & honesty

- **Licensing:** RF-DETR (Apache-2.0) over AGPL YOLO; local VLMs via Ollama; nothing copyleft in the path.
- **What is validated:** mining recall *generalizes* (held-out â‰ˆ cross-validated). Diagnosis is
  reproducible and critic-checked. Taxonomy consolidates real recurring modes.
- **What is *not* (and the report says so):** perception-failure causes aren't independently validated
  (Nexar has no detector-failure labels â€” that needs CARLA counterfactuals or human review); the
  domain is US/clear-day-heavy; this **mines** events in recorded video, it does **not predict** crashes
  before they are visible.

---

## Repository layout

```
src/sotif_miner/
  io/          video & frame readers
  datasets/    Nexar loader (+ held-out slicing)
  perception/  RF-DETR detector Â· CMC-SORT tracker Â· embedders
  mining/      4 signals Â· rolling scorer Â· learned Stage-1 model
  evidence/    keyframe compression Â· ROI crops Â· bundles
  vlm/         Ollama / vLLM / Claude / scripted backends Â· prompts
  diagnosis/   deterministic attribution Â· critic Â· 4-step agent
  taxonomy/    clustering Â· naming Â· Area 2/3
  remediation/ COCO manifest Â· fine-tune wrapper
  eval/        mining Â· bootstrap CIs Â· closed-loop
  report.py    self-contained HTML deliverable
  cli.py       mine Â· diagnose Â· taxonomy Â· train-stage1 Â· report Â· export-manifest Â· finetune Â· eval-closed-loop
tests/         73 CPU tests (gpu/network tests opt-in)
```

---

*Built iteratively and validated on real footage, correcting several honest failures along the way
(signal saturation on real video, in-sample overfitting, and an evaluation that measured the wrong
task) â€” each caught, fixed, and documented. Aligned with ISO 21448 (SOTIF) and Euro NCAP 2026 scenarios.*
