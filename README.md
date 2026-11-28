# SOTIF-Aligned Dashcam Edge-Case Miner & Causal Root-Cause Agent

> Turn raw monocular dashcam video into a **ranked, explained, safety-standard-tagged** catalog of the
> driving situations where perception is most likely to fail — the input an ADAS/AV team needs to
> decide what to fix next.

![python](https://img.shields.io/badge/python-3.11-blue)
[![CI](https://github.com/madhureddy25/SOTIF-Aligned/actions/workflows/ci.yml/badge.svg)](https://github.com/madhureddy25/SOTIF-Aligned/actions/workflows/ci.yml)
![license](https://img.shields.io/badge/license-Apache--2.0-informational)
![detector](https://img.shields.io/badge/detector-RF--DETR%20(Apache--2.0)-success)
![standard](https://img.shields.io/badge/aligned-ISO%2021448%20SOTIF-orange)

A fleet records thousands of hours of dashcam video. The valuable 1% — the near-misses, the
occluded pedestrian, the car that braked hard ahead — is buried in it. This system **finds those
moments automatically, explains *why* each was dangerous, and groups them into recurring failure
modes** mapped to the ISO 21448 (SOTIF) safety standard. It runs on a single laptop GPU and every
headline number is reported with a confidence interval and an honest limitations section.

---

## Headline result (real Nexar footage)

Current Stage-1 model: motion + **interaction cues**, trained on 1,300 clips.

| Metric | Held-out: 200 clips never seen (95% CI) | 5-fold CV, 1,300 clips |
|---|---|---|
| **Near-miss recall** | **0.92** [0.87, 0.97] | 0.92 |
| Flagged **before** the official alert time | 0.76 [0.68, 0.84] | 0.72 |
| Median early-warning lead | 2.6 s [2.4, 3.2] | 2.6 s |
| **False windows / minute** (normal driving) | **2.1** [1.8, 2.4] | 1.8 |
| **Video discarded** (compression) | **0.75** [0.73, 0.77] | 0.77 |
| Normal clips with ≥ 1 window | 0.78 | 0.72 |
| Step-level AUC | — | **0.917** |
| VLM root-cause JSON validity (3,360 calls) | 99.97% | — |

Interaction cues — the lead vehicle's proximity and closing rate, road users moving toward the ego path
(camera motion removed), signed ego braking — lifted step AUC from **0.850 to 0.917** on the *same*
re-featured clips (ablation: `train-stage1 --legacy-features`). At the same recall they cut false windows
nearly in half and raise the video discarded from 62% to 77%. The price: warnings come ~0.5 s later.
1,142 windows from all 650 near-miss clips were diagnosed; the 856 with a root cause consolidate into
**56 failure modes** (35 recurring, 21 singletons), led by daytime (246) and night (139) closing conflicts with cars.

### How we got here (the same 200 held-out clips throughout)

| Same held-out clips | Trained on 200 | Trained on 1,300 | + Stage-1 fixes | **+ interaction cues** |
|---|---|---|---|---|
| Recall | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.95 [0.90, 0.99] | 0.92 [0.87, 0.97] |
| Flagged before alert | 0.73 | 0.76 [0.67, 0.84] | 0.78 [0.69, 0.86] | 0.76 [0.68, 0.84] |
| Median early-warning lead | 3.3 s | **4.2 s** [3.6, 5.2] | 3.5 s [2.7, 4.5] | 2.6 s [2.4, 3.2] |
| False windows / minute | 3.1 [2.8, 3.4] | 3.4 [3.1, 3.8] | 3.8 [3.4, 4.2] | **2.1** [1.8, 2.4] |
| Video discarded | 0.67 [0.65, 0.69] | 0.63 [0.61, 0.66] | 0.60 [0.58, 0.63] | **0.75** [0.73, 0.77] |
| Step-level AUC (5-fold CV) | 0.855 | 0.851 | 0.850 | **0.917** |

**Finding: Stage 1 was feature-limited, not data-limited.** 6.5× more data and two bug fixes
([journal](docs/WHAT_WE_DID.md) #6–#8) all stayed on one trade-off curve (AUC ≈ 0.85): earlier warnings
or fewer false windows, never both. A new *kind* of signal moved the curve itself (#9): on held-out clips,
false windows fall from 3.8 to 2.1 per minute (non-overlapping CIs) at statistically unchanged recall.

The 200-clip model was also tested on **1,100 further held-out clips** (recall 0.88 [0.86, 0.91], early
recall 0.65), inside its cross-validated intervals — evidence that CV here predicts held-out performance.
The current model trained on those clips, so they can no longer test it.

---

## High-level architecture

<p align="center">
  <img src="docs/architecture.png" alt="SOTIF miner architecture — five-stage pipeline from dashcam video to an HTML report" width="100%">
</p>

The five stages are **decoupled** and trade throughput for depth: cheap detection runs on every
frame; the expensive vision-language reasoning runs **only** on verified anomaly windows.

---

## Five things that make it different

1. **Zero-noise evidence gating.** The detector/tracker act as a *trigger and cropper*, not as prompt
   filler. The VLM sees semantically-downsampled frames + zoomed ROI crops — avoiding the documented
   "pipe raw detections into the prompt and inject noise" failure mode.
2. **Counterfactual optical-vs-model blame.** A second detection + VLM pass on the *enhanced crop*
   separates **aleatory** optical impossibility (glare/darkness → an operating-domain limit) from
   **epistemic** model blind spots (fixable with training data).
3. **Deterministic, auditable attribution.** The VLM never chooses the root cause — fixed rules over
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
| ① Mining | Flag 4–8 s edge-case windows, discard routine video | RF-DETR · CMC-SORT · gradient boosting | ✅ validated (held-out recall 0.92 at 2.1 false windows/min) |
| ② Evidence | Keyframes + zoomed ROI crops + telemetry | OpenCV · CLAHE | ✅ |
| ③ Diagnosis | Describe + attribute root cause, critic-checked | Qwen3-VL (local) · deterministic rules | ✅ 99.97% valid JSON (3,360 calls) |
| ④ Taxonomy | Cluster into ranked SOTIF-tagged failure modes | UMAP/PCA · HDBSCAN | ✅ 56 modes from 856 events |
| ⑤ Remediation | Co-training manifest → fine-tune → prove gains | COCO export · RF-DETR train | ⚙️ machinery built (label-free + CODA-ready) |

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

- 📋 **[What we built — project journal](docs/WHAT_WE_DID.md)** — everything done, and the honest failures we caught and fixed.
- 🗂️ **[Full build plan](docs/PLAN.md)**

---

## Design notes & honesty

- **Licensing:** RF-DETR (Apache-2.0) over AGPL YOLO; local VLMs via Ollama; nothing copyleft in the path.
- **What is validated:** mining *generalizes* (held-out ≈ cross-validated, on 1,100 and on 200 clips) and
  *times the event* rather than the clip type (within-clip AUC 0.938 on held-out clips). Diagnosis is
  reproducible and critic-checked. Taxonomy consolidates real recurring modes.
- **What is *not* (and the report says so):** perception-failure causes aren't independently validated
  (Nexar has no detector-failure labels — that needs CARLA counterfactuals or human review); the
  domain is US/clear-day-heavy; this **mines** events in recorded video, it does **not predict** crashes
  before they are visible.
- **Diagnosis explains a window; it does not verify it.** Root-cause labels on 1,142 near-miss windows and
  on 123 false-alarm windows from held-out *normal* clips are distributed the same way: 54 / 25 / 10 / 9 / 2 %
  vs. 55 / 24 / 9 / 8 / 3 % (behaviour / inconclusive / occlusion / perception gap / optical). The rules
  read the same telemetry that fired the window (TTC ≤ 2 s or strong ego-path looming → AGENT_BEHAVIOR),
  so a card says *why* the miner flagged a moment, not that a near-miss happened, and the taxonomy groups
  triggers rather than verified failures. Whether false-alarm windows are unlabelled close calls needs
  human review.
- **Localizer, not a clip classifier:** 78% of normal-driving clips still get at least one window (down
  from 93%). The value is in narrowing each clip to a few seconds (75% of video discarded, ~2 false
  windows/min), not in declaring whole clips safe.
- **Confound check (passed for the model, partial for one cue):** `sotif-miner confound` recomputes AUC
  only *within* strata. On the 200 held-out clips the model's score is **0.938 within the same clip**
  (event moment vs. that clip's quiet moments), 0.912 scene-matched and 0.911 traffic-matched, vs. 0.913
  pooled — it times the event; scene, traffic and anything else constant per clip do not explain it.
  Individual cues differ: lead-vehicle size drops from 0.70 pooled to 0.61 within clips (0.56 held-out),
  so it is partly a clip marker (closer traffic in near-miss clips); corridor occupancy, active tracks
  and glare fall to ~0.50 within clips — pure clip markers. Ego-motion jerks *rise* to 0.71–0.80 within
  clips: the driver's reaction times the event but varies a lot between drivers. Training without the
  clip-marker inputs changes nothing (AUC 0.917 both; held-out 2.3 vs 2.1 false windows/min, overlapping
  CIs), so the model does not lean on them; removing absolute levels instead nearly doubles false windows
  (journal #11).
- **High-recall mode:** the default threshold targets recall ≥ 0.9 with the most video discarded. For
  "miss nothing" triage, threshold 0.4 reaches **0.97 recall (0.83 before the alert)** while still
  discarding 65% of video (5-fold CV, 1,300 clips).

---

## Repository layout

```
src/sotif_miner/
  io/          video & frame readers
  datasets/    Nexar loader (+ held-out slicing)
  perception/  RF-DETR detector · CMC-SORT tracker · embedders
  mining/      4 signals · rolling scorer · learned Stage-1 model
  evidence/    keyframe compression · ROI crops · bundles
  vlm/         Ollama / vLLM / Claude / scripted backends · prompts
  diagnosis/   deterministic attribution · critic · 4-step agent
  taxonomy/    clustering · naming · Area 2/3
  remediation/ COCO manifest · fine-tune wrapper
  eval/        mining · bootstrap CIs · closed-loop
  report.py    self-contained HTML deliverable
  cli.py       mine · diagnose · taxonomy · train-stage1 · report · export-manifest · finetune · eval-closed-loop
tests/         73 CPU tests (gpu/network tests opt-in)
```

---

*Built iteratively and validated on real footage, correcting several honest failures along the way
(signal saturation on real video, in-sample overfitting, and an evaluation that measured the wrong
task) — each caught, fixed, and documented. Aligned with ISO 21448 (SOTIF) and Euro NCAP 2026 scenarios.*
