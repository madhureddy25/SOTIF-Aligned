# What we built — project journal

An honest, chronological account of this project: what was built, what it achieves, and the real
failures we caught and fixed along the way. Nothing here is estimated — every number was measured on
your hardware against real Nexar footage.

---

## The goal

Build, end-to-end, an **autonomous dashcam edge-case miner and causal root-cause agent**: take raw
monocular dashcam video, find the rare dangerous moments, explain *why* each was dangerous, group them
into recurring failure modes aligned to ISO 21448 (SOTIF), and close the loop by exporting training
data that fixes them. Constraint: must run on a single laptop GPU (RTX 3050, 6 GB).

---

## What exists now

A complete Python package (`sotif-miner`), **73 passing tests**, clean linting, and a one-command
self-contained HTML report. Five decoupled stages:

| Stage | What it does | Status |
|---|---|---|
| ① **Mining** | RF-DETR detector + camera-motion-compensated tracker; four trigger signals; a learned gradient-boosting scorer; slices 4–8 s candidate windows and discards routine video | ✅ validated |
| ② **Evidence** | Semantic keyframe compression, zoomed + contrast-enhanced ROI crops, a 5-line telemetry header | ✅ |
| ③ **Diagnosis** | A 4-step agent (local vision-language model) with **deterministic** root-cause attribution and a hallucination critic | ✅ validated |
| ④ **Taxonomy** | Clusters events into ranked, SOTIF-tagged (Area 2 known / Area 3 novel) failure modes | ✅ |
| ⑤ **Remediation** | COCO co-training manifest → RF-DETR fine-tune → paired before/after proof | ⚙️ machinery built |

---

## What we actually measured (real Nexar footage, your GPU)

- Mined **200 clips**, diagnosed **141 events** → **99.8 % valid JSON** from the local 4B model, 1 failure in 416 calls.
- Built a **21-mode failure taxonomy**, 100 events in recurring (Area 2) modes, 13 novel (Area 3).
- **Held-out mining recall 0.91** (95 % CI 0.85–0.96) on 100 clips the model never saw — matching the
  cross-validated 0.90, which proves the model **generalizes** (no overfitting).
- Flags 73 % of near-misses **before** the official alert time, ~3.3 s median early warning, discards
  67 % of video, at ~3.1 false windows/min on normal driving.

---

## The honest failures we caught and fixed

This is the part that makes the results credible — we did not paper over anything.

1. **Signals saturated on real video.** The first trigger signals fired on *every* clip (uncertainty
   and optical stress sat at maximum on real footage). → Rebuilt them to score *change relative to each
   clip's own recent baseline*, not absolute levels. False-alarm rate dropped sharply.

2. **Diagnosis collapsed to one label.** The first real diagnosis run labelled 27 of 28 events the same
   ("model blind spot"). → Traced it to loose rules (any object in a crop counted as "recovered";
   normal confidence jitter counted as "unstable") and to ranking actors by instability and then blaming
   instability (circular). Fixed with the real telemetry — **no video hand-labelling required**.

3. **Reported recall was overfit.** An early report showed recall 1.00 — because the model was scored on
   the same clips it trained on. → Switched the report to honest **cross-validated** numbers and added a
   loud in-sample warning so it can't happen silently.

4. **The evaluation measured the wrong task.** The Nexar test split scores *predicting a crash before it
   is visible* — a different job from mining events in recorded video. Our near-chance AP there was
   expected and meaningless for mining. → Built the correct **held-out mining test** (re-mine unseen
   full-length clips), which gave the real 0.91, and relegated the anticipation AP to a labelled aside.

Each fix is a separate commit with the evidence in its message.

---

## Industry-grade layer

- **Confidence intervals** on every headline number (clip-level bootstrap).
- **Held-out test** on clips never used for tuning — the number an external reviewer trusts.
- **Provenance** (code version, config, dataset counts) and a standing **limitations** section in the report.
- **Licensing discipline:** RF-DETR (Apache-2.0) over AGPL YOLO; local VLMs; nothing copyleft in the path.

---

## Honestly not done (clearly scoped)

- **Phase 4 run:** the closed-loop machinery is built and tested, but a real *detection-accuracy* gain
  needs box labels Nexar lacks (CODA or BDD). The label-free path runs today and proves the mechanics.
- **Scale:** used 200 of 1,500 training clips. More clips → tighter CIs.
- **Real-time / on-vehicle:** out of scope; this is an offline analysis and triage tool.
- **Perception-failure causes** aren't independently validated (needs CARLA counterfactuals or human review).

---

## Where it fits in the real world

The industry is shifting from "collect more miles" to "curate the long tail," and 2026 regulations
(UNECE automated-driving framework, ISO 21448 continuous monitoring, Euro NCAP 2026) *mandate* exactly
this kind of auditable failure catalog. Because dashcams are cheap and everywhere, an automated miner
like this is the natural first step between raw fleet video and everything a safety or ML team does next:
curating training data, building a SOTIF safety case, triaging fleet incidents, and steering synthetic
world-model generators toward the verified blind spots.
