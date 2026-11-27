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
- Retrained Stage 1 on **1,300 clips**: 5-fold CV recall 0.91 [0.89, 0.93], early recall 0.72 [0.68, 0.75];
  on the 200 untouched held-out clips recall 0.91 [0.85, 0.96], median lead 4.2 s — but AUC unchanged (see #6).
- Mined the **full Nexar train split: all 1,500 clips** (200 to train Stage 1; 200 + 1,100 held out), ~46 s per clip on the 3050.
- **Held-out mining recall 0.88** (95 % CI 0.86–0.91) on **1,100 clips** the model never saw (trained on
  200). 5-fold CV on the training clips predicted 0.90 [0.84, 0.95] recall and 0.67 [0.58, 0.76] early
  recall; held-out gave 0.88 and 0.65 — inside both CV intervals, so the model **generalizes**.
- Flags 65 % (CI 0.61–0.69) of near-misses **before** the official alert time, 3.1 s median early warning
  (CI 2.7–3.4), discards 65 % of video, at 3.2 false windows/min (CI 3.1–3.4) on normal driving.

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

5. **A small held-out test was optimistic.** The first held-out test (100 clips) read recall 0.91 and
   early recall 0.73. Scaling the held-out set to 1,100 clips gave recall 0.88 (CIs overlap — consistent)
   but early recall 0.65, with the old 0.73 *outside* the new CI — and 0.65 matches the 0.67 that
   cross-validation had predicted, so the 100-clip figure was the outlier. → We report the 1,100-clip numbers and
   keep the 100-clip result only as history. It also surfaced that 93 % of normal clips get ≥1 window:
   the miner localizes events within a clip rather than classifying clips, now stated in the limitations.

6. **More data did not make the model better — and we say so.** Retraining Stage 1 on all 1,300 clips
   (6.5× more) left step-level AUC unchanged (0.855 → 0.851). On the same 200 held-out clips, recall stayed
   0.91, warnings came earlier (median lead 3.3 → 4.2 s), but false windows rose (3.1 → 3.4/min) and
   compression fell (0.67 → 0.63). → The model is **feature-limited, not data-limited**; the next
   improvement has to come from better signals. The threshold sweep also exposed a window-slicer quirk
   (recall *drops* at very low thresholds because long active stretches are clamped to one 8 s window
   around a single peak) — documented as a limitation, not hidden.

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
- **Better Stage-1 signals:** scaling data 200 → 1,300 clips did not raise AUC, so fewer false alarms
  need new features (e.g. ego-path looming), plus splitting long active stretches in the window slicer.
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
