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
- **Confound check:** the model's held-out score is 0.938 AUC *within the same clip* — it times the event,
  not the clip type (see #10).
- **Current Stage-1 model** (interaction cues, 1,300 clips): 5-fold CV AUC 0.917, recall 0.92 at 1.8 false
  windows/min with 77% of video discarded; held-out (200 clips) recall 0.92 [0.87, 0.97], 2.1 [1.8, 2.4]
  false windows/min, 75% discarded (see #9).
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

7. **A feature that was silently always zero.** The per-feature AUC table showed `c.mid_terminations`
   at exactly 0.000 on all 1,300 clips. Cause: the tracker reports a lost track ~3 s after it vanishes,
   but the feature looked up its last box in a 2.5 s history — every lookup missed. → Vanishes are now
   taken at the track's *last sighting* from the whole clip (also catching vanishes near the clip end).
   The feature came alive (AUC 0.556) but is weak: the learned model's AUC stayed 0.850, and as a
   hand-weighted rule it *hurt* the rule-based score (0.716 → 0.638), so the Nexar preset gives it rule
   weight 0 while the learned model keeps it.

8. **Recall fell at low thresholds.** The window slicer cut any long active stretch to one 8 s window around
   its single peak, dropping events elsewhere in the stretch (recall 0.79 at threshold 0.05 vs 0.94 at
   0.4). → Long stretches are now tiled into several windows. Recall is monotone again (0.99 at 0.05),
   which opens a high-recall triage mode: **0.98 recall, 0.88 before the alert, 38% of video discarded**
   at threshold 0.2. The default operating point is unchanged (0.92 recall, 62% discarded). On the
   200 untouched held-out clips (re-featured), the fixed model reached recall 0.95 [0.90, 0.99] and early
   recall 0.78, at 3.8 false windows/min and 60% discarded — more coverage for more windows, not a
   better curve (AUC 0.850).

9. **Breaking the ceiling with a new kind of signal.** Every change so far stayed at AUC ≈ 0.85, and the
   per-feature table showed the signal lived in motion. We added *interaction* cues computed from the
   cached tracks (no detection re-run): the lead road user in the ego corridor (proximity, closing rate,
   its change), road users moving toward the ego path with the camera's own motion removed, nearest VRU,
   and signed ego braking. A clean ablation — same re-featured clips, trained with and without the cues
   (`--legacy-features`) — gave AUC **0.850 → 0.917**; the legacy run reproduced the previous table
   exactly. On the 200 untouched held-out clips: recall 0.92 [0.87, 0.97], false windows **3.8 → 2.1/min**
   (non-overlapping CIs), video discarded **0.60 → 0.75**, at the cost of ~0.9 s less lead time.
   Honest caveats: the top cue is lead-vehicle size (AUC 0.70), which may partly reflect denser traffic in
   near-miss clips (a scene confound held-out clips cannot rule out); the VRU and ego-speed cues are ~0.50
   on Nexar. The cues feed the learned model only, so the rule-based scorer is unchanged.

10. **Checking the confound we flagged ourselves.** With lead-vehicle size as the top cue, the gain in #9
    could have come from near-miss clips simply having denser, closer traffic. `sotif-miner confound`
    recomputes every AUC only within strata: the same clip (event band vs. the clip's own steps > 2.5 s
    away), the same light/weather/scene labels (26 groups), and the same traffic-density tertile.
    Result on the 200 held-out clips: the model's score is **0.938 within clips** (0.913 pooled, 0.912
    scene-matched, 0.911 traffic-matched) — it times the event. The suspicion was partly right for the
    cue itself: lead-vehicle size falls from 0.70 to 0.61 within clips (0.56 held-out), and corridor
    occupancy, active tracks and glare fall to ~0.50 — clip markers, not moments. Ego-motion jerks rise
    from ~0.60 to 0.71–0.80 within clips: the driver's reaction marks the moment but its size varies
    between drivers, which pooled AUC hides.

11. **Where do the remaining false alarms come from? Not from shortcuts.** 72% of normal clips still got a
    window. Two hypotheses, each a clean ablation on the same 1,300 re-featured clips (5-fold CV, recall 0.92):
    - *Clip markers* (`--drop markers`: corridor occupancy, active tracks, saturation, lead present — the
      model inputs that are ~0.5 within clips): AUC 0.917 → 0.917, 1.80 → 1.83 false windows/min. On the
      200 held-out clips: recall 0.92 both, 2.1 → 2.3 false windows/min (overlapping CIs). No effect.
    - *Absolute levels* (`--no-raw`, z-scores vs. each clip's own baseline only): AUC 0.917 → 0.902, false
      windows 1.80 → **3.11**/min, normal clips with a window 0.72 → 0.95. Absolute context (how close the
      lead really is, how fast the ego car moves) is what separates routine change from danger.
    So the model is not leaning on "busy clip" shortcuts; the remaining false alarms are risky-*looking*
    moments in normal driving — some likely unlabelled close calls. Next: diagnose the false-alarm
    windows of normal clips with the VLM to see what they are. The no-markers model is equivalent and
    drops the inputs the confound check flagged, so it is the recommended model
    (`stage1_model_drop.joblib`). Writing the tests also showed `c.active_tracks` and `o.glare` were
    never model inputs (stored only), so they were never a shortcut risk.

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
- **Inspect the remaining false alarms:** clip markers and absolute levels are ruled out as their cause
  (#11). Diagnose the windows in normal clips (`diagnose --only negative`) to see whether they are
  unlabelled close calls or genuine misses of context.
- **Lane-relative cues (Phase B):** lane position needs pixels the perception cache does not keep; a
  decode-only pass (no detector) would add it. Worth it only if the confound check leaves headroom.
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
