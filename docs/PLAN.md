# Plan: SOTIF-Aligned Causal Dashcam Miner (`sotif-miner`), all 4 phases

## Progress (resume point)

**Done:**
- Directory skeleton under `/home/user/sotif-miner`.
- `pyproject.toml`.
- `src/sotif_miner/schemas.py`, with all stage records.
- Dependencies installed in `/home/user/.venv`.

**Update 24 Nov:**
- Data pivot: the user downloaded **Nexar** (31 GB, `D:\data\nexar`). It replaces BDD100K/ACDC/IDD-AW/CODA as the main data. Planned additions: CARLA, the user's own phone videos, and comma2k19.
- Stage 1 is committed (`705184c`): configs, video IO, RF-DETR wrapper (class names verified against rfdetr 1.12.1), CMCSort, signals, scorer, synth clips, Nexar loader + eval, CLI (`demo` / `mine` / `nexar` / `nexar-eval`). 18 tests pass.
- **Next:** Phase 2 (evidence bundles, VLM adapters, attribution, critic, agent).

**Next (original):** Phase 1, starting with configs, then `io/video.py`, the dataset loaders, the detector and tracker, the mining signals and scorer, `synth/clips.py`, and their tests. Phases 2–4 follow as planned below.

**Deviation:** default YAML configs ship inside the package at `src/sotif_miner/configs/`, so they install with it. A user override YAML is merged on top.

**Datasets (user downloads; every source needs login and licence acceptance, so there are no anonymous direct file URLs):**
- **BDD100K** and **BDD100K MOT:** https://bdd-data.berkeley.edu (docs: https://doc.bdd100k.com). Get the video parts, `100K Images` + detection labels, and the MOT 2020 images + labels.
- **ACDC:** https://acdc.vision.ee.ethz.ch. Get the RGB images and the semantic, panoptic and invIds ground truth.
- **IDD-AW:** https://iddaw.github.io, with the actual download via the IDD Insaan account (Dataset > Download > IDD-AW, which issues a 24 h token).
- **CODA:** https://coda-dataset.github.io/download.html.
- **User machine notes:** Windows PowerShell 5.1 with conda `(base)` on Python 3.14. The project venv must use Python 3.11 (torch/rfdetr wheels). Data goes on `D:\data` (393 GB free). The Kaggle CLI 2.2.1 is installed; the token is not set up yet.
- **Priority:** start with the small ones (CODA, ACDC val, a few hundred BDD clips). The synthetic tests need no data at all.

## Context

You shared a research audit for a system that mines edge cases from monocular dashcam video and diagnoses their causes. It has five stages: kinematic mining, evidence bundling, counterfactual VLM root-cause diagnosis mapped to ISO 21448, corpus-level failure taxonomy, and closed-loop co-training plus synthetic-seed export. `/home/user` is empty, so this is a new Python repo.

Decisions from your answers:
- **Scope:** all 4 phases, end to end.
- **Compute:** build here on CPU (4 cores, no GPU). Real RF-DETR and VLM runs happen later on your 24 GB GPU box. Every model sits behind an interface that has a CPU stub.
- **VLM backends:** vLLM (OpenAI-compatible), Claude API, and Ollama, all behind one adapter.
- **Data:** loaders for BDD100K, ACDC, IDD-AW and CODA that you point at your own downloads. Tests use generated synthetic clips, so no gated data is needed to run them.

Design stance: the paper's quantitative claims (benchmark numbers, the DeepAccident noise finding, Ultralytics defaults) motivate the design but are not hard dependencies. Every claim the build relies on is turned into an ablation the repo can run.

Three corrections the plan builds in:
1. **ACDC, IDD-AW and CODA are still images, not video.** The temporal triggers (flicker, ID switches, looming) only run on BDD100K videos and on BDD100K MOT sequences. Image datasets run in a *single-frame mode*: no tracker, only optical, density and rarity triggers, and the Stage 3 ROI audit runs on the detector's low-confidence or missed boxes.
2. **Licensing:** the plan avoids Ultralytics (AGPL) and BoxMOT (also AGPL). Tracking uses the Apache-2.0 `trackers` package (Roboflow) where it covers a tracker, plus an in-repo BoT-SORT-style tracker with camera-motion compensation built on OpenCV. Depth uses Depth-Anything-V2-**Small** (Apache 2.0); the larger variants are non-commercial.
3. **The ISO 21448 Area 2 vs Area 3 label is decided by code, not by the VLM.** An event is Area 3 (unknown) when it matches no existing failure cluster or known-trigger catalog entry; otherwise it is Area 2.

## Repo layout (`/home/user/sotif-miner`, git init locally, no remote)

```
pyproject.toml            # extras: [detect] rfdetr,torch; [vllm]/[ollama] openai,httpx; [claude] anthropic; [embed] transformers,sentence-transformers; [cluster] umap-learn; [viz] fiftyone; [eval] pycocotools,trackeval; [dev] pytest,ruff
configs/
  default.yaml            # thresholds, window sizes, weights, backend selection
  ontology.yaml           # dataset class → unified driving ontology (ped, rider, bicycle, motorcycle, car, truck, bus, autorickshaw, animal, debris, other)
  sotif_catalog.yaml      # triggering-condition enums, Euro NCAP 2026 tag list, known-trigger catalog (seeds Area 2)
src/sotif_miner/
  schemas.py              # pydantic: Detection, Track, FrameSignals, CandidateWindow, EvidenceBundle, SceneAudit, RoiAudit, TriggerCard, CriticReport, Cluster, Manifest
  io/video.py             # MP4 / frame-dir readers (cv2), optional MCAP via mcap-ros2 (lazy import)
  datasets/{bdd100k,bdd_mot,acdc,iddaw,coda}.py   # common Sample/Sequence interface + GT boxes (ACDC/IDD-AW boxes from panoptic/polygon instances) + ACDC invIds masks
  perception/detector.py  # Detector protocol; RFDETRDetector (rfdetr pkg, size N/S/M/L, optional -Seg); ScriptedDetector (tests)
  perception/tracking.py  # Tracker protocol; CMCSort (Kalman + ByteTrack two-pass association + OpenCV sparse-flow/RANSAC global motion compensation, BoT-SORT-style); OCSortAdapter / ByteTrackAdapter wrapping `trackers` if installed
  perception/embed.py     # Embedder protocol; SigLIP/DINOv2 image embedder (transformers); sentence-transformers text embedder; HashEmbedder stub
  mining/signals.py       # four scorers: detection entropy/flicker (0.25–0.60 band), track continuity (mid-frame terminations, ID-switch bursts, looming rate / TTC proxy), optical & density stress (luminance, specular saturation, Laplacian blur, track count >15/20), embedding rarity (kNN distance, sklearn NearestNeighbors)
  mining/scorer.py        # rolling 0.5 s Edge-Case Candidate Score, weighted + hysteresis → CandidateWindow (4–8 s: pre / peak / post)
  evidence/downsample.py  # semantic keyframe compression: dense 4–6 fps around signal transitions, prune redundant steady-state frames (frame-diff + embedding similarity)
  evidence/roi.py         # pick 1–3 conflicting actors; ROI crop with occluder-edge padding; upscale + CLAHE contrast normalization; minimal numeric ID marks on those actors only
  evidence/bundle.py      # EvidenceBundle: frames + ROI patches + 5-line telemetry header; also builds ablation arms (video_only, raw_cv_dump)
  vlm/base.py             # VLMBackend protocol: generate_structured(images, prompt, schema: type[BaseModel]) -> BaseModel; text_structured(...) for judge / cluster naming
  vlm/openai_compat.py    # vLLM: OpenAI client, response_format json_schema, extra_body chat_template_kwargs.enable_thinking=False
  vlm/ollama.py           # native /api/chat with format=<schema>, think=false, pinned *-instruct tag; warns on untagged qwen3-vl
  vlm/claude.py           # anthropic SDK (details below)
  vlm/scripted.py         # deterministic fake backend for tests and the CPU demo
  vlm/prompts/*.md        # Q3 neutral scene audit, ROI verification, trigger-card synthesis, repair, judge, cluster-naming
  diagnosis/attribution.py# deterministic Optical-vs-Epistemic-vs-Occlusion decision table
  diagnosis/critic.py     # schema guard + hallucination critic (rules over evidence)
  diagnosis/agent.py      # 4-step state machine with one critic-guided repair retry, then needs_review
  taxonomy/cluster.py     # card embedding (structured one-hot + text emb + ROI emb) → UMAP → sklearn HDBSCAN; LLM cluster naming; ranking
  taxonomy/memory.py      # failure-memory index keyed by static scene structure + dynamic actor cue (DriveVLA-M0-style decoupled retrieval); match() → Area 2/3
  remediation/manifest.py # core-replay co-training manifest → COCO dataset dirs (rfdetr layout: train/valid/test/_annotations.coco.json)
  remediation/retrieve.py # labelled look-alike retrieval: nearest GT-labelled frames (CODA / IDD-AW / BDD train) to each cluster centroid
  remediation/seeds.py    # AutoAWG-style control bundle: depth (Depth-Anything-V2-Small), object-masked lineart (Canny × M_obj), sky-excluded sketch (×(1−M_sky)); boxes JSON locked
  remediation/finetune.py # thin wrapper around rfdetr .train() + before/after eval
  eval/{detection,tracking,mining,ablation,attribution,judge,forgetting}.py
  viz/fiftyone_export.py  # optional: load clips, cards, cluster labels into a FiftyOne dataset
  cli.py                  # typer: mine | diagnose | taxonomy | export | finetune | eval {perception,mining,ablation,closed-loop} | run (end-to-end) | demo
  synth/clips.py          # synthetic video generator + matching scripted detections and VLM replies
tests/                    # pytest, CPU only; @pytest.mark.gpu / .network for real-model smoke tests
scripts/gpu_smoke.sh      # one-shot GPU-box check: RF-DETR-M + vLLM Qwen3-VL-8B-Instruct on 1 BDD clip
README.md                 # setup, dataset layout, GPU-box runbook, licence notes
```

## Key designs

**Stage 1 scoring.** Each scorer emits a normalized [0, 1] value every 0.5 s. The combined score is a weighted max-plus-mean, with weights and threshold set in YAML. A window opens on crossing `t_on` and closes below `t_off`. The window's dominant signal is recorded, because Stage 3 uses it. The rarity scorer keeps a reservoir of keyframe embeddings per run, and per corpus when you pass `--corpus-index`.

**Stage 3 state machine (`diagnosis/agent.py`).**
1. **Scene audit (Q3, unprimed):** the VLM sees the downsampled full frames only, with no IDs or telemetry, and returns `SceneAudit` (free description, hazards_mentioned, weather, illumination, road_layout, vru_density). This record is also the unprompted-recall measurement.
2. **Counterfactual ROI audit:** re-run the detector on the enhanced, upscaled crop, then ask the VLM a verification prompt on the crop that returns `RoiAudit` (object class, visibility 0–3, occluder present). `attribution.py` then decides deterministically:
   - **EPISTEMIC_PERCEPTION_GAP:** the crop detection matches a class that was missed or flickering in the original, *or* the VLM sees the object clearly (visibility ≥ 2) while the detector missed it.
   - **ALEATORY_OPTICAL:** optical stress is high, the crop detection fails, and visibility ≤ 1.
   - **DYNAMIC_OCCLUSION:** the track is born or re-born adjacent to and overlapping another box or an image edge, with clean visibility after emergence.
   - **AGENT_BEHAVIOR:** perception is stable but the kinematic hazard is high (looming/TTC, cut-in).
3. **Trigger card synthesis:** the VLM fills `TriggerCard`, with the evidence and the *fixed* attribution label supplied as input. Fields are enum-constrained: triggering_condition, functional_insufficiency, causal_layer, kinematic_hazard, euro_ncap_2026_tag[], actionable_fix, cited evidence frame IDs. `iso_21448_area` is filled afterwards by `taxonomy/memory.match()`.
4. **Critic:** checks schema validity, then rules. The claimed actor class must agree with detector or crop classes. A claimed occlusion must match the track birth evidence. A claimed weather must agree with the optical metrics. A claimed "clearly visible" must not contradict visibility ≤ 1. Cited frames must exist. On failure: one repair call with the critic's findings, then the card is marked `needs_review`.

**Claude backend (`vlm/claude.py`).** Uses `anthropic.Anthropic()` with the default credential chain. The default model is `claude-opus-5-5` (configurable), with `output_config={"effort": "medium"}` and no `thinking` param, since thinking can't be disabled on this model. Structured output comes from `messages.parse(..., output_format=<pydantic model>)`, reading `.parsed_output`. Server-side refusal fallback is on by default: beta `server-side-fallback-2026-07-01` with `fallbacks="default"`. Images are sent as base64 JPEG blocks before the text. `stop_reason` is checked before parsing. The stable system prompt is prompt-cached. Errors are caught in order: `RateLimitError` → `APIStatusError` → `APIConnectionError`. Whether the beta `parse` path accepts `fallbacks` with `output_format` gets checked against the installed SDK at implementation time; if not, the code falls back to `beta.messages.create` with `output_config.format` json_schema. The same backend serves as the text-only LLM judge.

**Stage 5 labels (the honest version).** Mined BDD clips have no box ground truth, so fine-tuning data comes from two sources:
- **(a) Labelled look-alikes:** GT-labelled frames from CODA, IDD-AW and BDD train splits that are nearest to each failure cluster's centroid. This is the default.
- **(b) Pseudo-labels:** crop-detector output confirmed by the VLM, written to a separate split flagged `review_required`. These are excluded from evaluation.

The co-training ratio is configurable: corner:core defaults to 1:3. Core frames are chosen by mid-band detector uncertainty plus k-center diversity on embeddings, following the VLM-C4L idea.

## Phased implementation

**Phase 1: foundation, perception, mining**
- pyproject, configs, schemas, video IO, the dataset loaders, and the ontology mapping.
- Detector and tracker protocols. `RFDETRDetector` uses lazy imports so CPU tests never import torch.
- `CMCSort`, plus the `trackers` adapters.
- The four signal scorers, the rolling scorer, and the window slicer.
- The `synth/clips.py` generator. Each scenario comes with ground-truth event times: an occlusion emergence, a confidence-flicker pedestrian, a luminance collapse (tunnel/glare), a dense crowd of more than 20 tracks, a looming vehicle, and a steady-state negative.
- `eval/mining.py`: event recall, compression ratio (fraction of video discarded) and window IoU against the synthetic ground truth, and against BDD100K MOT for tracker HOTA/IDF1 using TrackEval.

**Phase 2: evidence and causal agent**
- Downsampler, ROI builder, bundle builder, and the three ablation arms.
- The VLM adapters (vLLM, Ollama, Claude, scripted) and the prompts.
- Attribution, critic, and the agent state machine.
- `eval/ablation.py`:
  - **Arms:** video_only, raw_cv_dump and evidence_gated, run over CODA and ACDC.
  - **Metrics:** Q3 unprompted recall and Q5 guided recall (both judged), JSON validity rate, critic-rejection rate, and judge score. Judge prompts are calibrated against a human-label CSV template.
  - **ACDC alignment** (`eval/attribution.py`): an ALEATORY_OPTICAL label should coincide with high ROI overlap with the invIds invalid mask. The output is a 2×2 agreement table plus Cohen's κ.

**Phase 3: taxonomy and unstructured traffic**
- Card embedding, UMAP + HDBSCAN, and LLM cluster naming. Clusters are ranked by frequency × severity, where severity is a min-TTC proxy plus a VRU flag.
- Failure-memory match, which drives the Area 2/3 label.
- Reports: a cluster table in Markdown and JSON with representative clip links, Euro NCAP and SOTIF breakdowns, and an optional FiftyOne export.
- IDD-AW density-stratified reporting: ≤10, 11–20 and >20 participants.

**Phase 4: closed loop**
- Manifest builder, look-alike retrieval, COCO export, the `finetune.py` wrapper, and the seed-bundle exporter.
- `eval/forgetting.py` compares before and after fine-tuning: AP50:95 on held-out CODA and IDD-AW subsets, cluster-matched and overall, and on BDD100K clean-day validation. A regression larger than a configurable ε (default 0.5 AP) on the clean set fails the run.
- `cli run` chains all stages. `cli demo` runs the whole pipeline on synthetic clips with scripted backends, on CPU, in under 2 minutes.

Each phase ends with a local commit after its tests pass.

## Verification

**In this container, CPU:**
- `pytest -q`:
  - unit tests for each scorer, the tracker (IDs held through synthetic occlusion, camera-motion compensation under a synthetic global pan), attribution table cases, critic rules, schema round-trips, manifest ratio and COCO validity (load with pycocotools), and seed-bundle shapes and mask locking;
  - adapter request-shape tests with mocked HTTP and SDK clients for vLLM, Ollama and Claude;
  - a stage-chaining integration test.
- `sotif-miner demo` end to end on synthetic clips. Expected: every injected event is recalled, the negative clip produces no window, all cards are valid JSON, clusters separate the synthetic archetypes, and the manifest and seed bundles are written.
- `ruff check`.
- Best effort: install `rfdetr` + CPU torch and run RF-DETR-Nano on one synthetic frame (`@pytest.mark.network`). If the weight download is blocked, I'll say so and leave the test skipped.

**On your GPU box** (README runbook plus `scripts/gpu_smoke.sh`):
1. Start vLLM with Qwen3-VL-8B-Instruct.
2. Run `sotif-miner run` on a few BDD100K clips.
3. Run `eval ablation` on a CODA/ACDC subset.
4. Run `eval closed-loop` after `finetune`.

What I can't verify here: real-model accuracy, ablation outcomes, and fine-tuning gains. Those numbers come from your GPU run. I'll report them as unverified rather than estimate them.
