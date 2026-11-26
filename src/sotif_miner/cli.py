"""sotif-miner command line.

Stage 1 commands available now:
  sotif-miner demo                         # synthetic clips, CPU only, no weights
  sotif-miner mine  --input <video|folder> # mine any dashcam videos
  sotif-miner nexar --root D:\\data\\nexar   # mine + score against Nexar near-miss ground truth
  sotif-miner nexar-eval --run <dir>       # re-score a finished Nexar run without recomputing
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from .config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="ADAS dashcam edge-case miner")

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def _tracker(cfg):
    from .perception.tracking import build_tracker

    return build_tracker(cfg)


def _write_clip(out_dir: Path, clip_id: str, samples, windows, duration_s: float, fps: float, extra=None,
                feats=None, implicated=None, last_frame: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "clip_id": clip_id, "duration_s": duration_s, "fps": fps, "last_frame": last_frame,
        "features": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in f.items()} for f in (feats or [])],
        "implicated": [sorted(i) for i in (implicated or [])],
        "windows": [w.model_dump() for w in windows],
        "samples": [{"t": round(s.t, 3), "frame": s.frame, "combined": round(s.combined, 4),
                     "u": round(s.uncertainty, 3), "c": round(s.continuity, 3),
                     "o": round(s.optical_density, 3), "r": round(s.rarity, 3)} for s in samples],
        **(extra or {}),
    }
    (out_dir / f"{clip_id}.json").write_text(json.dumps(rec))


def _load_clip(path: Path):
    from .schemas import CandidateWindow, SignalSample

    rec = json.loads(path.read_text())
    samples = [SignalSample(t=s["t"], frame=s["frame"], combined=s["combined"], uncertainty=s["u"],
                            continuity=s["c"], optical_density=s["o"], rarity=s["r"]) for s in rec["samples"]]
    windows = [CandidateWindow(**w) for w in rec["windows"]]
    return rec, samples, windows


def _mine_video(path: Path, cfg, detector, stride: int, cache: Path | None = None):
    from .io.video import VideoFile
    from .mining.runner import run_perception, save_log
    from .mining.scorer import compute_features, rescore
    from .perception.embed import TinyEmbedder

    src = VideoFile(path, stride=stride)
    log = run_perception(src, detector, _tracker(cfg), TinyEmbedder(), cfg["mining"]["step_s"])
    if cache is not None:
        save_log(log, cache)
    feats, implicated = compute_features(log, cfg)
    last = log.frames[-1].frame if log.frames else 0
    samples, windows = rescore(feats, implicated, cfg, log.fps, last, src.source_id, str(path))
    duration = (log.frames[-1].t + 1 / log.fps) if log.frames else 0.0
    return samples, windows, duration, log.fps, feats, implicated, last


@app.command()
def demo(out: Path = typer.Option(Path("runs/demo"), help="Output folder"),
         config: Path | None = typer.Option(None, help="Override YAML")):
    """Run Stage 1 on the built-in synthetic scenarios (CPU, no weights, no data)."""
    from .io.video import ArraySource
    from .mining.runner import run_perception
    from .mining.scorer import mine_log
    from .perception.detector import ScriptedDetector
    from .perception.embed import TinyEmbedder
    from .synth.clips import render, scenarios

    cfg = load_config(config)
    for name, sc in scenarios().items():
        frames, dets = render(sc)
        log = run_perception(ArraySource(frames, 10.0, name), ScriptedDetector(dets), _tracker(cfg),
                             TinyEmbedder(), cfg["mining"]["step_s"])
        samples, windows = mine_log(log, cfg)
        _write_clip(out / "clips", name, samples, windows, len(frames) / 10.0, 10.0,
                    {"ground_truth_event_s": sc.event})
        spans = ", ".join(f"{w.start_frame / w.fps:.1f}-{w.end_frame / w.fps:.1f}s [{w.dominant_signal}]"
                          for w in windows) or "none"
        typer.echo(f"{name:22s} event={sc.event!s:12s} windows: {spans}")
    typer.echo(f"\nWritten to {out.resolve()}")


@app.command()
def mine(input: Path = typer.Option(..., "--input", help="Video file or folder of videos"),
         out: Path = typer.Option(Path("runs/mine"), help="Output folder"),
         config: Path | None = typer.Option(None, help="Override YAML"),
         stride: int = typer.Option(3, help="Process every Nth frame (3 = 10 fps from 30 fps)"),
         limit: int = typer.Option(0, help="Max videos (0 = all)")):
    """Mine candidate edge-case windows from dashcam videos."""
    from .perception.detector import build_detector

    cfg = load_config(config)
    vids = [input] if input.is_file() else sorted(p for p in input.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    vids = vids[:limit] if limit else vids
    detector = build_detector(cfg)
    n_win = 0
    out.mkdir(parents=True, exist_ok=True)
    with (out / "windows.jsonl").open("a") as wf:
        for i, v in enumerate(vids, 1):
            if (out / "clips" / f"{v.stem}.json").exists():
                continue
            t0 = time.time()
            samples, windows, dur, fps, feats, impl, last = _mine_video(
                v, cfg, detector, stride, out / "perception" / f"{v.stem}.json.gz")
            _write_clip(out / "clips", v.stem, samples, windows, dur, fps, {"source": str(v)}, feats, impl, last)
            for w in windows:
                wf.write(w.model_dump_json() + "\n")
            n_win += len(windows)
            typer.echo(f"[{i}/{len(vids)}] {v.name}: {len(windows)} windows ({time.time() - t0:.1f}s)")
    typer.echo(f"Done: {n_win} new windows -> {out.resolve()}")


@app.command()
def nexar(root: Path = typer.Option(..., help="Nexar dataset root, e.g. D:\\data\\nexar"),
          split: str = typer.Option("train", help="train | test-public | test-private"),
          out: Path = typer.Option(None, help="Output folder (default runs/nexar_<split>)"),
          config: Path | None = typer.Option(None, help="Override YAML"),
          stride: int = typer.Option(3, help="Process every Nth frame"),
          limit: int = typer.Option(0, help="Clips per label (0 = all)"),
          skip: int = typer.Option(0, help="Skip the first N clips per label (mine a disjoint held-out slice)")):
    """Mine Nexar clips and score against near-miss ground truth (resumable)."""
    from .datasets.nexar import load_nexar
    from .perception.detector import build_detector

    cfg = load_config(config)
    out = out or Path(f"runs/nexar_{split}")
    clips = []
    for lab in ("positive", "negative"):
        clips += load_nexar(root, split, labels=(lab,), limit=limit or None, skip=skip)
    detector = build_detector(cfg)
    for i, c in enumerate(clips, 1):
        if (out / "clips" / f"{c.clip_id}.json").exists():
            continue
        t0 = time.time()
        samples, windows, dur, fps, feats, impl, last = _mine_video(
            c.path, cfg, detector, stride, out / "perception" / f"{c.clip_id}.json.gz")
        _write_clip(out / "clips", c.clip_id, samples, windows, dur, fps,
                    {"source": str(c.path), "positive": c.positive, "event_s": c.event_in_clip_s(),
                     "alert_s": c.alert_in_clip_s()}, feats, impl, last)
        typer.echo(f"[{i}/{len(clips)}] {c.path.name} {'POS' if c.positive else 'neg'}: "
                   f"{len(windows)} windows ({time.time() - t0:.1f}s)")
    nexar_eval(run=out, root=root, split=split, bootstrap=0, out_metrics=None, metrics_kind="")


@app.command()
def diagnose(run: Path = typer.Option(..., help="Run folder from `nexar` or `mine` (needs perception cache)"),
             config: Path | None = typer.Option(None, help="Override YAML"),
             only: str = typer.Option("all", help="all | positive | negative (Nexar runs)"),
             limit: int = typer.Option(0, help="Max clips to diagnose (0 = all)"),
             skip_vlm: bool = typer.Option(False, help="Only build evidence (pass A)"),
             skip_evidence: bool = typer.Option(False, help="Only run the VLM on existing evidence (pass B)"),
             rebuild: bool = typer.Option(False, help="Delete existing evidence + cards first (after code changes)")):
    """Stage 2+3: build evidence bundles (pass A, RF-DETR on crops) then diagnose with the VLM (pass B).

    The two passes run one after the other so RF-DETR and the VLM never share GPU memory."""
    cfg = load_config(config)
    ev_root = run / "evidence"
    if rebuild and ev_root.exists():
        import shutil

        shutil.rmtree(ev_root)
        (run / "cards.jsonl").unlink(missing_ok=True)
        typer.echo("Removed previous evidence and cards.")
    if not skip_evidence:
        _evidence_pass(run, ev_root, cfg, only, limit)
    if not skip_vlm:
        _vlm_pass(run, ev_root, cfg)


def _evidence_pass(run: Path, ev_root: Path, cfg: dict, only: str, limit: int):
    from .evidence.bundle import build_bundle, select_keyframes
    from .io.video import native_fps, read_frames
    from .mining.runner import load_log
    from .perception.detector import build_detector
    from .schemas import CandidateWindow

    todo = []
    for f in sorted((run / "clips").glob("*.json")):
        rec = json.loads(f.read_text())
        if only == "positive" and not rec.get("positive"):
            continue
        if only == "negative" and rec.get("positive", True):
            continue
        wins = sorted((CandidateWindow(**w) for w in rec["windows"]), key=lambda w: -w.peak_score)
        wins = wins[: cfg["diagnosis"]["max_windows_per_clip"]]
        cache = run / "perception" / f"{rec['clip_id']}.json.gz"
        if wins and cache.exists() and rec.get("source"):
            todo.append((rec, wins, cache))
    todo = todo[:limit] if limit else todo
    pending = [(r, w, c) for r, ws, c in todo for w in ws if not (ev_root / w.clip_id / "evidence.json").exists()]
    if not pending:
        typer.echo("Pass A: evidence already built (or no windows).")
        return
    typer.echo(f"Pass A: building evidence for {len(pending)} windows (RF-DETR on zoomed crops)...")
    detector = build_detector(cfg)
    logs: dict[str, object] = {}
    for i, (rec, w, cache) in enumerate(pending, 1):
        log = logs.get(rec["clip_id"]) or load_log(cache)
        logs = {rec["clip_id"]: log}  # keep one clip in memory
        src = Path(rec["source"])
        stride = max(1, round(native_fps(src) / rec["fps"]))
        keys = select_keyframes(w, cfg["evidence"]["max_frames"], cfg["evidence"]["keyframe_fps_dense"],
                                cfg["evidence"]["keyframe_fps_sparse"])
        near = range(max(w.start_frame, w.peak_frame - int(w.fps)), min(w.end_frame, w.peak_frame + int(w.fps)) + 1)
        images = read_frames(src, sorted(set(keys) | set(near)), stride)
        out = ev_root / w.clip_id
        gated = build_bundle(w, log, images, out, cfg, "evidence_gated")
        plain = build_bundle(w, log, images, out, cfg, "video_only")
        crop_dets, n_all = [], 0
        for j, r in enumerate(gated.rois):
            import cv2

            from .evidence.bundle import actor_box_in_crop
            from .perception.tracking import box_iou

            crop = cv2.imread(r.path)
            if crop is None:
                continue
            dets = detector.detect(crop, None)
            n_all += len(dets)
            if j == 0:  # only the primary actor decides "recovered on the crop"
                ab = actor_box_in_crop(r)
                thr = cfg["diagnosis"].get("crop_match_iou", 0.3)
                crop_dets += [d.model_dump() for d in dets if ab is None or box_iou(d.box, ab) >= thr]
        seen = sorted({d.cls for fr in log.frames if w.start_frame <= fr.frame <= w.end_frame for d in fr.detections})
        (out / "evidence.json").write_text(json.dumps({
            "bundle": gated.model_dump(mode="json"), "unmarked": plain.frame_paths, "crop_dets": crop_dets,
            "seen_classes": seen, "source": rec["source"], "positive": rec.get("positive"),
            "event_s": rec.get("event_s"), "alert_s": rec.get("alert_s")}))
        typer.echo(f"  [{i}/{len(pending)}] {w.clip_id}: {len(gated.frame_paths)} frames, {len(gated.rois)} crops, "
                   f"detections on crops {n_all} (on the primary actor: {len(crop_dets)})")
    del detector
    try:  # free GPU memory before the VLM pass
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _vlm_pass(run: Path, ev_root: Path, cfg: dict, backend=None):
    from collections import Counter

    from .diagnosis.agent import Diagnoser
    from .schemas import Detection, EvidenceBundle
    from .vlm.base import VLMError
    from .vlm.factory import build_backend

    evs = sorted(ev_root.glob("*/evidence.json"))
    pending = [e for e in evs if not (e.parent / "card.json").exists()]
    typer.echo(f"Pass B: {len(pending)} windows to diagnose ({len(evs) - len(pending)} already done)")
    vlm = backend or build_backend(cfg)
    agent = Diagnoser(vlm, cfg)
    for i, e in enumerate(pending, 1):
        d = json.loads(e.read_text())
        bundle = EvidenceBundle(**d["bundle"])
        try:
            card = agent.diagnose(bundle, d["unmarked"], [Detection(**x) for x in d["crop_dets"]],
                                  set(d["seen_classes"]), d["source"])
        except VLMError as err:
            (e.parent / "error.txt").write_text(str(err))
            typer.echo(f"  [{i}/{len(pending)}] {bundle.clip_id}: FAILED ({err})")
            continue
        (e.parent / "card.json").write_text(card.model_dump_json(indent=2))
        with (run / "cards.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(card.model_dump_json() + "\n")
        typer.echo(f"  [{i}/{len(pending)}] {bundle.clip_id}: {card.attribution.value} / {card.status}"
                   f"{' (repaired)' if card.repaired else ''} - {card.draft.summary[:90]}")
    cards = [json.loads((c.parent / "card.json").read_text()) for c in evs if (c.parent / "card.json").exists()]
    summary = {"cards": len(cards), "attribution": dict(Counter(c["attribution"] for c in cards)),
               "status": dict(Counter(c["status"] for c in cards)),
               "iso_area": dict(Counter(c["iso_21448_area"] for c in cards)),
               "repaired": sum(c["repaired"] for c in cards),
               "coerced": sum(bool(c.get("coerced")) for c in cards),
               "first_draft_critic_rules": dict(Counter(f["rule"] for c in cards for f in c.get("initial_findings", []))),
               "vlm": vlm.stats.as_dict()}
    (run / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def taxonomy(runs: str = typer.Option(..., help="Comma-separated run folders (each with evidence/*/card.json)"),
             out: Path = typer.Option(Path("runs/taxonomy"), help="Output folder"),
             config: Path | None = typer.Option(None, help="Override YAML"),
             include_inconclusive: bool = typer.Option(False, help="Include INCONCLUSIVE cards (Stage-1 false positives)")):
    """Stage 4: cluster trigger cards into ranked, SOTIF-tagged fleet failure modes (offline, CPU)."""
    from .taxonomy.cluster import build_taxonomy, render_markdown

    cfg = load_config(config)
    cards = []
    for rd in [Path(r.strip()) for r in runs.split(",") if r.strip()]:
        for cj in sorted((rd / "evidence").glob("*/card.json")):
            c = json.loads(cj.read_text())
            if include_inconclusive or c.get("attribution") != "INCONCLUSIVE":
                cards.append(c)
    if not cards:
        raise typer.BadParameter("no cards found in the given runs")
    tx = cfg["taxonomy"]
    result = build_taxonomy(cards, tx["min_cluster_size"], tx["umap_components"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "clusters.json").write_text(json.dumps(result.clusters, indent=2))
    (out / "assignments.json").write_text(json.dumps(result.assignments, indent=2))
    (out / "clusters.md").write_text(render_markdown(result))
    area2 = sum(c["size"] for c in result.clusters if c["iso_21448_area"].startswith("AREA_2"))
    total = sum(c["size"] for c in result.clusters)
    typer.echo(f"{len(cards)} cards -> {len(result.clusters)} clusters "
               f"({sum(not c['noise'] for c in result.clusters)} recurring, {sum(c['noise'] for c in result.clusters)} novel singletons)")
    typer.echo(f"ISO 21448: {area2}/{total} events in Area 2 (known), {total - area2} in Area 3 (unknown)")
    typer.echo("\nTop failure modes:")
    for i, c in enumerate(result.clusters[:10], 1):
        typer.echo(f"  {i}. [{c['size']:2d}] sev {c['severity']:.2f}  {c['name']}")
    typer.echo(f"\nReport: {(out / 'clusters.md').resolve()}")


@app.command()
def report(run: Path = typer.Option(..., help="Run folder (diagnosis_summary.json etc.)"),
           taxonomy: Path | None = typer.Option(None, help="Taxonomy folder (clusters.json)"),
           out: Path | None = typer.Option(None, help="Output HTML (default <run>/report.html)"),
           data_root: Path | None = typer.Option(None, help="Nexar root, to record dataset counts"),
           test_metrics: Path | None = typer.Option(None, help="Anticipation AP JSON from `nexar-eval` on a test split"),
           train_metrics: Path | None = typer.Option(None, help="Out-of-fold CV metrics (stage1_cv_metrics.json)"),
           heldout_metrics: Path | None = typer.Option(None, help="Mining metrics on unseen full-length clips (nexar-eval --split train --metrics-kind held_out)")):
    """Build the consolidated, self-contained HTML report (the stakeholder-facing deliverable)."""
    from .report import build_report

    counts = None
    if data_root is not None:
        from .datasets.nexar import SPLITS, load_nexar

        counts = {}
        for sp in SPLITS:
            try:
                cl = load_nexar(data_root, sp, require_video=False)
                counts[sp] = f"{sum(c.positive for c in cl)} pos / {sum(not c.positive for c in cl)} neg"
            except (FileNotFoundError, ValueError):
                pass
    tm = json.loads(test_metrics.read_text()) if test_metrics else None
    trm = json.loads(train_metrics.read_text()) if train_metrics else None
    hom = json.loads(heldout_metrics.read_text()) if heldout_metrics else None
    html_str = build_report(run, taxonomy, counts, tm, trm, hom)
    out = out or run / "report.html"
    out.write_text(html_str, encoding="utf-8")
    typer.echo(f"Report written: {out.resolve()}")
    typer.echo(f"Open it with:  start {out}")


@app.command("export-manifest")
def export_manifest(mine_run: Path = typer.Option(..., help="Mined run (perception cache + clips)"),
                    taxonomy: Path = typer.Option(..., help="Taxonomy folder (clusters.json)"),
                    out: Path = typer.Option(..., help="Output COCO dataset folder"),
                    diagnose_run: Path | None = typer.Option(None, help="Diagnosis run, for crop-recovered boxes"),
                    config: Path | None = typer.Option(None, help="Override YAML"),
                    top_clusters: int = typer.Option(8, help="Top failure clusters to pull corner frames from")):
    """Stage 5a: build a COCO co-training dataset (corner failure frames + core replay frames)."""
    from .io.video import native_fps, read_frames
    from .remediation.manifest import plan_label_free, write_coco

    cfg = load_config(config)
    clusters = json.loads((taxonomy / "clusters.json").read_text())
    plan = plan_label_free(mine_run, clusters, diagnose_run, pseudo_thresh=cfg["detector"]["threshold"] + 0.25,
                           top_clusters=top_clusters, ratio_core=1.0 / cfg["remediation"]["corner_to_core"])
    # map clip_id -> (source video, stride) from the run's clip records
    srcinfo = {}
    for cj in (mine_run / "clips").glob("*.json"):
        rec = json.loads(cj.read_text())
        if rec.get("source"):
            stride = max(1, round(native_fps(rec["source"]) / rec["fps"]))
            srcinfo[rec["clip_id"]] = (rec["source"], stride)
    # read the needed frames, grouped by source clip
    need: dict[str, set] = {}
    for fl in plan.corner + plan.core:
        need.setdefault(fl.source_clip, set()).add(fl.frame)
    images = {}
    for clip_id, frames in need.items():
        if clip_id not in srcinfo:
            continue
        src, stride = srcinfo[clip_id]
        got = read_frames(src, sorted(frames), stride)
        for fr, img in got.items():
            images[(clip_id, fr)] = img
    summary = write_coco(plan, images, out, val_frac=0.15)
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"COCO dataset: {out.resolve()}  (fine-tune with: sotif-miner finetune --dataset {out} ...)")


@app.command()
def finetune(dataset: Path = typer.Option(..., help="COCO dataset folder from export-manifest"),
             out: Path = typer.Option(Path("checkpoints/ft"), help="Checkpoint output folder"),
             size: str = typer.Option("nano", help="nano | small | medium | large (nano fits ~6 GB)"),
             epochs: int = typer.Option(20), batch_size: int = typer.Option(2),
             grad_accum: int = typer.Option(8), device: str = typer.Option("cuda")):
    """Stage 5b: fine-tune RF-DETR on the co-training dataset (GPU)."""
    from .remediation.finetune import finetune as _ft

    best = _ft(dataset, out, size, epochs, batch_size, grad_accum, device=device)
    typer.echo(f"Best checkpoint: {best.resolve()}")
    typer.echo("Re-mine a held-out run with it using a config whose detector.weights points here, then:")
    typer.echo("  sotif-miner eval-closed-loop --before <baseline_run> --after <finetuned_run> --root D:\\data\\nexar")


@app.command("eval-closed-loop")
def eval_closed_loop(before: Path = typer.Option(..., help="Held-out run mined with the BASE detector"),
                     after: Path = typer.Option(..., help="Same clips mined with the FINE-TUNED detector"),
                     root: Path = typer.Option(..., help="Nexar root"),
                     split: str = typer.Option("train"),
                     bootstrap: int = typer.Option(2000),
                     out_metrics: Path | None = typer.Option(None)):
    """Stage 5 eval: paired before/after mining comparison with CIs + anti-forgetting gate."""
    from .datasets.nexar import load_nexar
    from .eval.closed_loop import compare

    clips = {c.clip_id: c for c in load_nexar(root, split)}

    def results(run: Path):
        out = []
        for f in sorted((run / "clips").glob("*.json")):
            rec, samples, windows = _load_clip(f)
            if rec["clip_id"] in clips:
                out.append(__import__("sotif_miner.eval.closed_loop", fromlist=["ClipResult"]).ClipResult(
                    clips[rec["clip_id"]], samples, windows, rec["duration_s"]))
        return out

    rb, ra = results(before), results(after)
    res = compare(rb, ra, n_boot=bootstrap)
    if out_metrics:
        out_metrics.write_text(json.dumps(res, indent=2))
    typer.echo(json.dumps(res["_verdict"], indent=2))
    v = res["_verdict"]
    typer.echo(f"\nRecall improved: {v['recall_improved']}   Anti-forgetting OK: {v['anti_forgetting_ok']}")


@app.command("cards-report")
def cards_report(run: Path = typer.Option(..., help="Run folder with evidence/*/card.json")):
    """One line per card: root cause, the measurements behind it, and the card summary."""
    cards = sorted((run / "evidence").glob("*/card.json"))
    if not cards:
        raise typer.BadParameter(f"no cards under {run / 'evidence'}")
    for c in cards:
        d = json.loads(c.read_text())
        t, r = d["telemetry"], d.get("roi_audit") or {}
        ttc = f"{t['min_ttc_s']:.1f}" if t.get("min_ttc_s") is not None else "-"
        typer.echo(f"{d['clip_id']}  {d['attribution']}  [{d['status']}]")
        typer.echo(f"    actor={t.get('focus_cls')} presence={t.get('focus_presence', 1):.2f} "
                   f"dropouts={t.get('focus_dropouts', 0)} gap={t.get('focus_reacquired_gap', 0)} "
                   f"lowconf={t.get('focus_low_conf_frac', 0):.2f} born_mid={t.get('focus_born_mid')} "
                   f"adjacent={t.get('focus_born_adjacent')}")
        typer.echo(f"    ego_looming={t.get('ego_looming', 0):.2f} ttc={ttc} jerk={t.get('ego_jerk', 0):.4f} "
                   f"lum={t.get('luminance', 0):.2f} sat={t.get('saturation_frac', 0):.3f} "
                   f"sharp={t.get('sharpness', 0):.2f} crop_hit={d['crop_detector_hit']} "
                   f"vis={r.get('visibility')} crop_class={r.get('object_class')}")
        typer.echo(f"    why: {'; '.join(d['attribution_reasons'])}")
        typer.echo(f"    card: {d['draft']['summary'][:160]}")


@app.command("refeature")
def refeature_cmd(run: Path = typer.Option(..., help="Run folder with perception/*.json.gz"),
                  config: Path | None = typer.Option(None, help="Override YAML"),
                  out: Path | None = typer.Option(None, help="Write to a new run folder (default: in place)")):
    """Recompute features + scores from cached perception (new feature code, no detection re-run)."""
    from .mining.runner import load_log
    from .mining.scorer import compute_features, rescore

    cfg = load_config(config)
    out = out or run
    n = nw = 0
    for g in sorted((run / "perception").glob("*.json.gz")):
        clip_id = g.name[: -len(".json.gz")]
        old = run / "clips" / f"{clip_id}.json"
        extra = {}
        if old.exists():
            rec = json.loads(old.read_text())
            extra = {k: v for k, v in rec.items() if k not in ("clip_id", "duration_s", "fps", "windows", "samples",
                                                               "features", "implicated", "last_frame")}
        log = load_log(g)
        feats, impl = compute_features(log, cfg)
        last = log.frames[-1].frame if log.frames else 0
        samples, windows = rescore(feats, impl, cfg, log.fps, last, clip_id, extra.get("source", ""))
        dur = (log.frames[-1].t + 1 / log.fps) if log.frames else 0.0
        _write_clip(out / "clips", clip_id, samples, windows, dur, log.fps, extra, feats, impl, last)
        n += 1
        nw += len(windows)
    typer.echo(f"Re-featured {n} clips -> {nw} windows in {out.resolve()}")


@app.command("train-stage1")
def train_stage1(run: Path = typer.Option(..., help="Nexar run folder with stored features (train split)"),
                 root: Path = typer.Option(..., help="Nexar dataset root"),
                 causal: bool = typer.Option(False, help="Past-only context (real-time anticipation)"),
                 folds: int = typer.Option(5, help="Cross-validation folds (grouped by clip)"),
                 target_recall: float = typer.Option(0.9, help="Pick the threshold with the best compression "
                                                     "among those reaching this recall"),
                 config: Path | None = typer.Option(None, help="Override YAML"),
                 threshold: float | None = typer.Option(None, help="Use this threshold instead of the automatic choice"),
                 out: Path | None = typer.Option(None, help="Model file (default <run>/stage1_model[_causal].joblib)")):
    """Train the learned Stage 1 scorer; report clip-grouped cross-validated metrics per threshold."""
    import copy

    import joblib
    import numpy as np

    from .datasets.nexar import load_nexar
    from .eval.nexar_eval import ClipResult, _auc, evaluate_train
    from .mining.learned import cross_validate, fit, learned_samples, step_labels
    from .mining.scorer import slice_windows

    cfg = load_config(config)
    recs = [json.loads(f.read_text()) for f in sorted((run / "clips").glob("*.json"))]
    recs = [r for r in recs if r.get("features") and "positive" in r]
    if not recs:
        raise typer.BadParameter("no clips with stored features + labels (mine with `sotif-miner nexar` v4+)")
    clips = {c.clip_id: c for c in load_nexar(root, "train")}
    typer.echo(f"{len(recs)} clips ({sum(bool(r['positive']) for r in recs)} positive); "
               f"{'causal' if causal else 'offline (+-2 s context)'}; {folds}-fold CV grouped by clip ...")
    oof = cross_validate(recs, cfg, causal, folds)

    pe, pn = [], []
    for r in recs:
        y = step_labels(r["features"], bool(r["positive"]), r.get("event_s"), r.get("alert_s"))
        p = oof[r["clip_id"]]
        pe += list(p[y == 1])
        if not r["positive"]:
            pn += list(p[y == 0])
    step_auc = _auc(np.array(pe), np.array(pn))
    typer.echo(f"Step-level AUC (out-of-fold, event band vs. normal clips): {step_auc:.3f}   "
               f"(rule-based combined score was ~0.685)")

    rows = []
    for thr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        c2 = copy.deepcopy(cfg)
        c2["mining"]["t_on"], c2["mining"]["t_off"] = thr, thr * 0.7
        results = []
        for r in recs:
            samples = learned_samples(r["features"], oof[r["clip_id"]], c2)
            wins = slice_windows(samples, [set(i) for i in r.get("implicated", [])], c2, r["fps"],
                                 r.get("last_frame", 0), r["clip_id"])
            if r["clip_id"] in clips:
                results.append(ClipResult(clips[r["clip_id"]], samples, wins, r["duration_s"]))
        m = evaluate_train(results)
        rows.append((thr, m))
    typer.echo(f"\n{'thr':>5} {'recall':>7} {'early':>6} {'lead_s':>7} {'neg_FA':>7} {'win/min':>8} {'compress':>9}")
    for thr, m in rows:
        typer.echo(f"{thr:5.2f} {m['recall']:7.2f} {m['early_recall_before_alert']:6.2f} "
                   f"{(m['median_lead_time_s'] or 0):7.2f} {m['negative_clip_false_alarm_rate']:7.2f} "
                   f"{(m['negative_windows_per_minute'] or 0):8.2f} {(m['compression'] or 0):9.2f}")
    ok = [(thr, m) for thr, m in rows if m["recall"] >= target_recall]
    if ok:
        best_thr, best = max(ok, key=lambda x: x[1]["compression"] or 0)
        typer.echo(f"\nChosen threshold {best_thr}: recall {best['recall']:.2f} >= target {target_recall}, "
                   f"with the most video discarded.")
    else:
        best_thr, best = max(rows, key=lambda x: (x[1]["recall"], x[1]["compression"] or 0))
        typer.echo(f"\nTarget recall {target_recall} NOT reached at any threshold. Using {best_thr}, the highest "
                   f"recall ({best['recall']:.2f}). Choose another with --threshold if you prefer.")

    if threshold is not None:
        best_thr = threshold
        best = next((m for t, m in rows if abs(t - threshold) < 1e-9), best)
        typer.echo(f"Using the threshold you set: {threshold}")
    if threshold is not None:
        best_thr = threshold
    # Out-of-fold operating-point metrics WITH bootstrap CIs at the chosen threshold. These are the
    # honest train numbers: every clip is scored by a model that did not see it. (Applying the final
    # model to the same train clips via `rescore` + `nexar-eval` is in-sample and reads ~1.0 recall;
    # do NOT report that.)
    from .eval.bootstrap import bootstrap_train

    c2 = copy.deepcopy(cfg)
    c2["mining"]["t_on"], c2["mining"]["t_off"] = best_thr, best_thr * 0.7
    oof_results = []
    for r in recs:
        samples = learned_samples(r["features"], oof[r["clip_id"]], c2)
        wins = slice_windows(samples, [set(i) for i in r.get("implicated", [])], c2, r["fps"],
                             r.get("last_frame", 0), r["clip_id"])
        if r["clip_id"] in clips:
            oof_results.append(ClipResult(clips[r["clip_id"]], samples, wins, r["duration_s"]))
    cv_ci = bootstrap_train(oof_results, ("recall", "early_recall_before_alert", "median_lead_time_s",
                                          "negative_windows_per_minute", "compression"), n=1000)
    cv_ci["_meta"] = {"kind": "cross_validated_out_of_fold", "threshold": best_thr, "folds": folds,
                      "n_clips": len(recs), "step_auc": step_auc, "causal": causal}
    cv_path = run / f"stage1_cv_metrics{'_causal' if causal else ''}.json"
    cv_path.write_text(json.dumps(cv_ci, indent=2))

    model = fit(recs, cfg, causal, best_thr)
    model.cv_report = {"step_auc": step_auc, "threshold": best_thr, "metrics_at_threshold": best,
                       "n_clips": len(recs), "folds": folds, "causal": causal}
    out = out or run / f"stage1_model{'_causal' if causal else ''}.joblib"
    joblib.dump(model, out)
    typer.echo(f"\nSaved model: {out.resolve()}")
    typer.echo(f"Saved cross-validated metrics (use for the report): {cv_path.resolve()}")
    typer.echo("Apply the model to a run with: sotif-miner rescore --run <run> --model " + str(out))


@app.command("rescore")
def rescore_cmd(run: Path = typer.Option(..., help="Run folder with clips/*.json"),
                config: Path | None = typer.Option(None, help="Override YAML with new thresholds"),
                out: Path | None = typer.Option(None, help="Write to a new run folder (default: in place)"),
                model: Path | None = typer.Option(None, help="Learned Stage 1 model (from train-stage1)")):
    """Re-score stored features with new settings or a learned model (seconds; no detection re-run)."""
    from .mining.scorer import rescore, slice_windows

    cfg = load_config(config)
    learned = None
    if model is not None:
        import joblib

        from .mining.learned import learned_samples

        learned = joblib.load(model)
        cfg["mining"]["t_on"], cfg["mining"]["t_off"] = learned.threshold, learned.threshold * 0.7
    out = out or run
    n = nw = 0
    for f in sorted((run / "clips").glob("*.json")):
        rec = json.loads(f.read_text())
        if not rec.get("features"):
            typer.echo(f"skip {f.name}: no stored features (mined with an older version; re-run mining)")
            continue
        impl = [set(i) for i in rec.get("implicated", [])]
        if learned is not None:
            samples = learned_samples(rec["features"], learned.predict_steps(rec["features"], cfg), cfg)
            windows = slice_windows(samples, impl, cfg, rec["fps"], rec.get("last_frame", 0), rec["clip_id"],
                                    rec.get("source", ""))
        else:
            samples, windows = rescore(rec["features"], impl, cfg, rec["fps"], rec.get("last_frame", 0),
                                       rec["clip_id"], rec.get("source", ""))
        extra = {k: v for k, v in rec.items() if k not in ("clip_id", "duration_s", "fps", "windows", "samples",
                                                           "features", "implicated", "last_frame")}
        _write_clip(out / "clips", rec["clip_id"], samples, windows, rec["duration_s"], rec["fps"], extra,
                    rec["features"], [set(i) for i in rec.get("implicated", [])], rec.get("last_frame", 0))
        n += 1
        nw += len(windows)
    typer.echo(f"Re-scored {n} clips -> {nw} windows in {out.resolve()}")


@app.command("nexar-diagnose")
def nexar_diagnose(run: Path = typer.Option(..., help="Nexar run folder"),
                   pre_s: float = typer.Option(1.5, help="Seconds before alert counted as 'event'"),
                   post_s: float = typer.Option(0.5, help="Seconds after event counted as 'event'")):
    """Which signals/features separate near-miss moments from normal driving? (step-level AUC)"""
    from .eval.nexar_eval import feature_separability

    recs = [json.loads(f.read_text()) for f in sorted((run / "clips").glob("*.json"))]
    table = feature_separability(recs, pre_s, post_s)
    typer.echo(f"{'feature':24s} {'AUC':>6s} {'event mean':>11s} {'normal mean':>12s}   (AUC 0.5 = useless, 1.0 = perfect)")
    for name, auc, me, mn in table:
        typer.echo(f"{name:24s} {auc:6.3f} {me:11.3f} {mn:12.3f}")


@app.command("nexar-eval")
def nexar_eval(run: Path = typer.Option(..., help="Folder written by `sotif-miner nexar`"),
               root: Path = typer.Option(..., help="Nexar dataset root"),
               split: str = typer.Option("train"),
               bootstrap: int = typer.Option(0, help="Bootstrap resamples for 95% CIs (e.g. 2000); 0 = off"),
               out_metrics: Path | None = typer.Option(None, help="Write metrics (with CIs) to this JSON"),
               metrics_kind: str = typer.Option("", help="Stamp _meta.kind in out-metrics, e.g. held_out")):
    """Score a Nexar run: recall, early recall before alert, lead time, false alarms, compression / AP."""
    from .datasets.nexar import load_nexar
    from .eval.nexar_eval import ClipResult, evaluate_test, evaluate_train

    clips = {c.clip_id: c for c in load_nexar(root, split)}
    results = []
    for f in sorted((run / "clips").glob("*.json")):
        rec, samples, windows = _load_clip(f)
        if rec["clip_id"] in clips:
            results.append(ClipResult(clips[rec["clip_id"]], samples, windows, rec["duration_s"]))
    if not results:
        raise typer.BadParameter(f"no finished clips under {run / 'clips'}")
    metrics = evaluate_train(results) if split == "train" else evaluate_test(results)
    if bootstrap:
        from .eval.bootstrap import bootstrap_test_ap, bootstrap_train

        if split == "train":
            metrics["ci"] = bootstrap_train(results, ("recall", "early_recall_before_alert", "median_lead_time_s",
                                                      "negative_windows_per_minute", "compression"), n=bootstrap)
        else:
            metrics["ci"] = bootstrap_test_ap(results, n=bootstrap)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if out_metrics:
        payload = metrics["ci"] if "ci" in metrics else metrics
        if metrics_kind:
            payload = {**payload, "_meta": {"kind": metrics_kind, "split": split, "n": len(results)}}
        out_metrics.write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps({k: v for k, v in metrics.items() if k != "by_group"}, indent=2, default=str))
    typer.echo(f"Full metrics (incl. weather/light/scene breakdown): {(run / 'metrics.json').resolve()}")


if __name__ == "__main__":
    app()
