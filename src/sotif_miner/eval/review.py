"""Blind review of Stage-1 windows: are "false alarms" on Nexar normal clips unlabelled close calls?

Three arms, shuffled together, judged on real video:

* flagged_normal   - every window the model fired on a normal (negative) clip: the false alarms;
* unflagged_normal - random windows of normal clips that the model did NOT flag: the base rate of close calls
                     in ordinary driving, so "the model finds close calls more often than chance" is testable;
* near_miss        - windows around the *labelled* event of positive clips (from the labels, not the model):
                     a positive control showing whether the review can recognise a near-miss at all.

`build_review` cuts each window (plus context) from the source video with ffmpeg into anonymously named MP4s
(the clip id reveals the Nexar label: positives 00xxx, negatives 01xxx) and writes a blind page, review.html,
and the answer key, review_key.json. `judge_review` runs an independent VLM judge on the same clips with no
telemetry (so, unlike diagnosis, it cannot just restate the trigger). `score_review` compares the arms with
Wilson CIs and a Fisher exact test, and reports inter-rater agreement (Cohen's kappa) between humans and the judge.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

VERDICTS = [("close_call", "Real close call / collision"), ("hard_brake", "Hard braking, no conflict"),
            ("normal", "Normal driving"), ("cant_tell", "Can't tell")]
VERDICT_KEYS = [v for v, _ in VERDICTS]
CAUSES = ["adjacent_lane", "lead_braking", "stopped_traffic", "turning_vehicle", "cut_in", "pedestrian_cyclist",
          "parked_vehicle", "glare_night_weather", "camera_artifact", "other"]
KINDS = ["flagged_normal", "unflagged_normal", "near_miss"]


# ----------------------------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion k/n with its Wilson 95% interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fisher_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided Fisher exact p-value that rate 1 (k1/n1) exceeds rate 2 (k2/n2)."""
    from math import comb

    total, m = n1 + n2, k1 + k2
    denom = comb(total, m)
    return sum(comb(n1, x) * comb(n2, m - x) for x in range(k1, min(n1, m) + 1)) / denom if denom else float("nan")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ----------------------------------------------------------------------------------------------- building

def _overlaps(a0: float, a1: float, spans: list[tuple[float, float]], margin: float) -> bool:
    return any(a0 < e + margin and a1 > s - margin for s, e in spans)


def collect_items(run: Path, n_near_miss: int = 50, n_unflagged: int = 50, seed: int = 0,
                  unflagged_len_s: float = 6.0, warmup_s: float = 2.0) -> list[dict]:
    rng = random.Random(seed)
    flagged, unflagged, near = [], [], []
    for f in sorted((run / "clips").glob("*.json")):
        rec = json.loads(f.read_text())
        src, dur = rec.get("source"), float(rec.get("duration_s") or 0)
        if not src or dur <= 0:
            continue
        spans = [(w["start_frame"] / w["fps"], w["end_frame"] / w["fps"]) for w in rec.get("windows", [])]
        base = {"clip_id": rec["clip_id"], "source": src, "duration_s": dur}
        if not rec.get("positive"):
            flagged += [{**base, "kind": "flagged_normal", "start_s": s, "end_s": e} for s, e in spans]
            starts = [warmup_s + 0.5 * i for i in range(int((dur - warmup_s - unflagged_len_s) / 0.5) + 1)]
            free = [t for t in starts if not _overlaps(t, t + unflagged_len_s, spans, 1.0)]
            if free:
                t = rng.choice(free)
                unflagged.append({**base, "kind": "unflagged_normal", "start_s": t, "end_s": t + unflagged_len_s})
        elif rec.get("event_s") is not None:
            ev = float(rec["event_s"])
            al = float(rec["alert_s"]) if rec.get("alert_s") is not None else ev
            s, e = max(0.0, al - 3.0), min(dur, ev + 1.5)
            if e - s < 4.0:
                s = max(0.0, e - 4.0)
            near.append({**base, "kind": "near_miss", "start_s": s, "end_s": e})
    items = (flagged + rng.sample(unflagged, min(n_unflagged, len(unflagged)))
             + rng.sample(near, min(n_near_miss, len(near))))
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["item"] = i
    return items


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError("ffmpeg not found: pip install -e \".[review]\" (bundles ffmpeg)") from e
    return imageio_ffmpeg.get_ffmpeg_exe()


def cut_clip(src: str, start_s: float, end_s: float, out: Path, height: int = 480, exe: str | None = None) -> None:
    """Re-encode [start_s, end_s] of `src` to a browser-playable, metadata-free H.264 MP4 without audio."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe or ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(0.0, start_s):.3f}", "-i", src,
           "-t", f"{max(0.5, end_s - start_s):.3f}", "-an", "-map_metadata", "-1", "-vf", f"scale=-2:{height}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


def build_review(run: Path, n_near_miss: int = 50, n_unflagged: int = 50, seed: int = 0, context_s: float = 1.0,
                 progress=None) -> tuple[Path, Path, dict[str, int]]:
    """Cut every item's clip into <run>/review_clips/NNN.mp4 and write review.html + review_key.json."""
    items = collect_items(run, n_near_miss, n_unflagged, seed)
    if not items:
        raise ValueError(f"no windows/clips with a source video under {run / 'clips'}")
    clips_dir = run / "review_clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    exe = ffmpeg_exe()
    for n, it in enumerate(items, 1):
        it["clip_start_s"] = max(0.0, it["start_s"] - context_s)
        it["clip_end_s"] = min(it["duration_s"], it["end_s"] + context_s)
        cut_clip(it["source"], it["clip_start_s"], it["clip_end_s"], clips_dir / f"{it['item']:03d}.mp4", exe=exe)
        if progress:
            progress(n, len(items))
    public = [{"item": it["item"], "src": f"review_clips/{it['item']:03d}.mp4"} for it in items]
    page = (PAGE.replace("__ITEMS__", json.dumps(public)).replace("__VERDICTS__", json.dumps(VERDICTS))
            .replace("__CAUSES__", json.dumps(CAUSES)).replace("__TAG__", json.dumps(f"{run.name}-{seed}")))
    out, key = run / "review.html", run / "review_key.json"
    out.write_text(page, encoding="utf-8")
    key.write_text(json.dumps({it["item"]: {k: it[k] for k in ("kind", "clip_id", "start_s", "end_s",
                                                                "clip_start_s", "clip_end_s")}
                               for it in items}, indent=2))
    return out, key, {k: sum(it["kind"] == k for it in items) for k in KINDS}


# ----------------------------------------------------------------------------------------------- VLM judge

class Judgement(BaseModel):
    verdict: Literal["close_call", "hard_brake", "normal", "cant_tell"]
    other_road_user: str = Field(default="", description="the road user involved, if any")
    reason: str = Field(description="one sentence: what is visible that supports the verdict")
    confidence: float = Field(ge=0.0, le=1.0)


JUDGE_SYSTEM = ("You review dashcam footage for a traffic-safety study. Judge only what is visible in the frames. "
                "Most footage is ordinary driving; do not assume something happened.")

JUDGE_PROMPT = """These {n} frames are in time order, about {dt:.2f} s apart, from a dashcam video of {dur:.1f} s.
Decide what happens to the camera car:
- close_call: a collision, or another road user comes dangerously close or forces an evasive action
  (hard braking or swerving to avoid a conflict);
- hard_brake: the camera car brakes or stops sharply, but no other road user is in conflict with it;
- normal: ordinary driving, including routine stops and slow traffic;
- cant_tell: the frames do not show enough to decide.
Reply with JSON only."""


def sample_frames(clip: Path, n: int) -> list:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = sorted({round(i * (total - 1) / max(1, n - 1)) for i in range(n)}) if total else []
        out = []
        for k in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, img = cap.read()
            if ok:
                out.append(img)
        return out
    finally:
        cap.release()


def judge_review(run: Path, backend, n_frames: int = 10, out_csv: Path | None = None, progress=None) -> Path:
    """Blind VLM verdict per review item (resumable). Writes a CSV in the same format as the human export."""
    key = json.loads((run / "review_key.json").read_text())
    out_csv = out_csv or run / "review_judge.csv"  # fixed name: model names contain ':' (invalid on Windows)
    done = set()
    if out_csv.exists():
        done = {r["item"] for r in csv.DictReader(out_csv.open(encoding="utf-8"))}
    new = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["item", "verdict", "causes", "note", "reviewer", "confidence"])
        items = sorted(key, key=int)
        for n, item in enumerate(items, 1):
            if item in done:
                continue
            k = key[item]  # only the clip times are used: the judge never sees the kind
            frames = sample_frames(run / "review_clips" / f"{int(item):03d}.mp4", n_frames)
            dur = k["clip_end_s"] - k["clip_start_s"]
            if not frames:
                w.writerow([item, "cant_tell", "", "no frames decoded", f"vlm:{backend.name}", 0.0])
                continue
            prompt = JUDGE_PROMPT.format(n=len(frames), dt=dur / max(1, len(frames) - 1), dur=dur)
            try:
                j = backend.generate(frames, prompt, Judgement, system=JUDGE_SYSTEM)
                w.writerow([item, j.verdict, "", j.reason, f"vlm:{backend.name}", f"{j.confidence:.2f}"])
            except Exception as e:  # noqa: BLE001 - one bad item must not stop the run
                w.writerow([item, "cant_tell", "", f"judge error: {e}", f"vlm:{backend.name}", 0.0])
            fh.flush()
            if progress:
                progress(n, len(items))
    return out_csv


# ----------------------------------------------------------------------------------------------- scoring

def _read_answers(path: Path) -> tuple[str, dict[int, dict]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rater = next((r.get("reviewer") for r in rows if r.get("reviewer")), "") or path.stem
    return rater, {int(r["item"]): {"verdict": r.get("verdict", ""),
                                     "causes": [c for c in (r.get("causes") or "").split(";") if c]}
                   for r in rows if r.get("verdict")}


def _arm_stats(ans: dict[int, dict], key: dict[int, dict], kind: str) -> dict:
    rs = [a for i, a in ans.items() if key.get(i, {}).get("kind") == kind]
    n = len(rs)
    out = {"n": n, "verdicts": {}}
    for v in VERDICT_KEYS:
        k = sum(a["verdict"] == v for a in rs)
        p, lo, hi = wilson(k, n)
        out["verdicts"][v] = {"k": k, "share": p, "ci95": [lo, hi]}
    errs = [a for a in rs if a["verdict"] in ("normal", "hard_brake")]
    out["causes_of_non_conflicts"] = {c: sum(c in a["causes"] for a in errs) for c in CAUSES}
    return out


def score_review(answer_csvs: list[Path], key_json: Path) -> dict:
    key = {int(k): v for k, v in json.loads(key_json.read_text()).items()}
    raters = {}
    for p in answer_csvs:
        name, ans = _read_answers(p)
        while name in raters:
            name += "'"
        raters[name] = ans
    out: dict = {"items": {k: sum(v["kind"] == k for v in key.values()) for k in KINDS}, "raters": {}}
    for name, ans in raters.items():
        arms = {kind: _arm_stats(ans, key, kind) for kind in KINDS}
        f, u = arms["flagged_normal"], arms["unflagged_normal"]
        kf, nf = f["verdicts"]["close_call"]["k"], f["n"]
        ku, nu = u["verdicts"]["close_call"]["k"], u["n"]
        out["raters"][name] = {
            "answered": len(ans), "arms": arms,
            "near_miss_recognised": arms["near_miss"]["verdicts"]["close_call"],
            "flagged_vs_unflagged_close_call": {
                "flagged": kf / nf if nf else float("nan"), "unflagged": ku / nu if nu else float("nan"),
                "difference": (kf / nf - ku / nu) if nf and nu else float("nan"),
                "fisher_p_one_sided": fisher_greater(kf, nf, ku, nu) if nf and nu else float("nan")}}
    names = list(raters)
    out["agreement"] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = raters[names[i]], raters[names[j]]
            common = sorted(set(a) & set(b))
            va, vb = [a[c]["verdict"] for c in common], [b[c]["verdict"] for c in common]
            out["agreement"][f"{names[i]} vs {names[j]}"] = {
                "n": len(common), "kappa_4class": cohen_kappa(va, vb),
                "kappa_close_call": cohen_kappa([x == "close_call" for x in va], [x == "close_call" for x in vb]),
                "raw_agreement": sum(x == y for x, y in zip(va, vb, strict=True)) / len(common) if common else None}
    return out


# ----------------------------------------------------------------------------------------------- page

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind Window Review</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#ddd;--acc:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9d9d97;--card:#21211f;--line:#3a3a37;--acc:#7ea0ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
header{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
h1{font-size:18px;margin:0 auto 0 0} .muted{color:var(--muted)}
button,input[type=text]{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button{cursor:pointer} button.primary{background:var(--acc);color:#fff;border-color:var(--acc)} button.on{outline:2px solid var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
video{display:block;width:100%;max-width:min(100%,calc(55vh*16/9));margin:0 auto;border-radius:6px;background:#000}
.ctl{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin:8px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:6px}
@media (max-width:700px){.cols{grid-template-columns:1fr}}
label{display:block;padding:3px 0;cursor:pointer} kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:12px}
textarea{width:100%;box-sizing:border-box;font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.bar{height:6px;background:var(--line);border-radius:3px;margin:8px 0}.bar i{display:block;height:100%;background:var(--acc);border-radius:3px}
</style></head><body><main>
<header><h1>Blind window review</h1><input type="text" id="who" placeholder="Your name" size="12">
<span class="muted" id="prog"></span>
<button onclick="go(-1)">&larr; Prev</button><button onclick="go(1)">Next &rarr;</button>
<button class="primary" onclick="exportCsv()">Export CSV</button></header>
<p class="muted">Each item is a short dashcam clip; the moment of interest is around the middle. Judge only what you see:
did the camera car collide, or nearly collide, with another road user? Keys: <kbd>1</kbd>&ndash;<kbd>4</kbd> verdict,
<kbd>&rarr;</kbd>/<kbd>&larr;</kbd> next/prev, <kbd>Space</kbd> play/pause, <kbd>S</kbd> slow motion, <kbd>R</kbd> replay.
Answers are kept in this browser; press <b>Export CSV</b> when done.</p>
<div class="bar"><i id="pbar"></i></div>
<div class="card"><video id="vid" muted playsinline loop autoplay></video>
<div class="ctl"><button onclick="rate(0.25)" id="r0.25">0.25&times;</button><button onclick="rate(0.5)" id="r0.5">0.5&times;</button>
<button onclick="rate(1)" id="r1">1&times;</button><button onclick="replay()">Replay</button></div>
<div class="cols"><div><b>Verdict</b><div id="verdicts"></div></div>
<div><b>What drove it?</b> <span class="muted">(any)</span><div id="causes"></div>
<textarea id="note" rows="2" placeholder="Note (optional)"></textarea></div></div></div>
</main><script>
const ITEMS=__ITEMS__, VERDICTS=__VERDICTS__, CAUSES=__CAUSES__, KEY="blind-review-"+__TAG__;
let ans={}, who=""; try{ans=JSON.parse(localStorage.getItem(KEY)||"{}");who=localStorage.getItem(KEY+"-who")||""}catch(e){}
let i=0, speed=1; const vid=document.getElementById("vid"), whoEl=document.getElementById("who"); whoEl.value=who;
whoEl.oninput=()=>{who=whoEl.value;try{localStorage.setItem(KEY+"-who",who)}catch(e){}};
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function cur(){return ITEMS[i]} function a(){return ans[cur().item]||(ans[cur().item]={v:"",c:[],n:""})}
function rate(r){speed=r;vid.playbackRate=r;document.querySelectorAll(".ctl button[id]").forEach(b=>b.classList.toggle("on",b.id==="r"+r))}
function replay(){vid.currentTime=0;vid.play()}
function render(){const r=a(); vid.src=cur().src; vid.onloadeddata=()=>{vid.playbackRate=speed}; vid.play().catch(()=>{});
 document.getElementById("verdicts").innerHTML=VERDICTS.map(([k,t],j)=>`<label><input type="radio" name="v" value="${k}" ${r.v===k?"checked":""}> <kbd>${j+1}</kbd> ${t}</label>`).join("");
 document.getElementById("causes").innerHTML=CAUSES.map(k=>`<label><input type="checkbox" value="${k}" ${r.c.includes(k)?"checked":""}> ${k.replace(/_/g," ")}</label>`).join("");
 document.getElementById("note").value=r.n;
 document.querySelectorAll('#verdicts input').forEach(e=>e.onchange=()=>{a().v=e.value;save();progress()});
 document.querySelectorAll('#causes input').forEach(e=>e.onchange=()=>{const c=a().c;e.checked?c.push(e.value):c.splice(c.indexOf(e.value),1);save()});
 progress()}
function progress(){const done=ITEMS.filter(x=>ans[x.item]&&ans[x.item].v).length;
 document.getElementById("prog").textContent=`Item ${i+1} / ${ITEMS.length} \\u00b7 ${done} answered`;
 document.getElementById("pbar").style.width=(100*done/ITEMS.length)+"%"}
function go(d){i=Math.min(ITEMS.length-1,Math.max(0,i+d));render()}
document.getElementById("note").oninput=e=>{a().n=e.target.value;save()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT")return;
 if(e.key>="1"&&e.key<="4"){a().v=VERDICTS[+e.key-1][0];save();render()}
 else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1);
 else if(e.key===" "){e.preventDefault();vid.paused?vid.play():vid.pause()}
 else if(e.key==="s"||e.key==="S")rate(speed===1?0.25:1); else if(e.key==="r"||e.key==="R")replay()});
function exportCsv(){const rows=[["item","verdict","causes","note","reviewer"]];
 ITEMS.forEach(x=>{const r=ans[x.item]||{v:"",c:[],n:""};rows.push([x.item,r.v,r.c.join(";"),r.n,who])});
 const csv=rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(",")).join("\\n");
 const b=new Blob([csv],{type:"text/csv"}),u=URL.createObjectURL(b),l=document.createElement("a");
 l.href=u;l.download="review_answers"+(who?"_"+who.replace(/\\W+/g,"_"):"")+".csv";l.click();URL.revokeObjectURL(u)}
rate(1); render();
</script></body></html>"""
