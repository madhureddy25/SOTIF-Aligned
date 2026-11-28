"""Blind human review of Stage-1 windows: are "false alarms" on Nexar normal clips unlabelled close calls?

`build_review` writes an HTML page into the run folder, copies of the keyframes under anonymous names
(`review_frames/<item>/<n>.jpg`), and a separate key file. The page shows only the plain keyframes (no model boxes, timestamp stamp cropped), in a
shuffled order that mixes the false-alarm windows of normal clips with near-miss *controls* (windows of
positive clips that overlap the labelled event). The reviewer never sees clip ids, labels or scores.
`score_review` joins the exported CSV with the key.
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
from pathlib import Path

VERDICTS = [("close_call", "Real close call / conflict"), ("hard_brake", "Hard braking, no conflict"),
            ("normal", "Normal driving"), ("cant_tell", "Can't tell")]
CAUSES = ["adjacent_lane", "lead_braking", "stopped_traffic", "turning_vehicle", "cut_in", "pedestrian_cyclist",
          "parked_vehicle", "glare_night_weather", "camera_artifact", "other"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion k/n with its Wilson 95% interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _frames(ev_dir: Path, ev: dict) -> list[str]:
    names = [Path(p.replace("\\", "/")).name for p in ev.get("unmarked") or []]
    return [n for n in names if (ev_dir / n).exists()]


def collect_items(run: Path, n_controls: int, seed: int, pre_s: float = 1.5, post_s: float = 0.5) -> list[dict]:
    """False-alarm windows of negative clips + up to n_controls positive windows overlapping the event."""
    fa, ctrl = [], []
    for f in sorted((run / "evidence").glob("*/evidence.json")):
        ev = json.loads(f.read_text())
        frames = _frames(f.parent, ev)
        if not frames:
            continue
        w = ev["bundle"]["window"]
        item = {"window": w["clip_id"], "dir": f.parent.name, "frames": frames, "positive": bool(ev.get("positive")),
                "start_s": w["start_frame"] / w["fps"], "end_s": w["end_frame"] / w["fps"]}
        if not item["positive"]:
            fa.append({**item, "kind": "false_alarm"})
        elif ev.get("event_s") is not None:
            al = ev.get("alert_s") if ev.get("alert_s") is not None else ev["event_s"]
            if item["start_s"] <= ev["event_s"] + post_s and item["end_s"] >= al - pre_s:
                ctrl.append({**item, "kind": "control"})
    rng = random.Random(seed)
    items = fa + rng.sample(ctrl, min(n_controls, len(ctrl)))
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["item"] = i
    return items


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind Window Review</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#ddd;--acc:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9d9d97;--card:#21211f;--line:#3a3a37;--acc:#7ea0ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
h1{font-size:18px;margin:0 auto 0 0} .muted{color:var(--muted)}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.big{position:relative;overflow:hidden;border-radius:6px;background:#000;max-width:min(100%,calc(52vh*16/9));margin:0 auto}
.big img{display:block;width:100%;margin-top:-3.9%}
.strip{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin:8px auto;max-width:min(100%,calc(52vh*16/9))}
.strip div{overflow:hidden;border-radius:4px;cursor:pointer;outline:2px solid transparent}
.strip div.on{outline-color:var(--acc)} .strip img{display:block;width:100%;margin-top:-3.9%}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px}
@media (max-width:700px){.cols{grid-template-columns:1fr}.strip{grid-template-columns:repeat(3,1fr)}}
label{display:block;padding:3px 0;cursor:pointer} kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:12px}
textarea{width:100%;box-sizing:border-box;font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.bar{height:6px;background:var(--line);border-radius:3px;margin:8px 0}.bar i{display:block;height:100%;background:var(--acc);border-radius:3px}
</style></head><body><main>
<header><h1>Blind window review</h1><span class="muted" id="prog"></span>
<button onclick="go(-1)">&larr; Prev</button><button onclick="go(1)">Next &rarr;</button>
<button class="primary" onclick="exportCsv()">Export CSV</button></header>
<p class="muted">Each item is a 4&ndash;8 s dashcam window. Judge only what you see: did a real conflict with another road
user happen (or nearly)? Keys: <kbd>1</kbd>&ndash;<kbd>4</kbd> verdict, <kbd>&rarr;</kbd>/<kbd>&larr;</kbd> next/prev, <kbd>Space</kbd> play.
Answers are kept in this browser; press <b>Export CSV</b> when done.</p>
<div class="bar"><i id="pbar"></i></div>
<div class="card"><div class="big"><img id="big" alt="keyframe"></div><div class="strip" id="strip"></div>
<div class="cols"><div><b>Verdict</b><div id="verdicts"></div></div>
<div><b>What drove it?</b> <span class="muted">(any)</span><div id="causes"></div>
<textarea id="note" rows="2" placeholder="Note (optional)"></textarea></div></div></div>
</main><script>
const ITEMS=__ITEMS__, VERDICTS=__VERDICTS__, CAUSES=__CAUSES__, KEY="blind-review-"+__TAG__;
let ans={}; try{ans=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0, f=0, timer=null;
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function cur(){return ITEMS[i]} function a(){return ans[cur().item]||(ans[cur().item]={v:"",c:[],n:""})}
function show(k){f=k;document.getElementById("big").src=cur().frames[f];
 document.querySelectorAll(".strip div").forEach((d,j)=>d.classList.toggle("on",j===f))}
function render(){const it=cur(),r=a();
 document.getElementById("strip").innerHTML=it.frames.map((p,j)=>`<div onclick="show(${j})"><img src="${p}" alt=""></div>`).join("");
 show(0);
 document.getElementById("verdicts").innerHTML=VERDICTS.map(([k,t],j)=>`<label><input type="radio" name="v" value="${k}" ${r.v===k?"checked":""}> <kbd>${j+1}</kbd> ${t}</label>`).join("");
 document.getElementById("causes").innerHTML=CAUSES.map(k=>`<label><input type="checkbox" value="${k}" ${r.c.includes(k)?"checked":""}> ${k.replace(/_/g," ")}</label>`).join("");
 document.getElementById("note").value=r.n;
 document.querySelectorAll('#verdicts input').forEach(e=>e.onchange=()=>{a().v=e.value;save();progress()});
 document.querySelectorAll('#causes input').forEach(e=>e.onchange=()=>{const c=a().c;e.checked?c.push(e.value):c.splice(c.indexOf(e.value),1);save()});
 progress()}
function progress(){const done=ITEMS.filter(x=>ans[x.item]&&ans[x.item].v).length;
 document.getElementById("prog").textContent=`Item ${i+1} / ${ITEMS.length} · ${done} answered`;
 document.getElementById("pbar").style.width=(100*done/ITEMS.length)+"%"}
function go(d){i=Math.min(ITEMS.length-1,Math.max(0,i+d));render()}
function play(){if(timer){clearInterval(timer);timer=null;return}timer=setInterval(()=>show((f+1)%cur().frames.length),350)}
document.getElementById("note").oninput=e=>{a().n=e.target.value;save()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA")return;
 if(e.key>="1"&&e.key<="4"){a().v=VERDICTS[+e.key-1][0];save();render()}
 else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1);
 else if(e.key===" "){e.preventDefault();play()}});
function exportCsv(){const rows=[["item","verdict","causes","note"]];
 ITEMS.forEach(x=>{const r=ans[x.item]||{v:"",c:[],n:""};rows.push([x.item,r.v,r.c.join(";"),r.n])});
 const csv=rows.map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(",")).join("\\n");
 const b=new Blob([csv],{type:"text/csv"}),u=URL.createObjectURL(b),l=document.createElement("a");
 l.href=u;l.download="review_answers.csv";l.click();URL.revokeObjectURL(u)}
render();
</script></body></html>"""


def build_review(run: Path, n_controls: int = 25, seed: int = 0) -> tuple[Path, Path, int, int]:
    """Write <run>/review.html (blind) and <run>/review_key.json (truth). Returns paths and counts."""
    items = collect_items(run, n_controls, seed)
    if not items:
        raise ValueError(f"no windows with evidence under {run / 'evidence'} (run `diagnose` pass A first)")
    # Copy frames under anonymous names: the evidence folder names carry the clip id, and on Nexar the
    # clip number reveals the label (positives 00xxx, negatives 01xxx), so they must not reach the page.
    import shutil

    frames_dir = run / "review_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    public = []
    for it in items:
        d = frames_dir / f"{it['item']:03d}"
        d.mkdir(parents=True)
        rel = []
        for j, n in enumerate(it["frames"], 1):
            shutil.copyfile(run / "evidence" / it["dir"] / n, d / f"{j}.jpg")
            rel.append(f"review_frames/{it['item']:03d}/{j}.jpg")
        public.append({"item": it["item"], "frames": rel})
    page = (PAGE.replace("__ITEMS__", json.dumps(public)).replace("__VERDICTS__", json.dumps(VERDICTS))
            .replace("__CAUSES__", json.dumps(CAUSES)).replace("__TAG__", json.dumps(html.escape(f"{run.name}-{seed}"))))
    out, key = run / "review.html", run / "review_key.json"
    out.write_text(page, encoding="utf-8")
    key.write_text(json.dumps({it["item"]: {k: it[k] for k in ("window", "kind", "start_s", "end_s")}
                               for it in items}, indent=2))
    n_fa = sum(it["kind"] == "false_alarm" for it in items)
    return out, key, n_fa, len(items) - n_fa


def score_review(answers_csv: Path, key_json: Path) -> dict:
    """Join answers with the key: verdict shares with Wilson CIs per kind, control recognition, cause counts."""
    key = {int(k): v for k, v in json.loads(key_json.read_text()).items()}
    rows = list(csv.DictReader(answers_csv.open(encoding="utf-8-sig")))
    out: dict = {"answered": 0, "unanswered": 0}
    by_kind: dict[str, list[dict]] = {"false_alarm": [], "control": []}
    for r in rows:
        k = key.get(int(r["item"]))
        if k is None:
            continue
        if not r.get("verdict"):
            out["unanswered"] += 1
            continue
        out["answered"] += 1
        by_kind[k["kind"]].append({"verdict": r["verdict"], "causes": [c for c in r.get("causes", "").split(";") if c]})
    for kind, rs in by_kind.items():
        n = len(rs)
        out[kind] = {"n": n, "verdicts": {}}
        for v, _ in VERDICTS:
            p, lo, hi = wilson(sum(x["verdict"] == v for x in rs), n)
            out[kind]["verdicts"][v] = {"k": sum(x["verdict"] == v for x in rs), "share": p, "ci95": [lo, hi]}
        out[kind]["causes"] = {c: sum(c in x["causes"] for x in rs) for c in CAUSES}
        if kind == "false_alarm":
            errs = [x for x in rs if x["verdict"] in ("normal", "hard_brake")]
            out[kind]["causes_of_true_errors"] = {c: sum(c in x["causes"] for x in errs) for c in CAUSES}
    return out
