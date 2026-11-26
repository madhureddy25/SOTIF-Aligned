"""Consolidated, self-contained HTML report: the industry-facing deliverable.

Pulls together Stage 1 mining metrics (train CV and, when present, held-out test AP with a
bootstrap CI), the Stage 3 diagnosis breakdown, and the Stage 4 fleet taxonomy, with a provenance
block (code version, config, dataset counts) and a standing limitations section. No JS, no external
assets -- one file a stakeholder can open or email.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ATTR_LABEL = {
    "AGENT_BEHAVIOR": "Closing / interaction conflict", "DYNAMIC_OCCLUSION": "Occluded-actor emergence",
    "EPISTEMIC_PERCEPTION_GAP": "Perception blind spot (fixable)", "ALEATORY_OPTICAL": "Optical-limit miss (ODD)",
    "INCONCLUSIVE": "Unresolved / false-positive window",
}
BAR = {"AGENT_BEHAVIOR": "var(--c1)", "DYNAMIC_OCCLUSION": "var(--c2)", "EPISTEMIC_PERCEPTION_GAP": "var(--c3)",
       "ALEATORY_OPTICAL": "var(--c4)", "INCONCLUSIVE": "var(--c5)"}


def _pkg_version() -> str:
    try:
        return version("sotif-miner")
    except PackageNotFoundError:
        return "dev"


def _bars(counts: dict[str, int], total: int) -> str:
    rows = []
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * v / total if total else 0
        rows.append(
            f'<div class="barrow"><span class="blab">{html.escape(ATTR_LABEL.get(k, k))}</span>'
            f'<span class="btrack"><span class="bfill" style="width:{pct:.1f}%;background:{BAR.get(k, "var(--c5)")}">'
            f'</span></span><span class="bval">{v} ({pct:.0f}%)</span></div>')
    return "\n".join(rows)


def _ci_str(ci) -> str:
    if not ci or ci[0] != ci[0]:  # nan
        return ""
    return f" <span class='ci'>95% CI [{ci[0]:.3f}, {ci[1]:.3f}]</span>"


def _metrics_table(metrics: dict, label: str) -> list[str]:
    rows = [f'<table><tr><th>Metric ({label})</th><th>Value</th><th>95% CI</th></tr>']
    for k, lab in [("recall", "Near-miss recall"), ("early_recall_before_alert", "Flagged before alert"),
                   ("median_lead_time_s", "Median lead time (s)"), ("negative_windows_per_minute", "False windows/min"),
                   ("compression", "Video discarded")]:
        d = metrics.get(k, {})
        v, ci = d.get("value"), d.get("ci95")
        vs = f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "—"
        cs = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] == ci[0] else "—"
        rows.append(f"<tr><td>{lab}</td><td>{vs}</td><td>{cs}</td></tr>")
    rows.append("</table>")
    return rows


def build_report(run: Path, taxonomy_dir: Path | None, dataset_counts: dict | None = None,
                 test_metrics: dict | None = None, train_metrics: dict | None = None,
                 heldout_metrics: dict | None = None) -> str:
    diag = json.loads((run / "diagnosis_summary.json").read_text()) if (run / "diagnosis_summary.json").exists() else {}
    clusters = json.loads((taxonomy_dir / "clusters.json").read_text()) if taxonomy_dir and (taxonomy_dir / "clusters.json").exists() else []
    n_cards = diag.get("cards", 0)
    attr = diag.get("attribution", {})
    vlm = diag.get("vlm", {})
    total_events = sum(c["size"] for c in clusters)
    area2 = sum(c["size"] for c in clusters if c["iso_21448_area"].startswith("AREA_2"))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def section(t):
        return f'<h2>{html.escape(t)}</h2>'

    out = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOTIF Dashcam Miner — Report</title><style>
:root{{--bg:#fff;--fg:#14202e;--mut:#5a6b7b;--card:#f4f7fb;--line:#d6dee8;--acc:#1f5fa8;
--c1:#2f6fb0;--c2:#3aa07a;--c3:#c9772e;--c4:#9a4ddb;--c5:#8a97a6;}}
:root:not([data-theme=light]) @media (prefers-color-scheme:dark){{}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0f1621;--fg:#e6edf5;--mut:#9fb0c2;
--card:#17202d;--line:#263241;--acc:#5fa0e8;--c5:#5d6b7b;}}}}
:root[data-theme=dark]{{--bg:#0f1621;--fg:#e6edf5;--mut:#9fb0c2;--card:#17202d;--line:#263241;--acc:#5fa0e8;--c5:#5d6b7b;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:980px;margin:0 auto;padding:28px 16px 80px}}
h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:19px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--mut);margin:0 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}
.kpi .n{{font-size:26px;font-weight:650}}.kpi .l{{color:var(--mut);font-size:13px;margin-top:2px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
th,td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}}
th{{background:var(--card)}}
.barrow{{display:flex;align-items:center;gap:10px;margin:5px 0}}
.blab{{flex:0 0 240px}}.btrack{{flex:1;background:var(--card);border-radius:6px;height:16px;overflow:hidden}}
.bfill{{display:block;height:100%}}.bval{{flex:0 0 90px;color:var(--mut);font-size:13px}}
.ci{{color:var(--mut);font-weight:400;font-size:13px}}
.pill{{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}}
.a2{{background:rgba(58,160,122,.15)}}.a3{{background:rgba(201,119,46,.15)}}
code{{background:var(--card);padding:1px 5px;border-radius:4px}}
.note{{background:var(--card);border-left:3px solid var(--acc);padding:10px 14px;border-radius:0 8px 8px 0;margin:10px 0}}
small{{color:var(--mut)}}
</style></head><body><div class="wrap">
<h1>SOTIF-Aligned Dashcam Edge-Case Miner — Run Report</h1>
<p class="sub">Generated {ts} · sotif-miner {html.escape(_pkg_version())} · VLM backend {html.escape(str(vlm.get("backend", diag.get("vlm", {}).get("name", "n/a")) or "n/a"))}</p>
<div class="grid">
<div class="kpi"><div class="n">{total_events or n_cards}</div><div class="l">diagnosed events</div></div>
<div class="kpi"><div class="n">{len(clusters)}</div><div class="l">failure modes</div></div>
<div class="kpi"><div class="n">{area2}</div><div class="l">in Area 2 (known)</div></div>
<div class="kpi"><div class="n">{(total_events or n_cards) - area2}</div><div class="l">in Area 3 (novel)</div></div>
<div class="kpi"><div class="n">{vlm.get("json_validity", 0) * 100:.1f}%</div><div class="l">VLM JSON validity</div></div>
</div>"""]

    # Stage 1 — this system MINES near-misses in recorded video; it is not a pre-event predictor.
    out.append(section("Stage 1 — edge-case mining"))
    out.append('<p>The miner flags near-miss windows in recorded dashcam video. Its job is <b>recall with high '
               'compression</b> (find the rare events, discard routine driving), measured on full-length clips.</p>')
    if heldout_metrics:
        out.append('<p><b>Held-out mining — the locked number</b> (unseen full-length clips, disjoint from training):</p>')
        out += _metrics_table(heldout_metrics, "held-out, unseen clips")
    if train_metrics:
        is_cv = train_metrics.get("_meta", {}).get("kind") == "cross_validated_out_of_fold"
        label = "train, cross-validated (out-of-fold)" if is_cv else "train (IN-SAMPLE — optimistic, not generalisation)"
        if not is_cv and train_metrics.get("_meta", {}).get("kind") != "held_out":
            out.append('<p class="note"><b>Warning:</b> the supplied train metrics are in-sample '
                       '(the model was evaluated on clips it trained on), so they overstate performance. '
                       'Pass the <code>stage1_cv_metrics.json</code> written by <code>train-stage1</code> instead.</p>')
        out += _metrics_table(train_metrics, label)
    if not heldout_metrics and not train_metrics:
        out.append('<p class="note">No mining metrics supplied. Run <code>train-stage1</code> for cross-validated '
                   'numbers, and the held-out mining protocol (runbook) for the locked number.</p>')
    if test_metrics:
        ap = test_metrics.get("ap", {})
        chance = test_metrics.get("prevalence", 0)
        out.append(f'<p class="note"><b>Aside — Nexar anticipation benchmark (different task):</b> '
                   f'AP {ap.get("value", float("nan")):.3f}{_ci_str(ap.get("ci95"))}, '
                   f'chance ≈ {chance:.2f}. The Nexar test split truncates every clip 0.5–1.5 s <b>before</b> the event, '
                   f'so it measures <b>predicting a crash before it is visible</b> — a different task this mining '
                   f'system is not designed for. A near-chance score here is expected and does not reflect mining quality.</p>')

    # Diagnosis
    out.append(section("Stage 3 — causal root-cause diagnosis"))
    out.append(f'<p>{n_cards} cards. Deterministic blame attribution (the VLM never chooses the label); '
               f'a critic checks each card against the evidence. Repairs: {diag.get("repaired", 0)}; '
               f'needs review: {diag.get("status", {}).get("needs_review", 0)}; VLM failures: {vlm.get("failures", 0)} '
               f'of {vlm.get("calls", 0)} calls.</p>')
    out.append('<div>' + _bars(attr, n_cards) + '</div>')

    # Taxonomy
    out.append(section("Stage 4 — fleet failure taxonomy"))
    out.append(f'<p>{total_events} events in {len(clusters)} modes. '
               f'<span class="pill a2">Area 2 (recurring, known): {area2}</span> '
               f'<span class="pill a3">Area 3 (novel): {total_events - area2}</span></p>')
    out.append('<table><tr><th>#</th><th>Failure mode</th><th>Size</th><th>Severity</th><th>ISO area</th>'
               '<th>Dominant cause</th><th>Representative clips</th></tr>')
    for i, c in enumerate([c for c in clusters if not c["noise"]][:20], 1):
        dom = max(c["attribution_hist"], key=c["attribution_hist"].get) if c["attribution_hist"] else "-"
        area = '<span class="pill a2">2 known</span>' if c["iso_21448_area"].startswith("AREA_2") else '<span class="pill a3">3 novel</span>'
        reps = ", ".join(c["representative_clip_ids"][:3])
        out.append(f'<tr><td>{i}</td><td>{html.escape(c["name"])}</td><td>{c["size"]}</td>'
                   f'<td>{c["severity"]:.2f}</td><td>{area}</td><td>{html.escape(ATTR_LABEL.get(dom, dom))}</td>'
                   f'<td><small>{html.escape(reps)}</small></td></tr>')
    out.append("</table>")

    # Provenance + limitations
    out.append(section("Provenance & reproducibility"))
    prov = [f"<li>Report generated {ts}; code version <code>{html.escape(_pkg_version())}</code>.</li>"]
    if dataset_counts:
        prov.append("<li>Dataset: " + ", ".join(f"{k} = {v}" for k, v in dataset_counts.items()) + ".</li>")
    prov.append(f"<li>Run folder: <code>{html.escape(str(run))}</code>.</li>")
    out.append("<ul>" + "".join(prov) + "</ul>")

    out.append(section("Limitations (read before use)"))
    out.append("""<ul>
<li><b>Dataset scope.</b> Numbers are from the Nexar Dashcam Collision Prediction set (US urban/suburban,
mostly clear/day). Dense unstructured traffic, animals and non-US road users are under-represented;
do not generalise beyond this operating domain without new data.</li>
<li><b>Perception-failure causes are not independently validated.</b> Nexar labels kinematic near-misses,
not detector failures, so the EPISTEMIC / ALEATORY attributions are internally consistent but have not been
checked against ground-truth perception labels (that needs CARLA counterfactuals or human review).</li>
<li><b>Local VLM.</b> Scene descriptions come from a 4B local model; a small fraction of cards are marked
<code>needs_review</code> and should be read by an engineer before action.</li>
<li><b>No closed loop yet.</b> The taxonomy is not yet fed back into detector fine-tuning (Phase 4).</li>
</ul>""")
    out.append("</div></body></html>")
    return "\n".join(out)
