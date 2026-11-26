"""Stage 4: corpus-level failure taxonomy.

Each validated trigger card becomes a feature vector built from its structured fields (root-cause
attribution, triggering condition, functional insufficiency, causal layer, kinematic hazard, scene
weather/illumination, primary actor, VRU density) plus scaled telemetry (ego-path looming, inverse
TTC, luminance, saturation, sharpness, active tracks) and a feature-hash of the summary text. The
cards are reduced (UMAP if installed, else PCA) and clustered (HDBSCAN), then each cluster is named
deterministically from its dominant fields and ranked by frequency x severity.

ISO 21448 area is decided by recurrence, per SOTIF: a cluster seen >= `min_cluster_size` times is a
recognised (Area 2, known) failure mode; HDBSCAN noise and singletons stay Area 3 (unknown).

Runs fully offline (CPU, no VLM). Optional VLM naming is available but off by default.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..config import is_vru, map_class
from ..perception.embed import HashTextEmbedder

CAT_FIELDS = {
    "attribution": ["EPISTEMIC_PERCEPTION_GAP", "ALEATORY_OPTICAL", "DYNAMIC_OCCLUSION", "AGENT_BEHAVIOR",
                    "INCONCLUSIVE"],
    "triggering_condition": ["none", "low_light", "glare", "rain", "fog", "snow", "motion_blur", "dense_traffic",
                             "occluding_object", "unusual_road_geometry", "out_of_ontology_object",
                             "unusual_actor_behavior"],
    "functional_insufficiency": ["missed_detection", "confidence_flicker", "misclassification", "id_switch",
                                 "track_loss_under_occlusion", "out_of_ontology", "late_detection",
                                 "none_perception_ok"],
    "causal_layer": ["Environment", "Perception", "Tracking", "Agent_Behavior"],
    "kinematic_hazard": ["none", "cut_in", "sudden_emergence", "rapid_closing", "crossing_vru", "dooring",
                         "wrong_way", "dense_weaving"],
    "weather": ["clear", "rain", "fog", "snow", "unknown"],
    "illumination": ["day", "dusk_dawn", "night", "glare", "tunnel", "unknown"],
    "vru_density": ["none", "low", "medium", "high"],
}
ACTORS = ["pedestrian", "rider", "bicycle", "motorcycle", "car", "truck", "bus", "autorickshaw", "animal", "other"]


def _onehot(value: str, vocab: list[str]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v


def card_features(card: dict, text_embedder: HashTextEmbedder | None, text_weight: float = 0.25) -> np.ndarray:
    d = card.get("draft", {})
    sc = card.get("scene", {})
    t = card.get("telemetry", {})
    parts = [
        _onehot(card.get("attribution", ""), CAT_FIELDS["attribution"]),
        _onehot(d.get("triggering_condition", ""), CAT_FIELDS["triggering_condition"]),
        _onehot(d.get("functional_insufficiency", ""), CAT_FIELDS["functional_insufficiency"]),
        _onehot(d.get("causal_layer", ""), CAT_FIELDS["causal_layer"]),
        _onehot(d.get("kinematic_hazard", ""), CAT_FIELDS["kinematic_hazard"]),
        _onehot(sc.get("weather", ""), CAT_FIELDS["weather"]),
        _onehot(sc.get("illumination", ""), CAT_FIELDS["illumination"]),
        _onehot(sc.get("vru_density", ""), CAT_FIELDS["vru_density"]),
        _onehot(map_class("coco", (d.get("primary_actor_class") or "").lower()), ACTORS),
    ]
    ttc = t.get("min_ttc_s")
    inv_ttc = 1.0 / ttc if ttc and ttc > 0 else 0.0
    num = np.array([
        min(t.get("ego_looming", 0.0) / 2.0, 1.5), min(inv_ttc, 1.5), t.get("luminance", 0.5),
        min(t.get("saturation_frac", 0.0) / 0.1, 1.5), t.get("sharpness", 0.9),
        min(t.get("active_tracks", 0) / 20.0, 1.5),
    ], dtype=np.float32)
    vec = np.concatenate(parts + [num])
    if text_embedder is not None:
        txt = (d.get("summary", "") + " " + sc.get("road_layout", ""))
        emb = text_embedder.embed([txt])[0] * text_weight
        vec = np.concatenate([vec, emb.astype(np.float32)])
    return vec


def _reduce(X: np.ndarray, n_components: int, use_umap_min_n: int = 50) -> np.ndarray:
    n_components = min(n_components, X.shape[1], max(2, X.shape[0] - 1))
    if X.shape[0] <= n_components + 1:
        return X
    # UMAP only helps on a large corpus and is non-deterministic on tiny sets; PCA is deterministic
    # and dependency-free, so it is the default and the small-N path.
    if X.shape[0] >= use_umap_min_n:
        try:
            import umap  # noqa: PLC0415

            return umap.UMAP(n_components=n_components, n_neighbors=min(15, X.shape[0] - 1), min_dist=0.0,
                             random_state=0).fit_transform(X)
        except Exception:  # noqa: BLE001 - umap not installed; PCA fallback below
            pass
    from sklearn.decomposition import PCA  # noqa: PLC0415

    return PCA(n_components=n_components, random_state=0).fit_transform(X)


def _cluster(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN  # noqa: PLC0415

    n = X.shape[0]
    if n < max(4, min_cluster_size):
        return np.zeros(n, dtype=int)  # too few: one cluster
    labels = HDBSCAN(min_cluster_size=max(2, min_cluster_size), min_samples=2, copy=True).fit_predict(X)
    if (labels >= 0).sum() == 0:  # everything noise: fall back to k-means
        from sklearn.cluster import KMeans  # noqa: PLC0415

        k = max(2, min(8, n // min_cluster_size))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    return labels


def _dominant(cards: list[dict], path) -> str:
    vals = [path(c) for c in cards if path(c)]
    return Counter(vals).most_common(1)[0][0] if vals else ""


SEV_CONDITION = {"low_light": 0.3, "glare": 0.3, "rain": 0.25, "fog": 0.4, "snow": 0.4}


def _severity(cards: list[dict]) -> float:
    """Mean per-card severity in [0,1]: short-TTC closing, VRU involvement, adverse trigger, and
    perception/optical blame (the actionable failure modes) raise it."""
    s = []
    for c in cards:
        t = c.get("telemetry", {})
        ttc = t.get("min_ttc_s")
        score = 0.0
        if ttc and ttc > 0:
            score += max(0.0, min(1.0, (3.0 - ttc) / 3.0)) * 0.5
        score += min(t.get("ego_looming", 0.0) / 2.0, 1.0) * 0.2
        if is_vru(map_class("coco", (c.get("draft", {}).get("primary_actor_class") or "").lower())):
            score += 0.3
        score += SEV_CONDITION.get(c.get("draft", {}).get("triggering_condition", ""), 0.0)
        if c.get("attribution") in ("EPISTEMIC_PERCEPTION_GAP", "ALEATORY_OPTICAL", "DYNAMIC_OCCLUSION"):
            score += 0.15
        s.append(min(1.0, score))
    return float(np.mean(s)) if s else 0.0


ATTR_PHRASE = {
    "AGENT_BEHAVIOR": "closing/interaction conflict", "DYNAMIC_OCCLUSION": "occluded-actor emergence",
    "EPISTEMIC_PERCEPTION_GAP": "perception blind spot", "ALEATORY_OPTICAL": "optical-limit miss",
    "INCONCLUSIVE": "unresolved window",
}


def _name(cards: list[dict]) -> tuple[str, str]:
    attr = _dominant(cards, lambda c: c.get("attribution"))
    illum = _dominant(cards, lambda c: c.get("scene", {}).get("illumination"))
    weather = _dominant(cards, lambda c: c.get("scene", {}).get("weather"))
    actor = _dominant(cards, lambda c: map_class("coco", (c.get("draft", {}).get("primary_actor_class") or "").lower()))
    cond = _dominant(cards, lambda c: c.get("draft", {}).get("triggering_condition"))
    ctx = " ".join(w for w in [weather if weather not in ("clear", "unknown", "") else "",
                               illum if illum not in ("day", "unknown", "") else ""] if w)
    ctx = ctx or "clear daytime"
    name = f"{ctx.capitalize()} {ATTR_PHRASE.get(attr, attr.lower())} with {actor or 'road user'}"
    trig = f", trigger: {cond}" if cond and cond != "none" else ""
    desc = (f"{len(cards)} events. Dominant root cause {attr}; actor {actor}; context {ctx}{trig}. "
            f"Example: {cards[0].get('draft', {}).get('summary', '')[:140]}")
    return name, desc


@dataclass
class TaxonomyResult:
    clusters: list[dict]
    assignments: dict[str, str]  # clip_id -> cluster_id


def build_taxonomy(cards: list[dict], min_cluster_size: int = 3, n_components: int = 8,
                   with_text: bool = True) -> TaxonomyResult:
    if not cards:
        return TaxonomyResult([], {})
    emb = HashTextEmbedder() if with_text else None
    X = np.vstack([card_features(c, emb) for c in cards])
    Xr = _reduce(X, n_components)
    labels = _cluster(Xr, min_cluster_size)

    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)

    clusters, assignments = [], {}
    cid = 0
    # named clusters first (label >= 0), then noise as per-point Area-3 singletons
    for lab in sorted(groups, key=lambda x: (x < 0, x)):
        idxs = groups[lab]
        members = [cards[i] for i in idxs]
        is_noise = lab < 0
        buckets = [[cards[i]] for i in idxs] if is_noise else [members]
        for bucket in buckets:
            recurring = (not is_noise) and len(bucket) >= min_cluster_size
            name, desc = _name(bucket)
            sev = _severity(bucket)
            ids = [c.get("clip_id", "") for c in bucket]
            cluster_id = f"C{cid:03d}"
            area = "AREA_2_KNOWN_HAZARDOUS" if recurring else "AREA_3_UNKNOWN_HAZARDOUS"
            # severity gives a nudge so a small-but-dangerous mode still ranks
            reps = sorted(bucket, key=lambda c: -(c.get("telemetry", {}).get("ego_looming", 0.0)))[:3]
            clusters.append({
                "cluster_id": cluster_id, "name": name, "description": desc, "size": len(bucket),
                "severity": round(sev, 3), "rank_score": round(len(bucket) * (0.5 + sev), 3),
                "iso_21448_area": area, "member_clip_ids": ids,
                "representative_clip_ids": [c.get("clip_id", "") for c in reps],
                "attribution_hist": dict(Counter(c.get("attribution") for c in bucket)),
                "triggering_hist": dict(Counter(c.get("draft", {}).get("triggering_condition") for c in bucket)),
                "ncap_hist": dict(Counter(x for c in bucket for x in c.get("draft", {}).get("euro_ncap_2026_tags", []))),
                "weather_hist": dict(Counter(c.get("scene", {}).get("weather") for c in bucket)),
                "illumination_hist": dict(Counter(c.get("scene", {}).get("illumination") for c in bucket)),
                "noise": is_noise,
            })
            for cid_ in ids:
                assignments[cid_] = cluster_id
            cid += 1
    clusters, assignments = _merge_by_signature(clusters, assignments, cards, min_cluster_size)
    clusters.sort(key=lambda c: -c["rank_score"])
    return TaxonomyResult(clusters, assignments)


def _context_bucket(cards: list[dict]) -> str:
    weather = _dominant(cards, lambda c: c.get("scene", {}).get("weather"))
    illum = _dominant(cards, lambda c: c.get("scene", {}).get("illumination"))
    w = weather if weather not in ("clear", "unknown", "") else ""
    il = illum if illum not in ("day", "unknown", "") else ""
    return (w + "|" + il) or "clear_day"


def _merge_by_signature(clusters: list[dict], assignments: dict, cards: list[dict], min_cluster_size: int
                        ) -> tuple[list[dict], dict]:
    """Collapse clusters with the same (dominant cause, dominant actor, context) into one fleet
    mode. HDBSCAN/PCA over-splits a dominant mode into several small clusters; for a fleet taxonomy
    the consolidated mode (and its true frequency) is what matters, and it makes the Area 2/3
    recurrence call honest."""
    by_id = {c.get("clip_id", ""): c for c in cards}
    groups: dict[tuple, list[dict]] = {}
    for cl in clusters:
        members = [by_id[i] for i in cl["member_clip_ids"] if i in by_id]
        attr = max(cl["attribution_hist"], key=cl["attribution_hist"].get) if cl["attribution_hist"] else ""
        actor = _dominant(members, lambda c: map_class("coco", (c.get("draft", {}).get("primary_actor_class") or "").lower()))
        sig = (attr, actor, _context_bucket(members))
        groups.setdefault(sig, []).extend(members)

    out, new_asg = [], {}
    for cid, (_sig, members) in enumerate(groups.items()):
        name, desc = _name(members)
        sev = _severity(members)
        ids = [c.get("clip_id", "") for c in members]
        recurring = len(members) >= min_cluster_size
        reps = sorted(members, key=lambda c: -(c.get("telemetry", {}).get("ego_looming", 0.0)))[:5]
        cluster_id = f"M{cid:03d}"
        out.append({
            "cluster_id": cluster_id, "name": name, "description": desc, "size": len(members),
            "severity": round(sev, 3), "rank_score": round(len(members) * (0.5 + sev), 3),
            "iso_21448_area": "AREA_2_KNOWN_HAZARDOUS" if recurring else "AREA_3_UNKNOWN_HAZARDOUS",
            "member_clip_ids": ids, "representative_clip_ids": [c.get("clip_id", "") for c in reps],
            "attribution_hist": dict(Counter(c.get("attribution") for c in members)),
            "triggering_hist": dict(Counter(c.get("draft", {}).get("triggering_condition") for c in members)),
            "ncap_hist": dict(Counter(x for c in members for x in c.get("draft", {}).get("euro_ncap_2026_tags", []))),
            "weather_hist": dict(Counter(c.get("scene", {}).get("weather") for c in members)),
            "illumination_hist": dict(Counter(c.get("scene", {}).get("illumination") for c in members)),
            "noise": not recurring,
        })
        for i in ids:
            new_asg[i] = cluster_id
    return out, new_asg


def render_markdown(result: TaxonomyResult, title: str = "Fleet failure taxonomy") -> str:
    cl = result.clusters
    total = sum(c["size"] for c in cl)
    area2 = sum(c["size"] for c in cl if c["iso_21448_area"].startswith("AREA_2"))
    lines = [f"# {title}", "",
             f"{total} diagnosed events in {len(cl)} failure modes. "
             f"ISO 21448: {area2} events in recurring (Area 2, known) modes, "
             f"{total - area2} in novel (Area 3, unknown) modes.", "",
             "| Rank | Cluster | Size | Severity | ISO area | Dominant cause | Context |",
             "|---|---|---|---|---|---|---|"]
    for i, c in enumerate(cl, 1):
        attr = max(c["attribution_hist"], key=c["attribution_hist"].get) if c["attribution_hist"] else "-"
        ctx = max(c["illumination_hist"], key=c["illumination_hist"].get) if c["illumination_hist"] else "-"
        area = "2 known" if c["iso_21448_area"].startswith("AREA_2") else "3 novel"
        lines.append(f"| {i} | {c['name']} | {c['size']} | {c['severity']:.2f} | {area} | {attr} | {ctx} |")
    lines += ["", "## Clusters", ""]
    for i, c in enumerate(cl, 1):
        lines += [f"### {i}. {c['name']}  (`{c['cluster_id']}`)",
                  f"- **Size:** {c['size']}  **Severity:** {c['severity']:.2f}  "
                  f"**Rank score:** {c['rank_score']:.2f}  **ISO 21448:** {c['iso_21448_area']}",
                  f"- **Root causes:** {c['attribution_hist']}",
                  f"- **Triggers:** {c['triggering_hist']}",
                  f"- **Euro NCAP 2026 tags:** {c['ncap_hist'] or 'none'}",
                  f"- **Representative clips:** {', '.join(c['representative_clip_ids'])}",
                  f"- {c['description']}", ""]
    return "\n".join(lines)
