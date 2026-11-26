"""Schema guard + hallucination critic: checks the VLM's card against measured evidence."""

from __future__ import annotations

from ..config import catalog
from ..schemas import (
    Attribution,
    CausalLayer,
    CriticFinding,
    CriticReport,
    EvidenceBundle,
    RoiAudit,
    SceneAudit,
    TriggerCardDraft,
    Weather,
)

SYNONYMS = {
    "person": "pedestrian", "people": "pedestrian", "man": "pedestrian", "woman": "pedestrian", "child": "pedestrian",
    "cyclist": "bicycle", "bike": "bicycle", "biker": "motorcycle", "scooter": "motorcycle", "motorbike": "motorcycle",
    "two-wheeler": "motorcycle", "two wheeler": "motorcycle", "motorcyclist": "motorcycle", "rider": "motorcycle",
    "vehicle": "car", "suv": "car", "van": "car", "pickup": "truck", "lorry": "truck", "auto": "autorickshaw",
    "rickshaw": "autorickshaw", "tuk-tuk": "autorickshaw", "dog": "animal", "cow": "animal",
}
VEHICLES = {"car", "truck", "bus"}
TWO_WHEEL = {"bicycle", "motorcycle"}

ALLOWED_LAYERS = {
    Attribution.ALEATORY_OPTICAL: {CausalLayer.Environment},
    Attribution.EPISTEMIC_PERCEPTION_GAP: {CausalLayer.Perception, CausalLayer.Tracking},
    Attribution.DYNAMIC_OCCLUSION: {CausalLayer.Tracking, CausalLayer.Agent_Behavior, CausalLayer.Perception},
    Attribution.AGENT_BEHAVIOR: {CausalLayer.Agent_Behavior},
}


def norm_cls(c: str | None) -> str:
    c = (c or "").strip().lower()
    return SYNONYMS.get(c, c)


def _compatible(a: str, b: str) -> bool:
    return a == b or (a in VEHICLES and b in VEHICLES) or (a in TWO_WHEEL and b in TWO_WHEEL)


def review(draft: TriggerCardDraft, attribution: Attribution, bundle: EvidenceBundle, scene: SceneAudit,
           roi: RoiAudit | None, seen_classes: set[str]) -> CriticReport:
    f: list[CriticFinding] = []
    t = bundle.telemetry
    actor = norm_cls(draft.primary_actor_class)
    known = {norm_cls(c) for c in seen_classes} | ({norm_cls(roi.object_class)} if roi else set())
    known.discard("")
    known.discard("none")
    if known and not any(_compatible(actor, k) for k in known):
        f.append(CriticFinding(rule="actor_class", message=f"primary_actor_class '{draft.primary_actor_class}' was "
                               f"not detected or seen in the crop (seen: {sorted(known)})"))
    if draft.claims_actor_clearly_visible and roi is not None and roi.visibility <= 1:
        f.append(CriticFinding(rule="visibility", message=f"card claims the actor is clearly visible but the zoomed "
                               f"crop check rated visibility {roi.visibility}/3"))
    occl_evidence = t.focus_born_adjacent or t.focus_reacquired_gap > 0 or (roi is not None and roi.occluder_present)
    if draft.claims_occlusion and not occl_evidence and attribution != Attribution.DYNAMIC_OCCLUSION:
        f.append(CriticFinding(rule="occlusion", message="card claims occlusion but tracking shows no emergence next "
                               "to another object, no track gap, and the crop check found no occluder"))
    if (draft.claimed_weather not in (Weather.unknown,) and scene.weather != Weather.unknown
            and draft.claimed_weather != scene.weather):
        f.append(CriticFinding(rule="weather", message=f"claimed weather '{draft.claimed_weather.value}' contradicts "
                               f"the unprimed scene audit '{scene.weather.value}'"))
    allowed = ALLOWED_LAYERS.get(attribution)
    if allowed and draft.causal_layer not in allowed:
        f.append(CriticFinding(rule="layer", message=f"causal_layer '{draft.causal_layer.value}' is inconsistent "
                               f"with attribution {attribution.value} (allowed: {sorted(x.value for x in allowed)})"))
    # The free-text summary must not contradict the (fixed) attribution. AGENT_BEHAVIOR asserts
    # perception was stable, so a card claiming the actor was "not detected" is inconsistent.
    text = (draft.summary + " " + draft.actionable_fix).lower()
    MISS = ("not detected", "undetected", "fails to detect", "failed to detect", "missed detection",
            "misses the", "not perceived", "does not detect")
    if attribution == Attribution.AGENT_BEHAVIOR and any(w in text for w in MISS):
        f.append(CriticFinding(rule="text_vs_label", message="summary claims a missed/undetected actor, but the "
                               "attribution is AGENT_BEHAVIOR (stable perception); describe the kinematic conflict "
                               "instead, or the root cause is a perception gap"))
    if not draft.cited_frame_ids:
        f.append(CriticFinding(rule="citations", message="no frames cited"))
    bad = [i for i in draft.cited_frame_ids if i not in bundle.frame_ids]
    if bad:
        f.append(CriticFinding(rule="citations", message=f"cited frames {bad} do not exist; cite the label numbers "
                               f"1-{len(bundle.frame_ids)} (F1-F{len(bundle.frame_ids)})"))
    tags = set(catalog()["euro_ncap_2026_tags"])
    badt = [x for x in draft.euro_ncap_2026_tags if x not in tags]
    if badt:
        f.append(CriticFinding(rule="ncap_tags", message=f"unknown Euro NCAP tags {badt}"))
    return CriticReport(passed=not f, findings=f)
