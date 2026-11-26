"""Stage 3: the 4-step causal diagnosis state machine.

  1. Unprimed scene audit   - full keyframes only, no marks/telemetry (also measures unprompted recall)
  2. Counterfactual ROI     - detector on the enhanced crop (done in pass A, GPU shared sequentially)
                              + VLM visibility check on the crop -> deterministic attribution
  3. Trigger card           - VLM fills the constrained card; attribution is given, not chosen
  4. Critic                 - rules vs. evidence; one repair round, then `needs_review`
"""

from __future__ import annotations

import json

from ..config import catalog
from ..schemas import (
    Detection,
    EvidenceBundle,
    RoiAudit,
    SceneAudit,
    TriggerCard,
    TriggerCardDraft,
)
from ..vlm import prompts as P
from ..vlm.base import VLMBackend
from .attribution import attribute
from .critic import ALLOWED_LAYERS, review


def iso_area(draft: TriggerCardDraft) -> str:
    """Area 2 if the (condition, insufficiency) pair is a known trigger; otherwise Area 3.
    Phase 3's failure-memory clusters extend the known set."""
    for k in catalog().get("known_triggers", []):
        if (k["triggering_condition"] == draft.triggering_condition.value
                and k["functional_insufficiency"] == draft.functional_insufficiency.value):
            return "AREA_2_KNOWN_HAZARDOUS"
    return "AREA_3_UNKNOWN_HAZARDOUS"


class Diagnoser:
    def __init__(self, vlm: VLMBackend, cfg: dict):
        self.vlm, self.cfg = vlm, cfg

    def scene_audit(self, bundle: EvidenceBundle, unmarked_paths: list[str]) -> SceneAudit:
        times = ", ".join(f"t={t:.1f}s" for t in bundle.frame_times)
        return self.vlm.generate(unmarked_paths, P.SCENE_AUDIT.format(n=len(unmarked_paths), times=times),
                                 SceneAudit, P.SYSTEM)

    def roi_audit(self, bundle: EvidenceBundle) -> RoiAudit | None:
        if not bundle.rois:
            return None
        r = bundle.rois[0]
        hint = f" (the tracker labelled it '{r.cls_hint}')" if r.cls_hint else ""
        return self.vlm.generate([r.path], P.ROI_VERIFY.format(hint=hint), RoiAudit, P.SYSTEM)

    def _card_view(self, bundle: EvidenceBundle) -> tuple[list[int], list[str]]:
        """Up to `card_frames` frames nearest the peak (chronological) + up to `card_rois` crops.
        Fewer images = faster and less confusing for a small local VLM."""
        n = self.cfg["diagnosis"].get("card_frames", 4)
        peak = bundle.window.peak_frame
        order = sorted(range(len(bundle.frame_ids)), key=lambda i: abs(bundle.frame_ids[i] - peak))[:n]
        idx = sorted(order)
        rois = [r.path for r in bundle.rois[: self.cfg["diagnosis"].get("card_rois", 2)]]
        return idx, [bundle.frame_paths[i] for i in idx] + rois

    @staticmethod
    def _coerce(draft: TriggerCardDraft, attr) -> tuple[TriggerCardDraft, list[str]]:
        """The causal layer follows from the attribution; when only one layer is allowed, set it
        by rule instead of spending a repair call (it was the only first-draft error on v7)."""
        allowed = ALLOWED_LAYERS.get(attr)
        if allowed and len(allowed) == 1 and draft.causal_layer not in allowed:
            layer = next(iter(allowed))
            return draft.model_copy(update={"causal_layer": layer}), [f"causal_layer {draft.causal_layer.value} -> {layer.value}"]
        return draft, []

    @staticmethod
    def _map_citations(draft: TriggerCardDraft, bundle: EvidenceBundle) -> TriggerCardDraft:
        """Labels F1..Fn -> real frame ids. Values that already are frame ids are kept."""
        ids = []
        for c in draft.cited_frame_ids:
            if 1 <= c <= len(bundle.frame_ids) and c not in bundle.frame_ids:
                ids.append(bundle.frame_ids[c - 1])
            else:
                ids.append(c)
        return draft.model_copy(update={"cited_frame_ids": ids})

    def diagnose(self, bundle: EvidenceBundle, unmarked_paths: list[str], crop_dets: list[Detection],
                 seen_classes: set[str], source: str = "") -> TriggerCard:
        scene = self.scene_audit(bundle, unmarked_paths)
        roi = self.roi_audit(bundle)
        attr, reasons = attribute(bundle.telemetry, roi, crop_dets, self.cfg)
        ncap = catalog()["euro_ncap_2026_tags"]
        idx, images = self._card_view(bundle)
        prompt = P.CARD.format(
            n=len(idx), labels=", ".join(f"F{i + 1}" for i in idx), telemetry=bundle.telemetry.header(),
            scene=scene.model_dump_json(), roi=roi.model_dump_json() if roi else "no crop available",
            attribution=attr.value, reasons="; ".join(reasons), ncap=ncap,
            layers=sorted(x.value for x in ALLOWED_LAYERS.get(attr, set())) or "any")
        draft = self._map_citations(self.vlm.generate(images, prompt, TriggerCardDraft, P.SYSTEM), bundle)
        report = review(draft, attr, bundle, scene, roi, seen_classes)
        initial = list(report.findings)
        draft, coerced = self._coerce(draft, attr)
        if coerced:
            report = review(draft, attr, bundle, scene, roi, seen_classes)
        repaired = False
        for _ in range(self.cfg["diagnosis"]["max_repairs"]):
            if report.passed:
                break
            fix = P.REPAIR.format(findings="\n".join(f"- {x.message}" for x in report.findings),
                                  previous=json.dumps(draft.model_dump(mode="json")))
            draft = self._map_citations(self.vlm.generate(images, prompt + "\n\n" + fix, TriggerCardDraft, P.SYSTEM),
                                        bundle)
            draft, more = self._coerce(draft, attr)
            coerced += more
            report = review(draft, attr, bundle, scene, roi, seen_classes)
            repaired = True
        return TriggerCard(
            clip_id=bundle.clip_id, source=source, window=bundle.window, scene=scene, roi_audit=roi,
            crop_detector_hit=any(d.score >= self.cfg["diagnosis"].get("crop_hit_score", 0.5) for d in crop_dets),
            attribution=attr, attribution_reasons=reasons, draft=draft, critic=report, initial_findings=initial,
            coerced=coerced,
            repaired=repaired, status="accepted" if report.passed else "needs_review",
            iso_21448_area=iso_area(draft), telemetry=bundle.telemetry, backend=self.vlm.name)
