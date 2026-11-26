"""Deterministic counterfactual blame attribution.

The VLM never chooses the root-cause class. It is decided here from measurements, so the most
safety-relevant field of every card is reproducible and auditable:

  ALEATORY_OPTICAL          the scene itself hides the actor: high optical stress AND the
                            detector fails on the enhanced crop AND the VLM cannot see it (vis <= 1)
                            -> fix with ODD limits / sensor cleaning, not training data
  EPISTEMIC_PERCEPTION_GAP  the information is there but the model missed it: the detector finds
                            THE ACTOR (not just any object) on the enhanced crop, or the VLM sees it
                            clearly (vis >= 2), while the original track really failed (>= 3
                            dropouts, a >= 2-frame gap, < 80% presence, or >= 30% low confidence)
                            -> fix with targeted training data
  DYNAMIC_OCCLUSION         the actor emerged mid-frame next to an occluder and is clearly
                            visible after emerging (checked before the epistemic rule)
  AGENT_BEHAVIOR            perception was stable; the hazard is kinematic (ego-path looming,
                            driver reaction)
  INCONCLUSIVE              none of the above holds with enough evidence
"""

from __future__ import annotations

from ..schemas import Attribution, Detection, RoiAudit, Telemetry

VEHICLE_OR_VRU = {"pedestrian", "rider", "bicycle", "motorcycle", "car", "truck", "bus", "autorickshaw", "animal"}


def optical_stress(t: Telemetry) -> float:
    dark = max(0.0, (0.30 - t.luminance) / 0.20)
    glare = t.saturation_frac / 0.08
    blur = max(0.0, (0.45 - t.sharpness) / 0.30)
    return float(min(1.0, max(dark, glare, blur)))


def attribute(t: Telemetry, roi: RoiAudit | None, crop_dets: list[Detection], cfg: dict
              ) -> tuple[Attribution, list[str]]:
    d = cfg["diagnosis"]
    reasons: list[str] = []
    stress = optical_stress(t)
    vis = roi.visibility if roi else None
    crop_hit = any(x.cls in VEHICLE_OR_VRU and x.score >= d.get("crop_hit_score", 0.5) for x in crop_dets)
    # "Unstable" must mean real perception failure, not the normal confidence jitter every real
    # detector shows (first Nexar diagnosis run: the loose rule labelled 27/28 windows epistemic).
    unstable = (t.focus_presence < d.get("presence_unstable", 0.8)
                or t.focus_dropouts >= d.get("dropouts_unstable", 3)
                or t.focus_reacquired_gap >= d.get("gap_unstable", 2)
                or t.focus_low_conf_frac >= d.get("low_conf_frac_unstable", 0.3))
    ttc_ok = t.min_ttc_s is not None and t.min_ttc_s <= d.get("ttc_hazard_s", 2.0)
    # An imminent closing hazard: short time-to-collision, OR strong ego-path looming on its own.
    # Camera jerk was dropped from the trigger: on real Nexar clips it sits at 0.008-0.021 on
    # every window (event and non-event alike), so it cannot separate anything.
    collision_imminent = ttc_ok or t.ego_looming >= d.get("ego_looming_strong", 1.0)

    if stress >= d["optical_stress_high"] and not crop_hit and (vis is None or vis <= 1):
        reasons += [f"optical stress {stress:.2f} >= {d['optical_stress_high']}",
                    "detector found nothing on the enhanced crop", f"VLM crop visibility {vis}"]
        return Attribution.ALEATORY_OPTICAL, reasons
    # A short-TTC closing conflict on a stably-tracked actor is a behaviour/kinematic hazard, even
    # if the actor's track happened to start next to another box (common in traffic) -- so it is
    # checked BEFORE occlusion.
    if collision_imminent and not unstable:
        reasons.append(f"stable perception; imminent closing conflict (ego-path looming {t.ego_looming:.2f}/s, "
                       f"TTC {t.min_ttc_s:.1f}s)" if ttc_ok else
                       f"stable perception; strong ego-path looming {t.ego_looming:.2f}/s")
        return Attribution.AGENT_BEHAVIOR, reasons
    if t.focus_born_mid and t.focus_born_adjacent and (vis is None or vis >= 2):
        reasons.append("focus actor emerged mid-frame from behind another object, no imminent closing conflict")
        return Attribution.DYNAMIC_OCCLUSION, reasons
    if unstable and (crop_hit or (vis is not None and vis >= 2)):
        reasons.append(f"focus track unstable (presence {t.focus_presence:.2f}, dropouts {t.focus_dropouts}, "
                       f"max gap {t.focus_reacquired_gap}, low-confidence share {t.focus_low_conf_frac:.2f})")
        reasons.append("detector recovers it on the enhanced crop" if crop_hit else f"VLM sees it clearly (vis {vis})")
        return Attribution.EPISTEMIC_PERCEPTION_GAP, reasons
    reasons.append(f"no rule met (stress {stress:.2f}, unstable={unstable}, crop_hit={crop_hit}, vis={vis}, "
                   f"TTC={t.min_ttc_s if t.min_ttc_s is not None else 'n/a'}, looming={t.ego_looming:.2f})")
    return Attribution.INCONCLUSIVE, reasons
