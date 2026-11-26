import json

import cv2
import pytest

from sotif_miner.diagnosis.attribution import attribute
from sotif_miner.diagnosis.critic import review
from sotif_miner.schemas import (
    Attribution,
    CandidateWindow,
    CausalLayer,
    Detection,
    EvidenceBundle,
    FunctionalInsufficiency,
    KinematicHazard,
    RoiAudit,
    SceneAudit,
    Telemetry,
    TriggerCardDraft,
    TriggeringCondition,
    Weather,
)
from sotif_miner.synth.clips import crop_detector_fn, render, scenarios


def roi(vis, occ=False, cls="pedestrian"):
    return RoiAudit(object_class=cls, visibility=vis, occluder_present=occ, notes="")


PED = [Detection(box=(0, 0, 10, 10), score=0.9, cls="pedestrian")]


@pytest.mark.parametrize("tele,r,crop,expect", [
    (Telemetry(luminance=0.03, sharpness=0.9, conf_min=0.9, conf_max=0.9), roi(0), [], Attribution.ALEATORY_OPTICAL),
    (Telemetry(luminance=0.5, sharpness=0.9, conf_min=0.3, conf_max=0.8, focus_dropouts=4), roi(2), PED,
     Attribution.EPISTEMIC_PERCEPTION_GAP),
    (Telemetry(luminance=0.03, sharpness=0.9, conf_min=0.3, conf_max=0.8, focus_low_conf_frac=0.5), roi(1), PED,
     Attribution.EPISTEMIC_PERCEPTION_GAP),  # dark, but the detector recovers it on the enhanced crop
    # normal confidence jitter (one dip, one 1-frame dropout) is NOT a perception failure:
    (Telemetry(luminance=0.5, sharpness=0.9, conf_min=0.3, conf_max=0.9, focus_dropouts=1, focus_reacquired_gap=1,
               focus_low_conf_frac=0.1, ego_looming=1.5), roi(3), PED, Attribution.AGENT_BEHAVIOR),
    (Telemetry(luminance=0.5, sharpness=0.9, conf_min=0.85, conf_max=0.9, focus_born_mid=True,
               focus_born_adjacent=True), roi(3, occ=True), PED, Attribution.DYNAMIC_OCCLUSION),
    (Telemetry(luminance=0.5, sharpness=0.9, conf_min=0.85, conf_max=0.9, ego_looming=1.5), roi(3), PED,
     Attribution.AGENT_BEHAVIOR),
    (Telemetry(luminance=0.5, sharpness=0.9, conf_min=0.85, conf_max=0.9), roi(3), PED, Attribution.INCONCLUSIVE),
])
def test_attribution_table(cfg, tele, r, crop, expect):
    a, reasons = attribute(tele, r, crop, cfg)
    assert a == expect, reasons


def _draft(**kw):
    d = dict(summary="s", primary_actor_class="pedestrian", triggering_condition=TriggeringCondition.low_light,
             functional_insufficiency=FunctionalInsufficiency.missed_detection, causal_layer=CausalLayer.Perception,
             kinematic_hazard=KinematicHazard.crossing_vru, euro_ncap_2026_tags=["AEB_pedestrian"],
             claims_actor_clearly_visible=False, claims_occlusion=False, claimed_weather=Weather.clear,
             actionable_fix="x", cited_frame_ids=[10])
    d.update(kw)
    return TriggerCardDraft(**d)


def _bundle(tele=None):
    w = CandidateWindow(clip_id="c", source="s", start_frame=0, onset_frame=5, peak_frame=10, end_frame=40, fps=10,
                        peak_score=0.9, dominant_signal="continuity", signal_peaks={})
    return EvidenceBundle(clip_id="c", frame_paths=["a"], frame_ids=[10, 12], frame_times=[1.0, 1.2],
                          telemetry=tele or Telemetry(), window=w)


SCENE = SceneAudit(description="d", hazards_mentioned=[], weather=Weather.clear, illumination="night",
                   road_layout="r", vru_density="low")


def test_critic_accepts_consistent_card():
    rep = review(_draft(), Attribution.EPISTEMIC_PERCEPTION_GAP, _bundle(), SCENE, roi(2), {"pedestrian", "car"})
    assert rep.passed, rep.findings


@pytest.mark.parametrize("kw,rule", [
    ({"primary_actor_class": "truck"}, "actor_class"),
    ({"claims_actor_clearly_visible": True}, "visibility"),
    ({"claims_occlusion": True}, "occlusion"),
    ({"claimed_weather": Weather.rain}, "weather"),
    ({"causal_layer": CausalLayer.Environment}, "layer"),
    ({"cited_frame_ids": [99]}, "citations"),
    ({"euro_ncap_2026_tags": ["made_up"]}, "ncap_tags"),
])
def test_critic_rules(kw, rule):
    rep = review(_draft(**kw), Attribution.EPISTEMIC_PERCEPTION_GAP, _bundle(), SCENE, roi(1), {"pedestrian"})
    assert not rep.passed and rule in {f.rule for f in rep.findings}, rep.findings


def test_critic_text_vs_label():
    # AGENT_BEHAVIOR asserts stable perception; a summary claiming a missed actor is inconsistent
    d = _draft(summary="A turning vehicle is not detected by the perception system.",
               causal_layer=CausalLayer.Agent_Behavior, primary_actor_class="car")
    rep = review(d, Attribution.AGENT_BEHAVIOR, _bundle(), SCENE, roi(3, cls="car"), {"car"})
    assert not rep.passed and "text_vs_label" in {x.rule for x in rep.findings}


def test_critic_synonyms_and_vehicle_family():
    rep = review(_draft(primary_actor_class="person"), Attribution.EPISTEMIC_PERCEPTION_GAP, _bundle(), SCENE,
                 roi(2), {"pedestrian"})
    assert rep.passed
    rep = review(_draft(primary_actor_class="van", causal_layer=CausalLayer.Agent_Behavior),
                 Attribution.AGENT_BEHAVIOR, _bundle(), SCENE, roi(3, cls="car"), {"car"})
    assert rep.passed, rep.findings


# ---------------------------------------------------------------------------
# End to end: synthetic video -> `mine` -> `diagnose` (evidence pass + VLM pass) via the real CLI
# ---------------------------------------------------------------------------

EXPECT = {"tunnel_collapse": "ALEATORY_OPTICAL", "flicker_pedestrian": "EPISTEMIC_PERCEPTION_GAP",
          "occlusion_emergence": "DYNAMIC_OCCLUSION", "looming_vehicle": "AGENT_BEHAVIOR"}


def _scripted_vlm(name, bad_first_card=False):
    """A fake VLM that answers like a well-behaved model for this scenario (it cannot see pixels)."""
    from sotif_miner.vlm.scripted import ScriptedBackend

    vis = 0 if name == "tunnel_collapse" else 3
    weather = "clear"
    state = {"cards": 0}

    def respond(prompt, schema, n):
        title = schema.get("title")
        if title == "SceneAudit":
            return json.dumps({"description": "synthetic road", "hazards_mentioned": ["road user"], "weather": weather,
                               "illumination": "tunnel" if vis == 0 else "day", "road_layout": "straight road",
                               "vru_density": "low"})
        if title == "RoiAudit":
            return json.dumps({"object_class": "car" if name in ("tunnel_collapse", "looming_vehicle") else "pedestrian",
                               "visibility": vis, "occluder_present": name == "occlusion_emergence", "notes": "none"})
        state["cards"] += 1
        layer = {"tunnel_collapse": "Environment", "flicker_pedestrian": "Perception",
                 "occlusion_emergence": "Tracking", "looming_vehicle": "Agent_Behavior"}[name]
        labels = prompt.split("labelled ")[1].split(" in their")[0]
        ids = [int(x.strip().lstrip("F")) for x in labels.split(",")]  # cite by label number, as instructed
        actor = "car" if name in ("tunnel_collapse", "looming_vehicle") else "pedestrian"
        if bad_first_card and state["cards"] == 1:
            actor = "elephant"  # hallucination the critic must catch
        return json.dumps({"summary": f"{name} event", "primary_actor_class": actor,
                           "triggering_condition": "none", "functional_insufficiency": "none_perception_ok",
                           "causal_layer": layer, "kinematic_hazard": "none", "euro_ncap_2026_tags": [],
                           "claims_actor_clearly_visible": vis >= 2, "claims_occlusion": name == "occlusion_emergence",
                           "claimed_weather": weather, "actionable_fix": "fix", "cited_frame_ids": ids[:2]})
    return ScriptedBackend(respond)


def _write_mp4(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


@pytest.mark.parametrize("name", list(EXPECT))
def test_end_to_end_mine_and_diagnose(tmp_path, monkeypatch, name):
    from typer.testing import CliRunner

    import sotif_miner.cli as cli
    import sotif_miner.perception.detector as det_mod
    import sotif_miner.vlm.factory as fac

    sc = scenarios()[name]
    frames, dets = render(sc)
    vid = tmp_path / f"{name}.mp4"
    _write_mp4(vid, frames)
    monkeypatch.setattr(det_mod, "build_detector", lambda cfg: det_mod.ScriptedDetector(dets, crop_detector_fn(sc)))
    backend = _scripted_vlm(name, bad_first_card=(name == "flicker_pedestrian"))
    monkeypatch.setattr(fac, "build_backend", lambda cfg: backend)
    run = tmp_path / "run"
    r = CliRunner().invoke(cli.app, ["mine", "--input", str(vid), "--out", str(run), "--stride", "1"])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(cli.app, ["diagnose", "--run", str(run)])
    assert r.exit_code == 0, r.output
    cards = [json.loads(p.read_text()) for p in (run / "evidence").glob("*/card.json")]
    assert cards, r.output
    top = max(cards, key=lambda c: c["window"]["peak_score"])
    assert top["attribution"] == EXPECT[name], (top["attribution_reasons"], top["telemetry"])
    assert top["status"] == "accepted", top["critic"]
    if name == "flicker_pedestrian":
        assert top["repaired"] is True  # the hallucinated 'elephant' was caught and fixed
    ev = json.loads(next((run / "evidence").glob("*/evidence.json")).read_text())
    assert ev["bundle"]["rois"] and len(ev["unmarked"]) == len(ev["bundle"]["frame_paths"]) <= 8
    summary = json.loads((run / "diagnosis_summary.json").read_text())
    assert summary["vlm"]["json_validity"] == 1.0


def test_citation_labels_map_to_frame_ids(cfg):
    from sotif_miner.diagnosis.agent import Diagnoser

    b = _bundle()  # frame_ids [10, 12]
    d = Diagnoser._map_citations(_draft(cited_frame_ids=[2, 10]), b)
    assert d.cited_frame_ids == [12, 10]  # label F2 -> frame 12; an actual frame id stays
