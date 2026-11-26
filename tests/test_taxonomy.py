"""Stage 4 taxonomy: feature vector, clustering, naming, Area 2/3 by recurrence, report + CLI."""

import json

from sotif_miner.taxonomy.cluster import build_taxonomy, card_features, render_markdown


def _card(cid, attr="AGENT_BEHAVIOR", actor="car", weather="clear", illum="day", cond="none",
          ttc=None, looming=0.0, summary="an event"):
    return {
        "clip_id": cid, "attribution": attr,
        "scene": {"weather": weather, "illumination": illum, "vru_density": "low", "road_layout": "urban road"},
        "draft": {"primary_actor_class": actor, "triggering_condition": cond,
                  "functional_insufficiency": "none_perception_ok", "causal_layer": "Agent_Behavior",
                  "kinematic_hazard": "rapid_closing", "euro_ncap_2026_tags": [], "summary": summary},
        "telemetry": {"min_ttc_s": ttc, "ego_looming": looming, "luminance": 0.4 if illum != "night" else 0.1,
                      "saturation_frac": 0.01, "sharpness": 0.95, "active_tracks": 4},
    }


def _corpus():
    cards = []
    # mode A: night closing conflict with a car (recurring)
    for i in range(6):
        cards.append(_card(f"A{i}", "AGENT_BEHAVIOR", "car", "clear", "night", "low_light", ttc=1.2, looming=1.4,
                           summary="closing on a car at night"))
    # mode B: daytime occlusion emergence (recurring)
    for i in range(5):
        cards.append(_card(f"B{i}", "DYNAMIC_OCCLUSION", "car", "clear", "day", "occluding_object", ttc=5.0,
                           looming=0.3, summary="car emerges from behind an object"))
    # mode C: rainy perception gap with a pedestrian (recurring, higher severity via VRU)
    for i in range(4):
        cards.append(_card(f"C{i}", "EPISTEMIC_PERCEPTION_GAP", "pedestrian", "rain", "night", "rain", ttc=2.0,
                           looming=0.5, summary="pedestrian missed in rain"))
    # one-off novelty
    cards.append(_card("N0", "ALEATORY_OPTICAL", "animal", "fog", "dusk_dawn", "fog", ttc=1.0, looming=0.8,
                       summary="animal in fog, unseen"))
    return cards


def test_card_features_shape_and_content():
    v = card_features(_card("x"), None)
    assert v.ndim == 1 and v.shape[0] > 40
    v2 = card_features(_card("x"), None)
    assert (v == v2).all()  # deterministic


def test_clusters_separate_modes_and_mark_area():
    r = build_taxonomy(_corpus(), min_cluster_size=3)
    # the three recurring modes land in distinct clusters
    asg = r.assignments
    for grp in (["A0", "A1", "A2"], ["B0", "B1", "B2"], ["C0", "C1", "C2"]):
        assert len({asg[g] for g in grp}) == 1, grp
    assert asg["A0"] != asg["B0"] != asg["C0"] and asg["A0"] != asg["C0"]
    byid = {c["cluster_id"]: c for c in r.clusters}
    assert byid[asg["A0"]]["iso_21448_area"] == "AREA_2_KNOWN_HAZARDOUS"
    assert byid[asg["N0"]]["iso_21448_area"] == "AREA_3_UNKNOWN_HAZARDOUS"  # singleton -> novel


def test_severity_and_ranking():
    r = build_taxonomy(_corpus(), min_cluster_size=3)
    byid = {c["cluster_id"]: c for c in r.clusters}
    ped = byid[r.assignments["C0"]]  # VRU + rain -> higher severity than the daytime car occlusion
    occ = byid[r.assignments["B0"]]
    assert ped["severity"] > occ["severity"]
    scores = [c["rank_score"] for c in r.clusters]
    assert scores == sorted(scores, reverse=True)  # sorted by rank


def test_names_mention_context_and_actor():
    r = build_taxonomy(_corpus(), min_cluster_size=3)
    a = next(c for c in r.clusters if r.assignments["A0"] == c["cluster_id"])
    assert "night" in a["name"].lower() and "car" in a["name"].lower()
    c = next(cl for cl in r.clusters if r.assignments["C0"] == cl["cluster_id"])
    assert "pedestrian" in c["name"].lower()


def test_render_markdown():
    r = build_taxonomy(_corpus(), min_cluster_size=3)
    md = render_markdown(r)
    assert "# Fleet failure taxonomy" in md and "ISO 21448" in md and "| Rank |" in md


def test_cli_taxonomy(tmp_path):
    from typer.testing import CliRunner

    from sotif_miner.cli import app

    run = tmp_path / "run"
    for c in _corpus():
        d = run / "evidence" / c["clip_id"]
        d.mkdir(parents=True)
        (d / "card.json").write_text(json.dumps(c))
    # an inconclusive card that must be excluded by default
    d = run / "evidence" / "INC"
    d.mkdir(parents=True)
    (d / "card.json").write_text(json.dumps(_card("INC", "INCONCLUSIVE")))
    out = tmp_path / "tax"
    res = CliRunner().invoke(app, ["taxonomy", "--runs", str(run), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "clusters.md").exists() and (out / "clusters.json").exists()
    clusters = json.loads((out / "clusters.json").read_text())
    assert sum(c["size"] for c in clusters) == 16  # 16 corpus cards, INCONCLUSIVE excluded
    assert "INC" not in json.loads((out / "assignments.json").read_text())
