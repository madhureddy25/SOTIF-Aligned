"""Typed records passed between pipeline stages.

Everything that crosses a stage boundary is a pydantic model so it can be
serialized to JSONL, validated, and handed to a VLM as a JSON schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class Detection(BaseModel):
    box: Box
    score: float
    cls: str  # unified-ontology class name


class TrackState(BaseModel):
    """One track observation at one frame."""

    track_id: int
    frame: int
    box: Box
    score: float
    cls: str
    is_new: bool = False  # first frame of a (re)born track
    reacquired_gap: int = 0  # >0 when a lost track is re-matched: number of frames it was missing


class FrameRecord(BaseModel):
    frame: int
    t: float  # seconds
    detections: list[Detection]
    tracks: list[TrackState] = Field(default_factory=list)
    luminance: float = 0.0  # mean luma [0,1]
    saturation_frac: float = 0.0  # fraction of near-white pixels
    sharpness: float = 0.0  # variance of Laplacian (normalized)
    terminated_ids: list[int] = Field(default_factory=list)  # tracks removed this frame
    ego_dx: float = 0.0  # camera motion vs. previous frame (pixels), from motion compensation
    ego_dy: float = 0.0
    ego_scale: float = 1.0


# ---------------------------------------------------------------------------
# Stage 1: mining
# ---------------------------------------------------------------------------

SignalName = Literal["uncertainty", "continuity", "optical_density", "rarity"]


class SignalSample(BaseModel):
    t: float
    frame: int
    uncertainty: float = 0.0
    continuity: float = 0.0
    optical_density: float = 0.0
    rarity: float = 0.0
    combined: float = 0.0
    detail: dict[str, float] = Field(default_factory=dict)


class CandidateWindow(BaseModel):
    clip_id: str
    source: str
    start_frame: int
    onset_frame: int  # first step where the score crossed t_on (used for lead-time metrics)
    peak_frame: int
    end_frame: int
    fps: float
    peak_score: float
    dominant_signal: SignalName
    signal_peaks: dict[str, float]
    focus_track_ids: list[int] = Field(default_factory=list)
    single_frame: bool = False


# ---------------------------------------------------------------------------
# Stage 2: evidence
# ---------------------------------------------------------------------------


class RoiPatch(BaseModel):
    track_id: int | None
    frame: int
    crop_box: Box  # region in the original frame
    path: str  # enhanced crop on disk
    cls_hint: str | None = None
    actor_box: Box | None = None  # the focus actor's box (original frame coordinates)
    scale: float = 1.0  # upscale factor applied when the crop was enhanced


class Telemetry(BaseModel):
    track_gap_frames: int = 0
    conf_min: float = 0.0
    conf_max: float = 0.0
    looming_rate: float = 0.0  # relative box-area growth per second
    min_ttc_s: float | None = None
    luminance: float = 0.0
    saturation_frac: float = 0.0
    sharpness: float = 0.0
    active_tracks: int = 0
    ego_looming: float = 0.0  # ego-path relative area growth per second
    ego_jerk: float = 0.0  # max camera pitch/yaw jerk in the window (driver reaction proxy)
    # Focus actor evidence (the 1-3 conflicting actors)
    focus_cls: str | None = None
    focus_presence: float = 1.0  # fraction of window frames where the focus track was matched
    focus_born_mid: bool = False  # track appeared mid-frame inside the window
    focus_born_adjacent: bool = False  # ... next to / overlapping another object (occluder)
    focus_reacquired_gap: int = 0  # longest gap (frames) before re-matching
    focus_dropouts: int = 0  # number of times the focus track was lost and re-matched in the window
    focus_low_conf_frac: float = 0.0  # fraction of its observations with score < 0.4

    def header(self) -> str:
        ttc = f"{self.min_ttc_s:.1f}s" if self.min_ttc_s is not None else "n/a"
        return "\n".join(
            [
                f"TRACK: max continuity gap {self.track_gap_frames} frames; active tracks {self.active_tracks}",
                f"CONF: focus-actor confidence range {self.conf_min:.2f}-{self.conf_max:.2f}",
                f"KINEMATICS: ego-path looming {self.ego_looming:+.2f}/s; min TTC proxy {ttc}; camera jerk {self.ego_jerk:.3f}",
                f"OPTICS: luminance {self.luminance:.2f}; specular saturation {self.saturation_frac:.3f}",
                f"CLARITY: sharpness {self.sharpness:.2f} (0=blurred, 1=crisp)",
            ]
        )


class EvidenceBundle(BaseModel):
    clip_id: str
    arm: Literal["evidence_gated", "video_only", "raw_cv_dump"] = "evidence_gated"
    frame_paths: list[str]  # timestamp-labelled keyframes (marked for evidence_gated)
    frame_ids: list[int]
    frame_times: list[float]
    rois: list[RoiPatch] = Field(default_factory=list)
    telemetry: Telemetry = Field(default_factory=Telemetry)
    text_dump: str | None = None  # raw_cv_dump arm only
    window: CandidateWindow


# ---------------------------------------------------------------------------
# Stage 3: diagnosis
# ---------------------------------------------------------------------------


class Weather(str, Enum):
    clear = "clear"
    rain = "rain"
    fog = "fog"
    snow = "snow"
    unknown = "unknown"


class Illumination(str, Enum):
    day = "day"
    dusk_dawn = "dusk_dawn"
    night = "night"
    glare = "glare"
    tunnel = "tunnel"
    unknown = "unknown"


class SceneAudit(BaseModel):
    """Q3-style unprimed description. The VLM sees only full frames."""

    model_config = ConfigDict(extra="forbid")
    description: str
    hazards_mentioned: list[str]
    weather: Weather
    illumination: Illumination
    road_layout: str
    vru_density: Literal["none", "low", "medium", "high"]


class RoiAudit(BaseModel):
    """VLM verification of a zoomed, contrast-normalized crop."""

    model_config = ConfigDict(extra="forbid")
    object_class: str
    visibility: int = Field(ge=0, le=3, description="0=not discernible .. 3=clearly visible")
    occluder_present: bool
    notes: str


class Attribution(str, Enum):
    EPISTEMIC_PERCEPTION_GAP = "EPISTEMIC_PERCEPTION_GAP"
    ALEATORY_OPTICAL = "ALEATORY_OPTICAL"
    DYNAMIC_OCCLUSION = "DYNAMIC_OCCLUSION"
    AGENT_BEHAVIOR = "AGENT_BEHAVIOR"
    INCONCLUSIVE = "INCONCLUSIVE"


class TriggeringCondition(str, Enum):
    none = "none"
    low_light = "low_light"
    glare = "glare"
    rain = "rain"
    fog = "fog"
    snow = "snow"
    motion_blur = "motion_blur"
    dense_traffic = "dense_traffic"
    occluding_object = "occluding_object"
    unusual_road_geometry = "unusual_road_geometry"
    out_of_ontology_object = "out_of_ontology_object"
    unusual_actor_behavior = "unusual_actor_behavior"


class FunctionalInsufficiency(str, Enum):
    missed_detection = "missed_detection"
    confidence_flicker = "confidence_flicker"
    misclassification = "misclassification"
    id_switch = "id_switch"
    track_loss_under_occlusion = "track_loss_under_occlusion"
    out_of_ontology = "out_of_ontology"
    late_detection = "late_detection"
    none_perception_ok = "none_perception_ok"


class CausalLayer(str, Enum):
    Environment = "Environment"
    Perception = "Perception"
    Tracking = "Tracking"
    Agent_Behavior = "Agent_Behavior"


class KinematicHazard(str, Enum):
    none = "none"
    cut_in = "cut_in"
    sudden_emergence = "sudden_emergence"
    rapid_closing = "rapid_closing"
    crossing_vru = "crossing_vru"
    dooring = "dooring"
    wrong_way = "wrong_way"
    dense_weaving = "dense_weaving"


class TriggerCardDraft(BaseModel):
    """What the VLM is asked to produce (constrained decoding target)."""

    model_config = ConfigDict(extra="forbid")
    summary: str
    primary_actor_class: str
    triggering_condition: TriggeringCondition
    functional_insufficiency: FunctionalInsufficiency
    causal_layer: CausalLayer
    kinematic_hazard: KinematicHazard
    euro_ncap_2026_tags: list[str]
    claims_actor_clearly_visible: bool
    claims_occlusion: bool
    claimed_weather: Weather
    actionable_fix: str
    cited_frame_ids: list[int]  # frame labels as stamped on the images (F3 -> 3); mapped to frame ids


class CriticFinding(BaseModel):
    rule: str
    message: str


class CriticReport(BaseModel):
    passed: bool
    findings: list[CriticFinding] = Field(default_factory=list)


IsoArea = Literal["AREA_2_KNOWN_HAZARDOUS", "AREA_3_UNKNOWN_HAZARDOUS"]


class TriggerCard(BaseModel):
    """Final, validated record for one event."""

    clip_id: str
    source: str
    window: CandidateWindow
    scene: SceneAudit
    roi_audit: RoiAudit | None
    crop_detector_hit: bool
    attribution: Attribution
    attribution_reasons: list[str]
    draft: TriggerCardDraft
    critic: CriticReport
    initial_findings: list[CriticFinding] = Field(default_factory=list)  # critic findings on the first draft
    coerced: list[str] = Field(default_factory=list)  # fields set by rule instead of a repair call
    repaired: bool = False
    status: Literal["accepted", "needs_review"] = "accepted"
    iso_21448_area: IsoArea | None = None
    matched_cluster: str | None = None
    telemetry: Telemetry
    backend: str = ""


# ---------------------------------------------------------------------------
# Stage 4/5
# ---------------------------------------------------------------------------


class ClusterName(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str


class Cluster(BaseModel):
    cluster_id: str
    name: str
    description: str
    size: int
    severity: float
    rank_score: float
    member_clip_ids: list[str]
    representative_clip_ids: list[str]
    attribution_hist: dict[str, int]
    triggering_hist: dict[str, int]
    ncap_hist: dict[str, int]
    iso_area_hist: dict[str, int]
    scene_key: str  # dominant static scene cue
    actor_key: str  # dominant dynamic actor cue


class ManifestEntry(BaseModel):
    image_path: str
    source: str
    role: Literal["corner", "core", "pseudo"]
    cluster_id: str | None = None
    review_required: bool = False
    boxes: list[Detection] = Field(default_factory=list)


class Manifest(BaseModel):
    ratio_corner_to_core: float
    entries: list[ManifestEntry]
    clusters: list[str]
