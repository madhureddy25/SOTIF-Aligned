"""Prompt templates for the 4-step diagnosis agent. Kept short and literal: small local VLMs
follow concrete instructions better than long policy text."""

SYSTEM = ("You are an ADAS safety analyst reviewing dashcam footage. Report only what is visible. "
          "If something cannot be seen, say so. Answer with one JSON object matching the schema.")

SCENE_AUDIT = """These are {n} frames from one dashcam video, in time order ({times}). Each frame is labelled F1, F2, ... in its top-left corner.
Describe the driving scene neutrally, as you would for a log entry.
- description: 2-4 sentences on what happens across the frames.
- hazards_mentioned: list any road users or objects that could create a safety risk (empty list if none).
- weather: clear, rain, fog, snow or unknown.
- illumination: day, dusk_dawn, night, glare, tunnel or unknown.
- road_layout: short phrase (e.g. "urban two-lane road with parked cars").
- vru_density: how many pedestrians/cyclists/motorcyclists are present: none, low, medium or high."""

ROI_VERIFY = """This image is a zoomed, contrast-enhanced crop from a dashcam frame, centred on one road user{hint}.
- object_class: what is in the centre (pedestrian, cyclist, motorcycle, car, truck, bus, autorickshaw, animal, debris, other, or none).
- visibility: 0 = cannot be made out at all, 1 = barely (only a shape), 2 = recognisable, 3 = clearly visible.
- occluder_present: true if another object partly hides it.
- notes: one sentence on what limits visibility (darkness, glare, rain drops, blur, occlusion) or "none"."""

CARD = """You are writing a SOTIF (ISO 21448) trigger card for a dashcam event.

Frames: {n} frames in time order, labelled {labels} in their top-left corners. Numbered coloured boxes mark the
actors of interest (1 = primary). Zoomed crops of the marked actors follow the frames.

Telemetry from the perception stack:
{telemetry}

Unprimed scene description (from step 1): {scene}
Zoomed-crop check (from step 2): {roi}

Root-cause attribution, already decided from measurements (do NOT change it): {attribution}
Reasons: {reasons}

Fill the card:
- summary: 1-2 sentences: what happened and why it is safety-relevant.
- primary_actor_class: class of actor 1.
- causal_layer: must be one of {layers} (fixed by the attribution).
- triggering_condition, functional_insufficiency, kinematic_hazard: choose from the allowed values; they must be consistent with the attribution above.
- euro_ncap_2026_tags: choose 0-2 from {ncap}.
- claims_actor_clearly_visible / claims_occlusion / claimed_weather: what YOU see in the images.
- actionable_fix: one concrete engineering action (e.g. "add night-time cyclist samples to detector training", "extend ODD exclusion for heavy glare").
- cited_frame_ids: the NUMBERS of the frame labels that show the key moment, e.g. [3, 4] for F3 and F4."""

REPAIR = """Your previous trigger card contradicted the measured evidence:
{findings}

Previous card:
{previous}

Return a corrected card. Keep everything that was consistent; fix only the listed problems."""
