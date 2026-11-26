"""Synthetic dashcam clips with scripted detections and known event times.

Each scenario renders a textured road scene (so camera-motion estimation has features)
with coloured boxes standing in for actors, plus a matching per-frame detection script
and a description of what the detector "sees" on an enhanced ROI crop. Used by the tests
and the CPU demo; no model weights or datasets needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..schemas import Detection

W, H, FPS, SECONDS = 640, 360, 10.0, 10.0
CLASS_COLOURS = {"car": (60, 60, 200), "truck": (40, 120, 40), "pedestrian": (30, 200, 230),
                 "motorcycle": (200, 80, 200)}


@dataclass
class Actor:
    cls: str
    box_fn: callable  # t -> (x1, y1, x2, y2) or None when not present
    score_fn: callable = lambda t: 0.9  # t -> detector score (None = missed)
    draw: bool = True


@dataclass
class Scenario:
    name: str
    actors: list[Actor]
    event: tuple[float, float] | None  # ground-truth (start_s, end_s); None for negatives
    expected_attribution: str | None = None
    crop_hit: bool = False  # does the detector find the actor on the enhanced crop?
    luminance_fn: callable = lambda t: 1.0
    pan_px_per_s: float = 0.0
    meta: dict = field(default_factory=dict)


def _background(t: float, pan: float, rng_seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    tex = rng.integers(60, 140, size=(H, W * 2), dtype=np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
    off = int(pan * t) % W
    g = tex[:, off:off + W]
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    img[: H // 3] = (img[: H // 3] * 0.5 + np.array([200, 160, 120]) * 0.5).astype(np.uint8)  # sky band
    cv2.line(img, (W // 2, H // 3), (W // 2, H), (220, 220, 220), 3)
    return img


def _lin(t, t0, t1, a, b):
    u = min(1.0, max(0.0, (t - t0) / max(t1 - t0, 1e-6)))
    return tuple(av + (bv - av) * u for av, bv in zip(a, b, strict=False))


def _crowd_box(i: int):
    """Box function for crowd member i: present 3-6 s, laid out on a grid with slight sway."""
    col, row = i % 11, i // 11

    def box(t: float):
        if not 3.0 <= t <= 6.0:
            return None
        x = 20 + col * 56 + 3 * np.sin(t + i)
        return (x, 210 + row * 50, x + 40, 255 + row * 50)

    return box


def scenarios() -> dict[str, Scenario]:
    car_lane = lambda t: (380 + 2 * t, 200, 460 + 2 * t, 260)  # noqa: E731
    return {
        "occlusion_emergence": Scenario(
            "occlusion_emergence",
            [Actor("truck", lambda t: (360, 150, 470, 290)),
             Actor("pedestrian", lambda t: None if t < 4.5 else _lin(t, 4.5, 6.5, (450, 200, 480, 280), (330, 210, 362, 300)),
                   score_fn=lambda t: 0.85)],
            event=(4.5, 6.5), expected_attribution="DYNAMIC_OCCLUSION", crop_hit=True),
        "flicker_pedestrian": Scenario(
            "flicker_pedestrian",
            [Actor("car", car_lane),
             Actor("pedestrian", lambda t: _lin(t, 0, 10, (150, 190, 175, 250), (210, 190, 237, 255)),
                   score_fn=lambda t: 0.8 if not (3.5 <= t <= 6.0) else (None if int(t * 10) % 3 == 0 else 0.3 + 0.25 * (int(t * 10) % 2)))],
            event=(3.5, 6.0), expected_attribution="EPISTEMIC_PERCEPTION_GAP", crop_hit=True),
        "tunnel_collapse": Scenario(
            "tunnel_collapse",
            [Actor("car", car_lane, score_fn=lambda t: None if 4.0 <= t <= 6.5 else 0.9)],
            event=(4.0, 6.5), expected_attribution="ALEATORY_OPTICAL", crop_hit=False,
            luminance_fn=lambda t: 0.08 if 4.0 <= t <= 6.5 else 1.0),
        "dense_crowd": Scenario(
            "dense_crowd",
            [Actor("motorcycle" if i % 2 else "pedestrian", _crowd_box(i), score_fn=lambda t: 0.75)
             for i in range(24)],
            event=(3.0, 6.0), expected_attribution="AGENT_BEHAVIOR", crop_hit=True),
        "looming_vehicle": Scenario(
            "looming_vehicle",
            [Actor("car", lambda t: (300, 190, 340, 220) if t < 4.0 else
                   _lin(t, 4.0, 5.5, (300, 190, 340, 220), (180, 120, 460, 330)))],
            event=(4.0, 5.5), expected_attribution="AGENT_BEHAVIOR", crop_hit=True),
        "steady_negative": Scenario(
            "steady_negative",
            [Actor("car", car_lane), Actor("truck", lambda t: (100 + t, 170, 200 + t, 250))],
            event=None),
    }


def render(sc: Scenario, fps: float = FPS, seconds: float = SECONDS
           ) -> tuple[list[np.ndarray], dict[int, list[Detection]]]:
    frames, dets = [], {}
    n = int(fps * seconds)
    for i in range(n):
        t = i / fps
        img = _background(t, sc.pan_px_per_s)
        frame_dets = []
        for a in sc.actors:
            box = a.box_fn(t)
            if box is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            if a.draw:
                cv2.rectangle(img, (x1, y1), (x2, y2), CLASS_COLOURS.get(a.cls, (128, 128, 128)), -1)
            s = a.score_fn(t)
            if s is not None:
                frame_dets.append(Detection(box=tuple(float(v) for v in box), score=float(s), cls=a.cls))
        img = np.clip(img.astype(np.float32) * sc.luminance_fn(t), 0, 255).astype(np.uint8)
        frames.append(img)
        dets[i] = frame_dets
    return frames, dets


def crop_detector_fn(sc: Scenario):
    """What the scripted detector returns on an enhanced ROI crop for this scenario."""
    main = next((a for a in sc.actors if a.cls in ("pedestrian", "motorcycle")), sc.actors[-1])

    def on_crop(img: np.ndarray) -> list[Detection]:
        if not sc.crop_hit:
            return []
        h, w = img.shape[:2]
        return [Detection(box=(w * 0.2, h * 0.2, w * 0.8, h * 0.8), score=0.88, cls=main.cls)]

    return on_crop
