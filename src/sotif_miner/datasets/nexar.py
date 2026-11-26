"""Nexar Dashcam Collision Prediction dataset loader.

Layout (Hugging Face `empiricearth/nexar_collision_prediction`):

    <root>/{train,test-public,test-private}/{positive,negative}/metadata.csv + NNNNN.mp4

metadata.csv columns:
    file_name, time_of_event, time_of_alert, light_conditions, weather, scene, time_to_accident

* train: ~40 s clips; positives carry time_of_event / time_of_alert in clip seconds.
* test-*: ~10 s clips that END `time_to_accident` seconds (0.5/1.0/1.5) BEFORE the event,
  so the event itself is never visible. Their time_of_event / time_of_alert refer to the
  original long clip, not to the truncated file; use `event_in_clip_s()` instead.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2

SPLITS = ("train", "test-public", "test-private")


def _f(x: str | None) -> float | None:
    x = (x or "").strip()
    return float(x) if x else None


@dataclass
class NexarClip:
    path: Path
    split: str
    positive: bool
    time_of_event: float | None
    time_of_alert: float | None
    light: str
    weather: str
    scene: str
    time_to_accident: float | None

    @property
    def clip_id(self) -> str:
        return f"nexar_{self.split}_{self.path.stem}"

    @property
    def truncated(self) -> bool:
        """Test clips stop before the event."""
        return self.split != "train"

    @cached_property
    def duration_s(self) -> float:
        cap = cv2.VideoCapture(str(self.path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            return float(n / fps)
        finally:
            cap.release()

    def event_in_clip_s(self) -> float | None:
        """Event time on this file's own timeline (may lie after the clip end for test clips)."""
        if not self.positive:
            return None
        if self.truncated:
            return self.duration_s + (self.time_to_accident or 0.0)
        return self.time_of_event

    def alert_in_clip_s(self) -> float | None:
        if not self.positive or self.time_of_alert is None or self.time_of_event is None:
            return None
        if self.truncated:
            ev = self.event_in_clip_s()
            return ev - (self.time_of_event - self.time_of_alert) if ev is not None else None
        return self.time_of_alert


def load_nexar(root: str | Path, split: str = "train", labels: tuple[str, ...] = ("positive", "negative"),
               limit: int | None = None, require_video: bool = True, skip: int = 0) -> list[NexarClip]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    root = Path(root)
    clips: list[NexarClip] = []
    for lab in labels:
        d = root / split / lab
        meta = d / "metadata.csv"
        if not meta.exists():
            raise FileNotFoundError(f"missing {meta}")
        with meta.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                p = d / row["file_name"]
                if require_video and not p.exists():
                    continue
                clips.append(NexarClip(
                    path=p, split=split, positive=lab == "positive",
                    time_of_event=_f(row.get("time_of_event")), time_of_alert=_f(row.get("time_of_alert")),
                    light=(row.get("light_conditions") or "").strip(), weather=(row.get("weather") or "").strip(),
                    scene=(row.get("scene") or "").strip(), time_to_accident=_f(row.get("time_to_accident")),
                ))
    clips.sort(key=lambda c: c.path.name)
    clips = clips[skip:]  # skip the first N per label (to mine a slice disjoint from an earlier run)
    return clips[:limit] if limit else clips


def iter_nexar(root: str | Path, splits: tuple[str, ...] = SPLITS, **kw) -> Iterator[NexarClip]:
    for s in splits:
        yield from load_nexar(root, s, **kw)
