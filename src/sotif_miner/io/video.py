"""Frame sources: MP4 files, image directories, single images, optional MCAP."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Frame:
    index: int
    t: float
    image: np.ndarray  # BGR uint8


class FrameSource:
    fps: float
    source_id: str

    def __iter__(self) -> Iterator[Frame]:  # pragma: no cover - interface
        raise NotImplementedError


class VideoFile(FrameSource):
    def __init__(self, path: str | Path, stride: int = 1):
        self.path = Path(path)
        self.stride = max(1, stride)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self.fps = native / self.stride
        self.source_id = self.path.stem

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        i = out = 0
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if i % self.stride == 0:
                    yield Frame(out, out / self.fps, img)
                    out += 1
                i += 1
        finally:
            cap.release()


class ImageSequence(FrameSource):
    """A directory of ordered frames (e.g. BDD100K MOT) or an explicit file list."""

    def __init__(self, paths: list[Path] | str | Path, fps: float = 5.0, source_id: str | None = None):
        if isinstance(paths, (str, Path)):
            d = Path(paths)
            paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
            source_id = source_id or d.name
        self.paths = list(paths)
        self.fps = fps
        self.source_id = source_id or (self.paths[0].stem if self.paths else "empty")

    def __iter__(self) -> Iterator[Frame]:
        for i, p in enumerate(self.paths):
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(p)
            yield Frame(i, i / self.fps, img)


class ArraySource(FrameSource):
    """In-memory frames (synthetic clips, tests)."""

    def __init__(self, frames: list[np.ndarray], fps: float, source_id: str):
        self.frames, self.fps, self.source_id = frames, fps, source_id

    def __iter__(self) -> Iterator[Frame]:
        for i, img in enumerate(self.frames):
            yield Frame(i, i / self.fps, img)


class McapSource(FrameSource):  # pragma: no cover - needs optional dep + real logs
    """ROS2 MCAP log with sensor_msgs/CompressedImage or Image on `topic`."""

    def __init__(self, path: str | Path, topic: str, fps: float = 30.0):
        self.path, self.topic, self.fps = Path(path), topic, fps
        self.source_id = self.path.stem

    def __iter__(self) -> Iterator[Frame]:
        try:
            from mcap_ros2.reader import read_ros2_messages
        except ImportError as e:
            raise ImportError("install sotif-miner[mcap] for MCAP support") from e
        t0 = None
        for i, msg in enumerate(read_ros2_messages(str(self.path), topics=[self.topic])):
            m = msg.ros_msg
            t = msg.log_time_ns / 1e9
            t0 = t if t0 is None else t0
            if hasattr(m, "format"):  # CompressedImage
                img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
            else:
                arr = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else arr
            yield Frame(i, t - t0, img)


def open_source(path: str | Path, stride: int = 1, fps: float = 5.0) -> FrameSource:
    p = Path(path)
    if p.is_dir():
        return ImageSequence(p, fps=fps)
    if p.suffix.lower() in IMG_EXT:
        return ImageSequence([p], fps=fps, source_id=p.stem)
    return VideoFile(p, stride=stride)


def frame_stats(img: np.ndarray) -> tuple[float, float, float]:
    """(mean luminance [0,1], near-white saturation fraction, normalized sharpness [0,1])."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (320, int(320 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    lum = float(small.mean() / 255.0)
    sat = float((small >= 250).mean())
    lap = cv2.Laplacian(small, cv2.CV_64F).var()
    sharp = float(1.0 - np.exp(-lap / 300.0))
    return lum, sat, sharp


def read_frames(path: str | Path, indices: list[int], stride: int = 1) -> dict[int, np.ndarray]:
    """Read specific frames, addressed in the *strided* index space used during mining.

    Decodes sequentially (robust across codecs; seeking in H.264 is not frame-accurate everywhere)
    and stops after the last wanted frame."""
    want = {i * stride: i for i in indices}
    if not want:
        return {}
    last = max(want)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    try:
        n = 0
        while n <= last:
            ok, img = cap.read()
            if not ok:
                break
            if n in want:
                out[want[n]] = img
            n += 1
    finally:
        cap.release()
    return out


def native_fps(path: str | Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        cap.release()
