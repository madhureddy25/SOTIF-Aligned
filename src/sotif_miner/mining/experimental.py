"""THE file the autoresearch loop edits (autoresearch/program.md). Nothing else is changed by the loop.

Add Stage-1 cues here; the fixed harness (`sotif-miner autoresearch-eval`) computes them from the cached perception
logs, trains the learned Stage 1 with them (5-fold CV over the 1,300 training clips, several seeds), and decides keep
or discard. The 200 held-out clips are never touched by the loop.

Contract:
* EXP_FEATURES: name -> (signal, direction, MAD floor), exactly like `scorer.FEATURES`. Names must start with "z.".
  The learned model reads them; the hand-weighted rule scorer ignores them.
* extra_features(frames, prev_frames, fps, w, h) -> {name: float} for one 0.5 s step. `frames` are the step's
  FrameRecords, `prev_frames` the preceding ~2 s; boxes are (x1, y1, x2, y2) pixels in a w x h image. Return every
  name in EXP_FEATURES (use 0.0 when undefined). Must be deterministic and fast (pure Python/numpy, no I/O).
Helpers worth reusing: `signals.in_ego_corridor`, `signals._lead_series`, `signals._approach_rates`, `signals.ROAD_USERS`.
"""

from __future__ import annotations

from ..schemas import FrameRecord

EXP_FEATURES: dict[str, tuple[str, int, float]] = {}


def extra_features(frames: list[FrameRecord], prev_frames: list[FrameRecord], fps: float, w: int, h: int
                   ) -> dict[str, float]:
    return {}
