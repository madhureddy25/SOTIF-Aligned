"""Stage 5b: fine-tune RF-DETR on the co-training COCO dataset. Thin wrapper over rfdetr .train().

6 GB laptop GPUs: default to RF-DETR-nano with batch_size=2 and grad_accum, so the effective batch
stays reasonable while VRAM stays low. If even nano OOMs, that single step needs a bigger GPU; the
manifest and the closed-loop evaluation still run on the laptop.
"""

from __future__ import annotations

from pathlib import Path


def finetune(dataset_dir: str | Path, out_dir: str | Path, size: str = "nano", epochs: int = 20,
             batch_size: int = 2, grad_accum: int = 8, resolution: int | None = None,
             device: str = "cuda", lr: float = 1e-4) -> Path:
    import rfdetr  # noqa: PLC0415

    model_cls = getattr(rfdetr, "RFDETR" + {"nano": "Nano", "small": "Small", "medium": "Medium",
                                            "large": "Large"}[size])
    model = model_cls()
    kw = dict(dataset_dir=str(dataset_dir), output_dir=str(out_dir), epochs=epochs, batch_size=batch_size,
              grad_accum_steps=grad_accum, device=device, lr=lr)
    if resolution:
        kw["resolution"] = resolution
    model.train(**kw)
    out = Path(out_dir)
    ckpts = sorted(out.glob("**/*.pth"))
    best = next((c for c in ckpts if "best" in c.name.lower()), ckpts[-1] if ckpts else None)
    if best is None:
        raise RuntimeError(f"training produced no checkpoint under {out}")
    return best
