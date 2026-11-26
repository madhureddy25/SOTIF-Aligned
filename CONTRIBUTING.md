# Contributing

Thanks for your interest. This repository is a research/engineering build of a SOTIF-aligned dashcam
edge-case miner.

## Development setup
```bash
conda create -n sotif python=3.11 -y && conda activate sotif
pip install -e ".[dev]"      # CPU-only; add [detect] for RF-DETR, [vllm]/[ollama]/[claude] for VLMs
pytest -q                    # 73 CPU tests
ruff check src tests
```

## Ground rules
- **Tests pass and `ruff` is clean** before a PR. GPU/network tests are opt-in (`-m gpu`, `-m network`).
- **Honest evaluation only.** Report cross-validated or held-out numbers with confidence intervals;
  never report in-sample results as generalisation. Keep the limitations section truthful.
- **Licensing.** Keep the dependency path permissive (Apache-2.0 / MIT); no AGPL in the runtime path.
- **Deterministic attribution.** The root-cause label is decided by rules over measurements, not by the
  VLM. New rules need a unit test with a concrete telemetry case.
