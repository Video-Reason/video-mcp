# Datasets (rinsed / processed)

This file is the running ledger of datasets that have been downloaded into `data/raw/` and processed into Video‑MCP outputs under `data/processed/`.

## How to update

- When you add support for a new dataset, add one entry below.
- When you re-run processing in a way that changes the output format or paths, update the entry.

## Ledger

| Dataset | Subset / task | Raw artifact(s) | Processed output | Status | Command(s) |
|---|---|---|---|---|---|
| CoreCognition | single-image MCQA VQA (753 samples) | `data/raw/corecognition/CoreCognition_20250622.zip` | `data/processed/corecognition_video_mcp/` | supported | `python -m video_mcp.dataset download --dataset corecognition`<br>`python -m video_mcp.dataset process --dataset corecognition` |

## Notes

- Some datasets require `HF_TOKEN` (gated access).
