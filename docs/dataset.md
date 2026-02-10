# Datasets (rinsed / processed)

This file is the running ledger of datasets that have been downloaded into `data/raw/`
and processed into Video-MCP outputs (default root: `questions/`).

## How to update

- When you add support for a new dataset, add one entry below.
- When you re-run processing in a way that changes the output format or paths, update the entry.

## Ledger

| Dataset | Subset / task | Raw artifact(s) | Processed output | Status | Command(s) |
|---|---|---|---|---|---|
| CoreCognition | single-image MCQA VQA (753 samples) | `data/raw/corecognition/CoreCognition_20250622.zip` | `questions/M-1_corecognition_data-generator/corecognition_task/` | supported | `python -m video_mcp.dataset download --dataset corecognition`<br>`python -m video_mcp.dataset process --dataset corecognition [--lit-style darken\|red_border] [--limit N]` |
| ScienceQA | image MCQA (HF `derek-thomas/ScienceQA`) | `data/raw/scienceqa/` (optional local snapshot) | `questions/M-2_scienceqa_data-generator/scienceqa_task/` | supported | `python -m video_mcp.dataset download --dataset scienceqa`<br>`python -m video_mcp.dataset process --dataset scienceqa [--lit-style darken\|red_border] [--limit N]` |

## Notes

- Some datasets require `HF_TOKEN` (gated access).
- See the [README](../README.md#adding-a-new-dataset) for how to add a new dataset adapter.
