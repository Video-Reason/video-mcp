# Datasets (rinsed / processed)

This file is the running ledger of datasets that have been downloaded into `data/raw/` and processed into VBVR-compatible outputs under `questions/`.

## How to update

- When you add support for a new dataset, add one entry below.
- When you re-run processing in a way that changes the output format or paths, update the entry.

## Ledger

| Generator ID | Dataset | Subset / task | Raw artifact(s) | Processed output | Status | Command(s) |
|---|---|---|---|---|---|---|
| `M-1` | CoreCognition | single-image MCQA VQA (753 samples) | `data/raw/corecognition/CoreCognition_20250622.zip` | `questions/M-1_corecognition_data-generator/corecognition_task/` | supported | `python -m video_mcp.dataset download --dataset corecognition`<br>`python -m video_mcp.dataset process --dataset corecognition` |
| `M-2` | ScienceQA | image MCQA (HF `derek-thomas/ScienceQA`) | `data/raw/scienceqa/` (optional local snapshot) | `questions/M-2_scienceqa_data-generator/scienceqa_task/` | supported | `python -m video_mcp.dataset download --dataset scienceqa`<br>`python -m video_mcp.dataset process --dataset scienceqa` |
| `M-3` | MathVision | competition math MCQA (HF `MathLLMs/MathVision`, test split) | HF datasets cache | `questions/M-3_mathvision_data-generator/mathvision_task/` | supported | `python -m video_mcp.dataset download --dataset mathvision`<br>`python -m video_mcp.dataset process --dataset mathvision` |
| `M-4` | PhyX | physics reasoning MCQA (HF `Cloudriver/PhyX`, test_mini split) | HF datasets cache | `questions/M-4_phyx_data-generator/phyx_task/` | supported | `python -m video_mcp.dataset download --dataset phyx`<br>`python -m video_mcp.dataset process --dataset phyx` |

## Notes

- Some datasets require `HF_TOKEN` (gated access).
- Output follows the [VBVR DataFactory](https://github.com/video-reason/VBVR-DataFactory) directory convention.
- See [ADD_DATASET.md](./ADD_DATASET.md) for how to add a new dataset adapter.
