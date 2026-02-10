# video-mcp
Model Context Protocol (MCP)-style **Video-MCQA** dataset utilities.

This repo’s main output is a **Video-MCP dataset**: short clips where the **prompt UI is part of the video** and the **answer is expressed by highlighting A/B/C/D** in later frames.

For the authoritative spec, see `docs/VIDEO_MCP_DATA.md` (kept in sync with code).

## Quickstart

### 1. Setup

Create a venv and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` (local, gitignored) with at least:

```bash
HF_TOKEN=...
```

### 2. Download and process

**IMPORTANT:** Always activate the venv before running commands:

```bash
source venv/bin/activate
```

Download raw data (default: `data/raw/`) and build Video-MCP outputs (default: `questions/`):

```bash
python -m video_mcp.dataset download --dataset corecognition
python -m video_mcp.dataset process  --dataset corecognition
```

Quick test run (50 samples):

```bash
python -m video_mcp.dataset download --dataset scienceqa
python -m video_mcp.dataset process  --dataset scienceqa --limit 50
```

### Video specifications (Wan2.2-I2V-A14B)

Default output specs are aligned with **Wan2.2-I2V-A14B** fine-tuning requirements:

- **Resolution**: 832×480 (480p tier)
- **Frames**: 81 @ 16 FPS (~5.06 seconds)
- **Codec**: H.264, yuv420p, MP4 container

Override with CLI flags:

```bash
# 720p, 81 frames (higher quality, more VRAM)
python -m video_mcp.dataset process --dataset corecognition --width 1280 --height 720

# 480p, 49 frames (lighter runs)
python -m video_mcp.dataset process --dataset corecognition --num-frames 49
```

**Constraints** (enforced by Pydantic validators):
- Width and height must be divisible by **8** (VAE spatial compression)
- Frame count must satisfy **1 + 4k** where k ≥ 0 (VAE temporal compression): 1, 5, 9, 13, ..., 49, ..., 81

### Additional options

- `--limit N` — Build only the first N samples (useful for quick testing)
- `--lit-style darken` (default) or `--lit-style red_border` — How the correct answer is highlighted

### Requirements

- `ffmpeg` must be on the system PATH (used to compile frames into MP4 video)

## Adding a new dataset

Every dataset lives in its own file under `video_mcp/datasets/`. The generic
processing pipeline (`process/`), CLI, and all build scripts work with any
dataset automatically — you never need to touch them.

### 1. Create the adapter file

Create `video_mcp/datasets/<name>.py` (e.g. `video_mcp/datasets/scienceqa.py`).

Your file must define a class that inherits from `DatasetAdapter` and implements
three methods:

| Method | Purpose |
|---|---|
| `name` (property) | Short slug used in `--dataset` flags, e.g. `"scienceqa"` |
| `download(*, out_dir)` | Download the raw data and return the local path |
| `iter_mcqa_vqa()` | Yield `(McqaVqaSample, image_bytes)` pairs |

Minimal skeleton:

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from video_mcp.process.adapter import (
    DatasetAdapter,
    McqaVqaSample,
    register_adapter,
)


@register_adapter("scienceqa")
class ScienceQaAdapter(DatasetAdapter):

    @property
    def name(self) -> str:
        return "scienceqa"

    def download(self, *, out_dir: Path) -> Path:
        # Download or locate the raw data; return the artifact path.
        ...

    def iter_mcqa_vqa(self) -> Iterator[tuple[McqaVqaSample, bytes]]:
        # Yield (sample, image_bytes) for every MCQA-VQA example.
        ...
```

See `video_mcp/datasets/corecognition.py` for a complete working example.

### 2. Register it

Open `video_mcp/datasets/__init__.py` and add one import line:

```python
from video_mcp.datasets import scienceqa as _scienceqa  # noqa: F401
```

That's it. The new dataset is now available everywhere:

```bash
python -m video_mcp.dataset download  --dataset scienceqa
python -m video_mcp.dataset process   --dataset scienceqa
```

## Output format

This project writes two kinds of outputs:

- **Raw downloads** (gitignored): `data/raw/<dataset>/...`
- **Processed Video-MCP clips** (gitignored, default): `questions/`

### Naming rules (must match code output)

Processing uses the **VBVR DataFactory** layout and naming conventions:

- Generator directory: `{generator_id}_{dataset}_data-generator/`
  - Example: `M-2_scienceqa_data-generator/`
- Task directory: `{dataset}_task/`
  - Example: `scienceqa_task/`
- Sample directory: `{dataset}_{NNNN}/` where `NNNN` is a **zero-based**, 4-digit, zero-padded index
  - Example: `scienceqa_0000/`, `scienceqa_0049/`

### Folder layout (per sample)

```
questions/
  M-2_scienceqa_data-generator/
  clip_config.json
  run_manifest.json
  run_manifests/
    20260210T192842Z.json
  scienceqa_task/
    scienceqa_0000/
      first_frame.png            # required: input image for i2v generation
      prompt.txt                 # required: question + choices + answer text
      final_frame.png            # optional: expected final frame for evaluation
      ground_truth.mp4           # optional: reference video for evaluation
      original/
        question.json            # structured question/choices/answer + provenance
        <original_image_file>    # preserved original (PNG/JPG/JPEG)
```

Notes:

- `first_frame.png` / `final_frame.png` are always written as PNG.
- The original image is preserved with its original filename/extension (commonly `.png`, `.jpg`, `.jpeg`).
- Intermediate per-frame PNGs (`frame_0000.png`...) are rendered in a temp directory and are not kept.

### `clip_config.json`

Written once per generator directory:

- `fps`, `seconds`, `num_frames`, `width`, `height`

### Run manifest (`run_manifest.json`)

Every `process` run writes:

- `questions/<generator>/run_manifest.json` (latest)
- `questions/<generator>/run_manifests/<UTC_RUN_ID>.json` (history)

It records:

- dataset identity (`hf_repo_id`, `hf_config`, `hf_split`, `hf_revision` when provided by the adapter)
- processing parameters (fps/seconds/num_frames/width/height, lit style, limit)
- code version (git commit/branch when available)
- runtime (`python_version`, `ffmpeg_version`)

Each frame uses a **two-column panel** (image on left, question + choices on right)
with A/B/C/D answer boxes in the four corners of the frame.

- **Frame 0**: Question panel visible, no answer highlighted.
- **Frames 1–80** (default): Correct answer gradually highlights across the full clip duration.

### Highlight styles (`--lit-style`)

| Style | Effect |
|---|---|
| `darken` (default) | Correct corner box gradually darkens |
| `red_border` | Thick red outline gradually appears around the correct corner box |

## S3 upload

Processed outputs can be synced to S3 for sharing. Prefer attaching an **IAM role**
to your EC2 instance instead of copying access keys.

If you do use local credentials, ensure your `.env` contains:

```bash
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
AWS_DEFAULT_REGION="us-east-2"
```

Then load the credentials and sync:

```bash
# Load AWS credentials from .env
source scripts/load_env.sh

# Sync to S3
aws s3 sync questions/ s3://video-mcp/questions/ --delete
```

**Alternative:** Set credentials permanently via `aws configure`, which creates `~/.aws/credentials`.
