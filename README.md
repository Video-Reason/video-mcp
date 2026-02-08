# video-mcp
Model Context Protocol (MCP)-style **Video-MCQA** dataset utilities.

This repo’s main output is a **Video-MCP dataset**: short clips where the **prompt UI is part of the video** and the **answer is expressed by highlighting A/B/C/D** in later frames.

For the authoritative spec, see `docs/VIDEO_MCP_DATA.md`.

## Quickstart

Create a venv, install pinned deps, and set your secrets:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` (local, gitignored) with at least:

```bash
HF_TOKEN=...
```

Download raw data and build processed Video-MCP outputs:

```bash
python -m video_mcp.dataset download --dataset corecognition
python -m video_mcp.dataset process  --dataset corecognition
```

Notes:
- Use `--limit N` to build only the first N samples (useful for quick testing).
- Use `--lit-style darken` (default) or `--lit-style red_border` to choose how the correct answer is highlighted.
- Requires `ffmpeg` on the system PATH (used to compile frames into MP4 video).

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

All outputs live under `data/` (gitignored):
- **Raw downloads**: `data/raw/`
- **Processed Video-MCP dataset**: `data/processed/`

CoreCognition processed output root:
- `data/processed/corecognition_video_mcp/`

Dataset-level config:
- `data/processed/corecognition_video_mcp/clip_config.json`

Each sample (ID format: `<datasetname>_<n>`):

```
data/processed/corecognition_video_mcp/
  clip_config.json
  corecognition_1/
    original/
      question.json              # question, choices, answer, source metadata
      <original_image_file>
    frames/
      frame_0000.png … frame_0047.png   # 48 rendered PNG frames
    video/
      clip.mp4                   # compiled MP4 video (H.264)
  corecognition_2/
    ...
```

### Frame layout

Each frame uses a **two-column panel** (image on left, question + choices on right)
with A/B/C/D answer boxes in the four corners of the frame.

- **Frame 0**: Question panel visible, no answer highlighted.
- **Frames 1–47**: Correct answer gradually highlights across the full clip duration.

### Highlight styles (`--lit-style`)

| Style | Effect |
|---|---|
| `darken` (default) | Correct corner box gradually darkens |
| `red_border` | Thick red outline gradually appears around the correct corner box |

## S3 upload

Processed datasets are synced to S3 for sharing:

```bash
aws s3 sync data/processed/ s3://video-mcp/data/processed/ --delete
```

Requires AWS credentials configured via `aws configure` or environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
