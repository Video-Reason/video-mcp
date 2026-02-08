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
python -m video_mcp.dataset process --dataset corecognition
```

Notes:
- `process` currently renders Video-MCP clips from the **complete** CoreCognition ZIP (`--config complete`).
- Use `--limit N` to build only the first N samples (useful for quick testing).

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
| `iter_mcqa_vqa(*, split)` | Yield `(McqaVqaSample, image_bytes)` pairs |

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

    def iter_mcqa_vqa(self, *, split: str) -> Iterator[tuple[McqaVqaSample, bytes]]:
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

Each sample:
- `data/processed/corecognition_video_mcp/<sample_id>/original/question.json`
- `data/processed/corecognition_video_mcp/<sample_id>/original/<original_image_file>`
- `data/processed/corecognition_video_mcp/<sample_id>/frames/frame_0000.png ... frame_0047.png`
