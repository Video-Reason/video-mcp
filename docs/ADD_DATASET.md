# Adding a New Dataset

This guide walks you through adding a new dataset adapter to video-mcp. The system is designed so that **all you need is one Python file** — the CLI, processing pipeline, rendering engine, and VBVR-compatible output all work automatically.

## Before you start

1. Your dataset must have **single-image MCQA (multiple-choice question-answer)** samples.
2. Each sample needs: a **question**, **2-4 answer choices** (A/B/C/D), the **correct answer**, and an **image**.
3. Pick a short slug name for your dataset (lowercase, no spaces). Examples: `scienceqa`, `mmstar`, `corecognition`.
4. Pick the next available generator ID. Current assignments:

| Generator ID | Name | Dataset |
|---|---|---|
| `M-1` | `corecognition` | CoreCognition |
| `M-2` | `scienceqa` | ScienceQA |
| `M-3` | `mathvision` | MathVision |
| `M-4` | `phyx` | PhyX |
| `M-5` | *(next available)* | |

## Step 1: Create the adapter file

Create a new file at `video_mcp/datasets/<name>.py`.

For example, to add a new dataset: `video_mcp/datasets/mydataset.py`.

### Full template

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from video_mcp.process.adapter import (
    DatasetAdapter,
    McqaVqaSample,
    register_adapter,
)


@register_adapter("mydataset")
class MyDatasetAdapter(DatasetAdapter):
    """Adapter for the MyDataset dataset."""

    @property
    def name(self) -> str:
        return "mydataset"

    @property
    def generator_id(self) -> str:
        return "M-5"

    def download(self, *, out_dir: Path) -> Path:
        """Download raw data and return the artifact path."""
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Your download logic here ---
        # Example using Hugging Face Hub:
        #
        # import os
        # from huggingface_hub import hf_hub_download
        #
        # token = os.environ.get("HF_TOKEN")
        # path = hf_hub_download(
        #     repo_id="org/dataset-name",
        #     repo_type="dataset",
        #     filename="data.zip",
        #     token=token,
        #     local_dir=str(out_dir),
        # )
        # return Path(path)

        raise NotImplementedError("TODO: implement download")

    def iter_mcqa_vqa(self) -> Iterator[tuple[McqaVqaSample, bytes]]:
        """Yield (sample, image_bytes) for every MCQA-VQA example."""

        # --- Your iteration logic here ---
        # For each valid sample in your dataset, yield a tuple of:
        #   1. McqaVqaSample (Pydantic model with question metadata)
        #   2. Raw image bytes (PNG or JPEG)
        #
        # Example:
        #
        # for row in load_my_dataset():
        #     image_bytes = read_image(row["image_path"])
        #     sample = McqaVqaSample(
        #         dataset="MyDataset",
        #         source_id=str(row["id"]),
        #         question=row["question"],
        #         choices={"A": row["choice_a"], "B": row["choice_b"], ...},
        #         answer="A",  # must be one of "A", "B", "C", "D"
        #         image_filename="image.png",
        #     )
        #     yield sample, image_bytes

        raise NotImplementedError("TODO: implement iteration")
```

## Step 2: Register the adapter

Open `video_mcp/datasets/__init__.py` and add one import line:

```python
from video_mcp.datasets import mydataset as _mydataset  # noqa: F401
```

The file should look like:

```python
# Import adapter submodules so their @register_adapter decorators execute.
from video_mcp.datasets import corecognition as _corecognition  # noqa: F401
from video_mcp.datasets import mathvision as _mathvision  # noqa: F401
from video_mcp.datasets import mydataset as _mydataset  # noqa: F401
from video_mcp.datasets import phyx as _phyx  # noqa: F401
from video_mcp.datasets import scienceqa as _scienceqa  # noqa: F401

__all__ = ["corecognition", "mathvision", "mydataset", "phyx", "scienceqa"]
```

That's it. The `@register_adapter` decorator runs at import time and makes your dataset available to the CLI and processing pipeline.

## Step 3: Test it

```bash
source venv/bin/activate

# Download the raw data
python -m video_mcp.dataset download --dataset mydataset

# Process a small batch first
python -m video_mcp.dataset process --dataset mydataset --limit 5

# Check the output
ls questions/M-5_mydataset_data-generator/mydataset_task/
```

You should see:

```
questions/
└── M-5_mydataset_data-generator/
    ├── clip_config.json
    └── mydataset_task/
        ├── mydataset_0000/
        │   ├── first_frame.png
        │   ├── prompt.txt
        │   ├── final_frame.png
        │   ├── ground_truth.mp4
        │   └── original/
        │       ├── question.json
        │       └── <source_image>.png
        ├── mydataset_0001/
        └── ...
```

## Step 4: Update the docs

After confirming everything works:

1. Add your generator to the table in `docs/DATASET.md` (the ledger).
2. Add your generator to the "Registered generators" table in `docs/VIDEO_MCP_DATA.md`.

## Reference

### DatasetAdapter interface

Your adapter class must implement these four members:

| Member | Type | Description |
|---|---|---|
| `name` | `@property` -> `str` | Short slug for CLI flags. Must match the string passed to `@register_adapter()`. Example: `"scienceqa"` |
| `generator_id` | `@property` -> `str` | VBVR-style prefix. Use `M-{N}` where N is the next available number. Example: `"M-2"` |
| `download(*, out_dir)` | method -> `Path` | Download the raw dataset artifact to `out_dir` and return the path to the downloaded file. |
| `iter_mcqa_vqa()` | method -> `Iterator[tuple[McqaVqaSample, bytes]]` | Yield `(sample, image_bytes)` pairs for every valid MCQA-VQA example in the dataset. |

Your adapter also gets a free derived property:

| Member | Type | Description |
|---|---|---|
| `generator_name` | `@property` -> `str` | Full VBVR directory name, e.g. `"M-2_scienceqa_data-generator"`. Computed automatically from `generator_id` and `name`. |

### McqaVqaSample fields

The Pydantic model you yield from `iter_mcqa_vqa()`:

| Field | Type | Description |
|---|---|---|
| `dataset` | `str` | Human-readable dataset name (e.g. `"ScienceQA"`) |
| `source_id` | `str` | Original row/sample ID from the source dataset |
| `question` | `str` | The question text |
| `choices` | `dict[str, str]` | Map of `"A"` -> choice text, `"B"` -> choice text, etc. (2-4 keys) |
| `answer` | `Choice` | The correct answer: `"A"`, `"B"`, `"C"`, or `"D"` |
| `image_filename` | `str` | Basename of the original image file (e.g. `"image_042.png"`) |

### VBVR naming convention

The output directory structure uses consistent naming across all three levels:

```
questions/
└── {generator_id}_{name}_data-generator/    <- generator level
    ├── clip_config.json
    └── {name}_task/                         <- task level
        └── {name}_{NNNN}/                   <- sample level (zero-padded)
            ├── first_frame.png              <- VBVR required
            ├── prompt.txt                   <- VBVR required
            ├── final_frame.png              <- VBVR optional
            ├── ground_truth.mp4             <- VBVR optional
            └── original/                    <- video-mcp extra
                ├── question.json
                └── <source_image>
```

### What the pipeline does with your adapter

When a user runs `python -m video_mcp.dataset process --dataset mydataset`, the pipeline:

1. Calls `adapter.iter_mcqa_vqa()` to get samples.
2. For each `(McqaVqaSample, image_bytes)` pair:
   - Saves the original image and structured `question.json` to `original/`.
   - Generates `prompt.txt` from the question, choices, and answer.
   - Renders 81 frames in a temp directory (frame 0 = no highlight, frames 1-80 = progressive answer reveal).
   - Saves frame 0 as `first_frame.png` and frame 80 as `final_frame.png`.
   - Compiles all 81 frames into `ground_truth.mp4` via ffmpeg.
   - Cleans up the temp directory.
3. Writes `clip_config.json` once at the generator root.

You don't need to implement any of this -- just yield the samples and the pipeline handles the rest.

## Tips

- **Filter aggressively** in `iter_mcqa_vqa()`. Skip samples that have videos instead of images, multiple images, missing choices, or invalid answers. Only yield clean MCQA-VQA examples.
- **Use Pydantic models** for raw data parsing. Define a model for the raw row format and use `model_validate()` -- this gives you automatic type checking and clear error messages.
- **Test with `--limit 3`** first. Rendering 81 frames per sample takes a few seconds, so start small.
- **Check existing adapters** for real-world examples that handle messy data (choice string parsing, image placeholder stripping, ZIP file extraction, etc.).

## Existing adapters

For reference, the current adapters handle a variety of data formats:

| Adapter | Key Challenges |
|---|---|
| `corecognition` (M-1) | Gated ZIP from HF Hub, CSV parsing, messy choice strings with `nan` values |
| `scienceqa` (M-2) | Multiple splits, answer normalization (index or letter), various image formats |
| `mathvision` (M-3) | Competition math, `<image>` tag stripping, letter-only options |
| `phyx` (M-4) | Physics reasoning, option prefix stripping (`A: 28.5`), combined question descriptions |

See the files in `video_mcp/datasets/` for full implementations.
