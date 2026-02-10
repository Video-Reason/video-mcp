## Video-MCP dataset: what "correct" means

This project defines **Video-MCP data** as **short video clips** that encode a
multiple-choice VQA interaction in the video itself.

The core idea is:
- The model first **sees a prompt UI** (question + choices + image).
- Then the model must **answer by "lighting up"** the correct option (**A/B/C/D**)
  across later frames.

## Clip spec (default, Wan2.2-I2V-A14B aligned)

- **FPS**: 16
- **Frames per clip**: 81
- **Duration**: 81 / 16 = **5.0625 seconds**
- **Resolution**: 832×480 (480p tier)
- **Output video**: H.264 MP4 (`ground_truth.mp4`) compiled via `ffmpeg`

## Frame layout

Each frame uses a **two-column panel** centered between the A/B/C/D corner boxes:

- **Left column**: the source image, scaled to fill the available area.
- **Right column**: question text at the top, then the four choice options.
- **Corners**: A (top-left), B (top-right), C (bottom-left), D (bottom-right).

## Frame semantics

For each clip:

- **Frame 0**:
  - Shows the full question panel (image + question + choices) and the four corner boxes.
  - **No highlight** on frame 0 (the model is "reading").

- **Frames 1..(N-1)**:
  - Panel remains visible.
  - The correct corner box **gradually highlights** across the clip duration.

## Highlight styles (`--lit-style`)

| Style | Effect |
|---|---|
| `darken` (default) | Correct corner box gradually darkens |
| `red_border` | Thick red outline gradually appears around the correct corner box |

## Output layout (VBVR DataFactory)

Processing uses a VBVR-compatible layout rooted at `questions/` by default:

```
questions/
  {generator_id}_{dataset}_data-generator/
    clip_config.json
    run_manifest.json
    run_manifests/
      <UTC_RUN_ID>.json
    {dataset}_task/
      {dataset}_0000/
        first_frame.png
        prompt.txt
        final_frame.png
        ground_truth.mp4
        original/
          question.json
          <original_image_file>
```

Naming rules:

- Sample indices are **zero-based** and **4-digit zero-padded**: `..._0000`, `..._0049`, ...
- `first_frame.png`/`final_frame.png` are always PNG; the original source image is preserved
  with its original filename/extension (commonly `.png`, `.jpg`, `.jpeg`).
- Intermediate per-frame PNGs are rendered in a temporary directory and are not kept.

### `original/question.json` schema (per sample)

Each sample's `original/question.json` contains (at minimum):

- `dataset`: source dataset name (e.g. `ScienceQA`, `CoreCognition`)
- `source_id`: original dataset id (traceability)
- `question`: question string
- `choices`: dict mapping `A/B/C/D` to choice text
- `answer`: one of `A/B/C/D`
- `original_image_filename`: filename of the preserved original image

### `clip_config.json` schema (dataset-level)

- `fps`, `seconds`, `num_frames`, `width`, `height`

### Run manifest

Each `process` run writes:

- `{generator}/run_manifest.json` (latest)
- `{generator}/run_manifests/<UTC_RUN_ID>.json` (history)

It records dataset identity (including HF fields when available), processing parameters,
code version (git commit/branch when available), and runtime info.

## CLI (how we build datasets)

Every registered dataset adapter is available through the same commands:

```bash
python -m video_mcp.dataset download --dataset <name>
python -m video_mcp.dataset process  --dataset <name>
```

Options for `process`:
- `--limit N` — build only the first N samples (useful for quick testing).
- `--lit-style darken|red_border` — choose the highlight style (default: `darken`).
- `--width/--height/--num-frames` — override the default video spec (validated).

Requires `ffmpeg` on the system PATH.

## Adding a new dataset

See the [README](../README.md#adding-a-new-dataset) for a step-by-step guide.
In short: create one file in `video_mcp/datasets/`, implement the `DatasetAdapter`
interface, and register it.
