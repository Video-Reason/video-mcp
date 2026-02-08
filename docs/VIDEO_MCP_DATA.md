## Video-MCP dataset: what "correct" means

This project defines **Video-MCP data** as **short video clips** that encode a multiple-choice VQA interaction in the video itself.

The core idea is:
- The model first **sees a prompt UI** (question + choices + image).
- Then the model must **answer by "lighting up"** the correct option (**A/B/C/D**) in the video frames.

### Clip spec (default)

- **FPS**: 16
- **Duration**: 3 seconds
- **Frames per clip**: 48 (`frame_0000.png` … `frame_0047.png`)
- **Resolution**: 1024×768 (configurable)
- **Output video**: `clip.mp4` (H.264, compiled from frames via ffmpeg)

### Frame layout

Each frame uses a **two-column panel** centred between the A/B/C/D corner boxes:

- **Left column (~55%)**: the source image, scaled to fill the available area.
- **Right column (~45%)**: question text at the top, a separator, then the four choice options.
- **Corners**: A (top-left), B (top-right), C (bottom-left), D (bottom-right) answer boxes.

### Frame semantics

For each clip:

- **Frame 0 (`frame_0000.png`)**:
  - Shows the full question panel (image + question + choices) and the four corner answer boxes.
  - **No highlight** on frame 0 (the model is "reading").

- **Frames 1–47 (`frame_0001.png` … `frame_0047.png`)**:
  - Question panel remains visible.
  - The correct answer's corner box **gradually highlights** across the full clip duration (linear fade-in from frame 1 to frame 47).

### Highlight styles (`--lit-style`)

| Style | Effect |
|---|---|
| `darken` (default) | Correct corner box gradually darkens; letter stays dark |
| `red_border` | Thick red outline gradually appears around the correct corner box |

### Sample ID format

Each sample folder is named `<datasetname>_<n>` where `<n>` is a sequential integer starting from 1 (e.g. `corecognition_1`, `corecognition_2`, …). The original source dataset ID is preserved inside `original/question.json` for traceability.

### Folder layout (canonical)

All dataset generation is organized under:

- `data/raw/`: raw downloads (zips, original files)
- `data/processed/`: processed Video-MCP-ready outputs

For a processed Video-MCP dataset, the layout is:

```
data/processed/<datasetname>_video_mcp/
  clip_config.json
  <datasetname>_1/
    original/
      question.json
      <original_image_file>
    frames/
      frame_0000.png
      frame_0001.png
      ...
      frame_0047.png
    video/
      clip.mp4
  <datasetname>_2/
    ...
```

### `original/question.json` schema (per sample)

Each sample's `original/question.json` contains (at minimum):

- **dataset**: source dataset name (e.g. `CoreCognition`)
- **source_id**: original dataset id (for traceability)
- **question**: question string
- **choices**: dict mapping `A/B/C/D` to choice text
- **answer**: one of `A/B/C/D`
- **original_image_filename**: the filename of the original image saved beside `question.json`

### `clip_config.json` schema (dataset-level)

- **fps**, **seconds**, **num_frames**, **width**, **height**

### CLI (how we build datasets)

Every registered dataset adapter is available through the same commands:

```bash
python -m video_mcp.dataset download --dataset <name>
python -m video_mcp.dataset process  --dataset <name>
```

Options for `process`:
- `--limit N` — build only the first N samples (useful for quick testing).
- `--lit-style darken|red_border` — choose the highlight style (default: `darken`).

Requires `ffmpeg` on the system PATH.

### Adding a new dataset

See the [README](../README.md#adding-a-new-dataset) for a step-by-step guide.
In short: create one file in `video_mcp/datasets/`, implement the `DatasetAdapter`
interface, and register it — the CLI and all build scripts pick it up automatically.

### CoreCognition specifics

- Current supported subset: **single-image MCQA VQA** from CoreCognition (753 samples).
- Source raw artifact: `CoreCognition_20250622.zip` (stored under `data/raw/corecognition/`).
- The adapter always uses the **complete** ZIP (real images required for rendering).
