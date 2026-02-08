## Video-MCP dataset: what “correct” means

This project defines **Video-MCP data** as **short video clips** that encode a multiple-choice VQA interaction in the video itself.

The core idea is:
- The model first **sees a prompt UI** (question + choices + image).
- Then the model must **answer by “lighting up”** the correct option (**A/B/C/D**) in the video frames.

### Clip spec (default)

- **FPS**: 16
- **Duration**: 3 seconds
- **Frames per clip**: 48 (`frame_0000.png` … `frame_0047.png`)
- **Resolution**: 1024×768 (configurable)

### Frame semantics (the important part)

For each clip:

- **Frame 0 (`frame_0000.png`)**:
  - Shows the **MCQA VQA UI** (like the reference figure):
    - A central panel with a “Questions” header
    - The question text
    - A/B/C/D options text
    - The associated image (rendered inside an image box)
  - Also shows the **four corner answer boxes** labeled **A, B, C, D**
  - **No highlight is shown** on frame 0 (the model is “reading”)

- **Frames 1..47 (`frame_0001.png` … `frame_0047.png`)**:
  - The central question panel is **hidden**
  - Only the **corner answer boxes** remain
  - The **correct answer’s box is “lit”** (filled dark)
  - All other boxes are unlit (light background)

This matches the intended supervision: the model should learn to output the correct “highlighted choice”.

### Folder layout (canonical)

All dataset generation is organized under:

- `data/raw/`: raw downloads (zips, original files)
- `data/processed/`: processed Video-MCP-ready outputs

For a processed Video-MCP dataset, the layout is:

```
data/processed/<datasetname>_video_mcp/
  clip_config.json
  <sample_id_1>/
    original/
      question.json
      <original_image_file>
    frames/
      frame_0000.png
      frame_0001.png
      ...
      frame_0047.png
  <sample_id_2>/
    original/
      question.json
      <original_image_file>
    frames/
      ...
```

### `original/question.json` schema (per sample)

Each sample’s `original/question.json` contains (at minimum):

- **dataset**: source dataset name (e.g. `CoreCognition`)
- **source_id**: original dataset id
- **question**: question string
- **choices**: dict mapping `A/B/C/D` to choice text
- **answer**: one of `A/B/C/D`
- **original_image_filename**: the filename of the original image saved beside `question.json`

### `clip_config.json` schema (dataset-level)

- **fps**, **seconds**, **num_frames**, **width**, **height**

### CLI (how we build datasets)

Canonical usage is via:

```bash
python -m video_mcp.dataset download --dataset corecognition
python -m video_mcp.dataset process --dataset corecognition
```

Notes:
- `download` places raw artifacts under `data/raw/...` (symlinked by default to avoid duplicating large files).
- `process` generates the **Video-MCP clip frames** under `data/processed/...`.

### CoreCognition specifics

- Current supported subset: **single-image MCQA VQA** from CoreCognition (753 samples).
- Source raw artifact: `CoreCognition_20250622.zip` (stored under `data/raw/corecognition/` as a symlink to the local HF cache).

