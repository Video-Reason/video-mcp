# Video-MCP dataset specification

This project defines **Video-MCP data** as **short video clips** that encode a multiple-choice VQA interaction in the video itself.

The core idea is:
- The model first **sees a prompt UI** (question + choices + image).
- Then the model must **answer by "lighting up"** the correct option (**A/B/C/D**) in the video frames.

Output follows the **[VBVR DataFactory](https://github.com/video-reason/VBVR-DataFactory)** directory convention.

## Clip spec (default)

Defaults are aligned with **Wan2.2-I2V-A14B** fine-tuning requirements:

- **FPS**: 16
- **Duration**: ~5 seconds
- **Frames per clip**: 81 (must satisfy `1 + 4k` for VAE temporal compression)
- **Resolution**: 832x480 (must be divisible by 8 for VAE spatial compression)
- **Output video**: `ground_truth.mp4` (H.264, yuv420p, compiled from frames via ffmpeg)

## VBVR-compatible folder layout

All output follows the VBVR DataFactory naming convention:

```
questions/
└── {generator_id}_{name}_data-generator/
    ├── clip_config.json
    └── {name}_task/
        ├── {name}_0000/
        │   ├── first_frame.png          # VBVR required
        │   ├── prompt.txt               # VBVR required
        │   ├── final_frame.png          # VBVR optional
        │   ├── ground_truth.mp4         # VBVR optional
        │   └── original/               # video-mcp extra (traceability)
        │       ├── question.json
        │       └── <source_image>
        ├── {name}_0001/
        │   └── [same structure]
        └── ...
```

### Concrete example (CoreCognition, generator `M-1`):

```
questions/
└── M-1_corecognition_data-generator/
    ├── clip_config.json
    └── corecognition_task/
        ├── corecognition_0000/
        │   ├── first_frame.png
        │   ├── prompt.txt
        │   ├── final_frame.png
        │   ├── ground_truth.mp4
        │   └── original/
        │       ├── question.json
        │       └── a0052.png
        ├── corecognition_0001/
        └── ...
```

### Naming convention

The generator name, task directory, and sample folders all share the same core name (matching VBVR convention):

| Level | Pattern | Example |
|---|---|---|
| Generator dir | `{generator_id}_{name}_data-generator/` | `M-1_corecognition_data-generator/` |
| Task dir | `{name}_task/` | `corecognition_task/` |
| Sample dir | `{name}_{NNNN}/` | `corecognition_0000/` |

- `generator_id`: VBVR-style prefix assigned per adapter (e.g. `M-1`, `M-2`)
- `name`: adapter slug (e.g. `corecognition`, `scienceqa`)
- `NNNN`: zero-padded 4-digit index starting at 0

## VBVR file contract

Per the [VBVR DataFactory validator](https://github.com/video-reason/VBVR-DataFactory):

| File | Required | Description |
|---|---|---|
| `first_frame.png` | Yes | Starting state -- rendered frame 0, no answer highlight |
| `prompt.txt` | Yes | Human-readable question, choices, and answer |
| `final_frame.png` | No | Ending state -- rendered last frame, answer fully highlighted |
| `ground_truth.mp4` | No | Full video with progressive answer reveal |

The `original/` subdirectory is a video-mcp addition (invisible to the VBVR validator since it only checks files, not subdirectories).

## Frame layout

Each frame uses a **two-column panel** centred between the A/B/C/D corner boxes:

- **Left column**: the source image, scaled to fill the available area.
- **Right column**: question text at the top, a separator, then the four choice options.
- **Corners**: A (top-left), B (top-right), C (bottom-left), D (bottom-right) answer boxes.

The renderer uses adaptive font sizing to prevent text truncation for long questions.

## Frame semantics

For each clip:

- **first_frame.png** (frame 0):
  - Shows the full question panel (image + question + choices) and the four corner answer boxes.
  - **No highlight** -- the model is "reading".

- **Intermediate frames** (frames 1 through N-2):
  - Question panel remains visible.
  - The correct answer's corner box **gradually highlights** (linear fade-in).
  - These frames are rendered in a temp directory for video compilation but are **not** saved to the output.

- **final_frame.png** (frame N-1):
  - Correct answer is **fully highlighted**.

- **ground_truth.mp4**:
  - All frames (0 through N-1) compiled at 16 FPS via ffmpeg.

## Highlight styles (`--lit-style`)

| Style | Effect |
|---|---|
| `darken` (default) | Correct corner box gradually darkens; letter stays dark |
| `red_border` | Thick red outline gradually appears around the correct corner box |

## prompt.txt format

Human-readable plain text with question, choices, and answer:

```
What color is the object in the image?

A: Red
B: Blue
C: Green
D: Yellow

Answer: A
```

## original/question.json schema (per sample)

Structured metadata preserved for traceability (Pydantic model):

- **dataset**: source dataset name (e.g. `CoreCognition`)
- **source_id**: original dataset id
- **question**: question string
- **choices**: dict mapping `A/B/C/D` to choice text
- **answer**: one of `A/B/C/D`
- **original_image_filename**: the filename of the original image saved beside `question.json`

## clip_config.json schema (generator-level)

Written once at the generator root directory. Fields:

- **fps**: frames per second (default: 16)
- **seconds**: clip duration derived from `num_frames / fps`
- **num_frames**: total frames per clip (default: 81)
- **width**: frame width in pixels (default: 832)
- **height**: frame height in pixels (default: 480)

## CLI

Every registered dataset adapter is available through the same commands:

```bash
python -m video_mcp.dataset download --dataset <name>
python -m video_mcp.dataset process  --dataset <name>
```

Options for `process`:
- `--out-dir PATH` -- output root (default: `questions/`)
- `--limit N` -- build only the first N samples (useful for quick testing)
- `--lit-style darken|red_border` -- choose the highlight style (default: `darken`)
- `--width N` -- frame width in px, must be divisible by 8 (default: 832)
- `--height N` -- frame height in px, must be divisible by 8 (default: 480)
- `--num-frames N` -- frames per clip, must satisfy `1 + 4k` (default: 81)

Requires `ffmpeg` on the system PATH.

## Adding a new dataset

See [ADD_DATASET.md](./ADD_DATASET.md) for the full step-by-step guide.
In short: create one file in `video_mcp/datasets/`, implement the `DatasetAdapter`
interface (including `name`, `generator_id`, `download`, and `iter_mcqa_vqa`),
register it -- the CLI and all build scripts pick it up automatically.

## Registered generators

| Generator ID | Name | Dataset | Output path |
|---|---|---|---|
| `M-1` | `corecognition` | CoreCognition (single-image MCQA VQA) | `questions/M-1_corecognition_data-generator/` |
| `M-2` | `scienceqa` | ScienceQA (image MCQA) | `questions/M-2_scienceqa_data-generator/` |
| `M-3` | `mathvision` | MathVision (competition math MCQA) | `questions/M-3_mathvision_data-generator/` |
| `M-4` | `phyx` | PhyX (physics reasoning MCQA) | `questions/M-4_phyx_data-generator/` |
