# Claude Notes (video-mcp)

This repo generates a **Video-MCP** dataset: short clips that embed a MCQA-VQA UI
in the frames, where the correct answer is revealed by highlighting one of
**A/B/C/D** over time.

## What to run

Primary CLI (recommended):

```bash
python -m video_mcp.dataset download --dataset <name>
python -m video_mcp.dataset process  --dataset <name> [--limit N]
```

Defaults:
- raw downloads: `data/raw/`
- processed outputs: `questions/`

## Output layout (what "process" writes)

`process` uses a VBVR-compatible layout:

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
- sample indices are zero-based, 4-digit zero-padded (`..._0000`, `..._0049`, ...)
- `first_frame.png` and `final_frame.png` are always PNG
- the original image is preserved with its original filename/extension

## Video spec (defaults and constraints)

Defaults (Wan2.2-I2V-A14B aligned):
- 832x480
- 81 frames @ 16 FPS (~5.06s)

Constraints (validated in `video_mcp/video_spec.py`):
- width/height divisible by 8
- num_frames must satisfy `1 + 4k` (e.g. 49, 81)

Override via CLI:

```bash
python -m video_mcp.dataset process --dataset <name> --width 1280 --height 720
python -m video_mcp.dataset process --dataset <name> --num-frames 49
```

## Key code locations

- CLI entrypoint: `video_mcp/dataset.py`
- Output writer: `video_mcp/process/build_video_mcp_clips.py`
- Adapter interface/registry: `video_mcp/process/adapter.py`
- Render overlay: `video_mcp/render/mcqa_overlay.py`
- Video constraints/spec: `video_mcp/video_spec.py`
- Built-in adapters:
  - `video_mcp/datasets/corecognition.py`
  - `video_mcp/datasets/scienceqa.py`

## Adding a new dataset adapter

1. Create `video_mcp/datasets/<name>.py`
2. Implement `DatasetAdapter`:
   - `name` (slug for `--dataset`)
   - `generator_id` (e.g. `M-3`)
   - `download(out_dir=...)`
   - `iter_mcqa_vqa()` yielding `(McqaVqaSample, image_bytes)`
3. Register it with `@register_adapter("<name>")`
4. Import it in `video_mcp/datasets/__init__.py` so registration executes
5. (Recommended) implement HF tracking properties for run manifests:
   - `hf_repo_id`, `hf_config`, `hf_split`, `hf_revision`

## Runtime requirements

- `ffmpeg` must be available on PATH (used to compile `ground_truth.mp4`)
- Some Hugging Face datasets require `HF_TOKEN` in `.env`

## Run traceability

Each `process` run writes:
- `run_manifest.json` (latest)
- `run_manifests/<UTC_RUN_ID>.json` (history)

The manifest includes:
- dataset identity (HF repo/config/split/revision when provided by adapter)
- processing parameters
- code version (git commit/branch when available)
- runtime versions (python, ffmpeg)

