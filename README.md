# video-mcp
Model Context Protocol (MCP)-style **video MCQA** dataset utilities.

We generate short video clips where:
- **FPS**: 16
- **Duration**: 3 seconds (\(48\) frames)
- **Frame 0**: shows the MCQA/VQA prompt UI (like the provided figure)
- **Frames 1..N**: the model should “light up” the correct answer by highlighting the matching corner box (**A/B/C/D**)

## Quickstart

Create a venv, install pinned deps, and generate a small dataset:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

python -m video_mcp --out-dir data/generated --split train --num-samples 100 --seed 0 --overlay-mode all_frames
```

## Output format

Generated files (ignored by git via `.gitignore`):
- **Frames**: `data/generated/train/<sample_id>/frames/frame_0000.png ... frame_0047.png`
- **Metadata**: `data/generated/train/metadata.jsonl` (one JSON per sample; includes `answer` and `frames_dir`)
