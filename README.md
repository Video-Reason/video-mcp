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

## Output format

All outputs live under `data/` (gitignored):
- **Raw downloads**: `data/raw/`
- **Processed Video-MCP dataset**: `data/processed/`

CoreCognition processed output root:
- `data/processed/corecognition_video_mcp/`

Each sample:
- `data/processed/corecognition_video_mcp/<sample_id>/original/question.json`
- `data/processed/corecognition_video_mcp/<sample_id>/original/<original_image_file>`
- `data/processed/corecognition_video_mcp/<sample_id>/frames/frame_0000.png ... frame_0047.png`
