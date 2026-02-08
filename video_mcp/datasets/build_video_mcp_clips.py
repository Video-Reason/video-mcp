from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from video_mcp.datasets.corecognition import download_corecognition_complete_zip, iter_corecognition_mcqa_single_image
from video_mcp.render.mcqa_overlay import Choice, make_fonts, render_video_mcp_frame
from video_mcp.video_spec import VideoSpec


class VideoMcpOriginalQuestion(BaseModel):
    dataset: str
    source_id: str
    question: str
    choices: dict[str, str]
    answer: Choice
    original_image_filename: str


class VideoMcpClipConfig(BaseModel):
    fps: int
    seconds: float
    num_frames: int
    width: int
    height: int


def build_video_mcp_clips_corecognition_mcqa_single_image(
    *,
    out_dir: Path,
    video: VideoSpec | None = None,
    limit: int | None = None,
) -> int:
    """
    Build Video-MCP clips:
    - 16 FPS, 3 seconds (48 frames) by default
    - frame_0000: MCQA VQA panel shown, no highlight
    - frame_0001..: only corner boxes; correct choice is lit
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v = video or VideoSpec()
    fonts = make_fonts()

    zip_path = download_corecognition_complete_zip()
    import zipfile

    z = zipfile.ZipFile(zip_path)
    available = set(z.namelist())

    n = 0
    # Dataset-level config (written once)
    cfg = VideoMcpClipConfig(
        fps=int(v.fps),
        seconds=float(v.seconds),
        num_frames=int(v.num_frames),
        width=int(v.width),
        height=int(v.height),
    )
    (out_dir / "clip_config.json").write_text(cfg.model_dump_json(), encoding="utf-8")

    for ex in iter_corecognition_mcqa_single_image(split="train", config="complete"):
        if limit is not None and n >= int(limit):
            break

        if ex.media_path not in available:
            continue

        img_bytes = z.read(ex.media_path)
        img_obj = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        sample_id = f"corecognition_{ex.id}"
        sample_dir = out_dir / sample_id
        original_dir = sample_dir / "original"
        frames_dir = sample_dir / "frames"
        original_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Save original image
        original_image_name = Path(ex.image).name
        (original_dir / original_image_name).write_bytes(img_bytes)

        # Save original question.json
        q = VideoMcpOriginalQuestion(
            dataset="CoreCognition",
            source_id=str(ex.id),
            question=ex.question,
            choices=ex.choices,
            answer=ex.answer,
            original_image_filename=original_image_name,
        )
        (original_dir / "question.json").write_text(q.model_dump_json(), encoding="utf-8")

        # Build choices list in A/B/C/D order for display.
        choices_display = [f"{k}: {ex.choices[k]}" for k in ["A", "B", "C", "D"] if k in ex.choices]

        for frame_idx in range(v.num_frames):
            show_panel = frame_idx == 0
            lit: Choice | None = None
            if frame_idx >= 1:
                lit = ex.answer  # correct answer

            frame = render_video_mcp_frame(
                width=int(v.width),
                height=int(v.height),
                question=ex.question,
                choices=choices_display,
                image=img_obj if show_panel else None,
                show_panel=show_panel,
                lit_choice=lit,
                fonts=fonts,
            )
            frame.save(frames_dir / f"frame_{frame_idx:04d}.png", format="PNG")

        n += 1

    z.close()
    return n

