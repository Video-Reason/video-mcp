from __future__ import annotations

import io
import subprocess
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from video_mcp.process.adapter import DatasetAdapter
from video_mcp.mcqa import CHOICE_ORDER, Choice, LitStyle
from video_mcp.render.mcqa_overlay import make_fonts, render_video_mcp_frame
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


def compile_frames_to_mp4(
    frames_dir: Path,
    output_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> Path:
    """Compile a directory of numbered PNGs into an MP4 using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-loglevel", "error",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def build_video_mcp_clips(
    adapter: DatasetAdapter,
    *,
    out_dir: Path,
    video: VideoSpec | None = None,
    limit: int | None = None,
    lit_style: LitStyle = "darken",
) -> int:
    """
    Build Video-MCP clips from **any** adapter.

    Wan2.2-I2V-A14B defaults: 480p (832×480), 16 FPS, 81 frames (~5 s).

    - frame_0000: MCQA VQA panel shown, no highlight
    - frame_0001..: MCQA VQA panel shown, correct choice is lit
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v = video or VideoSpec()
    fonts = make_fonts(width=int(v.width), height=int(v.height))

    # Dataset-level config (written once)
    cfg = VideoMcpClipConfig(
        fps=int(v.fps),
        seconds=float(v.seconds),
        num_frames=int(v.num_frames),
        width=int(v.width),
        height=int(v.height),
    )
    (out_dir / "clip_config.json").write_text(cfg.model_dump_json(), encoding="utf-8")

    total = f"/{limit}" if limit is not None else ""
    n = 0
    for sample, image_bytes in adapter.iter_mcqa_vqa():
        if limit is not None and n >= int(limit):
            break

        n += 1
        sample_id = f"{adapter.name}_{n}"
        print(f"[{n}{total}] {sample_id} (src: {sample.source_id}, {v.num_frames} frames)")

        img_obj = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        sample_dir = out_dir / sample_id
        original_dir = sample_dir / "original"
        frames_dir = sample_dir / "frames"
        video_dir = sample_dir / "video"
        original_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        # Save original image
        (original_dir / sample.image_filename).write_bytes(image_bytes)

        # Save original question.json
        q = VideoMcpOriginalQuestion(
            dataset=sample.dataset,
            source_id=sample.source_id,
            question=sample.question,
            choices=sample.choices,
            answer=sample.answer,
            original_image_filename=sample.image_filename,
        )
        (original_dir / "question.json").write_text(q.model_dump_json(), encoding="utf-8")

        # Build choices list in A/B/C/D order for display.
        choices_display = [f"{k}: {sample.choices[k]}" for k in CHOICE_ORDER if k in sample.choices]

        for frame_idx in range(v.num_frames):
            lit: Choice | None = None
            progress = 0.0
            if frame_idx >= 1:
                lit = sample.answer  # correct answer
                # Spread the fade-in across the entire clip (frames 1 → last).
                progress = frame_idx / (v.num_frames - 1)

            frame = render_video_mcp_frame(
                width=int(v.width),
                height=int(v.height),
                question=sample.question,
                choices=choices_display,
                image=img_obj,
                show_panel=True,
                lit_choice=lit,
                lit_progress=progress,
                lit_style=lit_style,
                fonts=fonts,
            )
            frame.save(frames_dir / f"frame_{frame_idx:04d}.png", format="PNG")

        # Compile frames into MP4 video
        mp4_path = video_dir / "clip.mp4"
        compile_frames_to_mp4(
            frames_dir,
            mp4_path,
            fps=int(v.fps),
            width=int(v.width),
            height=int(v.height),
        )
        print(f"  -> {mp4_path}")

    return n
