from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from video_mcp.process.adapter import DatasetAdapter
from video_mcp.mcqa import CHOICE_ORDER, Choice, LitStyle
from video_mcp.render.mcqa_overlay import make_fonts, render_video_mcp_frame
from video_mcp.video_spec import VideoSpec


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

TASK_NAME_FALLBACK = "mcqa_vqa"
"""Fallback task name when adapter name is unavailable."""


class VideoMcpOriginalQuestion(BaseModel):
    """Structured metadata preserved in ``original/question.json``."""

    dataset: str
    source_id: str
    question: str
    choices: dict[str, str]
    answer: Choice
    original_image_filename: str


class VideoMcpClipConfig(BaseModel):
    """Dataset-level video configuration saved at the generator root."""

    fps: int
    seconds: float
    num_frames: int
    width: int
    height: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_prompt_txt(
    question: str,
    choices: dict[str, str],
    answer: Choice,
) -> str:
    """Format a human-readable ``prompt.txt`` from MCQA fields."""
    lines = [question, ""]
    for key in CHOICE_ORDER:
        if key in choices:
            lines.append(f"{key}: {choices[key]}")
    lines.append("")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines) + "\n"


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


# ---------------------------------------------------------------------------
# Main builder — VBVR-compatible output
# ---------------------------------------------------------------------------


def build_video_mcp_clips(
    adapter: DatasetAdapter,
    *,
    out_dir: Path,
    video: VideoSpec | None = None,
    limit: int | None = None,
    lit_style: LitStyle = "darken",
) -> int:
    """Build Video-MCP clips following the **VBVR DataFactory** output layout.

    The task name is derived from the adapter name so that generator and task
    are consistent (matching VBVR convention).

    Output structure (per sample)::

        {out_dir}/
        └── {generator_id}_{name}_data-generator/
            ├── clip_config.json
            └── {name}_task/
                └── {name}_{0000}/
                    ├── first_frame.png
                    ├── prompt.txt
                    ├── final_frame.png
                    ├── ground_truth.mp4
                    └── original/
                        ├── question.json
                        └── <source_image>

    Wan2.2-I2V-A14B defaults: 480p (832×480), 16 FPS, 81 frames (~5 s).

    - first_frame  : MCQA panel shown, no answer highlight
    - final_frame  : MCQA panel shown, correct answer fully highlighted
    - ground_truth : full clip with progressive answer reveal
    """
    out_dir = Path(out_dir)

    v = video or VideoSpec()
    fonts = make_fonts(width=int(v.width), height=int(v.height))

    # VBVR convention: task name == adapter name (consistent naming)
    # Generator name includes the ID prefix, e.g. M-1_corecognition_data-generator
    task_name = adapter.name
    generator_name = adapter.generator_name
    generator_dir = out_dir / generator_name
    task_dir = generator_dir / f"{task_name}_task"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Dataset-level config (written once at generator root)
    cfg = VideoMcpClipConfig(
        fps=int(v.fps),
        seconds=float(v.seconds),
        num_frames=int(v.num_frames),
        width=int(v.width),
        height=int(v.height),
    )
    (generator_dir / "clip_config.json").write_text(
        cfg.model_dump_json(indent=2), encoding="utf-8"
    )

    total = f"/{limit}" if limit is not None else ""
    n = 0
    for sample, image_bytes in adapter.iter_mcqa_vqa():
        if limit is not None and n >= int(limit):
            break

        # VBVR uses zero-padded 4-digit indices
        sample_folder = f"{task_name}_{n:04d}"
        sample_dir = task_dir / sample_folder

        n += 1
        print(
            f"[{n}{total}] {generator_name}/{task_name}_task/{sample_folder} "
            f"(src: {sample.source_id}, {v.num_frames} frames)"
        )

        img_obj = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Create VBVR-compatible sample directory
        sample_dir.mkdir(parents=True, exist_ok=True)

        # --- original/ subfolder (preserves source data) -----------------
        original_dir = sample_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)

        (original_dir / sample.image_filename).write_bytes(image_bytes)

        q = VideoMcpOriginalQuestion(
            dataset=sample.dataset,
            source_id=sample.source_id,
            question=sample.question,
            choices=sample.choices,
            answer=sample.answer,
            original_image_filename=sample.image_filename,
        )
        (original_dir / "question.json").write_text(
            q.model_dump_json(indent=2), encoding="utf-8"
        )

        # --- prompt.txt (VBVR required) ----------------------------------
        prompt_text = format_prompt_txt(sample.question, sample.choices, sample.answer)
        (sample_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

        # --- Render frames in a temp directory ---------------------------
        choices_display = [
            f"{k}: {sample.choices[k]}" for k in CHOICE_ORDER if k in sample.choices
        ]

        with tempfile.TemporaryDirectory() as tmp_frames:
            tmp_frames_path = Path(tmp_frames)

            for frame_idx in range(v.num_frames):
                lit: Choice | None = None
                progress = 0.0
                if frame_idx >= 1:
                    lit = sample.answer
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
                frame.save(tmp_frames_path / f"frame_{frame_idx:04d}.png", format="PNG")

                # Save first and last frames as VBVR output files
                if frame_idx == 0:
                    frame.save(sample_dir / "first_frame.png", format="PNG")
                elif frame_idx == v.num_frames - 1:
                    frame.save(sample_dir / "final_frame.png", format="PNG")

            # --- ground_truth.mp4 (VBVR optional) -----------------------
            compile_frames_to_mp4(
                tmp_frames_path,
                sample_dir / "ground_truth.mp4",
                fps=int(v.fps),
                width=int(v.width),
                height=int(v.height),
            )

        print(f"  -> {sample_dir}")

    return n
