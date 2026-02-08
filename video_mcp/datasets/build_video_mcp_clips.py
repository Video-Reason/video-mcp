from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from video_mcp.datasets.corecognition import download_corecognition_complete_zip, iter_corecognition_mcqa_single_image
from video_mcp.render.mcqa_overlay import Choice, make_fonts, render_video_mcp_frame
from video_mcp.video_spec import VideoSpec


class VideoMcpClipRecord(BaseModel):
    dataset: str
    split: str
    source_id: str

    fps: int
    seconds: float
    num_frames: int
    width: int
    height: int

    question: str
    choices: dict[str, str]
    answer: Choice

    frames_dir: str


def build_video_mcp_clips_corecognition_mcqa_single_image(
    *,
    out_dir: Path,
    split: str = "train",
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
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    meta_path = split_dir / "metadata.jsonl"

    v = video or VideoSpec()
    fonts = make_fonts()

    zip_path = download_corecognition_complete_zip()
    import zipfile

    z = zipfile.ZipFile(zip_path)
    available = set(z.namelist())

    n = 0
    with meta_path.open("w", encoding="utf-8") as f:
        for ex in iter_corecognition_mcqa_single_image(split=split, config="complete"):
            if limit is not None and n >= int(limit):
                break

            # Load the referenced image from the ZIP.
            img_obj = None
            if ex.media_path in available:
                img_bytes = z.read(ex.media_path)
                img_obj = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Per-sample frame directory
            sample_id = f"corecognition_{ex.id}"
            frames_dir = split_dir / sample_id / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

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

            rec = VideoMcpClipRecord(
                dataset="CoreCognition",
                split=split,
                source_id=str(ex.id),
                fps=int(v.fps),
                seconds=float(v.seconds),
                num_frames=int(v.num_frames),
                width=int(v.width),
                height=int(v.height),
                question=ex.question,
                choices=ex.choices,
                answer=ex.answer,
                frames_dir=str(frames_dir.relative_to(split_dir)),
            )
            f.write(rec.model_dump_json() + "\n")
            n += 1

    z.close()
    return n

