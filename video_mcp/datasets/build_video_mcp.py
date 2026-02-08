from __future__ import annotations

from pathlib import Path

from video_mcp.datasets.corecognition import download_corecognition_complete_zip, iter_corecognition_mcqa_single_image
from video_mcp.datasets.video_mcp_format import VideoMcpSample


def build_video_mcp_from_corecognition_mcqa_single_image(
    *,
    out_dir: Path,
    split: str = "train",
    config: str = "complete",
) -> int:
    """
    Build a standardized Video-MCP dataset from CoreCognition:

    out_dir/
      train/
        metadata.jsonl
        images/
          <source_id>__<original_filename>.png
    """
    out_dir = Path(out_dir)
    split_dir = out_dir / split
    images_dir = split_dir / "images"
    split_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # We export real images only when using the complete ZIP.
    zip_path = download_corecognition_complete_zip() if config == "complete" else None
    z = None
    if zip_path is not None:
        import zipfile

        z = zipfile.ZipFile(zip_path)
        available = set(z.namelist())
    else:
        available = set()

    metadata_path = split_dir / "metadata.jsonl"

    n = 0
    with metadata_path.open("w", encoding="utf-8") as f:
        for ex in iter_corecognition_mcqa_single_image(split=split, config=config):
            # Write/copy image asset (complete only).
            image_rel = ""
            if z is not None and ex.media_path in available:
                out_name = f"{ex.id}__{Path(ex.image).name}"
                out_file = images_dir / out_name
                if not out_file.exists():
                    out_file.write_bytes(z.read(ex.media_path))
                image_rel = str(out_file.relative_to(split_dir))

            sample = VideoMcpSample(
                dataset="CoreCognition",
                split=split,
                source_id=str(ex.id),
                question=ex.question,
                choices=ex.choices,
                answer=ex.answer,
                image_path=image_rel,
            )
            f.write(sample.model_dump_json() + "\n")
            n += 1

    if z is not None:
        z.close()

    return n

