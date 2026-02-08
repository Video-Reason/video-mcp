from __future__ import annotations

from pathlib import Path

from video_mcp.process.adapter import DatasetAdapter
from video_mcp.process.video_mcp_format import VideoMcpSample


def build_video_mcp(
    adapter: DatasetAdapter,
    *,
    out_dir: Path,
    split: str = "train",
) -> int:
    """
    Build a standardised Video-MCP dataset from **any** adapter.

    out_dir/
      <split>/
        metadata.jsonl
        images/
          <source_id>__<original_filename>
    """
    out_dir = Path(out_dir)
    split_dir = out_dir / split
    images_dir = split_dir / "images"
    split_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = split_dir / "metadata.jsonl"

    n = 0
    with metadata_path.open("w", encoding="utf-8") as f:
        for sample, image_bytes in adapter.iter_mcqa_vqa(split=split):
            n += 1
            print(f"[{n}] {adapter.name}_{sample.source_id}")

            out_name = f"{sample.source_id}__{sample.image_filename}"
            out_file = images_dir / out_name
            if not out_file.exists():
                out_file.write_bytes(image_bytes)
            image_rel = str(out_file.relative_to(split_dir))

            vmc = VideoMcpSample(
                dataset=sample.dataset,
                split=split,
                source_id=sample.source_id,
                question=sample.question,
                choices=sample.choices,
                answer=sample.answer,
                image_path=image_rel,
            )
            f.write(vmc.model_dump_json() + "\n")
            n += 1

    return n
