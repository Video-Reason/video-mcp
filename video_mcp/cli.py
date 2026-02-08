from __future__ import annotations

import argparse
from pathlib import Path

from video_mcp.datasets.corecognition import extract_corecognition_mcqa_single_image
from video_mcp.datasets.build_video_mcp_clips import build_video_mcp_clips_corecognition_mcqa_single_image
from video_mcp.datasets.build_video_mcp import build_video_mcp_from_corecognition_mcqa_single_image
from video_mcp.env import ensure_hf_cache_dirs, load_env_file


def main(argv: list[str] | None = None) -> None:
    load_env_file(".env")
    ensure_hf_cache_dirs()

    p = argparse.ArgumentParser(prog="video_mcp", description="Video-MCP dataset processing utilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    cc = sub.add_parser(
        "extract-corecognition",
        help="Extract MCQA single-image VQA from CoreCognition (Hugging Face).",
    )
    cc.add_argument("--out", type=Path, default=Path("data/corecognition/mcqa_single_image.jsonl"))
    cc.add_argument("--split", type=str, default="train")
    cc.add_argument(
        "--config",
        type=str,
        default="default",
        help="HF dataset config: use 'default' for metadata preview, or 'complete' for full ZIP.",
    )
    cc.add_argument(
        "--export-media-dir",
        type=Path,
        default=None,
        help="If set (and --config complete), export referenced media files into this directory.",
    )

    bcc = sub.add_parser(
        "build-video-mcp-corecognition",
        help="Build Video-MCP dataset from CoreCognition MCQA single-image VQA.",
    )
    bcc.add_argument("--out-dir", type=Path, default=Path("data/video_mcp/corecognition_mcqa_single_image"))
    bcc.add_argument("--split", type=str, default="train")
    bcc.add_argument("--config", type=str, default="complete")

    bclips = sub.add_parser(
        "build-video-mcp-clips-corecognition",
        help="Build Video-MCP clip frames (frame0 prompt, later highlight) from CoreCognition single-image MCQA.",
    )
    bclips.add_argument("--out-dir", type=Path, default=Path("data/video_mcp_clips/corecognition_mcqa_single_image"))
    bclips.add_argument("--split", type=str, default="train")
    bclips.add_argument("--limit", type=int, default=None, help="Optional: only build first N samples (for testing).")

    args = p.parse_args(argv)
    if args.cmd == "extract-corecognition":
        extract_corecognition_mcqa_single_image(
            out_path=args.out,
            split=args.split,
            config=args.config,
            export_media_dir=args.export_media_dir,
        )
    if args.cmd == "build-video-mcp-corecognition":
        n = build_video_mcp_from_corecognition_mcqa_single_image(
            out_dir=args.out_dir,
            split=args.split,
            config=args.config,
        )
        print(f"Wrote {n} samples to {args.out_dir}/{args.split}/metadata.jsonl")

    if args.cmd == "build-video-mcp-clips-corecognition":
        n = build_video_mcp_clips_corecognition_mcqa_single_image(
            out_dir=args.out_dir,
            split=args.split,
            limit=args.limit,
        )
        print(f"Wrote {n} clip samples to {args.out_dir}/{args.split}/metadata.jsonl")

