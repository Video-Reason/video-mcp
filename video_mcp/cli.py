from __future__ import annotations

import argparse
from pathlib import Path

from video_mcp.process.adapter import get_adapter, list_adapters
from video_mcp.process.build_video_mcp import build_video_mcp
from video_mcp.process.build_video_mcp_clips import build_video_mcp_clips
from video_mcp.env import ensure_hf_cache_dirs, load_env_file


def main(argv: list[str] | None = None) -> None:
    load_env_file(".env")
    ensure_hf_cache_dirs()

    available = list_adapters()

    p = argparse.ArgumentParser(prog="video_mcp", description="Video-MCP dataset processing utilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- download --------------------------------------------------------
    dl = sub.add_parser("download", help="Download raw dataset files.")
    dl.add_argument("--dataset", type=str, required=True, choices=available)
    dl.add_argument("--out-dir", type=Path, default=Path("data/raw"))

    # ---- build (metadata.jsonl + images) ---------------------------------
    build = sub.add_parser(
        "build",
        help="Build Video-MCP dataset (metadata.jsonl + images).",
    )
    build.add_argument("--dataset", type=str, required=True, choices=available)
    build.add_argument("--out-dir", type=Path, default=Path("data/video_mcp"))
    build.add_argument("--split", type=str, default="train")

    # ---- build-clips (rendered MCQA overlay frames) ----------------------
    clips = sub.add_parser(
        "build-clips",
        help="Build Video-MCP clip frames (rendered MCQA overlays).",
    )
    clips.add_argument("--dataset", type=str, required=True, choices=available)
    clips.add_argument("--out-dir", type=Path, default=Path("data/video_mcp_clips"))
    clips.add_argument("--split", type=str, default="train")
    clips.add_argument("--limit", type=int, default=None, help="Only build first N samples (for testing).")

    args = p.parse_args(argv)

    if args.cmd == "download":
        adapter = get_adapter(args.dataset)
        out = adapter.download(out_dir=Path(args.out_dir) / args.dataset)
        print(f"Raw {args.dataset} data available at {out}")

    elif args.cmd == "build":
        adapter = get_adapter(args.dataset)
        out_root = Path(args.out_dir) / args.dataset
        n = build_video_mcp(adapter, out_dir=out_root, split=args.split)
        print(f"Wrote {n} samples to {out_root / args.split}/metadata.jsonl")

    elif args.cmd == "build-clips":
        adapter = get_adapter(args.dataset)
        out_root = Path(args.out_dir) / args.dataset
        n = build_video_mcp_clips(adapter, out_dir=out_root, split=args.split, limit=args.limit)
        print(f"Wrote {n} clip samples to {out_root}")
