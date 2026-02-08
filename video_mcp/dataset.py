from __future__ import annotations

import argparse
from pathlib import Path

from video_mcp.mcqa import LIT_STYLES, LitStyle
from video_mcp.process.adapter import get_adapter, list_adapters
from video_mcp.process.build_video_mcp_clips import build_video_mcp_clips
from video_mcp.video_spec import VideoSpec
from video_mcp.env import load_env_file


def main(argv: list[str] | None = None) -> None:
    load_env_file(".env")

    available = list_adapters()

    p = argparse.ArgumentParser(
        prog="python -m video_mcp.dataset",
        description="Dataset download + processing into Video-MCP format.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download raw datasets into data/raw/...")
    dl.add_argument("--dataset", type=str, required=True, choices=available)
    dl.add_argument("--out-dir", type=Path, default=Path("data/raw"))

    proc = sub.add_parser("process", help="Process raw datasets into data/processed/...")
    proc.add_argument("--dataset", type=str, required=True, choices=available)
    proc.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    proc.add_argument("--limit", type=int, default=None)
    proc.add_argument(
        "--lit-style",
        type=str,
        choices=list(LIT_STYLES),
        default="darken",
        help="How the correct answer is highlighted: 'darken' darkens the box, 'red_border' draws a red outline.",
    )
    proc.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width in px (must be divisible by 8). Default: 832 (480p).",
    )
    proc.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height in px (must be divisible by 8). Default: 480 (480p).",
    )
    proc.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frames per clip (must satisfy 1+4k). Default: 81 (~5 s @ 16 FPS).",
    )

    args = p.parse_args(argv)

    if args.cmd == "download":
        adapter = get_adapter(args.dataset)
        out = adapter.download(out_dir=Path(args.out_dir) / args.dataset)
        print(f"Raw {args.dataset} data available at {out}")

    elif args.cmd == "process":
        adapter = get_adapter(args.dataset)
        lit_style: LitStyle = args.lit_style

        spec_kw: dict[str, int] = {}
        if args.width is not None:
            spec_kw["width"] = args.width
        if args.height is not None:
            spec_kw["height"] = args.height
        if args.num_frames is not None:
            spec_kw["num_frames"] = args.num_frames
        video_spec = VideoSpec(**spec_kw)

        out_root = Path(args.out_dir) / f"{args.dataset}_video_mcp"
        n = build_video_mcp_clips(
            adapter,
            out_dir=out_root,
            limit=args.limit,
            lit_style=lit_style,
            video=video_spec,
        )
        print(
            f"Wrote {n} Video-MCP samples to {out_root} "
            f"({video_spec.width}x{video_spec.height}, "
            f"{video_spec.num_frames} frames @ {video_spec.fps} FPS)"
        )


if __name__ == "__main__":
    main()
