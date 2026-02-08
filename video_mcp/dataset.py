from __future__ import annotations

import argparse
from pathlib import Path

from video_mcp.process.adapter import get_adapter, list_adapters
from video_mcp.process.build_video_mcp_clips import build_video_mcp_clips
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

    args = p.parse_args(argv)

    if args.cmd == "download":
        adapter = get_adapter(args.dataset)
        out = adapter.download(out_dir=Path(args.out_dir) / args.dataset)
        print(f"Raw {args.dataset} data available at {out}")

    elif args.cmd == "process":
        adapter = get_adapter(args.dataset)
        out_root = Path(args.out_dir) / f"{args.dataset}_video_mcp"
        n = build_video_mcp_clips(adapter, out_dir=out_root, limit=args.limit)
        print(f"Wrote {n} Video-MCP samples to {out_root}")


if __name__ == "__main__":
    main()
