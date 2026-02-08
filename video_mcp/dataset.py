from __future__ import annotations

import argparse
from pathlib import Path

from video_mcp.datasets.build_video_mcp_clips import build_video_mcp_clips_corecognition_mcqa_single_image
from video_mcp.datasets.corecognition import download_corecognition_complete_zip
from video_mcp.env import ensure_hf_cache_dirs, load_env_file


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _materialize_zip_to_raw(*, zip_path: Path, raw_dir: Path) -> Path:
    _ensure_dir(raw_dir)
    out = raw_dir / zip_path.name
    if out.exists():
        return out
    # Prefer symlink to avoid duplicating 6.4GB.
    out.symlink_to(zip_path)
    return out


def main(argv: list[str] | None = None) -> None:
    load_env_file(".env")
    ensure_hf_cache_dirs()

    p = argparse.ArgumentParser(
        prog="python -m video_mcp.dataset",
        description="Dataset download + processing into Video-MCP format.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download raw datasets into data/raw/...")
    dl.add_argument("--dataset", type=str, required=True, choices=["corecognition"])
    dl.add_argument("--out-dir", type=Path, default=Path("data/raw"))

    proc = sub.add_parser("process", help="Process raw datasets into data/processed/...")
    proc.add_argument("--dataset", type=str, required=True, choices=["corecognition"])
    proc.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    proc.add_argument("--limit", type=int, default=None)

    args = p.parse_args(argv)

    if args.cmd == "download" and args.dataset == "corecognition":
        zip_path = download_corecognition_complete_zip()
        raw_dir = Path(args.out_dir) / "corecognition"
        out = _materialize_zip_to_raw(zip_path=zip_path, raw_dir=raw_dir)
        print(f"Raw CoreCognition ZIP available at {out}")

    if args.cmd == "process" and args.dataset == "corecognition":
        out_root = Path(args.out_dir) / "corecognition_video_mcp"
        n = build_video_mcp_clips_corecognition_mcqa_single_image(
            out_dir=out_root,
            limit=args.limit,
        )
        print(f"Wrote {n} Video-MCP samples to {out_root}")


if __name__ == "__main__":
    main()

