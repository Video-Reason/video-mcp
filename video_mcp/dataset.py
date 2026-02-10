from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video_mcp.mcqa import LIT_STYLES, LitStyle
from video_mcp.process.adapter import get_adapter, list_adapters
from video_mcp.process.build_video_mcp_clips import build_video_mcp_clips
from video_mcp.video_spec import VideoSpec
from video_mcp.env import load_env_file


def _run_cmd_output(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def _fmt_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_run_manifest(
    *,
    adapter_name: str,
    adapter_generator: str,
    out_root: Path,
    process_args: dict[str, Any],
    sample_count: int,
    run_started: datetime,
    run_finished: datetime,
    command_argv: list[str],
) -> tuple[Path, Path]:
    generator_dir = out_root / adapter_generator
    generator_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task_dir = generator_dir / f"{adapter_name}_task"
    repo_root = Path(__file__).resolve().parent.parent

    git_commit = _run_cmd_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_branch = _run_cmd_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    ffmpeg_line = _run_cmd_output(["ffmpeg", "-version"])
    ffmpeg_version = ffmpeg_line.splitlines()[0] if ffmpeg_line else None

    manifest = {
        "manifest_version": 1,
        "run": {
            "id": run_id,
            "started_at_utc": _fmt_utc(run_started),
            "finished_at_utc": _fmt_utc(run_finished),
            "command": "python -m video_mcp.dataset process",
            "argv": command_argv,
        },
        "dataset": {
            "name": process_args["dataset_name"],
            "hf_repo_id": process_args["hf_repo_id"],
            "hf_config": process_args["hf_config"],
            "hf_split": process_args["hf_split"],
            "hf_revision": process_args["hf_revision"],
        },
        "processing": {
            "lit_style": process_args["lit_style"],
            "limit": process_args["limit"],
            "fps": process_args["fps"],
            "seconds": process_args["seconds"],
            "num_frames": process_args["num_frames"],
            "width": process_args["width"],
            "height": process_args["height"],
        },
        "code": {
            "git_commit": git_commit,
            "git_branch": git_branch,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "ffmpeg_version": ffmpeg_version,
        },
        "output": {
            "out_dir": str(out_root.resolve()),
            "generator_dir": str(generator_dir.resolve()),
            "task_dir": str(task_dir.resolve()),
        },
        "result": {
            "num_samples_written": sample_count,
        },
    }

    latest_path = generator_dir / "run_manifest.json"
    latest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    history_dir = generator_dir / "run_manifests"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{run_id}.json"
    history_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return latest_path, history_path


def main(argv: list[str] | None = None) -> None:
    load_env_file(".env")
    command_argv = list(argv) if argv is not None else list(sys.argv[1:])

    available = list_adapters()

    p = argparse.ArgumentParser(
        prog="python -m video_mcp.dataset",
        description="Dataset download + processing into Video-MCP format.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download raw datasets into data/raw/...")
    dl.add_argument("--dataset", type=str, required=True, choices=available)
    dl.add_argument("--out-dir", type=Path, default=Path("data/raw"))

    proc = sub.add_parser("process", help="Process raw datasets into questions/...")
    proc.add_argument("--dataset", type=str, required=True, choices=available)
    proc.add_argument("--out-dir", type=Path, default=Path("questions"))
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
        run_started = datetime.now(timezone.utc)

        spec_kw: dict[str, int] = {}
        if args.width is not None:
            spec_kw["width"] = args.width
        if args.height is not None:
            spec_kw["height"] = args.height
        if args.num_frames is not None:
            spec_kw["num_frames"] = args.num_frames
        video_spec = VideoSpec(**spec_kw)

        # out_dir is the VBVR root (e.g. questions/); the builder creates
        # {adapter}_data-generator/{task}_task/{task}_{NNNN}/ inside it.
        out_root = Path(args.out_dir)
        n = build_video_mcp_clips(
            adapter,
            out_dir=out_root,
            limit=args.limit,
            lit_style=lit_style,
            video=video_spec,
        )
        run_finished = datetime.now(timezone.utc)
        generator_dir = out_root / adapter.generator_name
        manifest_path, history_path = _write_run_manifest(
            adapter_name=adapter.name,
            adapter_generator=adapter.generator_name,
            out_root=out_root,
            process_args={
                "dataset_name": adapter.name,
                "hf_repo_id": adapter.hf_repo_id,
                "hf_config": adapter.hf_config,
                "hf_split": adapter.hf_split,
                "hf_revision": adapter.hf_revision,
                "lit_style": lit_style,
                "limit": args.limit,
                "fps": int(video_spec.fps),
                "seconds": float(video_spec.seconds),
                "num_frames": int(video_spec.num_frames),
                "width": int(video_spec.width),
                "height": int(video_spec.height),
            },
            sample_count=n,
            run_started=run_started,
            run_finished=run_finished,
            command_argv=command_argv,
        )
        print(
            f"Wrote {n} VBVR-format samples to {generator_dir} "
            f"({video_spec.width}x{video_spec.height}, "
            f"{video_spec.num_frames} frames @ {video_spec.fps} FPS)"
        )
        print(f"Run manifest: {manifest_path}")
        print(f"Run manifest history: {history_path}")


if __name__ == "__main__":
    main()
