from __future__ import annotations

import argparse
import random
from pathlib import Path

from pydantic import BaseModel

from video_mcp.render import draw_mcqa_overlay, make_fonts, render_horizon_frame
from video_mcp.schemas import Choice, DatasetConfig, OverlayMode, SampleRecord


class _CliArgs(BaseModel):
    out_dir: Path | None = None
    split: str | None = None
    num_samples: int | None = None
    seed: int | None = None
    overlay_mode: OverlayMode | None = None
    fps: int | None = None
    seconds: float | None = None
    width: int | None = None
    height: int | None = None
    highlight_start: int | None = None
    highlight_end: int | None = None


def _build_config(cli: _CliArgs) -> DatasetConfig:
    cfg = DatasetConfig()
    if cli.out_dir is not None:
        cfg.out_dir = cli.out_dir
    if cli.split is not None:
        cfg.split_name = cli.split
    if cli.num_samples is not None:
        cfg.num_samples = cli.num_samples
    if cli.seed is not None:
        cfg.seed = cli.seed
    if cli.overlay_mode is not None:
        cfg.overlay_mode = cli.overlay_mode
    if cli.fps is not None:
        cfg.video.fps = cli.fps
    if cli.seconds is not None:
        cfg.video.seconds = cli.seconds
    if cli.width is not None:
        cfg.video.width = cli.width
    if cli.height is not None:
        cfg.video.height = cli.height
    if cli.highlight_start is not None:
        cfg.highlight.start_frame = cli.highlight_start
    if cli.highlight_end is not None:
        cfg.highlight.end_frame = cli.highlight_end
    return cfg


def _sun_question_and_answer(sun_choice: Choice) -> tuple[str, list[str], Choice]:
    # A=left, B=right, C=down, D=up (matches renderer mapping)
    choices = [
        "A: left",
        "B: right",
        "C: down",
        "D: up",
    ]
    question = "Where is the sun in the scene? Choose A/B/C/D."
    return question, choices, sun_choice


def _iter_samples(cfg: DatasetConfig) -> list[tuple[str, Choice]]:
    rng = random.Random(int(cfg.seed))
    ids: list[tuple[str, Choice]] = []
    for i in range(int(cfg.num_samples)):
        sample_id = f"{i:06d}"
        sun_choice: Choice = rng.choice(["A", "B", "C", "D"])
        ids.append((sample_id, sun_choice))
    return ids


def generate_dataset(cfg: DatasetConfig) -> None:
    out_split = cfg.out_dir / cfg.split_name
    out_split.mkdir(parents=True, exist_ok=True)

    metadata_path = out_split / "metadata.jsonl"
    fonts = make_fonts()

    num_frames = cfg.video.num_frames
    highlight_end = (num_frames - 1) if cfg.highlight.end_frame is None else int(cfg.highlight.end_frame)
    highlight_start = int(cfg.highlight.start_frame)

    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for sample_id, sun_choice in _iter_samples(cfg):
            sample_dir = out_split / sample_id
            frames_dir = sample_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            question, choices, answer = _sun_question_and_answer(sun_choice)

            for frame_idx in range(num_frames):
                base = render_horizon_frame(
                    width=int(cfg.video.width),
                    height=int(cfg.video.height),
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    sun_choice=sun_choice,
                )

                show_panel = (cfg.overlay_mode == OverlayMode.all_frames) or (frame_idx == 0)
                lit_choice: Choice | None = None
                if highlight_start <= frame_idx <= highlight_end:
                    lit_choice = answer

                composed = draw_mcqa_overlay(
                    base,
                    question=question,
                    choices=choices,
                    lit_choice=lit_choice,
                    show_panel=show_panel,
                    fonts=fonts,
                )

                frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
                composed.save(frame_path, format="PNG")

            rec = SampleRecord(
                sample_id=sample_id,
                split=cfg.split_name,
                width=int(cfg.video.width),
                height=int(cfg.video.height),
                fps=int(cfg.video.fps),
                seconds=float(cfg.video.seconds),
                num_frames=int(num_frames),
                question=question,
                choices=choices,
                answer=answer,
                frames_dir=str(frames_dir.relative_to(out_split)),
            )
            meta_f.write(rec.model_dump_json() + "\n")


def _parse_args(argv: list[str] | None = None) -> _CliArgs:
    p = argparse.ArgumentParser(description="Generate MCQA-video dataset (frames + metadata).")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overlay-mode", type=str, choices=[m.value for m in OverlayMode], default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--highlight-start", type=int, default=None)
    p.add_argument("--highlight-end", type=int, default=None)

    ns = p.parse_args(argv)
    overlay_mode = OverlayMode(ns.overlay_mode) if ns.overlay_mode is not None else None
    return _CliArgs(
        out_dir=ns.out_dir,
        split=ns.split,
        num_samples=ns.num_samples,
        seed=ns.seed,
        overlay_mode=overlay_mode,
        fps=ns.fps,
        seconds=ns.seconds,
        width=ns.width,
        height=ns.height,
        highlight_start=ns.highlight_start,
        highlight_end=ns.highlight_end,
    )


def main(argv: list[str] | None = None) -> None:
    cli = _parse_args(argv)
    cfg = _build_config(cli)
    generate_dataset(cfg)


if __name__ == "__main__":
    main()

