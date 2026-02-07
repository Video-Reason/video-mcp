from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic.types import PositiveFloat, PositiveInt


Choice = Literal["A", "B", "C", "D"]


class OverlayMode(str, Enum):
    first_frame_only = "first_frame_only"
    all_frames = "all_frames"


class VideoSpec(BaseModel):
    fps: PositiveInt = Field(default=16, description="Frames per second.")
    seconds: PositiveFloat = Field(default=3.0, description="Clip duration.")
    width: PositiveInt = Field(default=1024, description="Frame width in pixels.")
    height: PositiveInt = Field(default=768, description="Frame height in pixels.")

    @property
    def num_frames(self) -> int:
        return int(round(float(self.fps) * float(self.seconds)))


class HighlightSpec(BaseModel):
    start_frame: int = Field(default=1, ge=0, description="First frame index to show highlight.")
    end_frame: int | None = Field(
        default=None, description="Last frame index (inclusive). None means last frame."
    )


class DatasetConfig(BaseModel):
    out_dir: Path = Field(default=Path("data/generated"), description="Output dataset root directory.")
    split_name: str = Field(default="train", description="Split subdirectory name.")
    num_samples: PositiveInt = Field(default=100, description="How many samples to generate.")
    seed: int = Field(default=0, description="RNG seed for deterministic generation.")

    video: VideoSpec = Field(default_factory=VideoSpec)
    overlay_mode: OverlayMode = Field(default=OverlayMode.all_frames)
    highlight: HighlightSpec = Field(default_factory=HighlightSpec)


class SampleRecord(BaseModel):
    sample_id: str
    split: str

    width: int
    height: int
    fps: int
    seconds: float
    num_frames: int

    question: str
    choices: list[str]
    answer: Choice

    frames_dir: str

