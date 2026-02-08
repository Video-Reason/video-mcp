from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic.types import PositiveFloat, PositiveInt


class VideoSpec(BaseModel):
    fps: PositiveInt = Field(default=16, description="Frames per second.")
    seconds: PositiveFloat = Field(default=3.0, description="Clip duration.")
    width: PositiveInt = Field(default=1024, description="Frame width in pixels.")
    height: PositiveInt = Field(default=768, description="Frame height in pixels.")

    @property
    def num_frames(self) -> int:
        return int(round(float(self.fps) * float(self.seconds)))

