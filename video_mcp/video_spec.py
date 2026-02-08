from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from pydantic.types import PositiveInt


# ── Wan2.2 VAE temporal compression grid: valid frame counts = 1 + 4k ──
WAN_TEMPORAL_GRID = tuple(1 + 4 * k for k in range(21))  # 1, 5, 9, … 81


def snap_to_temporal_grid(n: int) -> int:
    """Return the closest integer satisfying ``1 + 4k`` (≥ 1)."""
    k = round((n - 1) / 4)
    return max(1, 1 + 4 * max(0, k))


class VideoSpec(BaseModel):
    """Video specification aligned with **Wan2.2-I2V-A14B** fine-tuning.

    Defaults: 480p (832×480), 81 frames @ 16 FPS (~5 s).
    """

    fps: PositiveInt = Field(
        default=16,
        description="Frames per second (Wan2.2 native: 16).",
    )
    num_frames: PositiveInt = Field(
        default=81,
        description=(
            "Frames per clip. Must satisfy 1+4k "
            "(VAE temporal compression). Common: 49 (lighter) or 81 (default)."
        ),
    )
    width: PositiveInt = Field(
        default=832,
        description="Frame width in pixels (must be divisible by 8). 480p→832, 720p→1280.",
    )
    height: PositiveInt = Field(
        default=480,
        description="Frame height in pixels (must be divisible by 8). 480p→480, 720p→720.",
    )

    # ── derived ──────────────────────────────────────────────────────────

    @property
    def seconds(self) -> float:
        """Clip duration derived from *num_frames / fps*."""
        return self.num_frames / self.fps

    # ── validators ───────────────────────────────────────────────────────

    @field_validator("width", "height")
    @classmethod
    def _divisible_by_8(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError(
                f"{v} is not divisible by 8 (VAE spatial compression factor)"
            )
        return v

    @field_validator("num_frames")
    @classmethod
    def _valid_temporal_grid(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            nearest = snap_to_temporal_grid(v)
            raise ValueError(
                f"num_frames={v} does not satisfy 1+4k; "
                f"nearest valid value: {nearest}"
            )
        return v
