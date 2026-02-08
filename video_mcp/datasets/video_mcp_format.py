from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VideoMcpSample(BaseModel):
    """
    Canonical sample format we use across datasets before any video rendering.
    """

    dataset: str = Field(description="Source dataset name, e.g. 'CoreCognition'.")
    split: str = Field(description="Split name, e.g. 'train'.")
    source_id: str = Field(description="Original dataset row id.")

    task: Literal["mcqa_vqa"] = "mcqa_vqa"
    question: str
    choices: dict[str, str]
    answer: str

    # Local asset paths (relative to the dataset root)
    image_path: str = Field(description="Relative path to the image file.")

