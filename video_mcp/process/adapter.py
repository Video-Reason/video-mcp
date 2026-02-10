from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

from video_mcp.mcqa import Choice


# ---------------------------------------------------------------------------
# Universal intermediate sample every adapter must yield
# ---------------------------------------------------------------------------


class McqaVqaSample(BaseModel):
    """
    Dataset-agnostic MCQA-VQA sample produced by every adapter.

    Paired with raw ``image_bytes`` when yielded by
    :py:meth:`DatasetAdapter.iter_mcqa_vqa`.
    """

    dataset: str = Field(description="Source dataset name, e.g. 'CoreCognition'.")
    source_id: str = Field(description="Row / sample id from the source dataset.")
    question: str
    choices: dict[str, str]
    answer: Choice
    image_filename: str = Field(description="Original image filename (basename only).")


# ---------------------------------------------------------------------------
# Abstract adapter that every dataset module must subclass
# ---------------------------------------------------------------------------


class DatasetAdapter(ABC):
    """
    Base class for Video-MCP dataset adapters.

    To add a new dataset:

    1. Create ``video_mcp/datasets/<name>.py``
    2. Subclass :class:`DatasetAdapter`
    3. Decorate with ``@register_adapter("<name>")``
    4. Add ``from video_mcp.datasets import <name> as _<name>``
       to ``video_mcp/datasets/__init__.py`` so the decorator runs at import.

    VBVR naming convention (all three levels share the same *name*)::

        {generator_id}_{name}_data-generator/
        └── {name}_task/
            └── {name}_0000/
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short slug used in CLI ``--dataset`` flags, e.g. ``'corecognition'``."""
        ...

    @property
    @abstractmethod
    def generator_id(self) -> str:
        """VBVR-style prefix, e.g. ``'M-1'``, ``'M-2'``."""
        ...

    @property
    def generator_name(self) -> str:
        """Full VBVR generator directory name, e.g. ``'M-1_corecognition_data-generator'``."""
        return f"{self.generator_id}_{self.name}_data-generator"

    @abstractmethod
    def download(self, *, out_dir: Path) -> Path:
        """Download the raw dataset artifact and return its local path."""
        ...

    @abstractmethod
    def iter_mcqa_vqa(self) -> Iterator[tuple[McqaVqaSample, bytes]]:
        """
        Yield ``(sample, image_bytes)`` pairs.

        * ``sample`` — question, choices, answer, and provenance metadata.
        * ``image_bytes`` — raw file contents of the image (PNG / JPEG).
        """
        ...


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_adapter(name: str):
    """Class decorator that registers a :class:`DatasetAdapter` subclass."""

    def _decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        ADAPTER_REGISTRY[name] = cls
        return cls

    return _decorator


def get_adapter(name: str) -> DatasetAdapter:
    """Instantiate a registered adapter by its slug name."""
    cls = ADAPTER_REGISTRY[name]
    return cls()


def list_adapters() -> list[str]:
    """Return a sorted list of registered adapter names."""
    return sorted(ADAPTER_REGISTRY)
