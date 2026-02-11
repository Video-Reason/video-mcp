from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Iterator

from video_mcp.mcqa import CHOICE_ORDER, Choice, normalize_choice
from video_mcp.process.adapter import DatasetAdapter, McqaVqaSample, register_adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Options in PhyX look like: "A: 28.5°", "B: 30.4°", etc.
# Strip the leading letter + colon to keep just the value text.
_OPTION_PREFIX_RE = re.compile(r"^[A-Da-d]\s*:\s*")


def _strip_option_prefix(text: str) -> str:
    """Remove leading ``A: `` / ``B: `` etc. from an option string."""
    return _OPTION_PREFIX_RE.sub("", text).strip()


def _build_question_text(
    question: str,
    description_simplified: str,
) -> str:
    """
    Combine the short *question* with *description_simplified* to form a
    self-contained question string.  PhyX ``question`` is often very terse
    (e.g. "Determine the angle θ'"), while the simplified description gives
    the essential physical context.
    """
    q = question.strip()
    desc = description_simplified.strip()
    if not desc:
        return q
    if not q:
        return desc
    # Avoid duplication when the question is already included in the description.
    if q in desc:
        return desc
    return f"{desc} {q}"


# ---------------------------------------------------------------------------
# HF loader
# ---------------------------------------------------------------------------

PHYX_REPO_ID = "Cloudriver/PhyX"
PHYX_CONFIG = "default"
PHYX_SPLIT = "test_mini"


def _load_phyx(*, split: str = PHYX_SPLIT):
    """Load the PhyX dataset from Hugging Face."""
    from datasets import load_dataset

    cache_dir = os.environ.get("HF_DATASETS_CACHE") or "hf_home/datasets"
    token = os.environ.get("HF_TOKEN") or None
    return load_dataset(
        PHYX_REPO_ID,
        split=split,
        cache_dir=str(cache_dir),
        token=token,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter("phyx")
class PhyXAdapter(DatasetAdapter):
    """
    Adapter for the *PhyX* dataset (``Cloudriver/PhyX``).

    PhyX is a physics-reasoning MCQA benchmark with 3 000 visually grounded
    questions spanning 6 physics domains: mechanics, electromagnetism,
    thermodynamics, wave/acoustics, optics, and modern physics.

    Uses the **test_mini** split (1 000 questions) by default.
    """

    @property
    def name(self) -> str:
        return "phyx"

    @property
    def generator_id(self) -> str:
        return "M-4"

    @property
    def hf_repo_id(self) -> str | None:
        return PHYX_REPO_ID

    @property
    def hf_config(self) -> str | None:
        return PHYX_CONFIG

    @property
    def hf_split(self) -> str | None:
        return PHYX_SPLIT

    # ---- download ---------------------------------------------------------

    def download(self, *, out_dir: Path) -> Path:
        """Download the PhyX dataset via HF datasets and write a summary."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = _load_phyx(split=PHYX_SPLIT)
        info_path = out_dir / "info.txt"
        info_path.write_text(
            f"PhyX dataset ({PHYX_REPO_ID})\n"
            f"Split: {PHYX_SPLIT}\n"
            f"Rows: {len(ds)}\n"
            f"Cached via HF datasets library.\n",
            encoding="utf-8",
        )
        return out_dir

    # ---- iter_mcqa_vqa ----------------------------------------------------

    def iter_mcqa_vqa(self) -> Iterator[tuple[McqaVqaSample, bytes]]:
        ds = _load_phyx(split=PHYX_SPLIT)

        for row in ds:
            # --- answer must be A/B/C/D ---
            answer = normalize_choice(str(row.get("answer", "")))
            if answer is None:
                continue

            # --- options list (always 4 items) ---
            options: list[str] = row.get("options") or []
            if len(options) != 4:
                continue

            choices: dict[str, str] = {}
            for i, label in enumerate(CHOICE_ORDER):
                choices[label] = _strip_option_prefix(options[i])

            if answer not in choices:
                continue

            # --- image (PIL Image) ---
            img = row.get("image")
            if img is None:
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # --- question text ---
            question = _build_question_text(
                str(row.get("question", "")),
                str(row.get("question_description_simplified", "")),
            )
            if not question:
                continue

            # --- source id ---
            source_id = str(row.get("id", ""))
            if not source_id:
                continue

            image_filename = f"phyx_{source_id}.png"

            sample = McqaVqaSample(
                dataset="PhyX",
                source_id=source_id,
                question=question,
                choices=choices,
                answer=answer,
                image_filename=image_filename,
            )
            yield sample, image_bytes
