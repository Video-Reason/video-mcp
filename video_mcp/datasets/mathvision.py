from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Iterator

from video_mcp.mcqa import CHOICE_ORDER, Choice, normalize_choice
from video_mcp.process.adapter import DatasetAdapter, McqaVqaSample, register_adapter


# Strip HF-style image placeholders like "<image1>" from question text.
_IMAGE_TAG_RE = re.compile(r"<image\d*>", flags=re.IGNORECASE)


def _clean_question(text: str) -> str:
    """Remove ``<image1>`` tags and collapse whitespace."""
    cleaned = _IMAGE_TAG_RE.sub("", text).strip()
    return " ".join(cleaned.split())


def _load_mathvision(*, split: str = "test"):
    """Load MathVision from Hugging Face (MIT license, no gating)."""
    from datasets import load_dataset

    cache_dir = os.environ.get("HF_DATASETS_CACHE") or "hf_home/datasets"
    token = os.environ.get("HF_TOKEN") or None
    return load_dataset(
        "MathLLMs/MathVision",
        split=split,
        cache_dir=str(cache_dir),
        token=token,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


MATHVISION_REPO_ID = "MathLLMs/MathVision"
MATHVISION_SPLIT = "test"


@register_adapter("mathvision")
class MathVisionAdapter(DatasetAdapter):
    """
    Adapter for the *MathVision* dataset (``MathLLMs/MathVision``).

    Filters for multiple-choice questions whose answer is in A/B/C/D.
    MathVision options are letter labels only — the detailed choice
    content is embedded in the competition image, which is displayed
    alongside the overlay.
    """

    @property
    def name(self) -> str:
        return "mathvision"

    @property
    def generator_id(self) -> str:
        return "M-3"

    @property
    def hf_repo_id(self) -> str | None:
        return MATHVISION_REPO_ID

    @property
    def hf_split(self) -> str | None:
        return MATHVISION_SPLIT

    # ---- download ---------------------------------------------------------

    def download(self, *, out_dir: Path) -> Path:
        """Trigger the HF datasets cache download and write a small manifest."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = _load_mathvision(split="test")
        info_path = out_dir / "info.txt"
        info_path.write_text(
            f"MathVision dataset (MathLLMs/MathVision)\n"
            f"Split: test\n"
            f"Rows: {len(ds)}\n"
            f"Cached via HF datasets library.\n",
            encoding="utf-8",
        )
        return out_dir

    # ---- iter_mcqa_vqa ----------------------------------------------------

    def iter_mcqa_vqa(self) -> Iterator[tuple[McqaVqaSample, bytes]]:
        ds = _load_mathvision(split="test")

        for row in ds:
            # --- filter: only MC questions with options ---
            options: list[str] = row.get("options") or []
            if not options:
                continue

            # --- filter: answer must be A/B/C/D ---
            answer = normalize_choice(str(row.get("answer", "")))
            if answer is None:
                continue

            # --- build choices dict (A‑D only) ---
            choices: dict[str, str] = {}
            for i, label in enumerate(CHOICE_ORDER):
                if i < len(options):
                    choices[label] = options[i]
            if answer not in choices:
                continue

            # --- image bytes from decoded_image (PIL Image) ---
            img = row.get("decoded_image")
            if img is None:
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # --- question text ---
            question = _clean_question(str(row.get("question", "")))
            if not question:
                continue

            # --- image filename ---
            image_col: str = row.get("image") or ""
            image_filename = Path(image_col).name if image_col else f"{row.get('id', 'unknown')}.png"

            sample = McqaVqaSample(
                dataset="MathVision",
                source_id=str(row.get("id", "")),
                question=question,
                choices=choices,
                answer=answer,
                image_filename=image_filename,
            )
            yield sample, image_bytes
