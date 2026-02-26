from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Iterator

from video_mcp.mcqa import CHOICE_ORDER, normalize_choice
from video_mcp.process.adapter import DatasetAdapter, McqaVqaSample, register_adapter

MMBENCH_REPO_ID = "HuggingFaceM4/MMBench"


def _build_question_text(question: str, hint: str) -> str:
    """Prepend the hint to the question when present, matching MMBench's
    own prompt construction."""
    q = question.strip()
    h = hint.strip()
    if not h:
        return q
    return f"Hint: {h}\n\n{q}"


def _load_mmbench() -> dict[str, Any]:
    from datasets import load_dataset

    cache_dir = os.environ.get("HF_DATASETS_CACHE") or "hf_home/datasets"
    token = os.environ.get("HF_TOKEN") or None

    local_repo = Path("data/raw/mmbench")
    if local_repo.exists() and any(local_repo.iterdir()):
        return load_dataset(str(local_repo), cache_dir=str(cache_dir), token=token)

    return load_dataset(MMBENCH_REPO_ID, cache_dir=str(cache_dir), token=token)


def _iter_mmbench_mcqa_vqa() -> Iterator[tuple[McqaVqaSample, bytes]]:
    ds_dict = _load_mmbench()

    for split_name, ds in ds_dict.items():
        for idx, row in enumerate(ds):
            question_raw = str(row.get("question", "")).strip()
            if not question_raw:
                continue

            # Build choices — skip rows missing any of A/B/C/D.
            choices: dict[str, str] = {}
            skip = False
            for label in CHOICE_ORDER:
                val = row.get(label)
                if val is None or str(val).strip() == "":
                    skip = True
                    break
                choices[label] = str(val).strip()
            if skip:
                continue

            answer = normalize_choice(str(row.get("answer", "")))
            if answer is None or answer not in choices:
                continue

            # Image is base64-encoded JPEG.
            raw_image = row.get("image")
            if not raw_image or not isinstance(raw_image, str) or not raw_image.strip():
                continue
            image_bytes = base64.b64decode(raw_image)

            source_index = row.get("index")
            if source_index is None:
                source_id = f"{split_name}_{idx:07d}"
            else:
                source_id = f"{split_name}_{source_index}"

            hint = str(row.get("hint") or "")
            question = _build_question_text(question_raw, hint)

            image_filename = f"mmbench_{source_id}.jpg"

            sample = McqaVqaSample(
                dataset="MMBench",
                source_id=source_id,
                question=question,
                choices=choices,
                answer=answer,
                image_filename=image_filename,
            )
            yield sample, image_bytes


@register_adapter("mmbench")
class MMBenchAdapter(DatasetAdapter):
    """Adapter for the MMBench dataset (``HuggingFaceM4/MMBench``).

    MMBench is a multimodal benchmark from OpenCompass with ~11k
    multiple-choice questions spanning 20 ability dimensions. Each
    question includes an image, a question, an optional hint, and
    2-4 answer choices (A/B/C/D). We filter for 4-option questions
    only since the renderer always draws four corner boxes.
    """

    @property
    def name(self) -> str:
        return "mmbench"

    @property
    def generator_id(self) -> str:
        return "M-5"

    @property
    def hf_repo_id(self) -> str:
        return MMBENCH_REPO_ID

    def download(self, *, out_dir: Path) -> Path:
        from huggingface_hub import snapshot_download

        out_dir.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN") or None
        local_path = snapshot_download(
            repo_id=MMBENCH_REPO_ID,
            repo_type="dataset",
            token=token,
            local_dir=str(out_dir),
        )
        return Path(local_path)

    def iter_mcqa_vqa(self) -> Iterator[tuple[McqaVqaSample, bytes]]:
        yield from _iter_mmbench_mcqa_vqa()
