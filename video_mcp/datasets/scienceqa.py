from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image

from video_mcp.mcqa import CHOICE_ORDER, Choice, normalize_choice
from video_mcp.process.adapter import DatasetAdapter, McqaVqaSample, register_adapter


SCIENCEQA_REPO_ID = "derek-thomas/ScienceQA"
SCIENCEQA_CONFIG = "default"
SCIENCEQA_SPLIT = "all"
SCIENCEQA_REVISION = "main"


def _choice_from_index(index: int) -> Choice | None:
    if index < 0 or index >= len(CHOICE_ORDER):
        return None
    return CHOICE_ORDER[index]


def _normalize_answer(raw: Any) -> Choice | None:
    direct = normalize_choice(str(raw))
    if direct is not None:
        return direct

    if isinstance(raw, int):
        return _choice_from_index(raw)

    txt = str(raw).strip()
    if not txt.isdigit():
        return None
    return _choice_from_index(int(txt))


def _normalize_choices(raw: Any) -> dict[str, str]:
    if not isinstance(raw, (list, tuple)):
        return {}

    out: dict[str, str] = {}
    for idx, value in enumerate(raw[: len(CHOICE_ORDER)]):
        txt = str(value).strip()
        if not txt:
            continue
        out[CHOICE_ORDER[idx]] = txt
    return out


def _extract_image_bytes_and_filename(image_obj: Any, source_id: str) -> tuple[bytes, str] | None:
    if image_obj is None:
        return None

    if isinstance(image_obj, dict):
        raw_bytes = image_obj.get("bytes")
        raw_path = image_obj.get("path")

        if isinstance(raw_bytes, (bytes, bytearray)) and len(raw_bytes) > 0:
            if isinstance(raw_path, str) and raw_path.strip():
                return bytes(raw_bytes), Path(raw_path).name
            return bytes(raw_bytes), f"{source_id}.png"

        if isinstance(raw_path, str) and raw_path.strip():
            p = Path(raw_path)
            if p.exists() and p.is_file():
                return p.read_bytes(), p.name
        return None

    if isinstance(image_obj, Image.Image):
        buf = io.BytesIO()
        image_obj.save(buf, format="PNG")
        return buf.getvalue(), f"{source_id}.png"

    if isinstance(image_obj, str) and image_obj.strip():
        p = Path(image_obj)
        if p.exists() and p.is_file():
            return p.read_bytes(), p.name

    return None


def _load_scienceqa() -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or None
    cache_dir = os.environ.get("HF_DATASETS_CACHE") or "hf_home/datasets"

    # Prefer local raw snapshot when it already exists.
    local_repo = Path("data/raw/scienceqa")
    if local_repo.exists() and any(local_repo.iterdir()):
        return load_dataset(str(local_repo), cache_dir=str(cache_dir), token=token)

    return load_dataset(SCIENCEQA_REPO_ID, cache_dir=str(cache_dir), token=token)


def iter_scienceqa_mcqa_vqa() -> Iterator[tuple[McqaVqaSample, bytes]]:
    ds_dict = _load_scienceqa()

    for split_name, ds in ds_dict.items():
        for idx, row in enumerate(ds):
            question = str(row.get("question", "")).strip()
            if not question:
                continue

            choices = _normalize_choices(row.get("choices"))
            if len(choices) != len(CHOICE_ORDER):
                continue

            answer = _normalize_answer(row.get("answer"))
            if answer is None or answer not in choices:
                continue

            raw_id = row.get("id")
            if raw_id is None or str(raw_id).strip() == "":
                source_id = f"{split_name}_{idx:07d}"
            else:
                source_id = f"{split_name}_{str(raw_id).strip()}"

            image_info = _extract_image_bytes_and_filename(row.get("image"), source_id)
            if image_info is None:
                continue
            image_bytes, image_filename = image_info

            sample = McqaVqaSample(
                dataset="ScienceQA",
                source_id=source_id,
                question=question,
                choices=choices,
                answer=answer,
                image_filename=image_filename,
            )
            yield sample, image_bytes


@register_adapter("scienceqa")
class ScienceQaAdapter(DatasetAdapter):
    @property
    def name(self) -> str:
        return "scienceqa"

    @property
    def generator_id(self) -> str:
        return "M-2"

    @property
    def hf_repo_id(self) -> str | None:
        return SCIENCEQA_REPO_ID

    @property
    def hf_config(self) -> str | None:
        return SCIENCEQA_CONFIG

    @property
    def hf_split(self) -> str | None:
        return SCIENCEQA_SPLIT

    @property
    def hf_revision(self) -> str | None:
        return SCIENCEQA_REVISION

    def download(self, *, out_dir: Path) -> Path:
        token = os.environ.get("HF_TOKEN") or None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        local_path = snapshot_download(
            repo_id=SCIENCEQA_REPO_ID,
            repo_type="dataset",
            token=token,
            local_dir=str(out_dir),
        )
        return Path(local_path)

    def iter_mcqa_vqa(self) -> Iterator[tuple[McqaVqaSample, bytes]]:
        yield from iter_scienceqa_mcqa_vqa()
