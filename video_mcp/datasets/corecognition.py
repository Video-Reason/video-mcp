from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CoreCognitionRawRow(BaseModel):
    id: str
    concept: str | None = None
    stage: str | None = None
    type: str = Field(description="'MC' for multiple-choice, 'TF' for true/false")
    question: str
    images: str | None = None
    videos: str | None = None
    answer: str
    choices: str


class CoreCognitionMcqaVqaExample(BaseModel):
    id: str
    concept: str | None = None
    stage: str | None = None

    source: str = Field(description="'default' (HF preview) or 'complete' (ZIP/CSV).")
    image: str
    media_path: str = Field(description="Path inside CoreCognition ZIP, e.g. CoreCognition_20250622/media/a0052.png")
    question: str
    choices: dict[str, str]
    answer: str


_IMAGE_PLACEHOLDER_RE = re.compile(r"^<image-placeholder:\s*([^>]+)>\s*", flags=re.IGNORECASE)
_VIDEO_PLACEHOLDER_RE = re.compile(r"<video-placeholder:", flags=re.IGNORECASE)


def _normalize_choices(raw: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue
        vs = str(v).strip()
        if not vs or vs.lower() == "nan":
            continue
        out[ks] = vs
    return out


_CHOICES_KV_RE = re.compile(
    r"""(?P<k>['"]?[A-Z]['"]?)\s*:\s*(?P<v>nan|None|['"].*?['"])""",
    flags=re.IGNORECASE,
)


def _parse_choices_str(s: str) -> dict[str, Any]:
    """
    CoreCognition 'choices' is a Python-dict-like string, e.g.:
      {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': nan, 'F': nan}
    Sometimes values are double-quoted and contain apostrophes.

    We parse it with regex (no try/except) and drop nan/None.
    """
    text = str(s).strip()
    if not text:
        return {}

    out: dict[str, Any] = {}
    for m in _CHOICES_KV_RE.finditer(text):
        k_raw = m.group("k").strip().strip("'").strip('"')
        v_raw = m.group("v").strip()

        if not k_raw:
            continue
        k = k_raw.upper()

        if v_raw.lower() in {"nan", "none"}:
            continue
        if len(v_raw) >= 2 and ((v_raw[0] == v_raw[-1] == '"') or (v_raw[0] == v_raw[-1] == "'")):
            v = v_raw[1:-1]
        else:
            v = v_raw
        out[k] = v

    return out


def _split_single_image(images: str | None) -> str | None:
    if images is None:
        return None
    s = str(images).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) != 1:
        return None
    return parts[0]


def _strip_image_placeholder(question: str) -> tuple[str | None, str]:
    q = str(question)
    m = _IMAGE_PLACEHOLDER_RE.match(q)
    if m is None:
        return None, q.strip()
    return m.group(1).strip(), q[m.end() :].strip()


def load_corecognition_dataset(*, split: str = "train", config: str = "default"):
    """
    Load CoreCognition using Hugging Face Datasets.

    - config='default' loads the lightweight preview metadata (recommended for filtering).
    - config='complete' is handled separately (ZIP + CSV), not via `datasets` parquet loader.
    """
    from datasets import load_dataset  # dependency is in requirements.txt

    import os

    cache_dir = os.environ.get("HF_DATASETS_CACHE") or "hf_home/datasets"
    token = os.environ.get("HF_TOKEN") or None
    return load_dataset(
        "williamium/CoreCognition",
        config,
        split=split,
        cache_dir=str(cache_dir),
        token=token,
    )


def download_corecognition_complete_zip() -> Path:
    """
    Download the complete ZIP from the HF dataset repo into the local HF cache.
    Requires HF_TOKEN for gated access.
    """
    import os

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or None
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "hf_home/hub"

    zip_path = hf_hub_download(
        repo_id="williamium/CoreCognition",
        repo_type="dataset",
        filename="CoreCognition_20250622.zip",
        token=token,
        cache_dir=str(cache_dir),
    )
    return Path(zip_path)


def iter_corecognition_rows(*, split: str = "train", config: str = "default"):
    """
    Yield raw rows as dicts matching `CoreCognitionRawRow`.
    """
    if config != "complete":
        ds = load_corecognition_dataset(split=split, config=config)
        for row in ds:
            yield row
        return

    # complete: read CoreCognition.csv inside the ZIP
    # (split is currently always train in this dataset release)
    _ = split

    zip_path = download_corecognition_complete_zip()
    csv_name = "CoreCognition_20250622/CoreCognition.csv"

    with zipfile.ZipFile(zip_path) as z:
        data = z.read(csv_name)

    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        yield row


def iter_corecognition_mcqa_single_image(*, split: str = "train", config: str = "default"):
    for row in iter_corecognition_rows(split=split, config=config):
        raw = CoreCognitionRawRow.model_validate(row)

        if str(raw.type).strip().upper() != "MC":
            continue
        if raw.videos is not None and str(raw.videos).strip():
            continue

        if _VIDEO_PLACEHOLDER_RE.search(raw.question) is not None:
            continue

        image_from_col = _split_single_image(raw.images)
        image_from_q, question_text = _strip_image_placeholder(raw.question)

        image = image_from_col or image_from_q
        if image is None:
            continue

        choices = _normalize_choices(_parse_choices_str(raw.choices))
        if not choices:
            continue

        yield CoreCognitionMcqaVqaExample(
            id=str(raw.id),
            concept=raw.concept,
            stage=raw.stage,
            source=str(config),
            image=image,
            media_path=f"CoreCognition_20250622/media/{image}",
            question=question_text,
            choices=choices,
            answer=str(raw.answer).strip(),
        )


def extract_corecognition_mcqa_single_image(
    *,
    out_path: Path,
    split: str = "train",
    config: str = "default",
    export_media_dir: Path | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    media_paths: set[str] = set()
    with out_path.open("w", encoding="utf-8") as f:
        for ex in iter_corecognition_mcqa_single_image(split=split, config=config):
            f.write(ex.model_dump_json() + "\n")
            media_paths.add(ex.media_path)

    if export_media_dir is None:
        return

    # Export only works for complete ZIP (real media files).
    if config != "complete":
        return

    export_media_dir = Path(export_media_dir)
    export_media_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_corecognition_complete_zip()
    with zipfile.ZipFile(zip_path) as z:
        available = set(z.namelist())
        for mp in sorted(media_paths):
            if mp not in available:
                continue
            # Keep only basename in export folder to make training ingestion simple.
            out_file = export_media_dir / Path(mp).name
            if out_file.exists():
                continue
            out_file.write_bytes(z.read(mp))

