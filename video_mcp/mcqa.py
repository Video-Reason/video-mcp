from __future__ import annotations

from typing import Literal, cast

Choice = Literal["A", "B", "C", "D"]
LitStyle = Literal["darken", "red_border"]

LIT_STYLES: tuple[LitStyle, ...] = ("darken", "red_border")
CHOICE_ORDER: tuple[Choice, ...] = ("A", "B", "C", "D")


def normalize_choice(value: str) -> Choice | None:
    """
    Normalize a raw answer string into an MCQA choice letter.
    Returns None if the value is not one of A/B/C/D.
    """
    v = str(value).strip().upper()
    if v in {"A", "B", "C", "D"}:
        return cast(Choice, v)
    return None

