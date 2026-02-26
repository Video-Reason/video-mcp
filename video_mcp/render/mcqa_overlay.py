from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from video_mcp.mcqa import Choice, LitStyle


@dataclass(frozen=True)
class Fonts:
    title: ImageFont.FreeTypeFont | ImageFont.ImageFont
    body: ImageFont.FreeTypeFont | ImageFont.ImageFont
    small: ImageFont.FreeTypeFont | ImageFont.ImageFont


_FONT_PATH: str | None = None  # resolved on first call
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a TrueType font at *size* pt (cached)."""
    global _FONT_PATH
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    if _FONT_PATH is None:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/Library/Fonts/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
            "/Library/Fonts/Verdana.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                _FONT_PATH = p
                break
        if _FONT_PATH is None:
            _FONT_PATH = ""  # sentinel: use default bitmap font

    if _FONT_PATH:
        font = ImageFont.truetype(_FONT_PATH, size=size)
    else:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _font_size(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    """Extract point size from a font object (fallback 12 for bitmap fonts)."""
    return getattr(font, "size", 12)


_REF_MIN_DIM = 768  # reference min(w,h) for the original 1024×768 layout
_REF_TITLE = 44
_REF_BODY = 30
_REF_SMALL = 26


def make_fonts(*, width: int = 832, height: int = 480) -> Fonts:
    """Create fonts scaled proportionally to the canvas resolution.

    The reference sizes (44/30/26 pt) were designed for 1024×768.
    """
    scale = min(width, height) / _REF_MIN_DIM
    return Fonts(
        title=_pick_font(max(16, int(round(_REF_TITLE * scale)))),
        body=_pick_font(max(12, int(round(_REF_BODY * scale)))),
        small=_pick_font(max(10, int(round(_REF_SMALL * scale)))),
    )


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width_px: int,
) -> list[str]:
    words = [w for w in str(text).split(" ") if w]
    if not words:
        return [""]

    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        candidate = " ".join([*cur, w])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width_px or not cur:
            cur.append(w)
            continue
        lines.append(" ".join(cur))
        cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _fit_into_box(img: Image.Image, *, box_w: int, box_h: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img.resize((box_w, box_h))
    scale = min(box_w / w, box_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh))


def _lerp_color(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Linearly interpolate between two RGB colours by factor *t* (0→a, 1→b)."""
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


# Colours for the corner choice boxes.
_BOX_BASE: tuple[int, int, int] = (245, 245, 245)
_LETTER_BASE: tuple[int, int, int] = (30, 30, 30)
_OUTLINE_BASE: tuple[int, int, int] = (60, 60, 60)

# "darken" style: box darkens, letter stays dark.
_BOX_DARKEN: tuple[int, int, int] = (140, 140, 140)

# "red_border" style: thick red outline.
_RED_BORDER: tuple[int, int, int] = (220, 30, 30)
_RED_BORDER_WIDTH = 6

# "circle" style: black ellipse around the correct corner box.
_CIRCLE_COLOR: tuple[int, int, int] = (0, 0, 0)
_CIRCLE_MAX_WIDTH = 5
_CIRCLE_PAD = 6


def draw_corner_choices(
    canvas: Image.Image,
    *,
    lit_choice: Choice | None,
    lit_progress: float = 1.0,
    lit_style: LitStyle = "darken",
    fonts: Fonts,
) -> None:
    """Draw A/B/C/D corner boxes. *lit_progress* (0‑1) fades the highlight."""
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    margin = int(min(w, h) * 0.06)
    box_w = int(min(w, h) * 0.14)
    box_h = int(box_w * 0.75)
    radius = int(box_h * 0.12)

    boxes: dict[Choice, tuple[int, int, int, int]] = {
        "A": (margin, margin, margin + box_w, margin + box_h),
        "B": (w - margin - box_w, margin, w - margin, margin + box_h),
        "C": (margin, h - margin - box_h, margin + box_w, h - margin),
        "D": (w - margin - box_w, h - margin - box_h, w - margin, h - margin),
    }

    for c, (x1, y1, x2, y2) in boxes.items():
        is_target = lit_choice == c
        t = lit_progress if is_target else 0.0

        if lit_style == "darken":
            fill = _lerp_color(_BOX_BASE, _BOX_DARKEN, t)
            outline = _OUTLINE_BASE
            outline_w = 3
        elif lit_style == "red_border":
            fill = _BOX_BASE
            outline = _lerp_color(_OUTLINE_BASE, _RED_BORDER, t)
            outline_w = int(round(3 + (_RED_BORDER_WIDTH - 3) * t)) if is_target else 3
        else:  # circle
            fill = _BOX_BASE
            outline = _OUTLINE_BASE
            outline_w = 3

        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=outline_w)

        bb = draw.textbbox((0, 0), c, font=fonts.title)
        lw = bb[2] - bb[0]
        lh = bb[3] - bb[1]
        draw.text((x1 + (box_w - lw) / 2, y1 + (box_h - lh) / 2 - 2), c, font=fonts.title, fill=_LETTER_BASE)

        if lit_style == "circle" and is_target and t > 0:
            ew = int(round(_CIRCLE_MAX_WIDTH * t))
            if ew >= 1:
                draw.ellipse(
                    [x1 - _CIRCLE_PAD, y1 - _CIRCLE_PAD, x2 + _CIRCLE_PAD, y2 + _CIRCLE_PAD],
                    outline=_CIRCLE_COLOR,
                    width=ew,
                )


def _measure_text_layout(
    draw: ImageDraw.ImageDraw,
    *,
    question: str,
    choice_list: list[str],
    q_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    c_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text_max_w: int,
    q_spacing: int = 8,
    c_line_spacing: int = 6,
    c_gap: int = 10,
    sep_h: int = 30,
) -> tuple[list[str], list[list[str]], int]:
    """Wrap all text and return (q_lines, c_wrapped, total_height)."""
    q_lines = _wrap_text(draw, question, q_font, max_width_px=text_max_w)
    q_h = 0
    for ql in q_lines:
        bb = draw.textbbox((0, 0), ql, font=q_font)
        q_h += (bb[3] - bb[1]) + q_spacing

    c_wrapped: list[list[str]] = []
    c_h = 0
    for opt in choice_list:
        lines = _wrap_text(draw, opt, c_font, max_width_px=text_max_w)[:3]
        for ol in lines:
            ob = draw.textbbox((0, 0), ol, font=c_font)
            c_h += (ob[3] - ob[1]) + c_line_spacing
        c_wrapped.append(lines)
        c_h += c_gap

    return q_lines, c_wrapped, q_h + sep_h + c_h


def draw_question_panel(
    canvas: Image.Image,
    *,
    question: str,
    choices: Iterable[str],
    image: Image.Image | None,
    fonts: Fonts,
) -> None:
    """
    Two-column panel layout with **adaptive font sizing**.

      - Left column : source image
      - Right column: question text + choice options

    When text is too long for the default fonts, the function progressively
    tries wider text columns and smaller fonts until everything fits
    without any truncation.
    """
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    # Mirror the corner-box geometry so the panel avoids them.
    cb_margin = int(min(w, h) * 0.06)
    cb_box_w = int(min(w, h) * 0.14)
    cb_box_h = int(cb_box_w * 0.75)
    gap = 14  # breathing room between corner boxes and panel

    px1 = cb_margin + cb_box_w + gap
    py1 = cb_margin + cb_box_h + gap
    px2 = w - cb_margin - cb_box_w - gap
    py2 = h - cb_margin - cb_box_h - gap
    panel_w = px2 - px1
    panel_h = py2 - py1

    draw.rounded_rectangle(
        [px1, py1, px2, py2], radius=18,
        fill=(235, 235, 235), outline=(120, 120, 120), width=3,
    )

    # --- header bar ---
    header_h = int(panel_h * 0.11)
    draw.rounded_rectangle(
        [px1, py1, px2, py1 + header_h], radius=18,
        fill=(245, 245, 245), outline=(120, 120, 120), width=3,
    )
    header_text = "Question"
    hb = draw.textbbox((0, 0), header_text, font=fonts.title)
    draw.text(
        (px1 + (panel_w - (hb[2] - hb[0])) / 2,
         py1 + (header_h - (hb[3] - hb[1])) / 2 - 2),
        header_text, font=fonts.title, fill=(30, 30, 30),
    )

    # --- content area below header ---
    pad = 20
    content_y1 = py1 + header_h + pad
    content_y2 = py2 - pad
    content_h = content_y2 - content_y1

    choice_list = list(choices)[:4]
    base_body_sz = _font_size(fonts.body)
    base_small_sz = _font_size(fonts.small)
    q_spacing = 8
    c_line_spacing = 6
    c_gap = 10
    sep_h = 30  # 12 gap + 2 line + 16 gap

    # --- Adaptive layout search -------------------------------------------
    # Try (image_column_fraction, font_scale) combos; pick the first where
    # ALL question lines AND ALL choices fit without truncation.
    _CONFIGS = [
        (0.55, 1.0),
        (0.45, 1.0),
        (0.55, 0.82),
        (0.45, 0.82),
        (0.40, 0.70),
        (0.35, 0.60),
        (0.30, 0.52),
    ]

    best: dict | None = None
    for img_frac, fscale in _CONFIGS:
        q_font = _pick_font(max(10, int(round(base_body_sz * fscale))))
        c_font = _pick_font(max(9, int(round(base_small_sz * fscale))))

        img_col_w = int(panel_w * img_frac)
        text_x1 = px1 + img_col_w + pad // 2
        text_x2 = px2 - pad
        text_max_w = text_x2 - text_x1
        if text_max_w < 80:
            continue

        q_lines, c_wrapped, total_h = _measure_text_layout(
            draw,
            question=question,
            choice_list=choice_list,
            q_font=q_font,
            c_font=c_font,
            text_max_w=text_max_w,
            q_spacing=q_spacing,
            c_line_spacing=c_line_spacing,
            c_gap=c_gap,
            sep_h=sep_h,
        )

        if total_h <= content_h:
            best = dict(
                img_col_w=img_col_w,
                text_x1=text_x1,
                text_x2=text_x2,
                text_max_w=text_max_w,
                q_font=q_font,
                c_font=c_font,
                q_lines=q_lines,
                c_wrapped=c_wrapped,
            )
            break

    # Fallback: smallest config, reserve choices first, fit what we can for
    # the question (should only trigger for extremely long text).
    if best is None:
        fscale = 0.52
        img_frac = 0.30
        q_font = _pick_font(max(10, int(round(base_body_sz * fscale))))
        c_font = _pick_font(max(9, int(round(base_small_sz * fscale))))

        img_col_w = int(panel_w * img_frac)
        text_x1 = px1 + img_col_w + pad // 2
        text_x2 = px2 - pad
        text_max_w = text_x2 - text_x1

        # Measure choices (always keep all).
        c_wrapped: list[list[str]] = []
        c_total = 0
        for opt in choice_list:
            lines = _wrap_text(draw, opt, c_font, max_width_px=text_max_w)[:3]
            for ol in lines:
                ob = draw.textbbox((0, 0), ol, font=c_font)
                c_total += (ob[3] - ob[1]) + c_line_spacing
            c_wrapped.append(lines)
            c_total += c_gap

        q_budget = content_h - c_total - sep_h
        all_q = _wrap_text(draw, question, q_font, max_width_px=text_max_w)
        q_lines: list[str] = []
        accum = 0
        for ql in all_q:
            bb = draw.textbbox((0, 0), ql, font=q_font)
            lh = (bb[3] - bb[1]) + q_spacing
            if accum + lh > q_budget and q_lines:
                break
            q_lines.append(ql)
            accum += lh

        best = dict(
            img_col_w=img_col_w,
            text_x1=text_x1,
            text_x2=text_x2,
            text_max_w=text_max_w,
            q_font=q_font,
            c_font=c_font,
            q_lines=q_lines,
            c_wrapped=c_wrapped,
        )

    # --- Draw image (left column) -----------------------------------------
    img_col_w = best["img_col_w"]
    img_x1 = px1 + pad
    img_x2 = px1 + img_col_w - pad // 2
    img_y1 = content_y1
    img_y2 = content_y2
    img_box_w = img_x2 - img_x1
    img_box_h = img_y2 - img_y1

    draw.rounded_rectangle(
        [img_x1, img_y1, img_x2, img_y2], radius=12,
        fill=(250, 250, 250), outline=(160, 160, 160), width=2,
    )

    if image is not None:
        fitted = _fit_into_box(image.convert("RGB"), box_w=img_box_w - 16, box_h=img_box_h - 16)
        fx, fy = fitted.size
        ox = img_x1 + (img_box_w - fx) // 2
        oy = img_y1 + (img_box_h - fy) // 2
        canvas.paste(fitted, (ox, oy))
    else:
        ph = "Image"
        pb = draw.textbbox((0, 0), ph, font=best["q_font"])
        draw.text(
            (img_x1 + (img_box_w - (pb[2] - pb[0])) / 2,
             img_y1 + (img_box_h - (pb[3] - pb[1])) / 2),
            ph, font=best["q_font"], fill=(100, 100, 100),
        )

    # --- Draw question text (right column) --------------------------------
    text_x1 = best["text_x1"]
    text_x2 = best["text_x2"]
    q_font = best["q_font"]
    c_font = best["c_font"]

    y = content_y1
    for ql in best["q_lines"]:
        draw.text((text_x1, y), ql, font=q_font, fill=(20, 20, 20))
        bb = draw.textbbox((0, 0), ql, font=q_font)
        y += (bb[3] - bb[1]) + q_spacing

    # --- Separator ---
    y += 12
    draw.line([(text_x1, y), (text_x2, y)], fill=(180, 180, 180), width=2)
    y += 16

    # --- Choice options ---
    for lines in best["c_wrapped"]:
        for ol in lines:
            draw.text((text_x1, y), ol, font=c_font, fill=(40, 40, 40))
            ob = draw.textbbox((0, 0), ol, font=c_font)
            y += (ob[3] - ob[1]) + c_line_spacing
        y += c_gap


def render_video_mcp_frame(
    *,
    width: int,
    height: int,
    question: str,
    choices: Iterable[str],
    image: Image.Image | None,
    show_panel: bool,
    lit_choice: Choice | None,
    lit_progress: float = 1.0,
    lit_style: LitStyle = "darken",
    fonts: Fonts,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    # Panel first, then corner choices on top so they're never hidden.
    if show_panel:
        draw_question_panel(canvas, question=question, choices=choices, image=image, fonts=fonts)
    draw_corner_choices(canvas, lit_choice=lit_choice, lit_progress=lit_progress, lit_style=lit_style, fonts=fonts)
    return canvas

