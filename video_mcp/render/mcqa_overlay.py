from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from video_mcp.mcqa import Choice


@dataclass(frozen=True)
class Fonts:
    title: ImageFont.FreeTypeFont | ImageFont.ImageFont
    body: ImageFont.FreeTypeFont | ImageFont.ImageFont
    small: ImageFont.FreeTypeFont | ImageFont.ImageFont


def _pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Avoid try/except: only load truetype if file exists.
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
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def make_fonts() -> Fonts:
    return Fonts(title=_pick_font(34), body=_pick_font(22), small=_pick_font(18))


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


def draw_corner_choices(
    canvas: Image.Image,
    *,
    lit_choice: Choice | None,
    fonts: Fonts,
) -> None:
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
        is_lit = lit_choice == c
        fill = (80, 80, 80) if is_lit else (245, 245, 245)
        outline = (60, 60, 60)
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=3)

        letter_fill = (255, 255, 255) if is_lit else (30, 30, 30)
        bb = draw.textbbox((0, 0), c, font=fonts.title)
        lw = bb[2] - bb[0]
        lh = bb[3] - bb[1]
        draw.text((x1 + (box_w - lw) / 2, y1 + (box_h - lh) / 2 - 2), c, font=fonts.title, fill=letter_fill)


def draw_question_panel(
    canvas: Image.Image,
    *,
    question: str,
    choices: Iterable[str],
    image: Image.Image | None,
    fonts: Fonts,
) -> None:
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    panel_w = int(w * 0.62)
    panel_h = int(h * 0.46)
    px1 = int((w - panel_w) / 2)
    py1 = int((h - panel_h) / 2)
    px2 = px1 + panel_w
    py2 = py1 + panel_h

    draw.rounded_rectangle([px1, py1, px2, py2], radius=18, fill=(235, 235, 235), outline=(120, 120, 120), width=3)

    header_h = int(panel_h * 0.22)
    draw.rectangle([px1, py1, px2, py1 + header_h], fill=(245, 245, 245), outline=(120, 120, 120), width=3)
    header_text = "Questions"
    hb = draw.textbbox((0, 0), header_text, font=fonts.title)
    draw.text((px1 + (panel_w - (hb[2] - hb[0])) / 2, py1 + (header_h - (hb[3] - hb[1])) / 2 - 2), header_text, font=fonts.title, fill=(30, 30, 30))

    padding = 18
    body_x1 = px1 + padding
    body_x2 = px2 - padding
    y = py1 + header_h + padding
    max_w = body_x2 - body_x1

    for line in _wrap_text(draw, question, fonts.body, max_width_px=max_w)[:3]:
        draw.text((body_x1, y), line, font=fonts.body, fill=(20, 20, 20))
        bb = draw.textbbox((0, 0), line, font=fonts.body)
        y += (bb[3] - bb[1]) + 6

    # Image box
    img_box_w = int(panel_w * 0.42)
    img_box_h = int(panel_h * 0.28)
    ix1 = int(px1 + (panel_w - img_box_w) / 2)
    iy1 = int(py1 + header_h + int(panel_h * 0.42))
    ix2 = ix1 + img_box_w
    iy2 = iy1 + img_box_h

    # Choices text between question and image box
    options_max_y = iy1 - 10
    for opt in list(choices)[:4]:
        if y >= options_max_y:
            break
        for ol in _wrap_text(draw, opt, fonts.small, max_width_px=max_w)[:2]:
            if y >= options_max_y:
                break
            draw.text((body_x1, y), ol, font=fonts.small, fill=(40, 40, 40))
            ob = draw.textbbox((0, 0), ol, font=fonts.small)
            y += (ob[3] - ob[1]) + 4
        y += 2

    draw.rectangle([ix1, iy1, ix2, iy2], fill=(250, 250, 250), outline=(120, 120, 120), width=3)

    if image is None:
        ph = "Image"
        pb = draw.textbbox((0, 0), ph, font=fonts.body)
        draw.text((ix1 + (img_box_w - (pb[2] - pb[0])) / 2, iy1 + (img_box_h - (pb[3] - pb[1])) / 2 - 2), ph, font=fonts.body, fill=(40, 40, 40))
        return

    fitted = _fit_into_box(image.convert("RGB"), box_w=img_box_w - 8, box_h=img_box_h - 8)
    fx, fy = fitted.size
    ox = ix1 + (img_box_w - fx) // 2
    oy = iy1 + (img_box_h - fy) // 2
    canvas.paste(fitted, (ox, oy))


def render_video_mcp_frame(
    *,
    width: int,
    height: int,
    question: str,
    choices: Iterable[str],
    image: Image.Image | None,
    show_panel: bool,
    lit_choice: Choice | None,
    fonts: Fonts,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    # Always show corner choices; highlight depends on frame.
    draw_corner_choices(canvas, lit_choice=lit_choice, fonts=fonts)
    if show_panel:
        draw_question_panel(canvas, question=question, choices=choices, image=image, fonts=fonts)
    return canvas

