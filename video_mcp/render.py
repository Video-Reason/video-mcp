from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from video_mcp.schemas import Choice


@dataclass(frozen=True)
class Fonts:
    title: ImageFont.FreeTypeFont | ImageFont.ImageFont
    body: ImageFont.FreeTypeFont | ImageFont.ImageFont
    small: ImageFont.FreeTypeFont | ImageFont.ImageFont


def _pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Avoid try/except per project rules: only attempt truetype if the file exists.
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
    return Fonts(
        title=_pick_font(34),
        body=_pick_font(22),
        small=_pick_font(18),
    )


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width_px: int,
) -> list[str]:
    words = [w for w in text.split(" ") if w]
    if not words:
        return [""]

    lines: list[str] = []
    current: list[str] = []
    for w in words:
        candidate = " ".join([*current, w])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width_px or not current:
            current.append(w)
            continue
        lines.append(" ".join(current))
        current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def render_horizon_frame(
    *,
    width: int,
    height: int,
    frame_idx: int,
    num_frames: int,
    sun_choice: Choice,
) -> Image.Image:
    """
    Make a simple synthetic "horizon" scene:
    - sky gradient + ground gradient
    - a bright sun whose position corresponds to the correct answer choice
    - subtle motion across frames
    """
    t = 0.0 if num_frames <= 1 else frame_idx / (num_frames - 1)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Horizon height with subtle bobbing
    base_horizon = int(height * 0.58)
    bob = int(6 * math.sin(2 * math.pi * (t * 1.0)))
    horizon_y = base_horizon + bob

    # Sky gradient
    sky_top = (168, 210, 255)
    sky_bottom = (220, 240, 255)
    for y in range(0, horizon_y):
        a = 0.0 if horizon_y <= 1 else y / (horizon_y - 1)
        r = int(sky_top[0] * (1 - a) + sky_bottom[0] * a)
        g = int(sky_top[1] * (1 - a) + sky_bottom[1] * a)
        b = int(sky_top[2] * (1 - a) + sky_bottom[2] * a)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Ground gradient
    ground_top = (120, 190, 120)
    ground_bottom = (70, 140, 70)
    for y in range(horizon_y, height):
        a = 0.0 if height - horizon_y <= 1 else (y - horizon_y) / (height - horizon_y - 1)
        r = int(ground_top[0] * (1 - a) + ground_bottom[0] * a)
        g = int(ground_top[1] * (1 - a) + ground_bottom[1] * a)
        b = int(ground_top[2] * (1 - a) + ground_bottom[2] * a)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # A slightly darker horizon line
    draw.line([(0, horizon_y), (width, horizon_y)], fill=(90, 120, 90), width=3)

    # Sun base position mapping to choices.
    # A=left, B=right, C=down, D=up.
    x_base = int(width * 0.25)
    y_base = int(height * 0.28)
    if sun_choice == "B":
        x_base = int(width * 0.75)
    if sun_choice == "C":
        y_base = int(height * 0.45)
    if sun_choice == "D":
        y_base = int(height * 0.18)

    # Subtle drift
    x = x_base + int(10 * math.sin(2 * math.pi * (t * 0.7)))
    y = y_base + int(6 * math.cos(2 * math.pi * (t * 0.9)))
    radius = int(min(width, height) * 0.045)

    # Sun glow
    glow_r = int(radius * 1.7)
    draw.ellipse(
        [x - glow_r, y - glow_r, x + glow_r, y + glow_r],
        fill=(255, 245, 190),
        outline=None,
    )
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=(255, 225, 110),
        outline=(240, 190, 40),
        width=3,
    )

    # A couple of simple clouds
    cloud_y = int(height * 0.20) + int(4 * math.sin(2 * math.pi * (t * 0.8)))
    cloud_x = int(width * (0.15 + 0.1 * math.sin(2 * math.pi * (t * 0.4))))
    _draw_cloud(draw, cloud_x, cloud_y, scale=1.0)
    _draw_cloud(draw, int(width * 0.55) + int(16 * math.cos(2 * math.pi * (t * 0.35))), int(height * 0.23), scale=1.2)

    return img


def _draw_cloud(draw: ImageDraw.ImageDraw, x: int, y: int, *, scale: float) -> None:
    w = int(160 * scale)
    h = int(60 * scale)
    fill = (245, 250, 255)
    outline = (220, 230, 240)
    # three puffs
    draw.ellipse([x, y, x + int(w * 0.45), y + int(h * 0.9)], fill=fill, outline=outline, width=2)
    draw.ellipse([x + int(w * 0.22), y - int(h * 0.2), x + int(w * 0.7), y + int(h * 0.8)], fill=fill, outline=outline, width=2)
    draw.ellipse([x + int(w * 0.5), y, x + w, y + int(h * 0.9)], fill=fill, outline=outline, width=2)
    # base
    draw.rounded_rectangle([x + int(w * 0.12), y + int(h * 0.35), x + int(w * 0.88), y + int(h * 0.95)], radius=int(18 * scale), fill=fill, outline=outline, width=2)


def draw_mcqa_overlay(
    img: Image.Image,
    *,
    question: str,
    choices: Iterable[str],
    lit_choice: Choice | None,
    show_panel: bool,
    fonts: Fonts,
) -> Image.Image:
    """
    Draw the MCQA layout:
    - central "Questions" panel (optional, for first-frame-only mode)
    - A/B/C/D boxes at four corners of the frame (always shown)
    - A. B. C. D. labels under the panel (when show_panel)
    - highlight: fill the chosen corner box
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Corner answer boxes (match the figure: A TL, B TR, C BL, D BR)
    margin = int(min(w, h) * 0.06)
    box_w = int(min(w, h) * 0.14)
    box_h = int(box_w * 0.75)

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
        draw.rounded_rectangle([x1, y1, x2, y2], radius=int(box_h * 0.12), fill=fill, outline=outline, width=3)

        letter_fill = (255, 255, 255) if is_lit else (30, 30, 30)
        letter_bbox = draw.textbbox((0, 0), c, font=fonts.title)
        lw = letter_bbox[2] - letter_bbox[0]
        lh = letter_bbox[3] - letter_bbox[1]
        draw.text(
            (x1 + (box_w - lw) / 2, y1 + (box_h - lh) / 2 - 2),
            c,
            font=fonts.title,
            fill=letter_fill,
        )

    if not show_panel:
        return img

    # Central panel
    panel_w = int(w * 0.62)
    panel_h = int(h * 0.46)
    panel_x1 = int((w - panel_w) / 2)
    panel_y1 = int((h - panel_h) / 2)
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h

    draw.rounded_rectangle(
        [panel_x1, panel_y1, panel_x2, panel_y2],
        radius=18,
        fill=(235, 235, 235),
        outline=(120, 120, 120),
        width=3,
    )

    # "Questions" header bar
    header_h = int(panel_h * 0.22)
    draw.rectangle(
        [panel_x1, panel_y1, panel_x2, panel_y1 + header_h],
        fill=(245, 245, 245),
        outline=(120, 120, 120),
        width=3,
    )

    header_text = "Questions"
    hb = draw.textbbox((0, 0), header_text, font=fonts.title)
    draw.text(
        (panel_x1 + (panel_w - (hb[2] - hb[0])) / 2, panel_y1 + (header_h - (hb[3] - hb[1])) / 2 - 2),
        header_text,
        font=fonts.title,
        fill=(30, 30, 30),
    )

    # Question body text just below header
    padding = 18
    body_x1 = panel_x1 + padding
    body_x2 = panel_x2 - padding
    body_y1 = panel_y1 + header_h + padding

    max_body_width = body_x2 - body_x1
    q_lines = _wrap_text(draw, question, fonts.body, max_width_px=max_body_width)
    y = body_y1
    for line in q_lines[:3]:
        draw.text((body_x1, y), line, font=fonts.body, fill=(20, 20, 20))
        lb = draw.textbbox((0, 0), line, font=fonts.body)
        y += (lb[3] - lb[1]) + 6

    # Inner image rectangle
    img_box_w = int(panel_w * 0.42)
    img_box_h = int(panel_h * 0.28)
    img_box_x1 = int(panel_x1 + (panel_w - img_box_w) / 2)
    img_box_y1 = int(panel_y1 + header_h + int(panel_h * 0.42))
    img_box_x2 = img_box_x1 + img_box_w
    img_box_y2 = img_box_y1 + img_box_h

    # Multiple-choice option text between question and image box
    options_max_y = img_box_y1 - 10
    for opt in list(choices)[:4]:
        if y >= options_max_y:
            break
        opt_lines = _wrap_text(draw, opt, fonts.small, max_width_px=max_body_width)
        for ol in opt_lines[:2]:
            if y >= options_max_y:
                break
            draw.text((body_x1, y), ol, font=fonts.small, fill=(40, 40, 40))
            ob = draw.textbbox((0, 0), ol, font=fonts.small)
            y += (ob[3] - ob[1]) + 4
        y += 2

    draw.rectangle([img_box_x1, img_box_y1, img_box_x2, img_box_y2], fill=(250, 250, 250), outline=(120, 120, 120), width=3)
    ph_text = "Image"
    pb = draw.textbbox((0, 0), ph_text, font=fonts.body)
    draw.text(
        (img_box_x1 + (img_box_w - (pb[2] - pb[0])) / 2, img_box_y1 + (img_box_h - (pb[3] - pb[1])) / 2 - 2),
        ph_text,
        font=fonts.body,
        fill=(40, 40, 40),
    )

    # A. B. C. D. labels under the panel
    labels = ["A.", "B.", "C.", "D."]
    label_y = panel_y2 - padding - 6
    step = panel_w / 4.0
    for i, lab in enumerate(labels):
        lx = panel_x1 + step * (i + 0.5)
        bb = draw.textbbox((0, 0), lab, font=fonts.small)
        draw.text((lx - (bb[2] - bb[0]) / 2, label_y), lab, font=fonts.small, fill=(60, 60, 60))

    return img

