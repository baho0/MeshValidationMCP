"""Pillow composition: contact sheet, view labels, axis gizmo, colorbar."""

from __future__ import annotations

import io
from typing import TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import RenderedTile

_AXIS_COLORS = {"X": (214, 39, 40), "Y": (44, 160, 44), "Z": (31, 119, 180)}
_LABEL_COLOR = (48, 52, 58)
_TILE_BORDER = (208, 212, 216)
_GUTTER = 12


class ColorbarSpec(TypedDict):
    lut: np.ndarray  # (256, 3) uint8
    vmin: float
    vmax: float
    label: str


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without the size kwarg
        return ImageFont.load_default()


def _draw_axis_gizmo(image: Image.Image, rotation: np.ndarray) -> None:
    """Orientation-only gizmo in the bottom-left corner (like CAD viewports)."""
    length = max(18, image.width // 28)
    margin = length // 2 + 8
    origin = (margin + length, image.height - margin - length)
    draw = ImageDraw.Draw(image)
    axes = (("X", (1.0, 0.0, 0.0)), ("Y", (0.0, 1.0, 0.0)), ("Z", (0.0, 0.0, 1.0)))
    for name, axis in axes:
        v = rotation @ np.asarray(axis)
        dx, dy = float(v[0]) * length, -float(v[1]) * length  # PIL y grows downward
        end = (origin[0] + dx, origin[1] + dy)
        draw.line([origin, end], fill=_AXIS_COLORS[name], width=2)
        tip = (origin[0] + dx * 1.4 - 4, origin[1] + dy * 1.4 - 6)
        draw.text(tip, name, fill=_AXIS_COLORS[name], font=_font(max(10, length // 2)))


def annotate_tile(tile: RenderedTile) -> Image.Image:
    image = tile.image
    _draw_axis_gizmo(image, tile.rotation)
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), tile.view.name, fill=_LABEL_COLOR, font=_font(max(13, image.width // 38)))
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=_TILE_BORDER, width=1)
    return image


def grid_shape(view_count: int) -> tuple[int, int]:
    return {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2), 5: (2, 3), 6: (2, 3)}[view_count]


def _draw_colorbar(sheet: Image.Image, spec: ColorbarSpec, top: int, width: int) -> None:
    strip_h = 18
    inner_w = width - 2 * _GUTTER
    vmin, vmax = spec["vmin"], spec["vmax"]
    uniform = not vmax > vmin
    if uniform:
        # Every value is identical: a gradient ramp would be misleading.
        ramp = np.tile(spec["lut"][0], (inner_w, 1))
    else:
        ramp = spec["lut"][np.linspace(0, 255, inner_w).astype(int)]  # (inner_w, 3)
    strip = np.repeat(ramp[np.newaxis, :, :], strip_h, axis=0).astype(np.uint8)
    sheet.paste(Image.fromarray(strip), (_GUTTER, top))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle(
        (_GUTTER, top, _GUTTER + inner_w - 1, top + strip_h - 1), outline=_TILE_BORDER, width=1
    )
    font = _font(12)
    y = top + strip_h + 2
    if uniform:
        draw.text((_GUTTER, y), f"uniform {vmin:.4g}", fill=_LABEL_COLOR, font=font)
    else:
        draw.text((_GUTTER, y), f"{vmin:.4g}", fill=_LABEL_COLOR, font=font)
        mid_text = f"{(vmin + vmax) / 2.0:.4g}"
        draw.text(
            (_GUTTER + inner_w // 2 - 4 * len(mid_text), y), mid_text,
            fill=_LABEL_COLOR, font=font,
        )
        max_text = f"{vmax:.4g}"
        draw.text(
            (_GUTTER + inner_w - 8 * len(max_text), y), max_text,
            fill=_LABEL_COLOR, font=font,
        )
    draw.text((_GUTTER, y + 16), spec["label"], fill=_LABEL_COLOR, font=font)


def compose_sheet(tiles: list[RenderedTile], colorbar: ColorbarSpec | None = None) -> Image.Image:
    rows, cols = grid_shape(len(tiles))
    tile_w, tile_h = tiles[0].image.size
    width = cols * tile_w + (cols + 1) * _GUTTER
    height = rows * tile_h + (rows + 1) * _GUTTER
    colorbar_h = 56 if colorbar is not None else 0
    sheet = Image.new("RGB", (width, height + colorbar_h), "white")

    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        x = _GUTTER + col * (tile_w + _GUTTER)
        y = _GUTTER + row * (tile_h + _GUTTER)
        sheet.paste(annotate_tile(tile), (x, y))

    if colorbar is not None:
        _draw_colorbar(sheet, colorbar, top=height, width=width)
    return sheet


def encode_png(image: Image.Image, max_edge: int | None = None) -> bytes:
    if max_edge is not None and max(image.size) > max_edge:
        image = image.copy()
        image.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()
