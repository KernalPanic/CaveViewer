"""
gui/bitmap_font.py

A minimal 5x7 pixel bitmap font, used to draw real text labels in the
overlay UI system (LightSlider, Minimap, RenderModeButtons), which only
knows how to draw vector rectangles/circles -- there's no font rasterizer
or glyph rendering available in that pipeline, so this hand-encodes just
the characters actually needed for CaveViewer's UI labels as 5-wide,
7-tall pixel grids, each pixel becoming one small quad when drawn.

This is intentionally not a general-purpose font system -- it only
includes the characters used in "MESH" and "TEXTURE" (plus a few extras
that are cheap to include for future labels: digits and common
punctuation). Add more characters here if a future label needs them.

Each glyph is a list of 7 strings (one per row, top to bottom), 5
characters each, where '#' is a filled pixel and '.' is empty.
"""

from __future__ import annotations

_GLYPHS: dict[str, list[str]] = {
    "M": [
        "#...#",
        "##.##",
        "#.#.#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
    ],
    "E": [
        "#####",
        "#....",
        "#....",
        "###..",
        "#....",
        "#....",
        "#####",
    ],
    "S": [
        ".####",
        "#....",
        "#....",
        ".###.",
        "....#",
        "....#",
        "####.",
    ],
    "H": [
        "#...#",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ],
    "T": [
        "#####",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
    ],
    "X": [
        "#...#",
        "#...#",
        ".#.#.",
        "..#..",
        ".#.#.",
        "#...#",
        "#...#",
    ],
    "U": [
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ],
    "R": [
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#..#.",
        "#...#",
        "#...#",
    ],
    "O": [
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ],
    "N": [
        "#...#",
        "##..#",
        "#.#.#",
        "#.#.#",
        "#..##",
        "#...#",
        "#...#",
    ],
    "F": [
        "#####",
        "#....",
        "#....",
        "###..",
        "#....",
        "#....",
        "#....",
    ],
    " ": [
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
    ],
    "A": [
        ".###.",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ],
    "B": [
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#...#",
        "#...#",
        "####.",
    ],
    "C": [
        ".####",
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        ".####",
    ],
    "D": [
        "####.",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "####.",
    ],
    "G": [
        ".####",
        "#....",
        "#....",
        "#.###",
        "#...#",
        "#...#",
        ".####",
    ],
    "I": [
        "#####",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "#####",
    ],
    "J": [
        "....#",
        "....#",
        "....#",
        "....#",
        "#...#",
        "#...#",
        ".###.",
    ],
    "K": [
        "#...#",
        "#..#.",
        "#.#..",
        "##...",
        "#.#..",
        "#..#.",
        "#...#",
    ],
    "L": [
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#####",
    ],
    "P": [
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#....",
        "#....",
        "#....",
    ],
    "Q": [
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        "#.#.#",
        "#..#.",
        ".##.#",
    ],
    "V": [
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".#.#.",
        ".#.#.",
        "..#..",
    ],
    "W": [
        "#...#",
        "#...#",
        "#...#",
        "#.#.#",
        "#.#.#",
        "#.#.#",
        ".#.#.",
    ],
    "Y": [
        "#...#",
        "#...#",
        ".#.#.",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
    ],
    "/": [
        "....#",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        "#....",
        "#....",
    ],
    "0": [
        ".###.",
        "#...#",
        "#..##",
        "#.#.#",
        "##..#",
        "#...#",
        ".###.",
    ],
    "1": [
        "..#..",
        ".##..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        ".###.",
    ],
    "2": [
        ".###.",
        "#...#",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        "#####",
    ],
    "3": [
        "####.",
        "....#",
        "....#",
        ".###.",
        "....#",
        "....#",
        "####.",
    ],
    "4": [
        "#...#",
        "#...#",
        "#...#",
        "#####",
        "....#",
        "....#",
        "....#",
    ],
    "5": [
        "#####",
        "#....",
        "#....",
        "####.",
        "....#",
        "....#",
        "####.",
    ],
    "6": [
        ".###.",
        "#....",
        "#....",
        "####.",
        "#...#",
        "#...#",
        ".###.",
    ],
    "7": [
        "#####",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        ".#...",
        ".#...",
    ],
    "8": [
        ".###.",
        "#...#",
        "#...#",
        ".###.",
        "#...#",
        "#...#",
        ".###.",
    ],
    "9": [
        ".###.",
        "#...#",
        "#...#",
        ".####",
        "....#",
        "....#",
        ".###.",
    ],
    ":": [
        ".....",
        "..#..",
        "..#..",
        ".....",
        "..#..",
        "..#..",
        ".....",
    ],
    "-": [
        ".....",
        ".....",
        ".....",
        "#####",
        ".....",
        ".....",
        ".....",
    ],
    "+": [
        ".....",
        "..#..",
        "..#..",
        "#####",
        "..#..",
        "..#..",
        ".....",
    ],
    "%": [
        "#...#",
        "#..#.",
        "...#.",
        "..#..",
        ".#...",
        "#..#.",
        "#...#",
    ],
    "(": [
        "...#.",
        "..#..",
        ".#...",
        ".#...",
        ".#...",
        "..#..",
        "...#.",
    ],
    ")": [
        ".#...",
        "..#..",
        "...#.",
        "...#.",
        "...#.",
        "..#..",
        ".#...",
    ],
}

GLYPH_COLS = 5
GLYPH_ROWS = 7


def text_width_px(text: str, pixel_size: float, letter_spacing: float = 1.0) -> float:
    """Total rendered width of `text` at the given pixel_size (size of one
    bitmap pixel in screen pixels), including letter_spacing gaps between
    characters but not trailing space."""
    n = len(text)
    if n == 0:
        return 0.0
    glyph_w = GLYPH_COLS * pixel_size
    spacing = letter_spacing * pixel_size
    return n * glyph_w + max(n - 1, 0) * spacing


def text_height_px(pixel_size: float) -> float:
    return GLYPH_ROWS * pixel_size


def iter_text_pixels(text: str, origin_x: float, origin_y: float, pixel_size: float,
                       letter_spacing: float = 1.0):
    """
    Yields (px_x0, px_y0, px_x1, px_y1) rectangles, one per filled pixel,
    for rendering `text` starting at (origin_x, origin_y) (top-left corner
    of the whole text block) at the given pixel_size. Unsupported
    characters are skipped silently (rendered as blank space) rather than
    raising, since a missing glyph shouldn't crash the UI -- worst case a
    label has a gap in it, which is recoverable by adding the glyph here.
    """
    cursor_x = origin_x
    for ch in text:
        glyph = _GLYPHS.get(ch.upper())
        if glyph is None:
            cursor_x += GLYPH_COLS * pixel_size + letter_spacing * pixel_size
            continue
        for row_idx, row in enumerate(glyph):
            for col_idx, cell in enumerate(row):
                if cell == "#":
                    px_x0 = cursor_x + col_idx * pixel_size
                    px_y0 = origin_y + row_idx * pixel_size
                    yield (px_x0, px_y0, px_x0 + pixel_size, px_y0 + pixel_size)
        cursor_x += GLYPH_COLS * pixel_size + letter_spacing * pixel_size
