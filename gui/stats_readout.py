"""
gui/stats_readout.py

A small text readout showing live FPS and chunk-loading stats (loaded /
pending), positioned directly above the minimap panel in the bottom-left
corner. This is the same information the console already prints every
couple seconds (see viewer_window.py's periodic "[CaveViewer] X fps |
chunks loaded=..." line), surfaced on-screen too -- it used to live next
to the old render-distance slider before that was replaced with a
plain +/- stepper button (which has no natural place to attach a readout
the way a slider's track did), so it's now its own small standalone
overlay instead.

Updates every frame using a smoothed rolling average (the same
_frame_time_history the spike-detector already maintains), rather than
only refreshing every 2 seconds like the console line -- a number that
visibly updates in real time is much more useful for judging the effect
of something you just changed (e.g. just bumped the render distance
stepper) than one that lags behind by up to 2 seconds.

Same drawing pattern as every other 2D overlay module here: a small
self-contained shader pass, independent of the main mesh rendering
pipeline.
"""

from __future__ import annotations

import moderngl
import numpy as np

from gui import bitmap_font


_VERT_SRC = """
#version 330
in vec2 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_color = in_color;
}
"""

_FRAG_SRC = """
#version 330
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = v_color;
}
"""


class StatsReadout:
    TEXT_SIZE = 1.6
    LINE_GAP = 4

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)
        self._max_verts = 800
        self._vbo = ctx.buffer(reserve=self._max_verts * 6 * 4)
        self._vao = ctx.vertex_array(
            self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
        )

    def total_height(self) -> float:
        """Total vertical space this readout occupies (two lines of text
        + the gap between them) -- used by the caller to know how much
        room to reserve above the minimap panel."""
        line_h = bitmap_font.text_height_px(self.TEXT_SIZE)
        return line_h * 2 + self.LINE_GAP

    def render(self, window_size: tuple[int, int], bottom_left_x: float, bottom_y: float,
               fps: float, chunks_loaded: int, chunks_pending: int) -> None:
        """
        Draws two lines of text -- "FPS xx.x" and "CHUNKS n (m pending)"
        -- with their BOTTOM-left corner at (bottom_left_x, bottom_y), so
        the caller can position this readout's bottom edge flush against
        the top of the minimap panel without needing to know this
        class's exact text-height internals first.
        """
        verts = []

        def add_quad_px(x0, y0, x1, y1, rgba):
            w, h = window_size
            nx0, ny0 = (x0 / w) * 2.0 - 1.0, 1.0 - (y0 / h) * 2.0
            nx1, ny1 = (x1 / w) * 2.0 - 1.0, 1.0 - (y1 / h) * 2.0
            top, bottom = max(ny0, ny1), min(ny0, ny1)
            left, right = min(nx0, nx1), max(nx0, nx1)
            quad = [
                (left, bottom), (right, bottom), (right, top),
                (left, bottom), (right, top), (left, top),
            ]
            for (x, y) in quad:
                verts.append((x, y, *rgba))

        def add_text(text, x, y, pixel_size, rgba):
            for (px0, py0, px1, py1) in bitmap_font.iter_text_pixels(text, x, y, pixel_size):
                add_quad_px(px0, py0, px1, py1, rgba)

        line_h = bitmap_font.text_height_px(self.TEXT_SIZE)

        chunks_line_y = bottom_y - line_h
        fps_line_y = chunks_line_y - self.LINE_GAP - line_h

        fps_text = f"FPS {fps:.0f}"
        chunks_text = f"CHUNKS {chunks_loaded}"
        if chunks_pending > 0:
            chunks_text += f" ({chunks_pending} PENDING)"

        fps_color = (0.7, 0.85, 0.7, 1.0) if fps >= 15.0 else (0.95, 0.75, 0.4, 1.0)
        chunks_color = (0.7, 0.8, 0.9, 1.0)

        add_text(fps_text, bottom_left_x, fps_line_y, self.TEXT_SIZE, fps_color)
        add_text(chunks_text, bottom_left_x, chunks_line_y, self.TEXT_SIZE, chunks_color)

        data = np.array(verts, dtype=np.float32)
        if len(verts) == 0:
            return
        if data.nbytes > self._max_verts * 6 * 4:
            self._vbo.release()
            self._max_verts = max(self._max_verts * 2, len(verts))
            self._vbo = self.ctx.buffer(reserve=self._max_verts * 6 * 4)
            self._vao = self.ctx.vertex_array(
                self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
            )

        self._vbo.write(data.tobytes())

        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self._vao.render(moderngl.TRIANGLES, vertices=len(verts))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
