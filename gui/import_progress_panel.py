"""
gui/import_progress_panel.py

A simple full-screen progress panel, drawn while a newly-opened map is
being imported and chunked for the FIRST time (no cache yet) -- the same
one-time cost the very first launch always pays, just now also reachable
mid-session via the OPEN button (see viewer_window.py's
_handle_open_button_click).

Important limitation, stated plainly: the actual import work (OBJ
parsing, chunk-building) runs synchronously on the main thread, the same
as it always has for the very first launch of any map. That means the
normal render loop is paused while it runs -- this panel can't animate
smoothly DURING the heavy parsing work itself, only at the discrete
progress checkpoints the parser already reports via its progress_cb
callback (see core/obj_parser.py / core/chunker.py). Each time that
callback fires, this panel redraws once and the frame is explicitly
swapped to the screen, so progress is genuinely visible as it happens,
just not as a continuously smooth animation -- an honest tradeoff against
the much larger work of moving the parser to a background thread.
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


class ImportProgressPanel:
    BAR_WIDTH = 480
    BAR_HEIGHT = 28

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)

        self._max_verts = 2000
        self._vbo = ctx.buffer(reserve=self._max_verts * 6 * 4)
        self._vao = ctx.vertex_array(
            self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
        )

    def render(self, window_size: tuple[int, int], map_name: str, stage: str, fraction: float) -> None:
        verts = []
        w, h = window_size

        def px_to_ndc(x, y):
            return (x / w) * 2.0 - 1.0, 1.0 - (y / h) * 2.0

        def add_quad_px(x0, y0, x1, y1, rgba):
            (nx0, ny0) = px_to_ndc(x0, y0)
            (nx1, ny1) = px_to_ndc(x1, y1)
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

        add_quad_px(0, 0, w, h, (0.0, 0.0, 0.0, 0.85))

        title = "IMPORTING NEW MAP"
        title_size = 3.2
        title_w = bitmap_font.text_width_px(title, title_size)
        title_y = h * 0.38
        add_text(title, (w - title_w) / 2.0, title_y, title_size, (0.95, 0.85, 0.55, 1.0))

        name_size = 1.7
        name_text = map_name.upper()
        name_w = bitmap_font.text_width_px(name_text, name_size)
        name_y = title_y + bitmap_font.text_height_px(title_size) + 16
        add_text(name_text, (w - name_w) / 2.0, name_y, name_size, (0.8, 0.8, 0.85, 1.0))

        bar_x0 = (w - self.BAR_WIDTH) / 2.0
        bar_y0 = name_y + bitmap_font.text_height_px(name_size) + 30
        bar_x1 = bar_x0 + self.BAR_WIDTH
        bar_y1 = bar_y0 + self.BAR_HEIGHT

        add_quad_px(bar_x0, bar_y0, bar_x1, bar_y1, (0.2, 0.2, 0.24, 0.95))
        fraction_clamped = max(0.0, min(1.0, fraction))
        fill_x1 = bar_x0 + fraction_clamped * self.BAR_WIDTH
        if fill_x1 > bar_x0:
            add_quad_px(bar_x0, bar_y0, fill_x1, bar_y1, (0.95, 0.7, 0.22, 1.0))
        border = 2.0
        border_color = (0.6, 0.6, 0.68, 0.9)
        add_quad_px(bar_x0, bar_y0, bar_x1, bar_y0 + border, border_color)
        add_quad_px(bar_x0, bar_y1 - border, bar_x1, bar_y1, border_color)
        add_quad_px(bar_x0, bar_y0, bar_x0 + border, bar_y1, border_color)
        add_quad_px(bar_x1 - border, bar_y0, bar_x1, bar_y1, border_color)

        stage_size = 1.6
        stage_text = f"{stage.upper()} -- {fraction_clamped*100:.0f}%"
        stage_w = bitmap_font.text_width_px(stage_text, stage_size)
        stage_y = bar_y1 + 14
        add_text(stage_text, (w - stage_w) / 2.0, stage_y, stage_size, (0.85, 0.9, 0.92, 1.0))

        note = "THIS IS A ONE-TIME COST -- FUTURE OPENS OF THIS MAP WILL BE INSTANT"
        note_size = 1.3
        note_w = bitmap_font.text_width_px(note, note_size)
        note_y = stage_y + bitmap_font.text_height_px(stage_size) + 20
        add_text(note, (w - note_w) / 2.0, note_y, note_size, (0.6, 0.6, 0.65, 1.0))

        data = np.array(verts, dtype=np.float32)
        if data.nbytes > self._max_verts * 6 * 4:
            self._vbo.release()
            self._max_verts = max(self._max_verts * 2, len(verts))
            self._vbo = self.ctx.buffer(reserve=self._max_verts * 6 * 4)
            self._vao = self.ctx.vertex_array(
                self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
            )

        self._vbo.write(data.tobytes())

        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self._vao.render(moderngl.TRIANGLES, vertices=len(verts))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
