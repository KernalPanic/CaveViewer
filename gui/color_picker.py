"""
gui/color_picker.py

A centered panel (toggled via the COLOR button, same pattern as the HELP
screen) with three horizontal sliders -- Red, Green, Blue -- for picking
the cave viewer's background ("void") color in real time, plus a live
preview swatch showing the resulting color as you drag.

Each slider covers the full 0-255 byte range for its channel. Dragging
any slider immediately updates self.color (a (r, g, b) float tuple in
0.0-1.0 range, what moderngl's ctx.clear() expects) -- viewer_window.py
reads this value each frame rather than this module reaching into the GL
context directly, keeping the same separation of concerns as every other
overlay module here (this one only knows about its own UI state, not
about moderngl-window/OpenGL specifics beyond the shader it draws with).

Like the other overlay modules, this owns its own tiny 2D shader pass and
geometry, independent of the main mesh rendering pipeline.
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


class ColorPicker:
    PANEL_WIDTH = 360
    PANEL_HEIGHT = 230
    TRACK_HEIGHT = 10
    TRACK_MARGIN_SIDE = 30
    HANDLE_WIDTH = 16
    HANDLE_HEIGHT = 24
    ROW_GAP = 56
    SWATCH_SIZE = 50

    _CHANNEL_NAMES = ["R", "G", "B"]
    _CHANNEL_COLORS = [
        (1.0, 0.35, 0.35, 1.0),
        (0.35, 0.9, 0.4, 1.0),
        (0.4, 0.55, 1.0, 1.0),
    ]

    def __init__(self, ctx: moderngl.Context, initial_color: tuple[float, float, float] = (0.02, 0.02, 0.03)):
        """
        initial_color: (r, g, b) floats in 0.0-1.0, matching the existing
        hardcoded ctx.clear() call this feature replaces -- defaults to
        the same near-black the viewer always used before this existed,
        so nothing changes for someone who never opens the picker.
        """
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)

        self._max_verts = 3000
        self._vbo = ctx.buffer(reserve=self._max_verts * 6 * 4)
        self._vao = ctx.vertex_array(
            self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
        )

        r, g, b = initial_color
        self.values = [
            int(round(max(0.0, min(1.0, r)) * 255)),
            int(round(max(0.0, min(1.0, g)) * 255)),
            int(round(max(0.0, min(1.0, b)) * 255)),
        ]

        self._active = False
        self._dragging_channel = None
        self._drag_grab_offset_x = 0.0

    @property
    def color(self) -> tuple[float, float, float]:
        """Current color as (r, g, b) floats in 0.0-1.0, what
        ctx.clear() expects -- viewer_window.py reads this every frame."""
        return tuple(v / 255.0 for v in self.values)

    @property
    def is_active(self) -> bool:
        return self._active

    def show(self) -> None:
        self._active = True

    def hide(self) -> None:
        self._active = False
        self._dragging_channel = None

    # -- layout ---------------------------------------------------------------

    def _panel_rect_px(self, window_size: tuple[int, int]) -> tuple[float, float, float, float]:
        w, h = window_size
        x0 = (w - self.PANEL_WIDTH) / 2.0
        y0 = (h - self.PANEL_HEIGHT) / 2.0
        return x0, y0, x0 + self.PANEL_WIDTH, y0 + self.PANEL_HEIGHT

    def _track_rect_px(self, channel: int, window_size: tuple[int, int]) -> tuple[float, float, float, float]:
        px0, py0, px1, py1 = self._panel_rect_px(window_size)
        track_x0 = px0 + self.TRACK_MARGIN_SIDE
        track_x1 = px1 - self.TRACK_MARGIN_SIDE
        row_top = py0 + 60 + channel * self.ROW_GAP
        track_y0 = row_top
        track_y1 = row_top + self.TRACK_HEIGHT
        return track_x0, track_y0, track_x1, track_y1

    def _handle_rect_px(self, channel: int, window_size: tuple[int, int]) -> tuple[float, float, float, float]:
        tx0, ty0, tx1, ty1 = self._track_rect_px(channel, window_size)
        t = self.values[channel] / 255.0
        handle_cx = tx0 + t * (tx1 - tx0)
        handle_cy = (ty0 + ty1) / 2.0
        return (
            handle_cx - self.HANDLE_WIDTH / 2.0,
            handle_cy - self.HANDLE_HEIGHT / 2.0,
            handle_cx + self.HANDLE_WIDTH / 2.0,
            handle_cy + self.HANDLE_HEIGHT / 2.0,
        )

    @staticmethod
    def _px_to_ndc(x: float, y: float, window_size: tuple[int, int]) -> tuple[float, float]:
        w, h = window_size
        nx = (x / w) * 2.0 - 1.0
        ny = 1.0 - (y / h) * 2.0
        return nx, ny

    # -- interaction ----------------------------------------------------------

    def _hit_test_handle(self, channel: int, x: float, y: float, window_size: tuple[int, int]) -> bool:
        hx0, hy0, hx1, hy1 = self._handle_rect_px(channel, window_size)
        pad = 6
        return (hx0 - pad) <= x <= (hx1 + pad) and (hy0 - pad) <= y <= (hy1 + pad)

    def _hit_test_track(self, channel: int, x: float, y: float, window_size: tuple[int, int]) -> bool:
        tx0, ty0, tx1, ty1 = self._track_rect_px(channel, window_size)
        pad_y = self.HANDLE_HEIGHT / 2.0
        return tx0 <= x <= tx1 and (ty0 - pad_y) <= y <= (ty1 + pad_y)

    def hit_test_panel(self, x: float, y: float, window_size: tuple[int, int]) -> bool:
        """True if (x, y) lands anywhere inside the panel's overall
        bounds -- used by viewer_window.py to know whether a click should
        be treated as 'interacting with the color picker' at all, before
        checking which specific slider it might be."""
        x0, y0, x1, y1 = self._panel_rect_px(window_size)
        return x0 <= x <= x1 and y0 <= y <= y1

    def on_mouse_press(self, x: float, y: float, window_size: tuple[int, int]) -> bool:
        """Returns True if the click was consumed by one of the three
        sliders (grabbed a handle, or clicked elsewhere on a track to
        jump there)."""
        for channel in range(3):
            if self._hit_test_handle(channel, x, y, window_size):
                self._dragging_channel = channel
                hx0, hy0, hx1, hy1 = self._handle_rect_px(channel, window_size)
                handle_cx = (hx0 + hx1) / 2.0
                self._drag_grab_offset_x = handle_cx - x
                return True
        for channel in range(3):
            if self._hit_test_track(channel, x, y, window_size):
                self._dragging_channel = channel
                self._drag_grab_offset_x = 0.0
                self._set_value_from_pixel_x(channel, x, window_size)
                return True
        return False

    def on_mouse_release(self) -> None:
        self._dragging_channel = None

    def on_mouse_drag(self, x: float, y: float, window_size: tuple[int, int]) -> bool:
        if self._dragging_channel is not None:
            self._set_value_from_pixel_x(self._dragging_channel, x + self._drag_grab_offset_x, window_size)
            return True
        return False

    @property
    def is_dragging(self) -> bool:
        return self._dragging_channel is not None

    def _set_value_from_pixel_x(self, channel: int, x: float, window_size: tuple[int, int]) -> None:
        tx0, ty0, tx1, ty1 = self._track_rect_px(channel, window_size)
        x_clamped = max(tx0, min(tx1, x))
        t = (x_clamped - tx0) / max(tx1 - tx0, 1e-6)
        self.values[channel] = int(round(max(0, min(255, t * 255))))

    # -- rendering --------------------------------------------------------------

    def render(self, window_size: tuple[int, int]) -> None:
        if not self._active:
            return

        verts = []

        def add_quad_px(x0, y0, x1, y1, rgba):
            (nx0, ny0) = self._px_to_ndc(x0, y0, window_size)
            (nx1, ny1) = self._px_to_ndc(x1, y1, window_size)
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

        px0, py0, px1, py1 = self._panel_rect_px(window_size)

        w, h = window_size
        add_quad_px(0, 0, w, h, (0.0, 0.0, 0.0, 0.55))

        add_quad_px(px0, py0, px1, py1, (0.07, 0.07, 0.10, 0.95))
        border = 2.0
        border_color = (0.6, 0.6, 0.68, 0.95)
        add_quad_px(px0, py0, px1, py0 + border, border_color)
        add_quad_px(px0, py1 - border, px1, py1, border_color)
        add_quad_px(px0, py0, px0 + border, py1, border_color)
        add_quad_px(px1 - border, py0, px1, py1, border_color)

        title = "BACKGROUND COLOR"
        title_size = 2.2
        title_w = bitmap_font.text_width_px(title, title_size)
        add_text(title, px0 + (self.PANEL_WIDTH - title_w) / 2.0, py0 + 16, title_size,
                  (0.95, 0.85, 0.55, 1.0))

        for channel in range(3):
            tx0, ty0, tx1, ty1 = self._track_rect_px(channel, window_size)
            accent = self._CHANNEL_COLORS[channel]

            add_quad_px(tx0, ty0, tx1, ty1, (0.25, 0.25, 0.28, 0.9))
            t = self.values[channel] / 255.0
            fill_x = tx0 + t * (tx1 - tx0)
            add_quad_px(tx0, ty0, fill_x, ty1, accent)

            hx0, hy0, hx1, hy1 = self._handle_rect_px(channel, window_size)
            add_quad_px(hx0, hy0, hx1, hy1, (0.92, 0.92, 0.95, 1.0))
            hborder = 1.5
            add_quad_px(hx0, hy0, hx1, hy0 + hborder, (accent[0], accent[1], accent[2], 1.0))
            add_quad_px(hx0, hy1 - hborder, hx1, hy1, (accent[0], accent[1], accent[2], 1.0))

            label_size = 1.8
            label = self._CHANNEL_NAMES[channel]
            add_text(label, tx0 - 22, (ty0 + ty1) / 2.0 - bitmap_font.text_height_px(label_size) / 2.0,
                      label_size, accent)

            value_text = str(self.values[channel])
            add_text(value_text, tx1 + 12, (ty0 + ty1) / 2.0 - bitmap_font.text_height_px(label_size) / 2.0,
                      label_size, (0.9, 0.9, 0.92, 1.0))

        swatch_x0 = px0 + (self.PANEL_WIDTH - self.SWATCH_SIZE) / 2.0
        swatch_y0 = py1 - self.SWATCH_SIZE - 16
        swatch_x1 = swatch_x0 + self.SWATCH_SIZE
        swatch_y1 = swatch_y0 + self.SWATCH_SIZE
        r, g, b = self.color
        add_quad_px(swatch_x0, swatch_y0, swatch_x1, swatch_y1, (r, g, b, 1.0))
        swatch_border = 1.5
        add_quad_px(swatch_x0, swatch_y0, swatch_x1, swatch_y0 + swatch_border, (0.8, 0.8, 0.85, 1.0))
        add_quad_px(swatch_x0, swatch_y1 - swatch_border, swatch_x1, swatch_y1, (0.8, 0.8, 0.85, 1.0))
        add_quad_px(swatch_x0, swatch_y0, swatch_x0 + swatch_border, swatch_y1, (0.8, 0.8, 0.85, 1.0))
        add_quad_px(swatch_x1 - swatch_border, swatch_y0, swatch_x1, swatch_y1, (0.8, 0.8, 0.85, 1.0))

        data = np.array(verts, dtype=np.float32)
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
