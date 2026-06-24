"""
gui/stepper_control.py

A small horizontal control: a "-" button, the current integer value, and
a "+" button -- replacing the draggable vertical sliders previously used
for headlamp brightness and render distance.

Why this replaced the sliders: dragging the slider handles was
unreliable for at least one person testing this (clicking the track
worked, but grabbing and dragging the handle itself did not reliably
keep the drag active between mouse-move events -- root cause never
pinned down with certainty before the decision was made to sidestep the
whole class of problem). Discrete +/-1 button clicks have no continuous
drag-tracking state to get out of sync in the first place, so this is a
structurally simpler, more robust interaction for the same underlying
adjustment (an integer value within a fixed range).

Used for two different controls, each its own StepperControl instance:
  - Headlamp brightness (0-10)
  - Render distance (1-10, chunk-radius units)

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


class StepperControl:
    BUTTON_SIZE = 32
    VALUE_BOX_WIDTH = 44
    GAP = 6

    def __init__(self, ctx: moderngl.Context, label: str, initial_value: int,
                 min_value: int, max_value: int):
        self.ctx = ctx
        self.label = label
        self.value = max(min_value, min(max_value, int(initial_value)))
        self.min_value = min_value
        self.max_value = max_value

        self.program = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)
        self._max_verts = 1200
        self._vbo = ctx.buffer(reserve=self._max_verts * 6 * 4)
        self._vao = ctx.vertex_array(
            self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
        )

    # -- value adjustment -------------------------------------------------------

    def decrement(self) -> None:
        self.value = max(self.min_value, self.value - 1)

    def increment(self) -> None:
        self.value = min(self.max_value, self.value + 1)

    # -- layout -----------------------------------------------------------------
    # anchor_x/anchor_y is the control's top-left corner -- the caller
    # (viewer_window.py) decides where that is for each instance (right
    # side for brightness, left side for render distance), this class
    # just lays out its three pieces relative to that one anchor point.

    def _minus_button_rect(self, anchor_x: float, anchor_y: float) -> tuple[float, float, float, float]:
        return (anchor_x, anchor_y, anchor_x + self.BUTTON_SIZE, anchor_y + self.BUTTON_SIZE)

    def _value_box_rect(self, anchor_x: float, anchor_y: float) -> tuple[float, float, float, float]:
        x0 = anchor_x + self.BUTTON_SIZE + self.GAP
        return (x0, anchor_y, x0 + self.VALUE_BOX_WIDTH, anchor_y + self.BUTTON_SIZE)

    def _plus_button_rect(self, anchor_x: float, anchor_y: float) -> tuple[float, float, float, float]:
        x0 = anchor_x + self.BUTTON_SIZE + self.GAP + self.VALUE_BOX_WIDTH + self.GAP
        return (x0, anchor_y, x0 + self.BUTTON_SIZE, anchor_y + self.BUTTON_SIZE)

    def total_width(self) -> float:
        return self.BUTTON_SIZE * 2 + self.VALUE_BOX_WIDTH + self.GAP * 2

    def total_height(self) -> float:
        return self.BUTTON_SIZE

    @staticmethod
    def _px_to_ndc(x: float, y: float, window_size: tuple[int, int]) -> tuple[float, float]:
        w, h = window_size
        nx = (x / w) * 2.0 - 1.0
        ny = 1.0 - (y / h) * 2.0
        return nx, ny

    # -- interaction --------------------------------------------------------------

    def on_mouse_press(self, x: float, y: float, anchor_x: float, anchor_y: float) -> bool:
        """
        Returns True if the click landed on the minus or plus button (and
        adjusted the value accordingly), False otherwise. No dragging,
        no held state at all -- a click either changes the value by
        exactly 1 right then, or it doesn't, which is the whole point of
        replacing the sliders with this.
        """
        mx0, my0, mx1, my1 = self._minus_button_rect(anchor_x, anchor_y)
        if mx0 <= x <= mx1 and my0 <= y <= my1:
            self.decrement()
            return True
        px0, py0, px1, py1 = self._plus_button_rect(anchor_x, anchor_y)
        if px0 <= x <= px1 and py0 <= y <= py1:
            self.increment()
            return True
        return False

    # -- rendering --------------------------------------------------------------

    def render(self, window_size: tuple[int, int], anchor_x: float, anchor_y: float,
               label_above: bool = True) -> None:
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

        def draw_button(rect, symbol):
            x0, y0, x1, y1 = rect
            at_limit = (symbol == "-" and self.value <= self.min_value) or \
                       (symbol == "+" and self.value >= self.max_value)
            bg = (0.18, 0.19, 0.22, 0.85) if at_limit else (0.20, 0.22, 0.26, 0.95)
            border_color = (0.4, 0.4, 0.44, 0.8) if at_limit else (0.55, 0.58, 0.62, 1.0)
            text_color = (0.45, 0.45, 0.48, 1.0) if at_limit else (0.92, 0.92, 0.95, 1.0)

            add_quad_px(x0, y0, x1, y1, bg)
            border = 2.0
            add_quad_px(x0, y0, x1, y0 + border, border_color)
            add_quad_px(x0, y1 - border, x1, y1, border_color)
            add_quad_px(x0, y0, x0 + border, y1, border_color)
            add_quad_px(x1 - border, y0, x1, y1, border_color)

            symbol_size = 2.6
            sym_w = bitmap_font.text_width_px(symbol, symbol_size)
            sym_h = bitmap_font.text_height_px(symbol_size)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            add_text(symbol, cx - sym_w / 2.0, cy - sym_h / 2.0, symbol_size, text_color)

        minus_rect = self._minus_button_rect(anchor_x, anchor_y)
        value_rect = self._value_box_rect(anchor_x, anchor_y)
        plus_rect = self._plus_button_rect(anchor_x, anchor_y)

        draw_button(minus_rect, "-")
        draw_button(plus_rect, "+")

        vx0, vy0, vx1, vy1 = value_rect
        add_quad_px(vx0, vy0, vx1, vy1, (0.10, 0.10, 0.13, 0.9))
        vborder = 2.0
        vborder_color = (0.5, 0.5, 0.55, 0.9)
        add_quad_px(vx0, vy0, vx1, vy0 + vborder, vborder_color)
        add_quad_px(vx0, vy1 - vborder, vx1, vy1, vborder_color)
        add_quad_px(vx0, vy0, vx0 + vborder, vy1, vborder_color)
        add_quad_px(vx1 - vborder, vy0, vx1, vy1, vborder_color)

        value_text = str(self.value)
        value_size = 2.4
        vw_px = bitmap_font.text_width_px(value_text, value_size)
        vh_px = bitmap_font.text_height_px(value_size)
        vcx, vcy = (vx0 + vx1) / 2.0, (vy0 + vy1) / 2.0
        add_text(value_text, vcx - vw_px / 2.0, vcy - vh_px / 2.0, value_size, (1.0, 0.92, 0.7, 1.0))

        label_size = 1.5
        label_w = bitmap_font.text_width_px(self.label, label_size)
        if label_above:
            label_x = anchor_x + (self.total_width() - label_w) / 2.0
            label_y = anchor_y - bitmap_font.text_height_px(label_size) - 8
        else:
            label_x = anchor_x
            label_y = anchor_y - bitmap_font.text_height_px(label_size) - 8
        add_text(self.label, label_x, label_y, label_size, (0.8, 0.8, 0.85, 1.0))

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
