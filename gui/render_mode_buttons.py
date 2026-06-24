"""
gui/render_mode_buttons.py

Five small buttons, stacked just below the headlamp brightness slider on
the right side of the screen:
  - "Mesh"    toggles wireframe display on/off (see the actual triangle
              edges/mesh density -- useful for inspecting scan quality).
  - "Texture" toggles whether the photo texture is sampled, or the surface
              renders as a plain lit gray (useful for inspecting pure
              geometry/shape without photo detail, and is a small free
              performance win since it skips texture sampling entirely).
  - "Help"    brings the controls reference screen back up (the same
              dimmed full-screen list shown while a map is loading),
              and hides it again on a second click. Unlike Mesh/Texture,
              this button is stateless on its own -- viewer_window.py
              checks ControlsOverlay.is_manual_mode to decide whether a
              click should show or hide it, rather than this module
              tracking a separate "is help showing" flag that could drift
              out of sync with the overlay's own actual state.
  - "Color"   opens/closes the background color picker panel (see
              gui/color_picker.py). Stateless here for the same reason as
              Help -- viewer_window.py checks ColorPicker.is_active.
  - "Open"    opens the folder-browse dialog to switch to a different
              map without closing the program. Always stateless/one-shot
              -- there's no "is open mode active" toggle state, a click
              just triggers viewer_window.py's map-switch flow once.

Mesh and Texture are independent toggles (not mutually exclusive), giving
four possible combined states:
    texture only        (default)            -- normal textured view
    texture + wireframe                       -- triangulation overlaid on the photo
    wireframe only                            -- pure geometry inspection
    neither (gray, no wireframe)               -- plain lit shape, no detail at all

Like LightSlider and Minimap, this owns its own tiny 2D shader pass and
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


class RenderModeButtons:
    # Layout, in pixels. The vertical starting position is no longer
    # fixed here -- it's passed in explicitly (see _button_rect_px's
    # top_y parameter) by viewer_window.py, which is the one place that
    # knows about the brightness/render-distance controls stacked above
    # this button block and can correctly anchor the WHOLE right-side
    # column (controls + buttons together) from a single position,
    # currently the bottom-right corner -- see
    # CaveViewerWindow._right_column_layout().
    BUTTON_WIDTH = 110
    BUTTON_HEIGHT = 34
    BUTTON_GAP = 10
    MARGIN_RIGHT = 18

    def __init__(self, ctx: moderngl.Context,
                 texture_enabled: bool = True, wireframe_enabled: bool = False):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)

        self.texture_enabled = texture_enabled
        self.wireframe_enabled = wireframe_enabled

        # Sized generously for three buttons' worth of: background, drop
        # shadow, top highlight strip, 4 border edges, and full text
        # labels (the longer "TEXTURE" at up to ~7 letters * 35 filled
        # pixels each, worst case) -- comfortably covers the redesigned
        # visuals (shadow + highlight added beyond the original flat
        # rectangle) without needing the auto-grow fallback on first render.
        self._max_verts = 3000
        self._vbo = ctx.buffer(reserve=self._max_verts * 6 * 4)
        self._vao = ctx.vertex_array(
            self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
        )

    # -- layout ---------------------------------------------------------------

    @classmethod
    def total_stack_height(cls, scale: float = 1.0) -> float:
        """Full height of all 5 buttons + gaps between them, at a given
        scale factor -- used by viewer_window.py to figure out how much
        vertical room this whole block needs when laying out the
        bottom-anchored right-side column."""
        n_buttons = 5
        return n_buttons * (cls.BUTTON_HEIGHT * scale) + (n_buttons - 1) * (cls.BUTTON_GAP * scale)

    def _button_rect_px(self, index: int, window_size: tuple[int, int], top_y: float) -> tuple[float, float, float, float]:
        """Returns (x0, y0, x1, y1) for button `index` (0=Mesh, 1=Texture, 2=Help, 3=Color, 4=Open).
        top_y is where the FIRST button (Mesh) starts -- passed in by the
        caller, which owns the overall column layout."""
        w, h = window_size

        n_buttons = 5
        full_stack_height = n_buttons * self.BUTTON_HEIGHT + (n_buttons - 1) * self.BUTTON_GAP

        # If the full stack (starting at `top_y`) would run past the
        # bottom of the window, shrink the per-button height and gap
        # proportionally so all five buttons stay visible and clickable,
        # rather than letting later buttons (Color, Open) run off-screen
        # on a short window. The floor here is deliberately low (0.35,
        # not the 0.5 used when there were fewer buttons) -- a HIGHER
        # floor than what's actually needed to fit defeats the entire
        # point of this calculation: at 640x480 with 5 buttons, the floor
        # was clamping the scale UP to a value that still overflowed the
        # window, which was a real, reproducible bug caught by testing
        # this exact window size after adding the 5th button. The floor
        # only matters at truly extreme window sizes (smaller than any
        # realistic usage) where some shrinkage is an acceptable
        # tradeoff against buttons being literally unreachable.
        available_height = h - top_y - 10  # 10px bottom breathing room
        scale = 1.0
        if full_stack_height > available_height and available_height > 0:
            scale = max(0.35, available_height / full_stack_height)

        button_h = self.BUTTON_HEIGHT * scale
        button_gap = self.BUTTON_GAP * scale

        x1 = w - self.MARGIN_RIGHT
        x0 = x1 - self.BUTTON_WIDTH
        y0 = top_y + index * (button_h + button_gap)
        y1 = y0 + button_h
        return x0, y0, x1, y1

    def _mesh_button_rect(self, window_size, top_y):
        return self._button_rect_px(0, window_size, top_y)

    def _texture_button_rect(self, window_size, top_y):
        return self._button_rect_px(1, window_size, top_y)

    def _help_button_rect(self, window_size, top_y):
        return self._button_rect_px(2, window_size, top_y)

    def _color_button_rect(self, window_size, top_y):
        return self._button_rect_px(3, window_size, top_y)

    def _open_button_rect(self, window_size, top_y):
        return self._button_rect_px(4, window_size, top_y)

    @staticmethod
    def _px_to_ndc(x: float, y: float, window_size: tuple[int, int]) -> tuple[float, float]:
        w, h = window_size
        nx = (x / w) * 2.0 - 1.0
        ny = 1.0 - (y / h) * 2.0
        return nx, ny

    # -- interaction ------------------------------------------------------------

    def hit_test_mesh(self, x: float, y: float, window_size: tuple[int, int], top_y: float) -> bool:
        x0, y0, x1, y1 = self._mesh_button_rect(window_size, top_y)
        return x0 <= x <= x1 and y0 <= y <= y1

    def hit_test_texture(self, x: float, y: float, window_size: tuple[int, int], top_y: float) -> bool:
        x0, y0, x1, y1 = self._texture_button_rect(window_size, top_y)
        return x0 <= x <= x1 and y0 <= y <= y1

    def hit_test_help(self, x: float, y: float, window_size: tuple[int, int], top_y: float) -> bool:
        x0, y0, x1, y1 = self._help_button_rect(window_size, top_y)
        return x0 <= x <= x1 and y0 <= y <= y1

    def hit_test_color(self, x: float, y: float, window_size: tuple[int, int], top_y: float) -> bool:
        x0, y0, x1, y1 = self._color_button_rect(window_size, top_y)
        return x0 <= x <= x1 and y0 <= y <= y1

    def hit_test_open(self, x: float, y: float, window_size: tuple[int, int], top_y: float) -> bool:
        x0, y0, x1, y1 = self._open_button_rect(window_size, top_y)
        return x0 <= x <= x1 and y0 <= y <= y1

    def on_mouse_press(self, x: float, y: float, window_size: tuple[int, int], top_y: float) -> str | None:
        """
        Returns a string identifying which button was clicked ("mesh",
        "texture", "help", "color", or "open"), or None if the click
        missed all five -- the caller (viewer_window.py) acts on the
        result, since Help/Color/Open's actual behavior depends on state
        this module doesn't have access to (see the module docstring for
        why they're intentionally stateless here). top_y is where this
        button block starts -- see total_stack_height()'s docstring for
        why the caller, not this class, owns that position.
        """
        if self.hit_test_mesh(x, y, window_size, top_y):
            self.wireframe_enabled = not self.wireframe_enabled
            return "mesh"
        if self.hit_test_texture(x, y, window_size, top_y):
            self.texture_enabled = not self.texture_enabled
            return "texture"
        if self.hit_test_help(x, y, window_size, top_y):
            return "help"
        if self.hit_test_color(x, y, window_size, top_y):
            return "color"
        if self.hit_test_open(x, y, window_size, top_y):
            return "open"
        return None

    # -- rendering --------------------------------------------------------------

    def render(self, window_size: tuple[int, int], top_y: float, help_active: bool = False,
               color_active: bool = False) -> None:
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

        # Pick ONE text pixel_size that fits the longer label ("TEXTURE"),
        # then use that same size for BOTH buttons. Previously each button
        # auto-fit its own label independently, so "MESH" (short) rendered
        # noticeably larger/differently-proportioned than "TEXTURE" (long)
        # sitting right next to it -- two adjacent buttons with
        # inconsistent text sizing is a large part of what reads as
        # unpolished. A single shared size keeps them visually matched.
        available_w = self.BUTTON_WIDTH - 16
        available_h = self.BUTTON_HEIGHT - 12
        shared_pixel_size = 2.4
        while shared_pixel_size > 0.5:
            w = bitmap_font.text_width_px("TEXTURE", shared_pixel_size)
            h = bitmap_font.text_height_px(shared_pixel_size)
            if w <= available_w and h <= available_h:
                break
            shared_pixel_size -= 0.1

        def draw_toggle_button(rect, is_on: bool, label: str):
            x0, y0, x1, y1 = rect

            # Soft drop shadow: a slightly offset, darker, larger rect
            # behind the button gives a sense of depth/elevation instead
            # of a flat color block sitting directly on the 3D view --
            # drawn first so everything else layers on top of it.
            shadow_offset = 3
            add_quad_px(x0 + shadow_offset, y0 + shadow_offset,
                        x1 + shadow_offset, y1 + shadow_offset,
                        (0.0, 0.0, 0.0, 0.35))

            # Button face: warm amber when active, cool dark slate when
            # inactive -- higher contrast than before between the two
            # states, and a less muddy "off" color (slate-blue-gray reads
            # more deliberately "inactive" than a plain flat gray).
            if is_on:
                bg = (0.95, 0.70, 0.22, 1.0)
            else:
                bg = (0.16, 0.18, 0.22, 0.95)
            add_quad_px(x0, y0, x1, y1, bg)

            # A thin brighter strip along the top edge of the button face
            # simulates a subtle highlight/bevel, the cheapest way to make
            # a flat-shaded rectangle read as a slightly raised, tactile
            # button rather than a painted-on color swatch.
            highlight_h = 3
            if is_on:
                highlight_color = (1.0, 0.85, 0.55, 0.9)
            else:
                highlight_color = (0.32, 0.35, 0.40, 0.8)
            add_quad_px(x0, y0, x1, y0 + highlight_h, highlight_color)

            # Crisp outer border, thicker than before (1.5px was too thin
            # to read clearly as a button edge) -- brighter and thicker
            # when active so the "on" state is unmistakable even at a
            # glance from across the room.
            border = 2.5 if is_on else 2.0
            border_color = (1.0, 1.0, 0.98, 1.0) if is_on else (0.45, 0.48, 0.55, 1.0)
            add_quad_px(x0, y0, x1, y0 + border, border_color)
            add_quad_px(x0, y1 - border, x1, y1, border_color)
            add_quad_px(x0, y0, x0 + border, y1, border_color)
            add_quad_px(x1 - border, y0, x1, y1, border_color)

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            text_color = (0.12, 0.10, 0.05, 1.0) if is_on else (0.88, 0.90, 0.95, 1.0)

            text_w = bitmap_font.text_width_px(label, shared_pixel_size)
            text_h = bitmap_font.text_height_px(shared_pixel_size)
            origin_x = cx - text_w / 2.0
            origin_y = cy - text_h / 2.0

            for (px0, py0, px1, py1) in bitmap_font.iter_text_pixels(label, origin_x, origin_y, shared_pixel_size):
                add_quad_px(px0, py0, px1, py1, text_color)

        draw_toggle_button(self._mesh_button_rect(window_size, top_y), self.wireframe_enabled, "MESH")
        draw_toggle_button(self._texture_button_rect(window_size, top_y), self.texture_enabled, "TEXTURE")
        draw_toggle_button(self._help_button_rect(window_size, top_y), help_active, "HELP")
        draw_toggle_button(self._color_button_rect(window_size, top_y), color_active, "COLOR")
        draw_toggle_button(self._open_button_rect(window_size, top_y), False, "OPEN")

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
