"""
gui/controls_overlay.py

A loading overlay that doubles as a controls reference diagram, shown:
  - Full-screen, right after the OpenGL window opens, while the first
    batch of chunks around the spawn point streams in.
  - As a smaller corner panel, briefly, after a minimap click teleports
    the camera somewhere new and that area's chunks need to stream in.

Both share the same content (full control list + UI feature summary) and
the same dismiss logic (auto-hides once enough chunks have loaded that the
person can actually see the cave they're standing in) -- they differ only
in how much of the screen they cover and how prominent they are while
visible.

Like the other overlay modules (LightSlider, Minimap, RenderModeButtons),
this draws its own vector shapes + bitmap-font text, independent of the
main mesh rendering pipeline.
"""

from __future__ import annotations

import os
import time

import moderngl
import numpy as np
from PIL import Image

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

# Separate shader pair for the spinning logo: unlike every other shape
# this overlay draws (flat-colored vector triangles), the logo is a
# textured image, so it needs UV coordinates and a texture sampler rather
# than per-vertex color -- different enough from the vector-shape
# pipeline above that it gets its own tiny program rather than trying to
# force one shader to do both jobs.
_LOGO_VERT_SRC = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_LOGO_FRAG_SRC = """
#version 330
uniform sampler2D u_texture;
uniform float u_alpha;
uniform float u_brightness;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 tex_color = texture(u_texture, v_uv);
    f_color = vec4(tex_color.rgb * u_brightness, tex_color.a * u_alpha);
}
"""

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_LOGO_PATH = os.path.join(_ASSETS_DIR, "loading_logo.png")


# Each row is (label, description) -- kept as plain text pairs rather than
# trying to draw little key-cap icons, since the bitmap font is already
# legible at small sizes and an icon-per-control would need a much bigger
# glyph set than the 5x7 font currently covers.
_CONTROL_ROWS = [
    ("W A S D", "MOVE / STRAFE"),
    ("SPACE", "MOVE UP"),
    ("CTRL", "MOVE DOWN"),
    ("RIGHT MOUSE", "HOLD AND DRAG TO LOOK"),
    ("SHIFT", "SPEED BOOST"),
    ("SCROLL", "ADJUST FLY SPEED"),
    ("BRIGHTNESS +/-", "ADJUST HEADLAMP BRIGHTNESS"),
    ("GLOBAL LIGHT +/-", "ADJUST AMBIENT FILL LIGHT"),
    ("MESH BUTTON", "TOGGLE WIREFRAME"),
    ("TEXTURE BUTTON", "TOGGLE PHOTO TEXTURE"),
    ("MINIMAP CLICK", "JUMP TO THAT SPOT"),
    ("VIEW DIST +/-", "ADJUST RENDER DISTANCE"),
    ("COLOR BUTTON", "CHANGE BACKGROUND COLOR"),
    ("OPEN BUTTON", "SWITCH TO A DIFFERENT MAP"),
    ("ESC", "QUIT"),
]


class ControlsOverlay:
    # Minimum number of loaded chunks (and zero pending) before the
    # overlay is considered satisfied and starts dismissing -- chosen as
    # "enough to actually see something", not "every chunk in the load
    # radius", since waiting for a perfectly full radius on a slow machine
    # could leave the overlay up far longer than necessary.
    MIN_CHUNKS_TO_DISMISS = 6

    # How long to hold the overlay up at minimum, even if chunks finish
    # loading instantly (e.g. a very fast machine or a small nearby area
    # already cached) -- a flash-then-gone overlay is more confusing than
    # informative, so there's a floor on how long it stays visible.
    MIN_DISPLAY_SECONDS_FULLSCREEN = 1.5
    MIN_DISPLAY_SECONDS_PANEL = 0.8

    # How long the fade-out transition takes once dismiss conditions are met.
    FADE_OUT_SECONDS = 0.5

    # Hard ceiling on how long the panel variant can stay up regardless of
    # load progress -- a safety net so an unusually slow machine (or a
    # teleport into an area needing far more chunks than MIN_CHUNKS_TO_DISMISS
    # to feel "settled") doesn't leave the panel lingering indefinitely.
    # The fullscreen variant has no such ceiling since it's meant to stay
    # up until the initial spawn area is properly ready.
    MAX_DISPLAY_SECONDS_PANEL = 4.0

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)

        self._max_verts = 6000
        self._vbo = ctx.buffer(reserve=self._max_verts * 6 * 4)
        self._vao = ctx.vertex_array(
            self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
        )

        self._active = False
        self._fullscreen = True
        self._manual_mode = False
        self._start_time = 0.0
        self._fade_start_time = None

        # Spinning logo: loaded once here (a one-time disk read + GPU
        # upload, not repeated every frame) and re-drawn each frame with
        # an updated rotation angle while this overlay is visible. A
        # missing/unreadable logo file degrades to simply not drawing the
        # logo (self._logo_texture stays None) rather than crashing the
        # whole loading screen over a missing image asset.
        self._logo_texture = None
        self._logo_aspect = 1.0
        try:
            img = Image.open(_LOGO_PATH).convert("RGBA")
            self._logo_aspect = img.size[0] / img.size[1]
            tex_data = img.tobytes()
            self._logo_texture = ctx.texture(img.size, 4, tex_data)
            self._logo_texture.build_mipmaps()
        except Exception as e:
            print(f"[ControlsOverlay] WARNING: could not load spinning logo "
                  f"({_LOGO_PATH}): {e}")

        self.logo_program = ctx.program(vertex_shader=_LOGO_VERT_SRC, fragment_shader=_LOGO_FRAG_SRC)
        # 4 verts (2f pos + 2f uv) per quad, drawn as a triangle strip --
        # rewritten every frame since the rotation angle changes constantly.
        self._logo_vbo = ctx.buffer(reserve=4 * 4 * 4)
        self._logo_vao = ctx.vertex_array(
            self.logo_program, [(self._logo_vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self._spin_start_time = time.perf_counter()

    # -- lifecycle ------------------------------------------------------------

    def show_fullscreen(self) -> None:
        """Call once, right after the window opens / first chunks start streaming."""
        self._active = True
        self._fullscreen = True
        self._manual_mode = False
        self._start_time = time.perf_counter()
        self._fade_start_time = None

    def show_panel(self) -> None:
        """Call after a minimap teleport, while the new area's chunks stream in."""
        self._active = True
        self._fullscreen = False
        self._manual_mode = False
        self._start_time = time.perf_counter()
        self._fade_start_time = None

    def show_help(self) -> None:
        """
        Call when the HELP button is clicked to bring the controls list
        back up manually, mid-flight. Unlike show_fullscreen/show_panel,
        this does NOT auto-dismiss based on chunk-loading progress --
        it's the fullscreen layout (dimmed background, centered text,
        same look as the startup screen), but stays open until
        hide_help() is called (i.e. the button is clicked again), since
        there's no "loading" actually happening to wait on here.
        """
        self._active = True
        self._fullscreen = True
        self._manual_mode = True
        self._fade_start_time = None

    def hide_help(self) -> None:
        """Call when the HELP button is clicked again to close the
        manually-toggled controls screen. Immediate -- no fade-out delay,
        since this is a direct response to a click rather than something
        dismissing itself once a background condition is met."""
        if self._manual_mode:
            self._active = False
            self._manual_mode = False
            self._fade_start_time = None

    @property
    def is_manual_mode(self) -> bool:
        """True while the HELP-triggered screen is showing -- lets the
        caller (viewer_window.py) know not to call show_panel()/
        show_fullscreen() over top of it, and lets the HELP button know
        whether a click should show or hide."""
        return self._manual_mode

    def update(self, streaming_stats: dict) -> None:
        """
        Call once per frame with the StreamingWorld.stats() dict. Handles
        the auto-dismiss timing: once enough chunks are loaded, starts a
        short fade-out (after the minimum display time); once the fade
        finishes, the overlay deactivates entirely (render() becomes a
        no-op).

        The fullscreen (startup) and panel (teleport) variants use
        slightly different completion criteria:
          - Fullscreen waits for `pending == 0` too -- at startup there's
            nothing else to look at yet, so it's fine (good, even) for it
            to stay up until the initial spawn area is fully settled.
          - Panel (teleport) does NOT require pending to fully reach zero
            -- a teleport can land somewhere needing many chunks loaded
            (especially if it's a totally new, previously-uncached area of
            a large map), and waiting for every single one to finish
            would make a supposedly-brief panel linger far longer than
            intended. It dismisses once a reasonable number have loaded,
            even if some are still streaming in the background -- chunks
            keep arriving after the panel is gone, same as they always do.

        The manually-toggled HELP screen (show_help/hide_help) ignores
        this method's auto-dismiss logic entirely -- it has no loading to
        wait on, so it just stays open until explicitly hidden.
        """
        if not self._active:
            return
        if self._manual_mode:
            return

        now = time.perf_counter()
        elapsed = now - self._start_time
        min_display = self.MIN_DISPLAY_SECONDS_FULLSCREEN if self._fullscreen else self.MIN_DISPLAY_SECONDS_PANEL

        loaded = streaming_stats.get("loaded", 0)
        pending = streaming_stats.get("pending", 0)

        if self._fullscreen:
            enough_loaded = loaded >= self.MIN_CHUNKS_TO_DISMISS and pending == 0
        else:
            enough_loaded = loaded >= self.MIN_CHUNKS_TO_DISMISS or elapsed >= self.MAX_DISPLAY_SECONDS_PANEL

        if self._fade_start_time is None:
            if enough_loaded and elapsed >= min_display:
                self._fade_start_time = now
        else:
            fade_elapsed = now - self._fade_start_time
            if fade_elapsed >= self.FADE_OUT_SECONDS:
                self._active = False
                self._fade_start_time = None

    @property
    def is_active(self) -> bool:
        return self._active

    def _current_alpha_multiplier(self) -> float:
        """1.0 = fully opaque, fading down to 0.0 during the dismiss fade."""
        if self._fade_start_time is None:
            return 1.0
        fade_elapsed = time.perf_counter() - self._fade_start_time
        t = min(fade_elapsed / self.FADE_OUT_SECONDS, 1.0)
        return 1.0 - t

    # -- spinning logo ------------------------------------------------------------

    def _render_spinning_logo(self, center_x: float, center_y: float, size_px: float,
                                window_size: tuple[int, int], alpha_mult: float) -> None:
        """
        Draws the logo as a textured quad continuously tumbling around a
        VERTICAL axis through its own center -- like a coin or a playing
        card spinning in place, rather than a flat clock-hand rotation.
        The logo stays upright the whole time; only its apparent WIDTH
        changes (narrowing to a thin sliver edge-on, then widening back
        out), which is the standard, convincing way to fake a 3D spin on
        a flat 2D image without actual 3D geometry.

        Two details make this read as "3D" rather than just "squashed":
          - Past the halfway point of each half-rotation (i.e. once we'd
            be looking at the "back" of the card), the U texture
            coordinates are mirrored left-right. A real object would show
            its back face there; since this flat artwork has no separate
            back face drawn, mirroring the same image reads as "now
            looking at it from the other side" rather than an unnatural
            instant flip or a frozen sliver.
          - Brightness dips slightly as the logo narrows toward edge-on
            (where a real spinning object would catch the least light
            face-on and look darkest), then recovers as it widens back
            toward the viewer -- a cheap but effective shading cue that
            sells the 3D illusion far better than width-scaling alone.
        """
        if self._logo_texture is None:
            return

        w, h = window_size

        # continuous tumble, independent of frame rate -- driven by
        # elapsed wall-clock time rather than incrementing per-frame by a
        # fixed step, so the spin speed stays consistent regardless of
        # how fast or slow the render loop is currently running
        elapsed = time.perf_counter() - self._spin_start_time
        angle = elapsed * 1.2  # radians/second; ~ one full tumble every ~5.2s

        # half-extents in pixels, preserving the logo's own aspect ratio
        if self._logo_aspect >= 1.0:
            half_w = size_px / 2.0
            half_h = (size_px / self._logo_aspect) / 2.0
        else:
            half_h = size_px / 2.0
            half_w = (size_px * self._logo_aspect) / 2.0

        # cos(angle) is the actual "3D" part: +1 = facing the viewer head
        # on (full width), 0 = edge-on (zero width, a vertical sliver),
        # -1 = facing fully away (full width again, but mirrored -- the
        # "back" of the card). Width scales by the ABSOLUTE value so it
        # never goes negative (which would otherwise invert the quad's
        # winding and could flicker/cull oddly); the sign is handled
        # separately, below, purely for the texture-mirroring decision.
        cos_a = np.cos(angle)
        width_scale = abs(cos_a)
        facing_away = cos_a < 0.0

        scaled_half_w = half_w * width_scale

        # brightness dips toward edge-on (width_scale near 0) and recovers
        # toward face-on (width_scale near 1) -- floor it so the logo
        # never goes fully black/invisible even at the thinnest sliver
        brightness = 0.35 + 0.65 * width_scale

        local_corners = [
            (-scaled_half_w, -half_h), (scaled_half_w, -half_h),
            (scaled_half_w, half_h), (-scaled_half_w, half_h),
        ]
        # V=0 at the TOP corners and V=1 at the BOTTOM corners, matching
        # standard image coordinate convention (row 0 = top of the
        # image) -- the original mapping had this inverted, which is what
        # made the logo render upside down.
        if facing_away:
            uvs = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        else:
            uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        verts = []
        for (lx, ly), (u, v) in zip(local_corners, uvs):
            px = center_x + lx
            py = center_y + ly
            nx = (px / w) * 2.0 - 1.0
            ny = 1.0 - (py / h) * 2.0
            verts.append((nx, ny, u, v))

        # two triangles from the 4 corners (0,1,2) and (0,2,3) -- a quad
        # drawn as 6 vertices rather than a triangle strip/fan, simplest
        # to get right and consistent with how every other shape in this
        # overlay system is built
        tri_order = [0, 1, 2, 0, 2, 3]
        data = np.array([verts[i] for i in tri_order], dtype=np.float32)

        if data.nbytes > self._logo_vbo.size:
            self._logo_vbo.release()
            self._logo_vbo = self.ctx.buffer(reserve=data.nbytes)
            self._logo_vao = self.ctx.vertex_array(
                self.logo_program, [(self._logo_vbo, "2f 2f", "in_pos", "in_uv")]
            )

        self._logo_vbo.write(data.tobytes())

        self._logo_texture.use(location=0)
        self.logo_program["u_texture"].value = 0
        self.logo_program["u_alpha"].value = alpha_mult
        self.logo_program["u_brightness"].value = float(brightness)

        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self._logo_vao.render(moderngl.TRIANGLES, vertices=6)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    # -- rendering --------------------------------------------------------------

    def render(self, window_size: tuple[int, int]) -> None:
        if not self._active:
            return

        alpha_mult = self._current_alpha_multiplier()
        if alpha_mult <= 0.0:
            return

        verts = []
        w, h = window_size

        def px_to_ndc(x, y):
            nx = (x / w) * 2.0 - 1.0
            ny = 1.0 - (y / h) * 2.0
            return nx, ny

        def add_quad_px(x0, y0, x1, y1, rgba):
            r, g, b, a = rgba
            a = a * alpha_mult
            (nx0, ny0) = px_to_ndc(x0, y0)
            (nx1, ny1) = px_to_ndc(x1, y1)
            top, bottom = max(ny0, ny1), min(ny0, ny1)
            left, right = min(nx0, nx1), max(nx0, nx1)
            quad = [
                (left, bottom), (right, bottom), (right, top),
                (left, bottom), (right, top), (left, top),
            ]
            for (vx, vy) in quad:
                verts.append((vx, vy, r, g, b, a))

        def add_text(text, x, y, pixel_size, rgba):
            for (px0, py0, px1, py1) in bitmap_font.iter_text_pixels(text, x, y, pixel_size):
                add_quad_px(px0, py0, px1, py1, rgba)

        if self._fullscreen:
            logo_spec = self._build_fullscreen(add_quad_px, add_text, window_size)
        else:
            logo_spec = self._build_panel(add_quad_px, add_text, window_size)

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

        # Spinning logo draws as its own pass (separate shader/texture)
        # on top of everything just drawn above -- logo_spec is
        # (center_x, center_y, size_px), returned by whichever layout
        # builder ran, so the logo's position adapts to fullscreen vs
        # panel layout without this method needing to know those details.
        if logo_spec is not None:
            logo_cx, logo_cy, logo_size = logo_spec
            self._render_spinning_logo(logo_cx, logo_cy, logo_size, window_size, alpha_mult)

    # -- layout: full-screen variant --------------------------------------------

    def _build_fullscreen(self, add_quad_px, add_text, window_size):
        w, h = window_size

        # dim the whole 3D view behind the overlay so the text reads
        # clearly regardless of what's loaded/visible underneath
        add_quad_px(0, 0, w, h, (0.0, 0.0, 0.0, 0.78))

        # Title/subtitle differ depending on why this screen is showing:
        # the original "loading" wording makes sense at startup, but
        # would read oddly if shown manually mid-flight via the HELP
        # button (there's no actual loading happening to wait on then).
        if self._manual_mode:
            title = "CONTROLS"
            subtitle = "PRESS HELP AGAIN TO CLOSE THIS SCREEN"
        else:
            title = "LOADING CAVE"
            subtitle = "WHILE THE MAP STREAMS IN, HERE ARE THE CONTROLS"

        title_size = 4.0
        title_w = bitmap_font.text_width_px(title, title_size)
        title_x = (w - title_w) / 2.0
        title_y = h * 0.14
        add_text(title, title_x, title_y, title_size, (0.95, 0.85, 0.55, 1.0))

        sub_size = 1.8
        sub_w = bitmap_font.text_width_px(subtitle, sub_size)
        sub_x = (w - sub_w) / 2.0
        sub_y = title_y + bitmap_font.text_height_px(title_size) + 18
        add_text(subtitle, sub_x, sub_y, sub_size, (0.8, 0.8, 0.85, 1.0))

        # Compute the table's real total width from actual content (same
        # approach now used for the panel variant) so the whole table can
        # be correctly centered as a unit, rather than assuming a fixed
        # label_col_width that may not match the real longest label.
        label_size = 2.2
        desc_size = 2.2
        gap = 24
        max_label_w = max(bitmap_font.text_width_px(label, label_size) for label, _ in _CONTROL_ROWS)
        max_desc_w = max(bitmap_font.text_width_px(desc, desc_size) for _, desc in _CONTROL_ROWS)
        table_w = max_label_w + gap + max_desc_w
        table_left_x = (w - table_w) / 2.0
        label_col_right_x = table_left_x + max_label_w

        self._draw_control_table(
            add_quad_px, add_text,
            label_col_right_x=label_col_right_x,
            top_y=sub_y + bitmap_font.text_height_px(sub_size) + 36,
            label_size=label_size,
            desc_size=desc_size,
            row_height=30,
            gap=gap,
        )

        # Logo sits centered in the space above the title -- sized to fit
        # comfortably within that space (capped at a reasonable max so it
        # doesn't dominate the screen on a very tall window), and never
        # smaller than a sensible minimum on a very short window either.
        logo_area_height = title_y
        logo_size = max(60.0, min(logo_area_height * 0.7, 180.0))
        logo_cx = w / 2.0
        logo_cy = logo_area_height / 2.0
        return (logo_cx, logo_cy, logo_size)

    # -- layout: small panel variant (used after teleport) ----------------------

    def _build_panel(self, add_quad_px, add_text, window_size):
        w, h = window_size

        title = "LOADING NEW AREA"
        title_size = 2.0
        label_size = 1.3
        desc_size = 1.3
        gap = 14
        side_margin = 24

        # Panel width is DERIVED from the actual content that has to fit
        # inside it, rather than a fixed guessed constant -- this is the
        # actual fix for the reported bug: the previous fixed panel_w=340
        # was sized independently of the real label/description text
        # widths, so the description column (positioned using the same
        # center-relative formula the unbounded fullscreen layout uses)
        # could extend past the panel's right edge. Measuring the real
        # content first and sizing the box to it guarantees this can't
        # happen regardless of what the control list's text says.
        max_label_w = max(bitmap_font.text_width_px(label, label_size) for label, _ in _CONTROL_ROWS)
        max_desc_w = max(bitmap_font.text_width_px(desc, desc_size) for _, desc in _CONTROL_ROWS)
        title_w = bitmap_font.text_width_px(title, title_size)

        table_w = max_label_w + gap + max_desc_w
        panel_w = max(table_w, title_w) + 2 * side_margin

        panel_h = 30 + len(_CONTROL_ROWS) * 22 + 20
        panel_x0 = (w - panel_w) / 2.0
        panel_y0 = 40.0
        panel_x1 = panel_x0 + panel_w
        panel_y1 = panel_y0 + panel_h

        add_quad_px(panel_x0, panel_y0, panel_x1, panel_y1, (0.05, 0.05, 0.08, 0.85))

        border = 2.0
        border_color = (0.6, 0.6, 0.68, 0.9)
        add_quad_px(panel_x0, panel_y0, panel_x1, panel_y0 + border, border_color)
        add_quad_px(panel_x0, panel_y1 - border, panel_x1, panel_y1, border_color)
        add_quad_px(panel_x0, panel_y0, panel_x0 + border, panel_y1, border_color)
        add_quad_px(panel_x1 - border, panel_y0, panel_x1, panel_y1, border_color)

        title_x = panel_x0 + (panel_w - title_w) / 2.0
        title_y = panel_y0 + 12
        add_text(title, title_x, title_y, title_size, (0.95, 0.85, 0.55, 1.0))

        # label_col_width is now exactly max_label_w (plus nothing extra)
        # since the panel itself was sized to fit it precisely -- the
        # table starts right at the panel's left content edge rather than
        # being centered via the center_x-relative formula, which is what
        # let the description column drift past the panel's actual right
        # edge before.
        self._draw_control_table(
            add_quad_px, add_text,
            label_col_right_x=panel_x0 + side_margin + max_label_w,
            top_y=title_y + bitmap_font.text_height_px(title_size) + 16,
            label_size=label_size,
            desc_size=desc_size,
            row_height=18,
            gap=gap,
        )

        # Small logo to the left of the title, inside the existing panel
        # bounds -- this keeps the panel's overall size and position
        # unchanged from before this feature existed (placing it above
        # the panel would risk running off the top of the screen on a
        # short window, since the panel already starts close to y=0).
        logo_size = min(28.0, title_y + bitmap_font.text_height_px(title_size) - panel_y0 + 4)
        logo_cx = panel_x0 + 22
        logo_cy = title_y + bitmap_font.text_height_px(title_size) / 2.0
        return (logo_cx, logo_cy, logo_size)

    # -- shared control-table drawing --------------------------------------------

    def _draw_control_table(self, add_quad_px, add_text, label_col_right_x, top_y,
                              label_size, desc_size, row_height, gap):
        """
        Draws the control list as two aligned columns: right-aligned key
        labels ending exactly at `label_col_right_x`, then descriptions
        starting `gap` pixels after that -- keeps every row's description
        starting at the same X regardless of how long each label text is,
        which reads much cleaner than left-aligning both columns
        independently (labels of very different lengths would otherwise
        produce a ragged, hard-to-scan list).

        label_col_right_x is an explicit boundary passed in by the caller
        (computed from real content measurements -- see _build_fullscreen
        and _build_panel) rather than derived here from a center point and
        an assumed column width. The earlier center-relative formula could
        place the description column past the edge of whatever container
        it was being drawn into, since it had no actual awareness of that
        container's real boundaries -- this is what let the panel
        variant's description text overflow its own box.
        """
        desc_col_x = label_col_right_x + gap

        y = top_y
        for label, desc in _CONTROL_ROWS:
            label_w = bitmap_font.text_width_px(label, label_size)
            label_x = label_col_right_x - label_w
            add_text(label, label_x, y, label_size, (1.0, 0.78, 0.35, 1.0))
            add_text(desc, desc_col_x, y, desc_size, (0.88, 0.9, 0.92, 1.0))
            y += row_height
