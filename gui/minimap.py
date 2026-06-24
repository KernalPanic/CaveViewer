"""
gui/minimap.py

A small top-down minimap overlay, bottom-left of the screen, showing a
crude outline of the entire cave map's footprint (computed once from the
chunk manifest's bounding boxes -- no extra rendering pass needed) with a
red dot tracking the camera's current X/Z position live as you fly.

Deliberately crude by design: this is a top-down (X/Z plane) silhouette of
"where does the cave occupy space", not a literal rendered view. For a cave
system with real vertical complexity (multiple levels, shafts), a literal
top-down render would just show overlapping passages on top of each other
and be more confusing than helpful -- an outline of occupied footprint is
the actually-useful version of "where am I in the whole system".

Like LightSlider, this owns its own tiny 2D shader pass and geometry,
independent of the main mesh rendering pipeline.
"""

from __future__ import annotations

import moderngl
import numpy as np


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


class Minimap:
    # Layout, in pixels from the bottom-left corner of the window.
    MARGIN = 18
    PANEL_SIZE = 200       # square panel, in pixels
    DOT_RADIUS = 5
    CELL_PIXEL_SIZE = 3.0  # how big each occupied chunk-cell renders as, in panel pixels

    def __init__(self, ctx: moderngl.Context, manifest: dict):
        """
        manifest: the same chunk manifest produced by core/chunker.py
        (build_cache) / loaded via chunker.load_manifest(). Used once here
        to compute the overall footprint outline -- this does not require
        any chunks to be loaded into memory, since bounding boxes are
        already stored in the manifest itself.
        """
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)

        # Generous fixed allocation: one quad (6 verts) per occupied cell
        # for the footprint outline, plus a handful more for the dot and
        # panel background/border. Sized from the actual chunk count so
        # large maps (thousands of chunks) don't overflow a fixed guess.
        n_chunks = len(manifest.get("chunks", {}))
        self._max_verts = max(256, (n_chunks + 20) * 6)
        self._vbo = ctx.buffer(reserve=self._max_verts * 6 * 4)  # 2f pos + 4f color
        self._vao = ctx.vertex_array(
            self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
        )

        self._compute_footprint(manifest)

    # -- footprint computation (done once, at startup) -----------------------

    def _compute_footprint(self, manifest: dict) -> None:
        """
        Collapses every chunk's 3D bounding box onto the X/Z plane (top-
        down) to get a set of occupied (x, z) cell coordinates, then finds
        the overall min/max extent for scaling into the panel later. This
        is the "crude top-down outline" -- a footprint silhouette, not a
        rendered view, which is the right level of detail for a cave with
        real vertical complexity (multiple levels would just overlap and
        confuse a literal top-down render).
        """
        chunk_size = manifest["chunk_size"]
        occupied_xz = set()
        min_x = min_z = float("inf")
        max_x = max_z = float("-inf")

        for cell_str in manifest["chunks"]:
            cx, cy, cz = (int(v) for v in cell_str.split("_"))
            occupied_xz.add((cx, cz))  # collapse Y (vertical) -- top-down footprint
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_z = min(min_z, cz)
            max_z = max(max_z, cz)

        self.chunk_size = chunk_size
        self.occupied_xz = occupied_xz
        self.min_cell_x = min_x
        self.max_cell_x = max_x
        self.min_cell_z = min_z
        self.max_cell_z = max_z

        # avoid division by zero for a degenerate single-chunk map
        self._span_x = max(max_x - min_x, 1)
        self._span_z = max(max_z - min_z, 1)

    # -- coordinate mapping ---------------------------------------------------

    def _panel_rect_px(self, window_size: tuple[int, int]) -> tuple[float, float, float, float]:
        """Returns (x0, y0, x1, y1) of the minimap panel in pixel coords,
        origin top-left, anchored to the bottom-left of the window."""
        w, h = window_size
        x0 = self.MARGIN
        y1 = h - self.MARGIN
        x1 = x0 + self.PANEL_SIZE
        y0 = y1 - self.PANEL_SIZE
        return x0, y0, x1, y1

    def _world_to_panel_px(self, world_x: float, world_z: float,
                             window_size: tuple[int, int]) -> tuple[float, float]:
        """
        Maps a world X/Z position to a pixel position inside the panel,
        preserving aspect ratio (no stretching) by fitting the longer axis
        to the panel and centering the shorter axis, with a small margin
        so the dot/outline doesn't touch the panel's edge.
        """
        x0, y0, x1, y1 = self._panel_rect_px(window_size)
        inner_pad = 10
        inner_x0, inner_y0 = x0 + inner_pad, y0 + inner_pad
        inner_w = (x1 - x0) - 2 * inner_pad
        inner_h = (y1 - y0) - 2 * inner_pad

        cell_x = world_x / self.chunk_size
        cell_z = world_z / self.chunk_size

        span = max(self._span_x, self._span_z)
        # uniform scale so the footprint isn't distorted even if the cave
        # is much longer in one direction than the other
        scale = min(inner_w, inner_h) / max(span, 1e-6)

        # center the (possibly non-square) footprint within the square panel
        center_cell_x = (self.min_cell_x + self.max_cell_x) / 2.0
        center_cell_z = (self.min_cell_z + self.max_cell_z) / 2.0

        px = inner_x0 + inner_w / 2.0 + (cell_x - center_cell_x) * scale
        # panel Y grows downward; world Z growing "away" maps to panel Y
        # growing downward too, which matches typical top-down map
        # conventions (north/forward = up on the map -- but since cave
        # coordinate conventions vary, this is a reasonable default and
        # easy to flip later if it reads backwards for a given map).
        py = inner_y0 + inner_h / 2.0 + (cell_z - center_cell_z) * scale

        return px, py

    def _panel_px_to_world_xz(self, px: float, py: float,
                                window_size: tuple[int, int]) -> tuple[float, float]:
        """
        Exact algebraic inverse of _world_to_panel_px: given a pixel
        position inside the panel, returns the corresponding world (x, z)
        coordinate. Used for click-to-teleport -- the person clicks
        somewhere on the minimap, and this turns that click into an actual
        world position to fly the camera to.
        """
        x0, y0, x1, y1 = self._panel_rect_px(window_size)
        inner_pad = 10
        inner_x0, inner_y0 = x0 + inner_pad, y0 + inner_pad
        inner_w = (x1 - x0) - 2 * inner_pad
        inner_h = (y1 - y0) - 2 * inner_pad

        span = max(self._span_x, self._span_z)
        scale = min(inner_w, inner_h) / max(span, 1e-6)

        center_cell_x = (self.min_cell_x + self.max_cell_x) / 2.0
        center_cell_z = (self.min_cell_z + self.max_cell_z) / 2.0

        cell_x = center_cell_x + (px - inner_x0 - inner_w / 2.0) / scale
        cell_z = center_cell_z + (py - inner_y0 - inner_h / 2.0) / scale

        world_x = cell_x * self.chunk_size
        world_z = cell_z * self.chunk_size
        return world_x, world_z

    def hit_test(self, x: float, y: float, window_size: tuple[int, int]) -> bool:
        """True if (x, y) in pixel coords (origin top-left) lands inside
        the minimap panel."""
        x0, y0, x1, y1 = self._panel_rect_px(window_size)
        return x0 <= x <= x1 and y0 <= y <= y1

    def world_xz_for_click(self, x: float, y: float,
                             window_size: tuple[int, int]) -> tuple[float, float] | None:
        """
        Returns the world (x, z) corresponding to a click at panel pixel
        (x, y), or None if the click landed outside the panel entirely.
        Caller (viewer_window.py) combines this with the camera's current
        Y to build a full teleport target -- the minimap only knows X/Z,
        so it can't and shouldn't decide what height to land at.
        """
        if not self.hit_test(x, y, window_size):
            return None
        return self._panel_px_to_world_xz(x, y, window_size)

    @staticmethod
    def _px_to_ndc(x: float, y: float, window_size: tuple[int, int]) -> tuple[float, float]:
        w, h = window_size
        nx = (x / w) * 2.0 - 1.0
        ny = 1.0 - (y / h) * 2.0
        return nx, ny

    # -- rendering -----------------------------------------------------------

    def render(self, window_size: tuple[int, int], camera_position: np.ndarray) -> None:
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

        def add_circle_px(cx, cy, radius, rgba, segments=12):
            w, h = window_size
            for i in range(segments):
                a0 = (i / segments) * 2 * np.pi
                a1 = ((i + 1) / segments) * 2 * np.pi
                for (px, py) in [(cx, cy),
                                  (cx + radius * np.cos(a0), cy + radius * np.sin(a0)),
                                  (cx + radius * np.cos(a1), cy + radius * np.sin(a1))]:
                    nx = (px / w) * 2.0 - 1.0
                    ny = 1.0 - (py / h) * 2.0
                    verts.append((nx, ny, *rgba))

        x0, y0, x1, y1 = self._panel_rect_px(window_size)

        # panel background (semi-transparent dark, so it reads as a
        # distinct HUD element over the 3D view behind it)
        add_quad_px(x0, y0, x1, y1, (0.05, 0.05, 0.07, 0.75))

        # footprint outline: one small square per occupied chunk cell,
        # collapsed onto X/Z. Drawn in a dim cool gray so the bright red
        # position dot stands out clearly against it.
        cell_px_size = self.CELL_PIXEL_SIZE
        for (cx, cz) in self.occupied_xz:
            world_x = (cx + 0.5) * self.chunk_size
            world_z = (cz + 0.5) * self.chunk_size
            px, py = self._world_to_panel_px(world_x, world_z, window_size)
            half = cell_px_size / 2.0
            add_quad_px(px - half, py - half, px + half, py + half,
                        (0.55, 0.58, 0.65, 0.9))

        # thin border around the panel so its edges are crisp against the
        # 3D scene regardless of what's behind it
        border = 2
        add_quad_px(x0, y0, x1, y0 + border, (0.7, 0.7, 0.75, 0.9))
        add_quad_px(x0, y1 - border, x1, y1, (0.7, 0.7, 0.75, 0.9))
        add_quad_px(x0, y0, x0 + border, y1, (0.7, 0.7, 0.75, 0.9))
        add_quad_px(x1 - border, y0, x1, y1, (0.7, 0.7, 0.75, 0.9))

        # live position dot (bright red, drawn last so it's always on top
        # of the footprint outline)
        cam_px, cam_py = self._world_to_panel_px(
            float(camera_position[0]), float(camera_position[2]), window_size
        )
        add_circle_px(cam_px, cam_py, self.DOT_RADIUS, (1.0, 0.15, 0.15, 1.0))

        data = np.array(verts, dtype=np.float32)
        if data.nbytes > self._max_verts * 6 * 4:
            # safety net: grow the buffer if the map's chunk count estimate
            # was somehow exceeded, rather than truncating the draw
            self._vbo.release()
            self._max_verts = max(self._max_verts * 2, len(verts))
            self._vbo = self.ctx.buffer(reserve=self._max_verts * 6 * 4)
            self._vao = self.ctx.vertex_array(
                self.program, [(self._vbo, "2f 4f", "in_pos", "in_color")]
            )

        self._vbo.write(data.tobytes())

        # Face culling is meaningless for a flat 2D overlay (there's no
        # "back" of a UI element that should ever be hidden), but the main
        # 3D mesh pass leaves CULL_FACE enabled globally. The quad helper
        # above happens to wind consistently with the enabled cull mode, so
        # quads render fine -- but the circle/fan helper winds the opposite
        # rotational direction, so without this disable, the position dot
        # (drawn as a fan) gets silently backface-culled every frame while
        # everything else keeps rendering normally. This was the actual
        # cause of the missing red dot.
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self._vao.render(moderngl.TRIANGLES, vertices=len(verts))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
