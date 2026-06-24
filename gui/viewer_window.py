"""
gui/viewer_window.py

The actual OpenGL window: owns the moderngl context, the free-fly camera,
the StreamingWorld (which decides what to load/unload), and the per-chunk
GPU buffers/textures. This is where everything else in core/ and gui/
gets wired together into a runnable program.

Each loaded chunk becomes a small set of moderngl VAOs, one per material
group within that chunk (so each can be drawn with its own bound texture).
We keep a dict: cell -> list[(vao, texture_material_name)] so unload is a
simple lookup-and-release.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window.context.base import KeyModifiers

from core import chunker
from core.streaming_world import StreamingWorld, StreamingConfig
from core.texture_manager import TextureManager
from gui.camera import FlyCamera
from gui.minimap import Minimap
from gui.render_mode_buttons import RenderModeButtons
from gui.controls_overlay import ControlsOverlay
from gui.stepper_control import StepperControl
from gui.color_picker import ColorPicker
from gui.import_progress_panel import ImportProgressPanel
from gui.stats_readout import StatsReadout

def _resource_base_dir() -> str:
    """
    Returns the correct base directory to resolve bundled resources (like
    the shaders/ folder) from, whether running normally from source or
    packaged into a standalone executable via PyInstaller.

    When PyInstaller builds a frozen executable, bundled data files are
    extracted to a temporary directory at runtime, exposed via
    `sys._MEIPASS` -- NOT the directory containing this .py file (which,
    in a frozen build, doesn't really exist as a normal file on disk at
    all). Checking for `sys.frozen` is the standard way to detect this and
    branch accordingly; see build_exe.py / CaveViewer.spec for the matching
    PyInstaller config that actually places shaders/ at the right spot
    inside the bundle.
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SHADER_DIR = os.path.join(_resource_base_dir(), "shaders")


class CaveViewerWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "CaveViewer 1.0"
    window_size = (1280, 800)
    resizable = True
    vsync = True
    aspect_ratio = None  # don't letterbox; we recompute from actual window size

    # Set on the class itself (not passed through __init__ kwargs) before
    # calling mglw.run_window_config(). Different moderngl-window versions
    # have changed how/whether run_window_config forwards extra keyword
    # arguments into WindowConfig.__init__, so relying on that passthrough
    # is fragile across versions. Class attributes are a stable mechanism
    # regardless of moderngl-window's internal arg handling -- run_viewer()
    # at the bottom of this file sets these right before launching.
    cave_cache_dir: str = None
    cave_textures_dir: str = None
    cave_manifest: dict = None

    # Alternative to the three attributes above: set THIS instead when the
    # map needs first-time import/chunking (no cache built yet) -- a dict
    # with keys "obj_path", "mtl_path", "textures_dir". When set, the
    # window opens immediately with no map loaded, and the actual import
    # runs from inside on_render()'s first frame (see _run_pending_import),
    # so the existing in-window ImportProgressPanel can show real progress
    # the same way it already does for the OPEN button's mid-session
    # imports -- rather than the old behavior of running the import
    # entirely before any window existed, which could only show a plain
    # console progress bar with nowhere graphical to draw into yet.
    cave_pending_import: dict = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        have_ready_cache = CaveViewerWindow.cave_cache_dir is not None
        have_pending_import = CaveViewerWindow.cave_pending_import is not None

        if not have_ready_cache and not have_pending_import:
            raise RuntimeError(
                "Neither CaveViewerWindow.cave_cache_dir nor .cave_pending_import "
                "was set before launch. One or the other must be set by "
                "run_viewer() / run_viewer_with_pending_import() before "
                "constructing this window."
            )

        with open(os.path.join(SHADER_DIR, "mesh.vert")) as f:
            vert_src = f.read()
        with open(os.path.join(SHADER_DIR, "mesh.frag")) as f:
            frag_src = f.read()
        self.program = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        self._keys_down = set()
        self._mouse_look_active = False
        self._last_mouse_pos = None
        self._frame_count = 0
        self._last_fps_print = time.time()
        self._frame_time_history: list[float] = []

        # Headlamp brightness control: a -/value/+ stepper, right side of
        # the screen. Replaced a draggable vertical slider -- dragging the
        # handle was unreliable for at least one person testing this
        # (clicking the track worked, grabbing the handle to drag did
        # not), so this sidesteps the whole class of problem by using
        # discrete +/-1 clicks instead of continuous drag-tracking.
        # Range/default unchanged from the old slider (0-10, default 3).
        self.light_stepper = StepperControl(self.ctx, "BRIGHTNESS", initial_value=3, min_value=0, max_value=10)

        # Render distance control: a -/value/+ stepper, left side of the
        # screen, mirroring the brightness control's placement logic but
        # on the opposite side. Directly drives
        # self.world.config.load_radius_cells live, same as the slider it
        # replaced. Range/default unchanged (1-10 chunk-radius units,
        # default 4). StreamingWorld's max_loaded_chunks safety valve
        # (see core/streaming_world.py) still applies underneath this as
        # a hard backstop regardless of what this is set to.
        self.render_distance_stepper = StepperControl(
            self.ctx, "VIEW DIST", initial_value=4, min_value=1, max_value=10
        )

        # "Global illumination" control: not actual simulated light
        # bouncing (a much bigger rendering undertaking), but an even
        # ambient fill light across the WHOLE cave, independent of the
        # headlamp -- raising this washes out shadows so the cave reads
        # clearly without the headlamp doing all the work, similar to
        # what people commonly mean by a one-button "GI toggle" in
        # smaller tools. Range 0-10 maps to the shader's u_ambient float
        # (see _AMBIENT_MIN/_AMBIENT_MAX below) -- 0 reproduces the
        # original fixed ambient value this app always used (0.04, a
        # tiny fill so unlit areas aren't pure black), so leaving this at
        # its default changes nothing from before this feature existed.
        self.ambient_stepper = StepperControl(self.ctx, "GLOBAL LIGHT", initial_value=0, min_value=0, max_value=10)

        # Mesh/Texture toggle buttons, stacked just below the brightness
        # slider. Mesh = wireframe overlay on/off; Texture = whether the
        # photo texture is sampled or the surface falls back to plain lit
        # gray. See gui/render_mode_buttons.py for the four resulting
        # combined display states.
        self.render_mode_buttons = RenderModeButtons(self.ctx, texture_enabled=True, wireframe_enabled=False)

        # Controls reference / loading overlay -- full-screen right now
        # while the first chunks around the spawn point stream in, and
        # again as a smaller panel any time a minimap click teleports the
        # camera somewhere new (see on_mouse_press_event's minimap-click
        # handling, which calls self.controls_overlay.show_panel()).
        self.controls_overlay = ControlsOverlay(self.ctx)
        self.controls_overlay.show_fullscreen()

        # Background ("void") color picker, toggled via the COLOR button.
        # Defaults to the same near-black the viewer always used, so
        # nothing changes for anyone who never opens it.
        self.color_picker = ColorPicker(self.ctx, initial_color=(0.02, 0.02, 0.03))

        # Shown only while a newly-opened map is being imported/chunked
        # for the first time (see _handle_open_button_click) -- never
        # active during normal viewing, so it has no on/off state of its
        # own the way the other overlays do.
        self.import_progress_panel = ImportProgressPanel(self.ctx)

        # Live FPS / chunk-loading readout, positioned above the minimap.
        # Map-independent (no per-map state), so it's set up once here
        # rather than rebuilt every time _load_map() runs for a new map.
        self.stats_readout = StatsReadout(self.ctx)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        # Map-specific state (world, manifest, camera, minimap, texture
        # manager, chunk GPU objects) lives in its own method, separate
        # from the one-time-per-window setup above, so the exact same
        # logic can run again later when switching to a different map via
        # the OPEN button -- see load_new_map() / _teardown_current_map().
        self._chunk_gpu_objects: dict[tuple, list] = {}
        self._has_map_loaded = False
        self._pending_import_started = False

        if have_ready_cache:
            self._load_map(
                CaveViewerWindow.cave_cache_dir,
                CaveViewerWindow.cave_textures_dir,
                CaveViewerWindow.cave_manifest,
            )
            self._has_map_loaded = True
        # else: have_pending_import is true instead -- the actual import
        # is deliberately NOT run here, before the window has rendered
        # even one frame. It's triggered from inside on_render() instead
        # (see _run_pending_import), once the window is confirmed to
        # actually be open and able to draw the in-window progress panel
        # -- starting the blocking import here, before super().__init__()
        # has truly finished and the window is on screen, would risk the
        # exact same "nothing to draw into yet" problem this feature
        # exists to avoid.

    def _load_map(self, cache_dir: str, textures_dir: str, manifest: dict) -> None:
        """
        Sets up everything specific to ONE map: the texture manager, the
        streaming world, the starting camera position, and the minimap.
        Called once from __init__ for the map the program launched with,
        and called again from load_new_map() when switching to a
        different map via the OPEN button -- _teardown_current_map() must
        be called first in that second case, to cleanly release the
        previous map's GPU/thread resources before this builds new ones.
        """
        self.cache_dir = cache_dir
        self.textures_dir = textures_dir
        self.manifest = manifest

        self.texture_manager = TextureManager(self.ctx, self.textures_dir, self.manifest["mtl_materials"])

        def predecode_textures_for_chunk(chunk_data):
            # Called from a background worker thread (see StreamingWorld) --
            # decodes JPEGs for every material this chunk uses, ahead of
            # time, so the eventual main-thread GPU upload in
            # _on_chunk_ready is just a fast texture() call on already-
            # decoded pixels rather than a slow decode-and-upload combined.
            for mat_name in chunk_data.groups.keys():
                self.texture_manager.decode_for_material(mat_name)

        chunk_size = self.manifest["chunk_size"]
        config = StreamingConfig(chunk_size=chunk_size, load_radius_cells=4, unload_radius_margin=1)
        self.world = StreamingWorld(self.cache_dir, config, on_decode_textures=predecode_textures_for_chunk)

        # pick a sane starting position: center of the first available chunk,
        # so the user doesn't spawn outside the mesh and see nothing
        first_cell_str = next(iter(self.manifest["chunks"]))
        first_info = self.manifest["chunks"][first_cell_str]
        start_pos = (np.array(first_info["bounds_min"]) + np.array(first_info["bounds_max"])) / 2.0
        self.camera = FlyCamera(position=tuple(start_pos))

        # Bottom-left minimap: a crude top-down outline of the whole cave's
        # footprint with a live red dot for current position. Built once
        # from the manifest's chunk bounding boxes -- no extra rendering
        # pass or GPU cost beyond this tiny 2D overlay.
        self.minimap = Minimap(self.ctx, self.manifest)

        # Render-distance slider's current value should drive the new
        # map's streaming config immediately, rather than resetting back
        # to the control's own default -- if someone already turned it up
        # for a previous large map, opening another large map shouldn't
        # silently reset that preference. (On first launch, from
        # __init__, this just re-applies the control's own initial value,
        # a harmless no-op.)
        if hasattr(self, "render_distance_stepper"):
            self.world.config.load_radius_cells = self.render_distance_stepper.value

        self.controls_overlay.show_fullscreen()

    def _teardown_current_map(self) -> None:
        """
        Cleanly releases everything specific to the CURRENTLY loaded map
        before _load_map() builds a new one -- stops StreamingWorld's
        background threads and waits for them to actually exit, then
        releases every currently-resident chunk's GPU buffers/VAOs and
        decrements the texture manager's reference counts via the exact
        same _on_chunk_unload() path used during normal streaming (so
        there's no separate cleanup logic to keep in sync with the
        regular unload path). The texture manager itself is then simply
        discarded -- a fresh one is constructed for the new map rather
        than trying to partially reuse the old one.

        Safe to call even if no map was ever loaded yet (e.g. the very
        first import, triggered from _run_pending_import, completing for
        the first time rather than switching away from an existing map)
        -- there's nothing to tear down in that case, so this just
        returns immediately rather than crashing on self.world not
        existing yet.
        """
        if not self._has_map_loaded:
            return

        self.world.shutdown()

        for cell in list(self._chunk_gpu_objects.keys()):
            self._on_chunk_unload(cell)

        # belt-and-suspenders: if anything was somehow left behind (it
        # shouldn't be, given the loop above), don't carry it into the
        # next map's state
        self._chunk_gpu_objects.clear()

    def load_new_map(self, cache_dir: str, textures_dir: str, manifest: dict) -> None:
        """
        Switches the viewer to a different map without closing the
        window -- called by the OPEN button's click handler once a new
        folder has been picked and imported/cached (see
        caveviewer.py's find_input_files/import_and_cache, reused as-is
        rather than duplicated here).

        Order matters: tear down the OLD map's GPU/thread state fully
        before constructing any NEW state, rather than interleaving the
        two -- this guarantees the old map's resources are genuinely
        released (not just about to be overwritten by Python references
        moving on, which would leak the GPU-side buffers/textures since
        those aren't cleaned up by garbage collection alone).
        """
        self._teardown_current_map()
        self._load_map(cache_dir, textures_dir, manifest)
        self._has_map_loaded = True

    def _handle_open_button_click(self) -> None:
        """
        Full OPEN button flow: shows the folder-browse dialog (same one
        used at startup), detects which supported format (.obj or
        .glb) the selected folder contains, imports/caches it if there's
        no valid cache yet (showing the progress panel while that one-
        time work runs), and finally calls load_new_map() to actually
        switch.

        Any failure along the way (cancelled dialog, no supported model
        file found, import error) prints a clear message and leaves the
        CURRENTLY loaded map running untouched -- a failed attempt to
        open a different map should never take down the map you already
        had open and were presumably still looking at.
        """
        # Local imports here (not at module top) since these pull in
        # tkinter and the parser/chunker modules, which the rest of this
        # file doesn't otherwise need -- same reasoning caveviewer.py
        # already uses for its own local imports of these.
        from caveviewer import pick_folder_dialog, find_model_file, import_and_cache_any
        from core import chunker as chunker_module

        folder = pick_folder_dialog()
        if not folder:
            print("[CaveViewer] Open cancelled -- no folder selected.")
            return

        folder = os.path.abspath(folder)
        print(f"[CaveViewer] Opening new map from: {folder}")

        try:
            model_descriptor = find_model_file(folder)
        except FileNotFoundError as e:
            print(f"[CaveViewer] Could not open this folder: {e}")
            return

        source_path = model_descriptor.get("obj_path") or model_descriptor.get("glb_path")
        map_name = os.path.basename(source_path)

        # If there's no valid cache yet, this is the same one-time import
        # cost as opening any brand-new map for the first time -- show
        # the progress panel so it's visible what's happening rather than
        # the window appearing to freeze with no explanation.
        already_cached = chunker_module.cache_is_valid(source_path)

        def on_progress(stage: str, fraction: float):
            self.import_progress_panel.render(self.wnd.size, map_name, stage, fraction)
            # Explicitly push this frame to the screen -- the normal
            # render loop is paused while import_and_cache_any() runs
            # synchronously below, so without this, nothing drawn here
            # would actually become visible until the import finishes
            # and the next regular frame happens to render.
            #
            # swap_buffers() is moderngl-window's standard, long-standing
            # method for this and should be present on any version in
            # use -- but since this project has already hit real
            # cross-version API differences before (see _resolve_key,
            # and the render()/on_render() hook rename), this is wrapped
            # defensively rather than assumed: if swap_buffers truly
            # isn't there on some version, ctx.finish() at least forces
            # the GPU to complete the draw rather than crashing outright,
            # even though it can't guarantee the frame reaches the
            # screen without a real swap.
            if hasattr(self.wnd, "swap_buffers"):
                self.wnd.swap_buffers()
            else:
                self.ctx.finish()

        try:
            if not already_cached:
                on_progress("starting import", 0.0)
                cache_dir = import_and_cache_any(model_descriptor, folder, force_rebuild=False,
                                                   extra_progress_cb=on_progress)
            else:
                cache_dir = chunker_module.get_cache_dir(source_path)
        except Exception as e:
            print(f"[CaveViewer] Failed to import this map: {e}")
            return

        try:
            new_manifest = chunker_module.load_manifest(cache_dir)
        except Exception as e:
            print(f"[CaveViewer] Failed to load the new map's manifest after import: {e}")
            return

        print(f"[CaveViewer] Switching to: {map_name}")
        self.load_new_map(cache_dir, folder, new_manifest)
        print(f"[CaveViewer] Now viewing: {map_name}")

    def _run_pending_import(self) -> None:
        """
        Runs the FIRST-TIME import for the map the program was launched
        with, when CaveViewerWindow.cave_pending_import was set instead
        of an already-built cache (see run_viewer_with_pending_import()
        at the bottom of this file, and main()'s use of it in
        caveviewer.py). Called once, from on_render()'s first frame --
        see the _has_map_loaded branch there for why it's deferred to
        that point rather than running before the window even opens.

        Format-agnostic: works the same regardless of whether the
        pending import is an .obj or .glb (see
        caveviewer.py's find_model_file()/import_and_cache_any(), which
        this delegates the actual format-specific parsing to) -- this
        method only deals with the progress-panel/window-lifecycle side
        of things, not anything about the source format itself.

        Shares the exact same import-with-progress-panel approach as
        _handle_open_button_click() (the OPEN button's mid-session
        equivalent of this), just sourced from the pending-import details
        already resolved by main() rather than a fresh folder-browse
        dialog + find_model_file() call.

        Unlike the OPEN button's failure handling (which can safely leave
        a previously-loaded map running untouched), a failure HERE means
        there was never a map to fall back to at all -- so this prints a
        clear error and closes the window instead, rather than leaving
        the person staring at a permanently blank screen with no map and
        no way to get one without restarting the program anyway.
        """
        pending = CaveViewerWindow.cave_pending_import
        model_descriptor = pending["model_descriptor"]
        textures_dir = pending["textures_dir"]
        fmt = model_descriptor["format"]
        source_path = model_descriptor.get("obj_path") or model_descriptor.get("glb_path")
        map_name = os.path.basename(source_path)

        from caveviewer import import_and_cache_any
        from core import chunker as chunker_module

        already_cached = chunker_module.cache_is_valid(source_path)

        def on_progress(stage: str, fraction: float):
            self.import_progress_panel.render(self.wnd.size, map_name, stage, fraction)
            if hasattr(self.wnd, "swap_buffers"):
                self.wnd.swap_buffers()
            else:
                self.ctx.finish()

        try:
            if not already_cached:
                on_progress("starting import", 0.0)
                cache_dir = import_and_cache_any(model_descriptor, textures_dir, force_rebuild=False,
                                                   extra_progress_cb=on_progress)
            else:
                cache_dir = chunker_module.get_cache_dir(source_path)

            new_manifest = chunker_module.load_manifest(cache_dir)
        except Exception as e:
            print(f"[CaveViewer] Failed to import this map: {e}")
            print("[CaveViewer] Closing -- there's no map to show without a successful import.")
            # wnd.close() is moderngl-window's standard way to request a
            # clean shutdown, but -- same reasoning as the swap_buffers
            # defensive check above -- this project has hit real cross-
            # version API differences before, so this is wrapped rather
            # than assumed. Worst case if .close() isn't present: the
            # window stays open showing the neutral background from the
            # on_render() guard above (since _has_map_loaded stays False
            # and the import won't be retried), rather than crashing
            # inside this already-error-handling block.
            if hasattr(self.wnd, "close"):
                self.wnd.close()
            return

        print(f"[CaveViewer] Now viewing: {map_name}")
        self.load_new_map(cache_dir, textures_dir, new_manifest)

    # -- chunk GPU lifecycle ------------------------------------------------

    def _on_chunk_ready(self, chunk_data):
        vao_list = []
        for mat_name, group in chunk_data.groups.items():
            n = len(group.positions)
            if n == 0:
                continue
            interleaved = np.empty((n, 8), dtype=np.float32)
            interleaved[:, 0:3] = group.positions
            interleaved[:, 3:5] = group.uvs
            interleaved[:, 5:8] = group.normals

            vbo = self.ctx.buffer(interleaved.tobytes())
            vao = self.ctx.vertex_array(
                self.program, [(vbo, "3f 2f 3f", "in_position", "in_uv", "in_normal")]
            )
            texture = self.texture_manager.acquire(mat_name)
            vao_list.append((vao, vbo, mat_name, texture))

        self._chunk_gpu_objects[chunk_data.cell] = vao_list

    def _on_chunk_unload(self, cell):
        vao_list = self._chunk_gpu_objects.pop(cell, [])
        for vao, vbo, mat_name, texture in vao_list:
            vao.release()
            vbo.release()
            self.texture_manager.release(mat_name)

    # -- moderngl_window hooks ------------------------------------------------
    #
    # moderngl-window renamed its per-frame/event hooks across major versions
    # (older releases used bare names like render()/key_event(), 3.x renamed
    # them to on_render()/on_key_event() etc). To work across versions without
    # guessing which exact release someone has installed, each hook below is
    # implemented under the new on_* name and aliased to the old bare name.

    # Right-side column layout: brightness stepper, then render-distance
    # stepper, then the Mesh/Texture/Help/Color/Open button block, all
    # stacked vertically and anchored as ONE group to the bottom-right
    # corner of the window (moved here from separate top-anchored
    # positions per request). Computed in this single method, used
    # identically by render() and the mouse-press handler, so the
    # clickable areas can never drift out of sync with what's actually
    # drawn -- the same reasoning the old per-control anchor helpers
    # already followed, just now covering the whole column at once since
    # a bottom anchor means every piece's position depends on the total
    # height of everything below the WINDOW bottom margin, not just its
    # own height.
    RIGHT_COLUMN_BOTTOM_MARGIN = 18
    RIGHT_COLUMN_GAP = 14  # vertical gap between each of the 4 blocks (brightness, render distance, global light, buttons)

    # Maps the GLOBAL LIGHT stepper's 0-10 integer range onto the
    # shader's actual u_ambient float. 0 -> _AMBIENT_MIN reproduces the
    # exact fixed ambient value this app always used before this feature
    # existed (a tiny fill so unlit areas aren't pure black, not truly
    # zero) -- so the default stepper value of 0 changes nothing for
    # anyone who never touches this control. 10 -> _AMBIENT_MAX is a
    # strong, even fill bright enough to read the whole cave clearly
    # without the headlamp doing any of the work, without fully blowing
    # out texture detail into flat white.
    _AMBIENT_MIN = 0.04
    _AMBIENT_MAX = 0.9

    def _right_column_layout(self, window_size: tuple[int, int]) -> dict:
        """
        Returns a dict with every position the right-side column needs:
        'brightness_anchor', 'ambient_anchor' (the GLOBAL LIGHT stepper),
        'render_distance_anchor' (note: this stepper moved to the right
        column per request, no longer on the left), and 'buttons_top_y'
        -- each stepper anchor already accounts for its own label space
        above it (see StepperControl.render's label_above handling), and
        the button block's top_y already accounts for RenderModeButtons'
        own height-shrinking safety net on short windows.

        Stack order, top to bottom: Brightness, Global Light, Render
        Distance, then the button block.
        """
        w, h = window_size

        # label reserve: matches StepperControl.render's own
        # label_size=1.5 text height + 8px gap, computed here once so
        # this stays correct if that label styling ever changes (rather
        # than a second hard-coded guess at the same number).
        from gui import bitmap_font
        label_reserve = bitmap_font.text_height_px(1.5) + 8

        button_block_height = RenderModeButtons.total_stack_height()

        # Build the stack from the BOTTOM up: button block's bottom sits
        # RIGHT_COLUMN_BOTTOM_MARGIN above the window's bottom edge.
        buttons_bottom_y = h - self.RIGHT_COLUMN_BOTTOM_MARGIN
        buttons_top_y = buttons_bottom_y - button_block_height

        render_distance_bottom_y = buttons_top_y - self.RIGHT_COLUMN_GAP
        render_distance_anchor_y = render_distance_bottom_y - self.render_distance_stepper.total_height()

        ambient_bottom_y = render_distance_anchor_y - label_reserve - self.RIGHT_COLUMN_GAP
        ambient_anchor_y = ambient_bottom_y - self.ambient_stepper.total_height()

        brightness_bottom_y = ambient_anchor_y - label_reserve - self.RIGHT_COLUMN_GAP
        brightness_anchor_y = brightness_bottom_y - self.light_stepper.total_height()

        right_x_brightness = w - 18 - self.light_stepper.total_width()
        right_x_ambient = w - 18 - self.ambient_stepper.total_width()
        right_x_render_distance = w - 18 - self.render_distance_stepper.total_width()

        return {
            "brightness_anchor": (right_x_brightness, brightness_anchor_y),
            "ambient_anchor": (right_x_ambient, ambient_anchor_y),
            "render_distance_anchor": (right_x_render_distance, render_distance_anchor_y),
            "buttons_top_y": buttons_top_y,
        }

    def on_render(self, current_time: float, frame_time: float):
        if not self._has_map_loaded:
            # First frame with no map loaded yet: just clear to a neutral
            # background and let this frame actually reach the screen
            # before doing anything else -- _handle_continuous_input,
            # world.update, the camera, etc all assume a loaded map and
            # would crash if touched here. The actual import is kicked
            # off AFTER this first real frame has rendered (see the
            # _pending_import_started check below), specifically so the
            # window is confirmed visibly open first, rather than risking
            # the blocking import starting before anything has actually
            # been drawn to the screen even once.
            self.ctx.clear(0.02, 0.02, 0.03)
            if self._pending_import_started:
                return
            self._pending_import_started = True
            self._run_pending_import()
            return

        frame_start = time.perf_counter()
        dt = max(frame_time, 1e-4)
        self._handle_continuous_input(dt)

        # Apply the render-distance control's current value before the
        # streaming world recalculates this frame -- a click on +/- takes
        # effect immediately rather than waiting for the camera to move
        # (see the matching check in StreamingWorld.update(), which
        # detects a changed load_radius_cells on its own, not just a
        # moved camera -- this assignment is what actually gives it a
        # changed value to detect).
        self.world.config.load_radius_cells = self.render_distance_stepper.value

        t0 = time.perf_counter()
        self.world.update(self.camera.position.astype(np.float32))
        self.world.drain_ready_chunks(
            self._on_chunk_ready, self._on_chunk_unload,
            max_per_frame=6, time_budget_ms=3.0,
        )
        streaming_ms = (time.perf_counter() - t0) * 1000.0

        self.ctx.clear(*self.color_picker.color)  # background ("void") color, adjustable via the COLOR button

        aspect = self.wnd.size[0] / max(self.wnd.size[1], 1)
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(aspect)
        model = np.identity(4, dtype=np.float32)

        self.program["u_view"].write(view.T.tobytes())
        self.program["u_projection"].write(proj.T.tobytes())
        self.program["u_model"].write(model.T.tobytes())
        self.program["u_camera_pos"].value = tuple(self.camera.position.astype(np.float32))
        self.program["u_light_color"].value = (1.0, 0.95, 0.85)  # warm headlamp tone
        self.program["u_light_intensity"].value = float(self.light_stepper.value)
        # GLOBAL LIGHT stepper (0-10) maps linearly onto the shader's
        # actual ambient range -- see _AMBIENT_MIN/_AMBIENT_MAX's
        # docstring above for why 0 reproduces the app's original fixed
        # ambient value rather than true darkness.
        ambient_t = self.ambient_stepper.value / self.ambient_stepper.max_value
        ambient_value = self._AMBIENT_MIN + ambient_t * (self._AMBIENT_MAX - self._AMBIENT_MIN)
        self.program["u_ambient"].value = ambient_value
        self.program["u_texture_enabled"].value = self.render_mode_buttons.texture_enabled

        t0 = time.perf_counter()

        # Solid pass (textured, or plain gray if Texture is off) only
        # draws when at least one of "show texture" or "wireframe is off"
        # is true. In other words: skip the solid pass entirely when the
        # person has explicitly turned Texture off AND turned Mesh
        # (wireframe) on -- that combination means "show me pure
        # wireframe, nothing else", and the solid pass would otherwise
        # always render underneath the wireframe lines regardless of the
        # Texture toggle, which defeats the point of turning texture off
        # in the first place when inspecting wireframe-only.
        show_solid_pass = self.render_mode_buttons.texture_enabled or not self.render_mode_buttons.wireframe_enabled
        if show_solid_pass:
            for cell, vao_list in self._chunk_gpu_objects.items():
                for vao, vbo, mat_name, texture in vao_list:
                    texture.use(location=0)
                    self.program["u_texture"].value = 0
                    vao.render(moderngl.TRIANGLES)

        # Wireframe pass: drawn whenever Mesh is toggled on. If the solid
        # pass also drew (texture or gray surface visible), this overlays
        # triangulation on top of it. If the solid pass was skipped (the
        # texture-off + wireframe-on combination above), this is the only
        # thing that draws -- true wireframe-only.
        if self.render_mode_buttons.wireframe_enabled:
            # NOTE: this draws coincident wireframe lines directly on top of
            # the solid pass's geometry, which can show minor z-fighting/
            # flicker on some GPUs since both passes write near-identical
            # depth values. A polygon-offset bias would clean this up, but
            # since the bias amount needs hand-tuning against moderngl's
            # actual ctx.polygon_offset API (left out here rather than
            # guess at a value that could silently do nothing or look
            # wrong), this is a known minor cosmetic rough edge -- the
            # wireframe is still fully readable, just not perfectly crisp
            # in rare cases.
            self.ctx.wireframe = True
            for cell, vao_list in self._chunk_gpu_objects.items():
                for vao, vbo, mat_name, texture in vao_list:
                    vao.render(moderngl.TRIANGLES)
            self.ctx.wireframe = False
        mesh_draw_ms = (time.perf_counter() - t0) * 1000.0

        # Overlay HUD elements draw last, on top of the 3D scene, each with
        # their own depth-disabled 2D pass.
        t0 = time.perf_counter()

        # Whole right-side column -- brightness, global light, render
        # distance, then the Mesh/Texture/Help/Color/Open buttons -- is
        # laid out as one group anchored to the bottom-right corner. See
        # _right_column_layout()'s docstring for why this is computed in
        # one place rather than each piece anchoring itself independently.
        column = self._right_column_layout(self.wnd.size)
        brightness_anchor_x, brightness_anchor_y = column["brightness_anchor"]
        ambient_anchor_x, ambient_anchor_y = column["ambient_anchor"]
        render_distance_anchor_x, render_distance_anchor_y = column["render_distance_anchor"]
        buttons_top_y = column["buttons_top_y"]

        self.light_stepper.render(self.wnd.size, brightness_anchor_x, brightness_anchor_y, label_above=True)
        self.ambient_stepper.render(self.wnd.size, ambient_anchor_x, ambient_anchor_y, label_above=True)
        self.render_distance_stepper.render(self.wnd.size, render_distance_anchor_x, render_distance_anchor_y,
                                              label_above=True)

        self.minimap.render(self.wnd.size, self.camera.position)

        # FPS / chunk-loading readout, positioned directly above the
        # minimap panel (a small gap between the two so they don't touch).
        # FPS is smoothed over the same rolling frame-time window the
        # spike-detector below already maintains, rather than the
        # coarser ~2-second console-print interval -- a readout that
        # only updates every 2 seconds would feel sluggish and
        # disconnected from whatever you just changed (e.g. clicking the
        # render-distance stepper).
        minimap_x0, minimap_y0, minimap_x1, minimap_y1 = self.minimap._panel_rect_px(self.wnd.size)
        if self._frame_time_history:
            avg_frame_ms = sum(self._frame_time_history) / len(self._frame_time_history)
            instantaneous_fps = 1000.0 / max(avg_frame_ms, 0.1)
        else:
            instantaneous_fps = 0.0
        world_stats = self.world.stats()
        readout_bottom_y = minimap_y0 - 8  # 8px gap above the minimap's top edge
        self.stats_readout.render(
            self.wnd.size, minimap_x0, readout_bottom_y,
            fps=instantaneous_fps, chunks_loaded=world_stats["loaded"], chunks_pending=world_stats["pending"],
        )

        self.render_mode_buttons.render(self.wnd.size, buttons_top_y,
                                          help_active=self.controls_overlay.is_manual_mode,
                                          color_active=self.color_picker.is_active)

        # Color picker panel draws on top of the regular HUD elements (it
        # dims the 3D view behind it, same visual language as the Help
        # screen) but still below the controls overlay, consistent with
        # Help also losing to a loading overlay if both somehow overlap.
        self.color_picker.render(self.wnd.size)

        # Controls/loading overlay draws last of all, on top of every
        # other UI element -- while it's showing, it's meant to be the
        # thing you're looking at (it's explaining what the other UI
        # pieces do), so it should never be obscured by them.
        self.controls_overlay.update(self.world.stats())
        self.controls_overlay.render(self.wnd.size)
        overlay_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - frame_start) * 1000.0

        # Spike detection: track a short rolling average of frame times, and
        # if a frame comes in notably above that average, print a one-line
        # breakdown of where the time went. This is the diagnostic for
        # tracking down any remaining stutter -- rather than guess at
        # causes, the next time a stutter happens this will print exactly
        # which section (chunk streaming, mesh draw, or overlay draw) was
        # responsible, plus chunk-loading stats at that moment.
        self._frame_time_history.append(total_ms)
        if len(self._frame_time_history) > 30:
            self._frame_time_history.pop(0)
        rolling_avg = sum(self._frame_time_history) / len(self._frame_time_history)

        if len(self._frame_time_history) >= 10 and total_ms > max(rolling_avg * 3, 25.0):
            stats = self.world.stats()
            print(f"[CaveViewer] FRAME SPIKE: {total_ms:.1f}ms (avg {rolling_avg:.1f}ms) | "
                  f"streaming={streaming_ms:.1f}ms mesh_draw={mesh_draw_ms:.1f}ms "
                  f"overlay={overlay_ms:.1f}ms | chunks loaded={stats['loaded']} "
                  f"pending={stats['pending']}")

        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_print > 2.0:
            fps = self._frame_count / (now - self._last_fps_print)
            stats = self.world.stats()
            print(f"[CaveViewer] {fps:.1f} fps | chunks loaded={stats['loaded']} "
                  f"pending={stats['pending']} | speed={self.camera.move_speed:.1f}m/s")
            self._frame_count = 0
            self._last_fps_print = now

    render = on_render  # back-compat alias for older moderngl-window releases

    def _resolve_key(self, keys, *candidate_names):
        """
        Different moderngl-window/pyglet versions have used different names
        for the same key (e.g. LEFT_CONTROL vs LEFT_CTRL). Rather than hard-
        code one name and risk another AttributeError crash on a different
        installed version, try each known alias in turn and cache whichever
        one actually exists on this version's Keys class.
        """
        cache = getattr(self, "_key_resolve_cache", None)
        if cache is None:
            cache = {}
            self._key_resolve_cache = cache
        cache_key = candidate_names
        if cache_key in cache:
            return cache[cache_key]
        for name in candidate_names:
            if hasattr(keys, name):
                value = getattr(keys, name)
                cache[cache_key] = value
                return value
        raise AttributeError(
            f"None of the key names {candidate_names} exist on this "
            f"moderngl-window version's Keys class. Available attributes: "
            f"{[a for a in dir(keys) if not a.startswith('_')]}"
        )

    def _handle_continuous_input(self, dt: float):
        keys = self.wnd.keys
        forward_amt = 0.0
        right_amt = 0.0
        up_amt = 0.0
        if keys.W in self._keys_down:
            forward_amt += 1.0
        if keys.S in self._keys_down:
            forward_amt -= 1.0
        if keys.D in self._keys_down:
            right_amt += 1.0
        if keys.A in self._keys_down:
            right_amt -= 1.0
        if keys.SPACE in self._keys_down:
            up_amt += 1.0

        ctrl_key = self._resolve_key(keys, "LEFT_CONTROL", "LEFT_CTRL", "LCTRL")
        if ctrl_key in self._keys_down:
            up_amt -= 1.0

        shift_key = self._resolve_key(keys, "LEFT_SHIFT", "LSHIFT")
        speed_mult = 3.0 if shift_key in self._keys_down else 1.0
        if forward_amt or right_amt or up_amt:
            self.camera.move(forward_amt, right_amt, up_amt, dt, speed_mult)

    def on_key_event(self, key, action, modifiers: KeyModifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            self._keys_down.add(key)
        elif action == keys.ACTION_RELEASE:
            self._keys_down.discard(key)

    key_event = on_key_event

    def on_mouse_position_event(self, x, y, dx, dy):
        # Color picker's RGB sliders still use continuous drag (a
        # separate feature from the brightness/render-distance controls
        # below, which were converted to discrete +/- steppers) -- this
        # still needs to take priority over camera look while one of its
        # sliders is being dragged, same reasoning as before.
        if self.color_picker.is_dragging:
            self.color_picker.on_mouse_drag(x, y, self.wnd.size)
            return
        if self._mouse_look_active:
            self.camera.look(dx, dy)

    mouse_position_event = on_mouse_position_event

    def on_mouse_press_event(self, x, y, button):
        if button == self.wnd.mouse.left:
            # Check order: all three steppers, then mesh/texture toggle
            # buttons, then minimap. All four pieces (brightness, global
            # light, render distance, button block) now live together in
            # the same bottom-right column -- check order only matters in
            # the sense that each needs to happen before falling through
            # to the next, since their hit areas don't overlap.
            column = self._right_column_layout(self.wnd.size)
            brightness_anchor_x, brightness_anchor_y = column["brightness_anchor"]
            ambient_anchor_x, ambient_anchor_y = column["ambient_anchor"]
            render_distance_anchor_x, render_distance_anchor_y = column["render_distance_anchor"]
            buttons_top_y = column["buttons_top_y"]

            if self.light_stepper.on_mouse_press(x, y, brightness_anchor_x, brightness_anchor_y):
                return

            if self.ambient_stepper.on_mouse_press(x, y, ambient_anchor_x, ambient_anchor_y):
                return

            if self.render_distance_stepper.on_mouse_press(x, y, render_distance_anchor_x, render_distance_anchor_y):
                return

            clicked_button = self.render_mode_buttons.on_mouse_press(x, y, self.wnd.size, buttons_top_y)
            if clicked_button == "help":
                # Toggle: if the help screen is already showing (manual
                # mode), a second click closes it; otherwise show it.
                # Showing help intentionally overrides whatever loading
                # overlay might currently be active (e.g. a brief teleport
                # panel) -- an explicit click is a clear request to see
                # the controls right now, which should win over a
                # transient loading indicator.
                if self.controls_overlay.is_manual_mode:
                    self.controls_overlay.hide_help()
                else:
                    self.controls_overlay.show_help()
                return
            elif clicked_button == "color":
                if self.color_picker.is_active:
                    self.color_picker.hide()
                else:
                    self.color_picker.show()
                return
            elif clicked_button == "open":
                self._handle_open_button_click()
                return
            elif clicked_button is not None:
                # "mesh" or "texture" -- already handled internally by
                # render_mode_buttons.on_mouse_press (it toggled its own
                # state before returning), nothing further needed here.
                return

            # While the color picker panel is open, it behaves like a
            # modal -- ANY left-click is consumed by it (a slider drag,
            # or simply a click that misses every slider), rather than
            # falling through to the minimap/3D view underneath. Without
            # this, clicking just outside a slider while picking a color
            # could accidentally teleport you via the minimap at the
            # same time, which would be a confusing side effect of what
            # was meant to be a color adjustment.
            if self.color_picker.is_active:
                self.color_picker.on_mouse_press(x, y, self.wnd.size)
                return

            minimap_target = self.minimap.world_xz_for_click(x, y, self.wnd.size)
            if minimap_target is not None:
                target_x, target_z = minimap_target
                # Land at an actual occupied height near that X/Z, rather
                # than blindly keeping the camera's previous Y -- a click
                # on the (top-down, height-blind) minimap doesn't tell us
                # which vertical level was meant, so we look up real chunk
                # bounds at that column and pick whichever level is
                # closest to the camera's current height (see
                # find_landing_position in core/chunker.py). This is what
                # prevents landing above or below the actual passage.
                landing_x, landing_y, landing_z = chunker.find_landing_position(
                    self.manifest, target_x, target_z,
                    preferred_y=float(self.camera.position[1]),
                )
                self.camera.position[0] = landing_x
                self.camera.position[1] = landing_y
                self.camera.position[2] = landing_z

                # Show the controls panel briefly while the newly-teleported
                # area's chunks stream in around the camera -- same content
                # as the full-screen startup overlay, just smaller since
                # teleporting is quick and shouldn't block the whole view.
                self.controls_overlay.show_panel()
                return
            return
        if button == self.wnd.mouse.right:
            self._mouse_look_active = True
            self.wnd.mouse_exclusivity = True

    mouse_press_event = on_mouse_press_event

    def on_mouse_release_event(self, x, y, button):
        if button == self.wnd.mouse.left:
            self.color_picker.on_mouse_release()
            return
        if button == self.wnd.mouse.right:
            self._mouse_look_active = False
            self.wnd.mouse_exclusivity = False

    mouse_release_event = on_mouse_release_event

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.camera.adjust_speed(y_offset)

    mouse_scroll_event = on_mouse_scroll_event

    def on_close(self):
        if self._has_map_loaded:
            self.world.shutdown()

    close = on_close



def run_viewer(cache_dir: str, textures_dir: str):
    manifest = chunker.load_manifest(cache_dir)

    # Set as class attributes rather than passing through run_window_config's
    # kwargs -- see the comment on CaveViewerWindow's class attributes above
    # for why. This sidesteps moderngl-window version differences in how
    # (or whether) run_window_config forwards extra keyword arguments.
    CaveViewerWindow.cave_cache_dir = cache_dir
    CaveViewerWindow.cave_textures_dir = textures_dir
    CaveViewerWindow.cave_manifest = manifest

    mglw.run_window_config(CaveViewerWindow, args=[])


def run_viewer_with_pending_import(model_descriptor: dict, textures_dir: str):
    """
    Launches the viewer window for a map that needs FIRST-TIME import
    (no .caveviewer_cache yet) -- used by caveviewer.py's main() instead
    of run_viewer() specifically so the import can run AFTER the window
    is open, showing real progress in the same in-window panel the OPEN
    button already uses, rather than the old behavior of running the
    import entirely before any window existed (which could only show a
    plain console progress bar, with nowhere graphical to draw into yet).

    model_descriptor is whatever caveviewer.py's find_model_file()
    returned -- a small dict identifying which format (.obj, .glb)
    and the relevant file path(s), format-agnostic so this single
    function/code path covers every supported source format rather than
    needing a separate pending-import entry point per format.

    The window opens immediately with no map loaded; the actual import
    is triggered from inside CaveViewerWindow.on_render()'s first frame
    (see _run_pending_import) once the window is confirmed to have
    rendered and is genuinely on screen.
    """
    CaveViewerWindow.cave_cache_dir = None
    CaveViewerWindow.cave_textures_dir = None
    CaveViewerWindow.cave_manifest = None
    CaveViewerWindow.cave_pending_import = {
        "model_descriptor": model_descriptor,
        "textures_dir": textures_dir,
    }

    mglw.run_window_config(CaveViewerWindow, args=[])
