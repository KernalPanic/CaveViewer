"""
core/streaming_world.py

Runtime chunk streaming: watches the camera's world position and keeps
only a radius of chunks around it loaded into GPU memory, uploading newly
needed chunks and evicting ones that fall out of range. This is the actual
mechanism that prevents lag on big maps -- the renderer never sees more
geometry/textures than fit within `load_radius_cells` of the camera,
regardless of how large the full cave map is.

This module is GPU-API-agnostic: it deals in ChunkData (CPU-side numpy
arrays) and calls back into caller-supplied upload/evict functions so the
moderngl-specific VBO/texture code lives in gui/viewer_window.py, not here.
This keeps the streaming *logic* unit-testable without an OpenGL context
(see the test suite -- we verify load/unload behavior with fake GPU hooks).
"""

from __future__ import annotations

import threading
import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from core import chunker
from core.chunker import ChunkData


@dataclass
class StreamingConfig:
    chunk_size: float
    load_radius_cells: int = 3     # ring radius kept loaded around camera
    # (unload_radius > load_radius prevents thrashing when camera sits
    #  near a cell boundary and jitters back and forth)
    unload_radius_margin: int = 1  # how many cells beyond load_radius before eviction
    max_loaded_chunks: int = 400   # hard cap as a safety valve regardless of radius

    @property
    def unload_radius_cells(self) -> int:
        """
        Derived from load_radius_cells rather than stored as an
        independent fixed value, so the hysteresis gap between "keep
        loaded" and "evict" stays correct automatically even when
        load_radius_cells changes at runtime (see the render-distance
        slider in viewer_window.py) -- a fixed unload_radius_cells set
        once at construction would otherwise need to be kept in sync by
        hand every time the load radius changes, and a bug there would
        be the kind of thing that's easy to miss (the gap silently
        shrinking or inverting) until someone actually notices chunks
        thrashing load/unload near a boundary.
        """
        return self.load_radius_cells + self.unload_radius_margin


class StreamingWorld:
    """
    Call `update(camera_position)` once per frame (or every N frames -- it's
    cheap, but you can throttle further if desired). It will:
      - compute which cells *should* be loaded given the camera position
      - kick off background loads (disk I/O + numpy unpacking) for missing
        ones via a worker thread, so disk reads never block the render
        thread / cause a frame hitch
      - call `on_chunk_ready(ChunkData)` on the main thread (via
        `drain_ready_chunks()`, which you call once per frame) for any
        chunks that finished loading
      - call `on_chunk_unload(cell)` for chunks that should be evicted
    """

    def __init__(self, cache_dir: str, config: StreamingConfig,
                 on_decode_textures: Optional[Callable[[ChunkData], None]] = None):
        """
        on_decode_textures, if given, is called from a background worker
        thread right after a chunk's geometry finishes loading, with the
        ChunkData as the argument. This is the hook used to pre-decode each
        chunk's textures (JPEG decode, pure CPU work, safe off-thread) so
        that by the time the chunk reaches the main thread for GPU upload,
        only the fast/predictable upload step remains there. Keeping this
        as an injected callback rather than importing TextureManager
        directly here preserves this module's GPU-API-agnostic design and
        keeps it unit-testable without any texture/GPU machinery at all.
        """
        self.cache_dir = cache_dir
        self.config = config
        self.on_decode_textures = on_decode_textures
        self.manifest = chunker.load_manifest(cache_dir)
        self.available_cells: set[tuple[int, int, int]] = {
            tuple(int(x) for x in cell_str.split("_"))
            for cell_str in self.manifest["chunks"]
        }

        self.loaded_cells: set[tuple[int, int, int]] = set()
        self._pending: set[tuple[int, int, int]] = set()
        self._ready_queue: "queue.Queue[ChunkData]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker_pool_size = 3
        self._stop_event = threading.Event()
        self._work_queue: "queue.Queue[tuple[int,int,int]]" = queue.Queue()
        self._workers = [
            threading.Thread(target=self._worker_loop, daemon=True)
            for _ in range(self._worker_pool_size)
        ]
        for w in self._workers:
            w.start()

        self._last_camera_cell: Optional[tuple[int, int, int]] = None
        self._last_load_radius: Optional[int] = None

    def shutdown(self):
        self._stop_event.set()
        for _ in self._workers:
            self._work_queue.put(None)  # sentinel to unblock get()
        for w in self._workers:
            w.join(timeout=2.0)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                cell = self._work_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if cell is None:
                continue
            try:
                data = chunker.load_chunk_file(self.cache_dir, cell)
                if self.on_decode_textures is not None:
                    try:
                        self.on_decode_textures(data)
                    except Exception as e:
                        # texture pre-decode is a best-effort optimization;
                        # a failure here should not block the chunk from
                        # becoming ready -- worst case, acquire() falls back
                        # to a synchronous decode on the main thread later.
                        print(f"[StreamingWorld] texture pre-decode failed for {cell}: {e}")
                self._ready_queue.put(data)
            except FileNotFoundError:
                # cell vanished from manifest expectations; ignore safely
                pass
            except Exception as e:
                # don't crash the worker thread on a single bad chunk file;
                # surface via print so it's visible without killing render
                print(f"[StreamingWorld] failed to load chunk {cell}: {e}")

    def cell_for_position(self, position: np.ndarray) -> tuple[int, int, int]:
        return chunker.world_to_cell(position, self.config.chunk_size)

    def update(self, camera_position: np.ndarray) -> None:
        """Call once per frame. Cheap if camera hasn't crossed a cell
        boundary AND the load radius hasn't changed since the last call
        (early-outs immediately in that case)."""
        cam_cell = self.cell_for_position(camera_position)
        current_radius = self.config.load_radius_cells

        # Recompute if the camera moved to a new cell OR the radius itself
        # changed (e.g. the person just dragged a render-distance slider
        # while standing still). Without the radius check, adjusting the
        # slider at a standstill would silently do nothing until the
        # camera happened to cross a cell boundary on its own -- the
        # slider would feel completely broken on first try.
        if cam_cell == self._last_camera_cell and current_radius == self._last_load_radius:
            return
        self._last_camera_cell = cam_cell
        self._last_load_radius = current_radius

        load_r = self.config.load_radius_cells
        wanted = self._cells_in_radius(cam_cell, load_r) & self.available_cells

        with self._lock:
            to_request = wanted - self.loaded_cells - self._pending
            # Dispatch closest-to-camera first. Without this, chunks load in
            # whatever arbitrary order set-iteration and thread scheduling
            # happen to produce -- so a chunk directly ahead of a fast-moving
            # camera (which causes a visible hole if it's late) can finish
            # loading AFTER a chunk behind the camera that doesn't matter yet.
            # Sorting by distance means the chunks the camera will reach
            # soonest are always the ones uploaded soonest.
            ordered = sorted(
                to_request,
                key=lambda cell: self._cell_distance_sq(cell, cam_cell),
            )
            for cell in ordered:
                self._pending.add(cell)
                self._work_queue.put(cell)

        # eviction uses a larger radius than load, so a chunk isn't dropped
        # the instant it's outside the tight load ring -- avoids reload
        # thrashing if the camera oscillates near a boundary.
        unload_r = self.config.unload_radius_cells
        keep = self._cells_in_radius(cam_cell, unload_r)
        self._cells_to_unload_next_drain = self.loaded_cells - keep
        self._last_cam_cell_for_priority = cam_cell

    def _cell_distance_sq(self, cell: tuple[int, int, int], center: tuple[int, int, int]) -> int:
        return (cell[0] - center[0]) ** 2 + (cell[1] - center[1]) ** 2 + (cell[2] - center[2]) ** 2

    def _cells_in_radius(self, center: tuple[int, int, int], radius: int) -> set[tuple[int, int, int]]:
        cx, cy, cz = center
        return {
            (cx + dx, cy + dy, cz + dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
        }

    def drain_ready_chunks(self, on_chunk_ready: Callable[[ChunkData], None],
                             on_chunk_unload: Callable[[tuple], None],
                             max_per_frame: int = 4,
                             time_budget_ms: float = 4.0) -> None:
        """
        Call once per frame on the render/main thread. Pulls finished
        background loads and hands them to `on_chunk_ready` (where the
        caller uploads to GPU -- the only part of this that has to happen
        on the main thread, since OpenGL calls aren't valid off-thread).

        Two throttles apply together:
          - `max_per_frame`: hard cap on number of chunks uploaded this call,
            as a simple worst-case backstop.
          - `time_budget_ms`: stops uploading once this many milliseconds
            have been spent in this call, even if under max_per_frame and
            even if more chunks are ready. This matters because individual
            chunks vary a lot in cost (a chunk needing a fresh texture
            decode is much more expensive than one reusing an already-
            resident texture) -- a fixed *count* can still spike frame time
            if several expensive chunks land in the same frame. The time
            budget catches that case; the count cap is just a backstop.

        Ready chunks are also re-sorted by distance to the camera's last
        known cell before draining, so if multiple chunks became ready
        between frames, the ones closest to the camera (most likely to
        cause a visible hole if delayed) are uploaded first.
        """
        # Drain everything currently sitting in the ready queue into a list
        # so we can sort by distance before uploading any of it -- this is
        # cheap (CPU-side list ops on ChunkData references, no GPU work yet).
        pending_ready = []
        while True:
            try:
                pending_ready.append(self._ready_queue.get_nowait())
            except queue.Empty:
                break

        cam_cell = getattr(self, "_last_cam_cell_for_priority", None)
        if cam_cell is not None and pending_ready:
            pending_ready.sort(key=lambda d: self._cell_distance_sq(d.cell, cam_cell))

        start = time.perf_counter()
        n = 0
        leftover = []
        for data in pending_ready:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if n >= max_per_frame or elapsed_ms >= time_budget_ms:
                leftover.append(data)
                continue
            with self._lock:
                self._pending.discard(data.cell)
                self.loaded_cells.add(data.cell)
            on_chunk_ready(data)
            n += 1

        # anything we didn't get to this frame goes back in the queue,
        # still in priority order, so the next frame picks up where this
        # one left off rather than re-sorting from scratch each time
        for data in leftover:
            self._ready_queue.put(data)

        unload_now = getattr(self, "_cells_to_unload_next_drain", set())
        if unload_now:
            for cell in list(unload_now):
                with self._lock:
                    self.loaded_cells.discard(cell)
                on_chunk_unload(cell)
            self._cells_to_unload_next_drain = set()

    def stats(self) -> dict:
        return {
            "loaded": len(self.loaded_cells),
            "pending": len(self._pending),
            "total_available": len(self.available_cells),
        }
