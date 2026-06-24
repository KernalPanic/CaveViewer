"""
core/chunker.py

Spatial partitioning of a parsed mesh into a 3D grid of chunks, cached to
disk in a fast-to-load binary format. This is the piece that makes large
cave maps viewable: instead of one giant draw call / VRAM blob for the
whole cave, we split the mesh into cells (default 8m cubes -- tune via
CHUNK_SIZE for your cave's scale) and load only the cells near the camera
at runtime (see core/streaming_world.py).

Cache layout on disk, under <obj_folder>/.caveviewer_cache/:
    manifest.json          - chunk grid metadata, bounds, cell size,
                              chunk_id -> required texture list, etc.
    chunks/<cx>_<cy>_<cz>.bin
                              - one binary blob per occupied cell:
                                packed positions/uvs/normals/indices,
                                grouped by material so each chunk can issue
                                one draw call per texture it touches.

A face is assigned to a cell based on its centroid. A mesh face never
spans multiple chunks even if its vertices are near a boundary -- this
intentionally avoids vertex splitting complexity. Cracks at chunk seams
are visually negligible at cave scale and the chunk overlap-load radius
(loading neighbor rings, not just the current cell) means seams are never
at the camera's center of attention for long.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np

from core.obj_parser import RawMesh, MaterialRange

CACHE_DIRNAME = ".caveviewer_cache"
MANIFEST_NAME = "manifest.json"
CHUNKS_DIRNAME = "chunks"

DEFAULT_CHUNK_SIZE = 8.0  # meters; tune based on cave passage scale

_MAGIC = b"CVCH"  # CaveViewer CHunk
_VERSION = 1


@dataclass
class ChunkData:
    """In-memory representation of one spatial cell's geometry, grouped by
    material so the renderer can do one draw call per texture per chunk."""
    cell: tuple[int, int, int]
    groups: dict[str, "ChunkMaterialGroup"]
    bounds_min: np.ndarray  # (3,) float32
    bounds_max: np.ndarray  # (3,) float32


@dataclass
class ChunkMaterialGroup:
    material_name: str
    positions: np.ndarray   # (N, 3) float32, flat (already expanded, not indexed)
    uvs: np.ndarray         # (N, 2) float32
    normals: np.ndarray     # (N, 3) float32


def world_to_cell(point: np.ndarray, chunk_size: float) -> tuple[int, int, int]:
    return tuple(np.floor(point / chunk_size).astype(np.int64).tolist())


def build_cache(obj_path: str, mesh: RawMesh, materials: dict,
                 chunk_size: float = DEFAULT_CHUNK_SIZE,
                 progress_cb=None) -> str:
    """
    Partition `mesh` into spatial chunks and write the disk cache next to
    `obj_path`. Returns the cache directory path.

    progress_cb(stage: str, fraction: float)
    """
    obj_dir = os.path.dirname(os.path.abspath(obj_path))
    cache_dir = os.path.join(obj_dir, CACHE_DIRNAME)
    chunks_dir = os.path.join(cache_dir, CHUNKS_DIRNAME)
    os.makedirs(chunks_dir, exist_ok=True)

    if progress_cb:
        progress_cb("computing face centroids", 0.0)

    tri_pos = mesh.positions[mesh.face_pos_idx]   # (Nf, 3, 3)
    centroids = tri_pos.mean(axis=1)              # (Nf, 3)
    cell_coords = np.floor(centroids / chunk_size).astype(np.int64)  # (Nf, 3)

    if progress_cb:
        progress_cb("grouping faces by cell", 0.1)

    n_faces = len(mesh.face_pos_idx)

    # IMPORTANT: key by *unique material name*, not by MaterialRange index.
    # A single material (e.g. "tile_A") can appear in multiple separate
    # usemtl ranges throughout the OBJ (common when Agisoft interleaves
    # texture tile usage across the file). If we keyed by range index here,
    # faces using the same texture tile but from different ranges would be
    # split into separate groups instead of merging -- wasting draw calls
    # and, worse, corrupting the cell-grouping logic below since range
    # boundaries don't align with cell boundaries.
    unique_material_names = sorted(set(mr.material_name for mr in mesh.material_ranges))
    material_name_to_id = {name: i for i, name in enumerate(unique_material_names)}
    material_names = unique_material_names  # used below for id -> name lookup

    face_material_id = np.full(n_faces, -1, dtype=np.int32)
    for mr in mesh.material_ranges:
        face_material_id[mr.start_face:mr.end_face] = material_name_to_id[mr.material_name]

    cell_min = cell_coords.min(axis=0)
    shifted = cell_coords - cell_min
    AXIS_BITS = 100_000
    cell_key = (shifted[:, 0].astype(np.int64) * AXIS_BITS * AXIS_BITS
                + shifted[:, 1].astype(np.int64) * AXIS_BITS
                + shifted[:, 2].astype(np.int64))
    combined_key = cell_key * (len(material_names) + 1) + (face_material_id.astype(np.int64) + 1)

    order = np.argsort(combined_key, kind="stable")
    sorted_keys = combined_key[order]

    boundaries = np.nonzero(np.diff(sorted_keys))[0] + 1
    run_starts = np.concatenate(([0], boundaries))
    run_ends = np.concatenate((boundaries, [len(sorted_keys)]))

    if progress_cb:
        progress_cb("writing chunk files", 0.3)

    manifest_chunks = {}
    total_runs = len(run_starts)

    current_cell_groups = []
    current_cell_coord = None

    def flush_cell():
        nonlocal current_cell_groups, current_cell_coord
        if current_cell_coord is None or not current_cell_groups:
            return
        cell_str = f"{current_cell_coord[0]}_{current_cell_coord[1]}_{current_cell_coord[2]}"
        bounds_min, bounds_max, used_materials = _write_chunk_file(
            chunks_dir, cell_str, mesh, current_cell_groups)
        manifest_chunks[cell_str] = {
            "materials": used_materials,
            "bounds_min": bounds_min.tolist(),
            "bounds_max": bounds_max.tolist(),
        }
        current_cell_groups = []

    for i in range(total_runs):
        if progress_cb and i % 200 == 0:
            progress_cb("writing chunk files", 0.3 + 0.65 * (i / max(total_runs, 1)))

        s, e = run_starts[i], run_ends[i]
        face_idx_in_order = order[s:e]
        key = sorted_keys[s]
        mat_id = int(key % (len(material_names) + 1)) - 1
        cell_packed = key // (len(material_names) + 1)
        cz = int(cell_packed % AXIS_BITS)
        cy = int((cell_packed // AXIS_BITS) % AXIS_BITS)
        cx = int(cell_packed // (AXIS_BITS * AXIS_BITS))
        real_cell = (cx + int(cell_min[0]), cy + int(cell_min[1]), cz + int(cell_min[2]))

        if real_cell != current_cell_coord:
            flush_cell()
            current_cell_coord = real_cell

        mat_name = material_names[mat_id] if mat_id >= 0 else "__no_material__"
        current_cell_groups.append((mat_name, face_idx_in_order))

    flush_cell()

    if progress_cb:
        progress_cb("writing manifest", 0.98)

    manifest = {
        "version": _VERSION,
        "chunk_size": chunk_size,
        "source_obj": os.path.basename(obj_path),
        "mtl_materials": {
            name: mat.diffuse_texture for name, mat in materials.items()
        },
        "chunks": manifest_chunks,
    }
    with open(os.path.join(cache_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f)

    if progress_cb:
        progress_cb("done", 1.0)

    return cache_dir


def _write_chunk_file(chunks_dir, cell_str, mesh, groups):
    """
    Write one chunk binary file containing all material groups for a cell.
    De-indexes faces into flat (position, uv, normal) vertex triples per
    group, since at render time we want simple flat VBOs, no index buffer
    juggling across materials.

    Binary format:
        MAGIC (4 bytes) | VERSION (uint32)
        n_groups (uint32)
        for each group:
            name_len (uint32) | name (utf8 bytes)
            n_verts (uint32)
            positions: n_verts * 3 float32
            uvs:       n_verts * 2 float32
            normals:   n_verts * 3 float32
    """
    path = os.path.join(chunks_dir, f"{cell_str}.bin")
    has_normals = mesh.normals.shape[0] > 0
    has_uvs = mesh.uvs.shape[0] > 0

    all_positions = []
    used_materials = []

    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<I", _VERSION))
        f.write(struct.pack("<I", len(groups)))

        for mat_name, face_idx in groups:
            pos_idx = mesh.face_pos_idx[face_idx].reshape(-1)
            uv_idx = mesh.face_uv_idx[face_idx].reshape(-1)
            nrm_idx = mesh.face_nrm_idx[face_idx].reshape(-1)

            flat_pos = mesh.positions[pos_idx].astype(np.float32)

            if has_uvs and (uv_idx >= 0).all():
                flat_uv = mesh.uvs[uv_idx].astype(np.float32)
            else:
                flat_uv = np.zeros((len(pos_idx), 2), dtype=np.float32)

            if has_normals and (nrm_idx >= 0).all():
                flat_nrm = mesh.normals[nrm_idx].astype(np.float32)
            else:
                flat_nrm = _compute_flat_normals(flat_pos)

            name_bytes = mat_name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(flat_pos)))
            f.write(flat_pos.tobytes())
            f.write(flat_uv.tobytes())
            f.write(flat_nrm.tobytes())

            all_positions.append(flat_pos)
            used_materials.append(mat_name)

    stacked = np.concatenate(all_positions, axis=0)
    return stacked.min(axis=0), stacked.max(axis=0), used_materials


def _compute_flat_normals(flat_pos: np.ndarray) -> np.ndarray:
    """Per-triangle face normal, duplicated across the triangle's 3 verts,
    used as a fallback when the OBJ didn't supply vertex normals."""
    tris = flat_pos.reshape(-1, 3, 3)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    n = n / lengths
    return np.repeat(n, 3, axis=0).astype(np.float32)


def load_manifest(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, MANIFEST_NAME), "r") as f:
        return json.load(f)


def load_chunk_file(cache_dir: str, cell: tuple[int, int, int]) -> ChunkData:
    cell_str = f"{cell[0]}_{cell[1]}_{cell[2]}"
    path = os.path.join(cache_dir, CHUNKS_DIRNAME, f"{cell_str}.bin")
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise ValueError(f"Bad chunk file magic in {path}")
        version = struct.unpack("<I", f.read(4))[0]
        n_groups = struct.unpack("<I", f.read(4))[0]

        groups = {}
        all_pos = []
        for _ in range(n_groups):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            n_verts = struct.unpack("<I", f.read(4))[0]

            pos_bytes = f.read(n_verts * 3 * 4)
            uv_bytes = f.read(n_verts * 2 * 4)
            nrm_bytes = f.read(n_verts * 3 * 4)

            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(n_verts, 3)
            uvs = np.frombuffer(uv_bytes, dtype=np.float32).reshape(n_verts, 2)
            normals = np.frombuffer(nrm_bytes, dtype=np.float32).reshape(n_verts, 3)

            groups[name] = ChunkMaterialGroup(name, positions, uvs, normals)
            all_pos.append(positions)

    stacked = np.concatenate(all_pos, axis=0) if all_pos else np.zeros((0, 3), np.float32)
    bmin = stacked.min(axis=0) if len(stacked) else np.zeros(3, np.float32)
    bmax = stacked.max(axis=0) if len(stacked) else np.zeros(3, np.float32)

    return ChunkData(cell=cell, groups=groups, bounds_min=bmin, bounds_max=bmax)


def cache_is_valid(obj_path: str) -> bool:
    """Cache is valid if it exists and is newer than the source OBJ (cheap
    staleness check so re-running on the same map doesn't reparse 2GB)."""
    obj_dir = os.path.dirname(os.path.abspath(obj_path))
    cache_dir = os.path.join(obj_dir, CACHE_DIRNAME)
    manifest_path = os.path.join(cache_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return False
    return os.path.getmtime(manifest_path) >= os.path.getmtime(obj_path)


def get_cache_dir(obj_path: str) -> str:
    obj_dir = os.path.dirname(os.path.abspath(obj_path))
    return os.path.join(obj_dir, CACHE_DIRNAME)


def find_landing_position(manifest: dict, target_x: float, target_z: float,
                            preferred_y: float, search_radius_cells: int = 12) -> tuple[float, float, float]:
    """
    Given a target world (x, z) -- e.g. from a minimap click, which only
    knows X/Z -- finds a world (x, y, z) that actually lands inside the
    cave's occupied space near that column, rather than blindly keeping
    whatever Y the camera happened to be at before.

    Strategy: look at every chunk cell whose (x, z) column matches the
    target cell (collapsing Y, same idea as the minimap's footprint). Each
    matching cell's vertical center (midpoint of its bounds_min/max Y) is
    a candidate landing height; pick whichever candidate is closest to
    `preferred_y` (typically the camera's current height) so a multi-level
    cave doesn't always snap you to the lowest or first-found level --
    if you're already up high and click a spot that has both a low and a
    high passage, you land in the one nearer to where you already were.

    If no chunk exists at that exact (x, z) column (a click slightly off
    from any real passage on the crude minimap outline, since chunk cells
    are coarse), the search expands outward ring by ring up to
    `search_radius_cells` until it finds the nearest occupied column, and
    targets the center of THAT column's cells instead -- so a near-miss
    click still lands you inside the cave rather than in empty space.

    search_radius_cells defaults to 12 (not a small number like 3) because
    a thin, winding cave passage drawn on a coarse minimap is easy to
    click slightly off of -- especially on a long straight stretch, where
    the click error needed to miss the passage entirely doesn't need to
    be large. A too-small search radius meant some clicks fell through
    every ring with nothing found, landing the camera in genuinely empty
    space with zero chunks anywhere nearby (visible as "CHUNKS 0" forever
    and a loading panel that never finds anything to load).

    If even the expanded ring search finds nothing (a pathological case,
    e.g. an extremely sparse or disconnected map), this falls back to the
    single closest occupied column anywhere in the ENTIRE map, rather
    than giving up and teleporting into empty space -- guaranteeing this
    function always lands you somewhere inside the cave if the cave has
    any chunks at all.

    Returns (landing_x, landing_y, landing_z). landing_x/z may differ
    significantly from target_x/z if the fallback search had to reach far
    to find any occupied column at all.
    """
    chunk_size = manifest["chunk_size"]
    target_cx = int(np.floor(target_x / chunk_size))
    target_cz = int(np.floor(target_z / chunk_size))

    # Build a quick lookup: (cx, cz) -> list of (y_center, cell_str) for
    # every cell in that column, across all Y levels.
    columns: dict[tuple[int, int], list[tuple[float, str]]] = {}
    for cell_str, info in manifest["chunks"].items():
        cx, cy, cz = (int(v) for v in cell_str.split("_"))
        y_center = (info["bounds_min"][1] + info["bounds_max"][1]) / 2.0
        columns.setdefault((cx, cz), []).append((y_center, cell_str))

    def best_y_in_column(cx: int, cz: int) -> float | None:
        candidates = columns.get((cx, cz))
        if not candidates:
            return None
        # closest to preferred_y, so multi-level caves keep you near your
        # current level rather than always jumping to one extreme
        return min(candidates, key=lambda c: abs(c[0] - preferred_y))[0]

    # exact column first
    y = best_y_in_column(target_cx, target_cz)
    if y is not None:
        return target_x, y, target_z

    # expand outward ring by ring looking for the nearest occupied column
    for radius in range(1, search_radius_cells + 1):
        best_dist = None
        best_col = None
        best_y_val = None
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if max(abs(dx), abs(dz)) != radius:
                    continue  # only the new outer ring at this radius, inner rings already checked
                col = (target_cx + dx, target_cz + dz)
                y_val = best_y_in_column(*col)
                if y_val is None:
                    continue
                dist = dx * dx + dz * dz
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_col = col
                    best_y_val = y_val
        if best_col is not None:
            landing_x = (best_col[0] + 0.5) * chunk_size
            landing_z = (best_col[1] + 0.5) * chunk_size
            return landing_x, best_y_val, landing_z

    # Ring search exhausted with nothing found -- rather than teleport
    # into empty space (the actual bug this fixes), fall back to a full
    # scan of every occupied column in the manifest and pick whichever is
    # closest to the original click. This is more expensive (O(number of
    # chunks)) but only runs in this rare fallback case, and guarantees a
    # minimap click always lands somewhere inside the cave if the cave
    # has any chunks loaded into the manifest at all.
    best_dist = None
    best_col = None
    best_y_val = None
    for (cx, cz), candidates in columns.items():
        dist = (cx - target_cx) ** 2 + (cz - target_cz) ** 2
        if best_dist is None or dist < best_dist:
            y_val = min(candidates, key=lambda c: abs(c[0] - preferred_y))[0]
            best_dist = dist
            best_col = (cx, cz)
            best_y_val = y_val

    if best_col is not None:
        landing_x = (best_col[0] + 0.5) * chunk_size
        landing_z = (best_col[1] + 0.5) * chunk_size
        return landing_x, best_y_val, landing_z

    # truly no chunks exist anywhere in the manifest (an empty/corrupt
    # cache) -- nothing sensible to land on, so fall back to the original
    # behavior of just keeping preferred_y rather than raising.
    return target_x, preferred_y, target_z
