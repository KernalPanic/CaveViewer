"""
core/obj_parser.py

Streaming parser for large Wavefront OBJ files produced by Agisoft Metashape.

Design constraints driving this implementation:
  - Source files can be 500MB-2GB+ with 5-20M triangles.
  - We must NOT load the file into a single Python list-of-tuples structure;
    Python object overhead would multiply memory use 10-20x.
  - We do a single streaming pass, writing directly into pre-sized numpy
    arrays. Since OBJ doesn't tell us face count up front, we do a fast
    line-count pre-pass (just counting 'f ' / 'v ' prefixes) to size arrays,
    then a second pass to actually fill them. Two passes over text is much
    cheaper than dynamic Python list growth + final conversion.

Agisoft OBJ exports are well-behaved: vertices are 'v x y z', texture coords
are 'vt u v', normals are 'vn x y z' (sometimes omitted), and faces are
'f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3' (always triangulated on export, but we
defensively fan-triangulate anything with >3 verts). Faces reference a
'usemtl <name>' that switches the active material/texture tile.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

_FACE_VERT_RE = re.compile(r"(-?\d+)(?:/(-?\d*)(?:/(-?\d*))?)?")


@dataclass
class MaterialRange:
    """A contiguous run of faces in the global face array that use one material."""
    material_name: str
    start_face: int   # inclusive
    end_face: int     # exclusive


@dataclass
class RawMesh:
    """
    Result of parsing one OBJ. Index arrays are 0-based into `positions` /
    `uvs` / `normals`, already resolved (OBJ's own indices are 1-based and
    support negative/relative indexing; we normalize both away here so
    nothing downstream has to think about OBJ quirks again).
    """
    positions: np.ndarray            # (Nv, 3) float32
    uvs: np.ndarray                  # (Nvt, 2) float32, may be zero-length
    normals: np.ndarray              # (Nvn, 3) float32, may be zero-length

    face_pos_idx: np.ndarray         # (Nf, 3) int32 -> positions
    face_uv_idx: np.ndarray          # (Nf, 3) int32 -> uvs (or -1 if none)
    face_nrm_idx: np.ndarray         # (Nf, 3) int32 -> normals (or -1 if none)

    material_ranges: list[MaterialRange] = field(default_factory=list)
    mtl_file: str | None = None


def _count_prepass(obj_path: str) -> tuple[int, int, int, int]:
    """
    Fast first pass: count vertices/uvs/normals/faces (post-triangulation
    estimate) so we can pre-allocate numpy arrays of the right size instead
    of growing them dynamically. Triangulation count assumes worst case of
    triangle-or-quad faces (Agisoft only ever emits these), counting a quad
    as 2 triangles.
    """
    n_v = n_vt = n_vn = n_f = 0
    with open(obj_path, "r", buffering=1024 * 1024, errors="replace") as fh:
        for line in fh:
            if not line:
                continue
            prefix = line[:2]
            if prefix == "v ":
                n_v += 1
            elif prefix == "vt":
                n_vt += 1
            elif prefix == "vn":
                n_vn += 1
            elif prefix == "f ":
                n_tokens = len(line.split()) - 1
                if n_tokens >= 3:
                    n_f += n_tokens - 2
    return n_v, n_vt, n_vn, n_f


def _resolve_index(raw: int, count_so_far: int) -> int:
    """OBJ indices are 1-based; negative indices count back from the most
    recently defined element. Convert both to a normal 0-based index."""
    if raw > 0:
        return raw - 1
    if raw < 0:
        return count_so_far + raw
    raise ValueError("OBJ index of 0 is invalid")


def parse_obj(obj_path: str, progress_cb=None) -> RawMesh:
    """
    Parse `obj_path` into a RawMesh.

    progress_cb(stage: str, fraction: float) is called periodically if given,
    so a GUI can show a progress bar during the (one-time, then cached)
    import of a large map.
    """
    if progress_cb:
        progress_cb("scanning file", 0.0)
    n_v, n_vt, n_vn, n_f_est = _count_prepass(obj_path)

    positions = np.empty((n_v, 3), dtype=np.float32)
    uvs = np.empty((n_vt, 2), dtype=np.float32)
    normals = np.empty((n_vn, 3), dtype=np.float32)

    face_pos_idx = np.empty((n_f_est, 3), dtype=np.int32)
    face_uv_idx = np.full((n_f_est, 3), -1, dtype=np.int32)
    face_nrm_idx = np.full((n_f_est, 3), -1, dtype=np.int32)

    material_ranges: list[MaterialRange] = []
    mtl_file = None

    vi = vti = vni = fi = 0
    current_material = None
    current_material_start_face = 0

    file_size = os.path.getsize(obj_path)
    bytes_read = 0
    last_reported = 0.0

    with open(obj_path, "r", buffering=1024 * 1024, errors="replace") as fh:
        for line in fh:
            bytes_read += len(line)
            if progress_cb and file_size:
                frac = bytes_read / file_size
                if frac - last_reported > 0.01:
                    progress_cb("parsing geometry", frac)
                    last_reported = frac

            if not line or line[0] == "#":
                continue

            if line[:2] == "v ":
                parts = line.split()
                positions[vi, 0] = float(parts[1])
                positions[vi, 1] = float(parts[2])
                positions[vi, 2] = float(parts[3])
                vi += 1

            elif line[:3] == "vt ":
                parts = line.split()
                uvs[vti, 0] = float(parts[1])
                uvs[vti, 1] = float(parts[2])
                vti += 1

            elif line[:3] == "vn ":
                parts = line.split()
                normals[vni, 0] = float(parts[1])
                normals[vni, 1] = float(parts[2])
                normals[vni, 2] = float(parts[3])
                vni += 1

            elif line[:2] == "f ":
                tokens = line.split()[1:]
                verts = []
                for tok in tokens:
                    m = _FACE_VERT_RE.match(tok)
                    if not m:
                        continue
                    p_raw = int(m.group(1))
                    p_idx = _resolve_index(p_raw, vi)
                    uv_idx = -1
                    nrm_idx = -1
                    if m.group(2):
                        uv_idx = _resolve_index(int(m.group(2)), vti)
                    if m.group(3):
                        nrm_idx = _resolve_index(int(m.group(3)), vni)
                    verts.append((p_idx, uv_idx, nrm_idx))

                # fan-triangulate (handles tris natively, n-gons defensively)
                for k in range(1, len(verts) - 1):
                    a, b, c = verts[0], verts[k], verts[k + 1]
                    face_pos_idx[fi] = (a[0], b[0], c[0])
                    face_uv_idx[fi] = (a[1], b[1], c[1])
                    face_nrm_idx[fi] = (a[2], b[2], c[2])
                    fi += 1

            elif line[:7] == "usemtl ":
                name = line.split(maxsplit=1)[1].strip()
                if current_material is not None and fi > current_material_start_face:
                    material_ranges.append(MaterialRange(
                        current_material, current_material_start_face, fi))
                current_material = name
                current_material_start_face = fi

            elif line[:7] == "mtllib ":
                mtl_file = line.split(maxsplit=1)[1].strip()

    if current_material is not None and fi > current_material_start_face:
        material_ranges.append(MaterialRange(
            current_material, current_material_start_face, fi))

    if progress_cb:
        progress_cb("done", 1.0)

    return RawMesh(
        positions=positions,
        uvs=uvs,
        normals=normals,
        face_pos_idx=face_pos_idx[:fi],
        face_uv_idx=face_uv_idx[:fi],
        face_nrm_idx=face_nrm_idx[:fi],
        material_ranges=material_ranges,
        mtl_file=mtl_file,
    )


@dataclass
class Material:
    name: str
    diffuse_texture: str | None  # filename relative to the mtl's folder


def parse_mtl(mtl_path: str) -> dict[str, Material]:
    """Parse a .mtl file into {material_name: Material}."""
    materials: dict[str, Material] = {}
    current_name = None
    current_tex = None

    with open(mtl_path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("newmtl "):
                if current_name is not None:
                    materials[current_name] = Material(current_name, current_tex)
                current_name = line.split(maxsplit=1)[1].strip()
                current_tex = None
            elif line.startswith("map_Kd "):
                current_tex = line.split(maxsplit=1)[1].strip()

    if current_name is not None:
        materials[current_name] = Material(current_name, current_tex)

    return materials
