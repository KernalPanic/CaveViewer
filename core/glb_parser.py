"""
core/glb_parser.py

Parses GLB (binary glTF) files into the same RawMesh shape
core/obj_parser.py produces for OBJ -- see that module's RawMesh
docstring for the exact field contract this must satisfy.

Built on the `pygltflib` library (PyPI: pygltflib) rather than hand-
writing a glTF/GLB parser. glTF is a structured, precisely-specified
format (JSON scene description + packed binary buffers for vertex/index
data, optionally with embedded images), and pygltflib already handles
the format's full structure correctly, including the buffer-view/
accessor indirection glTF uses to describe how raw bytes map to typed
arrays.

Key structural differences from OBJ that this parser has to bridge:
  - glTF organizes geometry into "meshes" containing "primitives" (each
    primitive is one draw call's worth of geometry + one material) --
    roughly analogous to OBJ's usemtl-delimited material ranges, but
    accessed very differently (each primitive has its own accessor
    indices into the shared binary buffer, rather than OBJ's flat
    contiguous-face-range convention). This parser concatenates all
    primitives from all meshes in the scene into one flat RawMesh, with
    one MaterialRange per primitive -- which is exactly the same shape
    obj_parser.py already produces per usemtl block, just sourced
    differently.
  - glTF positions/normals/UVs are NOT separately indexed per attribute
    the way OBJ's v/vt/vn can be -- a glTF primitive's single index
    buffer addresses one shared set of "vertices" where each vertex
    already has its position+normal+UV bundled together. This is
    actually a SIMPLER shape than OBJ's, and converts cleanly: this
    parser treats each primitive's own vertex range as if it were
    OBJ-style "one shared index per attribute," which is correct since
    that's exactly what a glTF vertex already is.
  - Textures are commonly EMBEDDED inside the .glb file's binary blob
    rather than referenced as separate files on disk. This parser
    extracts embedded image bytes directly (see _extract_embedded_images)
    and returns them as raw bytes rather than filenames -- TextureManager
    (core/texture_manager.py) was extended specifically to accept either
    a filename OR raw bytes per material for this reason.

IMPORTANT CAVEAT: this module's actual pygltflib API calls are NOT
verified against a real install in this development environment (no
internet access to install third-party packages here) -- only the
RawMesh conversion logic around them has been tested directly, using a
hand-built fake object matching pygltflib's documented GLTF2/accessor/
bufferView shape. If something about the real pygltflib API differs from
what's assumed below, that's the most likely place an issue would
surface on first real use.
"""

from __future__ import annotations

import os
import struct
from typing import Optional

import numpy as np

from core.obj_parser import RawMesh, MaterialRange


# glTF accessor component type codes -> numpy dtype, per the glTF 2.0 spec
# (these are fixed, standardized integer codes, not something that varies
# by exporter -- this mapping is safe to hard-code).
_COMPONENT_TYPE_TO_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_TO_NUM_COMPONENTS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}


def parse_glb(glb_path: str, progress_cb=None) -> tuple[RawMesh, dict]:
    """
    Parses a GLB file into a RawMesh, plus {material_name: embedded_image_bytes}
    for any materials whose texture was embedded directly inside the file
    (the common case for GLB specifically, as opposed to .gltf+.bin+loose
    images, though this parser handles either since pygltflib abstracts
    over both).

    Returns (mesh, material_to_embedded_bytes) -- the second dict is
    merged into the material-to-texture mapping caveviewer.py builds,
    alongside any materials that instead reference an external image file
    by relative path (handled the same way an OBJ's .mtl reference would
    be, since pygltflib resolves those to plain filenames too).
    """
    from pygltflib import GLTF2

    if progress_cb:
        progress_cb("reading GLB file", 0.0)

    gltf = GLTF2().load(glb_path)

    if progress_cb:
        progress_cb("reading GLB file", 0.2)

    # GLB embeds its binary buffer data directly in the file; pygltflib
    # exposes this via get_data_from_buffer_uri / binary_blob() depending
    # on version, but the stable, documented way to get a buffer's raw
    # bytes for any glTF (embedded or external) is via this helper.
    def get_buffer_bytes(buffer_index: int) -> bytes:
        return gltf.get_data_from_buffer_uri(gltf.buffers[buffer_index].uri) \
            if gltf.buffers[buffer_index].uri else gltf.binary_blob()

    def read_accessor(accessor_index: int) -> np.ndarray:
        """Resolves one glTF accessor into a numpy array, following the
        accessor -> bufferView -> buffer indirection the format uses to
        describe how raw bytes map to typed values."""
        accessor = gltf.accessors[accessor_index]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        raw_bytes = get_buffer_bytes(buffer_view.buffer)

        dtype = _COMPONENT_TYPE_TO_DTYPE[accessor.componentType]
        n_components = _TYPE_TO_NUM_COMPONENTS[accessor.type]

        start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        count = accessor.count

        # byteStride, if present, means the data isn't tightly packed
        # (interleaved with other attributes) -- read it out element by
        # element in that case rather than assuming a flat contiguous run.
        stride = buffer_view.byteStride
        element_size = n_components * np.dtype(dtype).itemsize

        if stride and stride != element_size:
            values = np.empty((count, n_components), dtype=dtype)
            for i in range(count):
                offset = start + i * stride
                values[i] = np.frombuffer(raw_bytes, dtype=dtype, count=n_components, offset=offset)
            return values

        flat = np.frombuffer(raw_bytes, dtype=dtype, count=count * n_components, offset=start)
        return flat.reshape((count, n_components)) if n_components > 1 else flat

    if progress_cb:
        progress_cb("reading mesh primitives", 0.35)

    all_positions = []
    all_uvs = []
    all_normals = []
    all_face_idx = []
    material_ranges = []

    vertex_offset = 0
    face_offset = 0

    # Walk every mesh's every primitive, in order -- this fixed,
    # deterministic order is what makes "one MaterialRange per primitive,
    # in the order encountered" a correct, stable mapping.
    for mesh_idx, gltf_mesh in enumerate(gltf.meshes):
        for prim_idx, primitive in enumerate(gltf_mesh.primitives):
            pos_accessor_idx = primitive.attributes.POSITION
            if pos_accessor_idx is None:
                continue  # a primitive with no positions is degenerate; skip it

            positions = read_accessor(pos_accessor_idx).astype(np.float32)
            n_verts_this_prim = positions.shape[0]
            all_positions.append(positions)

            uv_accessor_idx = primitive.attributes.TEXCOORD_0
            if uv_accessor_idx is not None:
                uvs = read_accessor(uv_accessor_idx).astype(np.float32)
            else:
                uvs = np.zeros((n_verts_this_prim, 2), dtype=np.float32)
            all_uvs.append(uvs)

            normal_accessor_idx = primitive.attributes.NORMAL
            if normal_accessor_idx is not None:
                normals = read_accessor(normal_accessor_idx).astype(np.float32)
            else:
                normals = np.zeros((n_verts_this_prim, 3), dtype=np.float32)
            all_normals.append(normals)

            if primitive.indices is not None:
                indices = read_accessor(primitive.indices).astype(np.int32).reshape(-1)
            else:
                # no index buffer means the vertex stream is already in
                # draw order, implicitly 0,1,2,3,4,5...
                indices = np.arange(n_verts_this_prim, dtype=np.int32)

            # glTF primitives are required to be triangle lists by
            # default (mode 4, TRIANGLES) -- the overwhelmingly common
            # case for any exported scan -- so no fan-triangulation
            # needed here the way OBJ/PLY can require; just reshape the
            # flat index stream into (N, 3) triangles directly.
            n_tris_this_prim = len(indices) // 3
            tris = indices[: n_tris_this_prim * 3].reshape((n_tris_this_prim, 3)) + vertex_offset
            all_face_idx.append(tris)

            material_name = f"gltf_material_{primitive.material}" if primitive.material is not None \
                else f"gltf_mesh{mesh_idx}_prim{prim_idx}_untextured"
            material_ranges.append(MaterialRange(
                material_name=material_name,
                start_face=face_offset,
                end_face=face_offset + n_tris_this_prim,
            ))

            vertex_offset += n_verts_this_prim
            face_offset += n_tris_this_prim

    if progress_cb:
        progress_cb("assembling mesh", 0.7)

    positions = np.concatenate(all_positions, axis=0) if all_positions else np.zeros((0, 3), dtype=np.float32)
    uvs = np.concatenate(all_uvs, axis=0) if all_uvs else np.zeros((0, 2), dtype=np.float32)
    normals = np.concatenate(all_normals, axis=0) if all_normals else np.zeros((0, 3), dtype=np.float32)
    face_pos_idx = np.concatenate(all_face_idx, axis=0) if all_face_idx else np.zeros((0, 3), dtype=np.int32)

    # glTF vertices already bundle position+UV+normal together (one
    # shared index per vertex, unlike OBJ's separate v/vt/vn index
    # streams) -- so the UV/normal index for any triangle corner is
    # simply the same index used for its position, since they're already
    # the same array length and correspondence.
    face_uv_idx = face_pos_idx.copy()
    face_nrm_idx = face_pos_idx.copy()

    mesh = RawMesh(
        positions=positions,
        uvs=uvs,
        normals=normals,
        face_pos_idx=face_pos_idx,
        face_uv_idx=face_uv_idx,
        face_nrm_idx=face_nrm_idx,
        material_ranges=material_ranges,
        mtl_file=None,
    )

    if progress_cb:
        progress_cb("extracting embedded textures", 0.9)

    material_to_embedded_bytes = _extract_embedded_images(gltf, get_buffer_bytes)

    if progress_cb:
        progress_cb("done", 1.0)

    return mesh, material_to_embedded_bytes


def _extract_embedded_images(gltf, get_buffer_bytes) -> dict:
    """
    Maps each material (by the same synthetic "gltf_material_<index>"
    name used above) to its raw embedded image bytes, for any material
    whose texture's image data lives inside a bufferView (the standard
    way GLB embeds images) rather than as an external file URI.

    Materials whose texture instead references an external file (a plain
    relative-path URI, common for .gltf+.bin+loose-images bundles rather
    than single-file .glb) are intentionally NOT included here -- the
    caller falls back to treating that case as an ordinary filename, the
    same as OBJ/.mtl's existing convention, since TextureManager already
    knows how to read a plain file from textures_dir.
    """
    result = {}

    for material_idx, material in enumerate(gltf.materials or []):
        pbr = getattr(material, "pbrMetallicRoughness", None)
        if pbr is None or pbr.baseColorTexture is None:
            continue

        texture_idx = pbr.baseColorTexture.index
        texture = gltf.textures[texture_idx]
        if texture.source is None:
            continue

        image = gltf.images[texture.source]

        if image.bufferView is not None:
            # embedded: the image's raw encoded bytes (JPEG/PNG file
            # bytes, exactly as if you'd opened the .jpg/.png on disk)
            # live directly in a bufferView, no separate accessor
            # indirection needed for images specifically (unlike vertex
            # data) -- bufferViews for images just point at a contiguous
            # byte range holding the already-encoded image file.
            buffer_view = gltf.bufferViews[image.bufferView]
            raw_bytes = get_buffer_bytes(buffer_view.buffer)
            start = buffer_view.byteOffset or 0
            length = buffer_view.byteLength
            image_bytes = raw_bytes[start:start + length]
            result[f"gltf_material_{material_idx}"] = image_bytes
        # else: image.uri points at an external file -- left for the
        # caller to handle as a plain filename, same as OBJ's convention.

    return result
