"""
core/texture_manager.py

Lazy GPU texture loading/eviction keyed by material name, kept in lockstep
with chunk streaming. This is the second half of the VRAM-saving strategy
(the first half is geometry chunking in chunker.py / streaming_world.py):
even though geometry chunks reference materials, we don't want to upload
every texture tile in the whole cave map upfront -- only the tiles actually
touched by currently-loaded chunks.

Reference-counted: a texture is only evicted once no currently-loaded chunk
still references it. This is necessary because multiple chunks can (and do)
share a texture tile.

Decode vs upload are deliberately split into two separate steps:
  - decode_from_disk() does JPEG decoding (Pillow) and pixel manipulation
    (numpy) only -- no OpenGL calls at all, so it is SAFE to run on a
    background worker thread. This is the expensive, variable-cost part
    (can take anywhere from <1ms to 10+ms depending on image size).
  - upload_decoded() takes already-decoded pixel bytes and does the actual
    ctx.texture(...) GPU call -- this MUST happen on the main/render thread
    (an OpenGL/driver requirement), but it's comparatively fast and
    consistent once decoding is already done.

This split exists because chunk streaming runs texture decode on background
worker threads (see core/streaming_world.py's _worker_loop, which now also
pre-decodes textures alongside loading geometry) so that by the time a
chunk reaches the main thread for GPU upload, the slow/unpredictable part
(JPEG decode) is already finished -- only the fast, predictable GPU upload
remains on the main thread, which is what keeps frame times smooth during
a fast flythrough that streams in many never-before-seen textures at once.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import numpy as np

# Pillow's default decompression-bomb guard rejects images above ~179
# million pixels, as a safety measure against maliciously crafted image
# files designed to exhaust memory when decoded. Photogrammetry/3D-scan
# texture atlases (including some downloaded from sites like Sketchfab)
# can legitimately exceed that -- a single 16000x16000 texture tile is
# 256 million pixels and is a completely normal thing for a textured
# scan to ship with, not an attack. Raise the limit rather than disable
# the check outright, so genuinely absurd files (a 1,000,000,000+ pixel
# image, which no legitimate texture tile would ever be) still get
# caught; this project's own decode path also wraps every Image.open()
# call in a try/except (see _decode_from_disk below) as a second layer
# of protection regardless of where the threshold ends up sitting.
Image.MAX_IMAGE_PIXELS = 1_000_000_000


@dataclass
class DecodedImage:
    """Result of decode_from_disk(): plain CPU-side data, no GPU objects,
    safe to pass between threads."""
    size: tuple[int, int]
    components: int
    data: bytes


@dataclass
class LoadedTexture:
    moderngl_texture: object   # the moderngl.Texture instance
    ref_count: int = 0


class TextureManager:
    def __init__(self, gl_context, textures_dir: str, material_to_file: dict):
        """
        gl_context: a moderngl.Context (or any object exposing .texture(size, components, data))
        textures_dir: folder containing texture files referenced by filename
            (ignored for any material whose value below is raw bytes
            rather than a filename, since there's nothing on disk to
            look up in that case).
        material_to_file: {material_name: value}, where value is one of:
            - a str filename, relative to textures_dir (OBJ/.mtl's
              convention -- a separate image file on disk)
            - raw image bytes (e.g. JPEG/PNG file bytes) -- used for
              formats like GLB/glTF, which commonly embed texture image
              data directly inside the model file rather than as
              separate files alongside it. The bytes themselves serve as
              the cache key (they're hashable), so two materials sharing
              the exact same embedded image are still deduplicated the
              same way two materials sharing one filename already are.
            - None -- no texture for this material (placeholder used)
        """
        self.ctx = gl_context
        self.textures_dir = textures_dir
        self.material_to_file = material_to_file
        self._loaded: dict[str, LoadedTexture] = {}  # keyed by material name
        # multiple materials can point at the same physical jpg (rare, but
        # cheap to dedupe so we don't decode the same file twice) -- or,
        # for embedded textures, the same raw bytes object/value
        self._file_cache: dict[object, object] = {}  # filename-or-bytes -> moderngl.Texture

        # Decoded-but-not-yet-uploaded images, populated by background
        # worker threads via decode_from_disk(), consumed on the main
        # thread via upload_decoded(). Guarded by a lock since multiple
        # worker threads may populate it concurrently.
        self._decode_cache: dict[object, DecodedImage] = {}
        self._decode_cache_lock = threading.Lock()

    def _placeholder_texture(self):
        """1x1 magenta texture used when a material's image file is missing,
        so a bad texture reference degrades visibly (obviously wrong color)
        instead of crashing the whole viewer mid-flythrough."""
        data = np.array([[255, 0, 255, 255]], dtype=np.uint8).tobytes()
        tex = self.ctx.texture((1, 1), 4, data)
        return tex

    # -- background-thread-safe decode step ----------------------------------

    def decode_for_material(self, material_name: str) -> None:
        """
        Safe to call from any thread (no OpenGL calls). Decodes the image
        for `material_name`'s texture (whether that's a file on disk or
        embedded raw bytes -- see __init__'s docstring), if not already
        decoded or already uploaded, and stashes the raw pixel data for
        upload_decoded() to pick up later on the main thread. No-op if the
        texture is already GPU-resident or already sitting decoded-and-
        waiting.
        """
        file_or_bytes = self.material_to_file.get(material_name)
        if not file_or_bytes:
            return  # no texture for this material; placeholder path handles it on upload
        if file_or_bytes in self._file_cache:
            return  # already GPU-resident
        with self._decode_cache_lock:
            if file_or_bytes in self._decode_cache:
                return  # already decoded, waiting for main-thread upload

        decoded = self._decode_image(file_or_bytes)
        with self._decode_cache_lock:
            # double check another thread didn't decode it in the meantime;
            # if so, just discard our redundant decode rather than overwrite
            if file_or_bytes not in self._decode_cache and file_or_bytes not in self._file_cache:
                self._decode_cache[file_or_bytes] = decoded

    def _decode_image(self, file_or_bytes) -> Optional[DecodedImage]:
        """
        Dispatches to the disk-file decode path or the in-memory-bytes
        decode path depending on the type of `file_or_bytes` -- str means
        "filename relative to textures_dir" (the existing OBJ/.mtl
        convention), bytes means "raw embedded image data" (the GLB/
        glTF convention). Both ultimately go through the same Pillow
        Image.open() + vertical-flip + RGB-conversion logic; only how the
        bytes are obtained differs.
        """
        if isinstance(file_or_bytes, bytes):
            return self._decode_from_bytes(file_or_bytes)
        return self._decode_from_disk(file_or_bytes)

    def _decode_from_bytes(self, raw_bytes: bytes) -> Optional[DecodedImage]:
        """Same decode logic as _decode_from_disk, but for image data
        that's already in memory (embedded in the model file) rather
        than a separate file to open from disk -- see GLB/glTF support
        in core/glb_parser.py."""
        try:
            import io
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            # same vertical flip as the disk-file path, for the same
            # reason: OBJ/OpenGL UV origin is bottom-left, most image
            # libraries decode top-left first.
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            return DecodedImage(size=img.size, components=3, data=img.tobytes())
        except Exception as e:
            print(f"[TextureManager] WARNING: failed to decode an embedded texture: {e}")
            return None

    def _decode_from_disk(self, filename: str) -> Optional[DecodedImage]:
        path = os.path.join(self.textures_dir, filename)
        if not os.path.exists(path):
            print(f"[TextureManager] WARNING: texture file missing: {path}")
            return None

        try:
            img = Image.open(path).convert("RGB")
            # flip vertically: OBJ/OpenGL UV origin is bottom-left, most
            # image libraries decode top-left first -- skipping this
            # produces an upside-down or mirrored-look texture that's
            # easy to misdiagnose as a UV bug in the OBJ export itself.
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            return DecodedImage(size=img.size, components=3, data=img.tobytes())
        except Exception as e:
            # Any decode failure -- a corrupt file, an unsupported/
            # unrecognized format, a truncated download, an image that's
            # still too large even after raising MAX_IMAGE_PIXELS above,
            # or anything else Pillow might raise -- degrades to a
            # missing-texture placeholder instead of taking down the
            # whole viewer. A wrong-looking (magenta) texture on one
            # chunk is recoverable; a crashed app mid-flythrough is not.
            print(f"[TextureManager] WARNING: failed to decode texture "
                  f"'{filename}': {e}")
            return None

    # -- main-thread-only GPU upload step ------------------------------------

    def acquire(self, material_name: str) -> object:
        """
        Increment refcount for `material_name`'s texture, uploading it to
        the GPU on first use. MUST be called from the main/render thread.

        If decode_for_material() was already called for this material on a
        background thread (the normal streaming path), this just does the
        fast GPU upload of already-decoded pixels. If not (e.g. a texture
        needed before its background decode finished, or this manager is
        used standalone), it falls back to decoding synchronously here --
        slower, but still correct.
        """
        if material_name in self._loaded:
            entry = self._loaded[material_name]
            entry.ref_count += 1
            return entry.moderngl_texture

        file_or_bytes = self.material_to_file.get(material_name)
        if file_or_bytes and file_or_bytes in self._file_cache:
            tex = self._file_cache[file_or_bytes]
        else:
            tex = self._upload_for_material(material_name, file_or_bytes)
            if file_or_bytes:
                self._file_cache[file_or_bytes] = tex

        self._loaded[material_name] = LoadedTexture(moderngl_texture=tex, ref_count=1)
        return tex

    def _upload_for_material(self, material_name: str, file_or_bytes) -> object:
        if not file_or_bytes:
            return self._placeholder_texture()

        decoded = None
        with self._decode_cache_lock:
            decoded = self._decode_cache.pop(file_or_bytes, None)

        if decoded is None:
            # fallback: decode synchronously on the main thread. Slower
            # (this is the case we're trying to avoid via pre-decoding),
            # but correctness matters more than speed here.
            decoded = self._decode_image(file_or_bytes)

        if decoded is None:
            return self._placeholder_texture()

        tex = self.ctx.texture(decoded.size, decoded.components, decoded.data)
        if hasattr(tex, "build_mipmaps"):
            tex.build_mipmaps()
        return tex

    def release(self, material_name: str) -> None:
        """Decrement refcount; free the GPU texture once it hits zero."""
        entry = self._loaded.get(material_name)
        if entry is None:
            return
        entry.ref_count -= 1
        if entry.ref_count <= 0:
            del self._loaded[material_name]
            # only release the underlying GPU texture if no other material
            # alias still points at the same file
            filename = self.material_to_file.get(material_name)
            still_used = any(
                self.material_to_file.get(m) == filename
                for m in self._loaded
            )
            if filename and not still_used and filename in self._file_cache:
                tex = self._file_cache.pop(filename)
                if hasattr(tex, "release"):
                    tex.release()

    def loaded_count(self) -> int:
        return len(self._file_cache)

    def stats(self) -> dict:
        with self._decode_cache_lock:
            n_decoded_waiting = len(self._decode_cache)
        return {
            "unique_materials_loaded": len(self._loaded),
            "unique_files_resident": len(self._file_cache),
            "decoded_waiting_for_upload": n_decoded_waiting,
        }
