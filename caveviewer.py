#!/usr/bin/env python3
"""
caveviewer.py

CaveViewer 1.0 -- entry point.

Workflow:
  1. User picks a folder containing the Agisoft export (.obj + .mtl + .jpg
     texture tiles).
  2. We find the .obj/.mtl, and check whether a valid chunk cache already
     exists (built on a previous run). If valid, skip straight to step 4.
  3. If no valid cache: parse the OBJ (streaming, handles 2GB+ files) and
     build the spatial chunk cache on disk -- this is the one-time cost
     that makes all future loads of this same map instant. Shows progress.
  4. Launch the OpenGL viewer window, which streams chunks in/out based on
     where the user flies, so frame rate stays smooth regardless of total
     map size.

Bare-bones UI for now per your request -- a Tkinter folder-picker dialog
and a console progress readout, nothing fancier. We can layer a nicer UI
on top later without touching any of the core/ engine code.
"""

import os
import sys
import glob
import time

# Single source of truth for the app's version. Compared against GitHub
# Releases' tag names by the auto-updater (gui/updater.py) -- bump this
# whenever a new release is tagged. Tags should be plain version strings
# WITHOUT a leading "v" (e.g. "1.1" not "v1.1") to keep the string
# comparison in updater.py simple; if a "v" prefix is preferred for
# GitHub convention later, updater.py's comparison would need a small
# adjustment to strip it.
__version__ = "1.0"

# Make sure 'core' and 'gui' packages are importable regardless of the
# directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_input_files(folder: str) -> tuple[str, str]:
    """Locate the .obj and its .mtl inside `folder`. Returns (obj_path, mtl_path).
    Raises a clear error if the folder doesn't contain what we expect, since
    a confusing stack trace here would be a bad first impression of the tool."""
    obj_candidates = glob.glob(os.path.join(folder, "*.obj"))
    if not obj_candidates:
        raise FileNotFoundError(
            f"No .obj file found in:\n  {folder}\n\n"
            f"Make sure you selected the folder that contains the exported "
            f".obj, .mtl, and .jpg texture tiles from Agisoft."
        )
    if len(obj_candidates) > 1:
        print(f"Note: multiple .obj files found, using the first one: {obj_candidates[0]}")
    obj_path = obj_candidates[0]

    from core.obj_parser import parse_obj  # local import; heavy-ish module

    # peek at just the mtllib line rather than a full parse, to find the mtl
    # filename quickly even on a multi-GB obj
    mtl_name = None
    with open(obj_path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("mtllib "):
                mtl_name = line.split(maxsplit=1)[1].strip()
                break

    if mtl_name:
        mtl_path = os.path.join(folder, mtl_name)
        if os.path.exists(mtl_path):
            return obj_path, mtl_path

    mtl_candidates = glob.glob(os.path.join(folder, "*.mtl"))
    if not mtl_candidates:
        raise FileNotFoundError(
            f"Found {os.path.basename(obj_path)} but no matching .mtl file in:\n  {folder}"
        )
    return obj_path, mtl_candidates[0]


# Supported model file extensions, checked in this priority order when a
# folder contains more than one kind (OBJ first, since it's the original
# and most-tested format here; GLB after). A folder genuinely
# containing multiple different model formats at once is an unusual case
# this doesn't try to be clever about -- it just picks by this fixed
# priority and proceeds, the same "use the first one found" philosophy
# find_input_files already uses for multiple .obj files.
#
# NOTE: .ply support was removed -- a PLY parser was built and
# integrated but caused crashes in practice (its core API calls were
# only ever tested against hand-built fakes matching `plyfile`'s
# documented shape, never against a real install of that library, since
# the development environment had no internet access to install it).
# If PLY support is revisited later, it needs real testing against an
# actual install of `plyfile` before being wired back in here.
_SUPPORTED_EXTENSIONS = [".obj", ".glb"]


def find_model_file(folder: str) -> dict:
    """
    Format-agnostic version of find_input_files -- detects which of the
    supported model formats (.obj, .glb) a folder contains, and
    returns a small descriptor dict import_and_cache_any() can dispatch
    on, rather than forcing every format through OBJ's specific
    (obj_path, mtl_path) two-tuple shape (which doesn't make sense for
    GLB -- typically one single self-contained file with no companion at all).

    Returns one of:
      {"format": "obj", "obj_path": ..., "mtl_path": ...}
      {"format": "glb", "glb_path": ...}

    Raises FileNotFoundError if no supported model file is found at all,
    with the same kind of clear, actionable message find_input_files
    already gives for the OBJ-specific case.
    """
    for ext in _SUPPORTED_EXTENSIONS:
        candidates = glob.glob(os.path.join(folder, f"*{ext}"))
        if not candidates:
            continue
        if len(candidates) > 1:
            print(f"Note: multiple {ext} files found, using the first one: {candidates[0]}")
        model_path = candidates[0]

        if ext == ".obj":
            obj_path, mtl_path = find_input_files(folder)
            return {"format": "obj", "obj_path": obj_path, "mtl_path": mtl_path}
        elif ext == ".glb":
            return {"format": "glb", "glb_path": model_path}

    raise FileNotFoundError(
        f"No supported model file found in:\n  {folder}\n\n"
        f"CaveViewer supports .obj (with a matching .mtl) and .glb files. "
        f"Make sure you selected the folder containing your exported map."
    )


def import_and_cache(obj_path: str, mtl_path: str, force_rebuild: bool = False,
                      extra_progress_cb=None) -> str:
    """Parse + chunk the mesh if needed, returning the cache directory.
    Skips straight to the existing cache if one's already valid, since
    re-parsing a 2GB OBJ on every launch would defeat the whole point.

    extra_progress_cb(stage: str, fraction: float), if given, is called
    alongside the built-in console progress bar at every same checkpoint
    -- this is how the OPEN button's in-window progress panel
    (gui/import_progress_panel.py) hooks into the same import process
    without needing its own separate copy of this function or changing
    the console output anyone running from a terminal already sees."""
    from core import chunker
    from core.obj_parser import parse_obj, parse_mtl

    if not force_rebuild and chunker.cache_is_valid(obj_path):
        print(f"Using existing chunk cache (delete the .caveviewer_cache "
              f"folder next to your .obj if you want to force a rebuild).")
        return chunker.get_cache_dir(obj_path)

    print(f"No valid cache found -- importing {os.path.basename(obj_path)}.")
    print(f"This is a one-time cost; subsequent opens of this map will be instant.\n")

    t_start = time.time()

    def progress(stage: str, frac: float):
        bar_width = 40
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r  [{bar}] {frac*100:5.1f}%  {stage:<28}")
        sys.stdout.flush()
        if extra_progress_cb:
            extra_progress_cb(stage, frac)

    mesh = parse_obj(obj_path, progress_cb=progress)
    print()  # newline after the parse progress bar

    materials = parse_mtl(mtl_path)

    cache_dir = chunker.build_cache(obj_path, mesh, materials, progress_cb=progress)
    print()

    elapsed = time.time() - t_start
    n_chunks = len(chunker.load_manifest(cache_dir)["chunks"])
    print(f"\nImport complete in {elapsed:.1f}s -- "
          f"{len(mesh.face_pos_idx):,} triangles split into {n_chunks:,} spatial chunks.")

    return cache_dir


def import_and_cache_any(model_descriptor: dict, textures_dir: str, force_rebuild: bool = False,
                          extra_progress_cb=None) -> str:
    """
    Format-agnostic version of import_and_cache() -- dispatches on
    model_descriptor["format"] (see find_model_file()) to the right
    parser, then feeds the result into the EXACT SAME chunker.build_cache()
    used for OBJ, since core/obj_parser.py's RawMesh shape is what every
    format's parser converts into (see core/glb_parser.py's module
    docstring for the conversion details).

    The one real bridging step this function does: GLB's embedded
    texture images (raw bytes living inside the .glb file itself) get
    written out to real files inside `textures_dir` here, ONCE, during
    import -- rather than ever trying to store raw image bytes inside the
    JSON manifest (which isn't JSON-serializable anyway, and would bloat
    the manifest badly even if it were). Once written to disk, an
    embedded GLB texture is indistinguishable from an OBJ's on-disk JPEG
    from every other part of the pipeline's perspective (chunker.py,
    TextureManager reading from textures_dir, the manifest format) --
    no format-specific code needed anywhere downstream of this point.
    """
    from core import chunker
    from core.obj_parser import Material

    fmt = model_descriptor["format"]

    if fmt == "obj":
        return import_and_cache(
            model_descriptor["obj_path"], model_descriptor["mtl_path"],
            force_rebuild=force_rebuild, extra_progress_cb=extra_progress_cb,
        )

    source_path = model_descriptor["glb_path"]

    if not force_rebuild and chunker.cache_is_valid(source_path):
        print(f"Using existing chunk cache (delete the .caveviewer_cache "
              f"folder next to your {os.path.basename(source_path)} if you want to force a rebuild).")
        return chunker.get_cache_dir(source_path)

    print(f"No valid cache found -- importing {os.path.basename(source_path)}.")
    print(f"This is a one-time cost; subsequent opens of this map will be instant.\n")

    t_start = time.time()

    def progress(stage: str, frac: float):
        bar_width = 40
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r  [{bar}] {frac*100:5.1f}%  {stage:<28}")
        sys.stdout.flush()
        if extra_progress_cb:
            extra_progress_cb(stage, frac)

    if fmt == "glb":
        from core.glb_parser import parse_glb
        mesh, embedded_textures = parse_glb(source_path, progress_cb=progress)

        # Write each embedded texture out to a real file in textures_dir,
        # once, so every downstream consumer (chunker.py's manifest,
        # TextureManager) just sees an ordinary on-disk filename -- see
        # this function's own docstring for why this is done here rather
        # than threading raw bytes through the manifest/cache format.
        materials = {}
        for mat_range in mesh.material_ranges:
            mat_name = mat_range.material_name
            if mat_name in embedded_textures:
                image_bytes = embedded_textures[mat_name]
                image_filename = _write_embedded_texture_to_disk(
                    image_bytes, textures_dir, mat_name
                )
                materials[mat_name] = Material(name=mat_name, diffuse_texture=image_filename)
            else:
                # no embedded texture found for this material under this
                # name -- leave it untextured (the placeholder-texture
                # path in TextureManager handles this the same as an
                # OBJ material with no map_Kd line)
                materials[mat_name] = Material(name=mat_name, diffuse_texture=None)

    else:
        raise ValueError(f"Unknown model format: {fmt!r}")

    print()  # newline after the parse progress bar

    cache_dir = chunker.build_cache(source_path, mesh, materials, progress_cb=progress)
    print()

    elapsed = time.time() - t_start
    n_chunks = len(chunker.load_manifest(cache_dir)["chunks"])
    print(f"\nImport complete in {elapsed:.1f}s -- "
          f"{len(mesh.face_pos_idx):,} triangles split into {n_chunks:,} spatial chunks.")

    return cache_dir


def _write_embedded_texture_to_disk(image_bytes: bytes, textures_dir: str, material_name: str) -> str:
    """
    Writes one GLB-embedded texture's raw bytes to a real file inside
    textures_dir, sniffing the actual image format from the bytes
    themselves (JPEG vs PNG, the two formats glTF supports for textures)
    rather than trusting any file extension, since embedded image data
    has no filename of its own to go by -- just the bytes. Returns the
    filename (not full path) that was written, which the caller stores
    as that material's diffuse_texture, the same as an OBJ's .mtl
    map_Kd line would.
    """
    # JPEG files start with FF D8; PNG files start with the fixed 8-byte
    # PNG signature -- checking the actual leading bytes is more reliable
    # than guessing from context, since glTF doesn't store a format tag
    # separately from the image bytes themselves for embedded images.
    if image_bytes[:2] == b"\xff\xd8":
        ext = ".jpg"
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    else:
        # unrecognized format -- write it anyway with a generic
        # extension; Pillow can often still sniff the real format from
        # content when TextureManager later opens it, and if it truly
        # can't, that degrades to the existing missing/corrupt-texture
        # placeholder path rather than crashing here at write time.
        ext = ".img"

    filename = f"{material_name}{ext}"
    os.makedirs(textures_dir, exist_ok=True)
    with open(os.path.join(textures_dir, filename), "wb") as f:
        f.write(image_bytes)
    return filename


def pick_folder_dialog() -> str | None:
    """Tkinter native folder picker. Tkinter ships with standard Python on
    Windows/Mac, so this needs no extra install for the bare-bones UI."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select folder containing your cave map (.obj, .mtl, .jpg)"
    )
    root.destroy()
    return folder or None


def main():
    print("=" * 60)
    print(f"  CaveViewer {__version__}")
    print("=" * 60)

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        from gui.splash_screen import show_splash_screen
        folder = show_splash_screen(program_name="CaveViewer", version=__version__)

    if not folder:
        print("No folder selected. Exiting.")
        return

    folder = os.path.abspath(folder)
    print(f"\nSelected folder: {folder}")

    try:
        model_descriptor = find_model_file(folder)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    fmt = model_descriptor["format"]
    source_path = model_descriptor.get("obj_path") or model_descriptor.get("glb_path")
    print(f"Found {fmt.upper()} mesh: {os.path.basename(source_path)}")
    if fmt == "obj":
        print(f"Found materials: {os.path.basename(model_descriptor['mtl_path'])}")

    print("\nLaunching viewer...")
    print("Controls: WASD = move, Space/Ctrl = up/down, hold Right-Mouse + move = look,")
    print("          Shift = speed boost, Scroll = adjust fly speed,")
    print("          Left-click +/- (bottom-right corner) = adjust headlamp brightness,")
    print("          Left-click +/- (bottom-right corner, below brightness) = adjust global ambient light,")
    print("          Left-click the minimap (bottom-left) = jump to that spot,")
    print("          Left-click +/- (bottom-right corner, below brightness) = adjust render distance,")
    print("          Left-click Mesh/Texture buttons (bottom-right corner) = toggle wireframe/texture,")
    print("          Left-click the Help button (bottom-right corner) = show/hide the controls list,")
    print("          Left-click the Color button (bottom-right corner) = open/close the background color picker,")
    print("          Left-click the Open button (bottom-right corner) = switch to a different map,")
    print("          Esc = quit\n")

    from core import chunker

    if chunker.cache_is_valid(source_path):
        # Fast path, unchanged: a cache already exists, so there's no
        # import to show progress for -- launch straight in, same as
        # this has always worked.
        print("Using existing chunk cache (delete the .caveviewer_cache "
              "folder next to your model file if you want to force a rebuild).")
        cache_dir = chunker.get_cache_dir(source_path)
        from gui.viewer_window import run_viewer
        run_viewer(cache_dir, textures_dir=folder)
    else:
        # No cache yet -- open the window FIRST, with no map loaded, and
        # let it run the import itself once it's actually on screen (see
        # gui/viewer_window.py's _run_pending_import), so the same
        # in-window progress panel the OPEN button uses can show real
        # progress here too, instead of the import running to completion
        # before any window exists (which could only show a plain console
        # progress bar).
        from gui.viewer_window import run_viewer_with_pending_import
        run_viewer_with_pending_import(model_descriptor, textures_dir=folder)


if __name__ == "__main__":
    main()

