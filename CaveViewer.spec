# CaveViewer.spec
#
# PyInstaller build spec for packaging CaveViewer into a single standalone
# .exe that runs on a Windows machine with no separate Python install.
#
# Build with:
#     pyinstaller CaveViewer.spec
#
# Output appears in dist/CaveViewer/CaveViewer.exe (a folder build, not a
# single-file build -- see the note on collect_data_files below for why
# this project intentionally avoids --onefile).
#
# IMPORTANT: this must be built ON Windows, with the project's own
# requirements.txt already pip-installed into the Python environment you
# run PyInstaller from. PyInstaller bundles whatever's importable in the
# environment it runs in -- it doesn't cross-compile, and it doesn't fetch
# packages itself.

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# moderngl-window ships some internal resource files (font data for its
# optional on-screen text/timer helpers, default icons, etc.) that aren't
# picked up by PyInstaller's default import analysis since they're loaded
# at runtime via importlib.resources / pkg_resources rather than plain
# `import` statements. collect_data_files grabs these automatically so the
# bundle doesn't silently break the first time moderngl-window reaches for
# one of them.
moderngl_window_datas = collect_data_files('moderngl_window')
pyglet_datas = collect_data_files('pyglet')

# Some packages (pyglet in particular, since it has backend-detection
# logic for different windowing systems) dynamically import submodules
# that PyInstaller's static analysis can miss. collect_submodules forces
# all of a package's submodules to be included rather than guessing.
hidden_imports = (
    collect_submodules('pyglet')
    + collect_submodules('moderngl_window')
    + ['PIL._tkinter_finder']  # Pillow's Tk integration, used indirectly via tkinter filedialog
)

a = Analysis(
    ['caveviewer.py'],
    pathex=[],
    binaries=[],
    datas=(
        # Our own shader source files -- read at runtime via SHADER_DIR in
        # gui/viewer_window.py, which checks sys.frozen and points at
        # sys._MEIPASS/shaders in a frozen build. The destination path
        # here ('shaders') must match that exactly.
        [('shaders/mesh.vert', 'shaders'), ('shaders/mesh.frag', 'shaders')]
        + moderngl_window_datas
        + pyglet_datas
    ),
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CaveViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    # console=True intentionally: this keeps the terminal window visible
    # so the person running it can see import progress, FPS, and any
    # error messages -- the same console output you've been pasting back
    # during testing. Setting this to False would hide all of that, which
    # makes diagnosing any future issue much harder for a non-technical
    # end user (no error text to send back, just a window that vanishes).
    icon='setup/icon/caveviewer.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CaveViewer',
)

# NOTE on --onefile vs folder builds:
#
# PyInstaller supports building a single .exe file (--onefile) instead of
# this folder-based COLLECT output. A single file is more convenient to
# share, but it has a real downside for this project specifically: a
# --onefile build re-extracts ALL bundled data (including moderngl-window
# and pyglet's resource files) to a fresh temp directory every single time
# the program launches, which adds a noticeable multi-second delay before
# the window even appears -- before the cave map import/chunking even
# starts. The folder build above extracts once at build time, so launches
# are instant. Given the existing import step for a large map already
# takes a minute or more, adding launch-time extraction overhead on top of
# that on every run is a worse experience than asking the person to keep
# one folder together (which they're already doing for the .obj/.mtl/.jpg
# inputs) instead of one single .exe file.
