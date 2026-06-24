"""
gui/splash_screen.py

The very first thing shown when CaveViewer launches: a small landing
window with the program name/version, the skull logo, and a Browse
button to pick the folder containing the cave map's .obj/.mtl/texture
files -- replacing the old behavior of jumping straight into a bare
native folder-picker dialog with zero context about what the program
even is.

Built with Tkinter (ships with standard Python on Windows/Mac, same
reasoning as the existing native folder-picker dialog already used
elsewhere in caveviewer.py -- no extra install needed). Styled to loosely
match the in-program overlays' dark background + amber accent look,
though Tkinter's native widgets can only approximate that so closely --
this is a real OS window with title bar and native buttons, not a custom-
drawn OpenGL overlay like the rest of the program's UI.

This is intentionally a SEPARATE function from pick_folder_dialog() in
caveviewer.py (which stays a quick bare native dialog) -- the splash
screen is for the very first launch, when the person hasn't seen the
program yet and benefits from the context; the OPEN button mid-session
(see viewer_window.py) is for someone already using the program, where a
quick plain dialog is the better fit and a full splash screen would just
be unnecessary ceremony.
"""

from __future__ import annotations

import os


# Resolve this once at import time -- same asset already used for the
# in-program loading-screen logo, reused here rather than shipping a
# second copy of the same image.
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "loading_logo.png")

_BG_COLOR = "#0a0a0d"           # near-black, matches the in-app overlay backgrounds
_PANEL_COLOR = "#12121a"        # slightly lighter panel background
_TITLE_COLOR = "#f2d98c"        # amber/gold, matches the in-app title text color
_SUBTITLE_COLOR = "#cccdd6"     # light gray, matches in-app subtitle/body text
_INSTRUCTION_COLOR = "#9a9aa6"  # dimmer gray, matches in-app secondary/note text
_BUTTON_BG = "#caa23e"          # amber button, matches the in-app active-button color
_BUTTON_FG = "#1a1408"          # dark text on the amber button, matches in-app active-button text
_BORDER_COLOR = "#5c5c6e"


def show_splash_screen(program_name: str = "CaveViewer", version: str = "1.0") -> str | None:
    """
    Shows the launch splash screen and blocks until the person either
    picks a folder (Browse -> select a folder -> OK) or closes the
    window. Returns the selected folder path, or None if the window was
    closed without picking one.
    """
    import tkinter as tk
    from tkinter import filedialog

    selected_folder: list[str | None] = [None]

    root = tk.Tk()
    root.title(f"{program_name} {version}")
    root.configure(bg=_BG_COLOR)
    root.resizable(False, False)

    window_w, window_h = 480, 490
    root.geometry(f"{window_w}x{window_h}")

    # Center the window on screen rather than letting the OS place it
    # arbitrarily -- a first-launch splash screen appearing somewhere
    # random/off-center is a small but noticeable rough edge.
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    pos_x = (screen_w - window_w) // 2
    pos_y = (screen_h - window_h) // 3  # slightly above true vertical center, reads better
    root.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")

    # Try to set the window icon to the same logo, falling back silently
    # if anything about icon-setting fails on this platform/Tk build --
    # a missing window-titlebar icon is a cosmetic-only issue, never
    # worth letting it block the splash screen from showing at all.
    try:
        from PIL import Image, ImageTk
        icon_img = Image.open(_LOGO_PATH)
        icon_photo = ImageTk.PhotoImage(icon_img)
        root.iconphoto(True, icon_photo)
    except Exception:
        pass

    # -- logo image, centered near the top --------------------------------------
    logo_photo = None
    try:
        from PIL import Image, ImageTk
        logo_img = Image.open(_LOGO_PATH)
        # scale down to a sensible splash-screen size if the source is
        # larger than needed, preserving aspect ratio -- keeps this
        # robust to the source asset's exact dimensions changing later
        max_logo_dim = 140
        scale = min(max_logo_dim / logo_img.width, max_logo_dim / logo_img.height, 1.0)
        if scale < 1.0:
            new_size = (int(logo_img.width * scale), int(logo_img.height * scale))
            logo_img = logo_img.resize(new_size, Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_img)
    except Exception as e:
        print(f"[CaveViewer] Note: could not load splash screen logo ({e}); continuing without it.")

    if logo_photo is not None:
        logo_label = tk.Label(root, image=logo_photo, bg=_BG_COLOR, borderwidth=0)
        logo_label.image = logo_photo  # keep a reference so it isn't garbage-collected
        logo_label.pack(pady=(22, 6))

    # -- title + version, centered top -------------------------------------------
    title_label = tk.Label(
        root, text=program_name, font=("Segoe UI", 22, "bold"),
        fg=_TITLE_COLOR, bg=_BG_COLOR,
    )
    title_label.pack(pady=(0, 0))

    version_label = tk.Label(
        root, text=f"Version {version}", font=("Segoe UI", 11),
        fg=_SUBTITLE_COLOR, bg=_BG_COLOR,
    )
    version_label.pack(pady=(0, 4))

    # Small, deliberately understated link-style button -- this is a
    # secondary action next to Browse, not something that should compete
    # visually with it. Clicking it opens a separate small dialog (see
    # gui/update_flow.py) that handles checking, confirming, downloading,
    # and handing off to the actual file-replacement step -- kept out of
    # this function entirely so the splash screen itself doesn't need to
    # know anything about how updating actually works.
    def on_check_updates():
        from gui.update_flow import run_update_check_flow
        run_update_check_flow(parent=root, current_version=version)

    update_button = tk.Button(
        root, text="Check for Updates", command=on_check_updates,
        font=("Segoe UI", 8, "underline"),
        bg=_BG_COLOR, fg=_INSTRUCTION_COLOR,
        activebackground=_BG_COLOR, activeforeground=_TITLE_COLOR,
        relief="flat", borderwidth=0,
        cursor="hand2",
    )
    update_button.pack(pady=(0, 14))

    credit_label = tk.Label(
        root,
        text="CaveViewer created by Brian Deatherage & Zsolt Zsabo of\n"
             "BottomLine Projects Scientific Dive Team",
        font=("Segoe UI", 9),
        fg=_INSTRUCTION_COLOR, bg=_BG_COLOR,
        justify="center",
    )
    credit_label.pack(pady=(0, 18))

    # -- separator line, subtle ---------------------------------------------------
    separator = tk.Frame(root, bg=_BORDER_COLOR, height=1)
    separator.pack(fill="x", padx=40, pady=(0, 22))

    # -- browse button + instructions ---------------------------------------------
    def on_browse():
        folder = filedialog.askdirectory(
            title="Select the folder containing your cave map (.obj, .mtl, .jpg)"
        )
        if folder:
            selected_folder[0] = folder
            root.destroy()

    browse_button = tk.Button(
        root, text="Browse...", command=on_browse,
        font=("Segoe UI", 12, "bold"),
        bg=_BUTTON_BG, fg=_BUTTON_FG,
        activebackground=_BUTTON_BG, activeforeground=_BUTTON_FG,
        relief="flat", borderwidth=0,
        padx=28, pady=10,
        cursor="hand2",
    )
    browse_button.pack(pady=(0, 14))

    instruction_label = tk.Label(
        root,
        text="Point this to the folder containing your map's\n"
             ".obj+.mtl or .glb file (with textures).",
        font=("Segoe UI", 10),
        fg=_INSTRUCTION_COLOR, bg=_BG_COLOR,
        justify="center",
    )
    instruction_label.pack(pady=(0, 0))

    # -- footer note ----------------------------------------------------------------
    footer_label = tk.Label(
        root, text="Close this window to exit without opening a map.",
        font=("Segoe UI", 8), fg=_INSTRUCTION_COLOR, bg=_BG_COLOR,
    )
    footer_label.pack(side="bottom", pady=(0, 14))

    root.mainloop()

    return selected_folder[0]
