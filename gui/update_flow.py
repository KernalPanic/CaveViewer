"""
gui/update_flow.py

Orchestrates the full "Check for Updates" experience triggered from the
splash screen's button: check GitHub -> show the result -> if an update
exists, confirm with the person -> download with a progress dialog ->
hand off to gui/updater.py (a separate process) to do the actual file
replacement -> close this process so the updater can safely overwrite
its files.

Kept separate from both gui/splash_screen.py (which shouldn't need to
know anything about HOW updating works, just that there's a button) and
gui/update_checker.py (pure check/download logic, no UI) -- this module
is the glue between the two, plus the Tkinter dialogs themselves.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile


def run_update_check_flow(parent, current_version: str) -> None:
    """
    Entry point called by the splash screen's "Check for Updates" button.
    parent is the splash screen's own Tk root, used so these dialogs
    appear centered relative to it and are modal to it (the splash
    screen is blocked from interaction while this runs, the same way any
    dialog-over-a-window normally behaves).
    """
    import tkinter as tk
    from tkinter import messagebox
    from gui.update_checker import check_for_update, download_update
    from gui.splash_screen import _BG_COLOR, _TITLE_COLOR, _SUBTITLE_COLOR, _INSTRUCTION_COLOR, \
        _BUTTON_BG, _BUTTON_FG, _BORDER_COLOR

    checking_dialog = tk.Toplevel(parent)
    checking_dialog.title("Checking for Updates")
    checking_dialog.configure(bg=_BG_COLOR)
    checking_dialog.resizable(False, False)
    checking_dialog.geometry("320x100")
    checking_dialog.transient(parent)
    _center_over_parent(checking_dialog, parent, 320, 100)

    tk.Label(checking_dialog, text="Checking for updates...", font=("Segoe UI", 11),
             fg=_SUBTITLE_COLOR, bg=_BG_COLOR).pack(expand=True)

    checking_dialog.update()

    result = check_for_update(current_version)

    checking_dialog.destroy()

    if result.error:
        messagebox.showinfo(
            "Check for Updates",
            f"Couldn't check for updates right now:\n\n{result.error}",
            parent=parent,
        )
        return

    if not result.update_available:
        messagebox.showinfo(
            "Check for Updates",
            f"You're up to date! (Version {current_version})",
            parent=parent,
        )
        return

    size_mb = (result.download_size_bytes or 0) / (1024 * 1024)
    notes_preview = (result.release_notes or "").strip()
    if len(notes_preview) > 500:
        notes_preview = notes_preview[:500] + "..."

    message = f"A new version is available: {result.latest_version} (you have {current_version})\n\n"
    if notes_preview:
        message += f"What's new:\n{notes_preview}\n\n"
    message += f"Download now (~{size_mb:.1f} MB)? CaveViewer will close and restart to finish updating."

    should_download = messagebox.askyesno("Update Available", message, parent=parent)
    if not should_download:
        return

    progress_dialog = tk.Toplevel(parent)
    progress_dialog.title("Downloading Update")
    progress_dialog.configure(bg=_BG_COLOR)
    progress_dialog.resizable(False, False)
    progress_dialog.geometry("360x120")
    progress_dialog.transient(parent)
    _center_over_parent(progress_dialog, parent, 360, 120)

    status_label = tk.Label(progress_dialog, text="Starting download...", font=("Segoe UI", 10),
                             fg=_SUBTITLE_COLOR, bg=_BG_COLOR)
    status_label.pack(pady=(20, 8))

    progress_canvas = tk.Canvas(progress_dialog, width=300, height=18, bg="#1c1c24",
                                  highlightthickness=1, highlightbackground=_BORDER_COLOR)
    progress_canvas.pack(pady=(0, 10))
    progress_bar = progress_canvas.create_rectangle(0, 0, 0, 18, fill=_BUTTON_BG, width=0)

    def on_progress(downloaded, total):
        frac = min(1.0, downloaded / total) if total else 0.0
        progress_canvas.coords(progress_bar, 0, 0, 300 * frac, 18)
        status_label.config(text=f"Downloading... {downloaded // 1024} / {total // 1024} KB")
        progress_dialog.update()

    download_dir = tempfile.mkdtemp(prefix="caveviewer_update_")
    zip_path = os.path.join(download_dir, "update.zip")

    try:
        download_update(result.download_url, result.download_size_bytes, zip_path, progress_cb=on_progress)
    except Exception as e:
        progress_dialog.destroy()
        messagebox.showerror(
            "Update Failed",
            f"Couldn't download the update:\n\n{e}\n\nCaveViewer has not been modified.",
            parent=parent,
        )
        return

    progress_dialog.destroy()

    main_script_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "caveviewer.py")
    main_script_candidate = os.path.normpath(main_script_candidate)
    if os.path.exists(main_script_candidate):
        install_dir = os.path.dirname(main_script_candidate)
        relaunch_script = main_script_candidate
    else:
        install_dir = os.getcwd()
        relaunch_script = None

    updater_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "updater.py")

    messagebox.showinfo(
        "Update Ready",
        "The update has been downloaded. CaveViewer will now close to finish installing it.",
        parent=parent,
    )

    updater_args = [sys.executable, updater_script, zip_path, install_dir]
    if relaunch_script:
        updater_args.append(relaunch_script)

    subprocess.Popen(updater_args)

    parent.destroy()


def _center_over_parent(window, parent, width, height) -> None:
    parent.update_idletasks()
    px = parent.winfo_rootx()
    py = parent.winfo_rooty()
    pw = parent.winfo_width()
    ph = parent.winfo_height()
    x = px + (pw - width) // 2
    y = py + (ph - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")
