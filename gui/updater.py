#!/usr/bin/env python3
"""
gui/updater.py

Standalone helper script that performs the actual update file-replacement
-- run as a SEPARATE process from the main CaveViewer app, launched right
before the main app exits.

Why a separate process: a running Python process has its own .py source
files open/imported. While Python source files (unlike a compiled .exe)
usually CAN be overwritten while "in use" on Windows -- the OS doesn't
memory-map a .py the way it does a running .exe -- it's still the
standard, safe pattern used by essentially every desktop auto-updater
(this is exactly what Electron's autoUpdater, Sparkle on Mac, and Squirrel
do under the hood) to fully separate "the app that's running" from "the
process that replaces the app's files," so there's no window where the
running app is trying to read a half-replaced version of itself, and so
a failure during replacement can't corrupt a process that's still trying
to execute from those same files.

Usage (invoked by gui/update_flow.py, not meant to be run manually):
    python updater.py <downloaded_zip_path> <install_dir> [<relaunch_script>]

Steps:
    1. Wait briefly for the main app's process to fully exit (it should
       already be gone by the time this runs, but a short grace period
       avoids a race on slow systems).
    2. Extract the downloaded zip to a temporary staging folder first
       (not directly over the install) -- this means a corrupt/partial
       zip fails at EXTRACTION time, before anything in the real install
       directory has been touched, rather than leaving a half-overwritten
       install if extraction fails partway through.
    3. Copy the staged files over the install directory.
    4. If a relaunch script path was given, relaunch the app.
    5. Clean up temp files.

Every step is logged to a file (update_log.txt, next to this script) since
this process runs detached with no visible console most of the time --
if something goes wrong, that log is the only record of what happened.
"""

import os
import sys
import time
import shutil
import zipfile
import tempfile
import subprocess


def log(log_path, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    if len(sys.argv) < 3:
        print("Usage: python updater.py <downloaded_zip_path> <install_dir> [<relaunch_script>]")
        sys.exit(1)

    zip_path = sys.argv[1]
    install_dir = sys.argv[2]
    relaunch_script = sys.argv[3] if len(sys.argv) > 3 else None

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_log.txt")
    log(log_path, f"Updater started. zip={zip_path}, install_dir={install_dir}, relaunch={relaunch_script}")

    time.sleep(1.5)

    if not os.path.exists(zip_path):
        log(log_path, f"ERROR: downloaded zip not found at {zip_path}. Aborting update.")
        sys.exit(1)

    staging_dir = tempfile.mkdtemp(prefix="caveviewer_update_staging_")
    log(log_path, f"Extracting to staging directory: {staging_dir}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(staging_dir)
    except zipfile.BadZipFile as e:
        log(log_path, f"ERROR: downloaded file is not a valid zip ({e}). Aborting update -- "
                       f"the install directory has NOT been touched.")
        shutil.rmtree(staging_dir, ignore_errors=True)
        sys.exit(1)

    staging_contents = os.listdir(staging_dir)
    if len(staging_contents) == 1 and os.path.isdir(os.path.join(staging_dir, staging_contents[0])):
        source_root = os.path.join(staging_dir, staging_contents[0])
    else:
        source_root = staging_dir

    log(log_path, f"Copying files from {source_root} to {install_dir}")

    try:
        for root, dirs, files in os.walk(source_root):
            rel_path = os.path.relpath(root, source_root)
            dest_dir = os.path.join(install_dir, rel_path) if rel_path != "." else install_dir
            os.makedirs(dest_dir, exist_ok=True)
            for filename in files:
                src_file = os.path.join(root, filename)
                dest_file = os.path.join(dest_dir, filename)
                shutil.copy2(src_file, dest_file)
    except Exception as e:
        log(log_path, f"ERROR during file copy: {e}. The install directory may be partially "
                       f"updated -- some files may be newer, some still old. Re-running the "
                       f"update is recommended.")
        shutil.rmtree(staging_dir, ignore_errors=True)
        sys.exit(1)

    log(log_path, "File copy completed successfully.")

    shutil.rmtree(staging_dir, ignore_errors=True)
    try:
        os.remove(zip_path)
    except Exception:
        pass

    if relaunch_script and os.path.exists(relaunch_script):
        log(log_path, f"Relaunching: {relaunch_script}")
        try:
            python_exe = sys.executable
            subprocess.Popen([python_exe, relaunch_script], cwd=install_dir)
        except Exception as e:
            log(log_path, f"WARNING: update succeeded but relaunch failed ({e}). "
                           f"Please start CaveViewer manually.")
    else:
        log(log_path, "No relaunch script given/found -- update complete. "
                       "Please start CaveViewer manually.")

    log(log_path, "Updater finished.")


if __name__ == "__main__":
    main()
