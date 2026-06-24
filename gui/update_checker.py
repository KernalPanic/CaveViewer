"""
gui/update_checker.py

Checks GitHub Releases for a newer version of CaveViewer than the one
currently running, and (if the person confirms) downloads the new
release's zip asset to a temp folder.

Does NOT do the actual file replacement -- that's deliberately handled
by a separate process (gui/updater.py), launched only after this
process has finished downloading and is about to exit. See
gui/updater.py's module docstring for why a separate process is the
standard, safe way to do this (the running app can't reliably overwrite
its own currently-imported source files).

Network failures (no internet, GitHub unreachable, rate-limited, repo
not found) are all treated as "couldn't check right now" rather than
crashes -- a failed update check should never block someone from using
the app offline, which is the whole point of keeping this feature
separate from the app's core offline-first design.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional


# PLACEHOLDER -- replace with the real GitHub repo once it exists.
# Format: "owner/repo", e.g. "octocat/Hello-World".
GITHUB_REPO = "YOUR_GITHUB_USERNAME/CaveViewer"

_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_REQUEST_TIMEOUT_SECONDS = 8


@dataclass
class UpdateCheckResult:
    update_available: bool
    current_version: str
    latest_version: Optional[str] = None
    download_url: Optional[str] = None
    download_size_bytes: Optional[int] = None
    release_notes: Optional[str] = None
    error: Optional[str] = None


def _parse_version(version_str: str) -> tuple:
    """
    Parses a version string like "1.2" or "1.2.3" into a tuple of ints
    for comparison, e.g. (1, 2) or (1, 2, 3) -- so "1.10" correctly
    compares as greater than "1.9" (plain string comparison would get
    this wrong: "1.10" < "1.9" alphabetically). Strips a leading "v" if
    present (some repos tag releases "v1.2" rather than "1.2"), and
    falls back to (0,) for anything that doesn't parse as dotted
    integers, so a malformed tag degrades to "treat as not newer" rather
    than crashing the whole check.
    """
    cleaned = version_str.strip()
    if cleaned.lower().startswith("v"):
        cleaned = cleaned[1:]
    parts = []
    for piece in cleaned.split("."):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            return (0,)
    return tuple(parts) if parts else (0,)


def check_for_update(current_version: str) -> UpdateCheckResult:
    """
    Synchronous -- intended to be called from a button click (the
    person already expects a brief pause for "checking..."), not from
    inside a render loop. Returns a result dict-like object; never
    raises -- every failure mode is captured in .error instead, so the
    caller can show a calm "couldn't check for updates right now"
    message rather than a stack trace.
    """
    if "YOUR_GITHUB_USERNAME" in GITHUB_REPO:
        return UpdateCheckResult(
            update_available=False,
            current_version=current_version,
            error="Update checking isn't configured yet -- gui/update_checker.py's "
                  "GITHUB_REPO placeholder needs to be set to a real repository."
        )

    try:
        request = urllib.request.Request(
            _API_URL,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "CaveViewer-UpdateChecker"},
        )
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            error_msg = "No releases found for this repository yet."
        else:
            error_msg = f"GitHub returned an error (HTTP {e.code})."
        return UpdateCheckResult(update_available=False, current_version=current_version, error=error_msg)
    except urllib.error.URLError:
        return UpdateCheckResult(
            update_available=False, current_version=current_version,
            error="Couldn't reach GitHub -- check your internet connection."
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return UpdateCheckResult(
            update_available=False, current_version=current_version,
            error=f"Got an unexpected response from GitHub: {e}"
        )

    latest_tag = data.get("tag_name", "")
    release_notes = data.get("body", "") or ""

    # Find a zip asset to download -- prefers an asset whose name
    # contains "caveviewer" (case-insensitive) if there are multiple
    # assets attached to the release, otherwise falls back to the first
    # .zip found, since most releases will only have one asset anyway.
    assets = data.get("assets", [])
    zip_assets = [a for a in assets if a.get("name", "").lower().endswith(".zip")]
    chosen_asset = None
    for asset in zip_assets:
        if "caveviewer" in asset.get("name", "").lower():
            chosen_asset = asset
            break
    if chosen_asset is None and zip_assets:
        chosen_asset = zip_assets[0]

    if chosen_asset is None:
        return UpdateCheckResult(
            update_available=False, current_version=current_version, latest_version=latest_tag,
            error="A newer release exists but has no downloadable .zip file attached."
        )

    is_newer = _parse_version(latest_tag) > _parse_version(current_version)

    return UpdateCheckResult(
        update_available=is_newer,
        current_version=current_version,
        latest_version=latest_tag,
        download_url=chosen_asset.get("browser_download_url"),
        download_size_bytes=chosen_asset.get("size"),
        release_notes=release_notes.strip(),
    )


def download_update(download_url: str, expected_size_bytes, dest_path: str,
                     progress_cb=None) -> None:
    """
    Downloads the release zip to dest_path. Raises on any failure
    (network error, size mismatch) -- the caller is expected to catch
    this and show a clear message, since a failed download should never
    silently proceed to the file-replacement step with a corrupt/partial
    file.

    progress_cb(downloaded_bytes, total_bytes), if given, is called
    periodically during the download for a progress indicator.
    """
    request = urllib.request.Request(download_url, headers={"User-Agent": "CaveViewer-UpdateChecker"})

    with urllib.request.urlopen(request, timeout=30) as response:
        total = expected_size_bytes or int(response.headers.get("Content-Length", 0)) or None
        downloaded = 0
        chunk_size = 65536

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total or downloaded)

    actual_size = os.path.getsize(dest_path)
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        os.remove(dest_path)
        raise IOError(
            f"Downloaded file size ({actual_size} bytes) doesn't match the "
            f"expected size ({expected_size_bytes} bytes) -- the download may "
            f"have been interrupted or corrupted. Please try again."
        )
