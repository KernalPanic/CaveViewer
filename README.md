# CaveViewer 1.0

A standalone viewer for large cave-survey 3D meshes, built for maps too
big to comfortably load all at once. Instead of loading the whole mesh
into memory/VRAM, CaveViewer splits it into a 3D grid of spatial chunks
and only keeps the chunks near your current position loaded -- so frame
rate stays smooth no matter how big the full cave system is.

Supports two source formats:
- **OBJ** (+ matching `.mtl` + tiled `.jpg` textures) -- the original
  format this was built for, exported from Agisoft Metashape.
- **GLB** (binary glTF) -- including maps whose textures are embedded
  directly inside the file rather than as separate images.

Whichever format a map is in, the rest of the program (chunking,
streaming, all the on-screen controls) behaves identically -- format only
matters at the moment a map is first opened.

## How it works

1. **First time opening a map:** CaveViewer parses your `.obj` (streaming,
   so it doesn't choke on multi-GB files) and splits it into a grid of
   spatial chunks (default 8m cubes), writing a cache folder
   `.caveviewer_cache` next to your `.obj` file. This is a one-time cost --
   for a 10-20 million triangle map, expect roughly 30-90 seconds depending
   on your CPU and disk speed.
2. **Every time after:** CaveViewer detects the existing cache and skips
   straight to launching the viewer -- near-instant.
3. **While flying around:** only the chunks within a few grid cells of your
   camera are loaded into GPU memory. As you fly deeper into the cave, far
   chunks behind you are unloaded and new chunks ahead of you stream in,
   automatically, in a background thread so it doesn't stall your frame
   rate.

## Setup (one-time, on your machine)

### Windows: the Setup GUI (recommended, no terminal needed)

Inside the `CaveViewer` folder, open the `setup` folder and double-click
**`Launch_Setup.bat`**. This opens a small window with one **Install**
button that does everything in sequence:

1. **Installs Python** -- checks if Python's already on your system; if
   not, downloads and installs it automatically (and correctly registers
   it on your system PATH, which is the step that's easy to miss when
   doing this by hand).
2. **Installs the required libraries** CaveViewer needs (moderngl, numpy,
   Pillow, etc).
3. **Creates a Desktop shortcut** -- adds a "CaveViewer" icon to your
   Desktop that launches the program directly, using the custom CaveViewer
   icon (`setup/icon/caveviewer.ico`) rather than a generic Python icon.

Click **Install** and watch the log box for progress. Windows will show a
security prompt asking for permission to install Python system-wide --
this is expected, click Yes. If any step fails, the button re-enables so
you can click Install again after fixing whatever the log box reported
(usually a connectivity issue for the Python download, or a missing file
if the setup folder got separated from the rest of the project).

Once everything finishes, the setup window closes itself automatically
after a couple seconds -- double-click the new **CaveViewer** icon on your
Desktop any time you want to run it -- no terminal, no typed commands.

### Mac

A one-click Mac setup script isn't available right now -- if you're on a
Mac, use the manual setup below instead (Python and the requirements
still install the same way; you'll just run a couple of terminal commands
yourself rather than double-clicking one file).

### Manual way (if you'd rather use a terminal, or a script hits an issue)

You'll need Python 3.10+ installed. Then, from inside the `CaveViewer`
folder:

```bash
pip install -r requirements.txt
```

This installs:
- `moderngl` + `moderngl-window` -- the OpenGL rendering layer
- `numpy` -- fast array math for mesh processing
- `Pillow` -- JPEG texture decoding

## Running it

If you used a setup script, just double-click the **CaveViewer** icon on
your Desktop.

If you're running manually:

```bash
python caveviewer.py
```
(On Mac, this is usually `python3 caveviewer.py` instead -- macOS doesn't
ship a `python` command by default, only `python3`.)

This shows the CaveViewer splash screen -- the program name and version,
the logo, and a **Browse...** button. Click Browse and pick the folder
containing your map -- a folder with an `.obj` (plus matching `.mtl` and
texture tile `.jpg` files) or a `.glb` file both work. (You can
also skip the splash screen entirely and pass the folder directly:
`python caveviewer.py "/path/to/your/cave folder"`.)

If the map you picked has never been opened before (no cache yet), the
viewer window opens right away and shows the same in-window progress
screen used when switching maps mid-session via the OPEN button --
title, map name, a progress bar, and the current import stage -- while
it imports and chunks the mesh. This is a one-time cost; once a map has
been opened, all future opens of it (including via this same screen, or
the OPEN button) are instant. The window can't be interacted with while
this is happening, the same way it couldn't before this screen existed --
just with real visible progress now instead of the program seeming to
hang with nothing on screen.

If the map's already been opened before, the viewer window opens
straight into it with no progress screen needed.

## Controls

| Input              | Action                          |
|---------------------|----------------------------------|
| `W` / `S`           | Fly forward / backward          |
| `A` / `D`           | Strafe left / right             |
| `Space` / `Ctrl`    | Move up / down                  |
| Hold Right-Mouse + move | Look around (yaw/pitch)     |
| `Shift` (held)      | 3x speed boost                  |
| Scroll wheel        | Increase/decrease fly speed     |
| Left-click +/- (bottom-right corner) | Adjust headlamp brightness |
| Left-click +/- (bottom-right corner, below brightness) | Adjust global ambient light |
| Left-click +/- (bottom-right corner, below global light) | Adjust render/view distance |
| Left-click the minimap (bottom-left) | Jump to that spot in the cave (lands inside the actual passage, near your current level) |
| Left-click the Mesh button (bottom-right corner) | Toggle wireframe overlay on/off |
| Left-click the Texture button (bottom-right corner, below Mesh button) | Toggle photo texture on/off (shows plain lit gray when off) |
| Left-click the Help button (bottom-right corner, below Texture button) | Show/hide the controls reference screen |
| Left-click the Color button (bottom-right corner, below Help button) | Open/close the background color picker |
| Left-click the Open button (bottom-right corner, below Color button) | Switch to a different map without closing the program |
| `Esc`               | Quit                            |

The fly camera has no gravity and no floor constraint -- it moves freely
in 3D like a no-clip/spectator camera, since cave diving isn't ground-locked
movement.

## Smoother streaming while flying fast

Two layers of work went into reducing stutter during fast flythroughs:

1. Chunk uploads to the GPU are prioritized by distance to the camera and
   spread across frames within a small time budget, so a fast flythrough
   that crosses several chunk boundaries per second doesn't dump a big
   batch of uploads into a single frame.
2. Texture decoding (the actual JPEG -> pixels work, which is the slowest
   and most variable-cost part of bringing a new chunk onto screen) now
   happens on the background loading threads, ahead of time, instead of on
   the main thread at the moment a chunk is ready to display. The main
   thread then only does the fast, predictable GPU upload of already-
   decoded pixel data. This was the main remaining cause of stutter on
   fast flythroughs through areas with many never-before-seen textures.

If you still notice hitching on a particularly fast or twisty section, try
increasing `load_radius_cells` in `StreamingConfig` (inside
`gui/viewer_window.py`) to give chunks more lead distance to finish loading
before the camera reaches them, at the cost of more chunks resident in
memory at once.

## Bottom-right control column

Everything that used to be split across the top-left and top-right of
the screen now lives together in one column anchored to the
**bottom-right corner**: brightness, then global light, then render
distance, then the five Mesh/Texture/Help/Color/Open buttons, stacked in
that order.

### Headlamp brightness control

The top item in the column. A small **-** / value / **+** control sets
headlamp brightness. Click **+** or **-** to step the value up or down by
1; each click takes effect immediately. Range is 0-10, default 3. The
buttons dim out when you're at either end of the range (0 or 10), so
it's clear when there's nowhere further to go.

(This replaced an earlier draggable slider -- dragging the handle proved
unreliable for at least one person, so it was swapped for plain click
buttons, which have no continuous drag state that can get out of sync.)

### Global light control

Sits directly below brightness. A small **-** / value / **+** control,
labeled GLOBAL LIGHT, that raises an even ambient fill light across the
**whole cave** -- not the headlamp's local cone of light, but a flat,
direction-independent brightening that washes out shadows everywhere at
once. Range is 0-10, default 0 (0 reproduces the exact ambient level
this app always used before this control existed, so leaving it alone
changes nothing).

This is **not** real simulated light bouncing off the cave walls (true
global illumination) -- that's a much bigger rendering undertaking than
a single control could reasonably do. What this actually does is raise
a flat ambient term in the shader, which is the same practical effect
most small tools mean when they offer a one-button "GI" toggle: a way to
see the whole space clearly without your headlamp doing all the work,
useful for inspecting geometry/texture detail in areas you're not
currently pointed at. At higher values, expect some loss of contrast and
shadow detail, since that's the direct tradeoff of an even, flat fill.

### Render distance control

Sits directly below global light in the same column. A second **-** /
value / **+** control that controls how far the cave map streams in
around you -- directly the same `load_radius_cells` setting described
under "Tuning for your maps" below, but adjustable live with a click
instead of needing to edit code and restart.

The number is how many chunk-rings out from your camera stay loaded
(each map's chunk size varies, but this is typically several meters per
ring -- so a setting of 4 covers noticeably more distance than a setting
of 1). Range is 1-10, default 4. Pushing it up on a very large or open
map will genuinely cost frame rate and cause a brief burst of loading
activity while the newly-included chunks stream in -- that's expected,
not a bug. Watch the console output (it prints your current FPS and
loaded chunk count every couple seconds) to judge whether a given setting
still runs smoothly on your map and machine.

If even the maximum still isn't far enough for a particularly
large/open map, that ceiling can be raised further -- it's deliberately
conservative for now since the right ceiling depends on real-world
testing across different maps and machines.

### Mesh / Texture / Help / Color / Open buttons

Five labeled buttons sit at the bottom of the column, below the render
distance control:

- **MESH** -- toggles a wireframe overlay showing the actual triangle
  edges of the scan, useful for inspecting mesh density/quality.
- **TEXTURE** -- toggles whether the photo texture is shown. With it off,
  surfaces render as plain lit gray (still shaded by your headlamp).
- **HELP** -- brings up the full controls reference screen (the same
  dimmed, centered list shown while a map is loading) any time mid-
  flight, in case you forget a key or want a refresher. Click it again
  to close. Unlike loading-related screens, this one never closes on its
  own -- it stays open until you click HELP a second time.
- **COLOR** -- opens a centered panel with three sliders (Red, Green,
  Blue) for changing the background color in the empty space ("the void")
  around the cave model. See below for details. Click COLOR again to
  close the panel.
- **OPEN** -- switches to a different map without closing the program.
  See below for details.

Mesh and Texture combine into four distinct views:

| Texture | Mesh | What you see |
|---|---|---|
| on  | off | Normal textured view (default) |
| on  | on  | Textured surface with wireframe overlaid on top |
| off | off | Plain lit gray surface, no photo detail |
| off | on  | **Pure wireframe only** -- no solid surface underneath at all |

Each button is highlighted (amber background) when its toggle/screen is
active, dark when inactive.

## Background color picker

Click the **COLOR** button to open a centered panel with three sliders --
**R**, **G**, **B** -- each covering the full 0-255 range. Drag any
slider to change that channel; the cave's background color updates live
as you drag, and a small swatch at the bottom of the panel shows a
preview of the resulting color. Grabbing a slider's handle directly
doesn't jump the value to wherever you clicked (it stays right where it
was relative to your cursor); clicking elsewhere on a slider's track
jumps straight to that position, the same behavior as the other sliders
in this program.

While the panel is open, clicks elsewhere on screen (the minimap, the 3D
view) are captured by the panel rather than passing through to whatever's
behind it -- this is intentional, so adjusting color doesn't accidentally
also teleport you via the minimap underneath. Click COLOR again to close
the panel and return to normal interaction.

The color defaults to the same near-black the viewer always used, so if
you never open this panel, nothing changes from before this feature
existed.

## Checking for updates

The splash screen (the very first window you see on launch) has a small
**Check for Updates** link below the version number. Clicking it:

1. Checks GitHub Releases for a newer version than the one you're
   running.
2. If one exists, shows you the version number and release notes, and
   asks whether to download it.
3. If you say yes, downloads it with a progress bar, then closes
   CaveViewer and hands off to a small separate helper script
   (`gui/updater.py`) that replaces the old files with the new ones and
   relaunches the program.

This entire feature is independent of CaveViewer's normal use --
nothing about viewing a map requires any network access, and a failed
or skipped update check never blocks or interferes with opening a map.
If there's no internet connection when you click the button, you'll get
a clear "couldn't check right now" message and nothing else happens.

**Setup required before this works:** `gui/update_checker.py` has a
`GITHUB_REPO` placeholder (`"YOUR_GITHUB_USERNAME/CaveViewer"`) that
needs to be set to a real GitHub repository before update checks will
do anything other than show a "not configured yet" message. Once you
have a repo:

1. Set `GITHUB_REPO = "your-username/your-repo-name"` in
   `gui/update_checker.py`.
2. Each time you want to publish a new version: bump `__version__` near
   the top of `caveviewer.py`, then create a GitHub Release tagged with
   that same version number (e.g. tag `1.1` to match `__version__ = "1.1"`)
   and attach a `.zip` of the project folder as the release asset.
3. Anyone running an older version can then use Check for Updates to
   pick it up automatically.

Version tags should be plain numbers without a leading "v" (`1.1`, not
`v1.1`) to match how `__version__` is written -- this keeps the
newer-than comparison simple. The release zip is expected to contain a
single top-level folder (matching how this project has always been
packaged for distribution) -- the updater unwraps that folder
automatically rather than nesting your install one level deeper.

## Switching maps with the Open button

Click **OPEN** to switch to a different map without closing and
relaunching the whole program. It shows the same folder-browse dialog you
see on startup -- pick a folder containing a different cave's
`.obj`/`.mtl`/texture files, and the program will:

1. Close out the map you currently have open (releases its memory/GPU
   resources cleanly).
2. If the new map has never been opened before (no cache yet), import and
   chunk it -- same one-time process as opening any brand-new map for the
   first time, with a progress screen showing what's happening.
3. Switch straight into the new map, landing you at its starting
   position.

If the new map has already been opened before (its `.caveviewer_cache`
folder already exists next to its `.obj`), the switch is fast -- no
import needed, similar to how reopening any previously-cached map works.

**A real limitation worth knowing:** if the new map needs that one-time
import, the window is unresponsive while it happens -- the same way the
very first time you ever open any brand-new map works today. The
progress screen updates at each real step of the import (not a smooth
continuous animation, just discrete real progress checkpoints), so it's
clear what's happening rather than the window looking frozen with no
explanation, but it genuinely can't be interacted with until that
one-time cost finishes. Once it's done, all future opens of that same
map -- including via this same Open button -- are fast.

If anything goes wrong while opening a new map (the folder doesn't
contain a valid `.obj`/`.mtl`, or the import fails partway through), the
map you already had open keeps running untouched -- a failed attempt to
open something else never takes down what you were already viewing.

## Minimap

A small panel in the bottom-left corner shows a crude top-down outline of
the entire cave system's footprint, with a red dot marking your current
position, updated live as you fly. The outline is computed once at
startup from the chunk cache's bounding boxes (no extra rendering cost),
and intentionally collapses all vertical levels into one flat silhouette --
for a cave with real depth/multiple levels, a literal top-down render would
just show overlapping passages on top of each other, so a flattened
footprint is the more readable option.

Left-click anywhere on the minimap to jump straight to that spot. Since
the minimap only knows X/Z (it's a flattened top-down view), it looks up
which actual chunk(s) exist at that column and lands you inside whichever
level is closest to your current height -- so in a multi-level cave,
clicking a spot keeps you on roughly the level you're already on rather
than always snapping to the lowest or highest passage there. If the exact
spot clicked doesn't have any chunk directly there (a near-miss on the
crude outline), it snaps to the nearest occupied spot instead of leaving
you in open space.

## FPS / chunk readout

A small two-line readout sits directly above the minimap, showing your
current frame rate and how many chunks are currently loaded (plus how
many are still pending, if any). This is the same information already
printed to the console every couple of seconds, just visible on-screen
too and updating every frame instead of every 2 seconds -- useful for
seeing the effect of something you just changed (e.g. clicking the
render distance +/- control) immediately rather than waiting for the
next console line. The FPS number turns amber below 15fps as a quick
visual cue that something is straining, rather than always staying the
same color regardless of how the map is actually performing.

## Controls overlay (loading screen / reference diagram)

A full-screen overlay appears right when the viewer window opens, showing
every control and UI feature (movement keys, mouse look, the brightness
control, the mesh/texture buttons, minimap click-to-jump) while the first
batch of chunks around your spawn point streams in. It dismisses itself
automatically once enough of that area has loaded -- there's nothing to
click or dismiss manually.

The same diagram appears again, as a smaller panel near the top of the
screen, briefly after any minimap click teleports you somewhere new --
since a teleport is meant to be quick, this panel doesn't wait for the
*entire* new area to finish loading the way the full-screen version does
at startup; it clears itself once a reasonable amount has loaded (with a
hard time limit either way, so it never lingers).

A spinning logo appears above the controls list while either version of
this overlay is visible (above the title on the full-screen version,
beside the title on the smaller panel). It's purely decorative -- tumbles
continuously around a vertical axis (like a coin or playing card spinning
in place: it narrows to a thin sliver edge-on, then widens back out,
darkening slightly at its thinnest point for a subtle 3D shading effect)
at a steady rate for as long as the overlay is shown, with no effect on
loading itself. The image lives at `gui/assets/loading_logo.png`; swap
that file for a different image (any size/aspect ratio) if you ever want
to change it.

## Tuning for your maps

`load_radius_cells` -- how many chunk-rings out from your camera stay
loaded -- is adjustable live via the render distance control in the
bottom-right column (see above), so you usually won't need to touch
code for this anymore. The control's range (1-10) and the map's starting
default are still set in `gui/viewer_window.py`:

```python
config = StreamingConfig(chunk_size=chunk_size, load_radius_cells=4, unload_radius_margin=1)
```

```python
self.render_distance_stepper = StepperControl(
    self.ctx, "VIEW DIST", initial_value=4, min_value=1, max_value=10
)
```

- `load_radius_cells`: the control's starting value when the map opens.
- `unload_radius_margin`: hysteresis -- chunks aren't evicted until this
  many cells beyond the current load radius, so chunks don't get
  loaded/unloaded repeatedly if you're hovering near a boundary. This
  automatically tracks whatever the control is currently set to (it's
  `load_radius_cells + unload_radius_margin`, recalculated live), so it
  doesn't need separate tuning.
- The control's `max_value=10` ceiling can be raised in code if even the
  maximum isn't far enough for a particular map -- it's deliberately
  conservative for now since the right ceiling depends on testing across
  different maps and machines (watch the console's periodic FPS/chunk-
  count printout to judge whether a higher ceiling would still run
  smoothly on your hardware).

In `core/chunker.py`, `DEFAULT_CHUNK_SIZE = 8.0` (meters) controls the grid
cell size used when building the cache. If your cave passages are very
tight and twisty, a smaller chunk size (e.g. 4m) gives finer-grained
streaming control. If your cave is mostly big open rooms, a larger chunk
size (e.g. 15-20m) means fewer draw calls and less per-chunk overhead. If
you change this, delete the `.caveviewer_cache` folder next to your `.obj`
to force a rebuild with the new size (or pass `force_rebuild=True` when
calling `import_and_cache`).

## If something looks wrong

- **Black/magenta triangles:** a texture file referenced by the `.mtl`
  couldn't be found in the folder you selected. Check the console output
  for `[TextureManager] WARNING: texture file missing: ...` -- it'll tell
  you the exact filename it expected. Make sure all the Agisoft-exported
  `.jpg` tiles are in the same folder as the `.obj`/`.mtl`.
- **Mesh looks "inside out" / faces missing from one direction:** Agisoft
  occasionally exports with a winding order CaveViewer doesn't expect.
  Try disabling backface culling temporarily by removing the
  `self.ctx.enable(moderngl.CULL_FACE)` line in `viewer_window.py` to
  confirm, then we can fix the winding order properly.
- **Re-importing a map you've already opened before:** delete the
  `.caveviewer_cache` folder sitting next to your `.obj` file to force a
  fresh import (useful if you re-export an updated version of the map from
  Agisoft with the same filename).

### Setup GUI specific issues

- **"Windows protected your PC" blue screen when double-clicking
  `Launch_Setup.bat`:** this is Windows SmartScreen, which flags any
  downloaded script it doesn't recognize -- it's not a virus warning
  specifically about this file, just a generic "I don't recognize this"
  notice for anything downloaded from the internet. Click **"More info"**,
  then **"Run anyway"**.
- **A blue PowerShell security prompt appears asking about execution
  policy:** click **Yes** or **"Run once"**. The bypass flag in
  `Launch_Setup.bat` is scoped to just that one launch, not a permanent
  change to your system.
- **Setup says Python installed but then fails finding it during the
  requirements step:** close the setup window and double-click
  `Launch_Setup.bat` again -- Windows sometimes needs a fresh process to
  pick up a PATH change from an install that just happened.
- **Nothing happens at all when double-clicking `Launch_Setup.bat`:** right-
  click it and choose "Run as administrator" manually instead -- some
  systems block the automatic elevation prompt from triggering on a
  double-click depending on local security settings.

## Packaging as a standalone .exe (no Python install needed to run it)

If you want to share CaveViewer with someone who shouldn't have to install
Python or run `pip install`, you can package it into a standalone Windows
executable using PyInstaller. This is a one-time build step **you** run on
your own machine (the one that already has everything working) -- the
person you share it with just gets a folder with a `.exe` they double-click.

**What this does and doesn't solve:** packaging bundles Python and all the
library code into the executable, so the *software* install step goes
away. It does **not** remove the need for a working graphics card with
OpenGL 3.3 support and reasonably current drivers -- that's a hardware/
driver requirement no packaging step can bundle away. Anyone you share
this with still needs a real GPU, same as you do now.

### Build steps (run once, on your Windows machine)

From inside the `CaveViewer` folder, with your existing virtual environment
/ Python install that already has `requirements.txt` installed:

```
pip install -r requirements-build.txt
pyinstaller CaveViewer.spec
```

This produces `dist/CaveViewer/` -- a folder containing `CaveViewer.exe`
plus all the bundled libraries it needs. Building can take a few minutes;
let it finish.

### Sharing it

Zip up the entire `dist/CaveViewer/` folder (not just the .exe -- it needs
everything alongside it) and share that. The person you send it to:

1. Unzips it anywhere on their machine.
2. Double-clicks `CaveViewer.exe` inside.
3. Picks their folder containing the `.obj`/`.mtl`/`.jpg` files, same as
   you've been doing.

No Python, no `pip install`, no terminal commands needed on their end. The
console window will still appear when it runs (intentionally -- it shows
import progress and any error messages, which matters if something goes
wrong on a different machine with different hardware/drivers than yours).

### If the build itself fails

PyInstaller bundles whatever is actually importable in the Python
environment you run it from -- if `pip install -r requirements.txt`
worked and the program runs fine via `python caveviewer.py`, the
PyInstaller build should succeed too. If it doesn't, the error it prints
will usually name the specific missing module; the most common cause is a
package that loads internal resource files at runtime in a way
PyInstaller's static analysis doesn't automatically detect (already
handled for moderngl-window and pyglet in `CaveViewer.spec` via
`collect_data_files`/`collect_submodules`, but a future library update
could introduce a new one needing the same treatment).

## What's next (not built yet, intentionally bare-bones for now)

- A real UI (drag-and-drop folder, minimap, depth/distance readout,
  bookmarked waypoints for survey stations)
- On-screen display of current chunk/coordinates for cross-referencing
  with dive notes
- Possibly: collision/proximity warning so you don't fly the camera
  straight through cave walls when moving fast
