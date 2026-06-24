"""
gui/camera.py

Free-fly (6DOF, "noclip"/spectator style) camera. Cave traversal needs
full pitch/yaw with no ground constraint -- divers move in 3D, not on a
walkable surface -- so this deliberately does NOT behave like an FPS
walking camera. Movement is always relative to the camera's current
look direction (forward = into the screen, regardless of pitch), like
a flight-sim free camera.

Controls (bound in gui/viewer_window.py, documented here for reference):
    W/S       - move forward/backward along view direction
    A/D       - strafe left/right
    Space/Ctrl- move up/down along world Y (or could be view-relative; see note)
    Mouse     - look (yaw/pitch), captured while right-mouse-button held
    Shift     - speed boost multiplier
    Scroll    - adjust base fly speed (useful since cave scale varies a lot)
"""

from __future__ import annotations

import math
import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


class FlyCamera:
    def __init__(self, position=(0.0, 0.0, 0.0), yaw_deg=-90.0, pitch_deg=0.0,
                 move_speed=4.0, mouse_sensitivity=0.12):
        self.position = np.array(position, dtype=np.float64)
        self.yaw = math.radians(yaw_deg)      # rotation around world Y
        self.pitch = math.radians(pitch_deg)  # rotation around local X
        self.move_speed = move_speed          # meters/second at 1x
        self.mouse_sensitivity = mouse_sensitivity
        self.fov_deg = 75.0
        self.near = 0.05
        self.far = 1000.0

        self._pitch_limit = math.radians(89.0)

    # -- orientation -----------------------------------------------------

    def forward(self) -> np.ndarray:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        return _normalize(np.array([cp * cy, sp, cp * sy], dtype=np.float64))

    def right(self) -> np.ndarray:
        world_up = np.array([0.0, 1.0, 0.0])
        return _normalize(np.cross(self.forward(), world_up))

    def up(self) -> np.ndarray:
        return _normalize(np.cross(self.right(), self.forward()))

    def look(self, dx_pixels: float, dy_pixels: float) -> None:
        """Apply mouse delta to yaw/pitch. dy positive = mouse moved down."""
        self.yaw += math.radians(dx_pixels * self.mouse_sensitivity)
        self.pitch -= math.radians(dy_pixels * self.mouse_sensitivity)
        self.pitch = max(-self._pitch_limit, min(self._pitch_limit, self.pitch))

    # -- movement ----------------------------------------------------------

    def move(self, forward_amt: float, right_amt: float, up_amt: float,
              dt: float, speed_multiplier: float = 1.0) -> None:
        """
        forward_amt/right_amt/up_amt are typically -1/0/1 from key state.
        up_amt moves along WORLD up (not view-relative pitch), which feels
        more controllable when navigating tight cave geometry -- you don't
        accidentally fly into the ceiling just because you're looking up.
        """
        speed = self.move_speed * speed_multiplier * dt
        delta = (self.forward() * forward_amt + self.right() * right_amt) * speed
        delta[1] += up_amt * speed  # world-vertical, independent of pitch
        self.position += delta

    def adjust_speed(self, scroll_amt: float) -> None:
        """Multiplicative scroll-wheel speed adjustment; cave passages range
        from <1m crawls to 50m+ rooms, so an additive adjustment would be
        annoying at either extreme -- multiplicative scales naturally."""
        factor = 1.1 ** scroll_amt
        self.move_speed = max(0.1, min(200.0, self.move_speed * factor))

    # -- matrices ------------------------------------------------------------

    def view_matrix(self) -> np.ndarray:
        f = self.forward()
        r = self.right()
        u = self.up()
        pos = self.position

        # standard lookAt-style matrix construction
        m = np.identity(4, dtype=np.float32)
        m[0, 0:3] = r
        m[1, 0:3] = u
        m[2, 0:3] = -f
        m[0, 3] = -np.dot(r, pos)
        m[1, 3] = -np.dot(u, pos)
        m[2, 3] = np.dot(f, pos)
        return m

    def projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        fov_rad = math.radians(self.fov_deg)
        f = 1.0 / math.tan(fov_rad / 2.0)
        near, far = self.near, self.far
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect_ratio
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m
