"""Gesture recognition: the middle finger is the click button.

Folding your middle finger clicks; unfolding it releases. Keeping the index
finger free throughout means aiming never breaks while you click, and holding
the middle finger down gives you a click-drag.

Why the two thresholds differ
-----------------------------
A single threshold makes the click chatter open and shut whenever a finger
lingers near the boundary. MediaPipe landmark noise is easily +-0.05 on the
extension ratio, which is larger than the gap you would want to use. So the
state machine has a dead band: it must cross ``fold_ratio`` to press and
``extend_ratio`` to release, and each takes ``confirm_frames`` consecutive
frames to be believed.
"""

from __future__ import annotations

import math

from . import landmarks as L

PRESS = "press"
RELEASE = "release"


def middle_extension(pts) -> float | None:
    """How extended the middle finger is, as a wrist-distance ratio.

    An extended middle finger puts the tip far beyond the PIP joint; a folded
    one collapses the tip back towards the wrist. The PIP joint is the
    reference because it barely moves when the finger folds.
    """
    wrist = pts[L.WRIST]
    pip_dist = math.dist(pts[L.MIDDLE_PIP], wrist)
    if pip_dist < 1e-6:
        return None
    return math.dist(pts[L.MIDDLE_TIP], wrist) / pip_dist


def finger_bend_degrees(pts, mcp: int, pip: int, tip: int) -> float | None:
    """Interior angle at the PIP joint. ~180 deg straight, <90 deg curled."""
    a = pts[mcp]
    b = pts[pip]
    c = pts[tip]

    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None

    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


class ClickFSM:
    """Debounced press/release state machine with a hold timeout."""

    def __init__(
        self,
        extend_ratio: float = 1.35,
        fold_ratio: float = 1.15,
        confirm_frames: int = 2,
        max_hold_ms: int = 2000,
    ) -> None:
        if fold_ratio >= extend_ratio:
            raise ValueError("fold_ratio must be below extend_ratio (hysteresis band)")

        self.extend_ratio = float(extend_ratio)
        self.fold_ratio = float(fold_ratio)
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_hold_ms = int(max_hold_ms)

        self.pressed = False
        self._armed = True
        self._press_run = 0
        self._release_run = 0
        self._pressed_at_ms: int | None = None
        self.auto_released = False

    def _emit_release(self) -> str:
        self.pressed = False
        self._pressed_at_ms = None
        self._press_run = 0
        self._release_run = 0
        return RELEASE

    def update(self, extension: float | None, now_ms: int) -> str | None:
        """Feed this frame's extension ratio. Returns PRESS, RELEASE or None."""
        if extension is None:
            return None

        # Holding timeout: never let a misdetected fold leave a button stuck.
        if self.pressed and self._pressed_at_ms is not None:
            if now_ms - self._pressed_at_ms >= self.max_hold_ms:
                self._armed = False
                self.auto_released = True
                return self._emit_release()

        if not self.pressed:
            self.auto_released = False
            if extension <= self.fold_ratio:
                self._press_run += 1
                self._release_run = 0
                if self._armed and self._press_run >= self.confirm_frames:
                    self.pressed = True
                    self._armed = False
                    self._press_run = 0
                    self._pressed_at_ms = now_ms
                    return PRESS
            else:
                self._press_run = 0
                if extension >= self.extend_ratio:
                    # Re-arm only once the finger is clearly out, so an
                    # auto-release does not immediately re-trigger.
                    self._armed = True
        else:
            self._release_run += 1
            self._press_run = 0
            if extension >= self.extend_ratio:
                self._armed = True
                if self._release_run >= self.confirm_frames:
                    return self._emit_release()

        return None

    def force_release(self) -> str | None:
        """Safety valve: hand lost, shutting down, user grabbed the mouse."""
        if self.pressed:
            self._armed = False
            return self._emit_release()
        return None