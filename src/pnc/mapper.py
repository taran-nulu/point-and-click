"""Turning hand landmarks into a screen position.

The pipeline is:

    landmarks (normalised image coords)
      -> aim_target()          the pointing model, pivoting about a chosen anchor
      -> to_screen_space()     aspect-corrected "camera space"
      -> Projector.__call__   calibrated homography, or a default fit
      -> screen points in Quartz coordinates

Why the middle step exists: the camera frame is 4:3 and the screen is roughly
16:10. Naively mapping normalised [0, 1] coordinates onto the screen stretches
the mapping by ~15%, which shows up as the cursor travelling faster
horizontally than vertically. Correcting for aspect first keeps the geometry
honest, and any residual error is absorbed by the calibration homography.

Pivot points
------------
The fingertip is projected along a ray anchored at some pivot point. Which one
you anchor at decides what motion moves the cursor:

``wrist``
    Rotating your wrist aims. The forearm barely contributes, because rotating
    your forearm about the elbow hardly moves your wrist.

``elbow``
    Rotating your forearm aims, which is a larger and more comfortable range of
    motion than wrist rotation. The lever is ~2.4x longer (elbow-to-fingertip
    ~44 cm against wrist-to-fingertip ~18 cm), so an elbow pivot needs a
    proportionally smaller ``gain`` for the same cursor sensitivity - otherwise
    every movement is exaggerated and the cursor shakes.

Note that at *matched* sensitivity the two reach about as far. The elbow buys
ergonomics, not range; range comes from ``gain`` and calibration, which trade
against jitter.
"""

from __future__ import annotations

import math

import numpy as np

from . import landmarks as L

PIVOT_WRIST = "wrist"
PIVOT_ELBOW = "elbow"
PIVOT_NAMES = (PIVOT_WRIST, PIVOT_ELBOW)


def hand_scale(pts) -> float:
    """A rigid bone pair, used as a depth proxy.

    The wrist-to-middle-knuckle distance is a constant physical length, so its
    apparent length in the image tells us how far the hand is from the camera.
    Dividing by it cancels out changes in apparent size, which is what stops
    the cursor sliding around when you lean towards the screen.
    """
    return math.dist(pts[L.WRIST], pts[L.MIDDLE_MCP])


def pivot_point(pts, pivot: str = PIVOT_WRIST, elbow=None):
    """The anchor the pointing ray radiates from.

    Always returns the wrist when the requested pivot is unavailable, so the
    fallback to a wrist pivot is structural rather than a special case that
    every caller has to remember to handle.
    """
    if pivot == PIVOT_ELBOW and elbow is not None:
        return elbow
    return pts[L.WRIST]


def aim_target(pts, gain: float, min_scale: float = 0.015, pivot: str = PIVOT_WRIST, elbow=None):
    """The pointing model.

    Places the index fingertip at a *fixed distance from the camera*, measured
    in units of hand size, anchored at the pivot::

        u = anchor + (index_tip - anchor) * (gain / scale)

    The returned point is in normalised image coordinates and is deliberately
    **not** clamped to [0, 1]: the gain is what lets you reach the screen
    edges, and clamping would quietly destroy it.

    Returns ``None`` if the hand is implausibly small to measure.
    """
    scale = hand_scale(pts)
    if scale < min_scale:
        return None

    k = gain / scale
    ax, ay = pivot_point(pts, pivot, elbow)
    tx, ty = pts[L.INDEX_TIP]
    return (ax + (tx - ax) * k, ay + (ty - ay) * k)


def to_screen_space(u: float, v: float, aspect: float) -> tuple[float, float]:
    """Normalised image coords -> aspect-corrected camera space."""
    return ((u - 0.5) * aspect, v - 0.5)


def from_screen_space(ax: float, ay: float, aspect: float) -> tuple[float, float]:
    """Camera space -> normalised image coords (inverse of the above)."""
    return (ax / aspect + 0.5, ay + 0.5)


class DefaultProjector:
    """Aspect-corrected scale-and-centre fit into the display bounds.

    Used before calibration exists, so the tool does something sane on its very
    first run instead of refusing to start. Analytically invertible, which the
    calibration overlay uses to show you where to point.
    """

    def __init__(self, aspect: float, width: float, height: float, margin: float = 1.0) -> None:
        self.aspect = aspect
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.scale = margin * min(width / aspect, height)

    def __call__(self, ax: float, ay: float) -> tuple[float, float]:
        return (self.cx + self.scale * ax, self.cy + self.scale * ay)

    def camera_from_screen(self, sx: float, sy: float) -> tuple[float, float]:
        if self.scale == 0:
            return (0.0, 0.0)
        return ((sx - self.cx) / self.scale, (sy - self.cy) / self.scale)


class HomographyProjector:
    """Maps camera space to the screen through the calibrated homography."""

    def __init__(self, matrix) -> None:
        self.matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
        self.inverse = np.linalg.inv(self.matrix)

    def _apply(self, m: np.ndarray, x: float, y: float) -> tuple[float, float]:
        v = m @ np.array([x, y, 1.0])
        w = v[2] if abs(v[2]) > 1e-12 else 1e-12
        return (float(v[0] / w), float(v[1] / w))

    def __call__(self, ax: float, ay: float) -> tuple[float, float]:
        return self._apply(self.matrix, ax, ay)

    def camera_from_screen(self, sx: float, sy: float) -> tuple[float, float]:
        return self._apply(self.inverse, sx, sy)


def load_projector(aspect: float, width: float, height: float, matrix=None) -> DefaultProjector | HomographyProjector:
    if matrix is not None:
        try:
            return HomographyProjector(matrix)
        except np.linalg.LinAlgError:
            # Degenerate calibration; fall back rather than crash mid-session.
            return DefaultProjector(aspect, width, height)
    return DefaultProjector(aspect, width, height)


def clamp_to_display(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    return (
        min(max(x, 0.0), width - 1.0),
        min(max(y, 0.0), height - 1.0),
    )