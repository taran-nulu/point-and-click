"""Guided calibration: learning where you actually point.

The webcam's field of view is much narrower than the screen, so there is no
mapping from "which pixel my fingertip is on" to "which pixel on the screen"
that is comfortable to use. Calibration learns one.

You are shown five targets on the *screen*, and the overlay simultaneously
shows a crosshair at the place in the camera frame you should be pointing to
reach each one. Align the fingertip with the crosshair and the sample is
captured automatically - so you get feedback rather than having to trust
yourself to aim at a physical bezel corner.

The four corners fit a 3x3 homography. The centre sample is not used for the
fit; it is a free accuracy check, because a homography that nailed all four
corners and missed the middle would look fine and feel awful.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

# (label, fractional x, fractional y) within the display.
TARGETS: tuple[tuple[str, float, float], ...] = (
    ("centre", 0.5, 0.5),
    ("top-left", 0.0, 0.0),
    ("top-right", 1.0, 0.0),
    ("bottom-right", 1.0, 1.0),
    ("bottom-left", 0.0, 1.0),
)
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


def screen_point(fx: float, fy: float, origin: tuple[float, float], size: tuple[float, float]):
    return (origin[0] + size[0] * fx, origin[1] + size[1] * fy)


def quad_area(points) -> float:
    """Area of a 4-point polygon. Guards against degenerate calibration input."""
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def build_homography(samples: dict, origin, size) -> np.ndarray:
    """Fit camera space -> screen points from the four corner samples."""
    src: list[list[float]] = []
    dst: list[list[float]] = []

    for name, fx, fy in TARGETS:
        if name == "centre":
            continue
        if name not in samples:
            raise ValueError(f"Missing calibration sample for {name!r}")
        ax, ay = samples[name]
        sx, sy = screen_point(fx, fy, origin, size)
        src.append([ax, ay])
        dst.append([sx, sy])

    if quad_area(src) < 1e-4:
        raise ValueError(
            "Calibration samples are nearly collinear - the hand probably did not "
            "move far enough between targets. Recalibrate and sweep wider."
        )

    matrix = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    if not np.all(np.isfinite(matrix)) or abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Calibration produced a degenerate mapping; try again.")
    return matrix


def validate(matrix, samples: dict, origin, size) -> tuple[float, float]:
    """Centre-sample error as ``(points, percent of screen diagonal)``."""
    if "centre" not in samples:
        return (float("nan"), float("nan"))

    ax, ay = samples["centre"]
    v = np.asarray(matrix, float) @ np.array([ax, ay, 1.0])
    if abs(v[2]) < 1e-12:
        return (float("inf"), float("inf"))

    expected = screen_point(0.5, 0.5, origin, size)
    error = math.dist((float(v[0] / v[2]), float(v[1] / v[2])), expected)
    diagonal = math.hypot(size[0], size[1])
    return (error, error / diagonal * 100.0)


def save(matrix, path: Path, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"homography": np.asarray(matrix, float).tolist(), **(meta or {})}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load(path: Path):
    """Saved homography matrix, or ``None`` if absent/unreadable."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    matrix = data.get("homography")
    if not matrix:
        return None
    array = np.asarray(matrix, float)
    return array if array.shape == (3, 3) and np.all(np.isfinite(array)) else None


def load_meta(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if k != "homography"}