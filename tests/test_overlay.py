"""Rendering tests for the preview overlay.

These call the drawing functions directly on a synthetic frame and assert the
image actually changed. That covers every drawing path without needing a window
server, so a typo in an overlay call cannot reach a live session unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pnc import landmarks as L
from pnc import overlay
from tests.test_pipeline import pointing_hand


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), np.uint8)


@pytest.fixture
def connections():
    """The real MediaPipe connection table, if mediapipe is importable."""
    try:
        from pnc.hands import connections as real

        return real()
    except Exception:  # pragma: no cover - exercised only without mediapipe
        return [type("C", (), {"start": 0, "end": 8})()]


def changed(before, after) -> bool:
    return not np.array_equal(before, after)


def test_draw_hand_marks_the_frame(frame, connections):
    before = frame.copy()
    overlay.draw_hand(frame, pointing_hand(dx=1), connections, highlight_middle=True)
    assert changed(before, frame)


def test_draw_aim_point_marks_the_frame(frame):
    before = frame.copy()
    overlay.draw_aim_point(frame, (0.6, 0.4))
    assert changed(before, frame)


def test_draw_target_ring_marks_the_frame(frame):
    before = frame.copy()
    overlay.draw_target_ring(frame, (0.5, 0.5), overlay._CYAN, "top-left")
    assert changed(before, frame)


def test_draw_progress_marks_the_frame(frame):
    before = frame.copy()
    overlay.draw_progress(frame, 0.5, overlay._GREEN)
    assert changed(before, frame)


def test_draw_hud_marks_the_frame(frame):
    before = frame.copy()
    overlay.draw_hud(frame, [("state TRACKING", overlay._GREEN), ("30 fps", overlay._WHITE)])
    assert changed(before, frame)


def test_banner_marks_the_frame(frame):
    before = frame.copy()
    overlay.banner(frame, "show your hand")
    assert changed(before, frame)


def test_minimap_marks_the_frame(frame):
    before = frame.copy()
    overlay.draw_minimap(frame, (0.0, 0.0, 1728.0, 1117.0), (864.0, 558.5))
    assert changed(before, frame)


def test_minimap_handles_out_of_range_positions(frame):
    """A cursor outside the display must not crash the render."""
    overlay.draw_minimap(frame, (0.0, 0.0, 1728.0, 1117.0), (-5000.0, 99999.0))
    overlay.draw_minimap(frame, (0.0, 0.0, 1728.0, 1117.0), (0.0, 0.0))


def test_drawing_survives_landmarks_outside_the_frame(frame, connections):
    """MediaPipe can report coordinates slightly outside [0, 1]."""
    hand = pointing_hand(dx=1)
    hand[L.INDEX_TIP] = (-0.2, 1.4)
    overlay.draw_hand(frame, hand, connections, highlight_middle=True)
    overlay.draw_aim_point(frame, (1.8, -0.6))