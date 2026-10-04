"""Tests for the aiming model.

The behaviour that matters: leaning towards the camera must not move the
cursor, and moving your hand must.
"""

from __future__ import annotations

import math

import pytest

from pnc import landmarks as L
from pnc.mapper import (
    DefaultProjector,
    HomographyProjector,
    aim_target,
    clamp_to_display,
    from_screen_space,
    hand_scale,
    to_screen_space,
)


def make_hand(wrist=(0.5, 0.5), tip=(0.7, 0.4), middle_mcp=(0.45, 0.45), middle_tip=(0.45, 0.3)):
    """A minimal 21-point hand; only four landmarks are ever read."""
    pts = [(0.0, 0.0)] * 21
    pts[L.WRIST] = wrist
    pts[L.INDEX_TIP] = tip
    pts[L.MIDDLE_MCP] = middle_mcp
    pts[L.MIDDLE_TIP] = middle_tip
    return pts


def test_hand_scale_is_the_bone_length():
    pts = make_hand(wrist=(0.5, 0.5), middle_mcp=(0.5, 0.45))
    assert hand_scale(pts) == pytest.approx(0.05)


def test_aim_target_is_anchored_at_the_wrist():
    pts = make_hand(wrist=(0.5, 0.5), tip=(0.7, 0.4), middle_mcp=(0.5, 0.45))
    gain = 2.0
    target = aim_target(pts, gain)

    scale = 0.05
    assert target[0] == pytest.approx(0.5 + (0.7 - 0.5) * gain / scale)
    assert target[1] == pytest.approx(0.5 + (0.4 - 0.5) * gain / scale)


def test_aim_target_ignores_depth_changes():
    """The whole point of dividing by hand size: leaning in must not move it.

    A hand twice as far from the camera is drawn at half size about the wrist,
    so every landmark offset shrinks - including the fingertip. If the model
    fails to cancel that out, the cursor slides when you lean.
    """
    wrist = (0.5, 0.5)
    base_mcp = (0.5, 0.45)
    base_tip = (0.7, 0.4)

    def at_depth(factor):
        def pull(point):
            return (
                wrist[0] + (point[0] - wrist[0]) * factor,
                wrist[1] + (point[1] - wrist[1]) * factor,
            )

        return make_hand(wrist=wrist, tip=pull(base_tip), middle_mcp=pull(base_mcp))

    near = at_depth(1.0)
    far = at_depth(0.5)

    assert hand_scale(near) == pytest.approx(2 * hand_scale(far))  # genuinely deeper
    assert aim_target(near, 2.6) == pytest.approx(aim_target(far, 2.6))


def test_aim_target_responds_to_lateral_hand_movement():
    left = aim_target(make_hand(wrist=(0.3, 0.5), tip=(0.5, 0.4), middle_mcp=(0.3, 0.45)), 2.6)
    right = aim_target(make_hand(wrist=(0.7, 0.5), tip=(0.9, 0.4), middle_mcp=(0.7, 0.45)), 2.6)
    assert right[0] > left[0]


def test_aim_target_can_exceed_the_frame():
    """The gain is what reaches the screen edges, so it must not be clamped."""
    target = aim_target(make_hand(wrist=(0.9, 0.9), tip=(0.95, 0.95), middle_mcp=(0.9, 0.88)), 3.0)
    assert target[0] > 1.0 or target[1] > 1.0


def test_aim_target_rejects_a_degenerate_hand():
    pts = make_hand(wrist=(0.5, 0.5), tip=(0.5, 0.5), middle_mcp=(0.5, 0.5))
    assert aim_target(pts, 2.6, min_scale=0.015) is None


def test_screen_space_roundtrip():
    aspect = 640 / 480
    ax, ay = to_screen_space(0.3, 0.7, aspect)
    assert from_screen_space(ax, ay, aspect) == pytest.approx((0.3, 0.7))


def test_camera_space_is_isotropic():
    """Equal steps in camera space must travel equal distances in pixels.

    Camera space has already folded the frame's aspect ratio into its x axis, so
    a unit step is the same number of pixels either way. The projector applying
    one uniform scale is therefore correct, and a 2:1 screen step would be the
    bug.
    """
    projector = DefaultProjector(16 / 10, 1000, 1000)
    x0, y0 = projector(0.0, 0.0)
    x1, y1 = projector(1.0, 1.0)
    assert (x1 - x0) == pytest.approx(y1 - y0)


def test_default_projector_fits_without_distortion():
    aspect = 16 / 10
    projector = DefaultProjector(aspect, 1600, 1600)
    # Fits the 1.6 x 1.0 region inside a square, so it is limited by width.
    assert projector.scale == pytest.approx(1600 / aspect)
    assert projector.aspect == aspect


def test_default_projector_inverts():
    projector = DefaultProjector(1.33, 800, 600)
    assert projector.camera_from_screen(*projector(0.2, -0.3)) == pytest.approx((0.2, -0.3))


def test_default_projector_centres_the_mapping():
    projector = DefaultProjector(1.5, 800, 600)
    x, y = projector(0.0, 0.0)
    assert (x, y) == pytest.approx((400.0, 300.0))


def test_homography_projector_roundtrip():
    matrix = [[1.2, 0.1, 40.0], [-0.05, 0.9, 20.0], [1e-4, -2e-4, 1.0]]
    projector = HomographyProjector(matrix)
    point = projector.camera_from_screen(*projector(0.1, 0.2))
    assert point == pytest.approx((0.1, 0.2))


def test_homography_maps_corners_as_fitted():
    """The four-corner fit must reproduce those corners exactly."""
    from pnc.calibration import build_homography

    samples = {
        "top-left": (0.10, 0.20),
        "top-right": (0.90, 0.22),
        "bottom-right": (0.88, 0.80),
        "bottom-left": (0.12, 0.78),
    }
    matrix = build_homography(samples, origin=(0.0, 0.0), size=(1728.0, 1117.0))
    projector = HomographyProjector(matrix)

    expected = {
        "top-left": (0.0, 0.0),
        "top-right": (1728.0, 0.0),
        "bottom-right": (1728.0, 1117.0),
        "bottom-left": (0.0, 1117.0),
    }
    for name, target in expected.items():
        assert math.dist(projector(*samples[name]), target) < 1e-3


def test_calibration_maps_the_centre_close_to_the_centre():
    """The centre sample is a free accuracy check on the fit."""
    from pnc.calibration import build_homography, validate

    samples = {
        "centre": (0.50, 0.50),
        "top-left": (0.10, 0.20),
        "top-right": (0.90, 0.22),
        "bottom-right": (0.88, 0.80),
        "bottom-left": (0.12, 0.78),
    }
    matrix = build_homography(samples, origin=(0.0, 0.0), size=(1728.0, 1117.0))
    error_px, error_pct = validate(matrix, samples, (0.0, 0.0), (1728.0, 1117.0))
    assert error_pct < 2.0
    assert error_px < 40.0


def test_clamp_to_display():
    assert clamp_to_display(-5.0, 9999.0, 100.0, 100.0) == (0.0, 99.0)