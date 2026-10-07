"""Tests for elbow detection and its trust rules.

A wrong elbow is worse than no elbow: the anchor sits far from the fingertip, so
an error there becomes a large cursor jump. Most of these tests are therefore
about *refusing* to emit an elbow we should not trust, which is why the decision
is a pure function that needs neither a camera nor a pose model.
"""

from __future__ import annotations

import pytest

from pnc.arm import LEFT_ELBOW, RIGHT_ELBOW, pick_index, select_elbow


class FakeLandmark:
    """Just the fields select_elbow reads."""

    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.z = x, y, 0.0
        self.visibility = visibility


def pose_of(
    left_visibility=0.9,
    right_visibility=0.9,
    left_xy=(0.2, 0.8),
    right_xy=(0.8, 0.8),
):
    """One pose: 33 landmarks, mostly unconfident, with two elbows placed."""
    points = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(33)]
    points[LEFT_ELBOW] = FakeLandmark(*left_xy, visibility=left_visibility)
    points[RIGHT_ELBOW] = FakeLandmark(*right_xy, visibility=right_visibility)
    return points


# ---------------------------------------------------------------- visibility


def test_high_visibility_elbow_is_returned():
    assert select_elbow(pose_of(), handedness="Right") == pytest.approx((0.8, 0.8))


def test_low_visibility_elbow_is_refused():
    """The important one.

    Pose keeps emitting coordinates even when the arm is out of frame; they just
    stop meaning anything. Trusting them would amplify garbage through a long
    lever.
    """
    points = pose_of(left_visibility=0.2, right_visibility=0.2)
    assert select_elbow(points, handedness="Right") is None


def test_visibility_threshold_is_configurable():
    points = pose_of(left_visibility=0.3, right_visibility=0.3)
    assert select_elbow(points, min_visibility=0.5) is None
    assert select_elbow(points, min_visibility=0.2) is not None


def test_the_confident_elbow_is_preferred_over_the_doubtful_one():
    """Asymmetric confidence must still yield the visible elbow."""
    points = pose_of(left_visibility=0.95, right_visibility=0.10)
    # Handedness says right, but right is not visible, so nothing is trusted.
    assert select_elbow(points, handedness="Right") is None
    # With no handedness we take the confident side.
    assert select_elbow(points) == pytest.approx((0.2, 0.8))


def test_missing_pose_yields_nothing():
    assert select_elbow(None) is None


# ------------------------------------------------------------- side selection


def test_side_comes_from_handedness():
    points = pose_of(left_xy=(0.2, 0.8), right_xy=(0.8, 0.8))
    assert pick_index(points, "Left") == LEFT_ELBOW
    assert pick_index(points, "Right") == RIGHT_ELBOW


def test_side_can_be_forced():
    """The override exists for the mirrored-frame handedness edge case."""
    points = pose_of()
    assert pick_index(points, "Right", side="left") == LEFT_ELBOW
    assert pick_index(points, "Left", side="right") == RIGHT_ELBOW


def test_side_override_is_case_insensitive():
    """Config spells it lowercase; MediaPipe capitalises. A mismatch would be
    silently ignored, aiming from the wrong elbow."""
    points = pose_of(left_visibility=0.95, right_visibility=0.10)
    # With no override, confidence picks the left elbow here.
    assert pick_index(points) == LEFT_ELBOW
    # An explicit lowercase override must win over the confidence heuristic.
    assert pick_index(points, side="right") == RIGHT_ELBOW
    assert pick_index(points, side="Left") == LEFT_ELBOW


def test_a_typo_in_the_side_override_is_ignored_not_fatal():
    points = pose_of()
    assert pick_index(points, side="leftt") in (LEFT_ELBOW, RIGHT_ELBOW)


def test_unknown_handedness_label_falls_back_to_confidence():
    points = pose_of(left_visibility=0.95, right_visibility=0.30)
    assert pick_index(points, "Ambidextrous") == LEFT_ELBOW


def test_unknown_side_falls_back_to_confidence():
    points = pose_of(left_visibility=0.30, right_visibility=0.95)
    assert pick_index(points, None, side="middle") == RIGHT_ELBOW


# ------------------------------------------------------------------- constants


def test_elbow_indices_match_the_pose_landmark_enum():
    """Guards the magic numbers against a MediaPipe version change."""
    mediapipe = pytest.importorskip("mediapipe")
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark

    assert LEFT_ELBOW == PoseLandmark.LEFT_ELBOW == 13
    assert RIGHT_ELBOW == PoseLandmark.RIGHT_ELBOW == 14
    assert mediapipe is not None