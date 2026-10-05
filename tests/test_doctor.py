"""Tests for doctor's threshold advice.

The gesture thresholds in config/default.toml are estimates, and they are
highly hand-specific. ``assess_thresholds`` is the logic that tells you your
estimates are wrong, so it matters that it gets the *direction* of the advice
right: telling someone to lower a threshold that they should raise would make
the problem worse.
"""

from __future__ import annotations

from pnc.doctor import assess_thresholds

MIDDLE_PRESS, MIDDLE_RELEASE = 1.15, 1.35
PINCH_PRESS, PINCH_RELEASE = 0.35, 0.50


def spread(lo, hi, n=1000):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def test_no_samples_is_reported():
    assert assess_thresholds([], 1.15, 1.35) == ["no samples"]


def test_well_separated_range_needs_no_advice():
    """The middle finger clearly spans 0.6 (folded) to 2.4 (extended)."""
    assert assess_thresholds(spread(0.6, 2.4), MIDDLE_PRESS, MIDDLE_RELEASE) == []


def test_a_clearly_pinchable_hand_needs_no_advice():
    """Pinch spans 0.15 (closed) to 0.8 (open), straddling 0.35/0.50."""
    assert assess_thresholds(spread(0.15, 0.8), PINCH_PRESS, PINCH_RELEASE) == []


def test_release_threshold_never_reached_is_reported():
    """This hand's extension stays under 1.35, so a click would never release."""
    warnings = assess_thresholds(spread(0.8, 1.30), MIDDLE_PRESS, MIDDLE_RELEASE)
    assert any("never reaches release" in w for w in warnings)
    assert any("Lower it" in w for w in warnings)


def test_press_threshold_never_reached_is_reported():
    """This hand never folds below 1.15, so clicks would never fire."""
    warnings = assess_thresholds(spread(1.40, 2.4), MIDDLE_PRESS, MIDDLE_RELEASE)
    assert any("never drops to press" in w for w in warnings)
    assert any("Raise it" in w for w in warnings)


def test_thresholds_inside_the_rest_range_are_reported():
    """Never moving the finger at all: any threshold here misfires."""
    warnings = assess_thresholds(spread(1.16, 1.34), MIDDLE_PRESS, MIDDLE_RELEASE)
    assert any("rest range" in w for w in warnings)


def test_suggested_thresholds_are_within_the_observed_range():
    """The advice must point at values the hand can actually produce."""
    warnings = assess_thresholds(spread(0.8, 1.30), MIDDLE_PRESS, MIDDLE_RELEASE)
    suggested = float(warnings[0].split("about ")[1].rstrip("."))
    assert 0.8 <= suggested <= 1.30


def test_configured_defaults_would_be_flagged_for_a_dead_hand():
    """Sanity check on the direction of the original complaint."""
    # A hand that never varies: both thresholds are unusable.
    warnings = assess_thresholds(spread(0.20, 0.24), PINCH_PRESS, PINCH_RELEASE)
    assert warnings