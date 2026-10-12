"""Tests for doctor's threshold advice.

The gesture thresholds in config/default.toml are estimates, and they are
highly hand-specific. ``assess_thresholds`` is the logic that tells you your
estimates are wrong, so it matters that it gets the *direction* of the advice
right: telling someone to lower a threshold that they should raise would make
the problem worse.
"""

from __future__ import annotations

import re

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
    assert warnings
    assert "never reached" in warnings[0]


def test_press_threshold_never_reached_is_reported():
    """This hand never folds below 1.15, so clicks would never fire."""
    warnings = assess_thresholds(spread(1.40, 2.4), MIDDLE_PRESS, MIDDLE_RELEASE)
    assert warnings
    assert "never reached" in warnings[0]


def test_thresholds_inside_the_rest_range_are_reported():
    """A band wide enough that ordinary hand movement lives inside it means
    clicks fire unreliably, even though both limits are reachable."""
    warnings = assess_thresholds(spread(0.2, 2.0), 0.6, 1.8)
    assert any("dead band" in w for w in warnings)


def test_suggested_thresholds_are_within_the_observed_range():
    """The advice must point at values the hand can actually produce."""
    warnings = assess_thresholds(spread(0.2, 2.0), 0.6, 1.8)
    numbers = [float(n) for n in re.findall(r"press=([\d.]+)", warnings[0])]
    assert numbers
    for value in numbers:
        assert 0.2 <= value <= 2.0


def test_a_narrow_band_in_the_right_place_is_accepted():
    """Both extremes reached, and most movement is outside the dead band."""
    assert assess_thresholds(spread(0.3, 2.2), 1.15, 1.35) == []


def test_configured_defaults_would_be_flagged_for_a_dead_hand():
    """A hand that never varies cannot use any threshold."""
    warnings = assess_thresholds(spread(0.20, 0.24), PINCH_PRESS, PINCH_RELEASE)
    assert warnings


# ------------------------------------------- distinguishing "not performed" from
#                                                          "threshold is wrong"


def test_a_resting_hand_is_reported_as_not_performed_not_mistuned():
    """The mistake this prevents.

    Real captured data from a resting hand: p5=0.60, median=1.17, p95=1.61. The
    old advice here was "raise press to about 1.02" - which is right where
    resting hands sit, so following it would make every resting frame click.
    """
    warnings = assess_thresholds(spread(0.60, 1.61), 0.60, 0.85)
    assert len(warnings) == 1
    assert "not actually performed" in warnings[0]
    assert "Raise it" not in warnings[0]
    assert "Lower it" not in warnings[0]


def test_the_resting_hand_report_quotes_the_misfire_share():
    """Tells the user how much of the capture was in each state, which is what
    makes 'you never did the gesture' credible."""
    warnings = assess_thresholds(spread(0.60, 1.61), 0.60, 0.85)
    assert "would press" in warnings[0]
    assert "%" in warnings[0]


def test_a_deliberate_sweep_is_accepted():
    """A genuine clench-to-open sweep reaches both extremes and gets no warning."""
    assert assess_thresholds(spread(0.2, 1.8), 0.6, 0.85) == []


def test_a_hand_that_only_ever_opens_is_not_accepted():
    """Reaching the release extreme alone is not evidence of a fist."""
    warnings = assess_thresholds(spread(1.0, 1.8), 0.6, 0.85)
    assert any("never reached" in w for w in warnings)