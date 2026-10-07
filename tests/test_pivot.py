"""Tests for the pivot point: what the pointing ray rotates about.

Two facts drive everything here, and the second one is counter-intuitive enough
that it is worth stating plainly.

**Translation is anchor-independent.** Since the projection is
``target = anchor + (tip - anchor) * k``, moving the fingertip by ``Δ`` moves the
target by exactly ``Δ * k`` no matter where the anchor is - the anchor terms
cancel. So the pivot choice does not change how responsive the cursor is to
translating your hand around. It only changes what happens when the finger
*rotates*.

**Rotation is not.** Rotating your forearm about the elbow swings the fingertip
through a long arc, and the elbow pivot amplifies that by the full lever length.
The wrist pivot sees the same rotation mostly as the wrist moving, so it
follows only weakly.

A practical consequence: the right ``gain_elbow`` depends on the user's arm
geometry and on which motion they intend to aim with, so it is a starting value
to tune with ``+``/``-`` rather than something derivable on paper. These tests
pin down the parts that *are* derivable.
"""

from __future__ import annotations

import math

import pytest

from pnc import landmarks as L
from pnc.config import Config
from pnc.mapper import (
    PIVOT_ELBOW,
    PIVOT_WRIST,
    aim_target,
    hand_scale,
    pivot_point,
)

GAIN = 0.5
SCALE = 0.05  # hand size, in normalised frame units

# Real proportions, in hand-lengths: wrist-to-fingertip ~18 cm, wrist-to-elbow
# ~24 cm, so elbow-to-fingertip ~44 cm against a ~9.5 cm hand.
ELBOW_TO_FINGERTIP = 4.6
WRIST_TO_FINGERTIP = 1.9


def make_hand(wrist=(0.5, 0.5), tip=None, middle_mcp=(0.5, 0.45)):
    if tip is None:
        tip = (wrist[0] + WRIST_TO_FINGERTIP * SCALE, wrist[1])
    pts = [wrist] * 21
    pts[L.WRIST] = wrist
    pts[L.INDEX_TIP] = tip
    pts[L.MIDDLE_MCP] = middle_mcp
    return pts


def rotate_hand(pts, centre, degrees):
    """Rigidly rotate the tracked landmarks about a fixed point.

    Models the forearm swinging about the elbow: every point moves on an arc of
    the same radius, and the hand keeps its shape.
    """
    theta = math.radians(degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rotate(p):
        dx, dy = p[0] - centre[0], p[1] - centre[1]
        return (
            centre[0] + dx * cos_t - dy * sin_t,
            centre[1] + dx * sin_t + dy * cos_t,
        )

    tracked = (L.WRIST, L.INDEX_TIP, L.MIDDLE_MCP)
    out = list(pts)
    for index in tracked:
        out[index] = rotate(pts[index])
    return out


def elbow_below_hand(wrist=(0.5, 0.5), distance=ELBOW_TO_FINGERTIP):
    """An elbow positioned so that wrist + lever lands on the fingertip."""
    return (wrist[0], wrist[1] + distance * SCALE)


def aim_delta(before, after, gain, pivot, elbow=None):
    return math.dist(
        aim_target(before, gain, pivot=pivot, elbow=elbow),
        aim_target(after, gain, pivot=pivot, elbow=elbow),
    )


# ------------------------------------------------------------ regression guard


def test_wrist_pivot_is_bit_identical_to_the_old_model():
    """The formula from before pivot support existed, spelled out longhand."""
    pts = make_hand()
    k = GAIN / hand_scale(pts)
    expected = (
        pts[L.WRIST][0] + (pts[L.INDEX_TIP][0] - pts[L.WRIST][0]) * k,
        pts[L.WRIST][1] + (pts[L.INDEX_TIP][1] - pts[L.WRIST][1]) * k,
    )

    got = aim_target(pts, GAIN)
    assert got == pytest.approx(expected)
    assert got == pytest.approx(aim_target(pts, GAIN, pivot=PIVOT_WRIST))
    assert got == pytest.approx(aim_target(pts, GAIN, pivot=PIVOT_ELBOW, elbow=None))


def test_wrist_pivot_ignores_a_supplied_elbow():
    """Asking for the wrist must not be able to pick up the elbow."""
    pts = make_hand()
    assert aim_target(pts, GAIN, pivot=PIVOT_WRIST) == pytest.approx(
        aim_target(pts, GAIN, pivot=PIVOT_WRIST, elbow=(0.1, 0.9))
    )


# ------------------------------------------------------------------ pivot_point


def test_pivot_point_returns_the_wrist_by_default():
    pts = make_hand()
    assert pivot_point(pts) == pts[L.WRIST]
    assert pivot_point(pts, PIVOT_WRIST) == pts[L.WRIST]


def test_pivot_point_returns_the_elbow_when_asked_and_available():
    assert pivot_point(make_hand(), PIVOT_ELBOW, (0.2, 0.8)) == (0.2, 0.8)


def test_pivot_point_falls_back_to_the_wrist_without_an_elbow():
    """Fallback is structural, so no caller can forget to handle it."""
    pts = make_hand()
    assert pivot_point(pts, PIVOT_ELBOW, None) == pts[L.WRIST]


# ------------------------------------------- translation is anchor-independent


@pytest.mark.parametrize("anchor_offset", [(0.5, 0.9), (0.2, 0.2), (0.9, 1.3)])
def test_translation_sensitivity_does_not_depend_on_the_anchor(anchor_offset):
    """The anchor cancels for translation, which is why the two pivots differ
    only in how they respond to rotation."""
    wrist = (0.5, 0.5)
    base = make_hand(wrist=wrist)
    moved = make_hand(wrist=wrist, tip=(base[L.INDEX_TIP][0] + 0.02, base[L.INDEX_TIP][1] - 0.02))

    wrist_move = aim_delta(base, moved, GAIN, PIVOT_WRIST)
    elbow_move = aim_delta(base, moved, GAIN, PIVOT_ELBOW, elbow=anchor_offset)

    assert wrist_move == pytest.approx(elbow_move, rel=1e-9)


def test_translation_response_is_proportional_to_gain():
    """Sanity check that the cancelling is real, not a degenerate case."""
    wrist = (0.5, 0.5)
    base = make_hand(wrist=wrist)
    moved = make_hand(wrist=wrist, tip=(base[L.INDEX_TIP][0] + 0.02, base[L.INDEX_TIP][1]))

    # Doubling gain doubles the cursor movement for the same hand motion.
    assert aim_delta(base, moved, 2 * GAIN, PIVOT_WRIST) == pytest.approx(
        2 * aim_delta(base, moved, GAIN, PIVOT_WRIST)
    )


# ------------------------------------------------ rotation is anchor-dependent


def test_elbow_pivot_responds_more_to_forearm_rotation():
    """The whole reason the elbow pivot exists."""
    wrist = (0.5, 0.5)
    elbow = elbow_below_hand(wrist)
    base = make_hand(wrist=wrist)
    rotated = rotate_hand(base, elbow, 20.0)

    wrist_swing = aim_delta(base, rotated, GAIN, PIVOT_WRIST)
    elbow_swing = aim_delta(base, rotated, GAIN, PIVOT_ELBOW, elbow=elbow)

    # Measured at ~2.5x for real arm proportions; assert the asymmetry without
    # pinning an exact figure that would break on a geometry tweak.
    assert elbow_swing > 2 * wrist_swing


def test_rotation_advantage_is_consistent_across_angles():
    """It should be a property of the geometry, not of one lucky angle."""
    wrist = (0.5, 0.5)
    elbow = elbow_below_hand(wrist)
    base = make_hand(wrist=wrist)

    ratios = []
    for degrees in (5, 15, 30):
        rotated = rotate_hand(base, elbow, degrees)
        ratios.append(
            aim_delta(base, rotated, GAIN, PIVOT_ELBOW, elbow=elbow)
            / aim_delta(base, rotated, GAIN, PIVOT_WRIST)
        )
    assert max(ratios) - min(ratios) < 0.05


def test_gain_elbow_is_tuned_to_equalise_rotation_sensitivity():
    """This is the derivation behind ``gain_elbow``.

    At the same gain the elbow pivot swings ~2.5x further for the same forearm
    rotation, because its lever is that much longer. Dividing the wrist gain by
    that ratio is what makes an elbow pivot feel like the wrist pivot did.
    """
    cfg = Config()
    wrist = (0.5, 0.5)
    elbow = elbow_below_hand(wrist)
    base = make_hand(wrist=wrist)
    rotated = rotate_hand(base, elbow, 20.0)

    wrist_swing = aim_delta(base, rotated, cfg.aim.gain, PIVOT_WRIST)
    elbow_swing = aim_delta(
        base, rotated, cfg.aim.gain_elbow, PIVOT_ELBOW, elbow=elbow
    )
    assert elbow_swing == pytest.approx(wrist_swing, rel=0.05)


def test_elbow_pivot_depends_on_where_the_elbow_is():
    pts = make_hand()
    near = aim_target(pts, GAIN, pivot=PIVOT_ELBOW, elbow=(0.5, 0.7))
    far = aim_target(pts, GAIN, pivot=PIVOT_ELBOW, elbow=(0.5, 1.4))
    assert near != pytest.approx(far)


# ------------------------------------------------------------------ guardrails


def test_degenerate_hand_is_rejected_under_both_pivots():
    pts = make_hand(wrist=(0.5, 0.5), tip=(0.5, 0.5), middle_mcp=(0.5, 0.5))
    assert aim_target(pts, GAIN, pivot=PIVOT_WRIST) is None
    assert aim_target(pts, GAIN, pivot=PIVOT_ELBOW, elbow=(0.5, 0.9)) is None


def test_config_defaults_are_a_sane_starting_point():
    """Not a derived constant - just a starting value, lower than the wrist's
    because the elbow's lever is longer. Worth asserting so a bad edit stands
    out."""
    cfg = Config()
    assert cfg.pivot == PIVOT_WRIST
    assert 0.0 < cfg.aim.gain_elbow < cfg.aim.gain


def test_config_parses_the_pivot_and_arm_sections():
    from pnc.config import load_config

    cfg = load_config()
    assert cfg.pivot in (PIVOT_WRIST, PIVOT_ELBOW)
    assert 0.0 < cfg.arm.min_visibility <= 1.0
    assert cfg.arm.side in ("auto", "left", "right")