"""Tests for the click state machine.

The property that matters most is the safety one: however badly the input
behaves, the machine must never leave a mouse button held down.
"""

from __future__ import annotations

import pytest

from pnc import landmarks as L
from pnc.gestures import PRESS, RELEASE, ClickFSM, finger_bend_degrees, middle_extension

EXTENDED = 1.6
FOLDED = 1.0


def make_hand(tip=(0.5, 0.3), pip=(0.5, 0.42), wrist=(0.5, 0.6)):
    pts = [(0.0, 0.0)] * 21
    pts[L.WRIST] = wrist
    pts[L.MIDDLE_PIP] = pip
    pts[L.MIDDLE_TIP] = tip
    return pts


def extension_for(tip, pip=(0.5, 0.42), wrist=(0.5, 0.6)):
    return middle_extension(make_hand(tip=tip, pip=pip, wrist=wrist))


def test_middle_extension_reports_extended_and_folded():
    # tip far beyond the PIP joint
    assert extension_for((0.5, 0.30)) > 1.35
    # tip collapsed back towards the wrist
    assert extension_for((0.5, 0.58)) < 1.15


def test_middle_extension_handles_a_degenerate_hand():
    pts = [(0.0, 0.0)] * 21  # PIP sits on the wrist
    assert middle_extension(pts) is None


def test_fold_then_extend_emits_press_then_release():
    fsm = ClickFSM(confirm_frames=2)

    assert fsm.update(FOLDED, 0) is None  # first frame is never enough
    assert fsm.update(FOLDED, 33) == PRESS

    assert fsm.update(EXTENDED, 100) is None
    assert fsm.update(EXTENDED, 133) == RELEASE


def test_single_noise_frame_does_not_click():
    fsm = ClickFSM(confirm_frames=3)

    assert fsm.update(FOLDED, 0) is None
    assert fsm.update(EXTENDED, 33) is None  # breaks the streak
    assert fsm.update(FOLDED, 66) is None
    assert fsm.update(EXTENDED, 99) is None
    assert fsm.update(FOLDED, 132) is None
    assert not fsm.pressed


def test_hysteresis_band_does_not_chatter():
    """A finger resting inside the band must not open and shut the button."""
    fsm = ClickFSM(extend_ratio=1.35, fold_ratio=1.15, confirm_frames=1)
    now = 0

    for _ in range(20):
        fsm.update(1.20, now)  # between fold and extend
        now += 33
        fsm.update(1.22, now)
        now += 33

    assert not fsm.pressed
    assert fsm.update(1.20, now) is None


def test_hold_timeout_releases_and_requires_unfold_first():
    fsm = ClickFSM(confirm_frames=1, max_hold_ms=500)

    assert fsm.update(FOLDED, 0) == PRESS
    assert fsm.update(FOLDED, 100) is None
    assert fsm.update(FOLDED, 600) == RELEASE  # timed out
    assert fsm.auto_released is True
    assert not fsm.pressed

    # Still folded: must NOT immediately re-press.
    assert fsm.update(FOLDED, 650) is None
    assert fsm.update(FOLDED, 700) is None
    assert not fsm.pressed

    # Unfolding re-arms it, and then the very next folded frame clicks again.
    assert fsm.update(EXTENDED, 750) is None
    assert fsm.update(FOLDED, 800) == PRESS


def test_force_release_is_a_safety_valve():
    fsm = ClickFSM(confirm_frames=1)
    assert fsm.update(FOLDED, 0) == PRESS
    assert fsm.force_release() == RELEASE
    assert not fsm.pressed
    assert fsm.force_release() is None  # idempotent


def test_none_extension_is_ignored():
    """No usable measurement this frame must not be read as 'folded'."""
    fsm = ClickFSM(confirm_frames=1)
    assert fsm.update(None, 0) is None
    assert fsm.update(None, 33) is None
    assert not fsm.pressed


def test_rejects_a_nonsensical_hysteresis_band():
    with pytest.raises(ValueError):
        ClickFSM(extend_ratio=1.1, fold_ratio=1.3)


def test_finger_bend_degrees():
    straight = [(0.0, 0.0), (0.0, -1.0), (0.0, -2.0)]
    assert finger_bend_degrees(straight, 0, 1, 2) == pytest.approx(180.0)

    curled = [(0.0, -1.0), (0.0, 0.0), (1.0, 0.0)]
    assert finger_bend_degrees(curled, 0, 1, 2) == pytest.approx(90.0)