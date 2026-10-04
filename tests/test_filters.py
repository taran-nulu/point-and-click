"""Tests for the smoothing filters and the physical-input guard."""

from __future__ import annotations

import random

import pytest

from pnc.controls import InputGuard
from pnc.filters import CursorSmoother, MedianFilter, OneEuroFilter


class FakeCursor:
    """Stands in for the Quartz cursor: just a position you can poke."""

    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y

    def position(self):
        return (self.x, self.y)


# --------------------------------------------------------------------- filters


def test_median_filter_removes_a_spike():
    f = MedianFilter(3)
    assert [f.filter(v) for v in (10.0, 10.0, 999.0, 10.0)] == [10.0, 10.0, 10.0, 10.0]


def test_median_filter_window_of_one_is_a_no_op():
    f = MedianFilter(1)
    assert f.filter(7.5) == 7.5


def test_one_euro_filter_settles_on_a_constant_input():
    f = OneEuroFilter(min_cutoff=1.2, beta=0.02)
    out = [f.filter(5.0, dt=1 / 30) for _ in range(200)]
    assert out[-1] == pytest.approx(5.0, abs=0.01)


def test_one_euro_filter_suppresses_jitter_when_still():
    """Noise around a fixed point must be damped."""
    rng = random.Random(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.005)
    raw, out = [], []
    for _ in range(400):
        value = 100.0 + rng.uniform(-2, 2)
        raw.append(value)
        out.append(f.filter(value, dt=1 / 30))

    raw_spread = max(raw[-50:]) - min(raw[-50:])
    out_spread = max(out[-50:]) - min(out[-50:])
    assert out_spread < raw_spread / 2


def test_one_euro_filter_catches_up_when_moving_fast():
    """The adaptive cutoff exists so fast motion is not laggy."""
    f = OneEuroFilter(min_cutoff=1.2, beta=0.02)
    start = f.filter(0.0, dt=1 / 30)
    value = 0.0
    for _ in range(6):
        value += 50.0
        f.filter(value, dt=1 / 30)
    assert f.filter(value, dt=1 / 30) > start + 40.0


def test_one_euro_filter_rejects_a_bad_cutoff():
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0)


def test_cursor_smoother_output_stays_finite_under_garbage():
    s = CursorSmoother(median_window=3, min_cutoff=1.2, beta=0.02)
    rng = random.Random(1)
    for _ in range(100):
        x, y = s.filter(rng.uniform(0, 1728), rng.uniform(0, 1117), dt=1 / 30)
    assert 0 <= x <= 1728 and 0 <= y <= 1117


def test_cursor_smoother_reset_stops_drifting():
    s = CursorSmoother()
    s.filter(100.0, 200.0, dt=1 / 30)
    s.reset()
    assert s.filter(500.0, 500.0, dt=1 / 30) == pytest.approx((500.0, 500.0))


# ----------------------------------------------------------------- inputguard


def test_guard_ignores_our_own_moves():
    guard = InputGuard()
    cursor = FakeCursor(10, 20)
    guard.note_write(cursor.position())

    assert not guard.detect(cursor, now_ms=0)


def test_guard_trips_when_a_human_moves_the_mouse():
    guard = InputGuard()
    cursor = FakeCursor(10, 20)
    guard.note_write(cursor.position())

    cursor.x = 400  # the user's real mouse
    assert guard.detect(cursor, now_ms=1000)
    assert guard.tripped


def test_guard_ignores_sub_pixel_drift():
    guard = InputGuard(tolerance=2.0)
    cursor = FakeCursor(10, 20)
    guard.note_write(cursor.position())

    cursor.x = 10.5
    assert not guard.detect(cursor, now_ms=0)


def test_guard_resumes_only_after_the_grace_period():
    guard = InputGuard()
    cursor = FakeCursor(10, 20)
    guard.detect(cursor, now_ms=0)  # first call only anchors the known position

    cursor.x = 400  # the user's real mouse
    assert guard.detect(cursor, now_ms=1000)
    assert guard.tripped

    assert not guard.should_resume(now_ms=1500, grace_ms=1500)
    assert guard.should_resume(now_ms=2600, grace_ms=1500)

    guard.reanchor(cursor)
    assert not guard.tripped
    assert guard.should_resume(now_ms=2600, grace_ms=1500)


def test_guard_learns_the_starting_position():
    guard = InputGuard()
    cursor = FakeCursor(77, 88)
    assert not guard.detect(cursor, now_ms=0)
    assert not guard.detect(cursor, now_ms=33)