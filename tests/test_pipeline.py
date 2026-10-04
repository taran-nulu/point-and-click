"""End-to-end tests for the frame loop.

These drive ``App._step`` with synthetic landmarks and a fake cursor, so the
whole chain - aiming, smoothing, mapping, gesture, safety - is exercised
without a camera, a display, or permission prompts.

The landmark geometry below is deliberately realistic: the index fingertip sits
about 1.1 hand-lengths from the wrist, and the middle finger is modelled at
either ~2.2 hand-lengths (extended) or ~0.3 (folded).
"""

from __future__ import annotations

import math

import pytest

from pnc import landmarks as L
from pnc.config import Config, load_config
from pnc.mapper import aim_target, to_screen_space, DefaultProjector

SCREEN_W, SCREEN_H = 1728.0, 1117.0
FRAME_ASPECT = 640 / 480
SCALE = 0.05  # hand size in normalised frame units


class FakeCursor:
    """Records what the app asked macOS to do, instead of doing it."""

    def __init__(self, display_index: int = 0) -> None:
        self.x0, self.y0 = 0.0, 0.0
        self.width, self.height = SCREEN_W, SCREEN_H
        self.bounds = (0.0, 0.0, SCREEN_W, SCREEN_H)
        self.x, self.y = 100.0, 100.0
        self.events: list[tuple] = []
        self._down = False

    def clamp(self, x, y):
        return (
            min(max(x, self.x0), self.x0 + self.width - 1.0),
            min(max(y, self.y0), self.y0 + self.height - 1.0),
        )

    def move(self, x, y):
        x, y = self.clamp(x, y)
        self.x, self.y = x, y
        self.events.append(("move", x, y))
        return (x, y)

    def position(self):
        return (self.x, self.y)

    def press(self):
        self._down = True
        self.events.append(("press",))

    def release(self):
        self._down = False
        self.events.append(("release",))

    @property
    def button_down(self):
        return self._down

    def moves(self):
        return [e for e in self.events if e[0] == "move"]


def pointing_hand(dx=1.0, dy=0.0, wrist=(0.5, 0.5), middle_folded=False, scale=SCALE):
    """A hand pointing in direction (dx, dy), index extended, middle optional."""
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm

    pts = [wrist] * 21
    pts[L.WRIST] = wrist
    pts[L.MIDDLE_MCP] = (wrist[0], wrist[1] - scale)
    pts[L.INDEX_TIP] = (wrist[0] + 1.1 * scale * ux, wrist[1] + 1.1 * scale * uy)

    # The middle finger points along the same direction as the index.
    pts[L.MIDDLE_PIP] = (wrist[0] + 1.2 * scale * ux, wrist[1] + 1.2 * scale * uy)
    pts[L.MIDDLE_TIP] = (
        wrist[0] + (0.35 if middle_folded else 2.2) * scale * ux,
        wrist[1] + (0.35 if middle_folded else 2.2) * scale * uy,
    )
    return pts


@pytest.fixture
def app(monkeypatch):
    """An App wired to a FakeCursor, with preview off."""
    from pnc import app as app_module

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    instance = app_module.App(Config())
    instance.preview = False
    instance._aspect = FRAME_ASPECT
    instance._refresh_projector(FRAME_ASPECT)
    return instance


def feed(app, points, now_ms: float, dt: float | None = None):
    app._step(points, now_ms, dt, None, None, None)


def settled(app, points, frames: int = 40, start_ms: float = 0.0):
    """Feed the same pose for a while so the smoothing filter converges."""
    now = start_ms
    for _ in range(frames):
        feed(app, points, now, dt=1 / 30)
        now += 33.3
    return now


# --------------------------------------------------------------------- aiming


def test_pointing_right_moves_the_cursor_right(app):
    settled(app, pointing_hand(dx=-1))
    left = app.cursor.x
    settled(app, pointing_hand(dx=1), start_ms=5000)
    assert app.cursor.x > left


def test_pointing_down_moves_the_cursor_down(app):
    settled(app, pointing_hand(dx=0, dy=-1))
    top = app.cursor.y
    settled(app, pointing_hand(dx=0, dy=1), start_ms=5000)
    assert app.cursor.y > top


def test_cursor_stays_on_screen_for_every_direction(app):
    """The whole point of a sane gain: no direction may pin to a corner."""
    projector = DefaultProjector(FRAME_ASPECT, SCREEN_W, SCREEN_H)
    gain = app.gain

    for direction in ((1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, -0.7)):
        hand = pointing_hand(*direction)
        u, v = aim_target(hand, gain)
        ax, ay = to_screen_space(u, v, FRAME_ASPECT)
        x, y = projector(ax, ay)
        # Generous but meaningful: each axis should land inside the display.
        assert -SCREEN_W * 0.5 < x < SCREEN_W * 1.5, direction
        assert -SCREEN_H * 0.5 < y < SCREEN_H * 1.5, direction


def test_hand_moving_right_moves_the_cursor_right(app):
    """Wrist translation must count, not just finger direction."""
    settled(app, pointing_hand(dx=1, wrist=(0.35, 0.5)))
    left = app.cursor.x
    settled(app, pointing_hand(dx=1, wrist=(0.65, 0.5)), start_ms=5000)
    assert app.cursor.x > left


def test_leaning_closer_does_not_drift_the_cursor(app):
    """Twice as close: every landmark offset halves, so the target must hold."""
    far = pointing_hand(dx=1)
    near = pointing_hand(dx=1, scale=SCALE / 2)

    settled(app, far)
    a = app.cursor.position()
    settled(app, near, start_ms=5000)
    b = app.cursor.position()

    assert math.dist(a, b) < 1.0


def test_no_hand_produces_no_cursor_movement(app):
    settled(app, pointing_hand(dx=1))
    app.cursor.events.clear()
    settled(app, None, start_ms=5000)
    assert app.cursor.moves() == []


# ------------------------------------------------------------------- clicking


def test_folding_the_middle_finger_clicks(app):
    settled(app, pointing_hand(dx=1, middle_folded=True))
    assert ("press",) in app.cursor.events
    assert app.cursor.button_down


def test_unfolding_releases(app):
    now = settled(app, pointing_hand(dx=1, middle_folded=True))
    feed(app, pointing_hand(dx=1, middle_folded=False), now, dt=1 / 30)
    feed(app, pointing_hand(dx=1, middle_folded=False), now + 33, dt=1 / 30)
    assert ("release",) in app.cursor.events
    assert not app.cursor.button_down


def test_extended_middle_finger_never_clicks(app):
    settled(app, pointing_hand(dx=1, middle_folded=False))
    assert ("press",) not in app.cursor.events


def test_losing_your_hand_releases_the_button(app):
    now = settled(app, pointing_hand(dx=1, middle_folded=True))
    assert app.cursor.button_down

    # Past hand_lost_release_ms with no hand.
    settled(app, None, start_ms=now + 400)
    assert not app.cursor.button_down
    assert ("release",) in app.cursor.events


def test_button_is_never_left_held_after_a_long_hold(app):
    """max_hold_ms is the backstop against a stuck button."""
    app.fsm.max_hold_ms = 300

    # Settle in fewer frames than the timeout allows, so the press is still live.
    now = settled(app, pointing_hand(dx=1, middle_folded=True), frames=6)
    assert app.cursor.button_down

    # Stay folded past the timeout: the button must let go on its own.
    settled(app, pointing_hand(dx=1, middle_folded=True), start_ms=now + 600)
    assert not app.cursor.button_down
    assert ("release",) in app.cursor.events


# ----------------------------------------------------------------- input guard


def test_using_the_real_mouse_pauses_the_tool(app):
    settled(app, pointing_hand(dx=1))

    app.cursor.x = 1500.0  # the human grabbed the mouse
    feed(app, pointing_hand(dx=-1), 5000.0, dt=1 / 30)
    assert app.guard.tripped

    app.cursor.events.clear()
    feed(app, pointing_hand(dx=-1), 5100.0, dt=1 / 30)
    assert app.cursor.moves() == []  # we are not fighting the human


def test_tool_resumes_after_the_grace_period_and_a_stable_hand(app):
    app.guard.trip(0)
    grace = app.cfg.safety.resume_grace_ms

    feed(app, pointing_hand(dx=1), 100.0, dt=1 / 30)
    assert app.guard.tripped  # inside the grace window

    later = grace + 1000.0
    feed(app, pointing_hand(dx=1), later, dt=1 / 30)  # hand spotted
    feed(app, pointing_hand(dx=1), later + 500, dt=1 / 30)  # held steady
    assert not app.guard.tripped

    app.cursor.events.clear()
    feed(app, pointing_hand(dx=1), later + 600, dt=1 / 30)
    assert app.cursor.moves()


def test_config_file_parses_and_overrides_defaults():
    cfg = load_config()
    assert cfg.camera.index == 0
    assert cfg.gesture.fold_ratio < cfg.gesture.extend_ratio  # hysteresis is sane
    assert 0.2 < cfg.aim.gain < 1.0  # a usable default, not a runaway
    assert cfg.preview.enabled is True