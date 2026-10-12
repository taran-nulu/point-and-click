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


def pointing_hand(
    dx=1.0,
    dy=0.0,
    wrist=(0.5, 0.5),
    middle_folded=False,
    pinched=False,
    scale=SCALE,
):
    """A hand pointing in direction (dx, dy).

    Pinching moves the index tip *back towards the wrist*, which is what really
    happens: the index curls to meet the thumb, and since the index tip is what
    aims the cursor, an unlatched pinch would drag the cursor off target.
    """
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm

    pts = [wrist] * 21
    pts[L.WRIST] = wrist
    pts[L.MIDDLE_MCP] = (wrist[0], wrist[1] - scale)

    index_reach = 0.7 if pinched else 1.1
    index_tip = (wrist[0] + index_reach * scale * ux, wrist[1] + index_reach * scale * uy)
    pts[L.INDEX_TIP] = index_tip

    # The middle finger points along the same direction as the index.
    pts[L.MIDDLE_PIP] = (wrist[0] + 1.2 * scale * ux, wrist[1] + 1.2 * scale * uy)
    pts[L.MIDDLE_TIP] = (
        wrist[0] + (0.35 if middle_folded else 2.2) * scale * ux,
        wrist[1] + (0.35 if middle_folded else 2.2) * scale * uy,
    )

    # The thumb sits off to the side at rest, and meets the index to pinch.
    pts[L.THUMB_MCP] = (wrist[0] - 0.6 * scale * ux, wrist[1] - 0.6 * scale * uy)
    if pinched:
        pts[L.THUMB_TIP] = (index_tip[0] + 0.15 * scale * ux, index_tip[1] + 0.15 * scale * uy)
    else:
        pts[L.THUMB_TIP] = (wrist[0] - 1.0 * scale * ux, wrist[1] - 1.0 * scale * uy)
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


@pytest.fixture
def app_pinch(monkeypatch):
    """An App wired to a FakeCursor, configured for pinch clicking."""
    from pnc import app as app_module

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    instance = app_module.App(Config(click_method="pinch"))
    instance.preview = False
    instance._aspect = FRAME_ASPECT
    instance._refresh_projector(FRAME_ASPECT)
    return instance


def as_hand(points, handedness="Right"):
    """Wrap raw landmarks in the Hand object the app now expects."""
    from pnc.hands import Hand

    return Hand(landmarks=points, handedness=handedness, score=0.99)


def feed(app, points, now_ms: float, dt: float | None = None):
    """One frame with a single (pointing) hand."""
    hand = as_hand(points) if points is not None else None
    app._step(hand, None, now_ms, dt, None, None, None)


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


# ---------------------------------------------------------------------- pinch


def test_pinch_gap_is_small_when_pinched_and_large_when_open():
    from pnc.gestures import pinch_gap

    closed = pinch_gap(pointing_hand(pinched=True))
    open_hand = pinch_gap(pointing_hand(pinched=False))

    assert closed < 0.35
    assert open_hand > 0.5


def test_pinch_gap_is_unchanged_by_hand_depth():
    """Normalising by hand size must make the gap depth-invariant."""
    from pnc.gestures import pinch_gap

    near = pinch_gap(pointing_hand(pinched=True, scale=SCALE))
    far = pinch_gap(pointing_hand(pinched=True, scale=SCALE / 2))
    assert near == pytest.approx(far)


def test_pinch_clicks(app_pinch):
    settled(app_pinch, pointing_hand(pinched=True))
    assert ("press",) in app_pinch.cursor.events
    assert app_pinch.cursor.button_down


def test_releasing_the_pinch_releases_the_button(app_pinch):
    now = settled(app_pinch, pointing_hand(pinched=True))
    assert app_pinch.cursor.button_down

    feed(app_pinch, pointing_hand(pinched=False), now, dt=1 / 30)
    feed(app_pinch, pointing_hand(pinched=False), now + 33, dt=1 / 30)
    assert ("release",) in app_pinch.cursor.events
    assert not app_pinch.cursor.button_down


def test_open_hand_never_clicks_with_pinch(app_pinch):
    settled(app_pinch, pointing_hand(pinched=False))
    assert ("press",) not in app_pinch.cursor.events


def test_folded_middle_finger_does_not_click_in_pinch_mode(app_pinch):
    """The two methods must not fire on each other's gesture."""
    settled(app_pinch, pointing_hand(middle_folded=True, pinched=False))
    assert ("press",) not in app_pinch.cursor.events


def test_pinch_freezes_the_cursor_on_press(app_pinch):
    """Once a pinch registers, the cursor must stop responding entirely.

    Pinching curls the index - the finger doing the aiming - so without latching
    the cursor slides under the click and can land on whatever was next to your
    target. The latch is taken when the press fires, so the drift from the
    fingers *closing* is unavoidable; what must never happen is the cursor
    continuing to move for the rest of the click.
    """
    assert app_pinch.click_method.freeze_on_press is True
    app_pinch.fsm.max_hold_ms = 60_000  # isolate this from the hold timeout

    # Aim somewhere with the hand open.
    settled(app_pinch, pointing_hand(dx=1, pinched=False))
    aimed = app_pinch.cursor.position()

    # Now pinch. The index tip retracts, so without latching the cursor moves.
    settled(app_pinch, pointing_hand(dx=1, pinched=True), start_ms=5000)
    assert app_pinch.cursor.button_down
    latched = app_pinch.cursor.position()
    assert latched != pytest.approx(aimed)  # the pinch really did move it

    # And it stays pinned for the whole hold.
    settled(app_pinch, pointing_hand(dx=-1, pinched=True), start_ms=9000)
    settled(app_pinch, pointing_hand(dx=0, dy=1, pinched=True), start_ms=13_000)
    assert app_pinch.cursor.button_down
    assert app_pinch.cursor.position() == pytest.approx(latched)

    # Releasing hands control straight back.
    feed(app_pinch, pointing_hand(dx=1, pinched=False), 20_000, dt=1 / 30)
    feed(app_pinch, pointing_hand(dx=1, pinched=False), 20_100, dt=1 / 30)
    assert not app_pinch.cursor.button_down


def test_pinch_drift_before_the_press_is_bounded(app_pinch):
    """The latch cannot help with drift that happens before the press fires."""
    app_pinch.fsm.max_hold_ms = 60_000

    settled(app_pinch, pointing_hand(dx=1, pinched=False))
    aimed = app_pinch.cursor.position()

    settled(app_pinch, pointing_hand(dx=1, pinched=True), start_ms=5000)
    assert app_pinch.cursor.button_down

    # The pinch itself nudges the cursor before the press fires. That is finite
    # and small - not a runaway - and it stops dead once pressed.
    drift = math.dist(app_pinch.cursor.position(), aimed)
    assert drift < SCREEN_W * 0.25


def test_middle_finger_method_does_not_freeze(app):
    """Freezing would break click-drag, which the middle finger supports."""
    assert app.click_method.freeze_on_press is False
    app.fsm.max_hold_ms = 60_000  # isolate this from the hold timeout

    settled(app, pointing_hand(dx=1, middle_folded=True))
    assert app.cursor.button_down
    before = app.cursor.position()

    settled(app, pointing_hand(dx=-1, middle_folded=True), start_ms=5000)
    assert app.cursor.button_down  # still held
    assert math.dist(app.cursor.position(), before) > 50  # and it moved: drag works


def test_click_method_flag_selects_the_method():
    from pnc import app as app_module
    from pnc.gestures import PINCH, MIDDLE_FINGER, build_click_method

    cfg = Config()
    assert build_click_method(MIDDLE_FINGER, cfg).name == MIDDLE_FINGER
    assert build_click_method(PINCH, cfg).name == PINCH
    assert app_module is not None

    with pytest.raises(ValueError):
        build_click_method("double-tap", cfg)


def test_cycling_click_methods_rebuilds_the_state_machine(app):
    from pnc.gestures import CLICK_METHOD_NAMES

    start = app.click_method.name
    app._cycle_click_method()
    assert app.click_method.name != start
    assert app.fsm.press_threshold == app.click_method.press_threshold
    assert app.click_method.name in CLICK_METHOD_NAMES

    # A full cycle returns to where it started, whatever the method count.
    for _ in range(len(CLICK_METHOD_NAMES) - 1):
        app._cycle_click_method()
    assert app.click_method.name == start


def test_switching_method_while_pressed_releases_the_button(app):
    """Never leave a button down just because the gesture changed."""
    settled(app, pointing_hand(dx=1, middle_folded=True))
    assert app.cursor.button_down

    app._cycle_click_method()
    assert not app.cursor.button_down


# ----------------------------------------------------------------- elbow pivot


@pytest.fixture
def elbow_app(monkeypatch):
    """An App pivoting from the elbow."""
    from pnc import app as app_module

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    instance = app_module.App(Config(pivot="elbow"))
    instance.preview = False
    instance._aspect = FRAME_ASPECT
    instance._refresh_projector(FRAME_ASPECT)
    instance.fsm.max_hold_ms = 60_000
    return instance


def feed_elbow(app, points, elbow, now_ms: float, dt: float | None = None):
    hand = as_hand(points) if points is not None else None
    app._step(hand, None, now_ms, dt, None, None, None, elbow)


def settle_elbow(app, points, elbow, frames=40, start_ms=0.0):
    now = start_ms
    for _ in range(frames):
        feed_elbow(app, points, elbow, now, dt=1 / 30)
        now += 33.3
    return now


def test_elbow_pivot_uses_a_smaller_gain_than_the_wrist(monkeypatch):
    """The two pivots need different gains because their levers differ."""
    from pnc import app as app_module

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    instance = app_module.App(Config(pivot="elbow"))
    assert instance.gain == pytest.approx(instance.cfg.aim.gain_elbow)
    assert instance.gain < instance.cfg.aim.gain


def test_default_pivot_is_the_wrist(app):
    assert app.pivot == "wrist"
    assert app.effective_pivot == "wrist"
    assert app.gain == pytest.approx(app.cfg.aim.gain)


def test_elbow_pivot_is_announced_only_once_it_is_usable(elbow_app):
    """Before the elbow is seen we are really using the wrist."""
    assert elbow_app.effective_pivot == "wrist"

    settle_elbow(elbow_app, pointing_hand(dx=1), elbow=(0.5, 0.73))
    assert elbow_app.effective_pivot == "elbow"


def test_missing_elbow_falls_back_to_the_wrist(elbow_app):
    settle_elbow(elbow_app, pointing_hand(dx=1), elbow=None)
    assert elbow_app.effective_pivot == "wrist"


def test_a_single_dropped_frame_does_not_switch_pivot(elbow_app):
    """One occluded frame must not teleport the cursor."""
    settle_elbow(elbow_app, pointing_hand(dx=1), elbow=(0.5, 0.73))
    assert elbow_app.effective_pivot == "elbow"

    feed_elbow(elbow_app, pointing_hand(dx=1), None, 5000.0, dt=1 / 30)
    assert elbow_app.effective_pivot == "elbow"  # still holding the elbow


def test_sustained_loss_switches_back_after_a_few_frames(elbow_app):
    from pnc.app import PIVOT_DROPOUT_FRAMES

    settle_elbow(elbow_app, pointing_hand(dx=1), elbow=(0.5, 0.73))
    assert elbow_app.effective_pivot == "elbow"

    now = 5000.0
    for _ in range(PIVOT_DROPOUT_FRAMES):
        feed_elbow(elbow_app, pointing_hand(dx=1), None, now, dt=1 / 30)
        now += 33.3
    assert elbow_app.effective_pivot == "wrist"


def test_pivot_is_frozen_while_a_button_is_held(elbow_app):
    """Changing the anchor mid-drag would yank the cursor under the user."""
    settle_elbow(elbow_app, pointing_hand(dx=1, middle_folded=True), elbow=(0.5, 0.73))
    assert elbow_app.cursor.button_down
    assert elbow_app.effective_pivot == "elbow"

    # Elbow disappears for many frames mid-hold: still elbow, still no teleport.
    for offset in range(10):
        feed_elbow(
            elbow_app,
            pointing_hand(dx=1, middle_folded=True),
            None,
            5000.0 + offset * 33.3,
            dt=1 / 30,
        )
        assert elbow_app.effective_pivot == "elbow"
    assert elbow_app.cursor.button_down


def test_losing_the_elbow_resets_the_smoother(elbow_app):
    """Otherwise the filter smears the jump across several frames and the
    cursor visibly trails behind your hand."""
    from pnc.app import PIVOT_DROPOUT_FRAMES

    settle_elbow(elbow_app, pointing_hand(dx=1), elbow=(0.5, 0.73))
    assert elbow_app.effective_pivot == "elbow"

    elbow_app.smoother._filters["x"]._last = 12345.0  # pretend it has history
    elbow_app._elbow_miss_run = PIVOT_DROPOUT_FRAMES - 1

    assert elbow_app._update_pivot(None, 5000.0) is None
    assert elbow_app.effective_pivot == "wrist"
    assert elbow_app.smoother._filters["x"]._last is None


def test_the_first_frame_after_a_dropout_is_not_dragged_by_filter_history(elbow_app):
    """The user-visible consequence: the cursor must not sweep across the screen
    on the frame the pivot changes."""
    from pnc.app import PIVOT_DROPOUT_FRAMES

    settle_elbow(elbow_app, pointing_hand(dx=1), elbow=(0.5, 0.73))
    settle_elbow(elbow_app, pointing_hand(dx=-1), None, start_ms=5000)
    # Poison the filter to stand in for a long pre-dropout history.
    elbow_app.smoother._filters["x"]._last = 12345.0
    elbow_app._elbow_miss_run = PIVOT_DROPOUT_FRAMES - 1

    before = elbow_app.cursor.position()
    feed_elbow(elbow_app, pointing_hand(dx=-1), None, 20_000, dt=1 / 30)
    jump = math.dist(elbow_app.cursor.position(), before)

    # A single frame moves at most a step or two, not a screenful.
    assert jump < SCREEN_W * 0.25


def test_cycling_to_elbow_picks_up_the_elbow_gain(monkeypatch):
    from pnc import app as app_module

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    instance = app_module.App(Config())
    instance.preview = False
    instance._aspect = FRAME_ASPECT
    instance._refresh_projector(FRAME_ASPECT)

    assert instance.pivot == "wrist"
    wrist_gain = instance.gain

    instance._cycle_pivot()
    assert instance.pivot == "elbow"
    assert instance.gain == pytest.approx(instance.cfg.aim.gain_elbow)
    assert instance.gain < wrist_gain

    instance._cycle_pivot()
    assert instance.pivot == "wrist"
    assert instance.gain == pytest.approx(wrist_gain)


def test_an_explicit_gain_survives_a_pivot_change(monkeypatch):
    """--gain is a deliberate override, so switching pivot must not clobber it."""
    from pnc import app as app_module

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    instance = app_module.App(Config(), gain=0.42)
    instance.preview = False
    instance._aspect = FRAME_ASPECT
    instance._refresh_projector(FRAME_ASPECT)

    instance._cycle_pivot()
    assert instance.pivot == "elbow"
    assert instance.gain == pytest.approx(0.42)


def test_cycling_pivot_while_pressed_releases_the_button(monkeypatch):
    from pnc import app as app_module

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    instance = app_module.App(Config())
    instance.preview = False
    instance._aspect = FRAME_ASPECT
    instance._refresh_projector(FRAME_ASPECT)
    instance.fsm.max_hold_ms = 60_000

    settled(instance, pointing_hand(dx=1, middle_folded=True))
    assert instance.cursor.button_down

    instance._cycle_pivot()
    assert not instance.cursor.button_down


def test_unknown_pivot_is_rejected(monkeypatch):
    from pnc import app as app_module

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    with pytest.raises(ValueError):
        app_module.App(Config(pivot="nose"))