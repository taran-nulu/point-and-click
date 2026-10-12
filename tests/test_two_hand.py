"""Tests for two-handed clicking: point with one hand, click with a fist in
the other.

The failure this guards against is a role mix-up - reading the click from the
hand that aims - because that would make the tool feel broken in a way that is
hard to diagnose from the outside. The most important test here is therefore
``test_two_hand_reads_the_other_hand``.
"""

from __future__ import annotations

import pytest

from pnc import gestures, landmarks as L
from pnc.config import Config
from pnc.hands import Hand
from pnc.roles import RoleResolver

SCALE = 0.05
WRIST = (0.5, 0.5)


def pointing_hand_points(wrist=WRIST, direction=1.0):
    """Index extended, other three relaxed: the aiming pose.

    The other fingers being relaxed matters - that asymmetry is precisely what
    ``pointer_score`` measures to tell a pointing hand from a fist.
    """
    pts = [wrist] * 21
    pts[L.WRIST] = wrist
    pts[L.MIDDLE_MCP] = (wrist[0], wrist[1] - SCALE)

    def finger(pip, tip, pip_reach, tip_reach):
        pts[pip] = (wrist[0] + pip_reach * SCALE * direction, wrist[1])
        pts[tip] = (wrist[0] + tip_reach * SCALE * direction, wrist[1])

    finger(L.INDEX_PIP, L.INDEX_TIP, 1.2, 1.9)  # pointing
    for pip, tip in gestures.FIST_FINGERS[1:]:
        finger(pip, tip, 1.2, 1.0)  # relaxed

    pts[L.THUMB_MCP] = (wrist[0] - 0.4 * SCALE, wrist[1])
    pts[L.THUMB_TIP] = (wrist[0] - 0.8 * SCALE, wrist[1])
    return pts


def open_hand_points(wrist=WRIST, direction=1.0):
    """All four fingers extended, as opposed to a deliberate fist."""
    pts = [wrist] * 21
    pts[L.WRIST] = wrist
    pts[L.MIDDLE_MCP] = (wrist[0], wrist[1] - SCALE)
    for pip, tip in gestures.FIST_FINGERS:
        pts[pip] = (wrist[0] + 1.2 * SCALE * direction, wrist[1])
        pts[tip] = (wrist[0] + 1.9 * SCALE * direction, wrist[1])
    return pts


def fist_points(wrist=(0.5, 0.75)):
    """All four fingers curled back towards the wrist."""
    pts = [wrist] * 21
    pts[L.WRIST] = wrist
    pts[L.MIDDLE_MCP] = (wrist[0], wrist[1] - SCALE)
    for pip, tip in gestures.FIST_FINGERS:
        pts[pip] = (wrist[0], wrist[1] - 1.2 * SCALE)
        pts[tip] = (wrist[0], wrist[1] - 0.3 * SCALE)
    return pts


def hand(points, handedness="Right", score=0.99):
    return Hand(landmarks=points, handedness=handedness, score=score)


# ------------------------------------------------------------ fist_openness


def test_fist_openness_separates_open_and_closed_widely():
    assert gestures.fist_openness(open_hand_points()) > gestures.fist_openness(fist_points())
    assert (
        gestures.fist_openness(open_hand_points()) - gestures.fist_openness(fist_points())
    ) > 0.8  # a decisive gap, not a marginal one


def test_fist_openness_lands_near_the_documented_ranges():
    """The config defaults only work if these hold, so pin them down.

    Measured on this synthetic geometry: deliberate fist ~0.25, loosely relaxed
    hand ~1.0, flat open hand ~1.58. The defaults (0.75 / 1.30) sit between
    them, which is the whole reason a fist reads as a fist and an open hand
    releases.
    """
    assert gestures.fist_openness(fist_points()) == pytest.approx(0.25, abs=0.2)
    assert gestures.fist_openness(pointing_hand_points()) == pytest.approx(1.0, abs=0.2)
    assert gestures.fist_openness(open_hand_points()) == pytest.approx(1.58, abs=0.2)


def test_the_shipped_config_and_the_dataclass_defaults_agree():
    """Guards a bug that actually happened.

    config/default.toml is overlaid on the dataclass defaults at load time, so
    when the two disagree the TOML silently wins and the tests - which build a
    bare Config() - validate thresholds that never run. That is exactly what
    happened once already with the fist thresholds.
    """
    from pnc.config import REPO_ROOT, load_config

    bare = Config()  # dataclass defaults, no TOML
    loaded = load_config(REPO_ROOT / "config" / "default.toml")

    for section, field in (
        ("two_hand", "fist_ratio"),
        ("two_hand", "open_ratio"),
        ("two_hand", "pointing_hand"),
        ("two_hand", "max_hold_ms"),
        ("pinch", "press_ratio"),
        ("pinch", "release_ratio"),
        ("gesture", "fold_ratio"),
        ("gesture", "extend_ratio"),
        ("aim", "gain"),
        ("aim", "gain_elbow"),
        ("click_method", None),
        ("pivot", None),
    ):
        expected = getattr(getattr(bare, section), field) if field else getattr(bare, section)
        actual = getattr(getattr(loaded, section), field) if field else getattr(loaded, section)
        assert actual == expected, f"{section}.{field or ''}: toml={actual!r} defaults={expected!r}"


def test_the_default_thresholds_separate_every_pose():
    """The regression that matters: a pointing hand must NOT read as a fist.

    A pointing hand scores ~1.02 openness because its relaxed fingers pull the
    mean down. Earlier defaults of 1.5/1.8 would have classified it as a fist.
    And a relaxed hand must read as *open*, so it releases - otherwise relaxing
    after a click leaves the button stuck down.
    """
    from pnc.config import Config

    cfg = Config()
    assert gestures.fist_openness(fist_points()) <= cfg.two_hand.fist_ratio
    assert gestures.fist_openness(pointing_hand_points()) > cfg.two_hand.fist_ratio
    assert gestures.fist_openness(pointing_hand_points()) >= cfg.two_hand.open_ratio
    assert gestures.fist_openness(open_hand_points()) >= cfg.two_hand.open_ratio


def test_fist_openness_ignores_the_thumb():
    """The thumb barely moves when a hand closes, so it must not dilute the
    signal."""
    with_thumb = fist_points()
    without_thumb = fist_points()
    assert with_thumb == without_thumb  # both have the thumb at the wrist
    assert gestures.fist_openness(without_thumb) is not None


def test_pointer_score_separates_a_pointing_hand_from_a_fist():
    """The margin is what lets the roles be told apart automatically."""
    pointing = gestures.pointer_score(pointing_hand_points())
    fist = gestures.pointer_score(fist_points())

    assert pointing > 0.5
    assert pointing > fist + 0.5


def test_a_fist_scores_about_zero_because_all_fingers_curl_equally():
    """Worth pinning down: the score measures index-vs-the-rest asymmetry, so a
    fist is not strongly negative, it is simply not-asymmetric-as-a-pointer."""
    assert gestures.pointer_score(fist_points()) == pytest.approx(0.0, abs=0.05)


def test_fist_openness_rejects_an_unmeasurable_hand():
    degenerate = [(0.0, 0.0)] * 21
    assert gestures.fist_openness(degenerate) is None


# --------------------------------------------------------------- two-hand FSM


def two_hand_app(monkeypatch):
    from pnc import app as app_module
    from tests.test_pipeline import FakeCursor

    monkeypatch.setattr(app_module, "Cursor", FakeCursor)
    instance = app_module.App(Config(click_method="two-hand"))
    instance.preview = False
    instance._aspect = 640 / 480
    instance._refresh_projector(instance._aspect)
    return instance


def feed_hands(app, pointing, clicking, now_ms, dt=None):
    app._step(
        as_hand(pointing) if pointing is not None else None,
        as_hand(clicking) if clicking is not None else None,
        now_ms, dt, None, None, None,
    )


def as_hand(points, handedness="Right"):
    return hand(points, handedness)


def settle_hands(app, pointing, clicking, frames=40, start_ms=0.0):
    now = start_ms
    for _ in range(frames):
        feed_hands(app, pointing, clicking, now, dt=1 / 30)
        now += 33.3
    return now


def test_two_hand_reads_the_other_hand(monkeypatch):
    """The wiring assertion that matters most: the click must come from the
    second hand, never the one doing the aiming."""
    app = two_hand_app(monkeypatch)
    assert app.click_method.hand_role == gestures.ROLE_CLICKING

    # Fist the click hand; the pointing hand stays wide open.
    settle_hands(app, pointing_hand_points(), fist_points())
    assert app.cursor.button_down

    # Now relax the click hand. The pointing hand is unchanged throughout, so
    # only the second hand can have driven the release.
    settle_hands(app, pointing_hand_points(), pointing_hand_points(), start_ms=5000)
    assert not app.cursor.button_down


def test_open_click_hand_never_clicks(monkeypatch):
    app = two_hand_app(monkeypatch)
    settle_hands(app, pointing_hand_points(), pointing_hand_points())
    assert not app.cursor.button_down


def test_single_hand_methods_read_the_pointing_hand():
    """Regression: the existing methods must not be redirected."""
    cfg = Config()
    for name in (gestures.MIDDLE_FINGER, gestures.PINCH):
        assert gestures.build_click_method(name, cfg).hand_role == gestures.ROLE_POINTING


def test_a_fist_holds_for_ten_seconds(monkeypatch):
    """A long hold is the point of this gesture, so the timeout is disabled."""
    app = two_hand_app(monkeypatch)
    assert app.fsm.max_hold_ms == 0

    settle_hands(app, pointing_hand_points(), fist_points())
    assert app.cursor.button_down

    # Ten seconds of held fist: 10_000 ms, twenty times the normal 2 s timeout.
    now = settle_hands(app, pointing_hand_points(), fist_points(), start_ms=10_000)
    assert app.cursor.button_down
    assert app.fsm.auto_released is False

    settled_more = settle_hands(
        app, pointing_hand_points(), fist_points(), start_ms=now
    )
    assert app.cursor.button_down
    assert settled_more > now


def test_opening_the_fist_releases(monkeypatch):
    app = two_hand_app(monkeypatch)
    now = settle_hands(app, pointing_hand_points(), fist_points())
    assert app.cursor.button_down

    settle_hands(app, pointing_hand_points(), pointing_hand_points(), start_ms=now)
    assert not app.cursor.button_down


def test_losing_the_click_hand_releases_after_the_grace(monkeypatch):
    app = two_hand_app(monkeypatch)
    grace = app.cfg.safety.hand_lost_release_ms

    now = settle_hands(app, pointing_hand_points(), fist_points())
    assert app.cursor.button_down

    # Click hand vanishes. The grace timer starts, and a single dropped
    # detection frame inside it must not drop a drag mid-gesture.
    feed_hands(app, pointing_hand_points(), None, now, dt=1 / 30)
    assert app.cursor.button_down

    feed_hands(app, pointing_hand_points(), None, now + grace * 0.5, dt=1 / 30)
    assert app.cursor.button_down

    feed_hands(app, pointing_hand_points(), None, now + grace + 50, dt=1 / 30)
    assert not app.cursor.button_down


def test_losing_the_pointing_hand_keeps_the_button_down(monkeypatch):
    """You are mid-drag, not finished, when your aiming hand leaves frame."""
    app = two_hand_app(monkeypatch)
    now = settle_hands(app, pointing_hand_points(), fist_points())
    assert app.cursor.button_down

    settle_hands(app, None, fist_points(), start_ms=now + 1000)
    assert app.cursor.button_down


def test_two_hand_supports_drag(monkeypatch):
    """The click hand is a different hand, so aiming is undisturbed by it."""
    app = two_hand_app(monkeypatch)
    now = settle_hands(app, pointing_hand_points(direction=1), fist_points())
    assert app.cursor.button_down
    before = app.cursor.position()

    settle_hands(app, pointing_hand_points(direction=-1), fist_points(), start_ms=now)
    assert app.cursor.button_down
    assert abs(app.cursor.position()[0] - before[0]) > 50


# ------------------------------------------------------------ role resolution


def test_auto_picks_the_hand_with_the_index_out():
    resolver = RoleResolver("auto")
    pointing = hand(pointing_hand_points(), "Right")
    clicking = hand(fist_points(), "Left")

    got_pointing, got_clicking = resolver.resolve([pointing, clicking])
    assert got_pointing is pointing
    assert got_clicking is clicking
    assert resolver.describe().startswith("Right")


def test_auto_is_sticky_across_identical_frames(monkeypatch):
    """Without stickiness the roles would flap and the cursor would jump
    between hands."""
    resolver = RoleResolver("auto")
    right = hand(pointing_hand_points(), "Right")
    left = hand(fist_points(), "Left")

    first, _ = resolver.resolve([right, left])
    assert first is right

    # Even when the gestures become ambiguous, the assignment holds.
    for _ in range(30):
        got, _ = resolver.resolve([right, left])
        assert got is right


def test_auto_switches_when_the_assigned_hand_leaves(monkeypatch):
    resolver = RoleResolver("auto")
    right = hand(pointing_hand_points(), "Right")
    left = hand(fist_points(), "Left")
    resolver.resolve([right, left])

    # Right hand gone: the only remaining hand must take over as the pointer,
    # rather than leaving the tool dead.
    got, other = resolver.resolve([left])
    assert got is left
    assert other is None


def test_a_pin_overrides_the_gesture():
    resolver = RoleResolver("left")
    right_pointing = hand(pointing_hand_points(), "Right")
    left_fist = hand(fist_points(), "Left")

    pointing, clicking = resolver.resolve([right_pointing, left_fist])
    assert pointing is left_fist
    assert clicking is right_pointing
    assert "pinned" in resolver.describe()


def test_a_pinned_hand_that_is_missing_does_not_fall_back(monkeypatch):
    """Silently aiming with the click hand would be chaos."""
    resolver = RoleResolver("left")
    right_only = hand(pointing_hand_points(), "Right")
    pointing, clicking = resolver.resolve([right_only])
    assert pointing is None
    assert clicking is None


def test_no_hands_resolves_to_nothing():
    resolver = RoleResolver("auto")
    assert resolver.resolve([]) == (None, None)


def test_ambiguous_first_frame_still_assigns_something():
    """Two identical hands: pick one and stick to it, rather than deadlock."""
    resolver = RoleResolver("auto")
    a = hand(fist_points(), "Left")
    b = hand(fist_points(), "Right")

    pointing, clicking = resolver.resolve([a, b])
    assert pointing in (a, b)
    assert clicking is not pointing

    for _ in range(10):
        again, _ = resolver.resolve([a, b])
        assert again is pointing


def test_an_invalid_pointing_hand_setting_is_rejected():
    with pytest.raises(ValueError):
        RoleResolver("sideways")


def test_unmeasurable_hand_does_not_win_the_pointing_role():
    """A hand with no usable landmarks scores nothing, so the measurable one
    takes over rather than leaving the tool dead."""
    resolver = RoleResolver("auto")
    junk = hand([(0.0, 0.0)] * 21, "Right")
    good = hand(fist_points(), "Left")

    pointing, _ = resolver.resolve([junk, good])
    assert pointing is good


def test_only_unmeasurable_hands_resolve_to_nothing():
    resolver = RoleResolver("auto")
    junk_a = hand([(0.0, 0.0)] * 21, "Right")
    junk_b = hand([(0.0, 0.0)] * 21, "Left")
    assert resolver.resolve([junk_a, junk_b]) == (None, None)