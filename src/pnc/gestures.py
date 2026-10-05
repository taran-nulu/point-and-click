"""Gesture recognition: two ways to click.

``middle-finger``
    Fold the middle finger to click, unfold to release. The index finger stays
    free, so aiming never breaks while you click, and holding it down gives a
    click-drag.

``pinch``
    Bring the thumb tip to the index tip to click. More precise than the
    middle finger, and the natural gesture most people try first - but
    pinching curls the index finger, which is the finger doing the aiming, so
    this method latches the cursor position at the moment of the click
    (see ``ClickMethod.freeze_on_press``). That also means it cannot drag.

Both methods share one property that makes the state machine reusable: in each
case, the measured signal *drops* when you press. The middle finger's extension
ratio falls as it folds; the pinch gap ratio falls as the fingers close.

Why the two thresholds differ
-----------------------------
A single threshold makes the click chatter open and shut whenever a finger
lingers near the boundary. MediaPipe landmark noise is easily +-0.05 on these
ratios, which is larger than the gap you would want to use. So the state
machine has a dead band: it must cross ``press_threshold`` to press and
``release_threshold`` to release, and each takes ``confirm_frames`` consecutive
frames to be believed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from . import landmarks as L
from .mapper import hand_scale

if TYPE_CHECKING:
    from .config import Config

PRESS = "press"
RELEASE = "release"

MIDDLE_FINGER = "middle-finger"
PINCH = "pinch"


def middle_extension(pts) -> float | None:
    """How extended the middle finger is, as a wrist-distance ratio.

    An extended middle finger puts the tip far beyond the PIP joint; a folded
    one collapses the tip back towards the wrist. The PIP joint is the
    reference because it barely moves when the finger folds.
    """
    wrist = pts[L.WRIST]
    pip_dist = math.dist(pts[L.MIDDLE_PIP], wrist)
    if pip_dist < 1e-6:
        return None
    return math.dist(pts[L.MIDDLE_TIP], wrist) / pip_dist


def pinch_gap(pts) -> float | None:
    """Distance between the thumb and index tips, in units of hand size.

    Normalising by hand size is what makes this usable: without it the
    threshold would mean something different every time you moved your hand
    towards or away from the camera, exactly like the aiming model. A pinched
    hand gives roughly 0.1-0.2; a hand at rest gives roughly 0.5-0.9.
    """
    scale = hand_scale(pts)
    if scale < 1e-6:
        return None
    return math.dist(pts[L.THUMB_TIP], pts[L.INDEX_TIP]) / scale


def finger_bend_degrees(pts, mcp: int, pip: int, tip: int) -> float | None:
    """Interior angle at the PIP joint. ~180 deg straight, <90 deg curled."""
    a = pts[mcp]
    b = pts[pip]
    c = pts[tip]

    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None

    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


@dataclass(frozen=True)
class ClickMethod:
    """One way to turn a hand pose into a button press.

    Bundles the measurement with its thresholds, so adding a method is a matter
    of defining one of these rather than threading conditionals through the app.
    """

    name: str
    label: str
    measure: Callable[[Sequence], float | None]
    press_threshold: float
    release_threshold: float
    freeze_on_press: bool
    highlight: tuple[int, ...]
    help: str

    def signal(self, points) -> float | None:
        """The raw measurement for this frame, or None if unmeasurable."""
        return self.measure(points)


def build_click_method(name: str, cfg: "Config") -> ClickMethod:
    """Assemble the named click method from configuration."""
    if name == MIDDLE_FINGER:
        return ClickMethod(
            name=MIDDLE_FINGER,
            label="middle finger",
            measure=middle_extension,
            press_threshold=cfg.gesture.fold_ratio,
            release_threshold=cfg.gesture.extend_ratio,
            # Aim uses the index tip, and folding the middle finger only nudges
            # it a little, so the cursor stays live while held. That is what
            # makes click-drag work.
            freeze_on_press=False,
            highlight=(L.MIDDLE_TIP, L.MIDDLE_PIP),
            help="fold your middle finger to click",
        )

    if name == PINCH:
        return ClickMethod(
            name=PINCH,
            label="pinch",
            measure=pinch_gap,
            press_threshold=cfg.pinch.press_ratio,
            release_threshold=cfg.pinch.release_ratio,
            # Pinching curls the index finger - the very finger doing the
            # aiming - so without latching, the cursor slides onto whatever was
            # under it as you closed the pinch.
            freeze_on_press=cfg.pinch.freeze_on_press,
            highlight=(L.THUMB_TIP, L.INDEX_TIP),
            help="pinch thumb to index to click",
        )

    raise ValueError(f"Unknown click method {name!r}. Choose from {sorted(CLICK_METHOD_NAMES)}.")


CLICK_METHOD_NAMES = (MIDDLE_FINGER, PINCH)


class ClickFSM:
    """Debounced press/release state machine with a hold timeout.

    It knows nothing about hands - it is handed a scalar that falls when you
    press, which is why the same state machine serves both click methods.
    """

    def __init__(
        self,
        press_threshold: float = 1.15,
        release_threshold: float = 1.35,
        confirm_frames: int = 2,
        max_hold_ms: int = 2000,
    ) -> None:
        if release_threshold <= press_threshold:
            raise ValueError(
                "release_threshold must exceed press_threshold (hysteresis band), "
                f"got press={press_threshold} release={release_threshold}"
            )

        self.press_threshold = float(press_threshold)
        self.release_threshold = float(release_threshold)
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_hold_ms = int(max_hold_ms)

        self.pressed = False
        self._armed = True
        self._press_run = 0
        self._release_run = 0
        self._pressed_at_ms: int | None = None
        self.auto_released = False

    @classmethod
    def for_method(cls, method: ClickMethod, cfg: "Config") -> "ClickFSM":
        return cls(
            press_threshold=method.press_threshold,
            release_threshold=method.release_threshold,
            confirm_frames=cfg.gesture.confirm_frames,
            max_hold_ms=cfg.gesture.max_hold_ms,
        )

    def _emit_release(self) -> str:
        self.pressed = False
        self._pressed_at_ms = None
        self._press_run = 0
        self._release_run = 0
        return RELEASE

    def update(self, signal: float | None, now_ms: int) -> str | None:
        """Feed this frame's signal. Returns PRESS, RELEASE or None."""
        if signal is None:
            return None

        # Holding timeout: never let a misdetected press leave a button stuck.
        if self.pressed and self._pressed_at_ms is not None:
            if now_ms - self._pressed_at_ms >= self.max_hold_ms:
                self._armed = False
                self.auto_released = True
                return self._emit_release()

        if not self.pressed:
            self.auto_released = False
            if signal <= self.press_threshold:
                self._press_run += 1
                self._release_run = 0
                if self._armed and self._press_run >= self.confirm_frames:
                    self.pressed = True
                    self._armed = False
                    self._press_run = 0
                    self._pressed_at_ms = now_ms
                    return PRESS
            else:
                self._press_run = 0
                if signal >= self.release_threshold:
                    # Re-arm only once the hand is clearly open, so an
                    # auto-release does not immediately re-trigger.
                    self._armed = True
        else:
            self._release_run += 1
            self._press_run = 0
            if signal >= self.release_threshold:
                self._armed = True
                if self._release_run >= self.confirm_frames:
                    return self._emit_release()

        return None

    def force_release(self) -> str | None:
        """Safety valve: hand lost, shutting down, user grabbed the mouse."""
        if self.pressed:
            self._armed = False
            return self._emit_release()
        return None