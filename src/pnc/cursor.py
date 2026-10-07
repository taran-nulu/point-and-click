"""Driving the real macOS cursor with Quartz CGEvent.

macOS has no public "put the cursor here" API. You post synthetic mouse events
to the HID event queue instead, which requires Accessibility permission. We use
``CGEventCreateMouseEvent`` + ``CGEventPost`` rather than
``CGWarpMouseCursorPosition`` because warping teleports the cursor without
generating a move event, so applications that listen for hover and drag events
would never see it.

Note ``CGPostMouseEvent`` is a *different*, legacy 4-argument function that
pyobjc also exports under that name. The modern one we want is
``CGEventPost(kCGHIDEventTap, event)``.

All coordinates are in **points**, not pixels, which matches ``CGDisplayBounds``
on Retina displays - so there is no scaling factor to get wrong.
"""

from __future__ import annotations

import ctypes

import Quartz

_APPLICATION_SERVICES = (
    "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
)


def is_accessibility_trusted() -> bool:
    """Whether we may post synthetic events.

    ``AXIsProcessTrusted`` lives in ApplicationServices, which pyobjc's Quartz
    wrapper does not re-export, so we bind it directly via ctypes rather than
    pulling in another pyobjc framework package.
    """
    try:
        fn = ctypes.cdll.LoadLibrary(_APPLICATION_SERVICES).AXIsProcessTrusted
    except OSError:
        return False
    fn.restype = ctypes.c_bool
    fn.argtypes = []
    return bool(fn())


def list_displays() -> list[tuple[float, float, float, float]]:
    """Every active display as ``(origin_x, origin_y, width, height)`` in points."""
    err, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
    if err != 0:
        return []

    out: list[tuple[float, float, float, float]] = []
    for display_id in ids[:count]:
        rect = Quartz.CGDisplayBounds(display_id)
        out.append((rect.origin.x, rect.origin.y, rect.size.width, rect.size.height))
    return out


class Cursor:
    """A single target display, plus press/release bookkeeping."""

    def __init__(self, display_index: int = 0) -> None:
        if display_index <= 0:
            # The main display is queryable directly, so do not depend on the
            # active-display list: it can come back empty in a session that has
            # no GUI window server attached, while the main display is still
            # perfectly usable.
            rect = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            self.bounds = (rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
        else:
            displays = list_displays()
            if not displays:
                raise RuntimeError(
                    "No active displays found, so display "
                    f"{display_index} cannot be selected. Use --display 0 for the main one."
                )
            if display_index > len(displays):
                raise RuntimeError(
                    f"Display {display_index} requested, but only {len(displays)} active."
                )
            self.bounds = displays[display_index - 1]

        self.x0, self.y0, self.width, self.height = self.bounds
        self._last_written: tuple[float, float] | None = None
        self._button_down = False

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, self.x0), self.x0 + self.width - 1.0),
            min(max(y, self.y0), self.y0 + self.height - 1.0),
        )

    def _post(self, event) -> None:
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def move(self, x: float, y: float) -> tuple[float, float]:
        x, y = self.clamp(x, y)
        event = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved, Quartz.CGPointMake(x, y), Quartz.kCGMouseButtonLeft
        )
        self._post(event)
        self._last_written = (x, y)
        return self._last_written

    def position(self) -> tuple[float, float]:
        """Where the cursor actually is right now."""
        point = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return (point.x, point.y)

    def _post_button(self, event_type: int) -> None:
        x, y = self.position()
        event = Quartz.CGEventCreateMouseEvent(
            None, event_type, Quartz.CGPointMake(x, y), Quartz.kCGMouseButtonLeft
        )
        self._post(event)

    def press(self) -> None:
        if self._button_down:
            return
        self._post_button(Quartz.kCGEventLeftMouseDown)
        self._button_down = True

    def release(self) -> None:
        if not self._button_down:
            return
        self._post_button(Quartz.kCGEventLeftMouseUp)
        self._button_down = False

    @property
    def button_down(self) -> bool:
        return self._button_down

    @property
    def last_written(self) -> tuple[float, float] | None:
        return self._last_written