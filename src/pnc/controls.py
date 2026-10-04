"""Letting the real mouse take over.

While this tool is running it owns the cursor, which would make the physical
mouse useless. The fix is to notice when you use it and get out of the way.

The obvious implementation is a ``CGEventTap`` that intercepts mouse events.
We poll instead: we already write the cursor position every frame, so if the
live cursor position is somewhere other than where we last put it, a human
moved it. Polling needs no special permission at all (unlike an event tap),
runs at the frame rate we are already at, and cannot be defeated by input
routing. At 30-60 Hz the difference is imperceptible.
"""

from __future__ import annotations

import math


class InputGuard:
    """Trips when the real mouse is moved; the app then pauses."""

    def __init__(self, tolerance: float = 2.0) -> None:
        self.tolerance = tolerance
        self._written: tuple[float, float] | None = None
        self._tripped_at_ms: int | None = None

    def note_write(self, position: tuple[float, float]) -> None:
        """Record the position we just put the cursor at."""
        self._written = position

    def detect(self, cursor, now_ms: int) -> bool:
        """True if the cursor moved and it was not us who moved it."""
        live = cursor.position()
        if self._written is None:
            self._written = live
            return False

        if math.dist(live, self._written) > self.tolerance:
            self._written = live
            self._tripped_at_ms = now_ms
            return True
        return False

    @property
    def tripped(self) -> bool:
        return self._tripped_at_ms is not None

    def trip(self, now_ms: int) -> None:
        """Force a pause (the manual pause key)."""
        self._tripped_at_ms = now_ms

    def should_resume(self, now_ms: int, grace_ms: int) -> bool:
        if self._tripped_at_ms is None:
            return True
        return (now_ms - self._tripped_at_ms) >= grace_ms

    def reanchor(self, cursor) -> None:
        """Resume trusting positions, starting from where the cursor is now."""
        self._written = cursor.position()
        self._tripped_at_ms = None

    def reset(self) -> None:
        self._written = None
        self._tripped_at_ms = None