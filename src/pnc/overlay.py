"""The preview window.

Two dots matter more than anything else drawn here:

* **yellow crosshair** - where your index fingertip actually is
* **magenta dot** - where the aiming model *thinks* it is pointing

Watching those two converge is how you tune ``gain`` and the smoothing
constants instead of guessing. The corner minimap then shows where that lands
on the screen.
"""

from __future__ import annotations

import cv2

from . import landmarks as L

_CYAN = (255, 200, 0)
_YELLOW = (0, 220, 255)
_MAGENTA = (255, 0, 220)
_GREEN = (120, 230, 120)
_RED = (80, 80, 255)
_GREY = (190, 190, 190)
_WHITE = (255, 255, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _px(point, width: int, height: int) -> tuple[int, int]:
    return (int(point[0] * width), int(point[1] * height))


def draw_hand(frame, points, connections, highlight: set | None = None) -> None:
    """Draw the skeleton. ``highlight`` marks landmarks drawn large.

    The app passes the landmarks of whichever finger is currently acting as the
    button, so you can see at a glance what the click detector is reading.
    """
    height, width = frame.shape[:2]

    for conn in connections:
        a = _px(points[conn.start], width, height)
        b = _px(points[conn.end], width, height)
        cv2.line(frame, a, b, _GREEN, 1, cv2.LINE_AA)

    for idx, point in enumerate(points):
        centre = _px(point, width, height)
        if highlight and idx in highlight:
            cv2.circle(frame, centre, 6, _CYAN, -1, cv2.LINE_AA)
        radius = 4 if idx == L.WRIST else 2
        colour = _YELLOW if idx == L.INDEX_TIP else _WHITE
        cv2.circle(frame, centre, radius, colour, -1, cv2.LINE_AA)

    tip = _px(points[L.INDEX_TIP], width, height)
    size = 9
    cv2.line(frame, (tip[0] - size, tip[1]), (tip[0] + size, tip[1]), _YELLOW, 1, cv2.LINE_AA)
    cv2.line(frame, (tip[0], tip[1] - size), (tip[0], tip[1] + size), _YELLOW, 1, cv2.LINE_AA)


def draw_pinch_gauge(frame, points, closed: bool) -> None:
    """The thumb-to-index gap that pinch clicking reads, drawn as a bar."""
    height, width = frame.shape[:2]
    a = _px(points[L.THUMB_TIP], width, height)
    b = _px(points[L.INDEX_TIP], width, height)
    colour = _RED if closed else _CYAN
    cv2.line(frame, a, b, colour, 2, cv2.LINE_AA)
    if closed:
        cv2.circle(frame, b, 11, colour, 2, cv2.LINE_AA)


def draw_aim_point(frame, point, ok: bool = True) -> None:
    """Where the aiming model resolved the fingertip to."""
    height, width = frame.shape[:2]
    centre = _px(point, width, height)
    colour = _MAGENTA if ok else _GREY
    cv2.circle(frame, centre, 5, colour, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, centre, colour, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)


def draw_target_ring(frame, point, colour, label: str = "") -> None:
    """The calibration target: a ring to align the fingertip with."""
    height, width = frame.shape[:2]
    centre = _px(point, width, height)
    cv2.circle(frame, centre, 18, colour, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, centre, colour, cv2.MARKER_CROSS, 30, 1, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 1)
        x = min(max(centre[0] - tw // 2, 2), width - tw - 2)
        cv2.putText(frame, label, (x, centre[1] - 26), _FONT, 0.55, colour, 1, cv2.LINE_AA)


def draw_progress(frame, fraction: float, colour) -> None:
    """Ring fill around the target showing sample progress."""
    height, width = frame.shape[:2]
    centre = (width // 2, height - 40)
    cv2.ellipse(frame, centre, (26, 26), 0, -90, -90 + 360 * max(0.0, min(1.0, fraction)),
                colour, 3, cv2.LINE_AA)


def draw_minimap(frame, screen_bounds, cursor_pos) -> None:
    """Tiny view of the display with the cursor's position in it."""
    height, width = frame.shape[:2]
    x0, y0, bw, bh = screen_bounds
    margin = 10
    box_w = min(150, width - 2 * margin)
    box_h = max(1, int(box_w * bh / bw)) if bw else 1
    top = height - box_h - margin

    cv2.rectangle(frame, (margin, top), (margin + box_w, top + box_h), _GREY, 1, cv2.LINE_AA)

    rel_x = (cursor_pos[0] - x0) / bw if bw else 0.0
    rel_y = (cursor_pos[1] - y0) / bh if bh else 0.0
    dot = (
        margin + int(min(max(rel_x, 0.0), 1.0) * box_w),
        top + int(min(max(rel_y, 0.0), 1.0) * box_h),
    )
    cv2.circle(frame, dot, 3, _MAGENTA, -1, cv2.LINE_AA)


def draw_hud(frame, lines: list[tuple[str, tuple]]) -> None:
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (10, 18 + i * 16), _FONT, 0.45, colour, 1, cv2.LINE_AA)


class Overlay:
    """Owns the cv2 window and keyboard input."""

    def __init__(self, window: str = "point-and-click") -> None:
        self.window = window
        self._open = False
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def show(self, frame) -> None:
        cv2.imshow(self.window, frame)
        self._open = True

    def poll_key(self) -> int:
        """Key pressed since the last call, or -1."""
        return cv2.waitKey(1) & 0xFF

    def resize(self, width: int, height: int) -> None:
        try:
            cv2.resizeWindow(self.window, width, height)
        except cv2.error:
            pass  # window backend can be briefly unavailable during teardown

    def close(self) -> None:
        if self._open:
            cv2.destroyWindow(self.window)
            self._open = False


def banner(frame, text: str, colour=_YELLOW) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height // 2 - 26), (width, height // 2 + 18), (30, 30, 30), -1)
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.7, 2)
    cv2.putText(frame, text, ((width - tw) // 2, height // 2 + 6), _FONT, 0.7, colour, 2, cv2.LINE_AA)