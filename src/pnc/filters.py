"""Smoothing primitives.

Two filters, both tuned for cursor control:

``MedianFilter``
    Cheap, non-causal outlier rejection. MediaPipe occasionally emits a
    landmark frame that is wildly wrong (a hand crossing a bright window, a
    partial occlusion). A 3-frame median deletes those spikes outright, which
    a smoother alone would smear across several frames instead.

``OneEuroFilter``
    The adaptive low-pass from Casiez, Roussel & Vogel (CHI 2012). A fixed
    low-pass has to choose between jitter and lag; the 1 Euro filter widens its
    cutoff with the speed of the signal, so it is heavily smoothed when you hold
    still and barely filtered when you whip across the screen. This single
    choice is most of the difference between "feels broken" and "feels good".
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

_TWO_PI = math.tau


class MedianFilter:
    """Reject outliers across a sliding window of samples."""

    def __init__(self, window: int) -> None:
        self.window = max(1, int(window))
        self._buf: deque[float] = deque(maxlen=self.window)

    def filter(self, x: float) -> float:
        if self.window == 1:
            return float(x)
        self._buf.append(float(x))
        return float(np.median(self._buf))

    def reset(self) -> None:
        self._buf.clear()


class _LowPass:
    def __init__(self) -> None:
        self.y: float | None = None

    def filter(self, x: float, alpha: float) -> float:
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y

    def reset(self) -> None:
        self.y = None


class OneEuroFilter:
    """Adaptive low-pass. See module docstring."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0:
            raise ValueError("min_cutoff must be positive")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last: float | None = None
        self._dt: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (_TWO_PI * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, dt: float | None = None) -> float:
        x = float(x)
        if self._last is None:
            # First sample: no velocity estimate yet, so settle rather than jump.
            self._dt = dt if dt and dt > 0 else None
            self._last = x
            return self._x.filter(x, 1.0)

        if dt is not None and dt > 0:
            self._dt = dt
        else:
            self._dt = self._dt or 1.0 / 60.0

        dx = (x - self._last) / self._dt
        self._last = x

        edx = self._dx.filter(dx, self._alpha(self.d_cutoff, self._dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.filter(x, self._alpha(cutoff, self._dt))

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last = None
        self._dt = None


class CursorSmoother:
    """Median + 1 Euro on both axes, which is what the app actually needs."""

    def __init__(self, median_window: int = 3, min_cutoff: float = 1.2, beta: float = 0.02) -> None:
        self._median_window = median_window
        self._filters = {
            "x": OneEuroFilter(min_cutoff, beta),
            "y": OneEuroFilter(min_cutoff, beta),
        }
        self._medians = {"x": MedianFilter(median_window), "y": MedianFilter(median_window)}

    def filter(self, x: float, y: float, dt: float | None = None) -> tuple[float, float]:
        mx = self._medians["x"].filter(x)
        my = self._medians["y"].filter(y)
        fx = self._filters["x"].filter(mx, dt)
        fy = self._filters["y"].filter(my, dt)
        return fx, fy

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()
        for m in self._medians.values():
            m.reset()