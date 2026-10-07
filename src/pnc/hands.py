"""MediaPipe hand landmark detection.

MediaPipe is imported lazily inside the class so that the pure-logic modules
(filters, mapper, gestures) stay importable - and testable - without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hand:
    """One detected hand, carrying the handedness needed to pick the right elbow."""

    landmarks: list
    handedness: str | None = None
    """MediaPipe's label for the tracked hand, or None if it is unsure."""
    score: float = 0.0

    def __len__(self) -> int:
        return len(self.landmarks)


class HandTracker:
    """Wraps ``HandLandmarker`` in VIDEO mode and returns plain Python points."""

    def __init__(
        self,
        model_path: str,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        self._mp = mp
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> list[Hand]:
        """Detect hands in a BGR->RGB frame. Returns a list of :class:`Hand`.

        Each landmark is ``(x, y)`` normalised to the frame, so resolution
        changes do not affect any downstream maths.
        """
        # VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame, dtype=np.uint8),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        hands: list[Hand] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            handedness, score = None, 0.0
            if index < len(result.handedness) and result.handedness[index]:
                best = max(result.handedness[index], key=lambda c: c.score)
                handedness, score = best.category_name, best.score
            hands.append(
                Hand(
                    landmarks=[(float(lm.x), float(lm.y)) for lm in landmarks],
                    handedness=handedness,
                    score=float(score),
                )
            )
        return hands

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def connections():
    """The 21-point hand skeleton, for drawing."""
    from mediapipe.tasks.python.vision import HandLandmarksConnections

    return HandLandmarksConnections.HAND_CONNECTIONS