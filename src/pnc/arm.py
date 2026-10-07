"""Locating the elbow, for the ``elbow`` pivot.

The hand landmarker sees 21 points on one hand and nothing above the wrist, so
there is no elbow in the data. This adds MediaPipe's pose landmarker purely to
read indices 13/14, which costs about 8 ms a frame.

Two failure modes drive the design:

**A bad elbow is worse than no elbow.** The elbow anchor is far from the
fingertip, so any error in the estimated elbow is amplified into a large error
in where the cursor goes. MediaPipe will happily *emit* elbow coordinates for a
person whose shoulders are outside the frame - they just stop meaning anything.
So we gate on the landmark's own ``visibility`` score and report nothing rather
than guessing, letting the caller fall back to the wrist.

**Wrong side is silent too.** A right-hand user's elbow chosen from the left
side of the image would aim at roughly double the true distance and feel like
pure noise. So we take the side from the hand tracker's handedness, and allow
forcing it for the mirrored-frame edge cases.
"""

from __future__ import annotations

# Pose landmark indices we care about. The full 33-point set includes both arms.
LEFT_ELBOW = 13
RIGHT_ELBOW = 14

_SIDE_TO_ELBOW = {"Left": LEFT_ELBOW, "Right": RIGHT_ELBOW}
_BY_LOWER = {"left": LEFT_ELBOW, "right": RIGHT_ELBOW}


def pick_index(points, handedness: str | None = None, side: str = "auto") -> int:
    """Choose which of the two elbows belongs to the tracked hand.

    ``side`` is the user's config override and is spelled lowercase
    ("auto"/"left"/"right"), while ``handedness`` arrives from MediaPipe as
    "Left"/"Right". Both are accepted either way: a typo'd override that was
    silently ignored would just quietly aim from the wrong elbow.
    """
    forced = _BY_LOWER.get(str(side).lower())
    if forced is not None:
        return forced
    if handedness in _SIDE_TO_ELBOW:
        return _SIDE_TO_ELBOW[handedness]
    if handedness and handedness.lower() in _BY_LOWER:
        return _BY_LOWER[handedness.lower()]

    # No usable handedness: fall back to whichever elbow pose is more confident,
    # which beats guessing a side.
    left, right = points[LEFT_ELBOW], points[RIGHT_ELBOW]
    return LEFT_ELBOW if left.visibility >= right.visibility else RIGHT_ELBOW


def select_elbow(
    points,
    handedness: str | None = None,
    side: str = "auto",
    min_visibility: float = 0.5,
):
    """The elbow as ``(x, y)``, or ``None`` when it cannot be trusted.

    Pure, so the trust rules can be tested without a camera or a pose model.
    """
    if points is None:
        return None

    elbow = points[pick_index(points, handedness, side)]
    if elbow.visibility < min_visibility:
        return None
    return (float(elbow.x), float(elbow.y))


class ArmTracker:
    """Reports the elbow position for the detected hand, or ``None``."""

    def __init__(self, model_path: str, min_visibility: float = 0.5, side: str = "auto") -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        self._mp = mp
        self.min_visibility = float(min_visibility)
        self.side = side

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
        )
        self._pose = PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def _pick_index(self, points, handedness: str | None) -> int:
        """Kept as a method so callers holding a tracker read naturally."""
        return pick_index(points, handedness, self.side)

    def elbow(self, rgb_frame, timestamp_ms: int, handedness: str | None = None):
        """Elbow in normalised image coords, or ``None`` if it cannot be trusted.

        ``rgb_frame`` must be the same pixels the hand tracker saw, so the two
        coordinate systems agree.
        """
        import numpy as np

        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame, dtype=np.uint8),
        )
        result = self._pose.detect_for_video(image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        points = result.pose_landmarks[0]
        return select_elbow(
            points,
            handedness=handedness,
            side=self.side,
            min_visibility=self.min_visibility,
        )

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self) -> "ArmTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()