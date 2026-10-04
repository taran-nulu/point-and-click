"""Frame sources: the live camera, or a recorded video for offline debugging.

``--from-video`` replays a recording through the identical
landmark -> aiming -> smoothing path as the live tool, so a bad session can be
inspected after the fact (or in CI, where no camera exists).
"""

from __future__ import annotations

import cv2


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None
        self.actual_size = (0, 0)

    def open(self) -> "Camera":
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Could not open camera index {self.index}. "
                "Grant Camera access in System Settings > Privacy & Security > Camera, "
                "then try `pnc doctor`."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise CameraError(
                f"Camera {self.index} opened but returned no frames. It may be in use "
                "by another app (Zoom, FaceTime, Photo Booth)."
            )

        self._cap = cap
        self.actual_size = (frame.shape[1], frame.shape[0])
        return self

    def read(self):
        """Next frame as BGR, or ``(False, None)`` at end of stream."""
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


class VideoSource:
    """Same interface as Camera, backed by a file on disk."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.actual_size = (0, 0)
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self.native_fps = 30.0

    def open(self) -> "VideoSource":
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise CameraError(f"Could not open video file {self.path!r}.")

        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise CameraError(f"Video {self.path!r} contains no frames.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = fps if fps and fps > 0 else 30.0
        self._cap = cap
        self.actual_size = (frame.shape[1], frame.shape[0])
        return self

    def read(self):
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if ok:
            self._frame_index += 1
        return ok, frame

    def timestamp_ms(self) -> int:
        """Playback time in ms, so tracking runs at real speed, not warp speed."""
        return int(self._frame_index * 1000.0 / self.native_fps)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def probe(index: int) -> str | None:
    """Describe camera ``index``, or ``None`` if it cannot be opened."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok:
            return "opened, but no frames"
        h, w = frame.shape[:2]
        return f"{w}x{h}"
    finally:
        cap.release()