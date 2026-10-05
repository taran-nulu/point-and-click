"""The main loop, and the guided calibration routine.

State machine::

    LOST    -> TRACKING -> CLICKING
       ^         |           |
       +---------+-----------+
                 |
      PAUSED (real mouse was used) - resumes after a grace period and a
      stable hand
"""

from __future__ import annotations

import signal
import time

import cv2
import numpy as np

from . import calibration as calib
from . import gestures, overlay
from .camera import Camera, VideoSource
from .config import CALIB_PATH, Config, ensure_model
from .controls import InputGuard
from .cursor import Cursor
from .filters import CursorSmoother
from .hands import HandTracker, connections
from .mapper import (
    DefaultProjector,
    aim_target,
    clamp_to_display,
    from_screen_space,
    load_projector,
    to_screen_space,
)

LOST = "LOST"
TRACKING = "TRACKING"
CLICKING = "CLICKING"
PAUSED = "PAUSED"

_HELP = "q quit | c calibrate | m pause | p preview | +/- gain | x click method"


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class App:
    def __init__(
        self,
        cfg: Config,
        display_index: int = 0,
        gain: float | None = None,
        preview: bool | None = None,
        video_path: str | None = None,
        recalibrate: bool = False,
        click_method: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.gain = cfg.aim.gain if gain is None else gain
        self.preview = cfg.preview.enabled if preview is None else preview
        self.video_path = video_path
        self.calib_path = CALIB_PATH

        self.cursor = Cursor(display_index)
        self.smoother = CursorSmoother(
            median_window=cfg.aim.median_window,
            min_cutoff=cfg.aim.min_cutoff,
            beta=cfg.aim.beta,
        )
        self.click_method = gestures.build_click_method(
            click_method or cfg.click_method, cfg
        )
        self.fsm = gestures.ClickFSM.for_method(self.click_method, cfg)
        self.guard = InputGuard()

        self.matrix = calib.load(self.calib_path)
        self.projector = None
        self.calib_was_warned = False
        self.running = True
        self.start_calibration = recalibrate
        self._click_anchor: tuple[float, float] | None = None

        self._lost_since: float | None = None
        self._hand_seen_since: float | None = None
        self._resume_since: float | None = None
        self._notice: tuple[str, float] | None = None
        self._last_ts: float | None = None
        self._fps = 0.0
        self._signals: list = []

    # -------------------------------------------------------------- lifecycle

    def _install_signal_handlers(self) -> None:
        """Turn SIGINT/SIGTERM into a clean shutdown.

        This tool holds the real mouse button, so being killed outright must not
        be how it exits: a SIGTERM with no handler would leave the button down.
        """

        def stop(signum, _frame):
            self.running = False

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._signals.append((sig, signal.getsignal(sig)))
                signal.signal(sig, stop)
            except (ValueError, OSError):
                # Not on the main thread, or the signal is unsupported.
                pass

    def _restore_signal_handlers(self) -> None:
        for sig, previous in self._signals:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass
        self._signals.clear()

    # ---------------------------------------------------------------- helpers

    def _notify(self, text: str, now_ms: float, ttl_ms: float = 2500) -> None:
        self._notice = (text, now_ms + ttl_ms)

    def _refresh_projector(self, aspect: float) -> None:
        x0, y0, w, h = self.cursor.bounds
        self.projector = load_projector(aspect, w, h, self.matrix)

    def _map_to_screen(self, points, now_ms: float, dt: float | None):
        """Fingertips -> smoothed screen point, or ``None`` if unmeasurable."""
        target = aim_target(points, self.gain, self.cfg.aim.min_scale)
        if target is None:
            return None, None

        aspect = self._aspect
        ax, ay = to_screen_space(target[0], target[1], aspect)
        raw_x, raw_y = self.projector(ax, ay)
        smooth_x, smooth_y = self.smoother.filter(raw_x, raw_y, dt)
        x, y = clamp_to_display(smooth_x, smooth_y, self.cursor.width, self.cursor.height)
        # Projectors work in display-local coordinates; Quartz wants absolute.
        return (x + self.cursor.x0, y + self.cursor.y0), target

    def _handle_key(self, key: int) -> bool:
        """Returns False when the user asked to quit."""
        if key in (-1, -1):
            return True
        if key in (27,) or key in (ord("q"), ord("Q")):
            self.running = False
            return False
        if key in (ord("c"), ord("C")):
            self.start_calibration = True
        elif key in (ord("p"), ord("P")):
            self.preview = not self.preview
            self._notify("preview off" if not self.preview else "preview on", _now_ms())
        elif key in (ord("m"), ord("M")):
            if self.guard.tripped:
                self.guard.reanchor(self.cursor)
                self._notify("resumed", _now_ms())
            else:
                self.guard.trip(_now_ms())
                self._notify("paused - press m to resume", _now_ms())
        elif key in (ord("+"), ord("=")):
            self.gain += 0.1
            self._notify(f"gain {self.gain:.2f}", _now_ms())
        elif key in (ord("-"), ord("_")):
            self.gain = max(0.1, self.gain - 0.1)
            self._notify(f"gain {self.gain:.2f}", _now_ms())
        elif key in (ord("x"), ord("X")):
            self._cycle_click_method()
        return True

    def _cycle_click_method(self) -> None:
        """Switch click gesture live, so you can compare without restarting.

        Worth having: which gesture feels right is personal, and the thresholds
        for the other one are easier to judge while you can see the signal.
        """
        if self.cursor.button_down:
            self.cursor.release()
        self.fsm.force_release()
        self._click_anchor = None
        self.smoother.reset()

        names = list(gestures.CLICK_METHOD_NAMES)
        current = self.click_method.name
        index = (names.index(current) + 1) % len(names) if current in names else 0

        self.click_method = gestures.build_click_method(names[index], self.cfg)
        self.fsm = gestures.ClickFSM.for_method(self.click_method, self.cfg)
        self._notify(f"click: {self.click_method.help}", _now_ms(), ttl_ms=3500)

    # ------------------------------------------------------------------- loop

    def run(self) -> int:
        model = ensure_model()
        if self.video_path:
            source = VideoSource(self.video_path).open()
        else:
            source = Camera(
                index=self.cfg.camera.index,
                width=self.cfg.camera.width,
                height=self.cfg.camera.height,
                fps=self.cfg.camera.fps,
            ).open()
        connections_list = connections()

        self._install_signal_handlers()
        with source, HandTracker(model) as tracker:
            view = None
            if self.preview:
                view = overlay.Overlay(self.cfg.preview.window)

            if self.start_calibration:
                # Calibration needs the preview: it is the feedback loop that
                # tells you when you are pointing at the right place.
                if view is None:
                    self.preview = True
                    view = overlay.Overlay(self.cfg.preview.window)
                self.calibrate(source, tracker, connections_list, view)
                self.start_calibration = False

            if self.matrix is None:
                print("No calibration found yet, using the default mapping.")
                print("Run `pnc calibrate` (or press c in the preview) to teach it where you point.")
            else:
                print(f"Calibration loaded from {self.calib_path}")

            last_frame_ms = _now_ms()
            while self.running:
                ok, frame = source.read()
                if not ok:
                    print("Video ended." if self.video_path else "Camera stream ended.")
                    break

                now = _now_ms()
                frame_ms = now - last_frame_ms
                last_frame_ms = now
                self._fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0

                # Mirror once, here, so detection, landmarks and the preview
                # all share one coordinate convention.
                if self.cfg.camera.mirror:
                    frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                self._aspect = width / height
                if self.projector is None:
                    self._refresh_projector(self._aspect)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = tracker.detect(rgb, int(now))

                dt = (now - self._last_ts) / 1000.0 if self._last_ts else None
                self._last_ts = now

                points = hands[0] if hands else None

                if self.start_calibration:
                    self.calibrate(source, tracker, connections_list, view)
                    self.start_calibration = False
                    continue

                self._step(points, now, dt, frame, connections_list, view)

                if self.preview and view is not None:
                    view.show(frame)
                    if not self._handle_key(view.poll_key()):
                        break

            if self.preview and view is not None:
                view.close()

        self._shutdown()
        self._restore_signal_handlers()
        return 0

    # ------------------------------------------------------------------- step

    def _step(self, points, now: float, dt, frame, connections_list, view) -> None:
        # 1. Has the real mouse been used? Check before we overwrite it.
        if self.guard.detect(self.cursor, now):
            if self.cursor.button_down:
                self.cursor.release()
            self.fsm.force_release()
            self._resume_since = None
            self._notify("physical mouse detected - pausing", now)

        has_hand = points is not None
        if has_hand and self.guard.tripped:
            if self._resume_since is None:
                self._resume_since = now
            elif (
                self.guard.should_resume(now, self.cfg.safety.resume_grace_ms)
                and (now - self._resume_since) > 400
            ):
                self.guard.reanchor(self.cursor)
                self.smoother.reset()
                self._lost_since = None
                self._notify("resumed", now)

        if not has_hand:
            self._resume_since = None
            if self._lost_since is None:
                self._lost_since = now
            elif (now - self._lost_since) >= self.cfg.safety.hand_lost_release_ms:
                if self.cursor.button_down:
                    self.cursor.release()
                self.fsm.force_release()
                self.smoother.reset()
        else:
            self._lost_since = None

        screen_point, aim_point = (None, None)
        if has_hand:
            screen_point, aim_point = self._map_to_screen(points, now, dt)
            if screen_point is None:
                has_hand = False

        # 2. Gestures.
        clicked_now = False
        if has_hand:
            event = self.fsm.update(self.click_method.signal(points), int(now))
            if event == gestures.PRESS:
                self.cursor.press()
                clicked_now = True
                if self.click_method.freeze_on_press and screen_point is not None:
                    # Latch where you were aiming when the click started, so the
                    # distortion of the click gesture itself cannot move it.
                    self._click_anchor = screen_point
            elif event == gestures.RELEASE:
                self.cursor.release()
            if self.fsm.auto_released:
                self.fsm.auto_released = False
                self._notify("auto-released (held too long)", now)

        if not self.fsm.pressed:
            # Covers every release path in one place: a normal release, the
            # hold timeout, a forced release on hand loss, and pause.
            self._click_anchor = None

        paused = self.guard.tripped
        if has_hand and not paused and screen_point is not None:
            target = self._click_anchor or screen_point
            self.guard.note_write(self.cursor.move(*target))

        # 3. Draw.
        if self.preview and view is not None:
            self._draw(frame, points, aim_point, connections_list, now, has_hand, paused, clicked_now)

    def _draw(self, frame, points, aim_point, connections_list, now, has_hand, paused, clicked_now) -> None:
        method = self.click_method
        if has_hand and points is not None:
            highlight = set(method.highlight) if self.fsm.pressed else None
            overlay.draw_hand(frame, points, connections_list, highlight=highlight)
            if method.name == gestures.PINCH:
                signal = method.signal(points)
                overlay.draw_pinch_gauge(
                    frame,
                    points,
                    closed=signal is not None and signal <= method.press_threshold,
                )
            if aim_point is not None:
                overlay.draw_aim_point(frame, aim_point)
            overlay.draw_minimap(frame, self.cursor.bounds, self.cursor.position())

        if self.fsm.pressed:
            state, colour = CLICKING, overlay._RED
        elif paused:
            state, colour = PAUSED, overlay._YELLOW
        elif has_hand:
            state, colour = TRACKING, overlay._GREEN
        else:
            state, colour = LOST, overlay._YELLOW

        signal = method.signal(points) if has_hand else None
        signal_text = f"{signal:.2f}" if signal is not None else "--"

        lines = [
            (f"{self._fps:4.0f} fps", overlay._WHITE),
            (f"state {state}", colour),
            (f"gain {self.gain:.2f}", overlay._WHITE),
            (f"click {method.label} {signal_text}", overlay._WHITE),
        ]
        if self._notice and now < self._notice[1]:
            lines.append((self._notice[0], overlay._CYAN))
        lines.append((_HELP, overlay._GREY))
        overlay.draw_hud(frame, lines)

        if state == LOST:
            overlay.banner(frame, "show your hand - index finger out")
        elif state == PAUSED:
            overlay.banner(frame, "paused - move the mouse or press m to resume")
        elif clicked_now:
            overlay.banner(frame, "click", overlay._RED)

    def _shutdown(self) -> None:
        # Never leave a button held down.
        if self.cursor.button_down:
            self.cursor.release()
            print("released mouse button on exit")

    # ----------------------------------------------------------- calibration

    def calibrate(self, source, tracker: HandTracker, connections_list, view) -> None:
        """Interactive five-target calibration."""
        self._refresh_projector(source.actual_size[0] / source.actual_size[1])
        default_projector = DefaultProjector(
            self._aspect,
            self.cursor.width,
            self.cursor.height,
        )

        origin = (self.cursor.x0, self.cursor.y0)
        size = (self.cursor.width, self.cursor.height)

        print()
        print("Calibration")
        print("----------")
        print("Hold your hand up and point your index finger at the target ON THE SCREEN.")
        print("Align the fingertip with the ring in the preview; the sample is taken")
        print("automatically once you hold it steady.")
        print()
        print("Use the pose you will actually click with - a slightly folded middle")
        print("finger nudges the index fingertip, and calibration must account for it.")
        print()

        samples: dict[str, tuple[float, float]] = {}
        tolerance = 26.0  # pixels in the preview
        hold_ms = 350.0

        for name, fx, fy in calib.TARGETS:
            sx, sy = calib.screen_point(fx, fy, origin, size)
            expected_uv = default_projector.camera_from_screen(sx, sy)
            target_uv = from_screen_space(*expected_uv, self._aspect)

            print(f"--> point at the {name}")
            best: tuple[float, float] | None = None
            stable_since: float | None = None
            captured = False

            while self.running and not captured:
                ok, frame = source.read()
                if not ok:
                    return
                if self.cfg.camera.mirror:
                    frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]
                aspect = width / height
                self._aspect = aspect
                self._refresh_projector(aspect)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = tracker.detect(rgb, int(_now_ms()))
                now = _now_ms()
                points = hands[0] if hands else None

                overlay.draw_target_ring(frame, target_uv, overlay._CYAN, f"point at the {name}")

                distance = None
                if points is not None:
                    overlay.draw_hand(frame, points, connections_list)
                    aim_point = aim_target(points, self.gain, self.cfg.aim.min_scale)
                    if aim_point is not None:
                        overlay.draw_aim_point(frame, aim_point)
                        distance = float(np.hypot(
                            (aim_point[0] - target_uv[0]) * width,
                            (aim_point[1] - target_uv[1]) * height,
                        ))
                    hint = "align the fingertip with the ring"
                else:
                    hint = "hand not visible"

                if distance is not None and distance <= tolerance:
                    if stable_since is None:
                        stable_since = now
                    progress = (now - stable_since) / hold_ms
                    overlay.draw_progress(frame, progress, overlay._GREEN)
                    hint = "hold..."
                    if progress >= 1.0:
                        best = aim_point
                        captured = True
                else:
                    stable_since = None

                overlay.draw_hud(frame, [
                    (f"CALIBRATING: {name}", overlay._CYAN),
                    (hint, overlay._WHITE),
                ])
                if self.preview and view is not None:
                    view.show(frame)
                    if not self._handle_key(view.poll_key()):
                        return

            if best is None:
                print(f"    could not capture {name}, skipping")
                continue

            # Average a short burst so a single noisy frame does not define a corner.
            samples[name] = best
            print(f"    captured {name}")

        if len(samples) < len(calib.CORNER_NAMES):
            print()
            print("Calibration incomplete - keeping the previous mapping.")
            return

        try:
            matrix = calib.build_homography(samples, origin, size)
        except ValueError as exc:
            print(f"Calibration failed: {exc}")
            return

        error_px, error_pct = calib.validate(matrix, samples, origin, size)
        calib.save(
            matrix,
            self.calib_path,
            meta={
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gain": self.gain,
                "display_origin": list(origin),
                "display_size": list(size),
                "centre_error_points": error_px,
            },
        )
        self.matrix = matrix
        self._refresh_projector(self._aspect)

        print()
        print(f"Calibration saved to {self.calib_path}")
        print(f"Centre check: {error_px:.0f} points off ({error_pct:.1f}% of the diagonal)")
        if error_pct > 4.0:
            print("That is larger than ideal - consider recalibrating with steadier hands.")
        print()