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
from .arm import ArmTracker
from .camera import Camera, VideoSource
from .config import CALIB_PATH, Config, ensure_model, ensure_pose_model
from .controls import InputGuard
from .cursor import Cursor
from .filters import CursorSmoother
from .hands import HandTracker, connections
from .mapper import (
    PIVOT_ELBOW,
    PIVOT_NAMES,
    PIVOT_WRIST,
    DefaultProjector,
    aim_target,
    clamp_to_display,
    from_screen_space,
    load_projector,
    to_screen_space,
)
from .roles import RoleResolver

LOST = "LOST"
TRACKING = "TRACKING"
CLICKING = "CLICKING"
PAUSED = "PAUSED"

_HELP = "q quit | c calibrate | m pause | p preview | b pivot | x click | +/- gain"

# Consecutive frames without a usable elbow before falling back to the wrist.
PIVOT_DROPOUT_FRAMES = 3


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
        pivot: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.preview = cfg.preview.enabled if preview is None else preview
        self.video_path = video_path
        self.calib_path = CALIB_PATH
        self._gain_override = gain

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
        self.roles = RoleResolver(cfg.two_hand.pointing_hand)
        self.guard = InputGuard()

        self.pivot = pivot or cfg.pivot
        if self.pivot not in PIVOT_NAMES:
            raise ValueError(
                f"Unknown pivot {self.pivot!r}. Choose from {sorted(PIVOT_NAMES)}."
            )
        self._aspect = 4.0 / 3.0
        self.matrix = calib.load(self.calib_path)
        self.projector = None
        # _apply_pivot derives the gain for the chosen pivot and touches the
        # projector, so it has to run after those two exist.
        self._apply_pivot()
        self.arm: ArmTracker | None = None

        self.running = True
        self.start_calibration = recalibrate
        self._click_anchor: tuple[float, float] | None = None
        self._elbow_live = False
        self._elbow_miss_run = 0
        self._elbow_used = None

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

    def _map_to_screen(self, points, now_ms: float, dt: float | None, elbow=None):
        """Fingertips -> smoothed screen point, or ``None`` if unmeasurable."""
        target = aim_target(
            points,
            self.gain,
            self.cfg.aim.min_scale,
            pivot=self.effective_pivot,
            elbow=elbow,
        )
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
        elif key in (ord("b"), ord("B")):
            self._cycle_pivot()
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
        with source, HandTracker(model, num_hands=self.cfg.camera.num_hands) as tracker:
            view = None
            if self.preview:
                view = overlay.Overlay(self.cfg.preview.window)

            # The pose model only loads if the elbow pivot is actually in use,
            # so the default path stays a single model and 30 fps.
            arm = self._maybe_open_arm()

            if self.start_calibration:
                # Calibration needs the preview: it is the feedback loop that
                # tells you when you are pointing at the right place.
                if view is None:
                    self.preview = True
                    view = overlay.Overlay(self.cfg.preview.window)
                self.calibrate(source, tracker, connections_list, view, arm)
                self.start_calibration = False

            if self.matrix is None:
                print("No calibration found yet, using the default mapping.")
                print("Run `pnc calibrate` (or press c in the preview) to teach it where you point.")
            else:
                print(f"Calibration loaded from {self.calib_path}")
            self._warn_calibration_pivot()

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

                # Which hand aims and which clicks. Sticky, so the two cannot
                # swap back and forth while both hands are open.
                pointing, clicking = self.roles.resolve(hands)

                # Pose must run on the same pixels, so its coordinates line up
                # with the hand landmarks.
                elbow = None
                if arm is not None:
                    elbow = arm.elbow(
                        rgb, int(now), pointing.handedness if pointing is not None else None
                    )

                if self.start_calibration:
                    self.calibrate(source, tracker, connections_list, view, arm)
                    self.start_calibration = False
                    continue

                self._step(pointing, clicking, now, dt, frame, connections_list, view, elbow)

                if self.preview and view is not None:
                    view.show(frame)
                    if not self._handle_key(view.poll_key()):
                        break

            if arm is not None:
                arm.close()
            if self.preview and view is not None:
                view.close()

        self._shutdown()
        self._restore_signal_handlers()
        return 0

    def _maybe_open_arm(self) -> ArmTracker | None:
        """Load the pose model only if the elbow pivot is selected."""
        if self.pivot != PIVOT_ELBOW:
            return None
        try:
            model = ensure_pose_model(self.cfg.arm.model)
            return ArmTracker(
                model,
                min_visibility=self.cfg.arm.min_visibility,
                side=self.cfg.arm.side,
            )
        except (RuntimeError, Exception) as exc:  # noqa: BLE001 - degrade, never die
            print(f"! Could not start arm tracking ({exc}); falling back to the wrist pivot.")
            self.pivot = PIVOT_WRIST
            self._apply_pivot()
            return None

    def _warn_calibration_pivot(self) -> None:
        """A homography fitted to one pivot is wrong for the other."""
        meta = calib.load_meta(self.calib_path)
        fitted = meta.get("pivot")
        if fitted is None or fitted == self.pivot:
            return
        print()
        print(f"! Calibration was fitted for the {fitted} pivot, but you are running")
        print(f"  the {self.pivot} pivot. The mapping will be off until you recalibrate:")
        print("    pnc calibrate")
        print()

    # ------------------------------------------------------------------- pivot

    def _update_pivot(self, elbow, now: float):
        """Decide whether to aim from the elbow this frame.

        Returns the elbow to aim from, or ``None`` to fall back to the wrist.

        The awkward part is that changing the anchor *teleports the cursor*,
        because the anchor is an additive term in the projection. Losing your
        arm mid-drag would yank the cursor under the user's finger, so:

        - while a button is held, the choice is frozen at whatever it was when
          the press began;
        - dropping out takes several consecutive misses, so a single occluded
          frame does not switch you over;
        - switching resets the smoother, which would otherwise smear the jump
          across several frames and leave the cursor visibly trailing.
        """
        if self.pivot != PIVOT_ELBOW:
            self._elbow_miss_run = 0
            self._elbow_live = False
            return None

        if self.fsm.pressed:
            # Hold the anchor steady for the duration of the click.
            return elbow if self._elbow_live else None

        if elbow is not None:
            self._elbow_miss_run = 0
            if not self._elbow_live:
                self._elbow_live = True
                self.smoother.reset()
                self._notify("aiming from your elbow", now)
            return elbow

        self._elbow_miss_run += 1
        if self._elbow_live and self._elbow_miss_run >= PIVOT_DROPOUT_FRAMES:
            self._elbow_live = False
            self.smoother.reset()
            self._notify("arm not visible - aiming from your wrist", now)
        return None

    @property
    def effective_pivot(self) -> str:
        """The pivot in force right now, which may differ from the request."""
        if self.pivot == PIVOT_ELBOW and self._elbow_live:
            return PIVOT_ELBOW
        return PIVOT_WRIST

    def _apply_pivot(self, pivot: str | None = None) -> None:
        """Switch pivot mode, or just re-derive state if ``pivot`` is None."""
        if pivot is not None:
            self.pivot = pivot
        self._elbow_live = False
        self._elbow_miss_run = 0
        self.smoother.reset()

        default_gain = (
            self.cfg.aim.gain_elbow
            if self.pivot == PIVOT_ELBOW
            else self.cfg.aim.gain
        )
        # An explicit --gain is a deliberate override and survives the switch.
        self.gain = default_gain if self._gain_override is None else self._gain_override

        if self.projector is not None:
            self._refresh_projector(self._aspect)

    def _cycle_pivot(self) -> None:
        """Toggle between wrist and elbow, live.

        Safe to do mid-click: the button is released first, because the two
        pivots are different mappings and continuing a drag across that change
        would fling the cursor.
        """
        if self.cursor.button_down:
            self.cursor.release()
        self.fsm.force_release()
        self._click_anchor = None

        nxt = PIVOT_WRIST if self.pivot == PIVOT_ELBOW else PIVOT_ELBOW
        self._apply_pivot(nxt)

        if nxt == PIVOT_ELBOW:
            self._notify("pivot: elbow (needs the pose model)", _now_ms())
        else:
            self._notify(f"pivot: wrist (gain {self.gain:.2f})", _now_ms())

    # ------------------------------------------------------------------- step

    def _step(self, pointing, clicking, now: float, dt, frame, connections_list, view, elbow=None) -> None:
        """One frame.

        ``pointing`` is the hand that aims, ``clicking`` the other one - which
        is the same object for single-handed methods, since they do not use the
        second role at all.
        """
        method = self.click_method
        signal_hand = pointing if method.hand_role == gestures.ROLE_POINTING else clicking

        # 1. Has the real mouse been used? Check before we overwrite it.
        if self.guard.detect(self.cursor, now):
            if self.cursor.button_down:
                self.cursor.release()
            self.fsm.force_release()
            self._resume_since = None
            self._notify("physical mouse detected - pausing", now)

        has_hand = pointing is not None
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

        # Losing the hand this method *reads* is what releases the button. Losing
        # only the aiming hand does not: the cursor freezes, but if you were
        # dragging and your aiming hand dropped out of frame you are mid-gesture,
        # not finished.
        if signal_hand is None:
            if self._resume_since is None:
                self._resume_since = now
            if self._lost_since is None:
                self._lost_since = now
            elif (now - self._lost_since) >= self.cfg.safety.hand_lost_release_ms:
                if self.cursor.button_down:
                    self.cursor.release()
                self.fsm.force_release()
                self.smoother.reset()
        else:
            self._lost_since = None

        # 3. Which pivot are we actually using this frame?
        self._elbow_used = self._update_pivot(elbow, now)

        screen_point, aim_point = (None, None)
        if has_hand:
            screen_point, aim_point = self._map_to_screen(
                pointing.landmarks, now, dt, self._elbow_used
            )
            if screen_point is None:
                has_hand = False

        # 4. Gestures.
        clicked_now = False
        signal = method.signal(signal_hand.landmarks) if signal_hand is not None else None
        if signal is not None:
            event = self.fsm.update(signal, int(now))
            if event == gestures.PRESS:
                self.cursor.press()
                clicked_now = True
                if method.freeze_on_press and screen_point is not None:
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

        # 5. Draw.
        if self.preview and view is not None:
            self._draw(
                frame, pointing, clicking, aim_point, connections_list,
                now, has_hand, paused, clicked_now,
            )

    def _draw(self, frame, pointing, clicking, aim_point, connections_list,
              now, has_hand, paused, clicked_now) -> None:
        method = self.click_method
        signal_hand = pointing if method.hand_role == gestures.ROLE_POINTING else clicking

        # Both hands are drawn when two-handed clicking is in use, because which
        # one is doing what is otherwise invisible.
        for hand, is_signal_hand in ((pointing, signal_hand is pointing),
                                     (clicking, signal_hand is clicking)):
            if hand is None or (hand is pointing and not has_hand):
                continue
            highlight = set(method.highlight) if (self.fsm.pressed and is_signal_hand) else None
            overlay.draw_hand(frame, hand.landmarks, connections_list, highlight=highlight)

        if has_hand and pointing is not None:
            points = pointing.landmarks
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
            if self.pivot == PIVOT_ELBOW:
                overlay.draw_pivot_axis(frame, points, self._elbow_used, self._elbow_live)

        if self.fsm.pressed:
            state, colour = CLICKING, overlay._RED
        elif paused:
            state, colour = PAUSED, overlay._YELLOW
        elif has_hand:
            state, colour = TRACKING, overlay._GREEN
        else:
            state, colour = LOST, overlay._YELLOW

        signal = method.signal(signal_hand.landmarks) if signal_hand is not None else None
        signal_text = f"{signal:.2f}" if signal is not None else "--"

        if self.pivot == PIVOT_ELBOW:
            if self._elbow_live:
                pivot_text, pivot_colour = "elbow", overlay._GREEN
            else:
                pivot_text, pivot_colour = "wrist (arm not visible)", overlay._YELLOW
        else:
            pivot_text, pivot_colour = "wrist", overlay._WHITE

        lines = [
            (f"{self._fps:4.0f} fps", overlay._WHITE),
            (f"state {state}", colour),
            (f"pivot {pivot_text}", pivot_colour),
            (f"gain {self.gain:.2f}", overlay._WHITE),
            (f"click {method.label} {signal_text}", overlay._WHITE),
        ]
        if method.hand_role == gestures.ROLE_CLICKING:
            # Print the assignment: if MediaPipe's handedness is mirrored the
            # wrong way round, this is the one line that makes it obvious.
            lines.append((f"pointing hand: {self.roles.describe()}", overlay._CYAN))
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

    def calibrate(self, source, tracker: HandTracker, connections_list, view, arm=None) -> None:
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
        sample_pivots: dict[str, str] = {}
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
                stamp = int(_now_ms())
                hands = tracker.detect(rgb, stamp)
                now = _now_ms()
                # Calibrate with the same hand that will aim at runtime, or the
                # fitted mapping describes a different geometry.
                pointing, _clicking = self.roles.resolve(hands)
                points = pointing.landmarks if pointing is not None else None

                # Calibrate against the same pivot that will be used at runtime,
                # or the fitted mapping describes a different geometry.
                elbow = None
                if arm is not None:
                    elbow = arm.elbow(
                        rgb, stamp, pointing.handedness if pointing is not None else None
                    )
                self._elbow_used = elbow
                pivot = PIVOT_ELBOW if elbow is not None else PIVOT_WRIST

                overlay.draw_target_ring(frame, target_uv, overlay._CYAN, f"point at the {name}")

                distance = None
                if points is not None:
                    overlay.draw_hand(frame, points, connections_list)
                    overlay.draw_pivot_axis(frame, points, elbow, elbow is not None)
                    aim_point = aim_target(
                        points,
                        self.gain,
                        self.cfg.aim.min_scale,
                        pivot=pivot,
                        elbow=elbow,
                    )
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
            sample_pivots[name] = pivot
            print(f"    captured {name}")

        if len(samples) < len(calib.CORNER_NAMES):
            print()
            print("Calibration incomplete - keeping the previous mapping.")
            return

        # Mixing pivots would fit a mapping describing a geometry that never
        # exists, so refuse rather than save something subtly wrong.
        used_pivots = {sample_pivots[n] for n in samples}
        if len(used_pivots) > 1:
            print()
            print("Calibration mixed wrist and elbow pivots (your arm came in and out")
            print("of frame). Run it again and keep your arm visible throughout.")
            return
        fitted_pivot = used_pivots.pop()

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
                "pivot": fitted_pivot,
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