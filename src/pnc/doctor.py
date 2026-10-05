"""``pnc doctor`` - check everything that can be wrong before you are mid-demo.

Hand tracking on macOS needs three things that can each fail silently and
confusingly: camera permission, Accessibility permission, and a model file. This
prints a plain verdict on each, plus a live frame-rate measurement, so problems
show up as one clear line instead of a cursor that mysteriously does not move.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

from . import calibration as calib
from .config import CALIB_PATH, ensure_model, load_config, model_path

_OK = "ok  "
_BAD = "FAIL"
_WARN = "warn"


def _line(ok: bool, label: str, detail: str = "") -> None:
    tag = _OK if ok else _BAD
    print(f"  [{tag}] {label:<22} {detail}")


def _warn(label: str, detail: str = "") -> None:
    print(f"  [{_WARN}] {label:<22} {detail}")


def run_doctor(camera_index: int = 0, benchmark: bool = True) -> int:
    failures = 0
    print("point-and-click doctor\n")

    # --- python -----------------------------------------------------------
    v = sys.version_info
    _line(v >= (3, 11), "python", f"{v.major}.{v.minor}.{v.micro}")

    # --- dependencies -----------------------------------------------------
    for module, label in (("numpy", "numpy"), ("cv2", "opencv"), ("mediapipe", "mediapipe")):
        try:
            mod = __import__(module)
            _line(True, label, getattr(mod, "__version__", "installed"))
        except ImportError as exc:
            _line(False, label, str(exc))
            failures += 1

    try:
        quartz = importlib.import_module("Quartz")
        _line(True, "quartz (pyobjc)", f"installed at {Path(quartz.__file__).parent.name}")
    except ImportError as exc:
        _line(False, "quartz (pyobjc)", str(exc))
        failures += 1
        return _finish(failures)

    # --- permissions ------------------------------------------------------
    from .cursor import is_accessibility_trusted, list_displays

    trusted = is_accessibility_trusted()
    _line(
        trusted,
        "accessibility",
        "granted"
        if trusted
        else "NOT granted - System Settings > Privacy & Security > Accessibility",
    )
    if not trusted:
        failures += 1

    # --- displays ---------------------------------------------------------
    displays = list_displays()
    if displays:
        for i, (x, y, w, h) in enumerate(displays):
            label = "display 0 (main)" if i == 0 else f"display {i}"
            _line(True, label, f"{w:.0f}x{h:.0f} at ({x:.0f},{y:.0f}) points")
    else:
        _line(False, "displays", "none active")
        failures += 1

    # --- model ------------------------------------------------------------
    if model_path().is_file():
        size_mb = model_path().stat().st_size / 1e6
        _line(True, "model", f"{model_path()} ({size_mb:.1f} MB)")
    else:
        try:
            path = ensure_model()
            _line(True, "model", f"downloaded to {path}")
        except RuntimeError as exc:
            _line(False, "model", str(exc))
            failures += 1
            return _finish(failures)

    # --- calibration ------------------------------------------------------
    matrix = calib.load(CALIB_PATH)
    if matrix is None:
        # Not a failure: the tool runs without it, it is just less accurate.
        _warn("calibration", f"none saved yet - run `pnc calibrate` ({CALIB_PATH})")
    else:
        meta = calib.load_meta(CALIB_PATH)
        detail = meta.get("created", "unknown date")
        if "centre_error_points" in meta:
            detail += f", centre error {meta['centre_error_points']:.0f} points"
        _line(True, "calibration", detail)

    # --- live camera + detector benchmark ---------------------------------
    if benchmark:
        print("\n  live check (hold your hand up, index finger out)")
        try:
            failures += _benchmark(camera_index)
        except Exception as exc:  # noqa: BLE001 - doctor reports, never crashes
            _line(False, "camera", f"{type(exc).__name__}: {exc}")
            failures += 1

    return _finish(failures)


def _benchmark(camera_index: int) -> int:
    import cv2

    from .camera import Camera
    from .config import ensure_model
    from .hands import HandTracker

    failures = 0
    cam = Camera(index=camera_index).open()
    _line(True, "camera", f"index {camera_index}, {cam.actual_size[0]}x{cam.actual_size[1]}")

    frames = 120
    with HandTracker(ensure_model()) as tracker:
        seen = 0
        start = time.perf_counter()
        for _ in range(frames):
            ok, frame = cam.read()
            if not ok:
                _line(False, "camera stream", "ended early")
                failures += 1
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if tracker.detect(rgb, int(time.perf_counter() * 1000)):
                seen += 1
        elapsed = time.perf_counter() - start
    cam.release()

    if elapsed > 0:
        fps = frames / elapsed
        per_frame = elapsed / frames * 1000
        detail = f"{fps:.0f} fps end-to-end ({per_frame:.1f} ms/frame)"
        _line(fps >= 20, "throughput", detail)
        if fps < 20:
            failures += 1
            print("       Below ~20 fps the cursor will feel laggy. Try a lower")
            print("       resolution in config/default.toml.")

    if seen:
        _line(True, "hand detection", f"found a hand in {seen}/{frames} frames")
    else:
        _line(False, "hand detection", "no hand seen in any frame")
        print("       Hold your hand in view of the camera, well lit, fingers apart.")
        print("       If it still fails, check lighting and try a plainer background.")
        failures += 1

    return failures


def _percentiles(values, points=(5, 25, 50, 75, 95)):
    import numpy as np

    a = np.asarray(values, dtype=float)
    return [float(np.percentile(a, p)) for p in points]


def assess_thresholds(values, press: float, release: float) -> list[str]:
    """Compare a signal's observed range against its configured thresholds.

    Returns warning lines; empty means the thresholds look usable for this hand.
    Pure, so it can be tested without a camera.
    """
    if not values:
        return ["no samples"]

    p5, p25, _, p75, p95 = _percentiles(values)
    warnings: list[str] = []

    if p95 < release:
        warnings.append(
            f"your hand never reaches release={release}; no click would ever release. "
            f"Lower it to about {p75:.2f}."
        )
    if p5 > press:
        warnings.append(
            f"your hand never drops to press={press}; clicks would never fire. "
            f"Raise it to about {p25:.2f}."
        )
    if p5 < press < release < p95:
        return warnings

    warnings.append(
        "the thresholds sit inside your hand's rest range, so clicks will misfire. "
        f"Try press={p5:.2f}, release={p95:.2f}, or move further from the camera."
    )
    return warnings


def _gesture_report(duration_s: float, config) -> int:
    """Report the live distribution of both click signals.

    The gesture thresholds in config/default.toml are starting estimates, not
    measured values, and they are very hand-specific. This shows where your
    actual hand sits relative to the configured numbers so you can set them
    from data instead of guessing - and spot a threshold that sits so close to
    your rest position that clicks fire by accident.
    """
    import time as _time

    import cv2

    from .camera import Camera
    from .config import ensure_model
    from .gestures import middle_extension, pinch_gap
    from .hands import HandTracker

    print("\n  gesture report")
    print("  For the next few seconds: point normally, then fold the middle")
    print("  finger, then pinch thumb to index. Do all three a few times.")

    middles: list[float] = []
    pinches: list[float] = []
    frames = 0
    hands_seen = 0

    with Camera(index=config.camera.index) as cam, HandTracker(ensure_model()) as tracker:
        deadline = _time.monotonic() + duration_s
        stamp = 0
        while _time.monotonic() < deadline:
            ok, frame = cam.read()
            if not ok:
                break
            if config.camera.mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stamp += 33
            found = tracker.detect(rgb, stamp)
            frames += 1
            if found:
                hands_seen += 1
                for hand in found:
                    m = middle_extension(hand)
                    p = pinch_gap(hand)
                    if m is not None:
                        middles.append(m)
                    if p is not None:
                        pinches.append(p)

    if not middles and not pinches:
        print("  No hand detected at all - check lighting and framing.")
        return 1

    print(f"  ({hands_seen}/{frames} frames had a hand)\n")

    failures = 0
    for name, values, press, release, lo_hint, hi_hint in (
        (
            "middle_extension (middle-finger click)",
            middles,
            config.gesture.fold_ratio,
            config.gesture.extend_ratio,
            "fold_ratio",
            "extend_ratio",
        ),
        (
            "pinch_gap (pinch click)",
            pinches,
            config.pinch.press_ratio,
            config.pinch.release_ratio,
            "press_ratio",
            "release_ratio",
        ),
    ):
        if not values:
            print(f"  {name}: no samples")
            continue

        p5, p25, p50, p75, p95 = _percentiles(values)
        print(f"  {name}")
        print(f"    p5={p5:.2f} p25={p25:.2f} mid={p50:.2f} p75={p75:.2f} p95={p95:.2f}")
        print(f"    configured {lo_hint}={press}  {hi_hint}={release}")
        print(
            f"    -> {(sum(1 for v in values if v <= press) / len(values)) * 100:.0f}% "
            f"of samples would press, "
            f"{(sum(1 for v in values if v >= release) / len(values)) * 100:.0f}% would release"
        )

        for warning in assess_thresholds(values, press, release):
            print(f"    ! {warning}")
            failures += 1
        print()

    return failures


def _finish(failures: int) -> int:
    print()
    if failures:
        print(f"{failures} problem(s) found - see the FAIL lines above.")
        return 1
    print("All checks passed. Run `pnc calibrate`, then `pnc run`.")
    return 0


def main(camera_index: int = 0, benchmark: bool = True, gestures: bool = False) -> int:
    # Config is loaded purely so a malformed default.toml surfaces here rather
    # than mid-session.
    config = load_config()
    config.camera.index = camera_index

    if gestures:
        return _finish(_gesture_report(6.0, config))
    return run_doctor(camera_index=camera_index, benchmark=benchmark)