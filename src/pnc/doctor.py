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


def _finish(failures: int) -> int:
    print()
    if failures:
        print(f"{failures} problem(s) found - see the FAIL lines above.")
        return 1
    print("All checks passed. Run `pnc calibrate`, then `pnc run`.")
    return 0


def main(camera_index: int = 0, benchmark: bool = True) -> int:
    # Config is loaded purely so a malformed default.toml surfaces here rather
    # than mid-session.
    load_config()
    return run_doctor(camera_index=camera_index, benchmark=benchmark)