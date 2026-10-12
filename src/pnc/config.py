"""Configuration for point-and-click.

Every tunable lives in a dataclass here, overridable from
``config/default.toml``. Tuning the feel of the tool should never require
editing code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
MODELS_DIR = REPO_ROOT / "models"
STATE_DIR = REPO_ROOT / "state"
CALIB_PATH = STATE_DIR / "calib.json"

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "{variant}/float16/1/{variant}.task"
)


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    """Flip the frame horizontally.

    The MacBook camera is not mirrored, so moving your hand to *your* right
    makes it appear to move to the frame's left. We flip once, at input, so
    that "further right" always means "further right on the cursor" and the
    preview reads as a mirror rather than as a video of someone else.
    """
    num_hands: int = 2
    """Hands to track.

    Two even when only one is needed, because it measures free (the palm
    detector dominates and runs once) and it lets the click method be switched
    without rebuilding the tracker.
    """


@dataclass
class AimConfig:
    """The wrist-anchored aiming model.

    The index fingertip is projected to a *fixed distance from the camera*,
    anchored at the wrist::

        u = wrist + (index_tip - wrist) * (gain / scale)

    Dividing by ``scale`` (a rigid bone pair, therefore a depth proxy) makes the
    target immune to you leaning towards or away from the camera.
    """

    gain: float = 0.5
    """How many hand-lengths beyond the wrist the fingertip projects to.

    The units matter, because this is easy to overshoot by an order of
    magnitude. ``(tip - wrist) / scale`` is the fingertip's distance from the
    wrist measured in hand-lengths, and ``gain`` multiplies it. An index finger
    is roughly 1.0-1.2 hand-lengths long, so:

      * ``gain = 1.0`` lands the target exactly on the fingertip: no gain at all.
      * ``gain < 1.0`` pulls it back towards the wrist, so the *direction* you
        point dominates over where your hand happens to be.
      * ``gain > 1.5`` overshoots hard: the cursor pins itself to a screen
        corner and stops responding.

    0.4-0.6 covers most of the display on its own. Calibration then stretches
    whatever range you actually use to fill the screen exactly.
    """
    min_scale: float = 0.015
    """Below this hand size the depth division is unsafe (practically never)."""

    gain_elbow: float = 0.2
    """``gain`` to use when pivoting from the elbow.

    The elbow-to-fingertip lever is roughly 2.4x the wrist-to-fingertip one
    (44 cm against 18 cm), so the same ``gain`` would move the cursor ~2.4x
    further for the same forearm rotation - every motion exaggerated, every
    wobble amplified. 0.5 / 2.4 lands at ~0.2, which is the setting that keeps
    an elbow pivot feeling like the wrist pivot did.

    Note this buys *ergonomics*, not range: at matched sensitivity both pivots
    reach about equally far.
    """
    median_window: int = 3
    """Landmark outlier rejection. 1 disables it."""
    min_cutoff: float = 1.2
    """1 Euro filter. Lower = smoother, laggier. Higher = jittery, snappy."""
    beta: float = 0.02
    """1 Euro filter speed coefficient. Higher = less lag when moving fast."""


@dataclass
class GestureConfig:
    extend_ratio: float = 1.35
    fold_ratio: float = 1.15
    confirm_frames: int = 2
    max_hold_ms: int = 2000


@dataclass
class PinchConfig:
    """Thumb-to-index pinch clicking.

    The gap between the thumb and index tips, divided by hand size. Pinching
    gives roughly 0.1-0.2; a hand at rest gives roughly 0.5-0.9.
    """

    press_ratio: float = 0.35
    """Below this the pinch counts as pressed."""
    release_ratio: float = 0.5
    """Above this the pinch counts as open. Must exceed press_ratio."""
    freeze_on_press: bool = True
    """Latch the cursor position when a pinch starts.

    Pinching curls the index finger, which is the finger doing the aiming, so
    without latching the cursor slides as you close the pinch and can land on
    whatever was next to your target. Latching pins the click where you aimed.
    The cost is that pinch cannot drag - use the middle-finger method for that.
    """


@dataclass
class TwoHandConfig:
    """Two-handed clicking: point with one hand, click with a fist in the other.

    The aiming hand is whichever the role resolver picks, and the click comes
    from the *other* one - so clicking cannot disturb your aim at all.
    """

    pointing_hand: str = "auto"
    """Which hand aims: "auto", "left", or "right".

    "auto" picks the hand whose index finger is extended relative to its other
    three, which separates a pointing hand from a fist by a wide margin. Pin it
    if the automatic guess ever picks wrong - the preview HUD prints which hand
    it settled on, so a swap is visible immediately.
    """
    fist_ratio: float = 0.6
    """Mean finger extension at or below this counts as a closed fist.

    Measured on a real resting hand: p5=0.52, median 0.92, p95=1.57. A
    deliberate fist must land well under that spread, so this sits below the
    resting p25 - otherwise resting triggers clicks on its own.
    """
    open_ratio: float = 0.85
    """Mean finger extension at or above this counts as open. Must exceed
    fist_ratio.

    Deliberately *below* the resting median (0.92): after a click you relax your
    hand rather than splaying it flat, so requiring a flat open hand would leave
    you holding the button.

    Both thresholds are close together because a relaxed hand and a fist are not
    hugely different signals. That is a thin margin, and it is the reason to run
    ``pnc doctor --gestures`` while actually clenching and releasing: it will
    tell you if your hand never reaches these.
    """
    max_hold_ms: int = 0
    """0 disables the hold timeout, so a fist stays down until you open it.

    A long hold is what this gesture is for - selecting text, dragging a window
    - so capping it would break the gesture rather than protect you. Every other
    safety net still applies: losing your hands, pausing, switching method, and
    quitting all force a release.
    """


@dataclass
class ArmConfig:
    """Elbow detection for the ``elbow`` pivot."""

    min_visibility: float = 0.5
    """Pose confidence required to trust the elbow.

    If your upper body leaves the frame, MediaPipe keeps *emitting* elbow
    coordinates - they just stop meaning anything. Gating on its own visibility
    score is what stops a confidently-wrong elbow from being amplified through a
    long lever arm into a wild cursor jump.
    """
    side: str = "auto"
    """Which elbow: "auto" (from the hand tracker's handedness), "left", "right"."""
    model: str = "lite"
    """Pose model size. lite is ~5.8 MB and plenty for an elbow."""


@dataclass
class SafetyConfig:
    hand_lost_release_ms: int = 250
    """Drop the cursor and release any button this soon after losing the hand."""
    resume_grace_ms: int = 1500
    """After the real mouse is touched, keep ignoring it for this long."""


@dataclass
class PreviewConfig:
    enabled: bool = True
    window: str = "point-and-click"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    aim: AimConfig = field(default_factory=AimConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    pinch: PinchConfig = field(default_factory=PinchConfig)
    two_hand: TwoHandConfig = field(default_factory=TwoHandConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)

    click_method: str = "middle-finger"
    """Which gesture clicks: "middle-finger" or "pinch".

    Overridden by ``--click-method``; here so the default can live in the config
    file alongside the thresholds it selects between.
    """

    pivot: str = "wrist"
    """What the pointing ray pivots about: "wrist" or "elbow".

    Overridden by ``--pivot``.
    """


def _merge_into(obj, data: dict) -> None:
    """Recursively overlay a TOML table onto a dataclass instance."""
    for f in fields(obj):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(obj, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into(current, value)
        elif isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
            # TOML gives us ints where a float field is declared.
            setattr(obj, f.name, float(value))
        else:
            setattr(obj, f.name, value)


def load_config(path: str | Path | None = None) -> Config:
    """Load defaults, then overlay ``config/default.toml`` (or an explicit path)."""
    cfg = Config()
    candidate = Path(path) if path else CONFIG_DIR / "default.toml"
    if candidate.is_file():
        with candidate.open("rb") as fh:
            _merge_into(cfg, tomllib.load(fh))
    return cfg


def model_path() -> Path:
    return MODELS_DIR / MODEL_FILENAME


def pose_model_path(variant: str = "lite") -> Path:
    return MODELS_DIR / f"pose_landmarker_{variant}.task"


def _download(url: str, destination) -> None:
    """Fetch ``url`` to ``destination``.

    Python's own cert store is used when it works, but a homebrew or
    framework build of Python often cannot verify the issuer chain for
    googleapis.com. ``certifi`` ships as a mediapipe dependency, so fall back to
    its bundle rather than failing the very first run.
    """
    import ssl
    import urllib.request

    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
        with opener.open(url, timeout=60) as response, destination.open("wb") as fh:
            fh.write(response.read())
        return
    except ImportError:
        pass

    urllib.request.urlretrieve(url, destination)


def ensure_model() -> Path:
    """Download the hand landmarker model on first use. Returns its path."""
    target = model_path()
    if target.is_file() and target.stat().st_size > 0:
        return target

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    print(f"==> downloading hand landmarker model to {target}")
    try:
        _download(MODEL_URL, tmp)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim, it is actionable
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the model from {MODEL_URL}: {exc}\n"
            f"Download it manually and save it to {target}."
        ) from exc
    tmp.replace(target)
    return target


def ensure_pose_model(variant: str = "lite") -> Path:
    """The pose model is only needed for the elbow pivot, so it downloads lazily."""
    target = pose_model_path(variant)
    if target.is_file() and target.stat().st_size > 0:
        return target

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = _POSE_MODEL_URL.format(variant=f"pose_landmarker_{variant}")
    tmp = target.with_suffix(".part")
    print(f"==> downloading pose landmarker model to {target}")
    try:
        _download(url, tmp)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim, it is actionable
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the pose model from {url}: {exc}\n"
            f"Download it manually and save it to {target}."
        ) from exc
    tmp.replace(target)
    return target