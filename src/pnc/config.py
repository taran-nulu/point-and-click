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
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)


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