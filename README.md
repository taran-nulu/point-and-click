# point-and-click

Control your computer cursor by pointing at the screen with your hand.

Hold your hand up to the camera with your **index finger extended**, like a
laser pointer. Where the fingertip lands in the camera view drives the cursor.
**Fold your middle finger to click**; unfold it to release. Keeping the click on
a different finger means aiming never breaks while you click, and holding it
down gives you a click-drag.

```
./run.sh doctor      # check camera, permissions, throughput
./run.sh calibrate   # teach it where you point
./run.sh run         # go
```

## Requirements

macOS. Python 3.11+. `run.sh` creates a virtualenv and installs everything on
first run.

You need two macOS permissions, both checked by `doctor`:

| Permission | Why | Where |
|---|---|---|
| **Camera** | sees your hand | System Settings → Privacy & Security → Camera |
| **Accessibility** | moves the real cursor | System Settings → Privacy & Security → Accessibility |

macOS attributes both to the app that launched the process (your terminal, or
whatever is running this), so grant them there. If the cursor moves but cannot
click, or nothing moves at all, this is almost always why.

## How the aiming works

A webcam sees a narrow slice of the world, and your screen is wide. There is no
mapping from "which pixel is my fingertip on" to "which pixel of the screen" that
is both accurate and comfortable to use, so the aiming model is built in two
stages.

**1. Wrist-anchored pointing.** The fingertip is projected to a fixed distance
from the camera, measured in units of hand size and anchored at the wrist:

```
scale = |wrist − middle_knuckle|          # a rigid bone pair, so a depth proxy
u     = wrist + (index_tip − wrist) · (gain / scale)
```

Dividing by `scale` is what stops the cursor sliding when you lean towards the
screen: the whole hand changes apparent size, and the division cancels it.

`gain` is in **hand-lengths**. `(tip − wrist) / scale` is the fingertip's
distance from your wrist, and an index finger is about 1.1 hand-lengths long,
so:

| `gain` | meaning |
|---|---|
| `1.0` | target sits exactly on the fingertip — no gain at all |
| `0.5` | halfway back to the wrist; pointing *direction* dominates (default) |
| `> 1.5` | overshoots; the cursor jams into a corner and stops responding |

**2. Calibration.** A 3×3 homography maps camera space to your screen. Five
targets (four corners plus the centre); the four corners fit the homography and
the centre is a free accuracy check, because a mapping that nails the corners
and misses the middle feels terrible.

During calibration the preview draws a ring showing where in the camera frame to
point, and captures the sample once your fingertip holds steady on it. You get
feedback instead of having to trust yourself to aim at a physical bezel corner.

**3. Smoothing.** Two filters, because they fix different problems:

- a 3-frame **median**, which deletes outright the rare frame where MediaPipe
  emits a wildly wrong landmark (hand crossing a bright window, partial
  occlusion);
- a **1 Euro filter** (`min_cutoff` 1.2, `beta` 0.02), which widens its cutoff
  with speed. A fixed low-pass forces a choice between jitter and lag; the 1 Euro
  filter is heavily smoothed when you hold still and barely filtered when you
  whip across the screen. This single choice is most of the difference between
  "feels broken" and "feels good".

## Clicking

Extension is `|middle_tip − wrist| / |middle_pip − wrist|`. Above `extend_ratio`
the finger is out, below `fold_ratio` it is folded, and each must hold for
`confirm_frames` consecutive frames to be believed.

The two thresholds are deliberately different — a **hysteresis band**. Landmark
noise is easily ±0.05 on this ratio, so a single threshold makes the click
chatter open and shut whenever your finger rests near the boundary.

Two things make a stuck mouse button impossible:

- `max_hold_ms` (2 s) auto-releases and then refuses to re-press until you
  unfold, so a misdetected fold cannot leave a button down;
- `mouseUp` is force-sent when your hand is lost, when the tool is paused, and
  on exit, including `SIGINT` and `atexit`.

## The mouse is still yours

While this is running it owns the cursor, which would make your trackpad
useless. Instead of an event tap (which needs its own permission), the tool
compares the live cursor position against where it last put it every frame: if
they differ, a human moved the mouse, so the tool pauses immediately, drops the
stale position, and waits for the grace period plus a steady hand before
resuming.

## Tuning

Everything lives in [`config/default.toml`](config/default.toml) with the
reasoning inline. The keys that matter, all adjustable live while running:

| Key | Raise it when | Lower it when |
|---|---|---|
| `aim.gain` | you cannot reach the screen edges | the cursor feels twitchy |
| `aim.min_cutoff` | the cursor jitters at rest | it feels laggy when you move |
| `aim.beta` | fast moves trail behind | fast moves overshoot |
| `aim.median_window` | occasional wild jumps | legitimate fast motion gets eaten |
| `gesture.fold_ratio` | clicks fire by accident | clicks miss |

While `run` is going: `q` quit · `c` calibrate · `m` pause · `p` preview ·
`+`/`-` gain. Gain changes are live; put the number in the config to keep it.

**Watch the two dots in the preview.** The yellow crosshair is your fingertip,
the magenta dot is where the aiming model resolved it. Converging, steady dots
mean a working setup. If the magenta dot sits still while you move, the gain is
saturating — lower it.

## Commands

```
pnc doctor [--camera N] [--no-benchmark]
pnc calibrate [--camera N]
pnc run [--camera N] [--display N] [--gain G] [--no-preview]
         [--recalibrate] [--from-video FILE]
pnc record [--seconds S] [--output FILE]
```

`--display 0` is the main display, `1+` selects the nth active display.

`--from-video` replays a recording through the identical
landmark → aiming → smoothing path, so a bad session can be diagnosed after the
fact. `pnc record` captures one to debug with.

## Tests

```
.venv/bin/python -m pytest
```

61 tests, no camera or permissions required. They cover the aiming model's
depth-invariance, the click state machine's safety guarantees, the smoothing
filters, the homography fit, and the full frame loop driven by synthetic
landmarks against a fake cursor.

## Known limits

- **One hand, index finger.** If you point with a different finger, or use the
  whole hand as a pointer, it will not track that.
- **Reaching the corners depends on calibration.** A narrow webcam field of view
  means you stretch to hit the edges; calibration amplifies whatever range you
  use, so a small range means more jitter near the edges. If corners are
  unreachable, lower `gain` and recalibrate.
- **Lighting matters.** MediaPipe needs contrast around the hand. A plain
  background and light from the front work best.
- **One display at a time.** Not a virtual-desktop or multi-monitor spanning
  tool.