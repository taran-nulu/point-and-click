# point-and-click

Control your computer cursor by pointing at the screen with your hand.

Hold your hand up to the camera with your **index finger extended**, like a
laser pointer. Where the fingertip lands in the camera view drives the cursor.

To click, pick one of two gestures:

| `--click-method` | Gesture | Notes |
|---|---|---|
| `middle-finger` *(default)* | Fold your middle finger | Aiming never breaks; **supports drag** |
| `pinch` | Pinch thumb tip to index tip | More precise; latches the target; no drag |

```
./run.sh doctor      # check camera, permissions, throughput
./run.sh calibrate   # teach it where you point
./run.sh run                                  # middle-finger click
./run.sh run --click-method pinch             # pinch click
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

Both methods reduce to the same measurement: **a signal that falls when you
press.** The middle finger's extension ratio drops as it folds; the pinch gap
drops as your fingers close. So one state machine serves both.

| Method | Signal | Press when | Release when |
|---|---|---|---|
| `middle-finger` | `\|middle_tip − wrist\| / \|middle_pip − wrist\|` | below `fold_ratio` | above `extend_ratio` |
| `pinch` | `\|thumb_tip − index_tip\| / hand_scale` | below `press_ratio` | above `release_ratio` |

Normalising both by a hand-sized reference is what makes the thresholds
reusable at any distance from the camera.

The two thresholds are deliberately different — a **hysteresis band**. Landmark
noise is easily ±0.05 on these ratios, so a single threshold makes the click
chatter open and shut whenever your hand rests near the boundary.

### Why pinch latches, and middle-finger does not

Pinching curls the index finger — the very finger doing the aiming. Without
intervention, the cursor slides as you close the pinch and the click lands on
whatever drifted underneath it. So pinch mode **latches the cursor position**
the instant the press fires and holds it there until release. Pinching therefore
cannot drag; use the middle-finger method for that.

Middle-finger mode does not latch, because folding the middle finger only
nudges the index tip, so the cursor stays live while held — which is exactly
what makes click-drag work.

### The thresholds are estimates, and they are personal

The defaults in `config/default.toml` are reasoned guesses, not measurements,
and gesture ranges vary a lot between hands. Rather than guess, measure yours:

```
./run.sh doctor --gestures
```

Point normally, fold the middle finger, and pinch a few times while it counts.
It prints the observed range for both signals against the configured
thresholds, and tells you which way to move them if clicks will never fire or
never release. Paste the suggested numbers into the config.

### Nothing gets stuck

Two things make a stuck mouse button impossible:

- `max_hold_ms` (2 s) auto-releases and then refuses to re-press until you open
  the gesture, so a misdetected press cannot leave a button down;
- `mouseUp` is force-sent when your hand is lost, when the tool is paused, when
  you switch click methods mid-press, and on exit, including `SIGINT`,
  `SIGTERM` and `atexit`.

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
| `pinch.press_ratio` | pinches fire without pinching | pinches are missed |

While `run` is going: `q` quit · `c` calibrate · `m` pause · `p` preview ·
`x` switch click method · `+`/`-` gain. Gain and click-method changes are live;
put the numbers in the config to keep them.

The HUD shows `click <method> <signal>` — that number is the raw measurement
being thresholded. Watching it while you fold or pinch is the fastest way to
set the thresholds by hand.

**Watch the two dots in the preview.** The yellow crosshair is your fingertip,
the magenta dot is where the aiming model resolved it. Converging, steady dots
mean a working setup. If the magenta dot sits still while you move, the gain is
saturating — lower it. In pinch mode a bar is also drawn between the thumb and
index tips: red means the gap is closed enough to click.

## Commands

```
pnc doctor [--camera N] [--no-benchmark] [--gestures]
pnc calibrate [--camera N]
pnc run [--camera N] [--display N] [--gain G] [--no-preview]
         [--recalibrate] [--from-video FILE] [--click-method METHOD]
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

89 tests, no camera or permissions required. They cover the aiming model's
depth-invariance, both click signals, the click state machine's safety
guarantees, pinch latching vs. middle-finger dragging, the smoothing filters,
the homography fit, doctor's threshold advice, and the full frame loop driven
by synthetic landmarks against a fake cursor.

## Known limits

- **One hand, index finger.** If you point with a different finger, or use the
  whole hand as a pointer, it will not track that.
- **Pinch cannot drag**, by design — latching the target is what stops the
  cursor sliding out from under the click. Use `middle-finger` for drag.
- **Reaching the corners depends on calibration.** A narrow webcam field of view
  means you stretch to hit the edges; calibration amplifies whatever range you
  use, so a small range means more jitter near the edges. If corners are
  unreachable, lower `gain` and recalibrate.
- **Click thresholds start as guesses.** They are hand-specific; run
  `pnc doctor --gestures` and set them from your own measured range.
- **Lighting matters.** MediaPipe needs contrast around the hand. A plain
  background and light from the front work best.
- **One display at a time.** Not a virtual-desktop or multi-monitor spanning
  tool.