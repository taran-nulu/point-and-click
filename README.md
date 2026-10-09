# point-and-click

Control your computer cursor by pointing at the screen with your hand.

Hold your hand up to the camera with your **index finger extended**, like a
laser pointer. Where the fingertip lands in the camera view drives the cursor.

Two things are configurable, because which feels best is personal:

| | Options | Flag |
|---|---|---|
| **How to click** | `middle-finger` (default, supports drag), `pinch` | `--click-method` |
| **What to pivot about** | `wrist` (default), `elbow` *(does not work well — see below)* | `--pivot` |

> ### ⚠️ The elbow pivot does not work well
>
> `--pivot elbow` is **known to perform poorly in practice.** It was built on a
> sound geometric argument and it does everything it was designed to do
> mechanically — the measurements back that up — but in the hand it feels
> jittery and imprecise. **Use the default `wrist` pivot.** Treat `elbow` as an
> experiment that did not pan out, not as a usable option.
>
> If you want to try it anyway, `b` switches pivots live so you can compare
> without restarting. The reasoning behind it, and the most likely reason it
> fails, are in [Why the elbow pivot underperforms](#why-the-elbow-pivot-underperforms).

```bash
./run.sh doctor              # check camera, permissions, throughput
./run.sh doctor --gestures   # measure your click gesture thresholds
./run.sh calibrate           # teach it where you point
./run.sh run                                  # wrist pivot, middle-finger click
./run.sh run --click-method pinch             # pinch click
./run.sh run --pivot elbow                    # experimental, see the warning above
```

`run.sh` creates a virtualenv and installs everything on first run. Requires
macOS with Python 3.11+, plus **Camera** and **Accessibility** permissions in
System Settings → Privacy & Security.

While `run` is going: `q` quit · `c` calibrate · `m` pause · `p` preview ·
`b` switch pivot · `x` switch click method · `+`/`-` gain.

## Other commands

```bash
./run.sh run --display 1        # use the 2nd active display
./run.sh record --seconds 10 --output clip.mp4   # capture a clip to debug with
./run.sh run --from-video clip.mp4               # replay a clip
```

## How the aiming works

A webcam sees a narrow slice of the world, and your screen is wide. There is no
mapping from "which pixel is my fingertip on" to "which pixel of the screen"
that is both accurate and comfortable, so aiming is built in three stages.

### 1. The pointing model

The fingertip is projected along a ray from a pivot point, to a fixed distance
from the camera measured in units of hand size:

```
scale  = |wrist − middle_knuckle|        # a rigid bone pair, so a depth proxy
target = pivot + (index_tip − pivot) · (gain / scale)
```

Dividing by `scale` is what stops the cursor sliding when you lean towards the
screen: the whole hand changes apparent size, and the division cancels it.

`gain` is in **hand-lengths**, because `(tip − pivot) / scale` is the
fingertip's distance from the pivot measured in hand-lengths. That makes it easy
to overshoot by an order of magnitude, so for reference: a `gain` of 1.0 lands
the target exactly on the fingertip (no gain at all), and anything above ~1.5
overshoots so badly that the cursor jams into a screen corner and stops
responding. Wrist pivot uses **0.5**.

### 2. Pivot point: wrist or elbow

**In practice the elbow pivot performs poorly and you should leave it alone.**
What follows is the design as built, kept because it explains why the elbow was
expected to help — and [why it doesn't](#why-the-elbow-pivot-underperforms).

- **`wrist`** — rotating your wrist aims. Your forearm barely contributes,
  because rotating it about the elbow hardly moves your wrist.
- **`elbow`** — rotating your forearm aims. That is a larger, more comfortable
  range of motion than wrist rotation.

There is a subtlety worth stating, because it is counter-intuitive and it
determines the gain:

> **Translating your hand moves the cursor by the same amount either way.** Since
> `target = pivot + (tip − pivot)·k`, shifting the tip by Δ shifts the target by
> exactly `Δ·k` — the pivot terms cancel. So the pivot choice has no effect on
> how responsive the cursor is to translating your hand around.

The pivots differ only in how they respond to **rotation**. Rotating your
forearm about the elbow swings the fingertip through a long arc; the elbow pivot
amplifies that by the full lever length, the wrist pivot follows only weakly.
Measured with realistic arm proportions, the elbow pivot swings **~2.5× further
for the same 20° forearm rotation**, which is why `gain_elbow` is set to
`0.5 / 2.5 ≈ 0.2` — the value that makes an elbow pivot feel like the wrist
pivot did. `tests/test_pivot.py` pins that relationship down.

Every one of those numbers is correct, and the feature still does not work well.
See [Why the elbow pivot underperforms](#why-the-elbow-pivot-underperforms).

**What the elbow pivot does not do is give you more reach.** Reach comes from
`gain` and calibration, and both trade against jitter. The elbow was meant to
buy ergonomics, not range.

Implementation notes, for anyone maintaining it rather than using it:

- Elbow tracking needs a second model, since MediaPipe's hand landmarker has no
  arm landmarks: a 5.8 MB pose model downloaded on first use, plus ~8 ms
  inference. That cost **no framerate** on this machine — the camera tops out
  near 33 fps and hand+pose inference (~20 ms) still fits the budget.
- The elbow is only trusted above `arm.min_visibility`. MediaPipe keeps
  *emitting* elbow coordinates when your shoulders are out of frame, where they
  stop meaning anything; below the threshold it reports nothing and the tool
  falls back to the wrist. The HUD shows which pivot is live.
- Switching pivot mid-drag would teleport the cursor, so while a button is held
  the choice is frozen, dropout takes several consecutive frames, and the
  smoother resets on a switch.
- Calibration records which pivot it was fitted under and `run` warns on a
  mismatch. Mixing pivots within one calibration is refused.

### 3. Calibration

A 3×3 homography maps camera space to your screen, from five targets (four
corners plus the centre). The four corners fit the homography; the centre is a
free accuracy check, because a mapping that nails the corners and misses the
middle feels terrible.

During calibration the preview draws a ring showing where in the camera frame to
point, and captures the sample once your fingertip holds steady on it. You get
feedback instead of having to trust yourself to aim at a physical bezel corner.

Hold your hand in the pose you will actually *click* with — a slightly folded
middle finger, or a pinch, nudges the index fingertip, and calibration has to
account for that.

**Calibration is the fix for not being able to reach the screen corners.** The
uncalibrated mapping preserves the camera's field of view; the homography
instead stretches whatever range you actually use to fill the screen. If corner
reach is your problem, calibrate and raise `gain` — that is what those two are
for. The elbow pivot cannot help with it.

## Why the elbow pivot underperforms

This is the section to read before spending time on `--pivot elbow`. The
short version: **the elbow pivot amplifies a motion the camera can barely see.**

The elbow pivot's only advantage over the wrist pivot is that it responds more
to forearm rotation — that was the entire point, since wrist rotation is a small
and uncomfortable range of motion. That advantage is real in the geometry, and
`tests/test_pivot.py` measures it: ~2.5× the cursor travel for the same 20° of
forearm rotation.

The problem is what happens between that ideal and a webcam pointed at you from
the top of a screen:

1. **The amplified motion is foreshortened.** Pointing at a screen means your
   forearm points roughly *at the camera*. Rotation around the elbow is then
   mostly a rotation about the viewing axis, which projects onto the image as
   very little movement. So the 4.6× lever is multiplying a signal that is
   already small — and the errors along with it.
2. **The lever amplifies noise, not just motion.** `tip − elbow` is a long
   vector, so any error in the estimated elbow position is scaled up by it. The
   hand landmarker tracks fingertips on a high-contrast object a few centimetres
   from the lens; MediaPipe's pose estimate of an elbow, partly occluded by your
   own torso and often at the frame edge, is much worse. The gain that makes the
   useful signal bigger makes the estimation noise bigger too.
3. **Those two compound.** More gain to compensate for a weak signal also
   amplifies a noisy anchor, so the net result is a cursor that drifts and
   twitches rather than one that aims confidently.

There is also a plain mismatch of intent: the elbow pivot was reached for to fix
*reach*, and it cannot fix reach at all. Reach comes from `gain` and
calibration. The elbow pivot could never have solved the problem it was added
for.

**To be clear about confidence:** items 1–3 are an explanation consistent with
the observed behaviour and with the geometry, not a diagnosis measured from your
hand. I have no instrumented recording of the failure. But the empirical result
is the thing to act on, and it is unambiguous: keep `pivot = "wrist"`.

If the elbow pivot is ever to be rescued rather than abandoned, the direction is
to stop amplifying the raw elbow and instead **estimate the forearm axis
directly** — fit a line through elbow, wrist and index tip and pivot about a
*damped* point on it, with the damping on your arm's actual motion. That trades
some of the ergonomic benefit for far less noise amplification. That work has
not been done.

## Clicking

Both methods reduce to the same measurement: **a signal that falls when you
press.** The middle finger's extension ratio drops as it folds; the pinch gap
drops as your fingers close. So one state machine serves both.

| Method | Signal | Press when | Release when |
|---|---|---|---|
| `middle-finger` | `\|middle_tip − wrist\| / \|middle_pip − wrist\|` | below `fold_ratio` | above `extend_ratio` |
| `pinch` | `\|thumb_tip − index_tip\| / hand_scale` | below `press_ratio` | above `release_ratio` |

Normalising both by a hand-sized reference is what makes the thresholds reusable
at any distance from the camera.

The two thresholds are deliberately different — a **hysteresis band**. Landmark
noise is easily ±0.05 on these ratios, so a single threshold makes the click
chatter open and shut whenever your hand rests near the boundary.

**Pinch latches the cursor, middle-finger does not.** Pinching curls the index
finger — the very finger doing the aiming — so the cursor would slide as you
close the pinch and the click would land on whatever drifted underneath. So
pinch mode latches the cursor position when the press fires. The cost is that
pinch cannot drag; use the middle-finger method for that. The latch is taken at
the press, so the small drift from your fingers closing is unavoidable — what is
guaranteed is that the cursor stops moving for the rest of the click.

### The thresholds are estimates, and they are personal

The defaults are reasoned guesses, not measurements, and gesture range varies a
lot between hands. Measure yours rather than guessing:

```
./run.sh doctor --gestures
```

Point normally, fold the middle finger, and pinch a few times while it counts.
It prints the observed range for both signals against the configured
thresholds, and tells you which way to move them if clicks would never fire or
never release.

### Nothing gets stuck

- `max_hold_ms` (2 s) auto-releases and then refuses to re-press until you open
  the gesture, so a misdetected press cannot leave a button down;
- `mouseUp` is force-sent when your hand is lost, when the tool is paused, when
  you switch click method or pivot mid-press, and on exit, including `SIGINT`,
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
reasoning inline.

| Key | Raise it when | Lower it when |
|---|---|---|
| `aim.gain` | you cannot reach the screen edges | the cursor feels twitchy |
| `aim.gain_elbow` | elbow pivot too slow | elbow pivot too twitchy |
| `aim.min_cutoff` | the cursor jitters at rest | it feels laggy when you move |
| `aim.beta` | fast moves trail behind | fast moves overshoot |
| `aim.median_window` | occasional wild jumps | legitimate fast motion gets eaten |
| `gesture.fold_ratio` | clicks fire by accident | clicks miss |
| `pinch.press_ratio` | pinches fire without pinching | pinches are missed |
| `arm.min_visibility` | elbow jumps around | it falls back to the wrist too eagerly |
| `arm.side` | `auto` picks the wrong arm | — |

The HUD shows `click <method> <signal>` — the raw measurement being thresholded.
Watching it while you fold or pinch is the fastest way to set thresholds by hand.

**Watch the two dots in the preview.** The yellow crosshair is your fingertip,
the magenta dot is where the aiming model resolved it. Converging, steady dots
mean a working setup. If the magenta dot sits still while you move, the gain is
saturating — lower it. In pinch mode a bar is drawn between thumb and index:
red means closed enough to click.

## Tests

```
.venv/bin/python -m pytest
```

130 tests, no camera or permissions required. They cover the aiming model's
depth-invariance, both pivot geometries and the gain that matches them, both
click signals, the click state machine's safety guarantees, pinch latching vs.
middle-finger dragging, elbow fallback and dropout latching, the smoothing
filters, the homography fit, doctor's threshold advice, and the full frame loop
driven by synthetic landmarks against a fake cursor.

## Known limits

- **One hand, index finger.** If you point with a different finger, or use the
  whole hand as a pointer, it will not track that.
- **Pinch cannot drag**, by design — latching is what stops the cursor sliding
  out from under the click. Use `middle-finger` for drag.
- **The elbow pivot does not work well.** It is retained for reference and left
  as a documented dead end — see
  [Why the elbow pivot underperforms](#why-the-elbow-pivot-underperforms). Do not
  expect it to be usable, and do not reach for it to solve reach: use `gain` and
  calibration, which can.
- **Reaching the corners depends on calibration.** A narrow webcam field of view
  means you stretch to hit the edges; calibration amplifies whatever range you
  use, so a small range means more jitter near the edges.
- **Click thresholds start as guesses.** They are hand-specific; run
  `pnc doctor --gestures` and set them from your own measured range.
- **Lighting matters.** MediaPipe needs contrast around the hand. A plain
  background and light from the front work best.
- **One display at a time.** Not a virtual-desktop or multi-monitor spanning
  tool.