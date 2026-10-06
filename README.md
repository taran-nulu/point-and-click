# point-and-click

## Commands

```bash
./run.sh doctor      # check camera, permissions, throughput
./run.sh doctor --gestures   # measure your click gesture thresholds
./run.sh calibrate   # teach it where you point
./run.sh run                                  # middle-finger click (supports drag)
./run.sh run --click-method pinch             # pinch click
```

`run.sh` creates a virtualenv and installs everything on first run. Requires
macOS with Python 3.11+, plus **Camera** and **Accessibility** permissions in
System Settings → Privacy & Security.

## Other commands

```bash
./run.sh run --display 1        # use the 2nd active display
./run.sh record --seconds 10 --output clip.mp4   # capture a clip to debug with
./run.sh run --from-video clip.mp4               # replay a clip
```

While `run` is going: `q` quit · `c` calibrate · `m` pause · `p` preview ·
`x` switch click method · `+`/`-` gain.
