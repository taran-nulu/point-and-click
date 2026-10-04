"""Command line entry point for point-and-click."""

from __future__ import annotations

import argparse
import sys

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pnc",
        description="Control your cursor by pointing at the screen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "getting started:\n"
            "  ./run.sh doctor      check camera, permissions and throughput\n"
            "  ./run.sh calibrate   teach it where you point\n"
            "  ./run.sh run         point your index finger at the screen\n"
        ),
    )
    parser.add_argument("--config", help="path to a config TOML (default config/default.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="control the cursor")
    run.add_argument("--camera", type=int, help="camera index (default 0)")
    run.add_argument("--display", type=int, default=0, help="0 = main display, 1+ = nth display")
    run.add_argument("--gain", type=float, help="aiming gain override")
    run.add_argument("--no-preview", action="store_true", help="hide the camera preview")
    run.add_argument("--recalibrate", action="store_true", help="calibrate before starting")
    run.add_argument("--from-video", help="replay a recording instead of using the camera")

    calibrate = sub.add_parser("calibrate", help="interactive five-target calibration")
    calibrate.add_argument("--camera", type=int, help="camera index (default 0)")

    doctor = sub.add_parser("doctor", help="diagnose camera, permissions and throughput")
    doctor.add_argument("--camera", type=int, default=0)
    doctor.add_argument("--no-benchmark", action="store_true", help="skip the live camera check")

    record = sub.add_parser("record", help="record the camera to a file for offline debugging")
    record.add_argument("--camera", type=int, help="camera index (default 0)")
    record.add_argument("--seconds", type=float, default=10.0)
    record.add_argument("--output", default="state/hand.mp4")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "doctor":
        from .doctor import main as doctor_main

        return doctor_main(camera_index=args.camera, benchmark=not args.no_benchmark)

    if args.command == "record":
        return _record(cfg, args)

    if args.camera is not None:
        cfg.camera.index = args.camera

    # Imported late so `pnc doctor` still works if the model or camera misbehave.
    from .app import App

    if args.command == "calibrate":
        app = App(cfg, display_index=0, preview=True, recalibrate=True)
    else:
        app = App(
            cfg,
            display_index=args.display,
            gain=args.gain,
            preview=False if args.no_preview else None,
            video_path=args.from_video,
            recalibrate=args.recalibrate,
        )

    try:
        return app.run()
    except KeyboardInterrupt:
        print("\ninterrupted")
        app._shutdown()
        return 130


def _record(cfg, args) -> int:
    """Save a clip of the camera, so a bad session can be replayed offline."""
    from pathlib import Path

    import cv2

    from .camera import Camera

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Camera(index=cfg.camera.index, width=cfg.camera.width, height=cfg.camera.height) as cam:
        fps = max(1, int(cfg.camera.fps))
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, cam.actual_size
        )
        if not writer.isOpened():
            print(f"Could not open {output} for writing.")
            return 1

        frames = int(args.seconds * fps)
        print(f"Recording {args.seconds:.0f}s to {output} - press Ctrl-C to stop early.")
        try:
            for _ in range(frames):
                ok, frame = cam.read()
                if not ok:
                    break
                writer.write(frame)
        except KeyboardInterrupt:
            print("\nstopped early")
        finally:
            writer.release()

    print(f"Saved {output}. Replay it with:  pnc run --from-video {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())