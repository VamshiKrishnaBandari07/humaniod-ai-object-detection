"""Command line for the spatial change comparison."""

from __future__ import annotations

import argparse

from spatial_change.pipeline import run_comparison
from spatial_change.synth import generate_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="spatial-change",
        description="Build a shared 3D scene from two phone recordings and describe what moved, appeared, or disappeared.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Render a synthetic room and run the full comparison")
    demo.add_argument("--out", default="runs/demo", help="Directory for videos, the report, and the viewer")
    demo.add_argument("--frames", type=int, default=12, help="Frames in each synthetic recording")

    run = commands.add_parser("run", help="Compare two recordings of the same small indoor area")
    run.add_argument("--before", required=True, help="First recording, the earlier visit")
    run.add_argument("--after", required=True, help="Second recording, the later visit")
    run.add_argument("--out", required=True, help="Directory for the report, point clouds, and viewer")
    run.add_argument("--intrinsics", default=None, help="JSON with fx, fy, cx, cy, width, height")
    run.add_argument("--fov", type=float, default=62.0, help="Vertical field of view in degrees, used when intrinsics are omitted")
    run.add_argument("--max-frames", type=int, default=12, help="Frames sampled from each video")
    run.add_argument("--gt", default=None, help="Optional ground-truth JSON, scored by change type and colour")

    ui = commands.add_parser("ui", help="Open a browser page to upload two recordings and compare them")
    ui.add_argument("--port", type=int, default=8765, help="Local port")
    ui.add_argument("--no-browser", action="store_true", help="Do not open a browser window")

    args = parser.parse_args(argv)
    if args.command == "ui":
        from spatial_change.server import serve
        serve(port=args.port, open_browser=not args.no_browser)
        return
    if args.command == "demo":
        print("Rendering a synthetic room with one moved, one removed, and one added object...")
        dataset = generate_dataset(args.out, frames=args.frames)
        run_comparison(
            dataset["before"],
            dataset["after"],
            args.out,
            intrinsics_path=dataset["intrinsics"],
            max_frames=args.frames,
            ground_truth=dataset["ground_truth"],
        )
        return
    run_comparison(
        args.before,
        args.after,
        args.out,
        intrinsics_path=args.intrinsics,
        fov_deg=args.fov,
        max_frames=args.max_frames,
        ground_truth=args.gt,
    )


if __name__ == "__main__":
    main()
