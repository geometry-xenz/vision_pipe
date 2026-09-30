#!/usr/bin/env python3
import argparse
import sys

VERSION = "vision_pipe 1.0"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vision_pipe",
        description="Detect objects in images and videos using a local ONNX model.",
    )
    p.add_argument(
        "input",
        nargs="?",
        help="Input image or video file",
    )
    p.add_argument(
        "--input", "-i",
        dest="input_explicit",
        default=None,
        help="Input image or video file",
    )
    p.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract from video")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0-1.0)")
    p.add_argument("--classes", nargs="*", default=None, metavar="CLASS",
                   help="Space-separated class names to filter")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p.add_argument("--text-only", "-t", action="store_true",
                   help="Plain English output instead of JSON")
    p.add_argument("--model", default="yolov8n",
                   help="Model name or path (for future extension)")
    return p


def main() -> int:
    parser = build_parser()
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()
    sys.stderr.write("not yet implemented\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())