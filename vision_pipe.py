#!/usr/bin/env python3
import argparse
import os.path
import sys

VERSION = "vision_pipe 1.0"

EXT_IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
EXT_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


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
    p.add_argument("--input", "-i", dest="input_flag", default=None,
                   help="Input image or video file")
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


def ext_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in EXT_IMAGES:
        return 'image'
    if ext in EXT_VIDEOS:
        return 'video'
    return None


def main() -> int:
    parser = build_parser()
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()

    if not args.input and not args.input_flag:
        sys.stderr.write("input required\n")
        return 2
    if args.input and args.input_flag:
        sys.stderr.write("give input either positionally or via --input, not both\n")
        return 2
    path = args.input or args.input_flag

    if not os.path.exists(path):
        sys.stderr.write(f"file not found: {path}\n")
        return 2

    kind = ext_type(path)
    if kind is None:
        ext = os.path.splitext(path)[1].lower()
        sys.stderr.write(
            f"unsupported format: {ext}. "
            f"Supported images: {sorted(EXT_IMAGES)} "
            f"Supported videos: {sorted(EXT_VIDEOS)}\n"
        )
        return 2

    sys.stderr.write(f"{kind}\n")
    sys.stderr.write("not yet implemented\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())