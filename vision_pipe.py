#!/usr/bin/env python3
import argparse
import json
import os.path
import sys
import threading

VERSION = "vision_pipe 1.0"

EXT_IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
EXT_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models', 'yolov8n.onnx'
)

_quiet = False

COCO_NAMES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
)


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
    p.add_argument("--model", default=None,
                   help="Path to ONNX model file. If omitted, uses bundled default.")
    p.add_argument("--caption", action="store_true",
                   help="Use Florence-2 caption model instead of YOLO detection")
    p.add_argument("--palette-only", action="store_true",
                   help="Output only the color palette (L0 pixel layer)")
    p.add_argument("--ocr-only", action="store_true",
                   help="Output only the OCR text (L1 OCR layer)")
    p.add_argument("--detections-only", action="store_true",
                   help="Output only YOLO detections (L2; default behavior)")
    p.add_argument("--no-ocr", action="store_true",
                   help="Skip OCR layer even in default mode")
    p.add_argument("--no-pixel", action="store_true",
                   help="Skip pixel layer even in default mode")
    return p


def ext_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in EXT_IMAGES:
        return 'image'
    if ext in EXT_VIDEOS:
        return 'video'
    return None


def _palette_kmeans(small_bgr, k=5):
    import cv2
    import numpy as np
    arr = small_bgr.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        arr, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    total = int(counts.sum())
    out = []
    for i in range(k):
        b, g, r = (int(v) for v in centers[i])
        out.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "pct": round(counts[i] / total, 3),
        })
    out.sort(key=lambda c: -c["pct"])
    return out


def analyze_image(path):
    import cv2
    import numpy as np
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    small = cv2.resize(img, (256, 256))
    arr = small.astype(np.float32)
    mean_bgr = arr.reshape(-1, 3).mean(axis=0)
    std_bgr = arr.reshape(-1, 3).std(axis=0)
    brightness = float(mean_bgr.mean() / 255.0)
    is_gray = bool(std_bgr.max() < 10)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edge_density = float(cv2.Sobel(gray, cv2.CV_32F, 1, 1).var() / 10000.0)
    return {
        "width": w,
        "height": h,
        "mean_rgb": [round(float(v), 1) for v in mean_bgr[::-1]],
        "std_rgb": [round(float(v), 1) for v in std_bgr[::-1]],
        "brightness": round(brightness, 3),
        "is_grayscale": is_gray,
        "edge_density": round(edge_density, 3),
        "palette": _palette_kmeans(small),
    }


def extract_text(path):
    import shutil
    import subprocess
    if not shutil.which("tesseract"):
        return None
    r = subprocess.run(
        ["tesseract", path, "-", "-l", "eng"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    words = r.stdout.split()
    text = " ".join(words)
    return {
        "text": text,
        "word_count": len(words),
        "language": "eng",
        "is_text_heavy": len(words) >= 50,
    }


def synthesize_summary(pixel, ocr, detections):
    parts = []
    if pixel:
        parts.append(f"{pixel['width']}x{pixel['height']}")
        parts.append(f"bright {pixel['brightness']:.2f}")
        if pixel["is_grayscale"]:
            parts.append("grayscale")
        elif pixel["palette"]:
            parts.append(f"dominant {pixel['palette'][0]['hex']}")
    if ocr:
        n = ocr["word_count"]
        parts.append("no text" if n == 0 else f"{n} words")
    if detections:
        counts = detections.get("counts", {})
        if counts:
            inv = ", ".join(f"{counts[c]} {c}" for c in sorted(counts))
            parts.append(f"detected {inv}")
    return ", ".join(parts) + "." if parts else ""


CAPTION_MODEL_ID = "microsoft/Florence-2-base-ft"
_caption_model = None
_caption_processor = None


def _venv_python():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '.venv', 'bin', 'python'
    )


def _load_caption_model():
    global _caption_model, _caption_processor
    if _caption_model is None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as e:
            venv_py = _venv_python()
            if os.path.exists(venv_py) and sys.executable != venv_py:
                sys.stderr.write(f'relaunching with {venv_py}\n')
                os.execv(venv_py, [venv_py] + sys.argv)
            sys.stderr.write(
                '--caption requires: pip install transformers torch sentencepiece einops Pillow timm\n'
                f'missing module: {e.name}\n'
            )
            sys.exit(1)
        if not _quiet:
            sys.stderr.write(f"loading {CAPTION_MODEL_ID}...\n")
        _caption_processor = AutoProcessor.from_pretrained(
            CAPTION_MODEL_ID, trust_remote_code=True
        )
        _caption_model = AutoModelForCausalLM.from_pretrained(
            CAPTION_MODEL_ID, trust_remote_code=True
        ).eval()
    return _caption_model, _caption_processor


def caption_image(path):
    model, processor = _load_caption_model()
    from PIL import Image
    import torch
    img = Image.open(path).convert("RGB")
    w, h = img.size
    task = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task, images=img, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
        )
    raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw, task=task, image_size=(w, h)
    )
    return {
        "tool": "vision_pipe",
        "version": "1.0",
        "input": path,
        "type": "image",
        "width": w,
        "height": h,
        "model": CAPTION_MODEL_ID,
        "caption": parsed[task].strip(),
    }


def ensure_model(override_path=None) -> str:
    if override_path is not None:
        if os.path.exists(override_path):
            return os.path.abspath(override_path)
        sys.stderr.write(f"model not found: {override_path}\n")
        sys.exit(2)
    if os.path.exists(MODEL_PATH):
        return os.path.abspath(MODEL_PATH)
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.stderr.write('Model not found and ultralytics not installed. Run: pip install ultralytics onnx\n')
        sys.exit(1)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not _quiet:
        sys.stderr.write('exporting model...\n')
    YOLO('yolov8n.pt').export(format='onnx')
    if not os.path.exists(MODEL_PATH):
        produced = os.path.join(os.getcwd(), 'yolov8n.onnx')
        if os.path.exists(produced):
            os.replace(produced, MODEL_PATH)
    return os.path.abspath(MODEL_PATH)


def detect_duration(path) -> float:
    import subprocess
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        try:
            return float(r.stdout.strip())
        except ValueError:
            pass
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0 and n > 0:
            return float(n) / float(fps)
    finally:
        cap.release()
    return 0.0


def extract_frames(path, fps):
    import subprocess
    from subprocess import PIPE
    import cv2
    import numpy as np

    proc = subprocess.Popen(
        ['ffmpeg', '-loglevel', 'error', '-i', path, '-vf', f'fps={fps}',
         '-f', 'image2pipe', '-vcodec', 'mjpeg', '-'],
        stdout=PIPE, stderr=PIPE, bufsize=10 ** 8,
    )
    err_chunks = []

    def _drain_err():
        try:
            while True:
                c = proc.stderr.read(4096)
                if not c:
                    break
                err_chunks.append(c)
        finally:
            try:
                proc.stderr.close()
            except Exception:
                pass

    t = threading.Thread(target=_drain_err, daemon=True)
    t.start()

    try:
        buf = bytearray()
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                start = buf.find(b'\xff\xd8')
                if start == -1:
                    break
                end = buf.find(b'\xff\xd9', start + 2)
                if end == -1:
                    break
                jpeg = bytes(buf[:end + 2])
                del buf[:end + 2]
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        proc.wait()
        t.join(timeout=1.0)
        if proc.returncode != 0:
            err = b''.join(err_chunks).decode('utf-8', errors='replace').strip()
            if err:
                sys.stderr.write(err + '\n')
            sys.exit(1)


def preprocess(frame, target=640):
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    scale = target / max(h, w)
    new_h = min(target, int(round(h * scale)))
    new_w = min(target, int(round(w * scale)))
    pad_h = target - new_h
    pad_w = target - new_w

    resized = cv2.resize(frame, (new_w, new_h))
    padded = cv2.copyMakeBorder(
        resized, 0, pad_h, 0, pad_w,
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    rgb = padded[..., ::-1]
    tensor = (
        rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    )
    tensor = np.ascontiguousarray(tensor)
    return tensor, scale


def infer(session, frame, conf=0.25, classes=None):
    tensor, scale = preprocess(frame)
    inv = 1.0 / scale
    out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    rows = out[0].T
    keep = None
    if classes is not None:
        keep = {c.lower() for c in classes}

    dets = []
    for row in rows:
        probs = row[4:]
        max_conf = float(probs.max())
        if max_conf < conf:
            continue
        class_id = int(probs.argmax())
        name = COCO_NAMES[class_id]
        if keep is not None and name.lower() not in keep:
            continue
        xc, yc, bw, bh = row[0], row[1], row[2], row[3]
        x1 = float((xc - bw / 2) * inv)
        y1 = float((yc - bh / 2) * inv)
        x2 = float((xc + bw / 2) * inv)
        y2 = float((yc + bh / 2) * inv)
        dets.append({
            'class': name,
            'confidence': max_conf,
            'bbox': [x1, y1, x2, y2],
        })
    return dets


def nms(dets, iou_thresh=0.45):
    if not dets:
        return []
    by_class = {}
    for d in dets:
        by_class.setdefault(d['class'], []).append(d)
    out = []
    for group in by_class.values():
        group.sort(key=lambda d: d['confidence'], reverse=True)
        kept = []
        for d in group:
            x1, y1, x2, y2 = d['bbox']
            ok = True
            for k in kept:
                kx1, ky1, kx2, ky2 = k['bbox']
                iw = max(0.0, min(x2, kx2) - max(x1, kx1))
                ih = max(0.0, min(y2, ky2) - max(y1, ky1))
                inter = iw * ih
                a1 = (x2 - x1) * (y2 - y1)
                a2 = (kx2 - kx1) * (ky2 - ky1)
                union = a1 + a2 - inter
                if union > 0 and inter / union >= iou_thresh:
                    ok = False
                    break
            if ok:
                kept.append(d)
        out.extend(kept)
    return out


def load_session(model_path):
    import onnxruntime as ort
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])


def _round_det(d):
    return {
        'class': d['class'],
        'confidence': round(d['confidence'], 2),
        'bbox': [int(round(v)) for v in d['bbox']],
    }


def aggregate(timeline, fps):
    counts = {}
    first_idx = {}
    last_idx = {}
    for entry in timeline:
        idx = entry['frame']
        for obj in entry['objects']:
            cls = obj['class']
            counts[cls] = counts.get(cls, 0) + 1
            if cls not in first_idx:
                first_idx[cls] = idx
            last_idx[cls] = idx
    classes_detected = sorted(counts.keys())
    inv = 1.0 / fps if fps else 0.0
    summary = {
        cls: {
            'count': counts[cls],
            'first_seen': first_idx[cls] * inv,
            'last_seen': last_idx[cls] * inv,
        }
        for cls in classes_detected
    }
    return classes_detected, summary


def format_json(d):
    return json.dumps(d, indent=2)


def _format_text_image(d):
    counts = {}
    for obj in d['objects']:
        counts[obj['class']] = counts.get(obj['class'], 0) + 1
    lines = [f"Image analysis ({d['width']}x{d['height']}):"]
    for cls in sorted(counts):
        lines.append(f"  - {cls}: {counts[cls]} detections")
    return "\n".join(lines)


def _format_text_video(d):
    n = d['total_frames']
    fps = d['fps']
    dur = d['duration_sec']
    lines = [
        f"Video analysis ({n} frames at {fps} fps, {dur}s total):",
        "",
        "Detections by class:",
    ]
    for cls in d['classes_detected']:
        s = d['summary'][cls]
        lines.append(
            f"  - {cls}: {s['count']} total, "
            f"first seen at {s['first_seen']}s, last seen at {s['last_seen']}s"
        )
    lines.append("")
    lines.append("Notable frames (2+ objects):")
    notable = [e for e in d['timeline'] if len(e['objects']) >= 2]
    cap = 10
    truncated = len(notable) > cap
    if truncated:
        notable = notable[:cap]
    for e in notable:
        classes = ", ".join(o['class'] for o in e['objects'])
        lines.append(f"  @ {e['timestamp']}s: {classes}")
    if truncated:
        lines.append("  ...")
    return "\n".join(lines)


def format_text(d):
    if d['type'] == 'image':
        return _format_text_image(d)
    return _format_text_video(d)


def main() -> int:
    parser = build_parser()
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()

    global _quiet
    _quiet = args.quiet

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

    if args.caption:
        if kind != 'image':
            sys.stderr.write("--caption supports images only (Florence-2 is an image model)\n")
            return 2
        payload = caption_image(path)
        print(payload["caption"] if args.text_only else format_json(payload))
        return 0

    only_modes = sum([args.palette_only, args.ocr_only, args.detections_only])
    if only_modes > 1:
        sys.stderr.write("pick only one of --palette-only, --ocr-only, --detections-only\n")
        return 2

    if args.palette_only:
        payload = _run_palette_only(path)
        print(format_json(payload))
        return 0
    if args.ocr_only:
        payload = _run_ocr_only(path)
        print(format_json(payload))
        return 0
    if args.detections_only:
        model_path = ensure_model(args.model)
        session = load_session(model_path)
        if kind == 'image':
            return run_image(path, session, args)
        return run_video(path, session, args)

    if kind == 'image':
        return run_image_full(path, args)
    if kind == 'video':
        model_path = ensure_model(args.model)
        session = load_session(model_path)
        return run_video(path, session, args)
    sys.stderr.write("not yet implemented\n")
    return 1


def _run_palette_only(path):
    pixel = analyze_image(path)
    return {
        "tool": "vision_pipe",
        "version": "1.2",
        "input": path,
        "type": "image",
        "pixel": pixel,
    }


def _run_ocr_only(path):
    ocr = extract_text(path)
    return {
        "tool": "vision_pipe",
        "version": "1.2",
        "input": path,
        "type": "image",
        "ocr": ocr,
    }


def run_image_full(path, args) -> int:
    pixel = None if args.no_pixel else analyze_image(path)
    ocr = None if args.no_ocr else extract_text(path)
    dets, w, h = _detect_image(path, args)
    counts = _class_counts(dets)
    payload = {
        "tool": "vision_pipe",
        "version": "1.2",
        "input": path,
        "type": "image",
        "width": w,
        "height": h,
        "pixel": pixel,
        "ocr": ocr,
        "detections": {"model": "yolov8n", "objects": dets, "counts": counts},
    }
    payload["summary"] = synthesize_summary(pixel, ocr, payload["detections"])
    if args.text_only:
        print(payload["summary"])
    else:
        print(format_json(payload))
    return 0


def _detect_image(path, args):
    import cv2
    model_path = ensure_model(args.model)
    session = load_session(model_path)
    img = cv2.imread(path)
    if img is None:
        sys.stderr.write(f"could not read image: {path}\n")
        sys.exit(2)
    h, w = img.shape[:2]
    sys.stderr.write(f"image: {w}x{h}\n")
    dets = nms(infer(session, img, args.conf, args.classes))
    norm = [_round_det(d) for d in dets]
    for d in norm:
        x1, y1, x2, y2 = d["bbox"]
        sys.stderr.write(
            f"class {d['class']} conf={d['confidence']:.2f} "
            f"bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]\n"
        )
    sys.stderr.write(f"image: {len(norm)} detections\n")
    return norm, w, h


def _class_counts(dets):
    counts = {}
    for d in dets:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    return counts


def run_image(path, session, args) -> int:
    import cv2
    img = cv2.imread(path)
    if img is None:
        sys.stderr.write(f"could not read image: {path}\n")
        return 2
    h, w = img.shape[:2]
    sys.stderr.write(f"image: {w}x{h}\n")
    dets = nms(infer(session, img, args.conf, args.classes))
    norm = [_round_det(d) for d in dets]
    for d in norm:
        x1, y1, x2, y2 = d['bbox']
        sys.stderr.write(
            f"class {d['class']} conf={d['confidence']:.2f} "
            f"bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]\n"
        )
    sys.stderr.write(f"image: {len(norm)} detections\n")
    payload = {
        'tool': 'vision_pipe',
        'version': '1.0',
        'input': path,
        'type': 'image',
        'width': w,
        'height': h,
        'model': 'yolov8n',
        'objects': norm,
    }
    print(format_text(payload) if args.text_only else format_json(payload))
    return 0


def run_video(path, session, args) -> int:
    duration = detect_duration(path)
    sys.stderr.write(f"video: duration={duration}s, fps={args.fps}\n")
    timeline = []
    width = height = 0
    n_frames = 0
    total_dets = 0
    inv_fps = 1.0 / args.fps if args.fps else 0.0
    for frame in extract_frames(path, args.fps):
        if n_frames == 0:
            height, width = frame.shape[:2]
        dets = nms(infer(session, frame, args.conf, args.classes))
        norm = [_round_det(d) for d in dets]
        timeline.append({
            'frame': n_frames,
            'timestamp': n_frames * inv_fps,
            'objects': norm,
        })
        n_frames += 1
        total_dets += len(norm)
        if not _quiet and n_frames % 10 == 0:
            sys.stderr.write(f"processed {n_frames} frames\n")
    if not _quiet:
        sys.stderr.write(f"processed {n_frames} frames\n")
    sys.stderr.write(f"video: {total_dets} detections across {n_frames} frames\n")
    classes_detected, summary = aggregate(timeline, args.fps)
    payload = {
        'tool': 'vision_pipe',
        'version': '1.0',
        'input': path,
        'type': 'video',
        'fps': args.fps,
        'duration_sec': duration,
        'total_frames': n_frames,
        'width': width,
        'height': height,
        'model': 'yolov8n',
        'classes_detected': classes_detected,
        'timeline': timeline,
        'summary': summary,
    }
    print(format_text(payload) if args.text_only else format_json(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())