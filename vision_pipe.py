#!/usr/bin/env python3
import argparse
import json
import os.path
import sys
import threading

VERSION = "vision_pipe 1.0"

EXT_IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
EXT_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
EXT_AUDIO = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.opus', '.wma'}

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
    p.add_argument("--audio", action="store_true",
                   help="Extract + transcribe audio track (videos only)")
    p.add_argument("--lang", default=None,
                   help="OCR language code (e.g. eng, chi_sim, ara, rus, hin). "
                        "Auto-detected from script if omitted.")
    p.add_argument("--no-audio-features", action="store_true",
                   help="Skip audio semantic feature extraction (loudness, music, mood)")
    p.add_argument("--no-scenes", action="store_true",
                   help="Skip scene change detection")
    p.add_argument("--no-narrative", action="store_true",
                   help="Skip cross-modal narrative synthesis (fusion block)")
    p.add_argument("--scene-threshold", type=float, default=0.5,
                   help="Scene cut sensitivity (0.1=very sensitive, 0.9=only hard cuts). Default 0.5")
    return p


def ext_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in EXT_IMAGES:
        return 'image'
    if ext in EXT_VIDEOS:
        return 'video'
    if ext in EXT_AUDIO:
        return 'audio'
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


SCRIPT_TO_LANG = {
    "Latin": "eng", "Cyrillic": "rus", "Greek": "ell",
    "Arabic": "ara", "Hebrew": "heb", "Devanagari": "hin",
    "Bengali": "ben", "Gurmukhi": "pan", "Gujarati": "guj",
    "Kannada": "kan", "Malayalam": "mal", "Tamil": "tam",
    "Telugu": "tel", "Oriya": "ori", "Thai": "tha",
    "Han": "chi_sim", "Hiragana": "jpn", "Katakana": "jpn",
    "Hangul": "kor",
}


def _detect_script_tesseract(path):
    import subprocess
    r = subprocess.run(
        ["tesseract", path, "-", "--psm", "0"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    for ln in r.stdout.splitlines():
        if ln.lower().startswith("script:"):
            return SCRIPT_TO_LANG.get(ln.split(":", 1)[1].strip(), "eng")
    return None


def _tesseract_text(path, lang):
    r = subprocess.run(
        ["tesseract", path, "-", "-l", lang],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    words = r.stdout.split()
    return {
        "text": " ".join(words),
        "word_count": len(words),
        "language": lang,
        "is_text_heavy": len(words) >= 50,
        "engine": "tesseract",
    }


_rapidocr_engine = None
_rapidocr_lang = None

RAPIDOCR_LANG_MAP = {
    "eng": "english",
    "chi_sim": "chinese",
    "chi_tra": "chinese",
    "jpn": "japan",
    "kor": "korean",
    "te": "te",
}


def _rapidocr_text(path, lang=None):
    global _rapidocr_engine, _rapidocr_lang
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        venv_py = _venv_python()
        if os.path.exists(venv_py) and sys.executable != venv_py:
            sys.stderr.write(f'relaunching with {venv_py}\n')
            os.execv(venv_py, [venv_py] + sys.argv)
        return None
    requested = lang or "eng"
    rec = RAPIDOCR_LANG_MAP.get(requested)
    if rec is None:
        sys.stderr.write(
            f'rapidocr does not support {requested}; '
            f'supported: {", ".join(sorted(RAPIDOCR_LANG_MAP))}\n'
            f'install tesseract for full 100+ language coverage\n'
        )
        requested = "eng"
        rec = "english"
    if _rapidocr_engine is None or _rapidocr_lang != rec:
        if not _quiet:
            sys.stderr.write(f'loading rapidocr ({rec})...\n')
        _rapidocr_engine = RapidOCR(params={"lang_rec": rec})
        _rapidocr_lang = rec
    try:
        result = _rapidocr_engine(path)
    except Exception as e:
        sys.stderr.write(f'ocr failed: {e}\n')
        return None
    if not result or not result[0]:
        return {"text": "", "word_count": 0, "language": "eng",
                "is_text_heavy": False, "engine": "rapidocr"}
    regions = []
    words = []
    for bbox, text, conf in result[0]:
        if not text:
            continue
        x1, y1 = bbox[0]
        x3, y3 = bbox[2]
        regions.append({
            "text": text,
            "bbox": [int(x1), int(y1), int(x3), int(y3)],
            "conf": round(float(conf), 3),
        })
        words.extend(text.split())
    return {
        "text": " ".join(words),
        "word_count": len(words),
        "language": requested,
        "is_text_heavy": len(words) >= 50,
        "regions": regions,
        "engine": "rapidocr",
    }


def extract_text(path, lang=None):
    import shutil
    if shutil.which("tesseract"):
        ocr_lang = lang or _detect_script_tesseract(path) or "eng"
        return _tesseract_text(path, ocr_lang)
    return _rapidocr_text(path, lang)


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
    if os.name == 'nt':
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '.venv', 'Scripts', 'python.exe',
        )
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
    # scenes
    scenes = d.get('scenes') or []
    if scenes:
        lines.append("")
        lines.append(f"Scenes detected: {len(scenes)}")
        for s in scenes[:20]:
            lines.append(
                f"  scene {s['scene_id']}: {s['start_ts']}s → {s['end_ts']}s "
                f"({s['frame_count']} frames)"
            )
        if len(scenes) > 20:
            lines.append(f"  ... and {len(scenes) - 20} more")
    # audio
    audio = d.get('audio') or {}
    feats = audio.get('features') if isinstance(audio, dict) else None
    if feats:
        lines.append("")
        lines.append("Audio semantic features:")
        lines.append(f"  loudness (rms): {feats.get('loudness_rms_mean')}")
        lines.append(f"  spectral centroid: {feats.get('spectral_centroid_mean_hz')} Hz")
        lines.append(f"  spectral flatness: {feats.get('spectral_flatness_mean')}")
        lines.append(f"  speech ratio: {feats.get('speech_ratio')}")
        lines.append(f"  mood: {feats.get('mood')}")
        tags = feats.get('tags') or []
        if tags:
            lines.append(f"  tags: {', '.join(tags)}")
    # narrative
    narrative = d.get('narrative') or []
    if narrative:
        lines.append("")
        lines.append(f"Cross-modal narrative ({len(narrative)} scenes):")
        for ns in narrative:
            lines.append(f"  scene {ns['scene_id']} [{ns['start_ts']}s-{ns['end_ts']}s]: {ns['summary']}")
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
        payload = _run_ocr_only(path, args.lang)
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
    if kind == 'audio':
        return run_audio(path, args)
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


def _run_ocr_only(path, lang=None):
    ocr = extract_text(path, lang)
    return {
        "tool": "vision_pipe",
        "version": "1.2",
        "input": path,
        "type": "image",
        "ocr": ocr,
    }


def run_image_full(path, args) -> int:
    pixel = None if args.no_pixel else analyze_image(path)
    ocr = None if args.no_ocr else extract_text(path, args.lang)
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
    # scene-change tracking state (inline during YOLO pass to avoid re-decoding)
    do_scenes = not args.no_scenes
    import cv2
    import numpy as np
    prev_hist = None
    scene_boundaries = []  # list of frame indices where a cut was detected
    if do_scenes:
        sys.stderr.write("scene detection: enabled\n")

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
        # scene-change detection: compare HSV histogram of small frame
        if do_scenes and n_frames > 0:
            small = cv2.resize(frame, (160, 90))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist = hist.flatten().astype(np.float32)
            hist /= (hist.sum() + 1e-12)
            if prev_hist is not None:
                dist = float(cv2.compareHist(
                    prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA,
                ))
                if dist > args.scene_threshold:
                    scene_boundaries.append(n_frames)
            prev_hist = hist
        n_frames += 1
        total_dets += len(norm)
        if not _quiet and n_frames % 10 == 0:
            sys.stderr.write(f"processed {n_frames} frames\n")
    if not _quiet:
        sys.stderr.write(f"processed {n_frames} frames\n")
    sys.stderr.write(f"video: {total_dets} detections across {n_frames} frames\n")

    # build scenes list from boundaries
    scenes = []
    if do_scenes and n_frames > 0:
        starts = [0] + scene_boundaries
        ends = scene_boundaries + [n_frames - 1]
        for i, (s, e) in enumerate(zip(starts, ends)):
            scenes.append({
                "scene_id": i,
                "start_frame": s,
                "end_frame": e,
                "start_ts": round(s * inv_fps, 3) if inv_fps else 0.0,
                "end_ts": round(e * inv_fps, 3) if inv_fps else 0.0,
                "frame_count": e - s + 1,
            })
        if not _quiet:
            sys.stderr.write(f"scenes: {len(scenes)} detected (threshold={args.scene_threshold})\n")

    classes_detected, summary = aggregate(timeline, args.fps)
    payload = {
        'tool': 'vision_pipe',
        'version': '1.3',
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
        'scenes': scenes,
    }
    audio_block = None
    if args.audio:
        if not _quiet:
            sys.stderr.write("transcribing audio...\n")
        audio_block = _transcribe(path)
    audio_features = None
    if not args.no_audio_features:
        if not _quiet:
            sys.stderr.write("analyzing audio features...\n")
        audio_features = analyze_audio_features(path)
    if audio_block or audio_features:
        payload['audio'] = {}
        if audio_block:
            payload['audio']['transcript'] = audio_block
        if audio_features:
            payload['audio']['features'] = audio_features
    # cross-modal narrative synthesis (fusion block)
    if not args.no_narrative and scenes:
        if not _quiet:
            sys.stderr.write("synthesizing narrative...\n")
        payload['narrative'] = synthesize_narrative(
            timeline, scenes, payload.get('audio'),
        )
    print(format_text(payload) if args.text_only else format_json(payload))
    return 0


def run_audio(path, args) -> int:
    payload = {
        'tool': 'vision_pipe',
        'version': '1.2',
        'input': path,
        'type': 'audio',
    }
    audio = _transcribe(path)
    if audio is None:
        return 1
    payload['audio'] = audio
    if args.text_only:
        print(audio['text'])
    else:
        print(format_json(payload))
    return 0


def _transcribe_video(path):  # kept for back-compat alias
    return _transcribe(path)


_whisper_model = None


def _transcribe(path):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        venv_py = _venv_python()
        if os.path.exists(venv_py) and sys.executable != venv_py:
            sys.stderr.write(f'relaunching with {venv_py}\n')
            os.execv(venv_py, [venv_py] + sys.argv)
        sys.stderr.write(
            '--audio requires: pip install faster-whisper\n'
            '(no system pkg, ~75MB model download on first run)\n'
        )
        return None
    global _whisper_model
    if _whisper_model is None:
        if not _quiet:
            sys.stderr.write('loading whisper base...\n')
        _whisper_model = WhisperModel('base', device='cpu', compute_type='int8')
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav = tmp.name
    try:
        r = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-y', '-i', path,
             '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav],
            capture_output=True, timeout=120,
        )
        if r.returncode != 0:
            sys.stderr.write(f'ffmpeg audio extract failed: {r.stderr.decode()[:200]}\n')
            return None
        segs, info = _whisper_model.transcribe(wav, beam_size=5)
        seg_list = list(segs)
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
    return {
        'language': info.language,
        'language_probability': round(float(info.language_probability), 3),
        'duration_sec': round(float(info.duration), 2),
        'segment_count': len(seg_list),
        'segments': [
            {'start': round(s.start, 2), 'end': round(s.end, 2),
             'text': s.text.strip()}
            for s in seg_list
        ],
        'text': ' '.join(s.text.strip() for s in seg_list),
        'engine': 'faster-whisper-base',
    }


def _extract_wav_to_temp(path):
    """Extract audio to a 16kHz mono WAV tempfile. Returns path or None."""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav = tmp.name
    try:
        r = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-y', '-i', path,
             '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav],
            capture_output=True, timeout=120,
        )
        if r.returncode != 0:
            try:
                os.unlink(wav)
            except OSError:
                pass
            return None
        return wav
    except Exception:
        try:
            os.unlink(wav)
        except OSError:
            pass
        return None


def _read_wav_mono(path):
    """Read a 16kHz mono PCM WAV file. Returns (samples_float32, sample_rate)."""
    import wave
    import numpy as np
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if w.getsampwidth() == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif w.getsampwidth() == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if samples.size == 0:
        return None, sr
    return samples, sr


def analyze_audio_features(path):
    """Extract audio semantic features from a video or audio file.

    Pure numpy/stdlib — no librosa, no model weights. Computes:
    - RMS energy (loudness curve + mean)
    - Zero crossing rate (noisiness / sibilance)
    - Spectral centroid (brightness — music proxy)
    - Spectral flatness (tonal vs noisy)
    - Voice activity detection (energy threshold)
    - Music likelihood (low flatness + high centroid)

    Returns dict or None on failure.
    """
    import numpy as np
    wav = _extract_wav_to_temp(path)
    if wav is None:
        return None
    try:
        samples, sr = _read_wav_mono(wav)
        if samples is None or samples.size < int(0.1 * sr):
            return {"duration_sec": 0.0, "tags": ["silent"]}
        n = samples.size
        duration = n / float(sr)

        # windowed analysis — 50ms windows, 25ms hop
        win = int(0.05 * sr)
        hop = int(0.025 * sr)
        if win < 64:
            win = 64
        if hop < 32:
            hop = 32

        n_frames = max(1, (n - win) // hop + 1)
        rms_curve = np.empty(n_frames, dtype=np.float32)
        zcr_curve = np.empty(n_frames, dtype=np.float32)
        centroid_curve = np.empty(n_frames, dtype=np.float32)
        flatness_curve = np.empty(n_frames, dtype=np.float32)

        for i in range(n_frames):
            start = i * hop
            seg = samples[start:start + win]
            if seg.size < 8:
                rms_curve[i] = 0.0
                zcr_curve[i] = 0.0
                centroid_curve[i] = 0.0
                flatness_curve[i] = 0.0
                continue
            rms_curve[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
            # zero crossing rate
            signs = np.sign(seg)
            signs[signs == 0] = 1
            zcr_curve[i] = float(np.mean(signs[:-1] != signs[1:]))
            # FFT-based: centroid + flatness
            spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
            freqs = np.fft.rfftfreq(seg.size, d=1.0 / sr)
            mag_sum = float(spec.sum()) + 1e-12
            centroid_curve[i] = float(np.sum(freqs * spec) / mag_sum)
            # spectral flatness (geometric mean / arithmetic mean of power)
            log_spec = np.log(spec + 1e-12)
            geo = float(np.exp(log_spec.mean()))
            arith = mag_sum / spec.size
            flatness_curve[i] = geo / (arith + 1e-12)

        # aggregate
        rms_mean = float(np.mean(rms_curve))
        rms_std = float(np.std(rms_curve))
        zcr_mean = float(np.mean(zcr_curve))
        centroid_mean = float(np.mean(centroid_curve))
        flatness_mean = float(np.mean(flatness_curve))

        # voice activity detection: rms above adaptive threshold + variance
        # adaptive: 1.2× mean captures above-average energy (speech peaks)
        vad_thresh = max(rms_mean * 1.2, 0.015)
        speech_ratio = float(np.mean(rms_curve > vad_thresh))
        # dynamic range: high std relative to mean = varying content (speech)
        dynamic_range = float(rms_std / (rms_mean + 1e-6))

        tags = []
        if rms_mean < 0.01:
            tags.append("silent")
        elif rms_mean < 0.05:
            tags.append("quiet")
        elif rms_mean < 0.15:
            tags.append("moderate")
        else:
            tags.append("loud")

        # speech detection: either enough above-threshold frames OR high dynamic range
        # (speech has pauses + bursts; piano drone has steady energy)
        if speech_ratio > 0.15 or (speech_ratio > 0.10 and dynamic_range > 0.5):
            tags.append("speech_present")
        # music detection: moderate-to-high centroid + not too noisy
        if centroid_mean > 400 and flatness_mean < 0.5:
            tags.append("music_likely")
        if flatness_mean < 0.15 and speech_ratio < 0.1:
            tags.append("tonal")
        if flatness_mean > 0.5:
            tags.append("noisy")
        if zcr_mean > 0.18:
            tags.append("bright")
        if dynamic_range > 0.7:
            tags.append("dynamic")

        # mood heuristic
        mood = "neutral"
        if "silent" in tags:
            mood = "silent"
        elif "music_likely" in tags and "loud" in tags:
            mood = "energetic"
        elif "music_likely" in tags and "speech_present" in tags and rms_mean < 0.15:
            mood = "intimate"
        elif "speech_present" in tags and rms_mean < 0.08:
            mood = "intimate"
        elif "loud" in tags and "music_likely" not in tags:
            mood = "intense"

        # energy curve sampled down to <= 64 points
        step = max(1, n_frames // 64)
        energy_curve = [round(float(rms_curve[i]), 4) for i in range(0, n_frames, step)]

        return {
            "duration_sec": round(duration, 2),
            "sample_rate": int(sr),
            "loudness_rms_mean": round(rms_mean, 4),
            "loudness_rms_std": round(rms_std, 4),
            "zero_crossing_rate_mean": round(zcr_mean, 4),
            "spectral_centroid_mean_hz": round(centroid_mean, 1),
            "spectral_flatness_mean": round(flatness_mean, 4),
            "speech_ratio": round(speech_ratio, 3),
            "tags": tags,
            "mood": mood,
            "energy_curve": energy_curve,
            "engine": "numpy-stdlib-v1",
        }
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


def detect_scene_changes_from_frames(frames_iter, fps, threshold=0.5):
    """Detect scene cuts by comparing HSV histograms between consecutive frames.

    Args:
        frames_iter: iterator yielding BGR frames (np.ndarray)
        fps: frames per second being extracted
        threshold: Bhattacharyya distance threshold (0..1, lower = more sensitive)

    Returns:
        list of {"scene_id", "start_frame", "end_frame", "start_ts", "end_ts"}
    """
    import cv2
    import numpy as np

    scenes = []
    cur_start = 0
    prev_hist = None
    idx = 0

    for frame in frames_iter:
        # downscale for speed
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = hist.flatten().astype(np.float32)
        hist /= (hist.sum() + 1e-12)

        if prev_hist is not None:
            dist = cv2.compareHist(
                prev_hist.astype(np.float32),
                hist,
                cv2.HISTCMP_BHATTACHARYYA,
            )
            if dist > threshold:
                # scene cut — close previous
                scenes.append({
                    "scene_id": len(scenes),
                    "start_frame": cur_start,
                    "end_frame": idx - 1,
                    "start_ts": round(cur_start / fps, 3) if fps else 0.0,
                    "end_ts": round((idx - 1) / fps, 3) if fps else 0.0,
                    "frame_count": idx - cur_start,
                })
                cur_start = idx
        prev_hist = hist
        idx += 1

    # close the final scene
    if idx > cur_start:
        scenes.append({
            "scene_id": len(scenes),
            "start_frame": cur_start,
            "end_frame": idx - 1,
            "start_ts": round(cur_start / fps, 3) if fps else 0.0,
            "end_ts": round((idx - 1) / fps, 3) if fps else 0.0,
            "frame_count": idx - cur_start,
        })

    return scenes


def synthesize_narrative(timeline, scenes, audio):
    """Cross-modal fusion: combine visual objects + scene boundaries + audio into
    a per-scene narrative readable by any text LLM.

    No ML — pure structural fusion. Each scene gets a one-line summary that
    joins the dominant objects in that scene's frame range with any audio that
    falls in that time range.
    """
    if not scenes:
        return []

    # index timeline by frame index for fast lookup
    by_frame = {e['frame']: e for e in timeline}
    transcript_segs = []
    if audio and isinstance(audio, dict):
        # audio block may have either flat "segments" or nested "transcript.segments"
        segs = audio.get('segments') or []
        if not segs:
            tr = audio.get('transcript') or {}
            if isinstance(tr, dict):
                segs = tr.get('segments') or []
        transcript_segs = [(s.get('start', 0.0), s.get('end', 0.0), s.get('text', ''))
                          for s in segs if s.get('text')]

    out = []
    for s in scenes:
        s_start = s['start_ts']
        s_end = s['end_ts']

        # aggregate objects in this scene
        obj_counts = {}
        for fidx in range(s['start_frame'], s['end_frame'] + 1):
            e = by_frame.get(fidx)
            if not e:
                continue
            for o in e['objects']:
                cls = o['class']
                obj_counts[cls] = obj_counts.get(cls, 0) + 1
        top_objs = sorted(obj_counts.items(), key=lambda kv: -kv[1])[:5]
        obj_str = ", ".join(f"{c}×{n}" if n > 1 else c for c, n in top_objs) or "none"

        # gather transcript segments in this scene's time range
        spoken = " ".join(
            txt for st, en, txt in transcript_segs
            if txt and not (en < s_start or st > s_end)
        ).strip()

        # audio mood if available
        mood = ""
        if audio and isinstance(audio, dict):
            af = audio.get('features') or {}
            if isinstance(af, dict):
                mood = af.get('mood', '')

        out.append({
            "scene_id": s['scene_id'],
            "start_ts": s_start,
            "end_ts": s_end,
            "duration_sec": round(s_end - s_start, 3),
            "frame_count": s['frame_count'],
            "objects": obj_str,
            "top_objects": [{"class": c, "count": n} for c, n in top_objs],
            "spoken": spoken,
            "mood": mood,
            "summary": _scene_summary_line(obj_str, spoken, mood, s_end - s_start),
        })

    return out


def _scene_summary_line(objects, spoken, mood, duration):
    """Build a one-line natural-language summary for a scene."""
    parts = []
    if objects and objects != "none":
        parts.append(f"visual: {objects}")
    if spoken:
        snippet = spoken if len(spoken) <= 200 else spoken[:197] + "..."
        parts.append(f"audio: \"{snippet}\"")
    if mood and mood != "neutral":
        parts.append(f"mood: {mood}")
    if duration > 0:
        parts.append(f"({duration:.1f}s)")
    return " | ".join(parts) if parts else "(no content)"


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())