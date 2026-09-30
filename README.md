# vision_pipe

**Turn any image or video into structured data that any AI can read.**

vision_pipe is a portable CLI tool that converts both images and videos into structured JSON using YOLO object detection. Drop it into any AI agent pipeline. Any model — vision-capable or text-only — can then reason about visual content by reading the output.

No GPU required. No API costs. No rate limits.

> **Supported inputs** — Images: `.jpg` `.jpeg` `.png` `.bmp` `.webp` `.gif` · Videos: `.mp4` `.avi` `.mov` `.mkv` `.webm` `.flv` `.wmv` · Same JSON schema for both; the only discriminator key is `type`: `"image"` or `"video"`.

---

## The Problem

### AI Models Can't Watch Videos Or Read Images Cheaply

Modern AI falls into two camps:

**Vision-capable models** (GPT-4V, Claude Vision, Gemini) — expensive:
- $0.01–0.25 per image
- $0.01–0.15 per second of video
- Rate limited to 50–500 requests per minute
- 1–10 second latency per call
- You pay per pixel sent

**Text-only models** (Llama, Mistral, Qwen, any local LLM) — free, but blind:
- Zero per-call cost
- No rate limits
- No latency beyond inference speed
- Cannot process images or videos natively

You want to use a local, free, fast text model to reason about image or video content. But it can't see.

### Existing Workarounds (And Why They Fail)

| Approach | The Problem |
|---|---|
| Send frames to GPT-4V | $0.50–5.00 per minute of video. Slow. Rate limited. |
| Use Claude Vision | Still per-frame costs. Max 100 images per request. |
| Run LLaVA locally | Heavy GPU required (7B+ parameters). Slow on CPU. |
| Build custom pipeline | Complex. Hard to integrate with agents. |

### The Insight

YOLO (You Only Look Once) is a fast, lightweight object detector:

- **YOLOv8n**: 6MB model, runs at 20–40 FPS on CPU
- Outputs structured data: `[class, confidence, bounding_box]` per detection
- Free. Local. No GPU required.

YOLO doesn't describe scenes — it reports what's where. But that structured output is exactly what a text model needs to reason about image or video content.

Instead of sending images or video directly to an expensive vision API:

```
Image or Video → expensive vision API → expensive natural language description
```

Do this — extract detections and let any text model reason about them:

```
Images ─┐
        ├─→ YOLO (fast, free, local) ─→ structured JSON ─→ any text model reasons
Video ──┘
```

The text model never "sees" the image or video. It reads a detection transcript. Same understanding, different modality, zero cost.

---

## What It Does

vision_pipe accepts an image or video file and outputs structured JSON containing:

- **Per-frame detections**: timestamp, detected objects, confidence scores, bounding boxes
- **Timeline summary**: which classes appeared, when they first/last appeared, total count
- **Metadata**: duration, frame rate, dimensions, model used

Any AI agent or model can then read this JSON and reason about the image, or about what happened in the video — without ever touching the pixels.

---

## How It Works

```
Input (image or video)
         │
         ▼
┌─────────────────────────────────────┐
│         vision_pipe.py               │
│                                     │
│  1. Detect input type               │
│  2. Extract frames (video → imgs)   │
│  3. YOLO inference per frame        │
│  4. Aggregate detections             │
│  5. Output JSON or plain text       │
└─────────────────────────────────────┘
         │
         ▼
Structured JSON output
         │
         ▼
Any AI model reads and reasons
(no vision capability needed)
```

### The Processing Pipeline

**For images:**
```
image.jpg → OpenCV loads it → YOLO inference → JSON output
```

**For videos:**
```
video.mp4 → FFmpeg extracts frames at N fps
         → each frame → OpenCV loads → YOLO inference
         → detections collected per frame
         → summary aggregated
         → JSON output
```

### The Model

vision_pipe uses **YOLOv8n** — the nano variant of YOLOv8:

| Property | Value |
|---|---|
| Model size | 6MB |
| Parameters | 3.2M |
| COCO mAP | 37.4% |
| CPU FPS (modern hardware) | 20–40 fps |
| GPU required | No |

On a modern 8-core CPU, processing 1 minute of video at 1 fps takes approximately 6–15 seconds total — frame extraction plus inference.

---

## Installation

### Requirements

**Python packages (runtime):**
```
pip install onnxruntime opencv-python
```

**System dependency:**
```
ffmpeg must be in PATH
```

On macOS: `brew install ffmpeg`
On Ubuntu/Debian: `sudo apt install ffmpeg`
On Windows: Download from ffmpeg.org and add to PATH

**First-run model export (optional):**
If `models/yolov8n.onnx` is not present, vision_pipe will auto-export it from the ultralytics weights on first run. For this, install:
```
pip install ultralytics onnx
```

### Quick Start

```bash
# Install dependencies
pip install onnxruntime opencv-python

# Run directly
python vision_pipe.py photo.jpg
```

On first run, the YOLOv8n ONNX model will be auto-downloaded and exported. Subsequent runs use the cached model.

---

## Usage

### Basic

```bash
# Process an image
python vision_pipe.py photo.jpg

# Process a video (1 frame per second)
python vision_pipe.py video.mp4

# Process a video faster (2 frames per second)
python vision_pipe.py video.mp4 --fps 2

# Only keep high-confidence detections
python vision_pipe.py video.mp4 --conf 0.7

# Filter to specific object classes only
python vision_pipe.py video.mp4 --classes person car bicycle

# Plain text output instead of JSON
python vision_pipe.py video.mp4 --text-only

# Suppress progress output
python vision_pipe.py video.mp4 --quiet
```

### Output Formats

**JSON (default):**
```bash
python vision_pipe.py video.mp4
```
```json
{
  "tool": "vision_pipe",
  "version": "1.0",
  "input": "video.mp4",
  "type": "video",
  "fps": 1.0,
  "duration_sec": 180.5,
  "total_frames": 180,
  "width": 1920,
  "height": 1080,
  "model": "yolov8n",
  "classes_detected": ["bicycle", "car", "person"],
  "timeline": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "objects": [
        {"class": "person", "confidence": 0.95, "bbox": [100, 50, 200, 400]},
        {"class": "person", "confidence": 0.88, "bbox": [500, 60, 600, 390]}
      ]
    },
    {
      "frame": 1,
      "timestamp": 1.0,
      "objects": [
        {"class": "person", "confidence": 0.94, "bbox": [105, 52, 203, 402]}
      ]
    }
  ],
  "summary": {
    "person": {"count": 45, "first_seen": 0.0, "last_seen": 179.0},
    "car": {"count": 12, "first_seen": 3.0, "last_seen": 175.0}
  }
}
```

**Plain text (--text-only):**
```bash
python vision_pipe.py video.mp4 --text-only
```
```
Video analysis (180 frames at 1.0 fps, 3 minutes 0.5 seconds total):

Detections by class:
  - bicycle: 8 total, first seen at 15.0s, last seen at 165.0s
  - car: 12 total, first seen at 3.0s, last seen at 175.0s
  - person: 45 total, first seen at 0.0s, last seen at 179.0s

Notable frames (2+ objects):
  @ 5.0s: person, car
  @ 23.0s: person, bicycle
  @ 47.0s: person, person, car
  ...
```

### All Options

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--input` | `-i` | path | (positional) | Input image or video file |
| `--fps` | | float | 1.0 | Frames per second to extract from video |
| `--conf` | | float | 0.25 | Confidence threshold (0.0–1.0) |
| `--classes` | | str[] | all | Space-separated class names to filter |
| `--quiet` | `-q` | flag | false | Suppress progress output |
| `--text-only` | `-t` | flag | false | Plain English output instead of JSON |
| `--model` | | str | yolov8n | Model name or path (for future extension) |
| `--help` | `-h` | flag | — | Show help and exit |
| `--version` | `-v` | flag | — | Show version and exit |

---

## Agent Integration

vision_pipe is designed for AI agent pipelines. Any agent that can run a subprocess or shell command can use it.

### Python Agent

```python
import subprocess, json

# Run the tool
result = subprocess.run(
    ["python", "vision_pipe.py", "video.mp4", "--fps", "2", "--conf", "0.4", "--quiet"],
    capture_output=True,
    text=True
)

# Parse structured output
detections = json.loads(result.stdout)

# Reason about it
for frame in detections["timeline"]:
    if frame["timestamp"] % 10 == 0:  # every 10 seconds
        objects = [o["class"] for o in frame["objects"]]
        print(f"At {frame['timestamp']}s: {objects}")
```

### Bash Script

```bash
#!/bin/bash
# Analyze video, get plain English summary
python vision_pipe.py "$1" --fps 1 --text-only --quiet

# Or get just the JSON and pipe it elsewhere
python vision_pipe.py "$1" --fps 1 --quiet | jq '.summary'
```

### Chain with a Local LLM

```bash
# Extract detections
python vision_pipe.py security_footage.mp4 --fps 0.5 --conf 0.5 > detections.json

# Ask any local text model to reason about it
llama "I ran object detection on a video. Here's the structured output:
$(cat detections.json)

What happened in this video? Give me a timeline of events."
```

### Even with Cloud Models — Still Cheaper

```
With vision_pipe on a single image:
  → Process locally with YOLO (free, fast)
  → Send one JSON payload to GPT-4 (~$0.0001)
  → Total: free vs. $0.01–0.25 per GPT-4V call

Traditional: Send 180 frames to GPT-4V
  → 180 images × ~$0.001–0.01 each = $0.18–1.80
  → Plus rate limits and latency

With vision_pipe on a 180-frame video:
  → Process locally with YOLO (free, fast)
  → Send one JSON payload to GPT-4 (~$0.0001)
  → Total: 99%+ cheaper
```

---

## Output Schema

### Image Output

```json
{
  "tool": "vision_pipe",
  "version": "1.0",
  "input": "photo.jpg",
  "type": "image",
  "width": 1920,
  "height": 1080,
  "model": "yolov8n",
  "objects": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": [145, 230, 320, 890]
    }
  ]
}
```

**bbox format:** `[x1, y1, x2, y2]` in pixel coordinates, origin at top-left of image.

### Video Output

```json
{
  "tool": "vision_pipe",
  "version": "1.0",
  "input": "video.mp4",
  "type": "video",
  "fps": 1.0,
  "duration_sec": 180.5,
  "total_frames": 180,
  "width": 1920,
  "height": 1080,
  "model": "yolov8n",
  "classes_detected": ["bicycle", "car", "person"],
  "timeline": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "objects": [
        {"class": "person", "confidence": 0.95, "bbox": [100, 50, 200, 400]}
      ]
    }
  ],
  "summary": {
    "person": {
      "count": 45,
      "first_seen": 0.0,
      "last_seen": 179.0
    }
  }
}
```

---

## Supported Formats

### Images
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.gif`

### Videos
`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`

FFmpeg handles most codecs natively. If a specific codec is missing on the target system, that's an environment issue — vision_pipe doesn't need modification.

---

## File Structure

```
vision_pipe/
├── vision_pipe.py       # Main script — single file, everything in it
├── requirements.txt     # Runtime dependencies
├── README.md            # This file
├── models/              # Cached ONNX model
│   └── yolov8n.onnx    # Auto-downloaded/exported on first run
└── tasks/               # Planning artifacts
    ├── plan.md
    └── todo.md
```

Single-file constraint: Everything lives in `vision_pipe.py`. No submodules, no packages. Drop the file anywhere and it works.

---

## Dependencies

### Runtime

| Package | Purpose | Install |
|---|---|---|
| `onnxruntime` | ONNX model inference | `pip install onnxruntime` |
| `opencv-python` | Image loading, video frame extraction | `pip install opencv-python` |

### First-Run Export (optional)

| Package | Purpose | Install |
|---|---|---|
| `ultralytics` | Download and export YOLO to ONNX | `pip install ultralytics` |
| `onnx` | ONNX format support | `pip install onnx` |

These are only needed if `models/yolov8n.onnx` doesn't exist yet. The tool will prompt you if needed.

### System

| Tool | Purpose |
|---|---|
| `ffmpeg` | Video frame extraction. Must be in PATH. |

---

## Performance

### Speed

On a modern 8-core CPU (e.g., AMD Ryzen 7, Intel i7 10th gen+):

| Operation | Time (1 min video @ 1 fps) |
|---|---|
| Frame extraction (60 frames) | 2–5 seconds |
| YOLO inference (60 frames) | 3–8 seconds |
| JSON serialization | <0.1 seconds |
| **Total** | **~6–15 seconds** |

On older or low-power hardware, expect 2–4× slower.

### Memory

- **Peak memory:** 2–4 GB (holding frames + model + inference buffers)
- **Model size:** 6MB on disk
- **Frame buffer:** One frame at a time (streaming — doesn't load entire video into memory)

### Throughput

| Hardware | FPS (YOLO inference only) |
|---|---|
| Modern 8-core CPU | 20–40 fps |
| Low-power / older CPU | 5–15 fps |
| With GPU (future) | 100+ fps |

---

## Comparison

| Approach | Cost | GPU | Output | Agent-Friendly |
|---|---|---|---|---|
| **vision_pipe** | Free | No | Structured JSON | Yes — CLI + JSON |
| GPT-4V API | $$$$ | No | Natural language | Sort of — API call |
| Claude Vision | $$$ | No | Natural language | Sort of — API call |
| LLaVA (7B) | Free | Yes (6GB+) | Natural language | Sort of — local server |
| YOLO CLI | Free | Optional | Text/JSON | Sort of — CLI only |
| OpenCV + YOLO | Free | Optional | Raw boxes | No — needs code |

vision_pipe's advantage: truly zero-cost, zero-setup, drop into any agent pipeline as a CLI call, structured output designed for machine consumption.

---

## Why This Exists

vision_pipe was built for a specific use case: AI agents that need to understand image or video content but can't afford vision API costs or GPU hardware.

If you have a local LLM running and you want it to answer "what's in this image" or "what happened in this video" — you don't need to give it vision. You need to give it a structured transcript of what's in each frame. That's what vision_pipe produces.

It's not trying to replace vision models. It's a bridge: it takes visual content and translates it into the language that text models speak.

---

## Troubleshooting

### "FFmpeg not found"

Install FFmpeg and ensure it's in your PATH:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: Download from ffmpeg.org, add to PATH

### "onnxruntime not found"

```bash
pip install onnxruntime opencv-python
```

### "Model not found, and ultralytics not installed"

For the first run, vision_pipe needs to export the ONNX model from ultralytics weights. Either:

```bash
pip install ultralytics onnx
python vision_pipe.py video.mp4
```

Or manually download and place the ONNX file at `models/yolov8n.onnx`.

### Slow performance on large videos

- Lower the `--fps` to extract fewer frames (e.g., `--fps 0.5` for one frame every 2 seconds)
- Increase `--conf` to filter out noise early
- Use `--classes person` to only detect people

### High memory usage

vision_pipe processes frames one at a time and cleans up temp files after. If you're seeing high memory, it may be from the ONNX Runtime holding inference buffers. This is normal and expected.

---

## Future Extensions (Not in v1)

- `--stream` flag: emit JSON Lines as each frame completes (for very long videos)
- `--gpu` flag: enable CUDA acceleration when available
- `--batch` flag: process a directory of images/videos
- Custom ONNX models: pass any YOLO-compatible ONNX file
- Multiple model options: YOLOv8s, YOLOv5s for higher accuracy
- FFmpeg bundling: static FFmpeg binary for true portability

---

## License

MIT. Use it however you want.
