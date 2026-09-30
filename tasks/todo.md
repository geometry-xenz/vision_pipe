# Tasks: vision_pipe

Each task maps to a phase in `plan.md`. Verification is one runnable check per task.

## Phase 1: Scaffold
- [ ] `vision_pipe.py` exists, `python vision_pipe.py --version` prints `vision_pipe 1.0`
- [ ] `python vision_pipe.py --help` shows every flag in the README options table
- [ ] `requirements.txt` lists `onnxruntime` and `opencv-python`
- [ ] `models/.gitkeep` present
- **Verify:** `python vision_pipe.py --version` exits 0

## Phase 2: CLI + routing
- [ ] `argparse` parses `--input`, `--fps`, `--conf`, `--classes`, `--quiet`, `--text-only`, `--model`
- [ ] Extension lookup picks `image` or `video`; unknown ext → exit 2 + supported list
- [ ] Missing file → exit 2 + clear message
- **Verify:** `python vision_pipe.py nope.jpg` exits nonzero with "file not found"

## Phase 3: Model auto-export
- [ ] `models/yolov8n.onnx` present → skip export
- [ ] Missing + `ultralytics` importable → export via `YOLO("yolov8n.pt").export(format="onnx")`
- [ ] Missing + `ultralytics` not importable → exit 1 with exact `pip install ultralytics onnx`
- **Verify:** delete `models/yolov8n.onnx`, run on an image; second run skips export

## Phase 4: Frame extraction
- [ ] Video → FFmpeg subprocess with `-vf fps={N} -f image2pipe -vcodec mjpeg -`, yield frames via `np.frombuffer` + `cv2.imdecode`
- [ ] Duration pulled from `ffprobe -v error -show_entries format=duration` (fallback to OpenCV)
- [ ] No temp dir created; one frame in memory at a time
- **Verify:** run on a 5-second clip with `--fps 2`; expect ~10 frames yielded

## Phase 5: Inference
- [ ] `preprocess`: keep aspect ratio with letterbox, resize to 640x640, BGR→RGB, `/255`, HWC→CHW, float32
- [ ] `infer`: run session, take output[0] shape `(84, 8400)`, transpose to `(8400, 84)`, split `xywh` + class probs
- [ ] Conf mask: `max(class_probs) >= conf`; argmax for class id; `xywh → xyxy`; scale back to original frame size
- [ ] NMS: greedy IoU dedupe per class
- [ ] `--classes` filter by name (case-insensitive against COCO names) before NMS
- **Verify:** one image with a person → at least one detection with `class: "person"` and conf > 0.25

## Phase 6: Aggregation + output
- [ ] Image output: matches `Output Schema / Image Output` in README (tool, version, input, type, width, height, model, objects)
- [ ] Video output: matches `Output Schema / Video Output` (adds fps, duration_sec, total_frames, timeline, summary, classes_detected)
- [ ] `summary[class] = {count, first_seen, last_seen}`; `classes_detected` sorted alphabetically
- [ ] `--text-only` produces the README's "Detections by class / Notable frames" block
- [ ] JSON goes to stdout; all progress to stderr
- **Verify:** pipe stdout into `python -c "import json,sys; json.load(sys.stdin)"` → exits 0

## Phase 7: End-to-end verify
- [ ] One image: `python vision_pipe.py samples/photo.jpg --quiet | jq '.type,.objects[0].class'` → `image`, then a class name
- [ ] One short video: `python vision_pipe.py samples/clip.mp4 --fps 1 --quiet | jq '.timeline | length'` → > 0
- [ ] README's Python and Bash agent snippets run unchanged against the built binary
- **Verify:** all three commands above exit 0

## Checkpoints
- After Phase 3: tool runs `--help`, can self-bootstrap the model
- After Phase 6: image and video both produce schema-correct JSON
- After Phase 7: ready to ship