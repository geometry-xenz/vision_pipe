# Implementation Plan: vision_pipe

## Overview
Single-file Python CLI that turns images and videos into structured JSON via YOLOv8n ONNX, so any text-only AI can reason about visual content by reading the output. CPU only, no GPU, agent-pipeline first.

## Problem
- Vision APIs cost per frame; rate-limited; text-only LLMs are blind.
- Local LLMs (llama/mistral/qwen) can reason over text but not pixels.
- YOLO gives structured `[class, conf, bbox]` per frame — exactly what a text model needs as a transcript.

## Core decisions

| Decision | Choice | Why this, not the alt |
|---|---|---|
| Interface | CLI → JSON on stdout, log on stderr | Any agent in any language can `subprocess` it; stdout stays parseable |
| Detector | YOLOv8n ONNX via `onnxruntime` | 20–40 FPS CPU, 6 MB, no GPU. Bigger models = +mAP but +RAM and slower; tiny is enough as a transcript |
| Frame source | FFmpeg pipe (`-vf fps=N -f image2pipe -vcodec mjpeg -`) | No temp dir to manage, no disk I/O. OpenCV `VideoCapture` is the lazy fallback if FFmpeg is missing |
| Input types | Image + video in one CLI | Same schema; `type` key discriminates. One tool beats two |
| Output | JSON (default) + `--text-only` plain English | JSON for code, text for direct LLM prompts |
| Model delivery | Auto-export on first run via `ultralytics`; cache `models/yolov8n.onnx` | One-time 6 MB download, then offline. Avoids shipping weights in repo |
| Layout | One file: `vision_pipe.py` | README's "drop anywhere" promise. No package, no init, no entry-point until asked |

## Single-file architecture (`vision_pipe.py`)

Order top-to-bottom, no class unless stateful:

1. Imports + constants — version string, class names (COCO 80), ext sets, model path
2. `ensure_model()` — export ONNX if missing; import `ultralytics` lazily so it's not a runtime dep
3. `load_session()` — `onnxruntime.InferenceSession`, returns session once
4. `preprocess(frame)` — resize letterbox to 640x640, BGR→RGB, normalize, HWC→CHW, add batch dim
5. `infer(session, tensor)` — run, transpose `(1,84,8400)→(8400,84)`, conf mask, xywh→xyxy, NMS
6. `extract_frames(path, fps)` — `subprocess` FFmpeg pipe, yield `np.ndarray` per frame
7. `aggregate(timeline)` — per-class `count`, `first_seen`, `last_seen`; sorted `classes_detected`
8. `format_image(out)` / `format_video(out)` — build JSON dict per documented output schema
9. `format_text(out)` — `--text-only` rendering (the README's "Notable frames" paragraph)
10. `cli()` — `argparse`; routes by extension; main entry

## Phases (source for `todo.md`)

1. **Scaffold** — `vision_pipe.py` stub responds to `--version`/`--help`; `requirements.txt`; `models/.gitkeep`
2. **CLI + routing** — argparse for all flags, ext → image|video, file-exists + ext-supported validation
3. **Model auto-export** — `ensure_model()`; clear error if `ultralytics` missing with install command
4. **Frame extraction** — FFmpeg subprocess with stdout pipe; fallback to OpenCV; duration via `ffprobe`
5. **Inference** — preprocess + infer + NMS + class filter; reuse for image and video (same code path)
6. **Aggregation + output** — timeline + summary; JSON formatter; `--text-only` formatter
7. **End-to-end verify** — one image and one short video through the whole pipe; assert JSON parses and contains expected keys

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| FFmpeg not in PATH | Detect via `shutil.which`; fall back to `cv2.VideoCapture`; warn clearly either way |
| Ultralytics not installed on first run | Print exact `pip install` command and exit 1; do not silently download ONNX from a URL we don't control in v1 |
| Very long video floods memory | Stream frames via FFmpeg pipe (one at a time); never `cv2.VideoCapture.read()` in a loop into a list |
| YOLO output shape changes between versions | Pin parsing to YOLOv8 `(1, 84, 8400)`; `--model` flag reserved for future variants |
| Agent can't parse JSON | `--text-only` covers it; also keep stderr completely separate from stdout |

## Deliberately deferred (YAGNI)
- `--stream` JSON Lines output
- `--gpu` CUDA provider
- `--batch` directory mode
- Multiple model variants (yolov8s, yolov5s)
- Bundled static FFmpeg
- `setup.py` / `pip install -e .` (the README mentions it; ship only when someone needs it)

## Open questions
- Pin YOLOv8n weights via `ultralytics` cache or commit `yolov8n.onnx` to repo for fully offline install?
- Should `--classes` accept class IDs as well as names (COCO IDs 0–79)?
- OpenCV fallback for frame extraction: keep it or drop it to shrink surface area?