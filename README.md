# Real-Time Camera Processing Pipeline

A multi-threaded image processing pipeline that simulates a camera capture stack: capture from a webcam, run a configurable chain of stages (debayer, noise reduction, tone mapping, histogram, edge detection), plus **neural inference stages** (scene classification, saliency, super-resolution), and display the result with per-stage latency and keyboard toggles.

## Features

- **Classic image processing:** Debayer, noise reduction, tone mapping, histogram, edge detection
- **Neural stages:** Scene classifier, saliency heatmap, super-resolution (all run asynchronously on a background thread)
- **Overlays:** Scene labels, saliency heatmap blend, PSNR/SSIM metrics

## Dependencies

- **C++17** compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.14+
- **OpenCV** 4.x with `dnn` module (for ONNX inference)
- **Python 3.8+** with PyTorch (for training models — optional)

### Install OpenCV (examples)

- **macOS (Homebrew):** `brew install opencv`
- **Ubuntu/Debian:** `sudo apt install libopencv-dev`
- **Windows:** Use vcpkg or build from [opencv.org](https://opencv.org/releases/).

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Run

**Quick start** (builds if needed and runs):

```bash
./run.sh
```

Or manually from the `build` directory:

```bash
./RealTimeCameraPipeline
```

Or from project root: `./build/RealTimeCameraPipeline`

## Neural Stages (Optional)

The pipeline includes three neural inference stages that run asynchronously:
- **Scene Classifier:** Displays top-k scene/object labels (ImageNet classes)
- **Saliency:** Overlays a heatmap showing where the model thinks you'll look
- **Super-Resolution:** Computes PSNR/SSIM quality metrics

### Quick Start (without models)

The pipeline works fine without trained models — neural overlays simply won't appear. This is the default behavior.

### Training and Exporting Models

To enable neural overlays, train and export ONNX models:

```bash
cd ml

# 1. Install Python dependencies
pip install torch torchvision torchaudio opencv-python pillow tqdm

# 2. Prepare data (see ml/DATA.md for real datasets or use synthetic)
python scripts/create_minimal_data.py  # Creates synthetic data for testing

# 3. Train models (disable wandb if not logged in)
WANDB_MODE=disabled python training/train_classifier.py --epochs 5
WANDB_MODE=disabled python training/train_saliency.py --epochs 5
WANDB_MODE=disabled python training/train_superres.py --epochs 5

# 4. Export to ONNX
python export/export_classifier.py
python export/export_saliency.py
python export/export_superres.py
```

Models are saved to `ml/models/` and the C++ pipeline automatically loads them on startup.

### Expected ONNX Paths

The C++ code looks for models at:
- `ml/models/scene_classifier.onnx`
- `ml/models/saliency.onnx`
- `ml/models/superres.onnx`

## Keyboard Controls

| Key | Action |
|-----|--------|
| `1` | Toggle Debayer stage |
| `2` | Toggle Noise Reduction stage |
| `3` | Toggle Tone Mapping stage |
| `4` | Toggle Histogram overlay stage |
| `5` | Toggle Edge Detection stage |
| `ESC` | Quit pipeline (graceful shutdown) |

*(Exact key bindings are defined in `StageController::handle_key()`; adjust as needed.)*

## Architecture (ASCII)

```
                    +------------------+
                    |   Webcam (OpenCV)|
                    +--------+---------+
                             |
                             v
  +------------------+  [Queue 0]  +------------------+
  |  FramePool       | ----------> |  DebayerStage     |
  |  (acquire/release)|             |  (thread 1)       |
  +--------+---------+             +--------+----------+
       ^   |                               |
       |   |  [Queue 1]                    v
       |   +------------------------> +------------------+
       |                              | NoiseReduction   |
       |                              | (thread 2)       |
       |                              +--------+--------+
       |                                       |
       |  [Queue 2]                            v
       |   +--------------------------> +------------------+
       |                                | ToneMappingStage |
       |                                | (thread 3)       |
       |                                +--------+--------+
       |                                         |
       |  [Queue 3]                               v
       |   +------------------------------> +------------------+
       |                                    | HistogramStage   |
       |                                    | (thread 4)       |
       |                                    +--------+--------+
       |                                             |
       |  [Queue 4]                                  v
       |   +---------------------------------> +------------------+
       |                                        | EdgeDetection    |
       |                                        | (thread 5)       |
       |                                        +--------+--------+
       |                                                 |
       |  [Queue 5]                                      v
       |   +---------------------------------------> +------------------+
       |                                             | Renderer (display)|
       |                                             | (thread 6)        |
       +---------------------------------------------+------------------+
                 release Frame back to pool
```

- **Capture thread:** Reads from webcam, acquires `Frame` from pool, pushes to Queue 0.
- **Stage threads:** Each pops from its input queue, optionally runs `process()` (if enabled via `StageController`), pushes to next queue. Uses `ScopedTimer` / `PipelineStats` for latency.
- **Display thread:** Pops from last queue, renders with overlay (active stages + latency), releases frame to pool.

## Project Layout

```
src/
  pipeline/       frame, thread_safe_queue, frame_pool, pipeline
  stages/         stage_base, debayer, noise_reduction, tone_mapping, histogram, edge_detection
  stages/neural/  scene_classifier, saliency, super_resolution, neural_dispatcher
  profiling/      scoped_timer, pipeline_stats
  controls/       config, stage_controller
  display/        renderer
  main.cpp
ml/
  training/       train_classifier.py, train_saliency.py, train_superres.py
  export/         export_classifier.py, export_saliency.py, export_superres.py
  evaluation/     eval_classifier.py, eval_saliency.py, eval_superres.py
  utils/          dataset_utils.py, model_utils.py, visualization.py
  models/         (ONNX models saved here)
  data/           (training data — not tracked in git)
CMakeLists.txt
README.md
```

## Optimization notes

- **Target:** 30 FPS ⇒ 33 ms per frame budget. Pipeline was stuck at ~24 FPS.
- **Bottleneck:** Profiling showed **NoiseReduction** at ~49 ms per frame — bilateral filter at full resolution was blowing the budget.
- **Fix:** Run bilateral at **half-resolution** then upsample (4× fewer pixels), and use a **smaller kernel** (d=5, sigma 50 instead of d=9, sigma 75). Latency drops into the single-digit ms range; FPS can reach 30.
- **Alternatives tried / available:** Reduce filter radius only; Gaussian blur to confirm bilateral was the cost; Joint Bilateral or box-filter approximation for further speed vs quality tradeoffs.

## Performance & Benchmarks

Tested on **MacBook Pro (M3 Pro)** with built-in webcam at **1280×720**.

| Stage | Latency (µs) | Status |
|-------|--------------|--------|
| Debayer | 238 µs | Well within 33ms budget |
| Noise Reduction | 2,304 µs | Well within 33ms budget |
| Tone Mapping | 211 µs | Well within 33ms budget |
| Histogram | 684 µs | Well within 33ms budget |
| Edge Detection | 518 µs | Well within 33ms budget |

**Observed FPS:** 24 fps

All five processing stages run concurrently on separate threads. Per-stage profiling confirms no single stage exceeds the 33 ms frame budget required for 30 fps throughput. The observed **24 fps ceiling is a hardware constraint** — the built-in MacBook webcam driver caps capture at 24 fps regardless of pipeline speed. This was confirmed by querying `CAP_PROP_FPS` directly, which returns 24. Pipeline processing overhead is not the limiting factor.

**To achieve 30 fps:** Use an external USB webcam or camera that supports 30 fps capture (e.g. Logitech C920). The processing pipeline has sufficient headroom to sustain 30 fps given a capable capture device.

## License

MIT License — see [LICENSE](LICENSE). Standard permissive license for portfolio projects.
