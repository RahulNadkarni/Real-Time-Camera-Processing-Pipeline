# Real-Time Camera Processing Pipeline

A multi-threaded image processing pipeline that simulates a camera capture stack: capture from a webcam, run a configurable chain of stages (debayer, noise reduction, tone mapping, histogram, edge detection), and display the result with per-stage latency and keyboard toggles.

## Dependencies

- **C++17** compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake** 3.14+
- **OpenCV** 4.x (with `videoio` and `highgui` for webcam and display)

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

Run:

```bash
./RealTimeCameraPipeline
```

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
  profiling/      scoped_timer, pipeline_stats
  controls/       config, stage_controller
  display/        renderer
  main.cpp
CMakeLists.txt
README.md
```

## License

Use as you like; this is a portfolio/skeleton project.
