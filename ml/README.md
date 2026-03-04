# ML Layer — Training, Export, and C++ Integration

This directory contains the Python ML training and export pipeline, plus instructions for how the exported ONNX models are loaded and run in the C++ real-time camera pipeline on Apple M3 Pro.

## Overview of the Three Neural Stages

1. **Scene Classifier** — MobileNetV3-Small fine-tuned on CIFAR-10. Produces top-k class labels and confidences per frame. Used for scene-aware overlays (e.g. "dog 0.92").
2. **Saliency Detection** — Lightweight U-Net trained on SALICON-style data. Outputs a saliency heatmap (0–1) per frame for attention overlay or masking.
3. **Super Resolution** — SRCNN (or ESRGAN-tiny) trained on DIV2K. Upscales low-resolution patches; optional PSNR/SSIM metrics for quality overlay.

All three models are trained in Python with **MPS (Metal Performance Shaders)** on Apple Silicon, exported to **ONNX**, and loaded in C++ via **OpenCV DNN** only (no LibTorch). The C++ pipeline runs inference asynchronously via a **NeuralDispatcher** so the main camera pipeline never blocks.

## Setup

- **Python**: 3.10+ recommended. Create a virtualenv and install dependencies:

  ```bash
  cd ml
  python3 -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```

- **MPS verification**: Ensure PyTorch sees MPS:

  ```python
  import torch
  assert torch.backends.mps.is_available(), "MPS not available"
  device = torch.device("mps")
  ```

- **W&B (Weights & Biases)**: Sign up at [wandb.ai](https://wandb.ai), then `wandb login` for training logging.

## Training Instructions (order)

1. **Scene classifier**  
   `python training/train_classifier.py`  
   Uses CIFAR-10 (downloaded automatically). Checkpoints go to `models/classifier.pt`. Logs loss and accuracy per epoch to W&B.

2. **Saliency**  
   `python training/train_saliency.py`  
   Requires SALICON (or similar) data at `data/salicon`. Saves `models/saliency.pt`. Logs loss and NSS to W&B.

3. **Super resolution**  
   `python training/train_superres.py`  
   Requires DIV2K at `data/div2k`. Saves `models/superres.pt`. Logs PSNR and SSIM per epoch to W&B.

## Export Instructions

After training, export each model to ONNX (used by C++):

- **Classifier**: `python export/export_classifier.py` → `models/scene_classifier.onnx`
- **Saliency**: `python export/export_saliency.py` → `models/saliency.onnx`
- **Super resolution**: `python export/export_superres.py` → `models/superres.onnx`

Each export script runs a quick `validate_onnx` check (PyTorch vs ONNX Runtime) to verify correctness.

## How ONNX Models Are Loaded in C++

- Model paths are defined in `src/controls/neural_config.h` (e.g. `ml/models/scene_classifier.onnx`).
- Each neural stage (`SceneClassifierStage`, `SaliencyStage`, `SuperResolutionStage`) loads its ONNX file via `cv::dnn::readNetFromONNX(path)` in `loadModel()`.
- Inference runs on a **single background thread** in `NeuralDispatcher::run()`. Frames are submitted with `submitFrame()`; the main pipeline never waits on inference. Results are read non-blockingly with `getSceneResult()`, `getSaliencyResult()`, `getSuperResResult()`.
- The display thread (renderer) calls the new overlay methods `overlaySceneLabels()`, `overlaySaliencyMap()`, `overlayNeuralMetrics()` using these cached results.

## Expected Inference Latency Targets (M3 Pro)

| Stage            | Target (per frame) | Note                          |
|------------------|--------------------|--------------------------------|
| Scene Classifier | &lt; 5 ms           | Runs every 15 frames           |
| Saliency         | &lt; 10 ms          | Runs every 5 frames             |
| Super Resolution | &lt; 15 ms          | Runs every frame if budget allows |

These are approximate; measure with the neural dispatcher’s `getNeuralFps()` and the pipeline’s existing latency profiler. If any stage exceeds the frame budget (~33 ms for 30 FPS), reduce input resolution or run that stage less often (e.g. every N frames).

## W&B Dashboard Setup

1. Create an account at [wandb.ai](https://wandb.ai).
2. Run `wandb login` and paste your API key.
3. Training scripts log `loss`, `accuracy` (classifier), `nss` (saliency), `psnr`/`ssim` (superres) per epoch. View runs and compare experiments in the W&B project dashboard.
