# MaskVLA

MaskVLA is a robotics/VLA workspace for RoboTwin simulation, MaskVLA fine-tuning, and checkpoint inference through a small Flask server.

## Structure

```text
RoboTwin/      Simulation tasks, assets, data collection, and policy evaluation
MaskVLA/   MaskVLA training, fine-tuning, and inference code
server/        HTTP model server used by RoboTwin evaluation
```

## Environment

Use separate Conda environments for simulation and VLA training/inference. The project is GPU/CUDA sensitive, so install the PyTorch build that matches your driver.

RoboTwin:

```bash
cd RoboTwin
conda env create -f environment-2.yml
conda activate robotwin2
bash script/_download_assets.sh
```

MaskVLA:

```bash
cd MaskVLA
conda create -n MaskVLA python=3.10 -y
conda activate MaskVLA
pip install -e .
```

For training, install FlashAttention after the editable install:

```bash
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

For inference-only runs where FlashAttention is unavailable:

```bash
export DISABLE_FLASH_ATTN=1
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
```

## Common Commands

Smoke test RoboTwin:

```bash
cd RoboTwin
conda activate robotwin2
bash test.sh
```

Collect one task:

```bash
cd RoboTwin
conda activate robotwin2
bash collect_data.sh put_object_cabinet demo_clean 0
```

Collect the task list in `sim-gen.sh`:

```bash
cd RoboTwin
conda activate robotwin2
bash sim-gen.sh
```

Run the inference server:

```bash
conda activate MaskVLA
export MASKVLA_MODEL_ROOT="$(pwd)/MaskVLA"
export MASKVLA_DEFAULT_CKPT="/path/to/checkpoint"
export MASKVLA_DEFAULT_UNNORM_KEY="your_dataset_key"
export MASKVLA_SERVER_PORT=8082
python server/robotwin-server.py
```

Health check:

```bash
curl http://127.0.0.1:8082/health
```

`POST /predict` requires `observation` and `instruction`. If the default checkpoint environment variables are not set, include `ckpt_path` and `unnorm_key` in the request.

## Key Files

- `RoboTwin/envs/`: task implementations.
- `RoboTwin/task_config/`: task, camera, and embodiment configs.
- `RoboTwin/description/task_instruction/all_tasks.json`: task instructions.
- `RoboTwin/policy/`: policy implementations and evaluation scripts.
- `MaskVLA/SETUP.md`: upstream MaskVLA setup notes.
- `server/README.md`: server-specific API notes.

## Core MaskVLA Change

The key MaskVLA implementation is in `MaskVLA/prismatic/extern/hf/modeling_prismatic.py`.

- Main entry: `apply_maskvla_training_mask(...)`.
- Default behavior: channel-wise masking on the primary camera with probability `0.1`.
- Ablations: change mask probability, switch to random-camera masking, or use patch-wise masking in the commented alternatives inside `apply_maskvla_training_mask(...)`.
- Debug images: call `save_maskvla_debug_images(...)` from the multimodal forward path when visual inspection of masked camera inputs is needed.

## Troubleshooting

- Asset path errors: run `bash script/_download_assets.sh` from `RoboTwin/`.
- Import errors: set `MASKVLA_MODEL_ROOT` to the absolute `MaskVLA` code path.
- Missing checkpoint defaults: set `MASKVLA_DEFAULT_CKPT` and `MASKVLA_DEFAULT_UNNORM_KEY`, or pass them in `/predict`.
- CUDA/PyTorch errors: reinstall PyTorch for the local CUDA driver.
