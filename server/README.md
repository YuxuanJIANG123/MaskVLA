# MaskVLA Model Server

Flask inference server for MaskVLA checkpoints. It exposes:

- `GET /health`
- `POST /reset`
- `POST /predict`

## Start

Run from the repository root:

```bash
conda activate MaskVLA
export CUDA_VISIBLE_DEVICES=0
export MASKVLA_MODEL_ROOT="$(pwd)/MaskVLA"
export MASKVLA_DEFAULT_CKPT="/path/to/checkpoint"
export MASKVLA_DEFAULT_UNNORM_KEY="your_dataset_key"
export MASKVLA_SERVER_PORT=8082
python server/robotwin-server.py
```

`MASKVLA_DEFAULT_CKPT` and `MASKVLA_DEFAULT_UNNORM_KEY` are optional only when each `/predict` request provides `ckpt_path` and `unnorm_key`.

## Requests

```bash
curl http://127.0.0.1:8082/health
```

```bash
curl -X POST http://127.0.0.1:8082/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

`POST /predict` JSON fields:

- `observation`: RoboTwin/OpenVLA observation payload.
- `instruction`: task instruction.
- `ckpt_path`: checkpoint path, optional when the default is set.
- `unnorm_key`: dataset normalization key, optional when the default is set.
