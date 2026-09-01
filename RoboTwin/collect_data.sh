#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

task_name=${1:?Usage: bash collect_data.sh <task_name> <task_config> <gpu_id>}
task_config=${2:?Usage: bash collect_data.sh <task_name> <task_config> <gpu_id>}
gpu_id=${3:?Usage: bash collect_data.sh <task_name> <task_config> <gpu_id>}

if [ -d "./assets/embodiments" ]; then
    python ./script/update_embodiment_config_path.py > /dev/null 2>&1 || true
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py "${task_name}" "${task_config}"
