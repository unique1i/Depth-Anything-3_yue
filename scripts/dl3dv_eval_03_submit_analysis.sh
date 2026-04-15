#!/bin/bash
#SBATCH --output=logs/dl3dv_eval_03_analyze_%A_%a.log
#SBATCH --error=logs/dl3dv_eval_03_analyze_%A_%a.log
#SBATCH --nodes=1
#SBATCH -p capacity
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
# Filtered evaluation split with one extra array slot; out-of-range tasks no-op safely.
#SBATCH --array=0-5

set -euo pipefail

cd /home/yli7/repos/Depth-Anything-3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BASH_ENV

PYTHON_BIN="${PYTHON_BIN:-/home/yli7/local/micromamba/envs/da3/bin/python}"
SPLIT_FILE="${SPLIT_FILE:-/home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yli7/scratch2/datasets/dl3dv_960p/evaluation}"
CONFIG_PATH="${CONFIG_PATH:-/home/yli7/repos/Depth-Anything-3/da3_streaming/configs/base_config.yaml}"
SCENES_PER_TASK="${SCENES_PER_TASK:-10}"
RUN_DIR_PREFIX="${RUN_DIR_PREFIX:-outputs/analyze_existing_dl3dv_eval}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}"
  exit 2
fi
if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Output root not found: ${OUTPUT_ROOT}"
  exit 2
fi
if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "Split file not found: ${SPLIT_FILE}"
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 2
fi
if ! [[ "${SCENES_PER_TASK}" =~ ^[0-9]+$ ]] || (( SCENES_PER_TASK <= 0 )); then
  echo "Invalid SCENES_PER_TASK=${SCENES_PER_TASK}; must be a positive integer."
  exit 2
fi

TOTAL_SCENES="$("${PYTHON_BIN}" - "${SPLIT_FILE}" <<'PY'
from pathlib import Path
import sys

count = 0
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    count += 1
print(count)
PY
)"
TASKS_NEEDED="$(( (TOTAL_SCENES + SCENES_PER_TASK - 1) / SCENES_PER_TASK ))"
START_IDX="$((SLURM_ARRAY_TASK_ID * SCENES_PER_TASK))"
END_IDX="$((START_IDX + SCENES_PER_TASK))"
if (( END_IDX > TOTAL_SCENES )); then
  END_IDX="${TOTAL_SCENES}"
fi

echo "Host: $(hostname)"
echo "Workflow step: dl3dv_eval_03_submit_analysis"
echo "Start: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Total scenes: ${TOTAL_SCENES}"
echo "Scenes per task: ${SCENES_PER_TASK}"
echo "Tasks needed for split: ${TASKS_NEEDED}"
echo "Range: [${START_IDX}, ${END_IDX})"

if (( START_IDX >= TOTAL_SCENES )); then
  echo "Nothing to do for this array task."
  exit 0
fi

RUN_DIR="${RUN_DIR_PREFIX}/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

env -u BASH_ENV bash scripts/dl3dv_eval_03_analyze_existing.sh \
  --python-bin "${PYTHON_BIN}" \
  --config "${CONFIG_PATH}" \
  --split-file "${SPLIT_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  --start-idx "${START_IDX}" \
  --end-idx "${END_IDX}" \
  --run-dir "${RUN_DIR}"

echo "Finished: $(date)"
