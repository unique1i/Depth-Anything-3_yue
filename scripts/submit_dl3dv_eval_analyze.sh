#!/bin/bash
#SBATCH --output=logs/dl3dv_eval_analyze_%A_%a.log
#SBATCH --error=logs/dl3dv_eval_analyze_%A_%a.log
#SBATCH --nodes=1
#SBATCH -p capacity
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
# 55 evaluation zips / 10 scenes per task => 6 array tasks.
#SBATCH --array=0-5

set -euo pipefail

source /home/yli7/.bashrc
micromamba activate da3
module load gnu12/12.4.0

cd /home/yli7/repos/Depth-Anything-3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${PYTHON_BIN:-/home/yli7/local/micromamba/envs/da3/bin/python}"
ZIP_ROOT="${ZIP_ROOT:-/home/yli7/scratch/datasets/dl3dv_960p/evaluation/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yli7/scratch/datasets/dl3dv_960p/evaluation/da3_streaming_gs}"
CONFIG_PATH="${CONFIG_PATH:-/home/yli7/repos/Depth-Anything-3/da3_streaming/configs/base_config.yaml}"
SCENES_PER_TASK="${SCENES_PER_TASK:-10}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}"
  exit 2
fi
if [[ ! -d "${ZIP_ROOT}" ]]; then
  echo "Zip root not found: ${ZIP_ROOT}"
  exit 2
fi
if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Output root not found: ${OUTPUT_ROOT}"
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

TOTAL_SCENES="$(find "${ZIP_ROOT}" -maxdepth 1 -type f -name '*.zip' | wc -l)"
TASKS_NEEDED="$(( (TOTAL_SCENES + SCENES_PER_TASK - 1) / SCENES_PER_TASK ))"
START_IDX="$((SLURM_ARRAY_TASK_ID * SCENES_PER_TASK))"
END_IDX="$((START_IDX + SCENES_PER_TASK))"
if (( END_IDX > TOTAL_SCENES )); then
  END_IDX="${TOTAL_SCENES}"
fi

echo "Host: $(hostname)"
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

RUN_DIR="outputs/analyze_existing_dl3dv_eval/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

bash scripts/analyze_existing_dl3dv_eval.sh \
  --python-bin "${PYTHON_BIN}" \
  --config "${CONFIG_PATH}" \
  --zip-root "${ZIP_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --start-idx "${START_IDX}" \
  --end-idx "${END_IDX}" \
  --run-dir "${RUN_DIR}"

echo "Finished: $(date)"
