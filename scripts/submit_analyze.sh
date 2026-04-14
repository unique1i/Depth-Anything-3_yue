#!/bin/bash
#SBATCH --output=logs/analyze_%A_%a.log
#SBATCH --error=logs/analyze_%A_%a.log
#SBATCH --nodes=1
#SBATCH -p capacity
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
# 4912 scenes / 50 scenes per task => 99 array tasks.
#SBATCH --array=0-98

set -euo pipefail

source /home/yli7/.bashrc
micromamba activate da3
module load gnu12/12.4.0

cd /home/yli7/repos/Depth-Anything-3

PYTHON_BIN="${PYTHON_BIN:-/home/yli7/local/micromamba/envs/da3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/yli7/scratch2/datasets/dl3dv_960p}"
SPLIT_FILE="${SPLIT_FILE:-/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200.txt}"
CONFIG_PATH="${CONFIG_PATH:-/home/yli7/repos/Depth-Anything-3/da3_streaming/configs/base_config.yaml}"
SCENES_PER_TASK="${SCENES_PER_TASK:-50}"
SCENE_OUTPUT_SUBDIR="${SCENE_OUTPUT_SUBDIR:-da3_streaming_output}"
DEPTH_THRESHOLD_M="${DEPTH_THRESHOLD_M:-100}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}"
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root not found: ${DATA_ROOT}"
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

TOTAL_SCENES="$(wc -l < "${SPLIT_FILE}")"
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

RUN_DIR="outputs/analyze_existing_dl3dv/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

bash scripts/analyze_existing_dl3dv.sh \
  --python-bin "${PYTHON_BIN}" \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --split-file "${SPLIT_FILE}" \
  --scene-output-subdir "${SCENE_OUTPUT_SUBDIR}" \
  --depth-threshold-m "${DEPTH_THRESHOLD_M}" \
  --start-idx "${START_IDX}" \
  --end-idx "${END_IDX}" \
  --run-dir "${RUN_DIR}" \
  --pose-condition \
  --undistort \
  --exclude-depth-above-100-for-points

echo "Finished: $(date)"
