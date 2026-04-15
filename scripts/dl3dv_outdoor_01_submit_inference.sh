#!/bin/bash
#SBATCH --output=logs/dl3dv_outdoor_01_infer_%A_%a.log
#SBATCH --error=logs/dl3dv_outdoor_01_infer_%A_%a.log
#SBATCH --nodes=1
#SBATCH -p capacity
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=9:00:00
# 5052 outdoor scenes / 50 scenes per task => 102 array tasks.
#SBATCH --array=0-101
##SBATCH --exclude=hipster-cn010,hipster-cn007


source /home/yli7/.bashrc
micromamba activate da3
module load gnu12/12.4.0

cd /home/yli7/repos/Depth-Anything-3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/home/yli7/local/micromamba/envs/da3/bin/python"
FUSE_PYTHON_BIN="${FUSE_PYTHON_BIN:-${PYTHON_BIN}}"
INFER_SCRIPT="scripts/dl3dv_outdoor_01_streaming_inference.py"
FUSE_SCRIPT="/home/yli7/scratch2/datasets/dataset_tools/dl3dv/fuse_da3_depth_maps.py"

CONFIG_PATH="/home/yli7/repos/Depth-Anything-3/da3_streaming/configs/base_config.yaml"
DATA_ROOT="/home/yli7/scratch2/datasets/dl3dv_960p"
SPLIT_FILE="/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200.txt"
SCENES_PER_TASK="${SCENES_PER_TASK:-50}"

if ! [[ "${SCENES_PER_TASK}" =~ ^[0-9]+$ ]] || (( SCENES_PER_TASK <= 0 )); then
  echo "Invalid SCENES_PER_TASK=${SCENES_PER_TASK}; must be a positive integer."
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}"
  exit 2
fi
if [[ ! -x "${FUSE_PYTHON_BIN}" ]]; then
  echo "Fuse python not found or not executable: ${FUSE_PYTHON_BIN}"
  exit 2
fi
if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "Inference script not found: ${INFER_SCRIPT}"
  exit 2
fi
if [[ ! -f "${FUSE_SCRIPT}" ]]; then
  echo "Fuse script not found: ${FUSE_SCRIPT}"
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

TOTAL_SCENES="$(wc -l < "${SPLIT_FILE}")"
TASKS_NEEDED="$(( (TOTAL_SCENES + SCENES_PER_TASK - 1) / SCENES_PER_TASK ))"
MAX_TASK_ID="$((TASKS_NEEDED - 1))"

START_IDX="$((SLURM_ARRAY_TASK_ID * SCENES_PER_TASK))"
END_IDX="$((START_IDX + SCENES_PER_TASK))"
if (( END_IDX > TOTAL_SCENES )); then
  END_IDX="${TOTAL_SCENES}"
fi

echo "Running on host $(hostname)"
echo "Workflow step: dl3dv_outdoor_01_submit_inference"
echo "Starting time: $(date)"
echo "Array task id: ${SLURM_ARRAY_TASK_ID}"
echo "Total scenes: ${TOTAL_SCENES}"
echo "Scenes per task: ${SCENES_PER_TASK}"
echo "Expected max task id for this split: ${MAX_TASK_ID}"
echo "Index range for this task: [${START_IDX}, ${END_IDX})"
echo ""

if (( START_IDX >= TOTAL_SCENES )); then
  echo "Task index is out of range for this split; nothing to run."
  echo "Finished time: $(date)"
  exit 0
fi

mapfile -t SCENE_RELPATHS < <(sed -n "$((START_IDX + 1)),$((END_IDX))p" "${SPLIT_FILE}")

INFER_EXIT=0
if "${PYTHON_BIN}" "${INFER_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --split-file "${SPLIT_FILE}" \
  --model-id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --device cuda \
  --start-idx "${START_IDX}" \
  --end-idx "${END_IDX}" \
  --pose-condition \
  --undistort \
  --save-sky-mask \
  --save-depth100-mask \
  --process-res 546 \
  --save-depth-conf-result \
  --exclude-depth-above-100-for-points \
  --skip-existing; then
  INFER_EXIT=0
else
  INFER_EXIT=$?
fi

FUSE_DONE=0
FUSE_SKIPPED=0
FUSE_FAILED=0
for scene_rel in "${SCENE_RELPATHS[@]}"; do
  if [[ -z "${scene_rel}" ]]; then
    continue
  fi

  scene_root="${DATA_ROOT}/${scene_rel}"
  results_dir="${scene_root}/da3_streaming_output/results_output"
  if [[ ! -d "${results_dir}" ]]; then
    echo "[FUSE][SKIP] ${scene_rel}: missing ${results_dir}"
    FUSE_SKIPPED="$((FUSE_SKIPPED + 1))"
    continue
  fi

  echo "[FUSE] ${scene_rel}"
  if "${FUSE_PYTHON_BIN}" "${FUSE_SCRIPT}" \
    --scene-root "${scene_root}" \
    --streaming-dir da3_streaming_output \
    --results-dir results_output \
    --images-dir images_4 \
    --output-file points_da3_fused.ply; then
    FUSE_DONE="$((FUSE_DONE + 1))"
  else
    echo "[FUSE][ERROR] ${scene_rel}"
    FUSE_FAILED="$((FUSE_FAILED + 1))"
  fi
done

echo ""
echo "Inference exit code: ${INFER_EXIT}"
echo "Fuse summary: done=${FUSE_DONE}, skipped=${FUSE_SKIPPED}, failed=${FUSE_FAILED}"
echo ""
echo "Finished time: $(date)"

if (( INFER_EXIT != 0 || FUSE_FAILED > 0 )); then
  exit 1
fi
