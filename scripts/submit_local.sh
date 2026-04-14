#!/bin/bash

# Ensure START_ID and END_ID are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <START_ID> <END_ID>"
    echo "Example: $0 0 100 (Processes indices 0 through 100 inclusive)"
    exit 1
fi

START_ID=$1
END_ID=$2

# 1. Environment Setup (matching your SLURM script)
source /home/yli7/.bashrc
micromamba activate da3
module load gnu12/12.4.0

cd /home/yli7/repos/Depth-Anything-3 || { echo "Failed to enter Depth-Anything-3 repo"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Path Variables
PYTHON_BIN="/home/yli7/local/micromamba/envs/da3/bin/python"
FUSE_PYTHON_BIN="${FUSE_PYTHON_BIN:-${PYTHON_BIN}}"
INFER_SCRIPT="scripts/da3_streaming_dl3dv_dev.py"
FUSE_SCRIPT="/home/yli7/scratch2/datasets/dataset_tools/dl3dv/fuse_da3_depth_maps.py"

CONFIG_PATH="/home/yli7/repos/Depth-Anything-3/da3_streaming/configs/base_config.yaml"
DATA_ROOT="/home/yli7/scratch2/datasets/dl3dv_960p"
SPLIT_FILE="/home/yli7/scratch2/datasets/dl3dv_960p/metadata/da3_missing_scenes_outdoor_min200.txt"

# 3. Basic Validation Checks
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found: ${PYTHON_BIN}"; exit 2; fi
if [[ ! -f "${INFER_SCRIPT}" ]]; then echo "Infer script not found: ${INFER_SCRIPT}"; exit 2; fi
if [[ ! -f "${FUSE_SCRIPT}" ]]; then echo "Fuse script not found: ${FUSE_SCRIPT}"; exit 2; fi
if [[ ! -d "${DATA_ROOT}" ]]; then echo "Data root not found: ${DATA_ROOT}"; exit 2; fi
if [[ ! -f "${SPLIT_FILE}" ]]; then echo "Split file not found: ${SPLIT_FILE}"; exit 2; fi

TOTAL_SCENES="$(wc -l < "${SPLIT_FILE}")"

echo "Running locally on host $(hostname)"
echo "Starting time: $(date)"
echo "Processing Index Range: [${START_ID}, ${END_ID}]"
echo "==========================================================="

# 4. Sequential Execution Loop
for (( idx=${START_ID}; idx<=${END_ID}; idx++ )); do
    if (( idx >= TOTAL_SCENES )); then
        echo "Index ${idx} is out of bounds (Max is $((TOTAL_SCENES-1))). Stopping here."
        break
    fi

    # Read the specific scene relative path from the split file (sed is 1-indexed)
    scene_rel=$(sed -n "$((idx + 1))p" "${SPLIT_FILE}")
    if [[ -z "${scene_rel}" ]]; then
        continue
    fi
    
    scene_root="${DATA_ROOT}/${scene_rel}"
    echo "--- Index ${idx} | Scene: ${scene_rel} ---"
    
    # Step A: Check & Run INFER
    points_ply="${scene_root}/points_da3.ply"
    if [[ ! -f "${points_ply}" ]]; then
        echo "[INFER] points_da3.ply not found. Running inference..."
        
        "${PYTHON_BIN}" "${INFER_SCRIPT}" \
          --config "${CONFIG_PATH}" \
          --data-root "${DATA_ROOT}" \
          --split-file "${SPLIT_FILE}" \
          --model-id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
          --device cuda \
          --start-idx "${idx}" \
          --end-idx "$((idx + 1))" \
          --pose-condition \
          --undistort \
          --save-sky-mask \
          --save-depth100-mask \
          --process-res 546 \
          --save-depth-conf-result \
          --exclude-depth-above-100-for-points \
          --skip-existing
          
        if [ $? -ne 0 ]; then
            echo "[INFER][ERROR] Failed on ${scene_rel}"
        fi
    else
        echo "[INFER][SKIP] points_da3.ply already exists."
    fi

    # Step B: Check & Run FUSE
    fused_ply="${scene_root}/points_da3_fused.ply"
    if [[ ! -f "${fused_ply}" ]]; then
        echo "[FUSE] points_da3_fused.ply not found. Running fusion..."
        
        "${FUSE_PYTHON_BIN}" "${FUSE_SCRIPT}" \
          --scene-root "${scene_root}" \
          --streaming-dir da3_streaming_output \
          --results-dir results_output \
          --images-dir images_4 \
          --output-file points_da3_fused.ply
          
        if [ $? -ne 0 ]; then
            echo "[FUSE][ERROR] Failed on ${scene_rel}"
        fi
    else
        echo "[FUSE][SKIP] points_da3_fused.ply already exists."
    fi
    echo ""
done

echo "Finished processing range [${START_ID}, ${END_ID}] at $(date)"