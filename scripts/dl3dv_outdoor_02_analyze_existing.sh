#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/yli7/local/micromamba/envs/da3/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/da3_streaming/configs/base_config.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/yli7/scratch2/datasets/dl3dv_960p}"
SPLIT_FILE="${SPLIT_FILE:-/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200.txt}"
SCENE_OUTPUT_SUBDIR="${SCENE_OUTPUT_SUBDIR:-da3_streaming_output}"
DEPTH_THRESHOLD_M="${DEPTH_THRESHOLD_M:-100}"
WORKERS="${WORKERS:-8}"
START_IDX=0
END_IDX=""
RUN_DIR=""
POSE_CONDITION=1
UNDISTORT=1
EXCLUDE_DEPTH_ABOVE_100=1

usage() {
  cat <<'EOF'
Step 02: analyze existing DL3DV outdoor outputs.

Usage:
  scripts/dl3dv_outdoor_02_analyze_existing.sh [options]

Options:
  --python-bin PATH
  --config PATH
  --data-root PATH
  --split-file PATH
  --scene-output-subdir NAME
  --depth-threshold-m FLOAT
  --workers INT
  --start-idx INT
  --end-idx INT
  --run-dir PATH
  --pose-condition / --no-pose-condition
  --undistort / --no-undistort
  --exclude-depth-above-100-for-points / --no-exclude-depth-above-100-for-points

Examples:
  scripts/dl3dv_outdoor_02_analyze_existing.sh
  scripts/dl3dv_outdoor_02_analyze_existing.sh --start-idx 0 --end-idx 100
EOF
}

while (($# > 0)); do
  case "$1" in
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --split-file) SPLIT_FILE="$2"; shift 2 ;;
    --scene-output-subdir) SCENE_OUTPUT_SUBDIR="$2"; shift 2 ;;
    --depth-threshold-m) DEPTH_THRESHOLD_M="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --end-idx) END_IDX="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --pose-condition) POSE_CONDITION=1; shift ;;
    --no-pose-condition) POSE_CONDITION=0; shift ;;
    --undistort) UNDISTORT=1; shift ;;
    --no-undistort) UNDISTORT=0; shift ;;
    --exclude-depth-above-100-for-points) EXCLUDE_DEPTH_ABOVE_100=1; shift ;;
    --no-exclude-depth-above-100-for-points) EXCLUDE_DEPTH_ABOVE_100=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root not found: ${DATA_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "Split file not found: ${SPLIT_FILE}" >&2
  exit 2
fi

TOTAL_SCENES="$(wc -l < "${SPLIT_FILE}")"
if [[ -z "${END_IDX}" ]]; then
  END_IDX="${TOTAL_SCENES}"
fi

if ! [[ "${START_IDX}" =~ ^[0-9]+$ ]] || ! [[ "${END_IDX}" =~ ^[0-9]+$ ]]; then
  echo "start/end must be non-negative integers." >&2
  exit 2
fi
if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]] || (( WORKERS <= 0 )); then
  echo "workers must be a positive integer." >&2
  exit 2
fi
if (( START_IDX < 0 || END_IDX < 0 || START_IDX > END_IDX )); then
  echo "Invalid index range: [${START_IDX}, ${END_IDX})" >&2
  exit 2
fi
if (( START_IDX >= TOTAL_SCENES )); then
  echo "Start index ${START_IDX} is out of range for ${TOTAL_SCENES} scenes."
  exit 0
fi
if (( END_IDX > TOTAL_SCENES )); then
  END_IDX="${TOTAL_SCENES}"
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${REPO_ROOT}/outputs/analyze_existing_dl3dv_$(date +%Y%m%d_%H%M%S)"
fi
LOG_DIR="${RUN_DIR}/logs"
STATUS_DIR="${RUN_DIR}/status"
mkdir -p "${LOG_DIR}"
mkdir -p "${STATUS_DIR}"

SUCCESS_LIST="${RUN_DIR}/success.txt"
FAILED_LIST="${RUN_DIR}/failed.txt"
MANIFEST="${RUN_DIR}/manifest.tsv"
: > "${SUCCESS_LIST}"
: > "${FAILED_LIST}"
printf "idx\tscene_rel\tstatus\tlog_path\n" > "${MANIFEST}"

mapfile -t SCENE_RELPATHS < <(sed -n "$((START_IDX + 1)),$((END_IDX))p" "${SPLIT_FILE}")

echo "Run dir: ${RUN_DIR}"
echo "Total scenes in split: ${TOTAL_SCENES}"
echo "Analyzing range: [${START_IDX}, ${END_IDX})"
echo "Workers: ${WORKERS}"
echo "Scene output subdir: ${SCENE_OUTPUT_SUBDIR}"
echo ""

run_one() {
  local idx="$1"
  local scene_rel="$2"
  local scene_root="${DATA_ROOT}/${scene_rel}"
  local log_path="${LOG_DIR}/$(printf "%05d" "${idx}").log"
  local status_path="${STATUS_DIR}/$(printf "%05d" "${idx}").tsv"

  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/dl3dv_outdoor_01_streaming_inference.py"
    --config "${CONFIG_PATH}"
    --analyze-existing-scene "${scene_root}"
    --scene-output-subdir "${SCENE_OUTPUT_SUBDIR}"
    --depth-threshold-m "${DEPTH_THRESHOLD_M}"
  )
  if (( POSE_CONDITION == 1 )); then
    cmd+=(--pose-condition)
  else
    cmd+=(--no-pose-condition)
  fi
  if (( UNDISTORT == 1 )); then
    cmd+=(--undistort)
  else
    cmd+=(--no-undistort)
  fi
  if (( EXCLUDE_DEPTH_ABOVE_100 == 1 )); then
    cmd+=(--exclude-depth-above-100-for-points)
  else
    cmd+=(--no-exclude-depth-above-100-for-points)
  fi

  if "${cmd[@]}" > "${log_path}" 2>&1; then
    printf "%s\t%s\t%s\t%s\n" "${idx}" "${scene_rel}" "ok" "${log_path}" > "${status_path}"
  else
    printf "%s\t%s\t%s\t%s\n" "${idx}" "${scene_rel}" "fail" "${log_path}" > "${status_path}"
  fi
}

active_jobs=0
for offset in "${!SCENE_RELPATHS[@]}"; do
  idx=$((START_IDX + offset))
  scene_rel="${SCENE_RELPATHS[offset]}"
  echo "[${idx}] ${scene_rel}"
  run_one "${idx}" "${scene_rel}" &
  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= WORKERS )); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done

wait

while IFS=$'\t' read -r idx scene_rel status log_path; do
  [[ -z "${idx}" || "${idx}" == "idx" ]] && continue
  printf "%s\t%s\t%s\t%s\n" "${idx}" "${scene_rel}" "${status}" "${log_path}" >> "${MANIFEST}"
  if [[ "${status}" == "ok" ]]; then
    echo "${scene_rel}" >> "${SUCCESS_LIST}"
  else
    echo "${scene_rel}" >> "${FAILED_LIST}"
    echo "  failed, see ${log_path}"
  fi
done < <(sort -t $'\t' -k1,1n "${STATUS_DIR}"/*.tsv 2>/dev/null)

success_count="$(wc -l < "${SUCCESS_LIST}")"
failed_count="$(wc -l < "${FAILED_LIST}")"
echo ""
echo "Analysis done."
echo "Success: ${success_count}"
echo "Failed: ${failed_count}"
echo "Manifest: ${MANIFEST}"
