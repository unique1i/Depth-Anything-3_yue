#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/yli7/local/micromamba/envs/da3/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/da3_streaming/configs/base_config.yaml}"
SPLIT_FILE="${SPLIT_FILE:-/home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yli7/scratch/datasets/dl3dv_960p/evaluation/da3_streaming}"
WORKERS="${WORKERS:-8}"
START_IDX=0
END_IDX=""
RUN_DIR=""

usage() {
  cat <<'EOF'
Usage:
  scripts/analyze_existing_dl3dv_eval.sh [options]

Options:
  --python-bin PATH
  --config PATH
  --split-file PATH
  --output-root PATH
  --workers INT
  --start-idx INT
  --end-idx INT
  --run-dir PATH

Examples:
  scripts/analyze_existing_dl3dv_eval.sh
  scripts/analyze_existing_dl3dv_eval.sh --start-idx 0 --end-idx 10 --workers 2
EOF
}

while (($# > 0)); do
  case "$1" in
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --split-file) SPLIT_FILE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --end-idx) END_IDX="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
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
if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "Split file not found: ${SPLIT_FILE}" >&2
  exit 2
fi
if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Output root not found: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if ! [[ "${START_IDX}" =~ ^[0-9]+$ ]]; then
  echo "start-idx must be a non-negative integer." >&2
  exit 2
fi
if [[ -n "${END_IDX}" ]] && ! [[ "${END_IDX}" =~ ^[0-9]+$ ]]; then
  echo "end-idx must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]] || (( WORKERS <= 0 )); then
  echo "workers must be a positive integer." >&2
  exit 2
fi

mapfile -t SCENE_IDS < <("${PYTHON_BIN}" - "${SPLIT_FILE}" <<'PY'
from pathlib import Path
import sys

for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.endswith(".zip"):
        line = line[:-4]
    print(Path(line.lstrip("/")).name)
PY
)
TOTAL_SCENES="${#SCENE_IDS[@]}"
if [[ -z "${END_IDX}" ]]; then
  END_IDX="${TOTAL_SCENES}"
fi
if (( START_IDX > END_IDX )); then
  echo "Invalid index range: [${START_IDX}, ${END_IDX})" >&2
  exit 2
fi
if (( START_IDX >= TOTAL_SCENES )); then
  echo "Start index ${START_IDX} is out of range for ${TOTAL_SCENES} zip scenes."
  exit 0
fi
if (( END_IDX > TOTAL_SCENES )); then
  END_IDX="${TOTAL_SCENES}"
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${REPO_ROOT}/outputs/analyze_existing_dl3dv_eval_$(date +%Y%m%d_%H%M%S)"
fi
LOG_DIR="${RUN_DIR}/logs"
STATUS_DIR="${RUN_DIR}/status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

SUCCESS_LIST="${RUN_DIR}/success.txt"
FAILED_LIST="${RUN_DIR}/failed.txt"
MANIFEST="${RUN_DIR}/manifest.tsv"
: > "${SUCCESS_LIST}"
: > "${FAILED_LIST}"
printf "idx\tscene_id\tscene_root\tstatus\tlog_path\n" > "${MANIFEST}"

echo "Run dir: ${RUN_DIR}"
echo "Total scenes: ${TOTAL_SCENES}"
echo "Analyzing range: [${START_IDX}, ${END_IDX})"
echo "Workers: ${WORKERS}"
echo ""

run_one() {
  local idx="$1"
  local scene_id="$2"
  local scene_root
  local log_path
  local status_path

  scene_root="${OUTPUT_ROOT}/${scene_id}"
  log_path="${LOG_DIR}/$(printf "%05d" "${idx}").log"
  status_path="${STATUS_DIR}/$(printf "%05d" "${idx}").tsv"

  if [[ ! -d "${scene_root}" ]]; then
    printf "Missing synthetic scene root: %s\n" "${scene_root}" > "${log_path}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${idx}" "${scene_id}" "${scene_root}" "fail" "${log_path}" > "${status_path}"
    return
  fi

  if "${PYTHON_BIN}" "${REPO_ROOT}/scripts/da3_streaming_dl3dv_eval.py" \
    --config "${CONFIG_PATH}" \
    --analyze-existing-scene "${scene_root}" > "${log_path}" 2>&1; then
    printf "%s\t%s\t%s\t%s\t%s\n" "${idx}" "${scene_id}" "${scene_root}" "ok" "${log_path}" > "${status_path}"
  else
    printf "%s\t%s\t%s\t%s\t%s\n" "${idx}" "${scene_id}" "${scene_root}" "fail" "${log_path}" > "${status_path}"
  fi
}

active_jobs=0
for ((offset = 0; offset < END_IDX - START_IDX; offset++)); do
  idx=$((START_IDX + offset))
  scene_id="${SCENE_IDS[idx]}"
  echo "[${idx}] ${scene_id}"
  run_one "${idx}" "${scene_id}" &
  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= WORKERS )); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done

wait

while IFS=$'\t' read -r idx scene_id scene_root status log_path; do
  [[ -z "${idx}" || "${idx}" == "idx" ]] && continue
  printf "%s\t%s\t%s\t%s\t%s\n" "${idx}" "${scene_id}" "${scene_root}" "${status}" "${log_path}" >> "${MANIFEST}"
  if [[ "${status}" == "ok" ]]; then
    echo "${scene_root}" >> "${SUCCESS_LIST}"
  else
    echo "${scene_root}" >> "${FAILED_LIST}"
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
