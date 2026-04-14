#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/yli7/local/micromamba/envs/da3/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/da3_streaming/configs/base_config.yaml}"
ZIP_ROOT="${ZIP_ROOT:-/home/yli7/scratch/datasets/dl3dv_960p/evaluation/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yli7/scratch/datasets/dl3dv_960p/evaluation/da3_streaming_gs}"
DEVICE="${DEVICE:-cuda}"
PROCESS_RES="${PROCESS_RES:-546}"
SCENE_OUTPUT_SUBDIR="${SCENE_OUTPUT_SUBDIR:-da3_streaming_output}"
START_IDX=0
END_IDX=""
MAX_FRAMES_PER_SCENE=-1
SAVE_SKY_MASK=1
SAVE_DEPTH100_MASK=1
SAVE_DEPTH_CONF_RESULT=1
SHARED_INTRINSICS=1
EXCLUDE_DEPTH_ABOVE_100=1
SKIP_EXISTING=1
FAIL_FAST=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_da3_streaming_dl3dv_eval.sh [options] [-- extra_runner_args...]

Options:
  --python-bin PATH
  --config PATH
  --zip-root PATH
  --output-root PATH
  --device NAME
  --process-res INT
  --scene-output-subdir NAME
  --start-idx INT
  --end-idx INT
  --max-frames-per-scene INT
  --save-sky-mask / --no-save-sky-mask
  --save-depth100-mask / --no-save-depth100-mask
  --save-depth-conf-result / --no-save-depth-conf-result
  --shared-intrinsics / --no-shared-intrinsics
  --exclude-depth-above-100-for-points / --no-exclude-depth-above-100-for-points
  --skip-existing / --no-skip-existing
  --fail-fast / --no-fail-fast

Examples:
  scripts/run_da3_streaming_dl3dv_eval.sh --start-idx 0 --end-idx 1
EOF
}

while (($# > 0)); do
  case "$1" in
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --zip-root) ZIP_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --process-res) PROCESS_RES="$2"; shift 2 ;;
    --scene-output-subdir) SCENE_OUTPUT_SUBDIR="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --end-idx) END_IDX="$2"; shift 2 ;;
    --max-frames-per-scene) MAX_FRAMES_PER_SCENE="$2"; shift 2 ;;
    --save-sky-mask) SAVE_SKY_MASK=1; shift ;;
    --no-save-sky-mask) SAVE_SKY_MASK=0; shift ;;
    --save-depth100-mask) SAVE_DEPTH100_MASK=1; shift ;;
    --no-save-depth100-mask) SAVE_DEPTH100_MASK=0; shift ;;
    --save-depth-conf-result) SAVE_DEPTH_CONF_RESULT=1; shift ;;
    --no-save-depth-conf-result) SAVE_DEPTH_CONF_RESULT=0; shift ;;
    --shared-intrinsics) SHARED_INTRINSICS=1; shift ;;
    --no-shared-intrinsics) SHARED_INTRINSICS=0; shift ;;
    --exclude-depth-above-100-for-points) EXCLUDE_DEPTH_ABOVE_100=1; shift ;;
    --no-exclude-depth-above-100-for-points) EXCLUDE_DEPTH_ABOVE_100=0; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --no-skip-existing) SKIP_EXISTING=0; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --no-fail-fast) FAIL_FAST=0; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
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
if [[ ! -d "${ZIP_ROOT}" ]]; then
  echo "Zip root not found: ${ZIP_ROOT}" >&2
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
if ! [[ "${PROCESS_RES}" =~ ^[0-9]+$ ]] || (( PROCESS_RES <= 0 )); then
  echo "process-res must be a positive integer." >&2
  exit 2
fi
if ! [[ "${MAX_FRAMES_PER_SCENE}" =~ ^-?[0-9]+$ ]] || (( MAX_FRAMES_PER_SCENE == 0 || MAX_FRAMES_PER_SCENE < -1 )); then
  echo "max-frames-per-scene must be -1 or a positive integer." >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/da3_streaming_dl3dv_eval.py"
  --config "${CONFIG_PATH}"
  --zip-root "${ZIP_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --device "${DEVICE}"
  --process-res "${PROCESS_RES}"
  --scene-output-subdir "${SCENE_OUTPUT_SUBDIR}"
  --start-idx "${START_IDX}"
  --max-frames-per-scene "${MAX_FRAMES_PER_SCENE}"
)

if [[ -n "${END_IDX}" ]]; then
  cmd+=(--end-idx "${END_IDX}")
fi
if (( SAVE_SKY_MASK == 1 )); then
  cmd+=(--save-sky-mask)
else
  cmd+=(--no-save-sky-mask)
fi
if (( SAVE_DEPTH100_MASK == 1 )); then
  cmd+=(--save-depth100-mask)
else
  cmd+=(--no-save-depth100-mask)
fi
if (( SAVE_DEPTH_CONF_RESULT == 1 )); then
  cmd+=(--save-depth-conf-result)
else
  cmd+=(--no-save-depth-conf-result)
fi
if (( SHARED_INTRINSICS == 1 )); then
  cmd+=(--shared-intrinsics)
else
  cmd+=(--no-shared-intrinsics)
fi
if (( EXCLUDE_DEPTH_ABOVE_100 == 1 )); then
  cmd+=(--exclude-depth-above-100-for-points)
else
  cmd+=(--no-exclude-depth-above-100-for-points)
fi
if (( SKIP_EXISTING == 1 )); then
  cmd+=(--skip-existing)
else
  cmd+=(--no-skip-existing)
fi
if (( FAIL_FAST == 1 )); then
  cmd+=(--fail-fast)
else
  cmd+=(--no-fail-fast)
fi
cmd+=("${EXTRA_ARGS[@]}")

echo "Running on host $(hostname)"
echo "Start: $(date)"
printf 'Command:'
printf ' %q' "${cmd[@]}"
echo ""
"${cmd[@]}"
echo "Finished: $(date)"
