# Depth-Anything-3 Local Agent Context

## Maintenance Rule
- `AGENTS.md` is startup context for this repo and should be kept in sync with code changes.
- When changing workflow entrypoints, dataset assumptions, CLI flags, save formats, or pickup commands, update this file in the same patch.

## Quick Pickup
This repo has two DL3DV streaming workflows plus one shared posthoc summarizer.
Logical phase numbers are shared across local runners and Slurm submit wrappers: the `*_submit_*.sh` script launches the same phase as its matching non-submit entrypoint.

Outdoor workflow:
- `scripts/dl3dv_outdoor_01_streaming_inference.py`: run DA3 streaming on posed DL3DV outdoor scenes.
  Inputs: `data-root`, outdoor split file, per-scene `transforms.json`, `images_4/`.
  Outputs: `<scene>/points_da3.ply`, masks, `da3_streaming_output/`, optional `points_da3_fused.ply`.
- `scripts/dl3dv_outdoor_01_submit_inference.sh`: Slurm array wrapper for outdoor phase 01.
- `scripts/dl3dv_outdoor_02_analyze_existing.sh`: analyze already-generated outdoor scene outputs.
  Inputs: outdoor scene roots with `da3_streaming_output/`.
  Outputs: analysis run dir with `success.txt`, `failed.txt`, `manifest.tsv`, per-scene logs.
- `scripts/dl3dv_outdoor_02_submit_analysis.sh`: Slurm array wrapper for outdoor phase 02.

Eval workflow:
- `scripts/dl3dv_eval_01_stage_scene_zips.py`: stage raw evaluation tar archives into per-scene zip files in split order.
  Inputs: `evaluation/images_tar`, eval split file.
  Outputs: `evaluation/images/<scene>.zip`.
- `scripts/dl3dv_eval_02_streaming_inference.py`: run DA3 streaming on staged eval scene zips.
  Inputs: eval zip root, eval split file.
  Outputs: `evaluation/<scene>/points_da3.ply`, masks, `da3_streaming_output/`, `scene_manifest.json`.
- `scripts/dl3dv_eval_02_run_inference.sh`: local shell wrapper for eval phase 02 with the standard defaults.
- `scripts/dl3dv_eval_02_submit_inference.sh`: Slurm array wrapper for eval phase 02.
- `scripts/dl3dv_eval_03_analyze_existing.sh`: analyze already-generated eval scene outputs.
  Inputs: `evaluation/<scene>` directories with `scene_manifest.json`.
  Outputs: analysis run dir with `success.txt`, `failed.txt`, `manifest.tsv`, per-scene logs.
- `scripts/dl3dv_eval_03_submit_analysis.sh`: Slurm array wrapper for eval phase 03.
- `scripts/dl3dv_eval_04_write_transforms.py`: write `transforms_da3.json` for eval scenes.
  Inputs: completed eval outputs plus an optional `--scene-ids-file` allowlist.
  Outputs: `<scene>/transforms_da3.json`.

Shared utility:
- `scripts/dl3dv_summarize_analysis.py`: summarize diagnostics and scene-level geometry metrics across a split.

## Environment
- Python env: `~/local/micromamba/envs/da3/bin/python`
- Expected GPU: 24GB class device such as L4 or A5000
- Streaming loop dependencies: `faiss`, `pypose`, `triton`, `open3d`

## Outdoor Assumptions
- Dataset root: `/home/yli7/scratch2/datasets/dl3dv_960p`
- Outdoor split: `/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200.txt`
- Scene poses come from `<scene>/transforms.json`
- Frames are read from `images_4/`, not `images/`
- Outdoor runs are normally pose-conditioned and undistorted
- `--shared-intrinsics` is only valid for unposed runs; do not combine it with pose conditioning

## Eval Assumptions
- Raw tar root: `/home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images_tar`
- Staged zip root: `/home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images`
- Eval split file: `/home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt`
- Per-scene output root: `/home/yli7/scratch2/datasets/dl3dv_960p/evaluation`
- Eval runner processes scenes in split-file order; it does not enumerate all zips blindly
- Eval inputs are read directly from zip archives without extraction
- Eval inference uses `nerfstudio/images_4/frame_*.png` only
- The active eval rerun subset is the filtered 50-scene split with consistent `960x540` `nerfstudio/images_4`
- Excluded from that split:
  - `18ae5d387baba346017f225c7661c885ab4ce1ef7dd028a9ad80773002528557`
  - `61b27b6168d547f1a0b08e62be1d8a490dffa280a8e4fce518a577c97190eeba`
  - `7a2e7f96ebe1a1023eb9a873bedd9cdcbdb490b628524e3b7522073fb87a8aa5`
  - `ec0d0f130fbe697564dbac8df4b8a07665868a6ed0441cd29458607c67f61836`
- Eval runs are unposed, do not use input intrinsics, and do not undistort frames
- `scripts/dl3dv_eval_02_run_inference.sh` enables `--shared-intrinsics` by default; disable with `--no-shared-intrinsics`
- `scripts/dl3dv_eval_01_stage_scene_zips.py` is the canonical tar-to-zip staging step; do not untar scene directories onto `scratch2`

## Outputs
For each scene `<scene_root>`:
- Final point cloud: `<scene_root>/points_da3.ply`
- Optional sky masks: `<scene_root>/sky_mask/masks.npz`
- Optional depth100 masks: `<scene_root>/depth100_mask/masks.npz`
- Optional per-frame depth/conf dumps: `<scene_root>/da3_streaming_output/results_output/<frame_stem>.npz`
- Outdoor fusion output, when run from the outdoor submitter: `<scene_root>/points_da3_fused.ply`

Extra eval outputs:
- `<scene_root>/scene_manifest.json`
  - Records zip source, selected frames, selected image size, runtime options, chunk indices, and scene status
  - Records the resolved shared intrinsics matrix and source chunk index when `--shared-intrinsics` is enabled
- `<scene_root>/transforms_da3.json`
  - Written from `scene_manifest.json`, `da3_streaming_output/camera_poses.txt`, `da3_streaming_output/intrinsic.txt`, and `points_da3.ply`
  - `frames[].file_path` is rewritten to `nerfstudio/images_4/frame_*.png`
  - Top-level `w/h` are fixed to the selected eval image size, currently `960x540`
  - Intrinsics are scaled from the DA3 processing plane to that `960x540` image plane
  - Distortion terms `k1`, `k2`, `p1`, `p2` are written as `0.0`
  - `bbox_obb_points` is derived from `points_da3.ply`
  - Eval transforms are analysis-gated: only scenes listed in analysis `success.txt` should be passed through `--scene-ids-file`

## Geometry And Filtering Rules
- Input `transform_matrix` in DL3DV is treated as OpenGL `c2w`
- Conversion to OpenCV world:
  - `c2w_cv = c2w_ogl @ diag(1, -1, -1, 1)`
  - `w2c_cv = inv(c2w_cv)`
- Pose conditioning uses `align_to_input_ext_scale=False` to preserve DA3 metric depth scale behavior
- Outdoor undistortion assumes an `OPENCV` camera model and scales intrinsics/distortion from `transforms.json` image size to the actual frame size
- Base confidence filtering follows the original DA3 streaming path
- Additional filtering can exclude sky pixels and optionally exclude depth-above-threshold pixels, default `100m`
- `--shared-intrinsics` is a full-geometry override for unposed streaming: it freezes `results_output/*.npz`, `intrinsic.txt`, loop geometry, and point projection to one scene-wide matrix

## Current Validated State
- The outdoor streaming workflow exists and has been run on the DL3DV outdoor split
- The eval workflow exists and has been rerun on the filtered 50-scene eval split
- Eval transforms are gated by analysis `success.txt` through `scripts/dl3dv_eval_04_write_transforms.py --scene-ids-file ...`
- The latest eval rerun produced analysis-passed `transforms_da3.json` for all 50 scenes in the current filtered split

## Common Commands
Outdoor single-scene smoke run:
```bash
~/local/micromamba/envs/da3/bin/python scripts/dl3dv_outdoor_01_streaming_inference.py \
  --start-idx 0 --end-idx 1 \
  --pose-condition --undistort \
  --save-sky-mask --save-depth100-mask \
  --exclude-depth-above-100-for-points
```

Outdoor cluster inference:
```bash
sbatch scripts/dl3dv_outdoor_01_submit_inference.sh
```

Outdoor analysis:
```bash
sbatch scripts/dl3dv_outdoor_02_submit_analysis.sh
```

Eval single-scene smoke run:
```bash
~/local/micromamba/envs/da3/bin/python scripts/dl3dv_eval_01_stage_scene_zips.py \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt \
  --tar-root /home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images_tar \
  --zip-root /home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images \
  --start-idx 0 --end-idx 1

~/local/micromamba/envs/da3/bin/python scripts/dl3dv_eval_02_streaming_inference.py \
  --zip-root /home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images \
  --output-root /home/yli7/scratch2/datasets/dl3dv_960p/evaluation \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt \
  --start-idx 0 --end-idx 1 \
  --process-res 546 \
  --save-sky-mask --save-depth100-mask \
  --save-depth-conf-result \
  --shared-intrinsics \
  --exclude-depth-above-100-for-points \
  --skip-existing
```

Eval cluster inference:
```bash
sbatch scripts/dl3dv_eval_02_submit_inference.sh
```

Eval analysis:
```bash
sbatch scripts/dl3dv_eval_03_submit_analysis.sh
```

Eval transforms:
```bash
~/local/micromamba/envs/da3/bin/python scripts/dl3dv_eval_04_write_transforms.py \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt \
  --output-root /home/yli7/scratch2/datasets/dl3dv_960p/evaluation \
  --zip-root /home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images \
  --scene-ids-file /path/to/analysis/success.txt
```

Posthoc analysis summary:
```bash
~/local/micromamba/envs/da3/bin/python scripts/dl3dv_summarize_analysis.py \
  --data-root /home/yli7/scratch2/datasets/dl3dv_960p/evaluation \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt \
  --output-dir /home/yli7/repos/Depth-Anything-3/outputs/analyze_existing_dl3dv_evaluation
```

## Notes
- Default model source is HuggingFace `depth-anything/DA3NESTED-GIANT-LARGE-1.1`
- `da3_streaming/weights/` local files are optional when using `--model-id`
- Keep `max-frames-per-scene` unset for normal loop-closure behavior
- `scripts/dl3dv_eval_02_submit_inference.sh` passes `--shared-intrinsics` explicitly so Slurm logs reflect the intended eval geometry mode
- Re-running an eval scene into an existing scene root overwrites prior generated scene artifacts before writing the new ns4-based outputs
- `scripts/dl3dv_summarize_analysis.py` reports explicit metric definitions for point-cloud extents, OBB extent, `obb_to_fused_extent_ratio`, and adjacent-frame camera-step statistics including `camera_step_max_to_fused_extent_ratio`
- `scripts/dl3dv_summarize_analysis.py` prints periodic progress during full-scene scans; default cadence is every 100 scenes and can be changed with `--progress-every`
- Conservative outdoor split filtering is based on existing `all_scene_metrics.csv` and writes `/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200_filtered.txt`
- Current conservative exclusion rules are:
  - `camera_step_max_to_fused_extent_ratio > 0.5`
  - `severity == critical`
  - `severity == watchlist` with any numeric-outlier reason in `severity_reasons`: `points_da3_extent_outlier`, `obb_extent_outlier`, `camera_step_median_outlier`
