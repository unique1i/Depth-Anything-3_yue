# Depth-Anything-3 Local Agent Context

## Maintenance Rule
- `AGENTS.md` is startup context for this repo and should be kept in sync with code changes.
- When changing supported datasets, CLI flags, save formats, renderer metadata assumptions, input formats, or debug workflows, update this file in the same patch.
- There is no built-in automatic sync from the runtime or the agent platform. The practical enforcement mechanism is to treat `AGENTS.md` updates as part of the implementation change.

## Scope
This repo contains DA3 base inference and `da3_streaming` long-sequence inference.
The DL3DV outdoor streaming workflow is implemented by:
- `scripts/da3_streaming_dl3dv.py`
- `scripts/da3_streaming_dl3dv_eval.py`
- `scripts/write_transforms_da3_dl3dv_eval.py`
- `da3_streaming/da3_streaming.py` (extended runtime hooks)
- `da3_streaming/loop_utils/loop_detector.py` (zip-backed loop image loading)
- `src/depth_anything_3/model/da3.py` (nested sky mask exposure)

## Environment
- Python env: `~/local/micromamba/envs/da3/bin/python`
- Expected GPU: 24GB class device (L4/A5000 style)
- Runtime dependencies required for streaming loop path: `faiss`, `pypose`, `triton`, `open3d`

## Dataset Assumptions (DL3DV Outdoor)
- Root: `/home/yli7/scratch2/datasets/dl3dv_960p`
- Outdoor split: `/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200.txt`
- Scene transforms: `<scene>/transforms.json`
- Actual frames are loaded from `images_4/` (not `images/`)
- Unposed runs can optionally use `--shared-intrinsics`, which freezes intrinsics to the median estimate from the first non-loop chunk; posed runs must not use this flag

## Dataset Assumptions (DL3DV Evaluation Zips)
- Zip root: `/home/yli7/scratch/datasets/dl3dv_960p/evaluation/images`
- Eval split file: `/home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt`
- Runner default synthetic output root: `/home/yli7/scratch/datasets/dl3dv_960p/evaluation/da3_streaming`
- Runner processes scenes in split-file order; it does not enumerate all zips blindly
- Inputs are read directly from each zip archive without extraction
- Inference uses `nerfstudio/images_4/frame_*.png` only
- The authoritative clean eval subset is 51 scenes with readable, consistent `960x540` `nerfstudio/images_4`
- The cleaned split excludes:
  - `18ae5d387baba346017f225c7661c885ab4ce1ef7dd028a9ad80773002528557`
  - `61b27b6168d547f1a0b08e62be1d8a490dffa280a8e4fce518a577c97190eeba`
  - `7a2e7f96ebe1a1023eb9a873bedd9cdcbdb490b628524e3b7522073fb87a8aa5`
  - `ec0d0f130fbe697564dbac8df4b8a07665868a6ed0441cd29458607c67f61836`
- There is no pose conditioning, no provided intrinsics, and no undistortion in this workflow
- Unposed evaluation runs default to per-frame predicted intrinsics; `--shared-intrinsics` switches to one scene-wide matrix estimated from the first non-loop chunk
- `scripts/run_da3_streaming_dl3dv_eval.sh` now enables `--shared-intrinsics` by default; disable explicitly with `--no-shared-intrinsics`
- Loop closure remains enabled by passing explicit zip-backed image lists/loaders into `DA3_Streaming` and SALAD
- `scripts/write_transforms_da3_dl3dv_eval.py` writes `transforms_da3.json` for completed eval scenes by directly copying DA3 `camera_poses.txt`; it does not support any SIM(3) align-to-DA3 mode

## Pose Convention
- Input `transform_matrix` in DL3DV is treated as OpenGL `c2w`.
- Conversion to OpenCV world:
  - `c2w_cv = c2w_ogl @ diag(1, -1, -1, 1)`
  - `w2c_cv = inv(c2w_cv)`
- Pose conditioning uses `align_to_input_ext_scale=False` to preserve DA3 metric depth scale behavior.

## Undistortion
- Camera model assumed `OPENCV` style from transforms keys (`k1`, `k2`, `p1`, `p2`, optional `k3..k6`).
- Intrinsics/distortion are scaled from transforms `w/h` to actual frame size (`images_4`, typically 960x540).
- Undistortion uses cached `cv2.initUndistortRectifyMap` + `cv2.remap` for efficiency.

## Outputs
For each scene `<scene_root>`:
- Final point cloud: `<scene_root>/points_da3.ply`
- Optional sky masks: `<scene_root>/sky_mask/masks.npz`
  - Keys are frame stems
  - Values are `int8` masks: `1` sky, `0` non-sky
- Optional depth100 masks: `<scene_root>/depth100_mask/masks.npz`
  - Keys are frame stems
  - Values are `int8` masks: `1` where `(depth > 100m) OR sky`, else `0`
- Optional per-frame depth/conf dumps (disabled by default in DL3DV runner):
  - `<scene_root>/da3_streaming_output/results_output/<frame_stem>.npz`
  - Contains `depth`, `conf`, `intrinsics` (+ `extrinsics`, `s`, `R`, `T` if debug extras are enabled)
- Evaluation-zip scenes also store `<scene_root>/scene_manifest.json`
  - Includes zip source path, fixed `source_image_tree="nerfstudio/images_4"`, selected frame keys, selected image size, chunk indices, runtime options, and scene status
  - When `--shared-intrinsics` is enabled, runtime metadata records the resolved shared intrinsics matrix and the source chunk index
- Evaluation-zip completed scenes can also store `<scene_root>/transforms_da3.json`
  - Written from `scene_manifest.json`, `da3_streaming_output/camera_poses.txt`, `da3_streaming_output/intrinsic.txt`, and `points_da3.ply`
  - `frames[].file_path` is rewritten to the scene-local path `nerfstudio/images_4/frame_*.png`
  - Top-level `w/h` are fixed to the selected `nerfstudio/images_4` size (`960x540` for the cleaned eval split)
  - Intrinsics are written on that `960x540` image plane by scaling the DA3 `intrinsic.txt` values from the processing plane (for current eval runs, typically `546x308`)
  - No `resize` key is written for eval scenes
  - `k1`, `k2`, `p1`, `p2` are all `0.0`
  - `bbox_obb_points` is computed from `points_da3.ply`
  - Skipped eval scenes do not get `transforms_da3.json`

## Streaming Filtering Behavior
- Base confidence filtering is unchanged from original DA3 streaming.
- Additional filtering can be enabled:
  - Always exclude sky pixels when sky is available.
  - Optionally exclude depth-above-threshold pixels (default threshold: 100m).

## Common Commands
Single-scene smoke run (by index range):
```bash
~/local/micromamba/envs/da3/bin/python scripts/da3_streaming_dl3dv.py \
  --start-idx 0 --end-idx 1 \
  --pose-condition --undistort \
  --save-sky-mask --save-depth100-mask \
  --exclude-depth-above-100-for-points
```

Cluster submission:
```bash
sbatch submit.sh
```

Evaluation-zip single-scene smoke run:
```bash
~/local/micromamba/envs/da3/bin/python scripts/da3_streaming_dl3dv_eval.py \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt \
  --start-idx 0 --end-idx 1 \
  --process-res 546 \
  --save-sky-mask --save-depth100-mask \
  --save-depth-conf-result \
  --shared-intrinsics \
  --exclude-depth-above-100-for-points \
  --skip-existing
```

Evaluation-zip cluster submission:
```bash
sbatch submit_dl3dv_eval.sh
sbatch submit_dl3dv_eval_analyze.sh
```

Evaluation-zip `transforms_da3.json` generation:
```bash
~/local/micromamba/envs/da3/bin/python scripts/write_transforms_da3_dl3dv_eval.py \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt \
  --output-root /home/yli7/scratch/datasets/dl3dv_960p/evaluation/da3_streaming \
  --zip-root /home/yli7/scratch/datasets/dl3dv_960p/evaluation/images
```

## Notes
- Default model source is HuggingFace `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
- `da3_streaming/weights/` local files are optional when using `--model-id`.
- Keep `max-frames-per-scene` unset for normal loop-closure behavior.
- `--shared-intrinsics` is a full-geometry override for unposed streaming; it freezes `results_output/*.npz`, `intrinsic.txt`, loop geometry, and point projection to one scene-wide intrinsics matrix.
- `submit_dl3dv_eval.sh` passes `--shared-intrinsics` explicitly so Slurm logs reflect the intended eval geometry mode.
- Re-running an eval scene into an existing scene root overwrites prior generated scene artifacts before writing the new ns4-based outputs.
- `scripts/analyze_existing_dl3dv_eval.sh` reads the eval split file and analyzes synthetic scene outputs under the evaluation output root in split order.
- Dataset-level analysis summaries from `scripts/summarize_existing_dl3dv_analysis.py` now include explicit metric definitions for point-cloud extents, OBB extent, `obb_to_fused_extent_ratio`, and adjacent-frame camera-step statistics including `camera_step_max_to_fused_extent_ratio`.
- `scripts/summarize_existing_dl3dv_analysis.py` prints periodic progress during full-scene scans; default cadence is every 100 scenes and can be changed with `--progress-every`.
- Conservative outdoor split filtering is based on existing `all_scene_metrics.csv` and writes `/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200_filtered.txt`.
- Current conservative exclusion rules are:
  - `camera_step_max_to_fused_extent_ratio > 0.5`
  - `severity == critical`
  - `severity == watchlist` with any numeric-outlier reason in `severity_reasons`: `points_da3_extent_outlier`, `obb_extent_outlier`, `camera_step_median_outlier`
