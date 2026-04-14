# DA3 Scene Inference Pipeline

This document captures the implementation plan and concrete decisions used for the modular scene inference pipeline in this folder.

## Scope

The pipeline supports:

1. Single-scene inference from a provided transforms JSON.
2. Dataset-batch inference for ScanNet and ScanNet++ (split-file driven).
3. Pose-conditioned depth inference with optional per-frame depth PNG saving.
4. Global point-cloud fusion via TSDF.
5. Global 3DGS export via a dedicated pass.
6. OOM-safe chunked inference with automatic chunk-size fallback.

Primary files:

- `src/depth_anything_3/pipelines/scene_inference.py`
- `scripts/da3_scene_inference.py`

## Core Design

### 1) Dataset adapters + scene parser

Implemented as `DatasetConfig` entries for:

- `scannet`
- `scannetpp`

Each adapter defines:

- split file location
- dataset root
- transforms JSON relative path
- image path resolver
- target-size resolver
- TSDF defaults

Scene JSON is normalized into `SceneSpec` and `FrameSpec`:

- frame order is preserved exactly as in `frames`
- `transform_matrix` convention is dataset-configurable:
  - ScanNet: `c2w`
  - ScanNet++: custom pose resolver for nerfstudio OpenGL `c2w`
    - conversion used: `c2w_cv = A @ c2w_ogl @ D`, `w2c_cv = inv(c2w_cv)`
    - `A` is axis remap, `D = diag(1, -1, -1, 1)` flips Y/Z camera axes
- per-frame `c2w` and `w2c` are both materialized in normalized frame specs
- intrinsics are loaded from top-level:
  - first preference: `fx/fy/cx/cy`
  - fallback: `fl_x/fl_y/cx/cy`
- intrinsics resize policy is dataset-configurable:
  - ScanNet: keep JSON intrinsics as-is
  - ScanNet++: pre-scale JSON intrinsics from original image size to target size
- image pre-resize policy is dataset-configurable:
  - ScanNet: pre-resize images from 1296x968 to 640x480
  - ScanNet++: pre-resize images from 1752x1168 to 876x584
- ScanNet++ `is_bad` frames are currently included (no filtering)

### 2) Inference engine with OOM-safe chunking

Depth pass is implemented in `_run_scene_depth_pass(...)`.

Flow:

1. Build chunk image paths.
2. Optionally pass chunk extrinsics/intrinsics if pose conditioning is enabled.
3. Call `model.inference(...)` for that chunk with explicit pose-scale handling.
4. On CUDA OOM:
   - run CUDA cleanup
   - halve chunk size
   - retry the same chunk
   - continue until `min_chunk_size`
5. If OOM persists at minimum chunk size:
   - fail the scene with explicit guidance

Implementation notes:

- chunk processing flushes work immediately (no full-scene prediction retention)
- successful chunk size is reused for following chunks
- pose-conditioned depth scaling is configurable via `pose_scale_mode`:
  - `none`: no pose-based scaling
  - `per_chunk`: per-chunk Umeyama scale
  - `global`: one scene-level calibration scale reused for all chunks

### 2.1) `pre_resize_images_to_target` behavior

This flag controls whether each frame is explicitly resized to dataset target size before entering DA3 `InputProcessor`.

Why this matters:

- DA3 still applies its own divisibility-by-14 processing after this step.
- But camera intrinsics in transforms JSON are often defined for a specific nominal resolution.
- If raw images are larger than that nominal resolution and we do not pre-resize, DA3's internal resizing will apply an extra scale to intrinsics.

Current dataset-specific policy:

- ScanNet:
  - raw images: `1296x968`
  - target: `640x480`
  - JSON intrinsics are already scaled for `640x480`
  - therefore:
    - `resize_intrinsics_to_target = False`
    - `pre_resize_images_to_target = True`
- ScanNet++:
  - raw images: `1752x1168`
  - target: `876x584`
  - JSON intrinsics are for original `1752x1168`
  - therefore:
    - `resize_intrinsics_to_target = True`
    - `pre_resize_images_to_target = True`

Rule of thumb:

- If intrinsics are already expressed at target resolution, pre-resize images to target and do not re-scale intrinsics.
- If intrinsics are expressed at raw image resolution, first scale intrinsics to target and pre-resize images to target.

### 3) Depth PNG export policy

When `save_depth=True`:

- each frame depth is encoded as:
  - `uint16 = clip(depth_m * depth_scale, 0, 65535)`
- PNG is saved using the frame stem (`00000.png`, `DSC06795.png`, etc.)
- DA3 snapped inference resolution is accepted internally
- saved PNGs are resized back to target size with nearest-neighbor

### 4) Global point-cloud fusion

TSDF integration runs incrementally during depth pass with Open3D:

- integrates each frame depth + RGB + intrinsics + extrinsics
- exports `global_pointcloud.ply` at scene end
- applies optional voxel downsample after sampling

Dataset defaults:

- ScanNet:
  - `voxel_length=0.02`
  - `sdf_trunc=0.08`
  - `max_depth=10.0`
- ScanNet++:
  - `voxel_length=0.02`
  - `sdf_trunc=0.15`
  - `max_depth=5.0`

### 5) Global 3DGS export

3DGS uses a dedicated scene pass (`_run_scene_gs_pass(...)`):

1. Select frame subset with automatic stride to cap count (`gs_max_frames`).
2. Run one `infer_gs=True` inference call on selected frames.
3. Export GS PLY via `export_to_gs_ply`.
4. Write final output as `global_3dgs.ply`.

OOM behavior for GS:

- halve selected frame cap and retry
- fail clearly if minimum viable frame count is reached

## Output Layout Decision

This was intentionally split for easier artifact management:

1. Managed results root (default):
   - `/home/yli7/repos/Depth-Anything-3/outputs/<dataset>/<scene>/`
   - contains:
     - `run_manifest.json`
     - `global_pointcloud.ply` (if enabled)
     - `global_3dgs.ply` (if enabled)

2. Original scene-root depth folder (only if depth saving is enabled):
   - `<scene_root>/<output_subdir>/` (default `<scene_root>/depth_est_da3/`)
   - contains:
     - per-frame depth PNGs

This behavior is controlled by:

- `RunOptions.output_root`
- `RunOptions.output_subdir`
- `save_depth` toggle

## Manifest and logging

Each scene writes `run_manifest.json` with:

- scene metadata
- run options
- output layout paths
- depth pass summary:
  - effective inference size
  - chunk logs
  - OOM retries
  - depth export stats
  - pointcloud stats
- GS pass summary:
  - selected frames and stride
  - attempt logs
  - final status

Console logs include:

- per-scene start/end
- chunk index ranges
- retries and OOM downscaling
- output directories used

## CLI contract

Entry script: `scripts/da3_scene_inference.py`

Modes:

- `single`
- `batch`

Shared options include:

- model/device
- pose-condition toggle
- save toggles (`depth`, `pointcloud`, `gs`)
- chunk and min-chunk size
- process resolution method
- pose scale controls (`pose-scale-mode`, `pose-scale-calib-frames`)
- alignment toggle
- GS frame cap and view interval
- output routing (`output-root`, `output-subdir`)

### Pose scale modes

`pose_scale_mode` controls how metric scale is applied in pose-conditioned depth fusion.

1. `none`
   - No Umeyama-based depth scaling is applied by this pipeline.
   - Uses raw DA3 depth output from each inference call.
   - Fastest but most fragile for chunked multi-view fusion.
   - Recommended only for debugging.

2. `per_chunk`
   - For each chunk, estimate one Umeyama scale from predicted-vs-input extrinsics, then scale only that chunk's depth.
   - Better than `none`, but chunk-to-chunk scale drift can still create seams and warping.
   - Useful when scenes are short or chunk size is large enough that drift is minimal.

3. `global` (default)
   - Run one calibration pass on a subset of scene frames (`pose_scale_calib_frames`) to estimate a single scene-level scale.
   - Apply the same scale to all chunks in depth pass.
   - Most stable for long scenes and batch processing.
   - Slight extra runtime for the calibration pass.

Recommended usage:

- Use `global` for production multi-scene runs.
- Use `per_chunk` only when global calibration is too expensive.
- Use `none` only for controlled experiments.

## Assumptions locked during implementation

1. `transform_matrix` convention is dataset-dependent and configured in `DatasetConfig`.
2. ScanNet++ `is_bad` frames are included.
3. DA3 size snapping is accepted internally.
4. Saved depth maps are resized to target size.
5. Default model is `depth-anything/DA3NESTED-GIANT-LARGE-1.1`.
6. Depth saving is off by default.
7. Point cloud and GS saving are on by default.
8. GS uses auto stride by default for long scenes.

## Extension guide

To add a new dataset:

1. Add a new `DatasetConfig` entry in `_dataset_configs(...)`.

2. Set data location fields:
   - `split_path`
   - `root_path`
   - `transforms_relpath`

3. Implement path/size resolvers:
   - `image_path_resolver(scene_root, file_path)`:
     - resolve exact input file path from frame `file_path`
     - include suffix fallback logic only if the dataset is inconsistent
   - `target_size_resolver(payload)`:
     - read from JSON (e.g. `resize`) when available, or return fixed dataset target

4. Set pose conversion:
   - `transform_matrix_convention`:
     - `c2w` if frame matrix maps camera to world
     - `w2c` if frame matrix maps world to camera
   - `apply_opengl_to_opencv`:
     - set `True` if poses are in OpenGL-style camera axes and must be converted for this pipeline
   - if dataset conventions are intricate, add `pose_matrix_resolver`:
     - function signature: `resolver(matrix_4x4) -> (c2w, w2c)`
     - this bypasses generic `transform_matrix_convention` handling
     - use this for dataset-specific transforms (ScanNet++ uses this path)

5. Set intrinsics/image scaling policy together:
   - `resize_intrinsics_to_target`
   - `pre_resize_images_to_target`
   - choose the pair using this checklist:
     - Intrinsics already at target resolution:
       - `resize_intrinsics_to_target=False`
       - `pre_resize_images_to_target=True` if images are larger than target
     - Intrinsics at raw resolution:
       - `resize_intrinsics_to_target=True`
       - `pre_resize_images_to_target=True` when inference target is downsampled

6. Set TSDF defaults:
   - `voxel_length`
   - `sdf_trunc`
   - `max_depth`
   - optional `sample_points` and `voxel_downsample`

7. Validate before full batch run:
   - parser smoke test on one scene:
     - confirm one frame resolves to an existing image
     - confirm `w2c @ c2w` is close to identity
   - print resulting intrinsics and verify they match intended target resolution
   - run a short chunked test (for example 24 to 72 frames) and inspect:
     - chunk scale logs
     - fused point cloud sanity
   - if chunk scales vary heavily, keep `pose_scale_mode=global`

## Validation checklist used

1. Parser smoke test on ScanNet scene.
2. Parser smoke test on ScanNet++ scene.
3. Single-scene mode parsing with explicit transforms path.
4. Depth PNG save check (count, dtype, resolution).
5. Pointcloud export existence and non-empty validation.
6. GS export existence validation.
7. OOM fallback validation with intentionally oversized chunk.
