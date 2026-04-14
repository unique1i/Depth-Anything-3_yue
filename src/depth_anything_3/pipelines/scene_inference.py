# Copyright (c) 2026
#
# Scene-level modular inference pipeline for Depth Anything 3.

from __future__ import annotations

import json
import math
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.gs import export_to_gs_ply
from depth_anything_3.utils.memory import cleanup_cuda_memory
from depth_anything_3.utils.pose_align import align_poses_umeyama

DEFAULT_SCANNET_SPLIT = "/home/yli7/scratch2/datasets/scannet/splits/mini_val.txt"
DEFAULT_SCANNET_ROOT = "/home/yli7/scratch2/datasets/scannet/scans"
DEFAULT_SCANNETPP_SPLIT = "/home/yli7/scratch2/datasets/scannetpp_v2/splits/mini_val.txt"
DEFAULT_SCANNETPP_ROOT = "/home/yli7/scratch2/datasets/scannetpp_v2/data"

DEFAULT_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_OUTPUT_ROOT = "/home/yli7/repos/Depth-Anything-3/outputs/without_pose_conditioning"
DEFAULT_OUTPUT_SUBDIR = "depth_est_da3"
DEFAULT_PROCESS_RES_METHOD = "upper_bound_resize"
DEFAULT_POSE_SCALE_MODE = "global"
DEFAULT_POSE_SCALE_CALIB_FRAMES = 72
SUPPORTED_POSE_SCALE_MODES = ("none", "per_chunk", "global")

_SCANNETPP_OPENGL_LEFT_TRANSFORM = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_SCANNETPP_YZ_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


@dataclass(frozen=True)
class TsdfSettings:
    voxel_length: float
    sdf_trunc: float
    max_depth: float
    sample_points: int = 1_000_000
    voxel_downsample: float | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split_path: str
    root_path: str
    transforms_relpath: str
    target_size_resolver: Callable[[dict], tuple[int, int] | None]
    image_path_resolver: Callable[[Path, str], Path]
    tsdf_settings: TsdfSettings
    pose_matrix_resolver: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
    transform_matrix_convention: Literal["c2w", "w2c"] = "c2w"
    apply_opengl_to_opencv: bool = False
    resize_intrinsics_to_target: bool = False
    pre_resize_images_to_target: bool = False


@dataclass(frozen=True)
class FrameSpec:
    index: int
    file_path: str
    image_path: Path
    output_stem: str
    c2w: np.ndarray
    w2c: np.ndarray


@dataclass(frozen=True)
class SceneSpec:
    dataset_name: str
    scene_name: str
    scene_root: Path
    transforms_json: Path
    frames: Sequence[FrameSpec]
    intrinsics: np.ndarray
    target_size: tuple[int, int]
    process_res: int
    transform_matrix_convention: Literal["c2w", "w2c"]
    apply_opengl_to_opencv: bool
    resize_intrinsics_to_target: bool
    pre_resize_images_to_target: bool
    tsdf_settings: TsdfSettings


@dataclass(frozen=True)
class RunOptions:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda"
    pose_condition: bool = True
    save_depth: bool = False
    save_pointcloud: bool = True
    save_gs: bool = True
    depth_scale: int = 1000
    chunk_size: int = 24
    min_chunk_size: int = 4
    process_res_method: str = DEFAULT_PROCESS_RES_METHOD
    pose_scale_mode: str = DEFAULT_POSE_SCALE_MODE
    pose_scale_calib_frames: int = DEFAULT_POSE_SCALE_CALIB_FRAMES
    align_to_input_ext_scale: bool = True
    gs_max_frames: int = 128
    gs_views_interval: int | None = None
    output_root: str = DEFAULT_OUTPUT_ROOT
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(scene_name: str, message: str) -> None:
    print(f"[{_now()}] [{scene_name}] {message}", flush=True)


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_split(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _ensure_intrinsics(payload: dict) -> np.ndarray:
    if all(k in payload for k in ("fx", "fy", "cx", "cy")):
        fx, fy, cx, cy = payload["fx"], payload["fy"], payload["cx"], payload["cy"]
    elif all(k in payload for k in ("fl_x", "fl_y", "cx", "cy")):
        fx, fy, cx, cy = payload["fl_x"], payload["fl_y"], payload["cx"], payload["cy"]
    else:
        raise ValueError("Missing intrinsics in transforms json (need fx/fy/cx/cy or fl_x/fl_y/cx/cy)")

    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = float(fx)
    intrinsics[1, 1] = float(fy)
    intrinsics[0, 2] = float(cx)
    intrinsics[1, 2] = float(cy)
    return intrinsics


def _payload_image_size(payload: dict) -> tuple[int, int] | None:
    width = payload.get("width") or payload.get("w")
    height = payload.get("height") or payload.get("h")
    if width is None or height is None:
        return None
    return int(width), int(height)


def _resize_intrinsics_to_target_size(
    intrinsics: np.ndarray,
    payload: dict,
    target_size: tuple[int, int],
) -> np.ndarray:
    source_size = _payload_image_size(payload)
    if source_size is None:
        raise ValueError(
            "resize_intrinsics_to_target requires source size in transforms json (width/height or w/h)"
        )
    source_w, source_h = source_size
    target_w, target_h = target_size
    if source_w <= 0 or source_h <= 0:
        raise ValueError(f"Invalid source image size in transforms json: {(source_w, source_h)}")

    if source_w == target_w and source_h == target_h:
        return intrinsics.copy().astype(np.float32)

    scaled = intrinsics.copy().astype(np.float32)
    scaled[0, :] *= float(target_w) / float(source_w)
    scaled[1, :] *= float(target_h) / float(source_h)
    return scaled


def _matrix_to_c2w_w2c(
    matrix_4x4: np.ndarray,
    convention: Literal["c2w", "w2c"],
    apply_opengl_to_opencv: bool,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix_4x4, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform_matrix must be 4x4, got {matrix.shape}")

    if convention == "c2w":
        c2w = matrix
        w2c = np.linalg.inv(c2w).astype(np.float32)
        return c2w, w2c

    w2c = matrix
    if apply_opengl_to_opencv:
        flip = np.eye(4, dtype=np.float32)
        flip[1, 1] = -1.0
        flip[2, 2] = -1.0
        w2c = (flip @ w2c).astype(np.float32)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    return c2w, w2c


def _scannetpp_pose_from_nerfstudio_transform(matrix_4x4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ScanNet++ nerfstudio transform_matrix to pipeline c2w/w2c.

    Input matrix is nerfstudio c2w in OpenGL convention.
    This matches the user-validated conversion:
      c2w_cv = A @ c2w_ogl @ D
      w2c_cv = inv(c2w_cv)
    where A is the axis transform and D flips Y/Z camera axes.
    """
    c2w_ogl = np.asarray(matrix_4x4, dtype=np.float32)
    if c2w_ogl.shape != (4, 4):
        raise ValueError(f"ScanNet++ transform_matrix must be 4x4, got {c2w_ogl.shape}")

    c2w_cv = (_SCANNETPP_OPENGL_LEFT_TRANSFORM @ c2w_ogl @ _SCANNETPP_YZ_FLIP).astype(np.float32)
    w2c_cv = np.linalg.inv(c2w_cv).astype(np.float32)
    return c2w_cv, w2c_cv


def _resolve_path_candidates(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve image path from candidates: {checked}")


def _scannet_target_size(payload: dict) -> tuple[int, int] | None:
    resize = payload.get("resize")
    if isinstance(resize, (list, tuple)) and len(resize) == 2:
        return int(resize[0]), int(resize[1])

    width = payload.get("width") or payload.get("w")
    height = payload.get("height") or payload.get("h")
    if width is not None and height is not None:
        return int(width), int(height)
    return None


def _scannet_image_path(scene_root: Path, file_path: str) -> Path:
    base = scene_root / file_path
    candidates = [
        base,
        base.with_suffix(".jpg"),
        base.with_suffix(".png"),
        base.with_suffix(".jpeg"),
        Path(str(base) + ".jpg"),
        Path(str(base) + ".png"),
        Path(str(base) + ".jpeg"),
    ]
    return _resolve_path_candidates(candidates)


def _scannetpp_target_size(_: dict) -> tuple[int, int]:
    # User-locked convention for this project.
    return 876, 584


def _scannetpp_image_path(scene_root: Path, file_path: str) -> Path:
    base = scene_root / "dslr" / "undistorted_images" / file_path
    return _resolve_path_candidates([base])


def _default_target_size_from_payload(payload: dict) -> tuple[int, int] | None:
    resize = payload.get("resize")
    if isinstance(resize, (list, tuple)) and len(resize) == 2:
        return int(resize[0]), int(resize[1])

    width = payload.get("width") or payload.get("w")
    height = payload.get("height") or payload.get("h")
    if width is not None and height is not None:
        return int(width), int(height)
    return None


def _custom_image_path(scene_root: Path, file_path: str, image_root: Path | None) -> Path:
    file_obj = Path(file_path)
    if file_obj.is_absolute():
        return _resolve_path_candidates([file_obj])

    roots = [image_root] if image_root is not None else []
    roots.append(scene_root)

    candidates: list[Path] = []
    for root in roots:
        if root is None:
            continue
        base = root / file_path
        candidates.extend(
            [
                base,
                base.with_suffix(".jpg"),
                base.with_suffix(".png"),
                base.with_suffix(".jpeg"),
                Path(str(base) + ".jpg"),
                Path(str(base) + ".png"),
                Path(str(base) + ".jpeg"),
            ]
        )
    return _resolve_path_candidates(candidates)


def _dataset_configs(
    scannet_split: str,
    scannet_root: str,
    scannetpp_split: str,
    scannetpp_root: str,
) -> dict[str, DatasetConfig]:
    return {
        "scannet": DatasetConfig(
            name="scannet",
            split_path=scannet_split,
            root_path=scannet_root,
            transforms_relpath="transforms_train.json",
            target_size_resolver=_scannet_target_size,
            image_path_resolver=_scannet_image_path,
            transform_matrix_convention="c2w",
            apply_opengl_to_opencv=False,
            resize_intrinsics_to_target=False,
            pre_resize_images_to_target=True,
            tsdf_settings=TsdfSettings(
                voxel_length=0.02,
                sdf_trunc=0.08,
                max_depth=10.0,
                sample_points=1_000_000,
                voxel_downsample=0.02,
            ),
        ),
        "scannetpp": DatasetConfig(
            name="scannetpp",
            split_path=scannetpp_split,
            root_path=scannetpp_root,
            transforms_relpath="dslr/nerfstudio/transforms_undistorted.json",
            target_size_resolver=_scannetpp_target_size,
            image_path_resolver=_scannetpp_image_path,
            pose_matrix_resolver=_scannetpp_pose_from_nerfstudio_transform,
            transform_matrix_convention="c2w",
            apply_opengl_to_opencv=False,
            resize_intrinsics_to_target=True,
            pre_resize_images_to_target=True,
            tsdf_settings=TsdfSettings(
                voxel_length=0.02,
                sdf_trunc=0.15,
                max_depth=10.0,
                sample_points=1_000_000,
                voxel_downsample=0.02,
            ),
        ),
    }


def _as_homogeneous_batch(extrinsics: np.ndarray | None) -> np.ndarray | None:
    if extrinsics is None:
        return None
    ext = np.asarray(extrinsics, dtype=np.float32)
    if ext.ndim == 2:
        ext = ext[None]
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] == (3, 4):
        padded = np.zeros((ext.shape[0], 4, 4), dtype=ext.dtype)
        padded[:, :3, :4] = ext
        padded[:, 3, 3] = 1.0
        return padded
    raise ValueError(f"Unexpected extrinsics shape: {ext.shape}")


def _ensure_depth_batch(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected depth shape: {arr.shape}")
    return arr


def _ensure_image_batch(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    return arr


def _ensure_intrinsics_batch(intrinsics: np.ndarray) -> np.ndarray:
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected intrinsics shape: {arr.shape}")
    return arr


def _load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _resolve_chunk_model_inputs(scene: SceneSpec, frames: Sequence[FrameSpec]) -> list[str | np.ndarray]:
    if not scene.pre_resize_images_to_target:
        return [str(frame.image_path) for frame in frames]

    target_w, target_h = scene.target_size
    images: list[np.ndarray] = []
    for frame in frames:
        rgb = _load_image_rgb(frame.image_path)
        h, w = rgb.shape[:2]
        if w != target_w or h != target_h:
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        images.append(rgb)
    return images


def _estimate_alignment_scale(
    prediction_extrinsics: np.ndarray,
    input_extrinsics: np.ndarray,
) -> float:
    pred_h = _as_homogeneous_batch(prediction_extrinsics)
    inp_h = _as_homogeneous_batch(input_extrinsics)
    if pred_h is None or inp_h is None:
        raise ValueError("Cannot estimate alignment scale without extrinsics")
    if pred_h.shape[0] != inp_h.shape[0]:
        raise ValueError(f"Alignment scale inputs mismatch: {pred_h.shape[0]} vs {inp_h.shape[0]}")
    _, _, scale = align_poses_umeyama(
        pred_h,
        inp_h,
        ransac=pred_h.shape[0] >= 10,
        random_state=42,
    )
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0:
        raise RuntimeError(f"Invalid alignment scale: {scale_value}")
    return scale_value


def _select_scale_calib_indices(total_frames: int, max_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if max_frames <= 0 or total_frames <= max_frames:
        return list(range(total_frames))

    count = min(total_frames, max_frames)
    sampled = np.linspace(0, total_frames - 1, num=count, dtype=np.int64).tolist()
    indices: list[int] = []
    seen: set[int] = set()
    for idx in sampled:
        idx_int = int(idx)
        if idx_int in seen:
            continue
        seen.add(idx_int)
        indices.append(idx_int)

    if len(indices) < count:
        for idx in range(total_frames):
            if idx in seen:
                continue
            indices.append(idx)
            seen.add(idx)
            if len(indices) >= count:
                break
    return indices


def _calibrate_scene_pose_scale(
    model: DepthAnything3,
    scene: SceneSpec,
    options: RunOptions,
) -> dict:
    if not options.pose_condition or options.pose_scale_mode != "global":
        return {
            "enabled": False,
            "mode": options.pose_scale_mode,
            "scale": 1.0,
            "selected_frames": 0,
            "message": "Pose-scale global calibration disabled",
        }

    total_frames = len(scene.frames)
    min_calib_frames = 8
    frame_cap = max(min_calib_frames, min(options.pose_scale_calib_frames, total_frames))
    attempt_logs: list[dict] = []

    while frame_cap >= min_calib_frames:
        indices = _select_scale_calib_indices(total_frames, frame_cap)
        if len(indices) < min_calib_frames:
            break

        frames = [scene.frames[i] for i in indices]
        model_inputs = _resolve_chunk_model_inputs(scene, frames)
        extrinsics_np = np.stack([frame.w2c for frame in frames]).astype(np.float32)
        intrinsics_np = np.repeat(scene.intrinsics[None, ...], repeats=len(frames), axis=0).astype(np.float32)
        attempt_record = {
            "requested_frame_cap": int(frame_cap),
            "selected_frames": int(len(frames)),
        }
        start_t = time.time()

        try:
            prediction = model.inference(
                image=model_inputs,
                extrinsics=extrinsics_np,
                intrinsics=intrinsics_np,
                align_to_input_ext_scale=False,
                infer_gs=False,
                process_res=scene.process_res,
                process_res_method=options.process_res_method,
                export_dir=None,
            )
            scale = _estimate_alignment_scale(prediction.extrinsics, extrinsics_np)
            attempt_record["elapsed_sec"] = round(time.time() - start_t, 4)
            attempt_record["success"] = True
            attempt_record["estimated_scale"] = scale
            attempt_logs.append(attempt_record)
            return {
                "enabled": True,
                "mode": "global",
                "scale": scale,
                "selected_frames": int(len(frames)),
                "attempt_logs": attempt_logs,
            }
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                attempt_record["success"] = False
                attempt_record["error"] = str(exc)
                attempt_logs.append(attempt_record)
                raise

            cleanup_cuda_memory()
            next_cap = frame_cap // 2
            attempt_record["success"] = False
            attempt_record["error"] = str(exc)
            attempt_record["next_frame_cap"] = int(next_cap)
            attempt_logs.append(attempt_record)
            _log(
                scene.scene_name,
                f"Global scale calibration OOM at {frame_cap} frames, reducing to {next_cap}",
            )
            if next_cap < min_calib_frames:
                break
            frame_cap = next_cap
        finally:
            if "prediction" in locals():
                del prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(
        "Failed to estimate global pose scale. "
        f"scene={scene.scene_name}, requested pose_scale_calib_frames={options.pose_scale_calib_frames}."
    )


def _select_gs_indices(total_frames: int, max_frames: int) -> tuple[list[int], int]:
    if total_frames <= 0:
        return [], 1
    if max_frames <= 0:
        return [], 1
    if total_frames <= max_frames:
        return list(range(total_frames)), 1

    stride = int(math.ceil(total_frames / float(max_frames)))
    indices = list(range(0, total_frames, stride))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    if len(indices) > max_frames:
        indices = indices[: max_frames - 1] + [total_frames - 1]
    return indices, stride


def _save_depth_png(
    depth_m: np.ndarray,
    output_path: Path,
    target_size: tuple[int, int],
    depth_scale: int,
) -> None:
    target_w, target_h = target_size
    depth_out = depth_m
    if depth_out.shape[1] != target_w or depth_out.shape[0] != target_h:
        depth_out = cv2.resize(depth_out, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    depth_u16 = np.clip(depth_out * float(depth_scale), 0, 65535).astype(np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), depth_u16):
        raise RuntimeError(f"Failed to write depth png: {output_path}")


def _build_scene_frames(
    payload: dict,
    scene_root: Path,
    image_resolver: Callable[[str], Path],
    pose_matrix_resolver: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
    transform_matrix_convention: Literal["c2w", "w2c"],
    apply_opengl_to_opencv: bool,
) -> list[FrameSpec]:
    frames_json = payload.get("frames", [])
    if not frames_json:
        raise ValueError("No frames found under key 'frames'")

    stem_counts: dict[str, int] = {}
    frames: list[FrameSpec] = []
    for idx, frame in enumerate(frames_json):
        file_path = frame.get("file_path")
        if not file_path:
            raise ValueError(f"Frame {idx} has no file_path")
        transform_matrix = frame.get("transform_matrix")
        if transform_matrix is None:
            raise ValueError(f"Frame {idx} has no transform_matrix")
        matrix = np.asarray(transform_matrix, dtype=np.float32)
        if pose_matrix_resolver is not None:
            c2w, w2c = pose_matrix_resolver(matrix)
        else:
            c2w, w2c = _matrix_to_c2w_w2c(
                matrix_4x4=matrix,
                convention=transform_matrix_convention,
                apply_opengl_to_opencv=apply_opengl_to_opencv,
            )

        image_path = image_resolver(file_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Resolved image does not exist: {image_path}")

        stem = Path(file_path).stem or Path(file_path).name
        count = stem_counts.get(stem, 0)
        stem_counts[stem] = count + 1
        output_stem = stem if count == 0 else f"{stem}_{count:03d}"

        frames.append(
            FrameSpec(
                index=idx,
                file_path=file_path,
                image_path=image_path,
                output_stem=output_stem,
                c2w=c2w,
                w2c=w2c,
            )
        )
    return frames


def _infer_target_size_from_first_image(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to load image to infer target size: {image_path}")
    h, w = image.shape[:2]
    return int(w), int(h)


def _make_scene_spec(
    dataset_name: str,
    scene_name: str,
    scene_root: Path,
    transforms_json: Path,
    payload: dict,
    image_resolver: Callable[[str], Path],
    target_size: tuple[int, int] | None,
    pose_matrix_resolver: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None,
    transform_matrix_convention: Literal["c2w", "w2c"],
    apply_opengl_to_opencv: bool,
    resize_intrinsics_to_target: bool,
    pre_resize_images_to_target: bool,
    tsdf_settings: TsdfSettings,
) -> SceneSpec:
    frames = _build_scene_frames(
        payload=payload,
        scene_root=scene_root,
        image_resolver=image_resolver,
        pose_matrix_resolver=pose_matrix_resolver,
        transform_matrix_convention=transform_matrix_convention,
        apply_opengl_to_opencv=apply_opengl_to_opencv,
    )

    if target_size is None:
        target_size = _default_target_size_from_payload(payload)
    if target_size is None:
        target_size = _infer_target_size_from_first_image(frames[0].image_path)
    target_w, target_h = int(target_size[0]), int(target_size[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target size: {(target_w, target_h)}")

    intrinsics = _ensure_intrinsics(payload)
    if resize_intrinsics_to_target:
        intrinsics = _resize_intrinsics_to_target_size(
            intrinsics=intrinsics,
            payload=payload,
            target_size=(target_w, target_h),
        )

    process_res = max(target_w, target_h)

    return SceneSpec(
        dataset_name=dataset_name,
        scene_name=scene_name,
        scene_root=scene_root,
        transforms_json=transforms_json,
        frames=frames,
        intrinsics=intrinsics,
        target_size=(target_w, target_h),
        process_res=process_res,
        transform_matrix_convention=transform_matrix_convention,
        apply_opengl_to_opencv=apply_opengl_to_opencv,
        resize_intrinsics_to_target=resize_intrinsics_to_target,
        pre_resize_images_to_target=pre_resize_images_to_target,
        tsdf_settings=tsdf_settings,
    )


def build_single_scene_spec(
    scene_root: str,
    transforms_json: str | None = None,
    image_root: str | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
) -> SceneSpec:
    scene_root_path = Path(scene_root).expanduser().resolve()
    transforms_path = (
        Path(transforms_json).expanduser().resolve()
        if transforms_json
        else (scene_root_path / "transforms_train.json")
    )
    if not transforms_path.exists():
        raise FileNotFoundError(f"Transforms json not found: {transforms_path}")

    payload = _read_json(transforms_path)
    image_root_path = Path(image_root).expanduser().resolve() if image_root else None

    if (target_width is None) != (target_height is None):
        raise ValueError("Both --target-width and --target-height must be provided together")
    user_target = (int(target_width), int(target_height)) if target_width is not None else None

    return _make_scene_spec(
        dataset_name="custom",
        scene_name=scene_root_path.name,
        scene_root=scene_root_path,
        transforms_json=transforms_path,
        payload=payload,
        image_resolver=lambda fp: _custom_image_path(scene_root_path, fp, image_root_path),
        target_size=user_target,
        pose_matrix_resolver=None,
        transform_matrix_convention="c2w",
        apply_opengl_to_opencv=False,
        resize_intrinsics_to_target=False,
        pre_resize_images_to_target=False,
        tsdf_settings=TsdfSettings(
            voxel_length=0.02,
            sdf_trunc=0.08,
            max_depth=10.0,
            sample_points=1_000_000,
            voxel_downsample=0.02,
        ),
    )


def build_batch_scene_specs(
    datasets: Sequence[str],
    scannet_split: str = DEFAULT_SCANNET_SPLIT,
    scannet_root: str = DEFAULT_SCANNET_ROOT,
    scannetpp_split: str = DEFAULT_SCANNETPP_SPLIT,
    scannetpp_root: str = DEFAULT_SCANNETPP_ROOT,
    scene_filter: Sequence[str] | None = None,
) -> list[SceneSpec]:
    configs = _dataset_configs(scannet_split, scannet_root, scannetpp_split, scannetpp_root)
    unknown = [name for name in datasets if name not in configs]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; expected any of {sorted(configs.keys())}")

    scene_filter_set = set(scene_filter or [])
    specs: list[SceneSpec] = []
    for dataset_name in datasets:
        cfg = configs[dataset_name]
        split_path = Path(cfg.split_path).expanduser().resolve()
        root_path = Path(cfg.root_path).expanduser().resolve()
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        scene_names = _read_split(split_path)
        if scene_filter_set:
            scene_names = [name for name in scene_names if name in scene_filter_set]

        for scene_name in scene_names:
            scene_root = root_path / scene_name
            transforms_path = scene_root / cfg.transforms_relpath
            if not transforms_path.exists():
                raise FileNotFoundError(f"Missing transforms file: {transforms_path}")

            payload = _read_json(transforms_path)
            target_size = cfg.target_size_resolver(payload)

            specs.append(
                _make_scene_spec(
                    dataset_name=dataset_name,
                    scene_name=scene_name,
                    scene_root=scene_root,
                    transforms_json=transforms_path,
                    payload=payload,
                    image_resolver=lambda fp, sr=scene_root, resolver=cfg.image_path_resolver: resolver(sr, fp),
                    target_size=target_size,
                    pose_matrix_resolver=cfg.pose_matrix_resolver,
                    transform_matrix_convention=cfg.transform_matrix_convention,
                    apply_opengl_to_opencv=cfg.apply_opengl_to_opencv,
                    resize_intrinsics_to_target=cfg.resize_intrinsics_to_target,
                    pre_resize_images_to_target=cfg.pre_resize_images_to_target,
                    tsdf_settings=cfg.tsdf_settings,
                )
            )
    return specs


class _TsdfAccumulator:
    def __init__(self, settings: TsdfSettings):
        try:
            import open3d as o3d
        except Exception as exc:
            raise RuntimeError("Open3D is required for --save-pointcloud") from exc

        self.o3d = o3d
        self.settings = settings
        self.integrated_frames = 0
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=settings.voxel_length,
            sdf_trunc=settings.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate(self, depth: np.ndarray, image_rgb: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> None:
        h, w = depth.shape
        image_contig = np.ascontiguousarray(image_rgb.astype(np.uint8))
        depth_contig = np.ascontiguousarray(depth.astype(np.float32))
        color_o3d = self.o3d.geometry.Image(image_contig)
        depth_o3d = self.o3d.geometry.Image(depth_contig)
        rgbd = self.o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.settings.max_depth,
            convert_rgb_to_intensity=False,
        )

        camera = self.o3d.camera.PinholeCameraIntrinsic(
            int(w),
            int(h),
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        )
        self.volume.integrate(rgbd, camera, extrinsics.astype(np.float64))
        self.integrated_frames += 1

    def export_pointcloud(self, output_path: Path) -> int:
        mesh = self.volume.extract_triangle_mesh()
        if len(mesh.vertices) == 0:
            pcd = self.o3d.geometry.PointCloud()
        else:
            try:
                pcd = mesh.sample_points_uniformly(number_of_points=self.settings.sample_points)
            except Exception:
                pcd = self.o3d.geometry.PointCloud()
                pcd.points = self.o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

        if self.settings.voxel_downsample and len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(self.settings.voxel_downsample)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.o3d.io.write_point_cloud(str(output_path), pcd):
            raise RuntimeError(f"Failed to write fused point cloud: {output_path}")
        return int(np.asarray(pcd.points).shape[0])


def _run_scene_depth_pass(
    model: DepthAnything3,
    scene: SceneSpec,
    options: RunOptions,
    results_output_dir: Path,
    depth_output_dir: Path | None,
    global_pose_scale: float | None = None,
) -> dict:
    total_frames = len(scene.frames)
    if total_frames == 0:
        raise RuntimeError("Scene has no frames")

    _log(scene.scene_name, f"Depth pass: {total_frames} frames, initial chunk={options.chunk_size}")

    if options.pose_scale_mode not in SUPPORTED_POSE_SCALE_MODES:
        raise ValueError(
            f"Unsupported pose_scale_mode={options.pose_scale_mode}. Expected one of {SUPPORTED_POSE_SCALE_MODES}"
        )
    if options.pose_condition and options.pose_scale_mode == "global" and global_pose_scale is None:
        raise RuntimeError("Missing global_pose_scale for pose_scale_mode='global'")

    current_chunk_size = max(options.chunk_size, options.min_chunk_size)
    min_chunk_size = max(1, options.min_chunk_size)
    frame_cursor = 0
    chunk_logs: list[dict] = []
    failed_chunks: list[dict] = []
    depth_saved = 0
    effective_size: tuple[int, int] | None = None

    tsdf = _TsdfAccumulator(scene.tsdf_settings) if options.save_pointcloud else None

    while frame_cursor < total_frames:
        attempt_chunk = min(current_chunk_size, total_frames - frame_cursor)
        retries = 0
        chunk_done = False

        while not chunk_done:
            end_idx = frame_cursor + attempt_chunk
            chunk_frames = scene.frames[frame_cursor:end_idx]
            image_inputs = _resolve_chunk_model_inputs(scene, chunk_frames)

            extrinsics_np = None
            intrinsics_np = None
            if options.pose_condition:
                extrinsics_np = np.stack([frame.w2c for frame in chunk_frames]).astype(np.float32)
                intrinsics_np = np.repeat(
                    scene.intrinsics[None, ...], repeats=len(chunk_frames), axis=0
                ).astype(np.float32)

            _log(
                scene.scene_name,
                f"Chunk [{frame_cursor}:{end_idx}) frames={len(chunk_frames)} chunk_size={attempt_chunk} retry={retries}",
            )
            start_t = time.time()
            try:
                prediction = model.inference(
                    image=image_inputs,
                    extrinsics=extrinsics_np,
                    intrinsics=intrinsics_np,
                    align_to_input_ext_scale=False if options.pose_condition else options.align_to_input_ext_scale,
                    infer_gs=False,
                    process_res=scene.process_res,
                    process_res_method=options.process_res_method,
                    export_dir=None,
                )
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise

                retries += 1
                next_chunk = max(min_chunk_size, attempt_chunk // 2)
                _log(
                    scene.scene_name,
                    f"OOM on chunk [{frame_cursor}:{end_idx}), reducing chunk_size {attempt_chunk} -> {next_chunk}",
                )
                cleanup_cuda_memory()

                if next_chunk >= attempt_chunk:
                    failed_chunks.append(
                        {
                            "start_index": frame_cursor,
                            "end_index": end_idx,
                            "chunk_size": attempt_chunk,
                            "error": str(exc),
                        }
                    )
                    raise RuntimeError(
                        "CUDA OOM persisted at minimum chunk size. "
                        f"scene={scene.scene_name}, chunk=[{frame_cursor}:{end_idx}), min_chunk_size={min_chunk_size}. "
                        "Try reducing target size (process resolution), disabling pose conditioning, or lowering --min-chunk-size."
                    ) from exc
                attempt_chunk = next_chunk
                continue

            elapsed = time.time() - start_t

            depth_batch = _ensure_depth_batch(prediction.depth)
            if depth_batch.shape[0] != len(chunk_frames):
                raise RuntimeError(
                    f"Depth batch size mismatch: got {depth_batch.shape[0]}, expected {len(chunk_frames)}"
                )

            chunk_pose_scale = None
            applied_depth_scale = 1.0
            if options.pose_condition:
                if extrinsics_np is None:
                    raise RuntimeError("Internal error: missing input extrinsics in pose-conditioned path")
                if prediction.extrinsics is None:
                    raise RuntimeError("Prediction extrinsics missing; cannot compute pose alignment scale")
                chunk_pose_scale = _estimate_alignment_scale(prediction.extrinsics, extrinsics_np)
                if options.pose_scale_mode == "per_chunk":
                    applied_depth_scale = chunk_pose_scale
                elif options.pose_scale_mode == "global":
                    applied_depth_scale = float(global_pose_scale)
                elif options.pose_scale_mode == "none":
                    applied_depth_scale = 1.0
                else:
                    raise RuntimeError(f"Unsupported pose_scale_mode: {options.pose_scale_mode}")

                depth_batch = (depth_batch / max(applied_depth_scale, 1e-8)).astype(np.float32, copy=False)

            effective_size = (int(depth_batch.shape[2]), int(depth_batch.shape[1]))

            if options.save_depth:
                if depth_output_dir is None:
                    raise RuntimeError("Internal error: depth_output_dir is None while save_depth is enabled")
                for local_idx, frame in enumerate(chunk_frames):
                    depth_png_path = depth_output_dir / f"{frame.output_stem}.png"
                    _save_depth_png(
                        depth_m=depth_batch[local_idx],
                        output_path=depth_png_path,
                        target_size=scene.target_size,
                        depth_scale=options.depth_scale,
                    )
                    depth_saved += 1

            if tsdf is not None:
                processed_images = _ensure_image_batch(prediction.processed_images)
                pred_intrinsics = _ensure_intrinsics_batch(prediction.intrinsics)
                if options.pose_condition:
                    pred_extrinsics = _as_homogeneous_batch(extrinsics_np)
                else:
                    pred_extrinsics = _as_homogeneous_batch(prediction.extrinsics)
                if pred_extrinsics is None:
                    raise RuntimeError("Prediction extrinsics missing; cannot run TSDF fusion")
                if pred_intrinsics.shape[0] != len(chunk_frames):
                    raise RuntimeError(
                        f"Pred intrinsics batch mismatch: {pred_intrinsics.shape[0]} vs {len(chunk_frames)}"
                    )
                if processed_images.shape[0] != len(chunk_frames):
                    raise RuntimeError(
                        f"Processed image batch mismatch: {processed_images.shape[0]} vs {len(chunk_frames)}"
                    )
                if pred_extrinsics.shape[0] != len(chunk_frames):
                    raise RuntimeError(
                        f"Pred extrinsics batch mismatch: {pred_extrinsics.shape[0]} vs {len(chunk_frames)}"
                    )

                for local_idx in range(len(chunk_frames)):
                    tsdf.integrate(
                        depth=depth_batch[local_idx],
                        image_rgb=processed_images[local_idx],
                        intrinsics=pred_intrinsics[local_idx],
                        extrinsics=pred_extrinsics[local_idx],
                    )

            chunk_logs.append(
                {
                    "start_index": frame_cursor,
                    "end_index": end_idx,
                    "frames": len(chunk_frames),
                    "chunk_size": attempt_chunk,
                    "oom_retries": retries,
                    "elapsed_sec": round(elapsed, 4),
                    "inference_size": {
                        "width": int(depth_batch.shape[2]),
                        "height": int(depth_batch.shape[1]),
                    },
                    "chunk_pose_scale": float(chunk_pose_scale) if chunk_pose_scale is not None else None,
                    "applied_depth_scale": float(applied_depth_scale),
                }
            )

            frame_cursor = end_idx
            current_chunk_size = attempt_chunk
            chunk_done = True
            del prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pointcloud_info = {"enabled": bool(options.save_pointcloud), "success": False}
    if tsdf is not None:
        pointcloud_path = results_output_dir / "global_pointcloud.ply"
        num_points = tsdf.export_pointcloud(pointcloud_path)
        pointcloud_info = {
            "enabled": True,
            "success": True,
            "path": str(pointcloud_path),
            "points": num_points,
            "integrated_frames": tsdf.integrated_frames,
        }

    return {
        "success": True,
        "frames_total": total_frames,
        "frames_processed": frame_cursor,
        "initial_chunk_size": int(max(options.chunk_size, options.min_chunk_size)),
        "final_chunk_size": int(current_chunk_size),
        "effective_inference_size": (
            {"width": int(effective_size[0]), "height": int(effective_size[1])}
            if effective_size
            else None
        ),
        "pose_scale": {
            "mode": options.pose_scale_mode if options.pose_condition else "none",
            "global_pose_scale": (
                float(global_pose_scale) if (options.pose_condition and global_pose_scale is not None) else None
            ),
        },
        "depth_png": {
            "enabled": bool(options.save_depth),
            "saved_frames": int(depth_saved),
            "depth_scale": int(options.depth_scale),
            "output_dir": str(depth_output_dir) if depth_output_dir is not None else None,
            "target_size": {"width": int(scene.target_size[0]), "height": int(scene.target_size[1])},
        },
        "pointcloud": pointcloud_info,
        "chunk_logs": chunk_logs,
        "failed_chunks": failed_chunks,
    }


def _run_scene_gs_pass(
    model: DepthAnything3,
    scene: SceneSpec,
    options: RunOptions,
    results_output_dir: Path,
) -> dict:
    if not options.save_gs:
        return {"enabled": False, "success": False, "message": "3DGS export disabled"}

    total_frames = len(scene.frames)
    if total_frames == 0:
        return {"enabled": True, "success": False, "message": "No frames available for 3DGS"}

    min_gs_frames = 4
    attempt_cap = max(1, min(options.gs_max_frames, total_frames))
    attempt_logs: list[dict] = []

    while attempt_cap >= min_gs_frames:
        indices, stride = _select_gs_indices(total_frames, attempt_cap)
        if not indices:
            break

        frames = [scene.frames[i] for i in indices]
        model_inputs = _resolve_chunk_model_inputs(scene, frames)

        extrinsics_np = None
        intrinsics_np = None
        if options.pose_condition:
            extrinsics_np = np.stack([frame.w2c for frame in frames]).astype(np.float32)
            intrinsics_np = np.repeat(scene.intrinsics[None, ...], repeats=len(frames), axis=0).astype(
                np.float32
            )

        _log(
            scene.scene_name,
            f"3DGS pass: trying {len(frames)} frames (cap={attempt_cap}, stride={stride})",
        )

        attempt_record = {
            "requested_frame_cap": int(attempt_cap),
            "selected_frames": int(len(frames)),
            "stride": int(stride),
        }
        start_t = time.time()

        try:
            prediction = model.inference(
                image=model_inputs,
                extrinsics=extrinsics_np,
                intrinsics=intrinsics_np,
                align_to_input_ext_scale=options.align_to_input_ext_scale,
                infer_gs=True,
                process_res=scene.process_res,
                process_res_method=options.process_res_method,
                export_dir=None,
            )
            temp_export_dir = results_output_dir / "_tmp_gs_export"
            if temp_export_dir.exists():
                shutil.rmtree(temp_export_dir)
            temp_export_dir.mkdir(parents=True, exist_ok=True)

            export_to_gs_ply(
                prediction=prediction,
                export_dir=str(temp_export_dir),
                gs_views_interval=options.gs_views_interval,
            )

            raw_ply_path = temp_export_dir / "gs_ply" / "0000.ply"
            if not raw_ply_path.exists():
                raise FileNotFoundError(f"Expected gs ply not found at {raw_ply_path}")

            final_ply_path = results_output_dir / "global_3dgs.ply"
            if final_ply_path.exists():
                final_ply_path.unlink()
            shutil.move(str(raw_ply_path), str(final_ply_path))
            shutil.rmtree(temp_export_dir, ignore_errors=True)

            elapsed = time.time() - start_t
            attempt_record["elapsed_sec"] = round(elapsed, 4)
            attempt_record["success"] = True
            attempt_logs.append(attempt_record)

            return {
                "enabled": True,
                "success": True,
                "path": str(final_ply_path),
                "selected_frames": int(len(frames)),
                "stride": int(stride),
                "gs_views_interval": options.gs_views_interval,
                "attempt_logs": attempt_logs,
            }
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                attempt_record["success"] = False
                attempt_record["error"] = str(exc)
                attempt_logs.append(attempt_record)
                return {
                    "enabled": True,
                    "success": False,
                    "error": str(exc),
                    "attempt_logs": attempt_logs,
                }

            cleanup_cuda_memory()
            next_cap = attempt_cap // 2
            attempt_record["success"] = False
            attempt_record["error"] = str(exc)
            attempt_record["next_frame_cap"] = int(next_cap)
            attempt_logs.append(attempt_record)

            _log(
                scene.scene_name,
                f"3DGS OOM at {attempt_cap} frames, reducing to {next_cap}",
            )
            if next_cap < min_gs_frames:
                return {
                    "enabled": True,
                    "success": False,
                    "error": (
                        "3DGS OOM even after reducing selected frames. "
                        f"Last attempted frame cap={attempt_cap}. "
                        "Try lowering --gs-max-frames or disable --save-gs."
                    ),
                    "attempt_logs": attempt_logs,
                }
            attempt_cap = next_cap
        finally:
            if "prediction" in locals():
                del prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "enabled": True,
        "success": False,
        "error": "3DGS did not run; no valid frame subset remained.",
        "attempt_logs": attempt_logs,
    }


def load_model(model_id: str, device: str) -> DepthAnything3:
    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()
    return model


def run_scene_inference(model: DepthAnything3, scene: SceneSpec, options: RunOptions) -> dict:
    results_output_dir = (
        Path(options.output_root).expanduser().resolve() / scene.dataset_name / scene.scene_name
    )
    results_output_dir.mkdir(parents=True, exist_ok=True)
    depth_output_dir = scene.scene_root / options.output_subdir if options.save_depth else None
    manifest_path = results_output_dir / "run_manifest.json"

    manifest: dict = {
        "status": "running",
        "scene": {
            "dataset": scene.dataset_name,
            "scene_name": scene.scene_name,
            "scene_root": str(scene.scene_root),
            "transforms_json": str(scene.transforms_json),
            "num_frames": len(scene.frames),
        },
        "options": {
            "model_id": options.model_id,
            "device": options.device,
            "pose_condition": options.pose_condition,
            "save_depth": options.save_depth,
            "save_pointcloud": options.save_pointcloud,
            "save_gs": options.save_gs,
            "depth_scale": options.depth_scale,
            "chunk_size": options.chunk_size,
            "min_chunk_size": options.min_chunk_size,
            "process_res_method": options.process_res_method,
            "pose_scale_mode": options.pose_scale_mode,
            "pose_scale_calib_frames": options.pose_scale_calib_frames,
            "align_to_input_ext_scale": options.align_to_input_ext_scale,
            "gs_max_frames": options.gs_max_frames,
            "gs_views_interval": options.gs_views_interval,
            "output_root": str(Path(options.output_root).expanduser().resolve()),
            "output_subdir": options.output_subdir,
        },
        "scene_config": {
            "target_size": {"width": scene.target_size[0], "height": scene.target_size[1]},
            "process_res": scene.process_res,
            "transform_matrix_convention": scene.transform_matrix_convention,
            "apply_opengl_to_opencv": scene.apply_opengl_to_opencv,
            "resize_intrinsics_to_target": scene.resize_intrinsics_to_target,
            "pre_resize_images_to_target": scene.pre_resize_images_to_target,
            "tsdf": {
                "voxel_length": scene.tsdf_settings.voxel_length,
                "sdf_trunc": scene.tsdf_settings.sdf_trunc,
                "max_depth": scene.tsdf_settings.max_depth,
                "sample_points": scene.tsdf_settings.sample_points,
                "voxel_downsample": scene.tsdf_settings.voxel_downsample,
            },
        },
        "output_layout": {
            "results_output_dir": str(results_output_dir),
            "depth_output_dir": str(depth_output_dir) if depth_output_dir is not None else None,
        },
        "start_time": time.time(),
    }

    _log(scene.scene_name, f"Starting scene inference ({len(scene.frames)} frames)")
    _log(scene.scene_name, f"Results output dir: {results_output_dir}")
    if depth_output_dir is not None:
        _log(scene.scene_name, f"Depth output dir: {depth_output_dir}")

    try:
        pose_scale_calibration = _calibrate_scene_pose_scale(
            model=model,
            scene=scene,
            options=options,
        )
        manifest["pose_scale_calibration"] = pose_scale_calibration
        if pose_scale_calibration.get("enabled", False):
            _log(
                scene.scene_name,
                "Pose-scale calibration: "
                f"mode={pose_scale_calibration.get('mode')} "
                f"scale={pose_scale_calibration.get('scale'):.6f} "
                f"frames={pose_scale_calibration.get('selected_frames')}",
            )

        depth_result = _run_scene_depth_pass(
            model=model,
            scene=scene,
            options=options,
            results_output_dir=results_output_dir,
            depth_output_dir=depth_output_dir,
            global_pose_scale=float(pose_scale_calibration.get("scale", 1.0)),
        )
        manifest["depth_pass"] = depth_result

        if options.save_gs:
            gs_result = _run_scene_gs_pass(
                model=model,
                scene=scene,
                options=options,
                results_output_dir=results_output_dir,
            )
            manifest["gs_pass"] = gs_result
            if not gs_result.get("success", False):
                raise RuntimeError(f"3DGS pass failed: {gs_result.get('error', 'unknown error')}")
        else:
            manifest["gs_pass"] = {"enabled": False, "success": False, "message": "3DGS disabled"}

        manifest["status"] = "success"
        _log(scene.scene_name, "Scene inference completed successfully")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        _log(scene.scene_name, f"Scene inference failed: {exc}")
    finally:
        manifest["end_time"] = time.time()
        manifest["elapsed_sec"] = round(manifest["end_time"] - manifest["start_time"], 4)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest
