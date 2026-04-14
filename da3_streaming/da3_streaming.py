# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import argparse
import gc
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.config_utils import load_config
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)
from safetensors.torch import load_file

from depth_anything_3.api import DepthAnything3

matplotlib.use("Agg")


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


def remove_duplicates(data_list):
    """
    data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])

        if key not in seen.keys():
            seen[key] = True
            result.append(item)

    return result


def compute_chunk_indices(num_frames: int, chunk_size: int, overlap: int) -> tuple[list[tuple[int, int]], int]:
    if num_frames <= chunk_size:
        return [(0, num_frames)], 1
    step = chunk_size - overlap
    num_chunks = (num_frames - overlap + step - 1) // step
    chunk_indices = []
    for i in range(num_chunks):
        start_idx = i * step
        end_idx = min(start_idx + chunk_size, num_frames)
        chunk_indices.append((start_idx, end_idx))
    return chunk_indices, num_chunks


def _safe_float(value) -> float | None:
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def _float_list(values: np.ndarray | Sequence[float]) -> list[float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [_safe_float(v) for v in arr]


def _summarize_depth_conf(
    depth: np.ndarray,
    conf: np.ndarray,
    sky: np.ndarray | None,
    depth_threshold_m: float,
) -> dict:
    depth_arr = np.asarray(depth, dtype=np.float32)
    conf_arr = np.asarray(conf, dtype=np.float32)
    sky_mask = None if sky is None else np.asarray(sky).astype(bool)
    total_pixels = int(depth_arr.size)
    finite_depth = np.isfinite(depth_arr)
    finite_vals = depth_arr[finite_depth]
    summary = {
        "pixel_count": total_pixels,
        "finite_depth_fraction": _safe_float(finite_depth.mean()) if total_pixels > 0 else None,
        "depth_threshold_m": _safe_float(depth_threshold_m),
        "depth_median_m": None,
        "depth_p90_m": None,
        "depth_p99_m": None,
        "depth_max_m": None,
        "frac_depth_above_threshold": None,
        "sky_fraction": _safe_float(sky_mask.mean()) if sky_mask is not None and total_pixels > 0 else None,
        "non_sky_depth_median_m": None,
        "non_sky_depth_p99_m": None,
        "conf_mean": _safe_float(conf_arr.mean()) if conf_arr.size > 0 else None,
        "conf_median": _safe_float(np.median(conf_arr)) if conf_arr.size > 0 else None,
        "frac_conf_positive": _safe_float((conf_arr > 1e-5).mean()) if conf_arr.size > 0 else None,
    }
    if finite_vals.size > 0:
        q = np.quantile(finite_vals, [0.5, 0.9, 0.99, 1.0])
        summary["depth_median_m"] = _safe_float(q[0])
        summary["depth_p90_m"] = _safe_float(q[1])
        summary["depth_p99_m"] = _safe_float(q[2])
        summary["depth_max_m"] = _safe_float(q[3])
        summary["frac_depth_above_threshold"] = _safe_float((finite_vals > depth_threshold_m).mean())

    if sky_mask is not None:
        non_sky_mask = finite_depth & (~sky_mask)
    else:
        non_sky_mask = finite_depth
    non_sky_vals = depth_arr[non_sky_mask]
    if non_sky_vals.size > 0:
        q_ns = np.quantile(non_sky_vals, [0.5, 0.99])
        summary["non_sky_depth_median_m"] = _safe_float(q_ns[0])
        summary["non_sky_depth_p99_m"] = _safe_float(q_ns[1])

    return summary


def _build_point_filter_stats(
    depth: np.ndarray,
    conf: np.ndarray,
    sky: np.ndarray | None,
    invalid_mask: np.ndarray | None,
    depth_threshold_m: float,
    exclude_depth_above_threshold: bool,
) -> dict:
    depth_arr = np.asarray(depth, dtype=np.float32)
    conf_arr = np.asarray(conf, dtype=np.float32)
    sky_mask = None if sky is None else np.asarray(sky).astype(bool)
    depth_above = depth_arr > depth_threshold_m
    conf_masked = np.asarray(conf_arr, dtype=np.float32).copy()
    if invalid_mask is not None:
        conf_masked[np.asarray(invalid_mask).astype(bool)] = -1e6
    return {
        "exclude_depth_above_threshold": bool(exclude_depth_above_threshold),
        "depth_threshold_m": _safe_float(depth_threshold_m),
        "sky_mask_available": sky_mask is not None,
        "raw_frac_depth_above_threshold": _safe_float(depth_above.mean()) if depth_above.size > 0 else None,
        "raw_sky_fraction": _safe_float(sky_mask.mean()) if sky_mask is not None and sky_mask.size > 0 else None,
        "invalid_fraction_after_raw_masking": (
            _safe_float(np.asarray(invalid_mask).astype(bool).mean()) if invalid_mask is not None else 0.0
        ),
        "frac_conf_positive_before_mask": _safe_float((conf_arr > 1e-5).mean()) if conf_arr.size > 0 else None,
        "frac_conf_positive_after_mask": _safe_float((conf_masked > 1e-5).mean()) if conf_masked.size > 0 else None,
    }


def _point_stats_from_points(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    summary = {
        "point_count": int(pts.shape[0]),
        "bbox_min": None,
        "bbox_max": None,
        "bbox_extent": None,
        "norm_quantiles": None,
    }
    if pts.shape[0] == 0:
        return summary
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    norms = np.linalg.norm(pts, axis=1)
    summary["bbox_min"] = _float_list(bbox_min)
    summary["bbox_max"] = _float_list(bbox_max)
    summary["bbox_extent"] = _float_list(bbox_max - bbox_min)
    summary["norm_quantiles"] = _float_list(np.quantile(norms, [0.5, 0.9, 0.99, 0.999, 1.0]))
    return summary


def _point_cloud_stats_from_ply(path: Path) -> dict:
    out = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    try:
        geom = trimesh.load(path, process=False)
        if isinstance(geom, trimesh.Scene):
            point_sets = []
            for item in geom.geometry.values():
                vertices = np.asarray(getattr(item, "vertices", np.zeros((0, 3))), dtype=np.float64)
                if vertices.size > 0:
                    point_sets.append(vertices.reshape(-1, 3))
            points = np.concatenate(point_sets, axis=0) if point_sets else np.zeros((0, 3), dtype=np.float64)
        else:
            points = np.asarray(getattr(geom, "vertices", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        out.update(_point_stats_from_points(points))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _camera_pose_stats_from_txt(path: Path) -> dict:
    out = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    try:
        poses = np.loadtxt(path, dtype=np.float64)
        if poses.ndim == 1:
            poses = poses[None, :]
        poses = poses.reshape(-1, 4, 4)
        centers = poses[:, :3, 3]
        center_norms = np.linalg.norm(centers, axis=1)
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros((0,), dtype=np.float64)
        bbox_min = centers.min(axis=0) if len(centers) > 0 else np.zeros((3,), dtype=np.float64)
        bbox_max = centers.max(axis=0) if len(centers) > 0 else np.zeros((3,), dtype=np.float64)
        out.update(
            {
                "num_poses": int(len(poses)),
                "bbox_min": _float_list(bbox_min) if len(centers) > 0 else None,
                "bbox_max": _float_list(bbox_max) if len(centers) > 0 else None,
                "bbox_extent": _float_list(bbox_max - bbox_min) if len(centers) > 0 else None,
                "center_norm_quantiles": (
                    _float_list(np.quantile(center_norms, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]))
                    if len(center_norms) > 0
                    else None
                ),
                "step_quantiles": (
                    _float_list(np.quantile(steps, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]))
                    if len(steps) > 0
                    else None
                ),
                "first_step_gt_100_index": int(np.where(steps > 100.0)[0][0] + 1) if np.any(steps > 100.0) else None,
                "first_step_gt_1000_index": int(np.where(steps > 1000.0)[0][0] + 1) if np.any(steps > 1000.0) else None,
            }
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _saved_range(chunk_start: int, save_indices: Sequence[int]) -> list[int] | None:
    if len(save_indices) == 0:
        return None
    return [int(chunk_start + save_indices[0]), int(chunk_start + save_indices[-1] + 1)]


def _saved_frame_entries(
    depth: np.ndarray,
    conf: np.ndarray,
    sky: np.ndarray | None,
    frame_keys: Sequence[str],
    chunk_idx: int,
    chunk_start: int,
    save_indices: Sequence[int],
    depth_threshold_m: float,
    depth_scale_applied: float,
) -> list[dict]:
    entries: list[dict] = []
    sky_arr = None if sky is None else np.asarray(sky).astype(bool)
    for local_idx in save_indices:
        global_idx = chunk_start + int(local_idx)
        frame_stats = _summarize_depth_conf(
            np.asarray(depth[local_idx], dtype=np.float32),
            np.asarray(conf[local_idx], dtype=np.float32),
            None if sky_arr is None else sky_arr[local_idx],
            depth_threshold_m,
        )
        frame_stats.update(
            {
                "chunk_idx": int(chunk_idx),
                "global_index": int(global_idx),
                "local_index": int(local_idx),
                "frame_key": frame_keys[global_idx],
                "depth_scale_applied": _safe_float(depth_scale_applied),
            }
        )
        entries.append(frame_stats)
    return entries


def _sim3_entries(transforms: Sequence[tuple], target_chunk_zero: bool) -> list[dict]:
    entries: list[dict] = []
    for idx, (s, R, t) in enumerate(transforms):
        t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
        entries.append(
            {
                "chunk_idx": int(idx + 1),
                "target_chunk_idx": 0 if target_chunk_zero else int(idx),
                "scale": _safe_float(s),
                "translation": _float_list(t_arr[:3]),
                "translation_norm": _safe_float(np.linalg.norm(t_arr[:3])) if t_arr.size >= 3 else None,
                "rotation_det": _safe_float(np.linalg.det(np.asarray(R, dtype=np.float64))),
            }
        )
    return entries


def _build_warning_messages(diagnostics: dict) -> list[str]:
    warnings: list[str] = []
    threshold = float(diagnostics.get("runtime_options", {}).get("depth_threshold_m", 100.0))

    saved_frames = diagnostics.get("saved_frame_diagnostics") or []
    for frame in saved_frames:
        ns_med = frame.get("non_sky_depth_median_m")
        if ns_med is not None and ns_med > (10.0 * threshold):
            warnings.append(
                "[WARN] Saved non-sky median depth exceeded "
                f"{10.0 * threshold:.3f} m at {frame['frame_key']} "
                f"(chunk {frame['chunk_idx']}, median={ns_med:.3f} m)."
            )
            break

    for entry in diagnostics.get("sim3", {}).get("cumulative", []):
        scale = entry.get("scale")
        if scale is not None and (scale < 0.1 or scale > 10.0):
            warnings.append(
                "[WARN] Cumulative Sim(3) scale left [0.1, 10] "
                f"at chunk {entry['chunk_idx']} (scale={scale:.6g})."
            )
            break

    camera_stats = diagnostics.get("artifacts", {}).get("camera_poses_txt", {})
    step_q = camera_stats.get("step_quantiles")
    if step_q is not None and len(step_q) >= 3 and step_q[2] is not None and step_q[2] > 100.0:
        warnings.append(
            "[WARN] Camera-pose step median exceeded 100 after alignment "
            f"(median={step_q[2]:.3f})."
        )
    first_large_step = camera_stats.get("first_step_gt_1000_index")
    if first_large_step is not None:
        step_vals = step_q if step_q is not None else []
        max_step = step_vals[-1] if step_vals else None
        warnings.append(
            "[WARN] Camera-pose step exceeded 1000 starting near frame "
            f"{first_large_step + 1} (max_step={max_step})."
        )

    points_stats = diagnostics.get("artifacts", {}).get("points_da3", {})
    bbox_extent = points_stats.get("bbox_extent")
    if bbox_extent is not None:
        max_extent = max(v for v in bbox_extent if v is not None)
        if max_extent > 1000.0:
            warnings.append(
                "[WARN] points_da3.ply bbox extent exceeded 1000 "
                f"(max_extent={max_extent:.3f})."
            )

    if diagnostics.get("runtime_options", {}).get("exclude_depth_above_100_for_points", False):
        mismatch_warned = False
        for chunk in diagnostics.get("chunks", []):
            post_alignment = chunk.get("post_alignment")
            raw_filter = chunk.get("point_filter_raw_stats")
            if not post_alignment or raw_filter is None:
                continue
            raw_frac = raw_filter.get("raw_frac_depth_above_threshold")
            scaled_frac = post_alignment.get("scaled_depth_stats", {}).get("frac_depth_above_threshold")
            if raw_frac is None or scaled_frac is None:
                continue
            if scaled_frac > raw_frac + 1e-6:
                warnings.append(
                    "[WARN] Chunk "
                    f"{chunk['chunk_idx']} uses pre-Sim(3) depth masking for point export "
                    f"(raw_frac>{threshold:.3f}m={raw_frac:.3f}, scaled_frac={scaled_frac:.3f})."
                )
                mismatch_warned = True
                break
        if not mismatch_warned:
            for chunk in diagnostics.get("chunks", []):
                if chunk.get("chunk_idx", 0) <= 0:
                    continue
                saved_stats = chunk.get("saved_output_stats") or {}
                scaled_frac = saved_stats.get("frac_depth_above_threshold")
                if scaled_frac is not None and scaled_frac > 0.0:
                    warnings.append(
                        "[WARN] Existing artifacts suggest aligned chunks can retain "
                        "post-Sim(3) large-scale points while export masking still uses pre-Sim(3) depth."
                    )
                    break

    return warnings


def write_scene_diagnostics_report(
    output_dir: str | Path,
    runtime_options: dict,
    chunk_indices: Sequence[tuple[int, int]] | None = None,
    frame_keys: Sequence[str] | None = None,
    chunk_diagnostics: Sequence[dict] | None = None,
    saved_frame_diagnostics: Sequence[dict] | None = None,
    sequential_sim3: Sequence[tuple] | None = None,
    cumulative_sim3: Sequence[tuple] | None = None,
    points_path: str | Path | None = None,
    fused_points_path: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir)
    diagnostics_path = output_dir / "diagnostics.json"

    chunk_entries = [dict(item) for item in (chunk_diagnostics or [])]
    if not chunk_entries and chunk_indices is not None:
        for chunk_idx, chunk_range in enumerate(chunk_indices):
            chunk_entries.append(
                {
                    "chunk_idx": int(chunk_idx),
                    "chunk_range": [int(chunk_range[0]), int(chunk_range[1])],
                    "saved_frame_range": None,
                    "num_frames": int(chunk_range[1] - chunk_range[0]),
                    "num_saved_frames": None,
                    "raw_chunk_stats": None,
                    "raw_saved_stats": None,
                    "point_filter_raw_stats": None,
                    "post_alignment": None,
                    "notes": ["Raw chunk diagnostics unavailable; report built from existing artifacts only."],
                }
            )

    pcd_dir = output_dir / "pcd"
    pcd_stats = []
    if pcd_dir.exists():
        for ply_path in sorted(pcd_dir.glob("*.ply")):
            pcd_stats.append(_point_cloud_stats_from_ply(ply_path))

    artifacts = {
        "pcd_chunks": pcd_stats,
        "points_da3": _point_cloud_stats_from_ply(Path(points_path)) if points_path is not None else {"exists": False},
        "points_da3_fused": (
            _point_cloud_stats_from_ply(Path(fused_points_path)) if fused_points_path is not None else {"exists": False}
        ),
        "camera_poses_txt": _camera_pose_stats_from_txt(output_dir / "camera_poses.txt"),
        "camera_poses_ply": _point_cloud_stats_from_ply(output_dir / "camera_poses.ply"),
        "results_output_dir": {
            "path": str(output_dir / "results_output"),
            "exists": (output_dir / "results_output").exists(),
        },
    }

    diagnostics = {
        "runtime_options": runtime_options,
        "num_chunks": len(chunk_entries),
        "num_saved_frames": len(saved_frame_diagnostics or []),
        "frame_count": len(frame_keys) if frame_keys is not None else None,
        "chunks": chunk_entries,
        "saved_frame_diagnostics": list(saved_frame_diagnostics or []),
        "sim3": {
            "sequential": _sim3_entries(sequential_sim3 or [], target_chunk_zero=False),
            "cumulative": _sim3_entries(cumulative_sim3 or [], target_chunk_zero=True),
        },
        "artifacts": artifacts,
        "notes": [
            "Aligned chunk point export masks use raw depth before Sim(3) scaling; post-alignment stats are diagnostic only."
        ],
    }
    diagnostics["warnings"] = _build_warning_messages(diagnostics)

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    for msg in diagnostics["warnings"]:
        print(msg)
    print(f"Saved scene diagnostics to {diagnostics_path}")
    return diagnostics_path


def analyze_existing_scene_outputs(
    output_dir: str | Path,
    runtime_options: dict,
    frame_keys: Sequence[str],
    chunk_indices: Sequence[tuple[int, int]],
    depth_threshold_m: float,
    points_path: str | Path | None = None,
    fused_points_path: str | Path | None = None,
    sky_mask_path: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir)
    results_dir = output_dir / "results_output"
    sky_masks = None
    if sky_mask_path is not None and Path(sky_mask_path).exists():
        sky_masks = np.load(Path(sky_mask_path))

    saved_frame_diagnostics: list[dict] = []
    per_frame_lookup: dict[str, dict] = {}
    if results_dir.exists():
        for frame_key in frame_keys:
            npz_path = results_dir / f"{frame_key}.npz"
            if not npz_path.exists():
                continue
            with np.load(npz_path) as payload:
                depth = np.asarray(payload["depth"], dtype=np.float32)
                conf = np.asarray(payload["conf"], dtype=np.float32)
            sky = None
            if sky_masks is not None and frame_key in sky_masks.files:
                sky = np.asarray(sky_masks[frame_key]).astype(bool)
            stats = _summarize_depth_conf(depth, conf, sky, depth_threshold_m)
            stats.update(
                {
                    "chunk_idx": None,
                    "global_index": int(frame_keys.index(frame_key)),
                    "local_index": None,
                    "frame_key": frame_key,
                    "depth_scale_applied": None,
                }
            )
            saved_frame_diagnostics.append(stats)
            per_frame_lookup[frame_key] = stats

    chunk_entries: list[dict] = []
    overlap = int(runtime_options.get("overlap", 0))
    overlap_e = overlap
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_indices):
        chunk_len = chunk_end - chunk_start
        if len(chunk_indices) == 1:
            save_indices = list(range(0, chunk_len))
        elif chunk_idx == 0:
            save_indices = list(range(0, chunk_len - overlap_e))
        elif chunk_idx == len(chunk_indices) - 1:
            save_indices = list(range(0, chunk_len))
        else:
            save_indices = list(range(0, chunk_len - overlap_e))
        save_keys = [frame_keys[chunk_start + i] for i in save_indices if (chunk_start + i) < len(frame_keys)]

        depth_stack = []
        conf_stack = []
        sky_stack = []
        for frame_key in save_keys:
            npz_path = results_dir / f"{frame_key}.npz"
            if not npz_path.exists():
                continue
            with np.load(npz_path) as payload:
                depth_stack.append(np.asarray(payload["depth"], dtype=np.float32))
                conf_stack.append(np.asarray(payload["conf"], dtype=np.float32))
            if sky_masks is not None and frame_key in sky_masks.files:
                sky_stack.append(np.asarray(sky_masks[frame_key]).astype(bool))

        saved_stats = None
        if depth_stack:
            saved_stats = _summarize_depth_conf(
                np.stack(depth_stack),
                np.stack(conf_stack),
                np.stack(sky_stack) if len(sky_stack) == len(depth_stack) and len(depth_stack) > 0 else None,
                depth_threshold_m,
            )

        chunk_entries.append(
            {
                "chunk_idx": int(chunk_idx),
                "chunk_range": [int(chunk_start), int(chunk_end)],
                "saved_frame_range": _saved_range(chunk_start, save_indices),
                "num_frames": int(chunk_end - chunk_start),
                "num_saved_frames": int(len(save_indices)),
                "saved_frame_keys": [save_keys[0], save_keys[-1]] if save_keys else None,
                "raw_chunk_stats": None,
                "raw_saved_stats": None,
                "point_filter_raw_stats": None,
                "saved_output_stats": saved_stats,
                "post_alignment": None,
                "notes": [
                    "Report built from existing results_output/PLY artifacts; raw chunk and Sim(3) internals are unavailable."
                ],
            }
        )
        for frame_key in save_keys:
            if frame_key in per_frame_lookup:
                per_frame_lookup[frame_key]["chunk_idx"] = int(chunk_idx)
                per_frame_lookup[frame_key]["local_index"] = int(frame_keys.index(frame_key) - chunk_start)

    return write_scene_diagnostics_report(
        output_dir=output_dir,
        runtime_options=runtime_options,
        chunk_indices=chunk_indices,
        frame_keys=frame_keys,
        chunk_diagnostics=chunk_entries,
        saved_frame_diagnostics=saved_frame_diagnostics,
        sequential_sim3=[],
        cumulative_sim3=[],
        points_path=points_path,
        fused_points_path=fused_points_path,
    )


class DA3_Streaming:
    def __init__(
        self,
        image_dir,
        save_dir,
        config,
        model: DepthAnything3 | None = None,
        model_id: str | None = None,
        image_paths: Sequence[str] | None = None,
        loop_image_paths: Sequence[str] | None = None,
        frame_keys: Sequence[str] | None = None,
        image_preprocessor: Callable[[str, int], np.ndarray | str] | None = None,
        loop_image_loader: Callable[[str], object] | None = None,
        pose_condition: bool = False,
        input_extrinsics: np.ndarray | None = None,
        input_intrinsics: np.ndarray | None = None,
        shared_intrinsics_mode: str = "none",
        process_res: int | None = None,
        process_res_method: str | None = None,
        save_sky_mask: bool = False,
        sky_mask_dir: str | None = None,
        save_depth100_mask: bool = False,
        depth100_mask_path: str | None = None,
        exclude_depth_above_100_for_points: bool = False,
        depth_threshold_m: float = 100.0,
        runtime_options: dict | None = None,
    ):
        self.config = config

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.overlap_s = 0
        self.overlap_e = self.overlap - self.overlap_s
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.dtype = (
                torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            )
        else:
            self.dtype = torch.float32

        self.img_dir = image_dir
        self.img_list = None
        self.image_paths_override = list(image_paths) if image_paths is not None else None
        self.loop_image_paths_override = list(loop_image_paths) if loop_image_paths is not None else None
        self.frame_keys_override = list(frame_keys) if frame_keys is not None else None
        self.output_dir = save_dir
        self.image_preprocessor = image_preprocessor
        self.loop_image_loader = loop_image_loader

        self.pose_condition = bool(pose_condition)
        self.shared_intrinsics_mode = str(shared_intrinsics_mode or "none")
        if self.shared_intrinsics_mode not in {"none", "first_chunk_median"}:
            raise ValueError(
                "shared_intrinsics_mode must be one of {'none', 'first_chunk_median'}; "
                f"got {self.shared_intrinsics_mode!r}."
            )
        if self.pose_condition and self.shared_intrinsics_mode != "none":
            raise ValueError(
                "shared_intrinsics_mode requires pose_condition=False; "
                "disable --pose-condition before enabling shared intrinsics."
            )
        self.input_extrinsics = (
            np.asarray(input_extrinsics, dtype=np.float32) if input_extrinsics is not None else None
        )
        if self.input_extrinsics is not None and self.input_extrinsics.shape[-2:] == (3, 4):
            pad = np.zeros((*self.input_extrinsics.shape[:-2], 4, 4), dtype=np.float32)
            pad[..., :3, :4] = self.input_extrinsics
            pad[..., 3, 3] = 1.0
            self.input_extrinsics = pad
        self.input_intrinsics = (
            np.asarray(input_intrinsics, dtype=np.float32) if input_intrinsics is not None else None
        )
        self.process_res = process_res
        self.process_res_method = process_res_method

        self.save_sky_mask = bool(save_sky_mask)
        self.save_depth100_mask = bool(save_depth100_mask)
        self.exclude_depth_above_100_for_points = bool(exclude_depth_above_100_for_points)
        self.depth_threshold_m = float(depth_threshold_m)
        self.runtime_options = {
            **(runtime_options or {}),
            "pose_condition": bool(pose_condition),
            "chunk_size": int(self.chunk_size),
            "overlap": int(self.overlap),
            "depth_threshold_m": float(self.depth_threshold_m),
            "shared_intrinsics": self.shared_intrinsics_mode != "none",
            "shared_intrinsics_mode": self.shared_intrinsics_mode,
            "resolved_shared_intrinsics": None,
            "shared_intrinsics_source_chunk_idx": None,
        }
        self._missing_sky_warned = False
        self._sky_mask_dict: dict[str, np.ndarray] = {}
        self._depth100_mask_dict: dict[str, np.ndarray] = {}
        self.shared_intrinsics_matrix: np.ndarray | None = None
        self.shared_intrinsics_source_chunk_idx: int | None = None

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.result_output_dir = os.path.join(save_dir, "results_output")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        self.sky_mask_dir = sky_mask_dir
        self.sky_mask_path = None
        if self.save_sky_mask:
            if self.sky_mask_dir is None:
                self.sky_mask_dir = os.path.join(save_dir, "sky_mask")
            os.makedirs(self.sky_mask_dir, exist_ok=True)
            self.sky_mask_path = os.path.join(self.sky_mask_dir, "masks.npz")

        self.depth100_mask_path = depth100_mask_path
        if self.save_depth100_mask:
            if self.depth100_mask_path is None:
                self.depth100_mask_path = os.path.join(save_dir, "depth100_mask", "masks.npz")
            Path(self.depth100_mask_path).parent.mkdir(parents=True, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []
        self.frame_keys = []
        self.chunk_diagnostics: list[dict] = []
        self.saved_frame_diagnostics: list[dict] = []
        self.sequential_sim3_list: list[tuple] = []
        self.cumulative_sim3_list: list[tuple] = []

        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        if model is not None:
            print("Using external preloaded model...")
            self.model = model.to(self.device)
            self.model.eval()
        elif model_id is not None:
            print(f"Loading model from pretrained id: {model_id}")
            self.model = DepthAnything3.from_pretrained(model_id).to(self.device)
            self.model.eval()
        else:
            print("Loading model from local DA3 weights...")
            with open(self.config["Weights"]["DA3_CONFIG"]) as f:
                config = json.load(f)
            self.model = DepthAnything3(**config)
            weight = load_file(self.config["Weights"]["DA3"])
            self.model.load_state_dict(weight, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)

        self.skyseg_session = None

        self.chunk_indices = None  # [(begin_idx, end_idx), ...]

        self.loop_list = []  # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config["Model"]["loop_enable"]
        self.loop_detector = None

        if self.loop_enable:
            loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
            self.loop_detector = LoopDetector(
                image_dir=image_dir,
                output=loop_info_save_path,
                config=self.config,
                image_paths=self.loop_image_paths_override,
                image_loader=self.loop_image_loader,
            )
            self.loop_detector.load_model()

        print("init done.")

    def _chunk_save_local_indices(self, chunk_idx: int) -> list[int]:
        chunk_start, chunk_end = self.chunk_indices[chunk_idx]
        chunk_len = chunk_end - chunk_start
        if len(self.chunk_indices) == 1:
            return list(range(0, chunk_len))
        if chunk_idx == 0:
            return list(range(0, chunk_len - self.overlap_e))
        if chunk_idx == len(self.chunk_indices) - 1:
            return list(range(self.overlap_s, chunk_len))
        return list(range(self.overlap_s, chunk_len - self.overlap_e))

    def _normalize_prediction_extrinsics(self, predictions) -> None:
        if predictions.extrinsics is None:
            return
        ext = np.asarray(predictions.extrinsics, dtype=np.float32)
        if ext.ndim != 3:
            raise ValueError(f"Unexpected prediction.extrinsics shape: {ext.shape}")
        if ext.shape[-2:] == (4, 4):
            predictions.extrinsics = ext[:, :3, :]
            return
        if ext.shape[-2:] == (3, 4):
            predictions.extrinsics = ext
            return
        raise ValueError(f"Unsupported prediction.extrinsics shape: {ext.shape}")

    def _get_chunk_pose_inputs(
        self, chunk_global_indices: Sequence[int]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.pose_condition:
            return None, None
        if self.input_extrinsics is None or self.input_intrinsics is None:
            raise ValueError(
                "pose_condition=True requires input_extrinsics and input_intrinsics aligned with frames."
            )
        ex_chunk = self.input_extrinsics[list(chunk_global_indices)].astype(np.float32, copy=False)
        in_chunk = self.input_intrinsics[list(chunk_global_indices)].astype(np.float32, copy=False)
        return ex_chunk, in_chunk

    def _prepare_chunk_images(
        self, chunk_image_paths: Sequence[str], chunk_global_indices: Sequence[int]
    ) -> list[str | np.ndarray]:
        if self.image_preprocessor is None:
            return list(chunk_image_paths)
        images = []
        for image_path, global_idx in zip(chunk_image_paths, chunk_global_indices):
            images.append(self.image_preprocessor(image_path, global_idx))
        return images

    def _build_invalid_mask(
        self, depth: np.ndarray, sky: np.ndarray | None
    ) -> np.ndarray | None:
        depth_batch = np.asarray(depth, dtype=np.float32)
        if depth_batch.ndim == 2:
            depth_batch = depth_batch[None]
        invalid_mask = None
        if sky is not None:
            sky_mask = np.asarray(sky).astype(bool)
            if sky_mask.ndim == 2:
                sky_mask = sky_mask[None]
            invalid_mask = sky_mask.copy()
        if self.exclude_depth_above_100_for_points:
            depth_mask = depth_batch > self.depth_threshold_m
            invalid_mask = depth_mask if invalid_mask is None else (invalid_mask | depth_mask)
        return invalid_mask

    def _update_shared_intrinsics_runtime_options(self) -> None:
        self.runtime_options["shared_intrinsics"] = self.shared_intrinsics_mode != "none"
        self.runtime_options["shared_intrinsics_mode"] = self.shared_intrinsics_mode
        self.runtime_options["resolved_shared_intrinsics"] = (
            None
            if self.shared_intrinsics_matrix is None
            else np.asarray(self.shared_intrinsics_matrix, dtype=np.float32).tolist()
        )
        self.runtime_options["shared_intrinsics_source_chunk_idx"] = (
            None
            if self.shared_intrinsics_source_chunk_idx is None
            else int(self.shared_intrinsics_source_chunk_idx)
        )

    def _estimate_shared_intrinsics(
        self, intrinsics: np.ndarray, chunk_idx: int | None
    ) -> np.ndarray:
        intrinsics_arr = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics_arr.ndim != 3 or intrinsics_arr.shape[-2:] != (3, 3):
            raise ValueError(
                "Expected predicted intrinsics with shape [N, 3, 3] for shared-intrinsics "
                f"estimation, got {intrinsics_arr.shape}."
            )
        if intrinsics_arr.shape[0] == 0:
            raise ValueError("Cannot estimate shared intrinsics from an empty chunk.")

        shared = np.eye(3, dtype=np.float32)
        shared[0, 0] = float(np.median(intrinsics_arr[:, 0, 0]))
        shared[1, 1] = float(np.median(intrinsics_arr[:, 1, 1]))
        shared[0, 2] = float(np.median(intrinsics_arr[:, 0, 2]))
        shared[1, 2] = float(np.median(intrinsics_arr[:, 1, 2]))
        self.shared_intrinsics_matrix = shared
        self.shared_intrinsics_source_chunk_idx = None if chunk_idx is None else int(chunk_idx)
        self._update_shared_intrinsics_runtime_options()
        print(
            "Using shared intrinsics estimated from first non-loop chunk "
            f"{self.shared_intrinsics_source_chunk_idx}: "
            f"fx={shared[0, 0]:.3f}, fy={shared[1, 1]:.3f}, "
            f"cx={shared[0, 2]:.3f}, cy={shared[1, 2]:.3f}"
        )
        return shared

    def _maybe_apply_shared_intrinsics(self, predictions, chunk_idx: int | None, is_loop: bool) -> None:
        if self.shared_intrinsics_mode == "none":
            return
        if predictions.intrinsics is None:
            raise RuntimeError(
                "shared_intrinsics_mode requested, but model inference did not return intrinsics."
            )
        intrinsics = np.asarray(predictions.intrinsics, dtype=np.float32)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics[None]
        if self.shared_intrinsics_matrix is None:
            if is_loop:
                raise RuntimeError(
                    "Loop chunk requested before shared intrinsics were estimated from a "
                    "regular streaming chunk."
                )
            self._estimate_shared_intrinsics(intrinsics, chunk_idx)

        if self.shared_intrinsics_matrix is None:
            raise RuntimeError("Failed to initialize shared intrinsics.")
        predictions.intrinsics = np.broadcast_to(
            self.shared_intrinsics_matrix[None, :, :], intrinsics.shape
        ).copy()

    def _mask_confidence(self, conf: np.ndarray, invalid_mask: np.ndarray | None) -> np.ndarray:
        conf_out = np.asarray(conf, dtype=np.float32).copy()
        if invalid_mask is None:
            return conf_out
        conf_out[invalid_mask] = -1e6
        return conf_out

    def _ensure_chunk_diag_entry(self, chunk_idx: int) -> dict:
        while len(self.chunk_diagnostics) <= chunk_idx:
            self.chunk_diagnostics.append({})
        return self.chunk_diagnostics[chunk_idx]

    def _record_chunk_raw_diagnostics(self, predictions, chunk_idx: int) -> None:
        chunk_start, chunk_end = self.chunk_indices[chunk_idx]
        save_indices = self._chunk_save_local_indices(chunk_idx)
        depth = np.asarray(predictions.depth, dtype=np.float32)
        conf = np.asarray(predictions.conf, dtype=np.float32)
        sky = getattr(predictions, "sky", None)
        sky_arr = None if sky is None else np.asarray(sky).astype(bool)

        entry = self._ensure_chunk_diag_entry(chunk_idx)
        entry.update(
            {
                "chunk_idx": int(chunk_idx),
                "chunk_range": [int(chunk_start), int(chunk_end)],
                "saved_frame_range": _saved_range(chunk_start, save_indices),
                "num_frames": int(chunk_end - chunk_start),
                "num_saved_frames": int(len(save_indices)),
                "saved_frame_keys": (
                    [self.frame_keys[chunk_start + save_indices[0]], self.frame_keys[chunk_start + save_indices[-1]]]
                    if len(save_indices) > 0
                    else None
                ),
                "raw_chunk_stats": _summarize_depth_conf(depth, conf, sky_arr, self.depth_threshold_m),
                "raw_saved_stats": (
                    _summarize_depth_conf(
                        depth[list(save_indices)],
                        conf[list(save_indices)],
                        None if sky_arr is None else sky_arr[list(save_indices)],
                        self.depth_threshold_m,
                    )
                    if len(save_indices) > 0
                    else None
                ),
                "point_filter_raw_stats": None,
                "saved_output_stats": None,
                "post_alignment": None,
                "notes": [],
            }
        )

    def _record_chunk_export_diagnostics(
        self,
        chunk_idx: int,
        depth: np.ndarray,
        conf: np.ndarray,
        sky: np.ndarray | None,
        invalid_mask: np.ndarray | None,
        world_points,
        cumulative_scale: float,
        aligned: bool,
    ) -> None:
        chunk_start, _ = self.chunk_indices[chunk_idx]
        save_indices = self._chunk_save_local_indices(chunk_idx)
        depth_arr = np.asarray(depth, dtype=np.float32)
        conf_arr = np.asarray(conf, dtype=np.float32)
        sky_arr = None if sky is None else np.asarray(sky).astype(bool)
        scaled_depth = depth_arr * float(cumulative_scale)
        entry = self._ensure_chunk_diag_entry(chunk_idx)
        entry["point_filter_raw_stats"] = _build_point_filter_stats(
            depth_arr,
            conf_arr,
            sky_arr,
            invalid_mask,
            self.depth_threshold_m,
            self.exclude_depth_above_100_for_points,
        )
        entry["saved_output_stats"] = (
            _summarize_depth_conf(
                scaled_depth[list(save_indices)],
                conf_arr[list(save_indices)],
                None if sky_arr is None else sky_arr[list(save_indices)],
                self.depth_threshold_m,
            )
            if len(save_indices) > 0
            else None
        )
        if aligned:
            world_points_np = world_points.detach().cpu().numpy() if torch.is_tensor(world_points) else np.asarray(world_points)
            entry["post_alignment"] = {
                "cumulative_scale": _safe_float(cumulative_scale),
                "scaled_depth_stats": _summarize_depth_conf(
                    scaled_depth,
                    conf_arr,
                    sky_arr,
                    self.depth_threshold_m,
                ),
                "world_point_stats": _point_stats_from_points(world_points_np),
                "export_filter_note": (
                    "Point export masking uses raw depth and sky before Sim(3) scaling; "
                    "scaled depth/world-point values are diagnostic only."
                ),
            }
        self.saved_frame_diagnostics = [item for item in self.saved_frame_diagnostics if item.get("chunk_idx") != chunk_idx]
        self.saved_frame_diagnostics.extend(
            _saved_frame_entries(
                scaled_depth,
                conf_arr,
                sky_arr,
                self.frame_keys,
                chunk_idx,
                chunk_start,
                save_indices,
                self.depth_threshold_m,
                float(cumulative_scale),
            )
        )

    def write_scene_diagnostics(
        self,
        points_path: str | Path | None = None,
        fused_points_path: str | Path | None = None,
    ) -> Path:
        return write_scene_diagnostics_report(
            output_dir=self.output_dir,
            runtime_options=self.runtime_options,
            chunk_indices=self.chunk_indices,
            frame_keys=self.frame_keys,
            chunk_diagnostics=self.chunk_diagnostics,
            saved_frame_diagnostics=self.saved_frame_diagnostics,
            sequential_sim3=self.sequential_sim3_list,
            cumulative_sim3=self.cumulative_sim3_list,
            points_path=points_path,
            fused_points_path=fused_points_path,
        )

    def _save_optional_masks(self, predictions, chunk_idx: int) -> None:
        if not self.save_sky_mask and not self.save_depth100_mask:
            return

        depth = np.asarray(predictions.depth, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None]
        sky = getattr(predictions, "sky", None)
        if sky is not None:
            sky = np.asarray(sky).astype(bool)
            if sky.ndim == 2:
                sky = sky[None]
        elif not self._missing_sky_warned:
            print("[WARN] prediction.sky is missing; sky-based mask terms are skipped.")
            self._missing_sky_warned = True

        chunk_start, _ = self.chunk_indices[chunk_idx]
        for local_idx in self._chunk_save_local_indices(chunk_idx):
            global_idx = chunk_start + local_idx
            frame_key = self.frame_keys[global_idx]

            sky_i = None
            if sky is not None:
                sky_i = sky[local_idx].astype(np.int8, copy=False)
                if self.save_sky_mask:
                    self._sky_mask_dict[frame_key] = sky_i

            if self.save_depth100_mask:
                mask = depth[local_idx] > self.depth_threshold_m
                if sky_i is not None:
                    mask = np.logical_or(mask, sky_i.astype(bool))
                self._depth100_mask_dict[frame_key] = mask.astype(np.int8, copy=False)

    def _flush_sky_masks(self) -> None:
        if not self.save_sky_mask or self.sky_mask_path is None:
            return
        if len(self._sky_mask_dict) == 0:
            print("[INFO] No sky masks to save.")
            return
        np.savez_compressed(self.sky_mask_path, **self._sky_mask_dict)
        print(f"Saved sky masks to {self.sky_mask_path}")

    def _flush_depth100_masks(self) -> None:
        if not self.save_depth100_mask or self.depth100_mask_path is None:
            return
        if len(self._depth100_mask_dict) == 0:
            print("[INFO] No depth100 masks to save.")
            return
        np.savez_compressed(self.depth100_mask_path, **self._depth100_mask_dict)
        print(f"Saved depth100+sky masks to {self.depth100_mask_path}")

    def get_loop_pairs(self):
        self.loop_detector.run()
        loop_list = self.loop_detector.get_loop_list()
        return loop_list

    def save_depth_conf_result(self, predictions, chunk_idx, s, R, T):
        if not self.config["Model"]["save_depth_conf_result"]:
            return
        os.makedirs(self.result_output_dir, exist_ok=True)

        chunk_start, _ = self.chunk_indices[chunk_idx]
        save_indices = self._chunk_save_local_indices(chunk_idx)

        print("[save_depth_conf_result] save_indices:")

        for local_idx in save_indices:
            global_idx = chunk_start + local_idx
            print(f"{global_idx}, ", end="")

            depth = predictions.depth[local_idx]  # [H, W] float32
            conf = predictions.conf[local_idx]  # [H, W] float32
            intrinsics = predictions.intrinsics[local_idx]  # [3, 3] float32

            frame_key = (
                self.frame_keys[global_idx]
                if global_idx < len(self.frame_keys)
                else f"frame_{global_idx:05d}"
            )
            filename = f"{frame_key}.npz"
            filepath = os.path.join(self.result_output_dir, filename)

            if self.config["Model"]["save_debug_info"]:
                np.savez_compressed(
                    filepath,
                    depth=depth,
                    conf=conf,
                    intrinsics=intrinsics,
                    extrinsics=predictions.extrinsics[local_idx],
                    s=s,
                    R=R,
                    T=T,
                )
            else:
                np.savez_compressed(
                    filepath, depth=depth, conf=conf, intrinsics=intrinsics
                )
        print("")

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = list(self.img_list[start_idx:end_idx])
        chunk_global_indices = list(range(start_idx, end_idx))
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += list(self.img_list[start_idx:end_idx])
            chunk_global_indices += list(range(start_idx, end_idx))

        print(f"Loaded {len(chunk_image_paths)} images")

        ref_view_strategy = self.config["Model"][
            "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"
        ]
        images = self._prepare_chunk_images(chunk_image_paths, chunk_global_indices)
        ex_chunk, in_chunk = self._get_chunk_pose_inputs(chunk_global_indices)

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                inference_kwargs = {
                    "image": images,
                    "ref_view_strategy": ref_view_strategy,
                }
                if self.process_res is not None:
                    inference_kwargs["process_res"] = self.process_res
                if self.process_res_method is not None:
                    inference_kwargs["process_res_method"] = self.process_res_method
                if self.pose_condition:
                    inference_kwargs["extrinsics"] = ex_chunk
                    inference_kwargs["intrinsics"] = in_chunk
                    inference_kwargs["align_to_input_ext_scale"] = False

                predictions = self.model.inference(**inference_kwargs)
                self._normalize_prediction_extrinsics(predictions)

                depth = np.asarray(predictions.depth, dtype=np.float32)
                if depth.ndim == 2:
                    depth = depth[None]
                predictions.depth = depth

                if predictions.conf is None:
                    raise RuntimeError("Prediction confidence map is required for DA3 streaming.")
                conf = np.asarray(predictions.conf, dtype=np.float32)
                if conf.ndim == 2:
                    conf = conf[None]
                predictions.conf = conf - 1.0

                if predictions.processed_images is not None:
                    proc = np.asarray(predictions.processed_images)
                    if proc.ndim == 3:
                        proc = proc[None]
                    predictions.processed_images = proc
                if predictions.intrinsics is not None:
                    intr = np.asarray(predictions.intrinsics, dtype=np.float32)
                    if intr.ndim == 2:
                        intr = intr[None]
                    predictions.intrinsics = intr
                self._maybe_apply_shared_intrinsics(predictions, chunk_idx=chunk_idx, is_loop=is_loop)
                if predictions.sky is not None:
                    sky = np.asarray(predictions.sky).astype(bool)
                    if sky.ndim == 2:
                        sky = sky[None]
                    predictions.sky = sky

                print(predictions.processed_images.shape)  # [N, H, W, 3] uint8
                print(predictions.depth.shape)  # [N, H, W] float32
                print(predictions.conf.shape)  # [N, H, W] float32
                if predictions.extrinsics is not None:
                    print(predictions.extrinsics.shape)  # [N, 3, 4] float32 (w2c)
                if predictions.intrinsics is not None:
                    print(predictions.intrinsics.shape)  # [N, 3, 3] float32
        torch.cuda.empty_cache()

        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions.extrinsics
            intrinsics = predictions.intrinsics
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        np.save(save_path, predictions)

        return predictions

    def get_chunk_indices(self):
        return compute_chunk_indices(len(self.img_list), self.chunk_size, self.overlap)

    def align_2pcds(
        self,
        point_map1,
        conf1,
        point_map2,
        conf2,
        chunk1_depth,
        chunk2_depth,
        chunk1_depth_conf,
        chunk2_depth_conf,
    ):

        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor_return, quality_score, method_used = precompute_scale_chunks_with_depth(
                chunk1_depth,
                chunk1_depth_conf,
                chunk2_depth,
                chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            print(
                f"[Depth Scale Precompute] scale: {scale_factor_return}, \
                    quality_score: {quality_score}, method_used: {method_used}"
            )
            scale_factor = scale_factor_return

        s, R, t = weighted_align_point_maps(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )
        print("Estimated Scale:", s)
        print("Estimated Rotation:\n", R)
        print("Estimated Translation:", t)

        return s, R, t

    def get_loop_sim3_from_loop_predict(self, loop_predict_list):
        loop_sim3_list = []
        for item in loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop_org = depth_to_point_cloud_vectorized(
                item[1].depth, item[1].intrinsics, item[1].extrinsics
            )

            chunk_a_s = 0
            chunk_a_e = chunk_a_len = chunk_a_range[1] - chunk_a_range[0]
            chunk_b_s = -chunk_b_range[1] + chunk_b_range[0]
            chunk_b_e = point_map_loop_org.shape[0]
            chunk_b_len = chunk_b_range[1] - chunk_b_range[0]

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_len
            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_len

            print("chunk_a align")

            point_map_loop_a = point_map_loop_org[chunk_a_s:chunk_a_e]
            conf_loop = item[1].conf[chunk_a_s:chunk_a_e]
            print(self.chunk_indices[chunk_idx_a])
            print(chunk_a_range)
            print(chunk_a_rela_begin, chunk_a_rela_end)
            chunk_data_a = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                allow_pickle=True,
            ).item()

            point_map_a = depth_to_point_cloud_vectorized(
                chunk_data_a.depth, chunk_data_a.intrinsics, chunk_data_a.extrinsics
            )
            point_map_a = point_map_a[chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_a_depth = np.squeeze(chunk_data_a.depth[chunk_a_rela_begin:chunk_a_rela_end])
                chunk_a_depth_conf = np.squeeze(
                    chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]
                )
                chunk_a_loop_depth = np.squeeze(item[1].depth[chunk_a_s:chunk_a_e])
                chunk_a_loop_depth_conf = np.squeeze(item[1].conf[chunk_a_s:chunk_a_e])
            else:
                chunk_a_depth = None
                chunk_a_loop_depth = None
                chunk_a_depth_conf = None
                chunk_a_loop_depth_conf = None

            s_a, R_a, t_a = self.align_2pcds(
                point_map_a,
                conf_a,
                point_map_loop_a,
                conf_loop,
                chunk_a_depth,
                chunk_a_loop_depth,
                chunk_a_depth_conf,
                chunk_a_loop_depth_conf,
            )

            print("chunk_b align")

            point_map_loop_b = point_map_loop_org[chunk_b_s:chunk_b_e]
            conf_loop = item[1].conf[chunk_b_s:chunk_b_e]
            print(self.chunk_indices[chunk_idx_b])
            print(chunk_b_range)
            print(chunk_b_rela_begin, chunk_b_rela_end)
            chunk_data_b = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                allow_pickle=True,
            ).item()

            point_map_b = depth_to_point_cloud_vectorized(
                chunk_data_b.depth, chunk_data_b.intrinsics, chunk_data_b.extrinsics
            )
            point_map_b = point_map_b[chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_b_depth = np.squeeze(chunk_data_b.depth[chunk_b_rela_begin:chunk_b_rela_end])
                chunk_b_depth_conf = np.squeeze(
                    chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]
                )
                chunk_b_loop_depth = np.squeeze(item[1].depth[chunk_b_s:chunk_b_e])
                chunk_b_loop_depth_conf = np.squeeze(item[1].conf[chunk_b_s:chunk_b_e])
            else:
                chunk_b_depth = None
                chunk_b_loop_depth = None
                chunk_b_depth_conf = None
                chunk_b_loop_depth_conf = None

            s_b, R_b, t_b = self.align_2pcds(
                point_map_b,
                conf_b,
                point_map_loop_b,
                conf_loop,
                chunk_b_depth,
                chunk_b_loop_depth,
                chunk_b_depth_conf,
                chunk_b_loop_depth_conf,
            )

            print("a -> b SIM 3")
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            print("Estimated Scale:", s_ab)
            print("Estimated Rotation:\n", R_ab)
            print("Estimated Translation:", t_ab)

            loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        return loop_sim3_list

    def plot_loop_closure(
        self, input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
    ):
        def extract_xyz(pose_tensor):
            poses = pose_tensor.cpu().numpy()
            return poses[:, 0], poses[:, 1], poses[:, 2]

        x0, _, y0 = extract_xyz(input_abs_poses)
        x1, _, y1 = extract_xyz(optimized_abs_poses)

        # Visual in png format
        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
        plt.plot(x1, y1, "o-", label="After Optimization")
        for i, j, _ in self.loop_sim3_list:
            plt.plot(
                [x0[i], x0[j]],
                [y0[i], y0[j]],
                "r--",
                alpha=0.25,
                label="Loop (Before)" if i == 5 else "",
            )
            plt.plot(
                [x1[i], x1[j]],
                [y1[i], y1[j]],
                "g-",
                alpha=0.25,
                label="Loop (After)" if i == 5 else "",
            )
        plt.gca().set_aspect("equal")
        plt.title("Sim3 Loop Closure Optimization")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) \
                    must be less than chunk size ({self.chunk_size})"
            )

        self.chunk_indices, num_chunks = self.get_chunk_indices()

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} \
                chunks of size {self.chunk_size} with {self.overlap} overlap"
        )

        pre_predictions = None
        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)}")
            cur_predictions = self.process_single_chunk(
                self.chunk_indices[chunk_idx], chunk_idx=chunk_idx
            )
            self._record_chunk_raw_diagnostics(cur_predictions, chunk_idx)
            torch.cuda.empty_cache()

            if chunk_idx > 0:
                print(
                    f"Aligning {chunk_idx-1} and {chunk_idx} (Total {len(self.chunk_indices)-1})"
                )
                chunk_data1 = pre_predictions
                chunk_data2 = cur_predictions

                point_map1 = depth_to_point_cloud_vectorized(
                    chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
                )

                point_map1 = point_map1[-self.overlap :]
                point_map2 = point_map2[: self.overlap]
                conf1 = chunk_data1.conf[-self.overlap :]
                conf2 = chunk_data2.conf[: self.overlap]

                if self.config["Model"]["align_method"] == "scale+se3":
                    chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                    chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                    chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                    chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                s, R, t = self.align_2pcds(
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    chunk1_depth,
                    chunk2_depth,
                    chunk1_depth_conf,
                    chunk2_depth_conf,
                )
                self.sim3_list.append((s, R, t))

            pre_predictions = cur_predictions

        if self.loop_enable:
            self.loop_list = self.get_loop_pairs()
            del self.loop_detector  # Save GPU Memory

            torch.cuda.empty_cache()

            print("Loop SIM(3) estimating...")
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1], range_2=item[3], is_loop=True
                )

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

            self.loop_sim3_list = self.get_loop_sim3_from_loop_predict(self.loop_predict_list)

            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )  # just for plot
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )  # just for plot

            self.plot_loop_closure(
                input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
            )

        self.sequential_sim3_list = [
            (float(s), np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64))
            for s, R, t in self.sim3_list
        ]
        print("Apply alignment")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        self.cumulative_sim3_list = [
            (float(s), np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64))
            for s, R, t in self.sim3_list
        ]
        chunk_data_first = np.load(
            os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
        ).item()
        np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
        points_first = depth_to_point_cloud_vectorized(
            chunk_data_first.depth,
            chunk_data_first.intrinsics,
            chunk_data_first.extrinsics,
        )
        colors_first = chunk_data_first.processed_images
        invalid_mask_first = self._build_invalid_mask(
            chunk_data_first.depth, getattr(chunk_data_first, "sky", None)
        )
        confs_first = self._mask_confidence(chunk_data_first.conf, invalid_mask_first)
        self._record_chunk_export_diagnostics(
            chunk_idx=0,
            depth=chunk_data_first.depth,
            conf=chunk_data_first.conf,
            sky=getattr(chunk_data_first, "sky", None),
            invalid_mask=invalid_mask_first,
            world_points=points_first,
            cumulative_scale=1.0,
            aligned=False,
        )
        conf_thresh_first = (
            float(np.mean(chunk_data_first.conf))
            * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
        )
        ply_path_first = os.path.join(self.pcd_dir, "0_pcd.ply")
        save_confident_pointcloud_batch(
            points=points_first,  # shape: (H, W, 3)
            colors=colors_first,  # shape: (H, W, 3)
            confs=confs_first,  # shape: (H, W)
            output_path=ply_path_first,
            conf_threshold=conf_thresh_first,
            sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
        )
        if self.config["Model"]["save_depth_conf_result"]:
            self.save_depth_conf_result(chunk_data_first, 0, 1, np.eye(3), np.array([0, 0, 0]))
        self._save_optional_masks(chunk_data_first, 0)

        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})")
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            aligned_chunk_data = {}

            aligned_chunk_data["world_points"] = depth_to_point_cloud_optimized_torch(
                chunk_data.depth, chunk_data.intrinsics, chunk_data.extrinsics
            )
            aligned_chunk_data["world_points"] = apply_sim3_direct_torch(
                aligned_chunk_data["world_points"], s, R, t
            )

            aligned_chunk_data["conf"] = chunk_data.conf
            aligned_chunk_data["images"] = chunk_data.processed_images

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, aligned_chunk_data)

            points = aligned_chunk_data["world_points"].reshape(-1, 3)
            colors = (aligned_chunk_data["images"].reshape(-1, 3)).astype(np.uint8)
            invalid_mask = self._build_invalid_mask(chunk_data.depth, getattr(chunk_data, "sky", None))
            confs = self._mask_confidence(aligned_chunk_data["conf"], invalid_mask).reshape(-1)
            self._record_chunk_export_diagnostics(
                chunk_idx=chunk_idx + 1,
                depth=chunk_data.depth,
                conf=aligned_chunk_data["conf"],
                sky=getattr(chunk_data, "sky", None),
                invalid_mask=invalid_mask,
                world_points=aligned_chunk_data["world_points"],
                cumulative_scale=float(s),
                aligned=True,
            )
            conf_thresh = (
                float(np.mean(aligned_chunk_data["conf"]))
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
            )
            ply_path = os.path.join(self.pcd_dir, f"{chunk_idx+1}_pcd.ply")
            save_confident_pointcloud_batch(
                points=points,  # shape: (H, W, 3)
                colors=colors,  # shape: (H, W, 3)
                confs=confs,  # shape: (H, W)
                output_path=ply_path,
                conf_threshold=conf_thresh,
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

            if self.config["Model"]["save_depth_conf_result"]:
                predictions = chunk_data
                predictions.depth = np.asarray(predictions.depth, dtype=np.float32) * float(s)
                self.save_depth_conf_result(predictions, chunk_idx + 1, s, R, t)
            self._save_optional_masks(chunk_data, chunk_idx + 1)

        self.save_camera_poses()
        self._flush_sky_masks()
        self._flush_depth100_masks()

        print("Done.")

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        if self.image_paths_override is not None:
            self.img_list = list(self.image_paths_override)
        else:
            self.img_list = sorted(
                glob.glob(os.path.join(self.img_dir, "*.jpg"))
                + glob.glob(os.path.join(self.img_dir, "*.png"))
                + glob.glob(os.path.join(self.img_dir, "*.jpeg"))
            )
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.frame_keys_override is not None:
            if len(self.frame_keys_override) != len(self.img_list):
                raise ValueError(
                    f"frame_keys length mismatch: {len(self.frame_keys_override)} vs {len(self.img_list)}"
                )
            self.frame_keys = list(self.frame_keys_override)
        else:
            self.frame_keys = [Path(path).stem for path in self.img_list]

        if self.loop_enable and self.loop_image_paths_override is not None:
            if len(self.loop_image_paths_override) != len(self.img_list):
                raise ValueError(
                    "loop_image_paths length mismatch: "
                    f"{len(self.loop_image_paths_override)} vs {len(self.img_list)}"
                )

        if self.pose_condition:
            if self.input_extrinsics is None or self.input_intrinsics is None:
                raise ValueError(
                    "pose_condition=True requires input_extrinsics/input_intrinsics to be provided."
                )
            if len(self.input_extrinsics) != len(self.img_list):
                raise ValueError(
                    f"input_extrinsics length mismatch: {len(self.input_extrinsics)} vs {len(self.img_list)}"
                )
            if len(self.input_intrinsics) != len(self.img_list):
                raise ValueError(
                    f"input_intrinsics length mismatch: {len(self.input_intrinsics)} vs {len(self.img_list)}"
                )

        self.process_long_sequence()

    def save_camera_poses(self):
        """
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        """
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        first_chunk_end = (
            first_chunk_range[1]
            if len(self.all_camera_poses) == 1
            else first_chunk_range[1] - self.overlap_e
        )

        for i, idx in enumerate(
            range(first_chunk_range[0], first_chunk_end)
        ):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i]
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1
            ]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            chunk_range_end = (
                chunk_range[1] - self.overlap_e
                if chunk_idx < len(self.all_camera_poses) - 1
                else chunk_range[1]
            )

            for i, idx in enumerate(range(chunk_range[0] + self.overlap_s, chunk_range_end)):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i + self.overlap_s]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i + self.overlap_s]

        for idx, pose in enumerate(all_poses):
            if pose is None:
                raise RuntimeError(
                    f"Missing reconstructed pose at frame index {idx}. "
                    "Check chunk overlap/save-index logic."
                )
        for idx, intrinsic in enumerate(all_intrinsics):
            if intrinsic is None:
                raise RuntimeError(
                    f"Missing reconstructed intrinsics at frame index {idx}. "
                    "Check chunk overlap/save-index logic."
                )

        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")

        print(f"Camera poses saved to {poses_path}")

        intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
        with open(intrinsics_path, "w") as f:
            for intrinsic in all_intrinsics:
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                f.write(f"{fx} {fy} {cx} {cy}\n")

        print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_poses)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(
                    f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n"
                )

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        """
        Clean up temporary files and calculate reclaimed disk space.

        This method deletes all temporary files generated during processing from three directories:
        - Unaligned results
        - Aligned results
        - Loop results

        ~50 GiB for 4500-frame KITTI 00,
        ~35 GiB for 2700-frame KITTI 05,
        or ~5 GiB for 300-frame short seq.
        """
        if not self.delete_temp_files:
            return

        total_space = 0

        temp_dirs = [
            self.result_unaligned_dir,
            self.result_aligned_dir,
            self.result_loop_dir,
        ]
        for temp_dir in temp_dirs:
            if not os.path.isdir(temp_dir):
                continue
            print(f"Deleting the temp files under {temp_dir}")
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    total_space += os.path.getsize(file_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        print("Deleting temp files done.")

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path

    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DA3-Streaming")
    parser.add_argument("--image_dir", type=str, required=True, help="Image path")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./configs/base_config.yaml",
        help="Image path",
    )
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="Output path")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional HuggingFace model id/path. If set, load model via DepthAnything3.from_pretrained.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir

    if args.output_dir is not None:
        save_dir = args.output_dir
    else:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        exp_dir = "./exps"
        save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"The exp will be saved under dir: {save_dir}")
        copy_file(args.config, save_dir)

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    da3_streaming = DA3_Streaming(image_dir, save_dir, config, model_id=args.model_id)
    da3_streaming.run()
    da3_streaming.close()

    del da3_streaming
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
    input_dir = os.path.join(save_dir, "pcd")
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print("DA3-Streaming done.")
    sys.exit()
