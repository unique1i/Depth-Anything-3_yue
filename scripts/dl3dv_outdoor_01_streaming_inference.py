#!/usr/bin/env python3
"""Step 01: run DL3DV outdoor DA3-Streaming inference."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

"""

### single scene test command example:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
~/local/micromamba/envs/da3/bin/python scripts/dl3dv_outdoor_01_streaming_inference.py \
  --config /home/yli7/repos/Depth-Anything-3/da3_streaming/configs/base_config.yaml \
  --data-root /home/yli7/scratch2/datasets/dl3dv_960p \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200.txt \
  --model-id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --device cuda \
  --start-idx 4807 \
  --end-idx 4808 \
  --undistort \
  --save-sky-mask \
  --save-depth100-mask \
  --process-res 546 \
  --save-depth-conf-result \
  --exclude-depth-above-100-for-points \
  --skip-existing \
  --pose-condition \

# Optional flags for mask exports and depth filtering:
#   --save-sky-mask
#   --save-depth100-mask
#   --exclude-depth-above-100-for-points
#   --shared-intrinsics; only applicable when --no-pose-condition is set, as it freezes intrinsics to a scene-wide estimate from the first non-loop chunk

"""


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DA3_STREAMING_ROOT = REPO_ROOT / "da3_streaming"
for p in (str(SRC_ROOT), str(DA3_STREAMING_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from depth_anything_3.api import DepthAnything3  # noqa: E402
from da3_streaming import (  # noqa: E402
    DA3_Streaming,
    analyze_existing_scene_outputs,
    compute_chunk_indices,
)
from loop_utils.config_utils import load_config  # noqa: E402
from loop_utils.sim3utils import merge_ply_files, warmup_numba  # noqa: E402


DEFAULT_DATA_ROOT = "/home/yli7/scratch2/datasets/dl3dv_960p"
DEFAULT_SPLIT_FILE = "/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200.txt"
DEFAULT_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# local ckpt paths
DEFAULT_DA3_MODEL_PATH = "/home/yli7/scratch2/datasets/ckpts/depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_DINO_SALAD_CKPT = "/home/yli7/repos/Depth-Anything-3/da3_streaming/weights/dino_salad.ckpt"
DEFAULT_DINOV2_HUB_DIR = "/home/yli7/.cache/torch/hub/facebookresearch_dinov2_main"
_DEFAULT_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})


def _read_split_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_existing_weight_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    raw = Path(path_str).expanduser()
    candidates = [
        raw,
        (REPO_ROOT / raw).resolve(),
        (DA3_STREAMING_ROOT / raw).resolve(),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        c = candidate.resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return c
    return None


def _resolve_existing_dir_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    raw = Path(path_str).expanduser()
    candidates = [
        raw,
        (REPO_ROOT / raw).resolve(),
        (DA3_STREAMING_ROOT / raw).resolve(),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        c = candidate.resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.exists() and c.is_dir():
            return c
    return None


def _install_dinov2_torchhub_local_shim(local_repo_dir: Path | None, allow_online_fallback: bool) -> None:
    repo_aliases = {"facebookresearch/dinov2", "facebookresearch/dinov2:main"}
    local_repo_str = str(local_repo_dir) if local_repo_dir is not None else None

    if not hasattr(torch.hub, "_da3_orig_load"):
        torch.hub._da3_orig_load = torch.hub.load
    orig_load: Callable = torch.hub._da3_orig_load

    def _shim(repo_or_dir, model, *args, **kwargs):
        repo_str = str(repo_or_dir)
        if repo_str not in repo_aliases:
            return orig_load(repo_or_dir, model, *args, **kwargs)

        forwarded_kwargs = dict(kwargs)
        forwarded_kwargs.pop("source", None)
        forwarded_kwargs.pop("pretrained", None)

        if local_repo_str and Path(local_repo_str).exists():
            forwarded_kwargs["source"] = "local"
            forwarded_kwargs["pretrained"] = False
            return orig_load(local_repo_str, model, *args, **forwarded_kwargs)

        if allow_online_fallback:
            forwarded_kwargs.setdefault("skip_validation", True)
            return orig_load("facebookresearch/dinov2:main", model, *args, **forwarded_kwargs)

        raise FileNotFoundError(
            "Local DINOv2 hub repo not found and online fallback disabled: "
            f"{local_repo_str if local_repo_str else '<none>'}"
        )

    torch.hub.load = _shim


def _resolve_image_path(scene_root: Path, file_path: str) -> Path:
    rel = Path(file_path)
    if rel.is_absolute():
        if rel.exists():
            return rel
        raise FileNotFoundError(f"Missing image file: {rel}")

    candidates: list[Path] = []
    candidates.append(scene_root / rel)

    parts = rel.parts
    if len(parts) > 0 and parts[0] == "images":
        candidates.append(scene_root / Path("images_4", *parts[1:]))
    if len(parts) > 0 and parts[0] == "images_4":
        candidates.append(scene_root / Path("images", *parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve image path for {file_path} under {scene_root}")


def _ensure_4x4(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {mat.shape}")
    return mat


def _c2w_opengl_to_w2c_opencv(c2w_ogl: np.ndarray) -> np.ndarray:
    c2w_cv = _ensure_4x4(c2w_ogl) @ _DEFAULT_GL_TO_CV
    w2c_cv = np.linalg.inv(c2w_cv).astype(np.float32)
    return w2c_cv


def _get_intrinsic_value(frame: dict, payload: dict, keys: tuple[str, ...]) -> float:
    for source in (frame, payload):
        for key in keys:
            value = source.get(key)
            if value is not None:
                return float(value)
    raise ValueError(f"Missing intrinsic key among {keys}")


def _build_scaled_intrinsics(payload: dict, frame: dict, width: int, height: int) -> np.ndarray:
    fx = _get_intrinsic_value(frame, payload, ("fx", "fl_x"))
    fy = _get_intrinsic_value(frame, payload, ("fy", "fl_y"))
    cx = _get_intrinsic_value(frame, payload, ("cx",))
    cy = _get_intrinsic_value(frame, payload, ("cy",))

    ref_w = float(frame.get("w", payload.get("w", width)))
    ref_h = float(frame.get("h", payload.get("h", height)))
    if ref_w <= 0 or ref_h <= 0:
        raise ValueError(f"Invalid reference image size in transforms: {(ref_w, ref_h)}")

    sx = float(width) / ref_w
    sy = float(height) / ref_h

    k = np.eye(3, dtype=np.float32)
    k[0, 0] = fx * sx
    k[1, 1] = fy * sy
    k[0, 2] = cx * sx
    k[1, 2] = cy * sy
    return k


class DL3DVUndistorter:
    """Undistort DL3DV OPENCV camera images with cached remap grids."""

    def __init__(self, payload: dict):
        self._global = self._extract_camera_params(payload)
        self._by_image: dict[str, dict] = {}
        self._map_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for frame in payload.get("frames", []):
            file_path = frame.get("file_path")
            if not file_path:
                continue
            image_name = Path(file_path).name
            params = dict(self._global)
            params.update(self._extract_camera_params(frame))
            self._by_image[image_name] = params

        self.enabled = self._has_distortion(self._global) or any(
            self._has_distortion(v) for v in self._by_image.values()
        )

    @staticmethod
    def _extract_camera_params(source: dict) -> dict:
        out: dict[str, float] = {}
        for key in (
            "fx",
            "fy",
            "fl_x",
            "fl_y",
            "cx",
            "cy",
            "k1",
            "k2",
            "p1",
            "p2",
            "k3",
            "k4",
            "k5",
            "k6",
            "w",
            "h",
        ):
            value = source.get(key)
            if value is not None:
                out[key] = float(value)
        return out

    @staticmethod
    def _has_distortion(params: dict) -> bool:
        return any(abs(float(params.get(k, 0.0))) > 1e-12 for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"))

    @staticmethod
    def _cache_key(params: dict, width: int, height: int) -> tuple:
        keys = (
            "fx",
            "fy",
            "fl_x",
            "fl_y",
            "cx",
            "cy",
            "k1",
            "k2",
            "p1",
            "p2",
            "k3",
            "k4",
            "k5",
            "k6",
            "w",
            "h",
        )
        vals = [float(params.get(k, 0.0)) for k in keys]
        vals.extend([float(width), float(height)])
        return tuple(round(v, 10) for v in vals)

    def _resolve_params(self, image_name: str) -> dict:
        return self._by_image.get(image_name, self._global)

    def _build_camera(self, params: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        fx = params.get("fx", params.get("fl_x"))
        fy = params.get("fy", params.get("fl_y"))
        cx = params.get("cx")
        cy = params.get("cy")
        if fx is None or fy is None or cx is None or cy is None:
            raise ValueError("Missing fx/fy/cx/cy for undistortion.")

        ref_w = float(params.get("w", width))
        ref_h = float(params.get("h", height))
        if ref_w <= 0 or ref_h <= 0:
            ref_w, ref_h = float(width), float(height)

        sx = float(width) / ref_w
        sy = float(height) / ref_h

        k = np.array(
            [
                [float(fx) * sx, 0.0, float(cx) * sx],
                [0.0, float(fy) * sy, float(cy) * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        k1 = float(params.get("k1", 0.0))
        k2 = float(params.get("k2", 0.0))
        p1 = float(params.get("p1", 0.0))
        p2 = float(params.get("p2", 0.0))
        k3 = float(params.get("k3", 0.0))
        k4 = float(params.get("k4", 0.0))
        k5 = float(params.get("k5", 0.0))
        k6 = float(params.get("k6", 0.0))

        if any(k in params for k in ("k4", "k5", "k6")):
            dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
        elif "k3" in params:
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        else:
            dist = np.array([k1, k2, p1, p2], dtype=np.float64)

        return k, dist

    def undistort(self, rgb: np.ndarray, image_name: str) -> np.ndarray:
        if not self.enabled:
            return rgb

        params = self._resolve_params(image_name)
        if not self._has_distortion(params):
            return rgb

        h, w = rgb.shape[:2]
        key = self._cache_key(params, w, h)
        if key not in self._map_cache:
            k, dist = self._build_camera(params, w, h)
            map_x, map_y = cv2.initUndistortRectifyMap(
                k,
                dist,
                None,
                k,
                (w, h),
                cv2.CV_32FC1,
            )
            self._map_cache[key] = (map_x, map_y)

        map_x, map_y = self._map_cache[key]
        return cv2.remap(
            rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def _make_scene_inputs(
    scene_root: Path,
    payload: dict,
    pose_condition: bool,
    max_frames_per_scene: int,
) -> tuple[list[str], list[str], np.ndarray | None, np.ndarray | None, tuple[int, int], DL3DVUndistorter]:
    frames = payload.get("frames", [])
    if len(frames) == 0:
        raise ValueError(f"No frames in transforms for scene {scene_root}")
    if max_frames_per_scene > 0:
        frames = frames[:max_frames_per_scene]

    image_paths: list[str] = []
    frame_keys: list[str] = []
    for frame in frames:
        image_path = _resolve_image_path(scene_root, frame["file_path"])
        image_paths.append(str(image_path))
        frame_keys.append(Path(image_path).stem)

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed reading image: {image_paths[0]}")
    h, w = first.shape[:2]

    undistorter = DL3DVUndistorter(payload)

    if not pose_condition:
        return image_paths, frame_keys, None, None, (h, w), undistorter

    extrinsics = []
    intrinsics = []
    for frame in frames:
        c2w_ogl = _ensure_4x4(np.asarray(frame["transform_matrix"], dtype=np.float32))
        extrinsics.append(_c2w_opengl_to_w2c_opencv(c2w_ogl))
        intrinsics.append(_build_scaled_intrinsics(payload, frame, w, h))

    return (
        image_paths,
        frame_keys,
        np.stack(extrinsics).astype(np.float32),
        np.stack(intrinsics).astype(np.float32),
        (h, w),
        undistorter,
    )


def _apply_config_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    out = copy.deepcopy(cfg)

    if args.chunk_size is not None:
        out["Model"]["chunk_size"] = int(args.chunk_size)
    if args.overlap is not None:
        out["Model"]["overlap"] = int(args.overlap)
    if args.loop_enable is not None:
        out["Model"]["loop_enable"] = bool(args.loop_enable)
    if args.align_lib is not None:
        out["Model"]["align_lib"] = args.align_lib
    if args.align_method is not None:
        out["Model"]["align_method"] = args.align_method
    if args.scale_compute_method is not None:
        out["Model"]["scale_compute_method"] = args.scale_compute_method
    if args.ref_view_strategy is not None:
        out["Model"]["ref_view_strategy"] = args.ref_view_strategy
    if args.ref_view_strategy_loop is not None:
        out["Model"]["ref_view_strategy_loop"] = args.ref_view_strategy_loop
    if args.delete_temp_files is not None:
        out["Model"]["delete_temp_files"] = bool(args.delete_temp_files)
    if args.sample_ratio is not None:
        out["Model"]["Pointcloud_Save"]["sample_ratio"] = float(args.sample_ratio)
    if args.conf_threshold_coef is not None:
        out["Model"]["Pointcloud_Save"]["conf_threshold_coef"] = float(args.conf_threshold_coef)
    if args.dino_salad_ckpt:
        out["Weights"]["SALAD"] = str(args.dino_salad_ckpt)
    out["Model"]["save_depth_conf_result"] = bool(args.save_depth_conf_result)

    return out


def _build_runtime_options(args: argparse.Namespace, scene_cfg: dict) -> dict:
    return {
        "pose_condition": bool(args.pose_condition),
        "undistort": bool(args.undistort),
        "exclude_depth_above_100_for_points": bool(args.exclude_depth_above_100_for_points),
        "depth_threshold_m": float(args.depth_threshold_m),
        "chunk_size": int(scene_cfg["Model"]["chunk_size"]),
        "overlap": int(scene_cfg["Model"]["overlap"]),
        "scene_output_subdir": args.scene_output_subdir,
        "shared_intrinsics": bool(args.shared_intrinsics),
        "shared_intrinsics_mode": "first_chunk_median" if args.shared_intrinsics else "none",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 01: run DL3DV outdoor DA3-Streaming inference")

    parser.add_argument("--config", default=str(REPO_ROOT / "da3_streaming" / "configs" / "base_config.yaml"))
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument(
        "--da3-model-path",
        default=DEFAULT_DA3_MODEL_PATH,
        help="Local DA3 model directory (contains config.json + model.safetensors).",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--dino-salad-ckpt",
        default=DEFAULT_DINO_SALAD_CKPT,
        help="Local SALAD checkpoint path for loop detector.",
    )
    parser.add_argument(
        "--dinov2-hub-dir",
        default=DEFAULT_DINOV2_HUB_DIR,
        help="Local DINOv2 torch.hub repository directory.",
    )
    _add_bool_flag(
        parser,
        "dinov2-online-fallback",
        True,
        "Allow online fallback for DINOv2 torch.hub loading when local hub dir is unavailable.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--analyze-existing-scene",
        default=None,
        help="Analyze an existing scene output directory and write da3_streaming_output/diagnostics.json without running inference.",
    )

    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--max-frames-per-scene", type=int, default=-1)

    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--align-lib", choices=["triton", "torch", "numba", "numpy"], default=None)
    parser.add_argument("--align-method", choices=["sim3", "se3", "scale+se3"], default=None)
    parser.add_argument("--scale-compute-method", choices=["auto", "ransac", "weighted"], default=None)
    parser.add_argument(
        "--ref-view-strategy",
        choices=["middle", "saddle_balanced", "first", "saddle_sim_range"],
        default=None,
    )
    parser.add_argument(
        "--ref-view-strategy-loop",
        choices=["middle", "saddle_balanced", "first", "saddle_sim_range"],
        default=None,
    )

    _add_bool_flag(parser, "pose-condition", True, "Enable conditioning on input poses/intrinsics")
    _add_bool_flag(
        parser,
        "shared-intrinsics",
        False,
        "Freeze unposed intrinsics to a scene-wide estimate from the first non-loop chunk",
    )
    _add_bool_flag(parser, "undistort", True, "Undistort OPENCV DL3DV frames before inference")
    _add_bool_flag(
        parser,
        "save-sky-mask",
        True,
        "Save per-scene sky mask archive (sky_mask/masks.npz, int8)",
    )
    _add_bool_flag(
        parser,
        "save-depth100-mask",
        True,
        "Save per-scene depth>100 or sky mask archive (depth100_mask/masks.npz)",
    )
    _add_bool_flag(
        parser,
        "exclude-depth-above-100-for-points",
        False,
        "Exclude points with predicted depth > threshold from output point cloud",
    )
    _add_bool_flag(parser, "skip-existing", False, "Skip scene if points_da3.ply already exists")
    _add_bool_flag(parser, "fail-fast", False, "Stop immediately on first scene failure")

    parser.add_argument(
        "--depth-threshold-m",
        type=float,
        default=100.0,
        help="Depth threshold used for depth100 mask and optional point filtering",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"],
    )

    parser.add_argument(
        "--scene-output-subdir",
        default="da3_streaming_output",
        help="Per-scene folder for DA3 streaming intermediates",
    )
    parser.add_argument("--sample-ratio", type=float, default=None)
    parser.add_argument("--conf-threshold-coef", type=float, default=None)
    _add_bool_flag(
        parser,
        "save-depth-conf-result",
        False,
        "Save per-frame depth/conf npz files under da3_streaming_output/results_output",
    )

    parser.add_argument(
        "--loop-enable",
        dest="loop_enable",
        action="store_true",
        help="Force-enable loop closure in streaming config",
    )
    parser.add_argument(
        "--no-loop-enable",
        dest="loop_enable",
        action="store_false",
        help="Disable loop closure in streaming config",
    )
    parser.set_defaults(loop_enable=None)

    parser.add_argument(
        "--delete-temp-files",
        dest="delete_temp_files",
        action="store_true",
        help="Force-enable temp cleanup in streaming config",
    )
    parser.add_argument(
        "--no-delete-temp-files",
        dest="delete_temp_files",
        action="store_false",
        help="Disable temp cleanup in streaming config",
    )
    parser.set_defaults(delete_temp_files=None)

    return parser


def _run_single_scene(
    scene_root: Path,
    scene_cfg: dict,
    model: DepthAnything3,
    args: argparse.Namespace,
) -> dict:
    transforms_path = scene_root / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if scene_cfg["Model"].get("loop_enable", False):
        salad_path = _resolve_existing_weight_path(scene_cfg.get("Weights", {}).get("SALAD"))
        if salad_path is None:
            print(
                "[WARN] Loop closure checkpoint not found "
                f"({scene_cfg.get('Weights', {}).get('SALAD')}); disabling loop closure."
            )
            scene_cfg["Model"]["loop_enable"] = False
        else:
            scene_cfg["Weights"]["SALAD"] = str(salad_path)

    image_paths, frame_keys, extrinsics, intrinsics, _, undistorter = _make_scene_inputs(
        scene_root=scene_root,
        payload=payload,
        pose_condition=args.pose_condition,
        max_frames_per_scene=args.max_frames_per_scene,
    )

    if args.max_frames_per_scene > 0 and scene_cfg["Model"].get("loop_enable", False):
        print("[WARN] max-frames-per-scene is set; disabling loop closure to keep indices consistent.")
        scene_cfg["Model"]["loop_enable"] = False

    save_dir = scene_root / args.scene_output_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    points_out = scene_root / "points_da3.ply"
    sky_mask_dir = scene_root / "sky_mask"
    sky_mask_path = sky_mask_dir / "masks.npz"
    depth100_mask_path = scene_root / "depth100_mask" / "masks.npz"
    pcd_dir = save_dir / "pcd"
    results_output_dir = save_dir / "results_output"
    points_fused_out = scene_root / "points_da3_fused.ply"

    if pcd_dir.exists():
        for ply_file in pcd_dir.glob("*.ply"):
            ply_file.unlink()
    if args.save_sky_mask:
        sky_mask_dir.mkdir(parents=True, exist_ok=True)
        if sky_mask_path.exists():
            sky_mask_path.unlink()
        for mask_file in sky_mask_dir.glob("*.npy"):
            mask_file.unlink()
    if args.save_depth100_mask and depth100_mask_path.exists():
        depth100_mask_path.unlink()
    if args.save_depth_conf_result:
        results_output_dir.mkdir(parents=True, exist_ok=True)
        for npz_file in results_output_dir.glob("*.npz"):
            npz_file.unlink()
    elif results_output_dir.exists():
        shutil.rmtree(results_output_dir, ignore_errors=True)

    def _preprocess_image(image_path: str, _global_idx: int) -> np.ndarray:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if args.undistort:
            image_rgb = undistorter.undistort(image_rgb, Path(image_path).name)
        return image_rgb

    streamer = DA3_Streaming(
        image_dir=str(scene_root / "images_4"),
        save_dir=str(save_dir),
        config=scene_cfg,
        model=model,
        image_paths=image_paths,
        frame_keys=frame_keys,
        image_preprocessor=_preprocess_image,
        pose_condition=args.pose_condition,
        input_extrinsics=extrinsics,
        input_intrinsics=intrinsics,
        shared_intrinsics_mode="first_chunk_median" if args.shared_intrinsics else "none",
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        save_sky_mask=args.save_sky_mask,
        sky_mask_dir=str(sky_mask_dir) if args.save_sky_mask else None,
        save_depth100_mask=args.save_depth100_mask,
        depth100_mask_path=str(depth100_mask_path) if args.save_depth100_mask else None,
        exclude_depth_above_100_for_points=args.exclude_depth_above_100_for_points,
        depth_threshold_m=args.depth_threshold_m,
        runtime_options=_build_runtime_options(args, scene_cfg),
    )

    streamer.run()
    streamer.close()

    merge_ply_files(streamer.pcd_dir, str(points_out))
    diagnostics_path = streamer.write_scene_diagnostics(
        points_path=points_out,
        fused_points_path=points_fused_out if points_fused_out.exists() else None,
    )

    return {
        "scene_root": str(scene_root),
        "num_frames": len(image_paths),
        "points_path": str(points_out),
        "diagnostics_path": str(diagnostics_path),
        "sky_mask_dir": str(sky_mask_dir) if args.save_sky_mask else None,
        "sky_mask_path": str(sky_mask_path) if args.save_sky_mask else None,
        "depth100_mask_path": str(depth100_mask_path) if args.save_depth100_mask else None,
    }


def _analyze_existing_scene(
    scene_root: Path,
    scene_cfg: dict,
    args: argparse.Namespace,
) -> Path:
    transforms_path = scene_root / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    image_paths, frame_keys, _, _, _, _ = _make_scene_inputs(
        scene_root=scene_root,
        payload=payload,
        pose_condition=False,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    chunk_indices, _ = compute_chunk_indices(
        len(image_paths),
        int(scene_cfg["Model"]["chunk_size"]),
        int(scene_cfg["Model"]["overlap"]),
    )
    save_dir = scene_root / args.scene_output_subdir
    diagnostics_path = analyze_existing_scene_outputs(
        output_dir=save_dir,
        runtime_options=_build_runtime_options(args, scene_cfg),
        frame_keys=frame_keys,
        chunk_indices=chunk_indices,
        depth_threshold_m=float(args.depth_threshold_m),
        points_path=scene_root / "points_da3.ply",
        fused_points_path=(scene_root / "points_da3_fused.ply") if (scene_root / "points_da3_fused.ply").exists() else None,
        sky_mask_path=(scene_root / "sky_mask" / "masks.npz") if (scene_root / "sky_mask" / "masks.npz").exists() else None,
    )
    return diagnostics_path


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.analyze_existing_scene and args.pose_condition and args.shared_intrinsics:
        raise ValueError(
            "--shared-intrinsics requires --no-pose-condition in scripts/dl3dv_outdoor_01_streaming_inference.py."
        )

    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    base_cfg = load_config(str(config_path))
    base_cfg = _apply_config_overrides(base_cfg, args)

    if args.analyze_existing_scene:
        scene_root = Path(args.analyze_existing_scene).expanduser().resolve()
        if not scene_root.exists():
            raise FileNotFoundError(f"Scene root not found: {scene_root}")
        diagnostics_path = _analyze_existing_scene(scene_root, copy.deepcopy(base_cfg), args)
        print(f"Analysis complete: {diagnostics_path}")
        return 0

    data_root = Path(args.data_root).expanduser().resolve()
    split_file = Path(args.split_file).expanduser().resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    da3_model_path = _resolve_existing_dir_path(args.da3_model_path)
    dinov2_hub_dir = _resolve_existing_dir_path(args.dinov2_hub_dir)
    if dinov2_hub_dir is None:
        fallback_text = "enabled" if args.dinov2_online_fallback else "disabled"
        print(
            f"[WARN] Local DINOv2 hub dir not found ({args.dinov2_hub_dir}); "
            f"online fallback is {fallback_text}."
        )
    _install_dinov2_torchhub_local_shim(dinov2_hub_dir, bool(args.dinov2_online_fallback))

    scene_relpaths = _read_split_lines(split_file)
    total = len(scene_relpaths)
    start = max(0, int(args.start_idx))
    end = total if args.end_idx is None else min(total, int(args.end_idx))
    if start >= end:
        print(f"No scenes to run for range [{start}, {end}).")
        return 0

    if base_cfg["Model"]["align_lib"] == "numba":
        warmup_numba()

    selected_indices = list(range(start, end))
    pre_skipped = 0
    if args.skip_existing:
        pending_indices: list[int] = []
        for idx in selected_indices:
            scene_rel = scene_relpaths[idx]
            points_out = data_root / scene_rel / "points_da3.ply"
            if points_out.exists():
                pre_skipped += 1
            else:
                pending_indices.append(idx)
        selected_indices = pending_indices
    if len(selected_indices) == 0:
        print(f"All scenes in [{start}, {end}) already have points_da3.ply. Nothing to run.")
        print("\n===== DL3DV DA3-Streaming Summary =====")
        print(f"Total requested: {end - start}")
        print(f"Completed: 0")
        print(f"Skipped: {end - start}")
        print("Failed: 0")
        return 0

    model_source = args.model_id
    if da3_model_path is not None:
        model_source = str(da3_model_path)
    elif args.da3_model_path:
        print(
            f"[WARN] Local DA3 path not found ({args.da3_model_path}); "
            f"falling back to --model-id={args.model_id}."
        )

    print(f"Loading model {model_source} on {args.device} ...")
    model = DepthAnything3.from_pretrained(model_source).to(args.device)
    model.eval()

    print(f"Running scenes [{start}, {end}) out of {total} total scenes")
    if pre_skipped > 0:
        print(f"Pre-skipped existing outputs: {pre_skipped}")

    failures: list[tuple[int, str, str]] = []
    completed = 0
    skipped = pre_skipped
    for idx in selected_indices:
        scene_rel = scene_relpaths[idx]
        scene_root = data_root / scene_rel
        print(f"\n[{idx}] scene={scene_rel}")

        try:
            result = _run_single_scene(scene_root, copy.deepcopy(base_cfg), model, args)
            completed += 1
            print(
                f"Scene done: frames={result['num_frames']} points={result['points_path']} diagnostics={result['diagnostics_path']}"
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failures.append((idx, scene_rel, msg))
            print(f"[ERROR] {scene_rel}: {msg}")
            traceback.print_exc()
            if args.fail_fast:
                break

    print("\n===== DL3DV DA3-Streaming Summary =====")
    print(f"Total requested: {end - start}")
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failures)}")
    if failures:
        for idx, scene_rel, msg in failures[:20]:
            print(f"  - [{idx}] {scene_rel}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
