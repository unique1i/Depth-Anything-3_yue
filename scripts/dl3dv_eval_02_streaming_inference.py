#!/usr/bin/env python3
"""Step 02: run DL3DV evaluation-zip DA3-Streaming inference."""

from __future__ import annotations

import argparse
import copy
import io
import json
import re
import shutil
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image


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
    write_scene_diagnostics_report,
)
from loop_utils.config_utils import load_config  # noqa: E402
from loop_utils.sim3utils import merge_ply_files, warmup_numba  # noqa: E402


DEFAULT_ZIP_ROOT = "/home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images"
DEFAULT_OUTPUT_ROOT = "/home/yli7/scratch2/datasets/dl3dv_960p/evaluation"
DEFAULT_SPLIT_FILE = "/home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt"
DEFAULT_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

# local ckpt paths
DEFAULT_DA3_MODEL_PATH = "/home/yli7/scratch2/datasets/ckpts/depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_DINO_SALAD_CKPT = "/home/yli7/repos/Depth-Anything-3/da3_streaming/weights/dino_salad.ckpt"
DEFAULT_DINOV2_HUB_DIR = "/home/yli7/.cache/torch/hub/facebookresearch_dinov2_main"

_FRAME_RE = re.compile(r"frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
_NS4_IMAGE_MARKER = "/nerfstudio/images_4/"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})


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
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
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
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir():
            return resolved
    return None


def _read_split_lines(path: Path) -> list[str]:
    scenes: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(".zip"):
                line = line[:-4]
            scenes.append(Path(line.lstrip("/")).name)
    return scenes


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


def _build_runtime_options(
    args: argparse.Namespace,
    scene_cfg: dict,
    zip_path: Path | None = None,
    source_frame_count: int | None = None,
    selected_image_size: tuple[int, int] | None = None,
) -> dict:
    return {
        "pose_condition": False,
        "undistort": False,
        "exclude_depth_above_100_for_points": bool(args.exclude_depth_above_100_for_points),
        "depth_threshold_m": float(args.depth_threshold_m),
        "chunk_size": int(scene_cfg["Model"]["chunk_size"]),
        "overlap": int(scene_cfg["Model"]["overlap"]),
        "scene_output_subdir": args.scene_output_subdir,
        "loop_enable": bool(scene_cfg["Model"].get("loop_enable", False)),
        "process_res": int(args.process_res),
        "process_res_method": args.process_res_method,
        "save_depth_conf_result": bool(args.save_depth_conf_result),
        "save_sky_mask": bool(args.save_sky_mask),
        "save_depth100_mask": bool(args.save_depth100_mask),
        "shared_intrinsics": bool(args.shared_intrinsics),
        "shared_intrinsics_mode": "first_chunk_median" if args.shared_intrinsics else "none",
        "source_image_tree": "nerfstudio/images_4",
        "zip_path": str(zip_path) if zip_path is not None else None,
        "source_frame_count": int(source_frame_count) if source_frame_count is not None else None,
        "selected_image_size": (
            {
                "width": int(selected_image_size[0]),
                "height": int(selected_image_size[1]),
            }
            if selected_image_size is not None
            else None
        ),
    }


def _resolve_zip_paths_from_split(zip_root: Path, split_file: Path) -> list[Path]:
    zip_paths: list[Path] = []
    missing: list[str] = []
    seen: set[str] = set()
    for scene_id in _read_split_lines(split_file):
        if scene_id in seen:
            raise ValueError(f"Duplicate scene in split file {split_file}: {scene_id}")
        seen.add(scene_id)
        zip_path = zip_root / f"{scene_id}.zip"
        if not zip_path.exists():
            missing.append(str(zip_path))
            continue
        zip_paths.append(zip_path)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else " ..."
        raise FileNotFoundError(
            f"Missing {len(missing)} zip scenes referenced by {split_file}: {preview}{suffix}"
        )
    return zip_paths


def _scene_manifest_path(scene_root: Path) -> Path:
    return scene_root / "scene_manifest.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_scene_manifest(scene_root: Path) -> dict:
    manifest_path = _scene_manifest_path(scene_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing scene manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _chunk_indices_to_lists(chunk_indices: list[tuple[int, int]]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in chunk_indices]


def _lists_to_chunk_indices(chunk_indices: list[list[int]] | None) -> list[tuple[int, int]]:
    if not chunk_indices:
        return []
    return [(int(item[0]), int(item[1])) for item in chunk_indices]


def _append_unique_note(notes: list[str], note: str) -> None:
    if note not in notes:
        notes.append(note)


def _augment_diagnostics_with_manifest(diagnostics_path: Path, manifest: dict, scene_root: Path) -> None:
    with diagnostics_path.open("r", encoding="utf-8") as f:
        diagnostics = json.load(f)

    diagnostics["scene_status"] = manifest.get("status")
    diagnostics["scene_manifest_path"] = str(_scene_manifest_path(scene_root))
    diagnostics["zip_source"] = {
        "zip_path": manifest.get("zip_path"),
        "source_image_tree": manifest.get("source_image_tree"),
        "source_frame_count": manifest.get("source_frame_count"),
        "selected_image_size": manifest.get("selected_image_size"),
    }
    diagnostics["selected_frame_count"] = manifest.get("selected_frame_count")
    diagnostics["frame_keys"] = manifest.get("frame_keys")
    if manifest.get("skip_reason"):
        diagnostics["skip_reason"] = manifest["skip_reason"]

    notes = list(diagnostics.get("notes", []))
    _append_unique_note(
        notes,
        "Scene metadata sourced from scene_manifest.json for evaluation-zip analysis.",
    )
    if manifest.get("status") and manifest.get("status") != "completed":
        _append_unique_note(
            notes,
            f"Scene did not complete inference: {manifest.get('status')}.",
        )
    diagnostics["notes"] = notes

    _write_json(diagnostics_path, diagnostics)


def _write_scene_manifest(scene_root: Path, payload: dict) -> Path:
    manifest = dict(payload)
    manifest["updated_at"] = _timestamp()
    manifest_path = _scene_manifest_path(scene_root)
    _write_json(manifest_path, manifest)
    return manifest_path


class EvalZipSceneArchive:
    def __init__(self, zip_path: Path):
        self.zip_path = zip_path
        self._zip_file: zipfile.ZipFile | None = None
        self.image_entries: list[str] = []

    def __enter__(self) -> "EvalZipSceneArchive":
        self._zip_file = zipfile.ZipFile(self.zip_path)
        self.image_entries = self._collect_entries(_NS4_IMAGE_MARKER)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None

    def _require_zip_file(self) -> zipfile.ZipFile:
        if self._zip_file is None:
            raise RuntimeError(f"Zip archive is not open: {self.zip_path}")
        return self._zip_file

    def _collect_entries(self, marker: str) -> list[str]:
        entries: list[tuple[int, str]] = []
        for name in self._require_zip_file().namelist():
            if marker not in name:
                continue
            match = _FRAME_RE.search(name)
            if match is None:
                continue
            entries.append((int(match.group(1)), name))
        entries.sort(key=lambda item: item[0])
        return [name for _, name in entries]

    def load_rgb(self, image_entry: str) -> np.ndarray:
        raw = self._require_zip_file().read(image_entry)
        image_np = np.frombuffer(raw, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed decoding image from archive: {image_entry}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def load_pil(self, image_entry: str) -> Image.Image:
        raw = self._require_zip_file().read(image_entry)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def validate_image_entries(self, image_entries: list[str]) -> tuple[int, int]:
        expected_size: tuple[int, int] | None = None
        for image_entry in image_entries:
            raw = self._require_zip_file().read(image_entry)
            if len(raw) == 0:
                raise ValueError(f"Zero-byte image entry: {image_entry}")
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    size = (int(image.size[0]), int(image.size[1]))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Unreadable image entry {image_entry}: {exc}") from exc
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"Invalid image size for {image_entry}: {size}")
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError(
                    "Inconsistent nerfstudio/images_4 size in "
                    f"{self.zip_path}: expected {expected_size[0]}x{expected_size[1]}, "
                    f"got {size[0]}x{size[1]} for {image_entry}"
                )
        if expected_size is None:
            raise ValueError(f"No nerfstudio/images_4 frames found in {self.zip_path}")
        return expected_size


def _build_scene_manifest(
    scene_root: Path,
    zip_path: Path,
    scene_cfg: dict,
    args: argparse.Namespace,
    source_frame_count: int,
    image_entries: list[str],
    frame_keys: list[str],
    selected_image_size: tuple[int, int] | None,
    status: str,
    skip_reason: str | None = None,
    error: str | None = None,
) -> dict:
    chunk_indices: list[tuple[int, int]] = []
    if image_entries:
        chunk_indices, _ = compute_chunk_indices(
            len(image_entries),
            int(scene_cfg["Model"]["chunk_size"]),
            int(scene_cfg["Model"]["overlap"]),
        )

    runtime_options = _build_runtime_options(
        args=args,
        scene_cfg=scene_cfg,
        zip_path=zip_path,
        source_frame_count=source_frame_count,
        selected_image_size=selected_image_size,
    )

    return {
        "scene_id": zip_path.stem,
        "scene_root": str(scene_root),
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "source_image_tree": "nerfstudio/images_4",
        "status": status,
        "skip_reason": skip_reason,
        "error": error,
        "source_frame_count": int(source_frame_count),
        "selected_frame_count": len(image_entries),
        "selected_image_size": (
            {
                "width": int(selected_image_size[0]),
                "height": int(selected_image_size[1]),
            }
            if selected_image_size is not None
            else None
        ),
        "frame_keys": list(frame_keys),
        "image_entries": list(image_entries),
        "chunk_indices": _chunk_indices_to_lists(chunk_indices),
        "scene_output_subdir": args.scene_output_subdir,
        "runtime_options": runtime_options,
        "created_at": _timestamp(),
    }


def _clear_previous_scene_artifacts(scene_root: Path, args: argparse.Namespace) -> None:
    artifact_dirs = [
        scene_root / args.scene_output_subdir,
        scene_root / "sky_mask",
        scene_root / "depth100_mask",
    ]
    artifact_files = [
        scene_root / "points_da3.ply",
        scene_root / "points_da3_fused.ply",
        scene_root / "transforms_da3.json",
        _scene_manifest_path(scene_root),
    ]

    for path in artifact_dirs:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for path in artifact_files:
        if path.exists():
            path.unlink()


def _prepare_scene_outputs(scene_root: Path, args: argparse.Namespace) -> None:
    save_dir = scene_root / args.scene_output_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    sky_mask_dir = scene_root / "sky_mask"
    sky_mask_path = sky_mask_dir / "masks.npz"
    depth100_mask_path = scene_root / "depth100_mask" / "masks.npz"
    pcd_dir = save_dir / "pcd"
    results_output_dir = save_dir / "results_output"

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


def _write_noncompleted_scene_diagnostics(scene_root: Path, manifest: dict) -> Path:
    save_dir = scene_root / manifest.get("scene_output_subdir", "da3_streaming_output")
    save_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = write_scene_diagnostics_report(
        output_dir=save_dir,
        runtime_options=manifest.get("runtime_options", {}),
        chunk_indices=_lists_to_chunk_indices(manifest.get("chunk_indices")),
        frame_keys=manifest.get("frame_keys", []),
        chunk_diagnostics=[],
        saved_frame_diagnostics=[],
        sequential_sim3=[],
        cumulative_sim3=[],
        points_path=scene_root / "points_da3.ply",
        fused_points_path=(scene_root / "points_da3_fused.ply") if (scene_root / "points_da3_fused.ply").exists() else None,
    )
    _augment_diagnostics_with_manifest(diagnostics_path, manifest, scene_root)
    return diagnostics_path


def _analyze_existing_scene(scene_root: Path) -> Path:
    manifest = _load_scene_manifest(scene_root)
    if manifest.get("status") != "completed":
        return _write_noncompleted_scene_diagnostics(scene_root, manifest)

    frame_keys = manifest.get("frame_keys")
    if frame_keys is None:
        raise ValueError(f"Manifest missing frame_keys: {_scene_manifest_path(scene_root)}")

    chunk_indices = _lists_to_chunk_indices(manifest.get("chunk_indices"))
    save_dir = scene_root / manifest.get("scene_output_subdir", "da3_streaming_output")
    runtime_options = manifest.get("runtime_options", {})
    depth_threshold_m = float(runtime_options.get("depth_threshold_m", 100.0))

    diagnostics_path = analyze_existing_scene_outputs(
        output_dir=save_dir,
        runtime_options=runtime_options,
        frame_keys=frame_keys,
        chunk_indices=chunk_indices,
        depth_threshold_m=depth_threshold_m,
        points_path=scene_root / "points_da3.ply",
        fused_points_path=(scene_root / "points_da3_fused.ply") if (scene_root / "points_da3_fused.ply").exists() else None,
        sky_mask_path=(scene_root / "sky_mask" / "masks.npz") if (scene_root / "sky_mask" / "masks.npz").exists() else None,
    )
    _augment_diagnostics_with_manifest(diagnostics_path, manifest, scene_root)
    return diagnostics_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 02: run DL3DV evaluation-zip DA3-Streaming inference")

    parser.add_argument("--config", default=str(REPO_ROOT / "da3_streaming" / "configs" / "base_config.yaml"))
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--split-file",
        default=DEFAULT_SPLIT_FILE,
        help="Text file listing evaluation scene IDs to run in order.",
    )
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
        help="Analyze an existing synthetic scene output directory without running inference.",
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

    _add_bool_flag(
        parser,
        "save-sky-mask",
        True,
        "Save per-scene sky mask archive (sky_mask/masks.npz, int8).",
    )
    _add_bool_flag(
        parser,
        "save-depth100-mask",
        True,
        "Save per-scene depth>100 or sky mask archive (depth100_mask/masks.npz).",
    )
    _add_bool_flag(
        parser,
        "shared-intrinsics",
        False,
        "Freeze unposed intrinsics to a scene-wide estimate from the first non-loop chunk.",
    )
    _add_bool_flag(
        parser,
        "exclude-depth-above-100-for-points",
        False,
        "Exclude points with predicted depth > threshold from output point cloud.",
    )
    _add_bool_flag(
        parser,
        "skip-existing",
        False,
        "Skip scenes that already have a completed or skipped manifest.",
    )
    _add_bool_flag(parser, "fail-fast", False, "Stop immediately on first scene failure.")

    parser.add_argument(
        "--depth-threshold-m",
        type=float,
        default=100.0,
        help="Depth threshold used for depth100 mask and optional point filtering.",
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
        help="Per-scene folder for DA3 streaming intermediates.",
    )
    parser.add_argument("--sample-ratio", type=float, default=None)
    parser.add_argument("--conf-threshold-coef", type=float, default=None)
    _add_bool_flag(
        parser,
        "save-depth-conf-result",
        False,
        "Save per-frame depth/conf npz files under da3_streaming_output/results_output.",
    )

    parser.add_argument(
        "--loop-enable",
        dest="loop_enable",
        action="store_true",
        help="Force-enable loop closure in streaming config.",
    )
    parser.add_argument(
        "--no-loop-enable",
        dest="loop_enable",
        action="store_false",
        help="Disable loop closure in streaming config.",
    )
    parser.set_defaults(loop_enable=None)

    parser.add_argument(
        "--delete-temp-files",
        dest="delete_temp_files",
        action="store_true",
        help="Force-enable temp cleanup in streaming config.",
    )
    parser.add_argument(
        "--no-delete-temp-files",
        dest="delete_temp_files",
        action="store_false",
        help="Disable temp cleanup in streaming config.",
    )
    parser.set_defaults(delete_temp_files=None)

    return parser


def _select_model_source(args: argparse.Namespace) -> str:
    da3_model_path = _resolve_existing_dir_path(args.da3_model_path)
    if da3_model_path is not None:
        return str(da3_model_path)
    if args.da3_model_path:
        print(
            f"[WARN] Local DA3 path not found ({args.da3_model_path}); "
            f"falling back to --model-id={args.model_id}."
        )
    return args.model_id


def _maybe_skip_existing(scene_root: Path) -> bool:
    manifest_path = _scene_manifest_path(scene_root)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        status = manifest.get("status")
        if status == "completed" or (isinstance(status, str) and status.startswith("skipped_")):
            return True
        if status in {"failed", "running", "pending"}:
            return False
    return (scene_root / "points_da3.ply").exists()


def _run_single_scene(
    zip_path: Path,
    scene_root: Path,
    scene_cfg: dict,
    model: DepthAnything3,
    args: argparse.Namespace,
) -> dict:
    streamer = None
    with EvalZipSceneArchive(zip_path) as archive:
        selected_entries = list(archive.image_entries)
        source_frame_count = len(selected_entries)
        if args.max_frames_per_scene > 0:
            selected_entries = selected_entries[: args.max_frames_per_scene]
        frame_keys = [Path(entry).stem for entry in selected_entries]

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

        if args.max_frames_per_scene > 0 and scene_cfg["Model"].get("loop_enable", False):
            print("[WARN] max-frames-per-scene is set; disabling loop closure to keep indices consistent.")
            scene_cfg["Model"]["loop_enable"] = False

        selected_image_size: tuple[int, int] | None = None
        status = "pending"
        skip_reason = None
        if len(selected_entries) == 0:
            status = "skipped_missing_ns4"
            skip_reason = f"No nerfstudio/images_4 frames found in {zip_path}"
        else:
            try:
                selected_image_size = archive.validate_image_entries(selected_entries)
            except Exception as exc:  # noqa: BLE001
                status = "skipped_invalid_ns4"
                skip_reason = f"Invalid nerfstudio/images_4 inputs: {exc}"

        manifest = _build_scene_manifest(
            scene_root=scene_root,
            zip_path=zip_path,
            scene_cfg=scene_cfg,
            args=args,
            source_frame_count=source_frame_count,
            image_entries=selected_entries,
            frame_keys=frame_keys,
            selected_image_size=selected_image_size,
            status=status,
            skip_reason=skip_reason,
        )
        manifest["scene_output_subdir"] = args.scene_output_subdir

        scene_root.mkdir(parents=True, exist_ok=True)
        _clear_previous_scene_artifacts(scene_root, args)
        scene_root.mkdir(parents=True, exist_ok=True)

        if manifest["status"] != "pending":
            _write_scene_manifest(scene_root, manifest)
            diagnostics_path = _write_noncompleted_scene_diagnostics(scene_root, manifest)
            manifest["diagnostics_path"] = str(diagnostics_path)
            _write_scene_manifest(scene_root, manifest)
            return {
                "status": manifest["status"],
                "skip_reason": manifest["skip_reason"],
                "diagnostics_path": str(diagnostics_path),
                "num_frames": len(selected_entries),
            }

        manifest["status"] = "running"
        _write_scene_manifest(scene_root, manifest)

        _prepare_scene_outputs(scene_root, args)

        save_dir = scene_root / args.scene_output_subdir
        points_out = scene_root / "points_da3.ply"
        sky_mask_dir = scene_root / "sky_mask"
        sky_mask_path = sky_mask_dir / "masks.npz"
        depth100_mask_path = scene_root / "depth100_mask" / "masks.npz"
        points_fused_out = scene_root / "points_da3_fused.ply"

        def _preprocess_image(image_path: str, _global_idx: int) -> np.ndarray:
            return archive.load_rgb(image_path)

        def _load_loop_image(image_path: str) -> Image.Image:
            return archive.load_pil(image_path)

        streamer = DA3_Streaming(
            image_dir=str(zip_path),
            save_dir=str(save_dir),
            config=scene_cfg,
            model=model,
            image_paths=selected_entries,
            loop_image_paths=selected_entries,
            frame_keys=frame_keys,
            image_preprocessor=_preprocess_image,
            loop_image_loader=_load_loop_image,
            pose_condition=False,
            input_extrinsics=None,
            input_intrinsics=None,
            shared_intrinsics_mode="first_chunk_median" if args.shared_intrinsics else "none",
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            save_sky_mask=args.save_sky_mask,
            sky_mask_dir=str(sky_mask_dir) if args.save_sky_mask else None,
            save_depth100_mask=args.save_depth100_mask,
            depth100_mask_path=str(depth100_mask_path) if args.save_depth100_mask else None,
            exclude_depth_above_100_for_points=args.exclude_depth_above_100_for_points,
            depth_threshold_m=args.depth_threshold_m,
            runtime_options=manifest["runtime_options"],
        )

        try:
            streamer.run()
            merge_ply_files(streamer.pcd_dir, str(points_out))
            diagnostics_path = streamer.write_scene_diagnostics(
                points_path=points_out,
                fused_points_path=points_fused_out if points_fused_out.exists() else None,
            )
        finally:
            if streamer is not None:
                streamer.close()

        manifest["status"] = "completed"
        manifest["runtime_options"] = dict(streamer.runtime_options)
        manifest["diagnostics_path"] = str(diagnostics_path)
        manifest["points_path"] = str(points_out)
        if args.save_sky_mask:
            manifest["sky_mask_path"] = str(sky_mask_path)
        if args.save_depth100_mask:
            manifest["depth100_mask_path"] = str(depth100_mask_path)
        manifest["results_output_dir"] = str(save_dir / "results_output")
        _augment_diagnostics_with_manifest(Path(diagnostics_path), manifest, scene_root)
        _write_scene_manifest(scene_root, manifest)

        return {
            "status": manifest["status"],
            "num_frames": len(selected_entries),
            "points_path": str(points_out),
            "diagnostics_path": str(diagnostics_path),
        }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.max_frames_per_scene == 0 or args.max_frames_per_scene < -1:
        raise ValueError("--max-frames-per-scene must be -1 or a positive integer")

    config_path = Path(args.config).expanduser().resolve()
    split_file = Path(args.split_file).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    base_cfg = load_config(str(config_path))
    base_cfg = _apply_config_overrides(base_cfg, args)

    if args.analyze_existing_scene:
        scene_root = Path(args.analyze_existing_scene).expanduser().resolve()
        if not scene_root.exists():
            raise FileNotFoundError(f"Scene root not found: {scene_root}")
        diagnostics_path = _analyze_existing_scene(scene_root)
        print(f"Analysis complete: {diagnostics_path}")
        return 0

    zip_root = Path(args.zip_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not zip_root.exists():
        raise FileNotFoundError(f"Zip root not found: {zip_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    dinov2_hub_dir = _resolve_existing_dir_path(args.dinov2_hub_dir)
    if dinov2_hub_dir is None:
        fallback_text = "enabled" if args.dinov2_online_fallback else "disabled"
        print(
            f"[WARN] Local DINOv2 hub dir not found ({args.dinov2_hub_dir}); "
            f"online fallback is {fallback_text}."
        )
    _install_dinov2_torchhub_local_shim(dinov2_hub_dir, bool(args.dinov2_online_fallback))

    zip_paths = _resolve_zip_paths_from_split(zip_root, split_file)
    total = len(zip_paths)
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
            scene_root = output_root / zip_paths[idx].stem
            if _maybe_skip_existing(scene_root):
                pre_skipped += 1
            else:
                pending_indices.append(idx)
        selected_indices = pending_indices

    if len(selected_indices) == 0:
        print(f"All scenes in [{start}, {end}) already have completed or skipped manifests. Nothing to run.")
        print("\n===== DL3DV Eval Zip DA3-Streaming Summary =====")
        print(f"Total requested: {end - start}")
        print("Completed: 0")
        print(f"Skipped: {end - start}")
        print("Failed: 0")
        return 0

    model_source = _select_model_source(args)
    model: DepthAnything3 | None = None

    print(f"Running scenes [{start}, {end}) out of {total} split scenes")
    if pre_skipped > 0:
        print(f"Pre-skipped existing outputs: {pre_skipped}")

    failures: list[tuple[int, str, str]] = []
    completed = 0
    skipped = pre_skipped
    for idx in selected_indices:
        zip_path = zip_paths[idx]
        scene_root = output_root / zip_path.stem
        print(f"\n[{idx}] zip={zip_path.name}")

        try:
            if model is None:
                print(f"Loading model {model_source} on {args.device} ...")
                model = DepthAnything3.from_pretrained(model_source).to(args.device)
                model.eval()

            result = _run_single_scene(
                zip_path=zip_path,
                scene_root=scene_root,
                scene_cfg=copy.deepcopy(base_cfg),
                model=model,
                args=args,
            )

            if result["status"] == "completed":
                completed += 1
                print(
                    "Scene done: "
                    f"frames={result['num_frames']} "
                    f"points={result['points_path']} "
                    f"diagnostics={result['diagnostics_path']}"
                )
            else:
                skipped += 1
                print(f"Scene skipped: {result['skip_reason']}")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failures.append((idx, zip_path.name, msg))
            print(f"[ERROR] {zip_path.name}: {msg}")
            traceback.print_exc()

            scene_root.mkdir(parents=True, exist_ok=True)
            if _scene_manifest_path(scene_root).exists():
                manifest = _load_scene_manifest(scene_root)
            else:
                manifest = {
                    "scene_id": zip_path.stem,
                    "scene_root": str(scene_root),
                    "zip_path": str(zip_path),
                    "zip_name": zip_path.name,
                    "source_image_tree": "nerfstudio/images_4",
                    "scene_output_subdir": args.scene_output_subdir,
                    "runtime_options": {},
                    "created_at": _timestamp(),
                }
            manifest["status"] = "failed"
            manifest["error"] = msg
            _write_scene_manifest(scene_root, manifest)

            if args.fail_fast:
                break

    print("\n===== DL3DV Eval Zip DA3-Streaming Summary =====")
    print(f"Total requested: {end - start}")
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failures)}")
    if failures:
        for idx, zip_name, msg in failures[:20]:
            print(f"  - [{idx}] {zip_name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
