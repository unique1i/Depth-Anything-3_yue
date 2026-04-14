#!/usr/bin/env python3
"""Generate transforms_da3.json for completed DL3DV evaluation scenes.

This script copies DA3 camera poses directly from `camera_poses.txt`.
It does not support any SIM(3) alignment against an existing transforms JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DA3_STREAMING_ROOT = REPO_ROOT / "da3_streaming"
for p in (str(SRC_ROOT), str(DA3_STREAMING_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


DEFAULT_SPLIT_FILE = "/home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt"
DEFAULT_OUTPUT_ROOT = "/home/yli7/scratch/datasets/dl3dv_960p/evaluation/da3_streaming"
DEFAULT_ZIP_ROOT = "/home/yli7/scratch/datasets/dl3dv_960p/evaluation/images"
DEFAULT_SCENE_OUTPUT_SUBDIR = "da3_streaming_output"
EXPECTED_SOURCE_IMAGE_TREE = "nerfstudio/images_4"
FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_APPLIED_TRANSFORM = [
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [-0.0, -0.0, -1.0, -0.0],
]


def _read_split_lines(path: Path) -> list[str]:
    scenes: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(".zip"):
                line = line[:-4]
            scenes.append(line.lstrip("/"))
    return scenes


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def _trimmed_mean_p5_p95(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot compute trimmed mean from an empty array.")
    p5 = float(np.quantile(arr, 0.05))
    p95 = float(np.quantile(arr, 0.95))
    kept = arr[(arr >= p5) & (arr <= p95)]
    if kept.size == 0:
        kept = arr
    return float(np.mean(kept))


def _load_intrinsics_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected Nx4 intrinsics in {path}, got {values.shape}")
    return values[:, 0], values[:, 1], values[:, 2], values[:, 3]


def _load_camera_poses_txt(path: Path) -> np.ndarray:
    poses: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            vals = [float(x) for x in FLOAT_PATTERN.findall(line)]
            if len(vals) != 16:
                raise ValueError(f"pose line {idx} in {path} has {len(vals)} values, expected 16")
            poses.append(vals)
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"Expected Nx16 camera poses in {path}, got {values.shape}")
    return values.reshape(-1, 4, 4)


def _obb_points_from_points_da3(path: Path) -> list[list[float]]:
    try:
        import open3d as o3d
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"open3d is required to compute bbox_obb_points from {path}: {exc}") from exc

    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        raise ValueError(f"No points found in {path}")
    obb = point_cloud.get_oriented_bounding_box()
    vertices = np.asarray(obb.get_box_points(), dtype=np.float64)
    if vertices.shape != (8, 3):
        raise ValueError(f"Expected 8x3 OBB vertices from {path}, got {vertices.shape}")
    return vertices.tolist()


def _zip_image_size(zip_path: Path, image_entry: str) -> tuple[int, int]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(image_entry) as f:
            with Image.open(f) as image:
                width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size from {zip_path}:{image_entry} -> {(width, height)}")
    return int(width), int(height)


def _processed_size_from_intrinsics(cx_vals: np.ndarray, cy_vals: np.ndarray) -> tuple[int, int, float, float]:
    cx = float(np.median(np.asarray(cx_vals, dtype=np.float64).reshape(-1)))
    cy = float(np.median(np.asarray(cy_vals, dtype=np.float64).reshape(-1)))
    proc_w = int(round(2.0 * cx))
    proc_h = int(round(2.0 * cy))
    if proc_w <= 0 or proc_h <= 0:
        raise ValueError(
            "Failed to infer processed image size from intrinsics: "
            f"cx={cx}, cy={cy}, inferred={(proc_w, proc_h)}"
        )
    return proc_w, proc_h, cx, cy


def _scene_id_from_split_entry(entry: str) -> str:
    return Path(entry).name


def _manifest_image_size(manifest_path: Path, manifest: dict) -> tuple[int, int]:
    size = manifest.get("selected_image_size")
    if not isinstance(size, dict):
        raise ValueError(f"Manifest missing selected_image_size: {manifest_path}")
    width = int(size.get("width", 0))
    height = int(size.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid selected_image_size in {manifest_path}: {size}")
    if (width, height) != (960, 540):
        raise ValueError(
            f"Expected eval selected_image_size=(960, 540) in {manifest_path}, got {(width, height)}"
        )
    return width, height


def _build_transforms_da3_payload(
    scene_root: Path,
    zip_root: Path,
    scene_output_subdir: str,
) -> dict:
    manifest_path = scene_root / "scene_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing scene manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    source_image_tree = manifest.get("source_image_tree")
    if source_image_tree != EXPECTED_SOURCE_IMAGE_TREE:
        raise ValueError(
            f"Scene {scene_root.name} uses source_image_tree={source_image_tree!r}; "
            f"expected {EXPECTED_SOURCE_IMAGE_TREE!r}"
        )

    status = manifest.get("status")
    if status != "completed":
        raise ValueError(
            f"Scene {scene_root.name} is not completed (status={status!r}); "
            "transforms_da3.json is only written for completed scenes."
        )

    image_entries = list(manifest.get("image_entries") or [])
    if not image_entries:
        raise ValueError(f"Manifest has no image_entries: {manifest_path}")
    invalid_entries = [entry for entry in image_entries if f"/{EXPECTED_SOURCE_IMAGE_TREE}/" not in entry]
    if invalid_entries:
        raise ValueError(
            f"Found {len(invalid_entries)} image entries outside {EXPECTED_SOURCE_IMAGE_TREE} in {manifest_path}: "
            f"{invalid_entries[:3]}"
        )

    frame_keys = list(manifest.get("frame_keys") or [])
    if frame_keys and len(frame_keys) != len(image_entries):
        raise ValueError(
            f"frame_keys/image_entries mismatch in {manifest_path}: "
            f"{len(frame_keys)} vs {len(image_entries)}"
        )

    zip_path_str = manifest.get("zip_path")
    zip_path = Path(zip_path_str).expanduser().resolve() if zip_path_str else (zip_root / f"{scene_root.name}.zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip source not found for {scene_root.name}: {zip_path}")

    output_dir = scene_root / scene_output_subdir
    poses_path = output_dir / "camera_poses.txt"
    intrinsics_path = output_dir / "intrinsic.txt"
    points_path = scene_root / "points_da3.ply"
    for required_path in (poses_path, intrinsics_path, points_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required eval artifact missing for {scene_root.name}: {required_path}")

    poses = _load_camera_poses_txt(poses_path)
    fx_vals, fy_vals, cx_vals, cy_vals = _load_intrinsics_txt(intrinsics_path)
    if len(poses) != len(image_entries):
        raise ValueError(
            f"camera_poses.txt length mismatch for {scene_root.name}: "
            f"{len(poses)} poses vs {len(image_entries)} frames"
        )
    if len(fx_vals) != len(image_entries):
        raise ValueError(
            f"intrinsic.txt length mismatch for {scene_root.name}: "
            f"{len(fx_vals)} rows vs {len(image_entries)} frames"
        )

    orig_w, orig_h = _manifest_image_size(manifest_path, manifest)
    first_w, first_h = _zip_image_size(zip_path, image_entries[0])
    if (first_w, first_h) != (orig_w, orig_h):
        raise ValueError(
            f"Manifest/zip image-size mismatch for {scene_root.name}: "
            f"manifest={(orig_w, orig_h)} zip={(first_w, first_h)}"
        )
    proc_w, proc_h, proc_cx, proc_cy = _processed_size_from_intrinsics(cx_vals, cy_vals)
    sx = float(orig_w) / float(proc_w)
    sy = float(orig_h) / float(proc_h)

    fl_x = _trimmed_mean_p5_p95(fx_vals) * sx
    fl_y = _trimmed_mean_p5_p95(fy_vals) * sy
    cx = proc_cx * sx
    cy = proc_cy * sy
    bbox_obb_points = _obb_points_from_points_da3(points_path)

    frames = []
    for idx, (image_entry, pose) in enumerate(zip(image_entries, poses, strict=True), start=1):
        local_parts = Path(image_entry).parts
        try:
            subtree_start = local_parts.index("nerfstudio")
        except ValueError as exc:
            raise ValueError(
                f"Expected eval image entry under nerfstudio/images_4 for {scene_root.name}, got {image_entry}"
            ) from exc
        local_image_entry = Path(*local_parts[subtree_start:]).as_posix()
        frames.append(
            {
                "file_path": local_image_entry,
                "transform_matrix": np.asarray(pose, dtype=np.float64).tolist(),
                "colmap_im_id": idx,
            }
        )

    payload = {
        "w": int(orig_w),
        "h": int(orig_h),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_model": "OPENCV",
        "frames": frames,
        "applied_transform": _APPLIED_TRANSFORM,
        "bbox_obb_points": bbox_obb_points,
        "zip_path": str(zip_path),
        "source_image_tree": source_image_tree,
    }
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write transforms_da3.json for completed DL3DV evaluation scenes by directly "
            "copying DA3 camera_poses.txt. No --align_to_da3_pose / SIM(3) alignment mode is supported."
        )
    )
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--scene-output-subdir", default=DEFAULT_SCENE_OUTPUT_SUBDIR)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    split_file = Path(args.split_file).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    zip_root = Path(args.zip_root).expanduser().resolve()
    scene_output_subdir = str(args.scene_output_subdir)

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")
    if not zip_root.exists():
        raise FileNotFoundError(f"Zip root not found: {zip_root}")

    scene_entries = _read_split_lines(split_file)
    total = len(scene_entries)
    start = max(0, int(args.start_idx))
    end = total if args.end_idx is None else min(total, int(args.end_idx))
    if start >= end:
        print(f"No scenes to process for range [{start}, {end}).")
        return 0

    completed = 0
    skipped = 0
    failed: list[tuple[int, str, str]] = []

    print(f"Writing transforms_da3.json for scenes [{start}, {end}) out of {total}")
    for idx in range(start, end):
        scene_entry = scene_entries[idx]
        scene_id = _scene_id_from_split_entry(scene_entry)
        scene_root = output_root / scene_id
        print(f"\n[{idx}] scene={scene_id}")

        manifest_path = scene_root / "scene_manifest.json"
        if not manifest_path.exists():
            msg = f"Missing scene_manifest.json: {manifest_path}"
            failed.append((idx, scene_id, msg))
            print(f"[ERROR] {msg}")
            if args.fail_fast:
                break
            continue

        manifest = _load_json(manifest_path)
        status = manifest.get("status")
        if status != "completed":
            skipped += 1
            print(f"Scene skipped: status={status!r}")
            continue

        try:
            out_path = scene_root / "transforms_da3.json"
            if args.skip_existing and out_path.exists():
                skipped += 1
                print(f"Scene skipped: existing {out_path.name}")
                continue
            payload = _build_transforms_da3_payload(
                scene_root=scene_root,
                zip_root=zip_root,
                scene_output_subdir=scene_output_subdir,
            )
            _write_json(out_path, payload)
            completed += 1
            print(f"Wrote {out_path}")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failed.append((idx, scene_id, msg))
            print(f"[ERROR] {msg}")
            traceback.print_exc()
            if args.fail_fast:
                break

    print("\n===== DL3DV Eval transforms_da3 Summary =====")
    print(f"Total requested: {end - start}")
    print(f"Written: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failed)}")
    if failed:
        for idx, scene_id, msg in failed[:20]:
            print(f"  - [{idx}] {scene_id}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
