"""Scene-level DA3 inference for ScanNet and ScanNet++."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from depth_anything_3.pipelines.scene_inference import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSE_SCALE_CALIB_FRAMES,
    DEFAULT_POSE_SCALE_MODE,
    DEFAULT_SCANNETPP_ROOT,
    DEFAULT_SCANNETPP_SPLIT,
    DEFAULT_SCANNET_ROOT,
    DEFAULT_SCANNET_SPLIT,
    RunOptions,
    SUPPORTED_POSE_SCALE_MODES,
    build_batch_scene_specs,
    build_single_scene_spec,
    load_model,
    run_scene_inference,
)
from depth_anything_3.utils.memory import cleanup_cuda_memory  # noqa: E402


def _add_bool_flag(
    parser: argparse.ArgumentParser, name: str, default: bool, help_text: str
) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(
        f"--no-{name}", dest=dest, action="store_false", help=f"Disable: {help_text}"
    )
    parser.set_defaults(**{dest: default})


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID, help="Model repo or local path"
    )
    parser.add_argument(
        "--device", default="cuda", help="Torch device, e.g., cuda or cpu"
    )
    _add_bool_flag(parser, "pose-condition", False, "Enable pose-conditioned inference")
    _add_bool_flag(parser, "save-depth", False, "Save per-frame metric depth PNGs")
    _add_bool_flag(parser, "save-pointcloud", True, "Save global fused point cloud PLY")
    _add_bool_flag(parser, "save-gs", True, "Save global 3DGS PLY")
    parser.add_argument(
        "--depth-scale",
        type=int,
        default=1000,
        help="Depth scale factor for uint16 PNG export",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=48,
        help="Initial chunk size for scene inference",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=16,
        help="Minimum chunk size for OOM fallback",
    )
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=[
            "upper_bound_resize",
            "lower_bound_resize",
            "upper_bound_crop",
            "lower_bound_crop",
        ],
        help="DA3 preprocessing resize policy",
    )
    parser.add_argument(
        "--pose-scale-mode",
        default=DEFAULT_POSE_SCALE_MODE,
        choices=list(SUPPORTED_POSE_SCALE_MODES),
        help="Depth scaling policy under pose conditioning: none, per_chunk, or global",
    )
    parser.add_argument(
        "--pose-scale-calib-frames",
        type=int,
        default=DEFAULT_POSE_SCALE_CALIB_FRAMES,
        help="Frame budget for global pose-scale calibration pass",
    )
    _add_bool_flag(
        parser,
        "align-to-input-ext-scale",
        True,
        "Align predicted poses/depth to input extrinsics scale when pose-conditioning is enabled",
    )
    parser.add_argument(
        "--gs-max-frames",
        type=int,
        default=128,
        help="Maximum selected frames for 3DGS pass",
    )
    parser.add_argument(
        "--gs-views-interval",
        type=int,
        default=8,
        help="Optional gs_ply views interval. If unset, exporter auto behavior is used.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for non-depth outputs (manifest, fused point cloud, global 3DGS)",
    )
    parser.add_argument(
        "--output-subdir",
        default="depth_est_da3",
        help="Depth PNG subfolder written under the original scene root when --save-depth is enabled",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular DA3 scene inference")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser(
        "single", help="Run one scene from a transforms JSON"
    )
    single.add_argument(
        "--scene-root", required=True, help="Path to scene root directory"
    )
    single.add_argument(
        "--transforms-json",
        default=None,
        help="Path to transforms JSON (default: <scene-root>/transforms_train.json)",
    )
    single.add_argument(
        "--image-root",
        default=None,
        help="Optional root used to resolve frame file_path entries",
    )
    single.add_argument(
        "--target-width", type=int, default=None, help="Optional target output width"
    )
    single.add_argument(
        "--target-height", type=int, default=None, help="Optional target output height"
    )
    _add_shared_args(single)

    batch = subparsers.add_parser("batch", help="Run all scenes from dataset splits")
    batch.add_argument(
        "--datasets",
        nargs="+",
        default=["scannet"],
        choices=["scannet", "scannetpp"],
        help="Datasets to run",
    )
    batch.add_argument(
        "--scannet-split",
        default=DEFAULT_SCANNET_SPLIT,
        help="Path to ScanNet split file",
    )
    batch.add_argument(
        "--scannet-root",
        default=DEFAULT_SCANNET_ROOT,
        help="Path to ScanNet scans root",
    )
    batch.add_argument(
        "--scannetpp-split",
        default=DEFAULT_SCANNETPP_SPLIT,
        help="Path to ScanNet++ split file",
    )
    batch.add_argument(
        "--scannetpp-root",
        default=DEFAULT_SCANNETPP_ROOT,
        help="Path to ScanNet++ data root",
    )
    batch.add_argument(
        "--scene-filter",
        nargs="*",
        default=None,
        help="Optional scene names to keep (space or comma separated)",
    )
    _add_shared_args(batch)

    return parser


def _normalize_scene_filter(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        out.extend(parts)
    return out or None


def _build_options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        model_id=args.model_id,
        device=args.device,
        pose_condition=args.pose_condition,
        save_depth=args.save_depth,
        save_pointcloud=args.save_pointcloud,
        save_gs=args.save_gs,
        depth_scale=args.depth_scale,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        process_res_method=args.process_res_method,
        pose_scale_mode=args.pose_scale_mode,
        pose_scale_calib_frames=args.pose_scale_calib_frames,
        align_to_input_ext_scale=args.align_to_input_ext_scale,
        gs_max_frames=args.gs_max_frames,
        gs_views_interval=args.gs_views_interval,
        output_root=args.output_root,
        output_subdir=args.output_subdir,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    options = _build_options(args)

    if options.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if options.min_chunk_size < 1:
        raise ValueError("--min-chunk-size must be >= 1")
    if options.chunk_size < options.min_chunk_size:
        raise ValueError("--chunk-size must be >= --min-chunk-size")
    if options.pose_scale_calib_frames < 1:
        raise ValueError("--pose-scale-calib-frames must be >= 1")

    if args.mode == "single":
        scene_specs = [
            build_single_scene_spec(
                scene_root=args.scene_root,
                transforms_json=args.transforms_json,
                image_root=args.image_root,
                target_width=args.target_width,
                target_height=args.target_height,
            )
        ]
    else:
        scene_specs = build_batch_scene_specs(
            datasets=args.datasets,
            scannet_split=args.scannet_split,
            scannet_root=args.scannet_root,
            scannetpp_split=args.scannetpp_split,
            scannetpp_root=args.scannetpp_root,
            scene_filter=_normalize_scene_filter(args.scene_filter),
        )

    if not scene_specs:
        print("No scenes selected. Nothing to run.", flush=True)
        return 0

    print(f"Loading model: {options.model_id} on {options.device}", flush=True)
    model = load_model(options.model_id, options.device)

    start_time = time.time()
    failed: list[str] = []
    for index, scene_spec in enumerate(scene_specs, start=1):
        print(
            f"\n[{index}/{len(scene_specs)}] Running scene {scene_spec.scene_name} ({scene_spec.dataset_name})",
            flush=True,
        )
        manifest = run_scene_inference(model=model, scene=scene_spec, options=options)
        if manifest.get("status") != "success":
            failed.append(scene_spec.scene_name)

    cleanup_cuda_memory()

    elapsed = time.time() - start_time
    succeeded = len(scene_specs) - len(failed)
    print(
        f"\nCompleted {len(scene_specs)} scenes in {elapsed:.2f}s | success={succeeded} failed={len(failed)}",
        flush=True,
    )
    if failed:
        print(f"Failed scenes: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
