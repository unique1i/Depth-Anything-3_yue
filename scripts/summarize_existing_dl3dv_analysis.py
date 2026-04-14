#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

"""
Example configs:

1. Original outdoor filtered split
~/local/micromamba/envs/da3/bin/python scripts/summarize_existing_dl3dv_analysis.py \
  --data-root /home/yli7/scratch2/datasets/dl3dv_960p \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_outdoor_min200_filtered.txt \
  --output-dir /home/yli7/repos/Depth-Anything-3/outputs/analyze_existing_dl3dv_all

2. Evaluation 55-scene split
~/local/micromamba/envs/da3/bin/python scripts/summarize_existing_dl3dv_analysis.py \
  --data-root /home/yli7/scratch/datasets/dl3dv_960p/evaluation/da3_streaming \
  --split-file /home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt \
  --output-dir /home/yli7/repos/Depth-Anything-3/outputs/analyze_existing_dl3dv_evaluation

For evaluation scenes, `transforms_da3.json` is usually absent. The script will still summarize
`diagnostics.json`; OBB-based columns stay empty in that case.
"""


DEFAULT_DATA_ROOT = Path("/home/yli7/scratch2/datasets/dl3dv_960p")
DEFAULT_SPLIT_FILE = DEFAULT_DATA_ROOT / "metadata" / "dl3dv_outdoor_min200.txt"
DEFAULT_OUTPUT_DIR = Path("/home/yli7/repos/Depth-Anything-3/outputs/analyze_existing_dl3dv_all")
DEFAULT_FILTERED_SPLIT_PATH = DEFAULT_DATA_ROOT / "metadata" / "dl3dv_outdoor_min200_filtered.txt"
DEFAULT_EXCLUDED_SCENES_TSV = DEFAULT_OUTPUT_DIR / "excluded_scenes_conservative_filter.tsv"
DEFAULT_FILTER_SUMMARY_MD = DEFAULT_OUTPUT_DIR / "conservative_filter_summary.md"
EVAL_DATA_ROOT = Path("/home/yli7/scratch/datasets/dl3dv_960p/evaluation/da3_streaming")
EVAL_SPLIT_FILE = Path("/home/yli7/scratch2/datasets/dl3dv_960p/metadata/dl3dv_evaluation.txt")
EVAL_OUTPUT_DIR = Path("/home/yli7/repos/Depth-Anything-3/outputs/analyze_existing_dl3dv_evaluation")
NUMERIC_OUTLIER_REASONS = {
    "points_da3_extent_outlier",
    "obb_extent_outlier",
    "camera_step_median_outlier",
}


def _read_split_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _load_json(path)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _max_list_value(values) -> float | None:
    if values is None:
        return None
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _warning_category(warning: str) -> str:
    text = warning.lower()
    if "saved non-sky median depth exceeded" in text:
        return "saved_non_sky_depth_gt_10x_threshold"
    if "cumulative sim(3) scale left [0.1, 10]" in text:
        return "cumulative_sim3_scale_out_of_range"
    if "camera-pose step median exceeded 100" in text:
        return "camera_step_median_gt_100"
    if "camera-pose step exceeded 1000" in text:
        return "camera_step_gt_1000"
    if "points_da3.ply bbox extent exceeded 1000" in text:
        return "points_da3_extent_gt_1000"
    if "pre-sim(3) depth masking" in text or "pre-sim(3) depth" in text:
        return "pre_sim3_masking_mismatch"
    return "other_warning"


def _obb_edge_lengths(points) -> tuple[list[float | None], float | None]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape != (8, 3):
        return [None, None, None], None
    dists = np.linalg.norm(arr[1:] - arr[0], axis=1)
    dists = [float(v) for v in dists if v > 1e-9 and math.isfinite(v)]
    if len(dists) < 3:
        return [None, None, None], None
    dists.sort()
    edges: list[float] = []
    for dist in dists:
        if not edges or all(abs(dist - prev) > 1e-6 for prev in edges):
            edges.append(dist)
        if len(edges) == 3:
            break
    while len(edges) < 3:
        edges.append(dists[len(edges)])
    edges.sort()
    return [float(v) for v in edges], float(max(edges))


def _quantile_outlier_threshold(values: Iterable[float | None]) -> float | None:
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return None
    q99 = float(np.quantile(arr, 0.99))
    q3 = float(np.quantile(arr, 0.75))
    q1 = float(np.quantile(arr, 0.25))
    iqr = q3 - q1
    return float(max(q99, q3 + 3.0 * iqr))


def _format_float(value: float | None, precision: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def _to_frame_key(frame_idx_1based: int | None) -> str:
    if frame_idx_1based is None:
        return ""
    return f"frame_{frame_idx_1based:05d}"


def _scene_sort_key(row: dict) -> tuple[int, float, float, float]:
    severity_rank = {"critical": 0, "watchlist": 1, "ok": 2}
    return (
        severity_rank[row["severity"]],
        -(row["points_da3_max_extent"] or -1.0),
        -(row["obb_max_extent"] or -1.0),
        -(row["camera_step_max"] or -1.0),
    )


def _classify_scene(row: dict, thresholds: dict[str, float | None]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    categories = set(filter(None, row["warning_categories"].split(";")))
    if "cumulative_sim3_scale_out_of_range" in categories:
        reasons.append("cumulative_sim3_scale_out_of_range")
    if "camera_step_gt_1000" in categories:
        reasons.append("camera_step_gt_1000")
    if "points_da3_extent_gt_1000" in categories:
        reasons.append("points_da3_extent_gt_1000")
    if "saved_non_sky_depth_gt_10x_threshold" in categories:
        reasons.append("saved_non_sky_depth_gt_10x_threshold")
    if row["obb_max_extent"] is not None and row["obb_max_extent"] > 1000.0:
        reasons.append("obb_max_extent_gt_1000")
    if (
        row["obb_max_extent"] is not None
        and row["points_da3_fused_max_extent"] is not None
        and row["points_da3_fused_max_extent"] > 0.0
        and row["obb_to_fused_extent_ratio"] is not None
        and row["obb_to_fused_extent_ratio"] > 20.0
    ):
        reasons.append("obb_to_fused_extent_ratio_gt_20")
    if reasons:
        return "critical", reasons

    watch_reasons: list[str] = []
    if row["warning_count"] > 0:
        watch_reasons.append("warning_present")
    if thresholds["points_da3_max_extent"] is not None and row["points_da3_max_extent"] is not None:
        if row["points_da3_max_extent"] > thresholds["points_da3_max_extent"]:
            watch_reasons.append("points_da3_extent_outlier")
    if thresholds["obb_max_extent"] is not None and row["obb_max_extent"] is not None:
        if row["obb_max_extent"] > thresholds["obb_max_extent"]:
            watch_reasons.append("obb_extent_outlier")
    if thresholds["camera_step_median"] is not None and row["camera_step_median"] is not None:
        if row["camera_step_median"] > thresholds["camera_step_median"]:
            watch_reasons.append("camera_step_median_outlier")
    if watch_reasons:
        return "watchlist", watch_reasons
    return "ok", []


def _read_metrics_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_filtered_split(
    metrics_csv_path: Path,
    source_split_file: Path,
    filtered_split_path: Path,
    excluded_scenes_tsv: Path,
    filter_summary_md: Path,
    camera_step_ratio_threshold: float,
) -> int:
    source_rels = _read_split_lines(source_split_file)
    rows = _read_metrics_csv(metrics_csv_path)
    by_scene = {row["scene_rel"]: row for row in rows}
    if len(by_scene) != len(rows):
        raise ValueError(f"Duplicate scene_rel rows found in {metrics_csv_path}")
    missing = [rel for rel in source_rels if rel not in by_scene]
    extra = sorted(set(by_scene) - set(source_rels))
    if missing or extra:
        raise ValueError(
            f"Split/metrics mismatch: missing_in_metrics={len(missing)} extra_in_metrics={len(extra)}"
        )

    excluded_rows: list[dict] = []
    retained_rels: list[str] = []
    severity_excluded: Counter[str] = Counter()
    ratio_count = 0
    critical_count = 0
    watchlist_numeric_count = 0
    watchlist_numeric_warning_count = 0
    watchlist_numeric_outlier_only_count = 0

    for rel in source_rels:
        row = dict(by_scene[rel])
        reasons = set(filter(None, row["severity_reasons"].split(";")))
        warning_count = int(row["warning_count"] or 0)
        ratio = _safe_float(row.get("camera_step_max_to_fused_extent_ratio"))
        exclude_ratio = ratio is not None and ratio > camera_step_ratio_threshold
        exclude_critical = row["severity"] == "critical"
        exclude_watchlist_numeric = row["severity"] == "watchlist" and any(
            reason in NUMERIC_OUTLIER_REASONS for reason in reasons
        )
        matched_rules = []
        if exclude_ratio:
            matched_rules.append("exclude_camera_step_ratio_gt_0_5")
            ratio_count += 1
        if exclude_critical:
            matched_rules.append("exclude_critical")
            critical_count += 1
        if exclude_watchlist_numeric:
            matched_rules.append("exclude_watchlist_numeric_outlier")
            watchlist_numeric_count += 1
            if warning_count > 0:
                watchlist_numeric_warning_count += 1
            else:
                watchlist_numeric_outlier_only_count += 1

        row["exclude_camera_step_ratio_gt_0_5"] = str(exclude_ratio)
        row["exclude_critical"] = str(exclude_critical)
        row["exclude_watchlist_numeric_outlier"] = str(exclude_watchlist_numeric)
        row["matched_rules"] = ";".join(matched_rules)

        if matched_rules:
            excluded_rows.append(row)
            severity_excluded[row["severity"]] += 1
        else:
            retained_rels.append(rel)

    retained_watchlist = [
        by_scene[rel]
        for rel in retained_rels
        if by_scene[rel]["severity"] == "watchlist"
    ]
    retained_watchlist_warning_only = sum(
        row["severity_reasons"] == "warning_present" for row in retained_watchlist
    )

    excluded_fieldnames = [
        "idx",
        "scene_rel",
        "severity",
        "severity_reasons",
        "warning_count",
        "warning_categories",
        "camera_step_max_to_fused_extent_ratio",
        "camera_step_max",
        "camera_step_median",
        "points_da3_max_extent",
        "points_da3_fused_max_extent",
        "obb_max_extent",
        "obb_to_fused_extent_ratio",
        "exclude_camera_step_ratio_gt_0_5",
        "exclude_critical",
        "exclude_watchlist_numeric_outlier",
        "matched_rules",
    ]
    excluded_scenes_tsv.parent.mkdir(parents=True, exist_ok=True)
    with excluded_scenes_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=excluded_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded_rows)

    _write_text_lines(filtered_split_path, retained_rels)

    summary_lines = [
        "# DL3DV Outdoor Conservative Filter Summary",
        "",
        "## Rule Log",
        f"- Source split: `{source_split_file}`",
        f"- Metrics CSV: `{metrics_csv_path}`",
        f"- Filtered split: `{filtered_split_path}`",
        f"- Excluded scenes TSV: `{excluded_scenes_tsv}`",
        "- Exclusion rules:",
        f"  - `camera_step_max_to_fused_extent_ratio > {camera_step_ratio_threshold:.1f}`",
        "- `severity == critical`",
        "- `severity == watchlist` and `severity_reasons` contains any of: "
        "`points_da3_extent_outlier`, `obb_extent_outlier`, `camera_step_median_outlier`",
        "",
        "## Counts",
        f"- Source scenes: {len(source_rels)}",
        f"- Excluded by ratio rule (before overlap): {ratio_count}",
        f"- Excluded critical (before overlap): {critical_count}",
        f"- Excluded watchlist numeric-outlier (before overlap): {watchlist_numeric_count}",
        f"- Watchlist numeric-outlier split: {watchlist_numeric_outlier_only_count} outlier-only + {watchlist_numeric_warning_count} warning+numeric",
        f"- Excluded scenes (union): {len(excluded_rows)}",
        f"- Retained scenes: {len(retained_rels)}",
        "",
        "## Excluded By Severity",
        f"- critical: {severity_excluded['critical']}",
        f"- watchlist: {severity_excluded['watchlist']}",
        f"- ok: {severity_excluded['ok']}",
        "",
        "## Retained Watchlist",
        f"- Retained watchlist scenes: {len(retained_watchlist)}",
        f"- Retained watchlist scenes with `warning_present` only: {retained_watchlist_warning_only}",
    ]
    _write_text_lines(filter_summary_md, summary_lines)

    print(f"Wrote {filtered_split_path}")
    print(f"Wrote {excluded_scenes_tsv}")
    print(f"Wrote {filter_summary_md}")
    print(
        f"Conservative filter kept {len(retained_rels)} / {len(source_rels)} scenes "
        f"and excluded {len(excluded_rows)}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize existing DL3DV DA3 analysis across all scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Outdoor filtered split:\n"
            f"    ~/local/micromamba/envs/da3/bin/python scripts/summarize_existing_dl3dv_analysis.py "
            f"--data-root {DEFAULT_DATA_ROOT} "
            f"--split-file {DEFAULT_FILTERED_SPLIT_PATH} "
            f"--output-dir {DEFAULT_OUTPUT_DIR}\n\n"
            "  Evaluation 55-scene split:\n"
            f"    ~/local/micromamba/envs/da3/bin/python scripts/summarize_existing_dl3dv_analysis.py "
            f"--data-root {EVAL_DATA_ROOT} "
            f"--split-file {EVAL_SPLIT_FILE} "
            f"--output-dir {EVAL_OUTPUT_DIR}"
        ),
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene-output-subdir", default="da3_streaming_output")
    parser.add_argument("--logs-dir", default="")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--export-filtered-split", action="store_true")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--source-split-file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--filtered-split-path", default=str(DEFAULT_FILTERED_SPLIT_PATH))
    parser.add_argument("--excluded-scenes-tsv", default=str(DEFAULT_EXCLUDED_SCENES_TSV))
    parser.add_argument("--filter-summary-md", default=str(DEFAULT_FILTER_SUMMARY_MD))
    parser.add_argument("--camera-step-max-to-fused-extent-threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    split_file = Path(args.split_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    logs_dir = Path(args.logs_dir).expanduser().resolve() if args.logs_dir else output_dir / "logs"
    metrics_csv_path = (
        Path(args.metrics_csv).expanduser().resolve()
        if args.metrics_csv
        else output_dir / "all_scene_metrics.csv"
    )
    source_split_file = Path(args.source_split_file).expanduser().resolve()
    filtered_split_path = Path(args.filtered_split_path).expanduser().resolve()
    excluded_scenes_tsv = Path(args.excluded_scenes_tsv).expanduser().resolve()
    filter_summary_md = Path(args.filter_summary_md).expanduser().resolve()

    if args.export_filtered_split:
        return _export_filtered_split(
            metrics_csv_path=metrics_csv_path,
            source_split_file=source_split_file,
            filtered_split_path=filtered_split_path,
            excluded_scenes_tsv=excluded_scenes_tsv,
            filter_summary_md=filter_summary_md,
            camera_step_ratio_threshold=args.camera_step_max_to_fused_extent_threshold,
        )

    rels = _read_split_lines(split_file)

    rows: list[dict] = []
    warning_counter: Counter[str] = Counter()
    complete_logs = 0
    warn_logs = 0
    missing_logs = 0
    warning_scene_count = 0
    transforms_da3_found = 0
    progress_every = max(int(args.progress_every), 1)
    start_time = time.time()

    print(
        f"Scanning {len(rels)} scenes from {split_file} "
        f"(progress every {progress_every} scenes)."
    )

    for idx, rel in enumerate(rels):
        scene_root = data_root / rel
        diag_path = scene_root / args.scene_output_subdir / "diagnostics.json"
        tda3_path = scene_root / "transforms_da3.json"
        log_path = logs_dir / f"{idx:05d}.log"

        diagnostics = _load_json(diag_path)
        transforms_da3 = _load_json_if_exists(tda3_path)
        if transforms_da3 is not None:
            transforms_da3_found += 1

        warnings = diagnostics.get("warnings", [])
        if warnings:
            warning_scene_count += 1
        warning_categories = sorted({_warning_category(w) for w in warnings})
        for cat in warning_categories:
            warning_counter[cat] += 1

        artifacts = diagnostics.get("artifacts", {})
        points_da3_max_extent = _max_list_value((artifacts.get("points_da3") or {}).get("bbox_extent"))
        points_da3_fused_max_extent = _max_list_value((artifacts.get("points_da3_fused") or {}).get("bbox_extent"))

        camera_txt = artifacts.get("camera_poses_txt") or {}
        step_quantiles = camera_txt.get("step_quantiles") or []
        camera_step_median = _safe_float(step_quantiles[2]) if len(step_quantiles) >= 3 else None
        camera_step_max = _safe_float(step_quantiles[-1]) if step_quantiles else None
        camera_step_max_to_fused_extent_ratio = None
        if (
            camera_step_max is not None
            and points_da3_fused_max_extent is not None
            and points_da3_fused_max_extent > 0.0
        ):
            camera_step_max_to_fused_extent_ratio = camera_step_max / points_da3_fused_max_extent
        first_step_gt_1000_index = camera_txt.get("first_step_gt_1000_index")
        first_large_step_to_frame_idx = int(first_step_gt_1000_index + 1) if first_step_gt_1000_index is not None else None

        saved_frames = diagnostics.get("saved_frame_diagnostics") or []
        max_saved_non_sky_depth = None
        first_saved_depth_offender_key = ""
        depth_threshold_m = _safe_float((diagnostics.get("runtime_options") or {}).get("depth_threshold_m")) or 100.0
        for frame in saved_frames:
            ns_med = _safe_float(frame.get("non_sky_depth_median_m"))
            if ns_med is not None and (max_saved_non_sky_depth is None or ns_med > max_saved_non_sky_depth):
                max_saved_non_sky_depth = ns_med
            if not first_saved_depth_offender_key and ns_med is not None and ns_med > 10.0 * depth_threshold_m:
                first_saved_depth_offender_key = frame.get("frame_key", "")

        if transforms_da3 is not None:
            obb_edge_lengths, obb_max_extent = _obb_edge_lengths(transforms_da3.get("bbox_obb_points"))
        else:
            obb_edge_lengths, obb_max_extent = [None, None, None], None
        obb_to_fused_extent_ratio = None
        if obb_max_extent is not None and points_da3_fused_max_extent is not None and points_da3_fused_max_extent > 0.0:
            obb_to_fused_extent_ratio = obb_max_extent / points_da3_fused_max_extent

        log_exists = log_path.exists()
        log_has_warn = False
        log_analysis_complete = False
        if log_exists:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            log_has_warn = "[WARN]" in text
            log_analysis_complete = "Analysis complete:" in text
            complete_logs += int(log_analysis_complete)
            warn_logs += int(log_has_warn)
        else:
            missing_logs += 1

        rows.append(
            {
                "idx": idx,
                "scene_rel": rel,
                "scene_root": str(scene_root),
                "diagnostics_path": str(diag_path),
                "transforms_da3_path": str(tda3_path),
                "log_path": str(log_path),
                "log_exists": log_exists,
                "log_has_warn": log_has_warn,
                "log_analysis_complete": log_analysis_complete,
                "transforms_da3_exists": transforms_da3 is not None,
                "warning_count": len(warnings),
                "warning_categories": ";".join(warning_categories),
                "warnings_json": json.dumps(warnings, ensure_ascii=True),
                "points_da3_max_extent": points_da3_max_extent,
                "points_da3_fused_max_extent": points_da3_fused_max_extent,
                "camera_step_median": camera_step_median,
                "camera_step_max": camera_step_max,
                "camera_step_max_to_fused_extent_ratio": camera_step_max_to_fused_extent_ratio,
                "first_step_gt_1000_index": first_step_gt_1000_index,
                "first_large_step_to_frame_idx": first_large_step_to_frame_idx,
                "first_large_step_to_frame_key": _to_frame_key(first_large_step_to_frame_idx),
                "max_saved_non_sky_depth_median": max_saved_non_sky_depth,
                "first_saved_depth_offender_key": first_saved_depth_offender_key,
                "obb_edge_len_1": _safe_float(obb_edge_lengths[0]),
                "obb_edge_len_2": _safe_float(obb_edge_lengths[1]),
                "obb_edge_len_3": _safe_float(obb_edge_lengths[2]),
                "obb_max_extent": obb_max_extent,
                "obb_to_fused_extent_ratio": obb_to_fused_extent_ratio,
                "depth_threshold_m": depth_threshold_m,
            }
        )

        if (idx + 1) % progress_every == 0 or (idx + 1) == len(rels):
            elapsed = max(time.time() - start_time, 1e-9)
            rate = (idx + 1) / elapsed
            print(
                f"[progress] scanned {idx + 1}/{len(rels)} scenes "
                f"({rate:.2f} scenes/s, elapsed={elapsed:.1f}s, "
                f"warning_scenes={warning_scene_count}, log_warns={warn_logs})"
            )

    thresholds = {
        "points_da3_max_extent": _quantile_outlier_threshold(row["points_da3_max_extent"] for row in rows),
        "obb_max_extent": _quantile_outlier_threshold(row["obb_max_extent"] for row in rows),
        "camera_step_median": _quantile_outlier_threshold(row["camera_step_median"] for row in rows),
    }

    severity_counter: Counter[str] = Counter()
    for row in rows:
        severity, reasons = _classify_scene(row, thresholds)
        row["severity"] = severity
        row["severity_reasons"] = ";".join(reasons)
        severity_counter[severity] += 1

    rows.sort(key=_scene_sort_key)
    flagged_rows = [row for row in rows if row["severity"] != "ok"]

    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "idx",
        "scene_rel",
        "scene_root",
        "severity",
        "severity_reasons",
        "warning_count",
        "warning_categories",
        "warnings_json",
        "points_da3_max_extent",
        "points_da3_fused_max_extent",
        "camera_step_median",
        "camera_step_max",
        "camera_step_max_to_fused_extent_ratio",
        "first_step_gt_1000_index",
        "first_large_step_to_frame_idx",
        "first_large_step_to_frame_key",
        "max_saved_non_sky_depth_median",
        "first_saved_depth_offender_key",
        "obb_edge_len_1",
        "obb_edge_len_2",
        "obb_edge_len_3",
        "obb_max_extent",
        "obb_to_fused_extent_ratio",
        "depth_threshold_m",
        "log_exists",
        "log_has_warn",
        "log_analysis_complete",
        "transforms_da3_exists",
        "diagnostics_path",
        "transforms_da3_path",
        "log_path",
    ]

    with (output_dir / "all_scene_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "flagged_scenes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged_rows)

    with (output_dir / "warning_category_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["warning_category", "scene_count"])
        writer.writeheader()
        for cat, count in sorted(warning_counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"warning_category": cat, "scene_count": count})

    top_points = sorted(rows, key=lambda row: row["points_da3_max_extent"] or -1.0, reverse=True)[:20]
    top_obb = sorted(rows, key=lambda row: row["obb_max_extent"] or -1.0, reverse=True)[:20]
    top_camera = sorted(rows, key=lambda row: row["camera_step_max"] or -1.0, reverse=True)[:20]
    critical_rows = [row for row in rows if row["severity"] == "critical"][:50]

    summary_lines = [
        "# DL3DV Outdoor DA3 Analysis Summary",
        "",
        "## Coverage",
        f"- Split scenes: {len(rels)}",
        f"- diagnostics.json found: {len(rows)}",
        f"- transforms_da3.json found: {transforms_da3_found}",
        f"- Logs with `Analysis complete:`: {complete_logs}",
        f"- Logs containing `[WARN]`: {warn_logs}",
        f"- Missing logs: {missing_logs}",
        "",
        "## Severity Counts",
        f"- critical: {severity_counter['critical']}",
        f"- watchlist: {severity_counter['watchlist']}",
        f"- ok: {severity_counter['ok']}",
        "",
        "## Outlier Thresholds",
        f"- points_da3_max_extent: {_format_float(thresholds['points_da3_max_extent'])}",
        f"- obb_max_extent: {_format_float(thresholds['obb_max_extent'])}",
        f"- camera_step_median: {_format_float(thresholds['camera_step_median'])}",
        "",
        "## Metric Definitions",
        "- `points_da3_max_extent`: largest axis-aligned bbox side length of `points_da3.ply`.",
        "- `points_da3_fused_max_extent`: largest axis-aligned bbox side length of `points_da3_fused.ply`.",
        "- `obb_max_extent`: largest OBB edge length derived from `transforms_da3.json[\"bbox_obb_points\"]`; empty when `transforms_da3.json` is unavailable.",
        "- `obb_to_fused_extent_ratio`: `obb_max_extent / points_da3_fused_max_extent`; empty when OBB metadata is unavailable.",
        "- `camera_step_max_adjacent`: largest Euclidean distance between neighboring camera centers from `camera_poses.txt`.",
        "- `camera_step_median_adjacent`: median Euclidean distance between neighboring camera centers from `camera_poses.txt`.",
        "- `camera_step_max_to_fused_extent_ratio`: `camera_step_max_adjacent / points_da3_fused_max_extent`; dimensionless severity ratio.",
        "- Camera-step values are computed only from adjacent frames via `np.diff(centers, axis=0)`, not all frame pairs.",
        "- A huge `camera_step_max_adjacent` with a small `camera_step_median_adjacent` indicates a local pose jump / teleport, not sustained motion.",
        "- Camera-step values use the saved DA3 world/pose units and should be interpreted as meters in this DL3DV workflow.",
        "",
        "## Warning Category Counts",
    ]
    for cat, count in sorted(warning_counter.items(), key=lambda item: (-item[1], item[0])):
        summary_lines.append(f"- {cat}: {count}")

    def add_table(title: str, scene_rows: list[dict], columns: list[tuple[str, str]]) -> None:
        header = "| scene | severity | " + " | ".join(label for label, _ in columns) + " | reasons |"
        align = "| --- | --- | " + " | ".join("---:" for _ in columns) + " | --- |"
        summary_lines.extend(["", f"## {title}", "", header, align])
        for row in scene_rows[:20]:
            values = " | ".join(_format_float(row.get(key)) for _, key in columns)
            summary_lines.append(f"| {row['scene_rel']} | {row['severity']} | {values} | {row['severity_reasons']} |")

    add_table(
        "Top points_da3 Extent",
        top_points,
        [
            ("points_da3_max_extent", "points_da3_max_extent"),
            ("points_da3_fused_max_extent", "points_da3_fused_max_extent"),
        ],
    )
    add_table(
        "Top OBB Extent",
        top_obb,
        [
            ("obb_max_extent", "obb_max_extent"),
            ("obb_to_fused_extent_ratio", "obb_to_fused_extent_ratio"),
        ],
    )
    add_table(
        "Top Camera Step Max",
        top_camera,
        [
            ("camera_step_max_adjacent", "camera_step_max"),
            ("camera_step_median_adjacent", "camera_step_median"),
            ("camera_step_max_to_fused_extent_ratio", "camera_step_max_to_fused_extent_ratio"),
        ],
    )

    summary_lines.extend(["", "## Critical Review", "", "| scene | points_da3_max | obb_max | camera_step_max | warnings | reasons |", "| --- | ---: | ---: | ---: | --- | --- |"])
    for row in critical_rows:
        summary_lines.append(
            f"| {row['scene_rel']} | {_format_float(row['points_da3_max_extent'])} | {_format_float(row['obb_max_extent'])} | "
            f"{_format_float(row['camera_step_max'])} | {row['warning_categories']} | {row['severity_reasons']} |"
        )

    (output_dir / "analysis_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    total_elapsed = time.time() - start_time
    print(f"Finished scanning {len(rels)} scenes in {total_elapsed:.1f}s.")
    print(f"Wrote {(output_dir / 'all_scene_metrics.csv')}")
    print(f"Wrote {(output_dir / 'flagged_scenes.csv')}")
    print(f"Wrote {(output_dir / 'warning_category_counts.csv')}")
    print(f"Wrote {(output_dir / 'analysis_summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
