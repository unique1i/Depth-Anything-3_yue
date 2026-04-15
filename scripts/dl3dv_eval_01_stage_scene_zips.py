#!/usr/bin/env python3
"""Step 01: stage DL3DV evaluation tar archives into per-scene zip archives."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import traceback
import zipfile
from pathlib import Path


DEFAULT_SPLIT_FILE = "/home/yli7/scratch2/datasets/dl3dv_960p/metadata/splits/dl3dv_evaluation_filtered.txt"
DEFAULT_TAR_ROOT = "/home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images_tar"
DEFAULT_ZIP_ROOT = "/home/yli7/scratch2/datasets/dl3dv_960p/evaluation/images"


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


def _tar_member_to_zip_info(member: tarfile.TarInfo) -> zipfile.ZipInfo:
    zip_info = zipfile.ZipInfo(filename=member.name)
    zip_info.compress_type = zipfile.ZIP_STORED
    zip_info.create_system = 3
    zip_info.external_attr = (member.mode & 0xFFFF) << 16
    return zip_info


def _stage_one_scene(tar_path: Path, zip_path: Path) -> tuple[int, int]:
    tmp_path = zip_path.with_name(f"{zip_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    file_count = 0
    dir_count = 0
    try:
        with tarfile.open(tar_path, mode="r:*") as tf, zipfile.ZipFile(
            tmp_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as zf:
            for member in tf:
                if not member.name:
                    continue

                zip_info = _tar_member_to_zip_info(member)
                if member.isdir():
                    if not zip_info.filename.endswith("/"):
                        zip_info.filename = f"{zip_info.filename}/"
                    zf.writestr(zip_info, b"")
                    dir_count += 1
                    continue
                if not member.isfile():
                    continue

                extracted = tf.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Failed to read tar member: {member.name}")
                with extracted:
                    with zf.open(zip_info, mode="w") as out_f:
                        shutil.copyfileobj(extracted, out_f, length=1024 * 1024)
                file_count += 1

        tmp_path.replace(zip_path)
        return file_count, dir_count
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 01: stage DL3DV evaluation tar archives into per-scene zip archives in split order "
            "without extracting scene directories to disk."
        )
    )
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--tar-root", default=DEFAULT_TAR_ROOT)
    parser.add_argument("--zip-root", default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    split_file = Path(args.split_file).expanduser().resolve()
    tar_root = Path(args.tar_root).expanduser().resolve()
    zip_root = Path(args.zip_root).expanduser().resolve()

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not tar_root.exists():
        raise FileNotFoundError(f"Tar root not found: {tar_root}")

    scene_ids = _read_split_lines(split_file)
    total = len(scene_ids)
    start = max(0, int(args.start_idx))
    end = total if args.end_idx is None else min(total, int(args.end_idx))
    if start >= end:
        print(f"No scenes to stage for range [{start}, {end}).")
        return 0

    zip_root.mkdir(parents=True, exist_ok=True)

    staged = 0
    skipped = 0
    failed: list[tuple[int, str, str]] = []

    print(f"Staging scenes [{start}, {end}) out of {total}")
    print(f"Tar root: {tar_root}")
    print(f"Zip root: {zip_root}")

    for idx in range(start, end):
        scene_id = scene_ids[idx]
        tar_path = tar_root / f"{scene_id}.tar"
        zip_path = zip_root / f"{scene_id}.zip"
        print(f"\n[{idx}] scene={scene_id}")

        if args.skip_existing and zip_path.exists():
            skipped += 1
            print(f"Skipped existing {zip_path}")
            continue

        try:
            if not tar_path.exists():
                raise FileNotFoundError(f"Tar archive not found: {tar_path}")

            file_count, dir_count = _stage_one_scene(tar_path, zip_path)
            staged += 1
            print(
                f"Staged {zip_path.name} from {tar_path.name} "
                f"(files={file_count}, dirs={dir_count})"
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failed.append((idx, scene_id, msg))
            print(f"[ERROR] {msg}")
            traceback.print_exc()
            if args.fail_fast:
                break

    print("\n===== DL3DV Eval Tar->Zip Staging Summary =====")
    print(f"Total requested: {end - start}")
    print(f"Staged: {staged}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failed)}")
    if failed:
        for idx, scene_id, msg in failed[:20]:
            print(f"  - [{idx}] {scene_id}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
