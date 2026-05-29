"""
Load face images from the dataset folder, encode with face_recognition,
and persist encodings in SQLite for recognition comparison.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from database import (
    add_user,
    find_user_id_by_alphanumeric_name,
    find_user_id_by_name,
    list_users_with_encodings,
    update_user_face_encoding,
)
from face_module import encode_face_from_image_bytes, load_known_users


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Matches filenames saved by add-user: Name_YYYYMMDD_HHMMSS
_STEM_WITH_TS = re.compile(r"^(.+)_(\d{8})_(\d{6})$")


def display_name_from_filename(path: Path) -> str:
    stem = path.stem
    m = _STEM_WITH_TS.match(stem)
    if m:
        return m.group(1).replace("_", " ").strip() or stem
    return stem.replace("_", " ").strip() or stem


def lookup_stem_from_filename(path: Path) -> str:
    stem = path.stem
    m = _STEM_WITH_TS.match(stem)
    if m:
        return m.group(1)
    return stem


def resolve_existing_user_id(db_path: str, path: Path) -> Optional[int]:
    display = display_name_from_filename(path)
    stem_key = lookup_stem_from_filename(path)
    uid = find_user_id_by_name(db_path, display)
    if uid is not None:
        return uid
    return find_user_id_by_alphanumeric_name(db_path, stem_key)


def list_dataset_image_paths(dataset_dir: str) -> List[Path]:
    root = Path(dataset_dir)
    if not root.is_dir():
        return []
    paths: List[Path] = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            paths.append(p)
    return sorted(paths, key=lambda x: x.name.lower())


def sync_dataset_to_database(
    db_path: str,
    dataset_dir: str,
    *,
    default_user_type: str = "permanent",
) -> Dict[str, Any]:
    """
    Encode each image in dataset_dir and upsert into the DB.
    New name  → INSERT (permanent, no expiry).
    Existing  → UPDATE face_encoding only (preserves user_type & expiry).
    Reloads in-memory known-user cache when done.
    """
    if default_user_type not in {"permanent", "guest"}:
        raise ValueError("default_user_type must be 'permanent' or 'guest'")

    stats: Dict[str, Any] = {
        "files": 0,
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
    }

    for path in list_dataset_image_paths(dataset_dir):
        stats["files"] += 1
        display_name = display_name_from_filename(path)
        try:
            with open(path, "rb") as f:
                raw = f.read()
            if not raw:
                stats["skipped"] += 1
                stats["errors"].append(f"{path.name}: empty file")
                continue

            encoding = encode_face_from_image_bytes(raw)
            existing_id = resolve_existing_user_id(db_path, path)

            if existing_id is None:
                add_user(db_path, display_name, encoding, default_user_type, None)
                stats["added"] += 1
            else:
                update_user_face_encoding(db_path, existing_id, encoding)
                stats["updated"] += 1
        except ValueError as e:
            stats["skipped"] += 1
            stats["errors"].append(f"{path.name}: {e}")
        except OSError as e:
            stats["skipped"] += 1
            stats["errors"].append(f"{path.name}: {e}")

    load_known_users(list_users_with_encodings(db_path))
    return stats


def print_dataset_sync_summary(stats: Dict[str, Any], dataset_dir: str) -> None:
    print(
        f"[dataset] folder={os.path.abspath(dataset_dir)} "
        f"files={stats['files']} added={stats['added']} "
        f"updated={stats['updated']} skipped={stats['skipped']}"
    )
    for err in stats.get("errors", [])[:20]:
        print(f"  - {err}")
    if len(stats.get("errors", [])) > 20:
        print(f"  ... and {len(stats['errors']) - 20} more")