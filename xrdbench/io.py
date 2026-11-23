from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile
from functools import lru_cache

import numpy as np

from .protocol import digest, rng_seed


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_npz(path, **arrays):
    """Stable ZIP metadata, no pickle, reproducible byte hashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for key, value in sorted(arrays.items()):
            value = np.asarray(value)
            if value.dtype.hasobject:
                raise ValueError("Object arrays are forbidden")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
            item = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def manifest(public):
    result = read_json(Path(public) / "manifest.json")
    body = {k: v for k, v in result.items() if k != "dataset_id"}
    if digest(body) != result["dataset_id"]:
        raise ValueError("Dataset manifest fingerprint mismatch")
    return result


@lru_cache(maxsize=8192)
def _contained_path(root, relative):
    # Releases are immutable. Resolve each file once, but verify bytes on EVERY read.
    # This avoids repeated NAS lstat calls for every ancestor of every shard.
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Path escapes artifact root")
    return path


def checked_load(root, record):
    path = _contained_path(str(Path(root).absolute()), record["path"])
    if file_hash(path) != record["sha256"]:
        raise ValueError(f"File checksum mismatch: {path}")
    return load_npz(path)


def iter_observations(public, split, view="full"):
    meta = manifest(public)
    if view not in ("full", "physics128"):
        raise ValueError("Unknown view; use full or physics128")
    groups = None
    if view == "physics128":
        all_groups = set()
        for record in meta["splits"][split]["observations"]:
            all_groups.update(checked_load(public, record)["group_ids"].tolist())
        groups = set(sorted(all_groups, key=lambda g: rng_seed("physics-view-v1", g))[:128])
    for record in meta["splits"][split]["observations"]:
        batch = checked_load(public, record)
        if groups is not None:
            selection = np.isin(batch["group_ids"], list(groups))
            batch = {k: v[selection] for k, v in batch.items()}
        if len(batch["ids"]):
            yield batch


def concatenate(parts):
    parts = list(parts)
    if not parts:
        raise ValueError("No records")
    if any(set(part) != set(parts[0]) for part in parts):
        raise ValueError("Inconsistent shard keys")
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


def load_targets(public, split, private=None):
    meta = manifest(public)
    if split == "test":
        if private is None:
            raise ValueError("Coordinator must provide --private for test scoring")
        secret = read_json(Path(private) / "manifest.json")
        if secret["dataset_id"] != meta["dataset_id"]:
            raise ValueError("Private labels belong to another dataset")
        records, root = secret["test_targets"], private
    else:
        records, root = meta["splits"][split]["targets"], public
    return concatenate(checked_load(root, record) for record in records)
