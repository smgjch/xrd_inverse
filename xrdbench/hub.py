"""Download immutable, complete benchmark releases, including public test targets."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from .io import file_hash, manifest, read_json

DATASET_REPO = "smgjch/PXRD-LatticeBench"
# Immutable v0.2-pilot snapshot; numerical data and original dataset IDs are frozen.
DATASET_REVISION = "a7f5ce0c62d7fc5f50f1c2ce2d34b23f2e58a3d2"


def _contained(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("Unsafe dataset member path")
    path = Path(root).joinpath(*relative.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset path escapes release root")
    return path


def verify_release(root, entry):
    root = Path(root)
    meta = manifest(root / "public")
    labels = read_json(root / "private/manifest.json")
    if meta["dataset_id"] != entry["dataset_id"] or labels["dataset_id"] != entry["dataset_id"]:
        raise ValueError("Downloaded release dataset_id mismatch")
    cfg = meta["config"]
    if (cfg["task"]["task"], cfg["track"], cfg.get("suite", "v0.1")) != (entry["task"], entry["track"], entry["suite"]):
        raise ValueError("Downloaded task/track/suite mismatch")
    checked = 0
    records = [(root / "public", meta["prototype"])]
    for split in meta["splits"].values():
        records.extend((root / "public", r) for r in split["observations"] + split["targets"])
    records.extend((root / "private", r) for r in labels["test_targets"])
    for base, record in records:
        if file_hash(_contained(base, record["path"])) != record["sha256"]:
            raise ValueError(f"Downloaded file checksum mismatch: {record['path']}")
        checked += 1
    return {"root": str(root), "dataset_id": meta["dataset_id"], "verified_files": checked,
            "test_labels_included": True}


def unpack_archive(archive, output, entry):
    """Verify before extraction, reject links/traversal, and publish atomically."""
    archive, output = Path(archive), Path(output)
    if file_hash(archive) != entry["archive"]["sha256"]:
        raise ValueError("Downloaded archive SHA-256 mismatch")
    if output.is_symlink():
        raise ValueError("Release output must not be a symlink")
    if output.exists():
        # The test-label manifest has no independent dataset digest. Compare
        # both manifests with the verified archive before trusting their lists.
        with tarfile.open(archive, "r:gz") as handle:
            for name in ("public/manifest.json", "private/manifest.json"):
                with handle.extractfile(name) as original:
                    if _contained(output, name).read_bytes() != original.read():
                        raise ValueError("Existing release manifest differs from the frozen archive")
        return {**verify_release(output, entry), "already_present": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xrdbench-download-", dir=output.parent) as temp:
        staging = Path(temp) / "release"
        staging.mkdir()
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            if len(members) != entry["archive"]["file_count"]:
                raise ValueError("Archive member count mismatch")
            if sum(m.size for m in members) != entry["archive"]["expanded_file_bytes"]:
                raise ValueError("Archive expanded size mismatch")
            names = set()
            for member in members:
                if not member.isfile() or member.name in names:
                    raise ValueError("Archive contains a link, non-file or duplicate member")
                names.add(member.name)
                path = _contained(staging, member.name)
                if PurePosixPath(member.name).parts[0] not in ("public", "private"):
                    raise ValueError("Unexpected archive root")
                path.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        result = verify_release(staging, entry)
        if output.exists():
            raise FileExistsError("Output appeared during download; refusing to replace it")
        staging.rename(output)
    return {**result, "root": str(output), "already_present": False}


def download(output, task="rutile", track="calibrated", suite="v0.1+v0.2",
             all_releases=False, revision=DATASET_REVISION, cache_dir=None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install the Hub extra: pip install -e '.[hub]'") from exc
    kwargs = {"repo_id": DATASET_REPO, "repo_type": "dataset", "revision": revision,
              "cache_dir": cache_dir, "token": False}
    catalog = read_json(hf_hub_download(filename="publication.json", **kwargs))
    entries = [e for e in catalog["releases"] if all_releases or
               (e["task"], e["track"], e["suite"]) == (task, track, suite)]
    if not entries:
        raise ValueError("Requested task/track/suite is absent from this release")
    results = []
    for entry in entries:
        destination = Path(output)
        if all_releases:
            destination = destination / entry["task"] / entry["track"] / entry["suite"]
        archive = hf_hub_download(filename=entry["archive"]["path"], **kwargs)
        results.append(unpack_archive(archive, destination, entry))
    return {"repo_id": DATASET_REPO, "revision": revision, "releases": results,
            "split_aliases": {"validation": "val"}, "training_performed": False}
