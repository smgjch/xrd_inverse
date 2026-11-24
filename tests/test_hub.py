import io
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from xrdbench.data import generate
from xrdbench.hub import download, unpack_archive, verify_release
from xrdbench.io import file_hash, iter_observations, load_targets, manifest, read_json, write_json
from xrdbench.submission import run, validate_submission


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    parent = tmp_path_factory.mktemp("hub")
    root = parent / "release"
    generate(root, "rutile", "blind", "smoke", suite="v0.2")
    archive = parent / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(root).as_posix())
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
    entry = {"task": "rutile", "track": "blind", "suite": "v0.2",
             "dataset_id": manifest(root / "public")["dataset_id"],
             "archive": {"path": "archives/release.tar.gz", "sha256": file_hash(archive),
                         "file_count": len(members), "expanded_file_bytes": sum(m.size for m in members)}}
    return root, archive, entry


def test_download_retains_test_labels_and_is_idempotent(frozen, tmp_path, monkeypatch):
    _, archive, entry = frozen
    catalog = tmp_path / "publication.json"
    write_json(catalog, {"releases": [entry]})
    calls = []
    def fetch(filename, **kwargs):
        calls.append(kwargs)
        return catalog if filename == "publication.json" else archive
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fetch))
    result = download(tmp_path / "data", task="rutile", track="blind", suite="v0.2", revision="fixed-commit")
    assert result["releases"][0]["test_labels_included"]
    assert len(load_targets(tmp_path / "data/public", "test", tmp_path / "data/private")["theta"]) == 48
    assert unpack_archive(archive, tmp_path / "data", entry)["already_present"]
    assert all(call["revision"] == "fixed-commit" and call["token"] is False for call in calls)
    all_result = download(tmp_path / "matrix", all_releases=True, revision="fixed-commit")
    assert all_result["releases"][0]["root"].endswith("rutile/blind/v0.2")


def test_corruption_and_existing_different_data_are_rejected(frozen, tmp_path):
    _, archive, entry = frozen
    corrupt = tmp_path / "corrupt.tar.gz"
    corrupt.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        unpack_archive(corrupt, tmp_path / "bad", entry)
    assert not (tmp_path / "bad").exists()
    unpack_archive(archive, tmp_path / "good", entry)
    target = next((tmp_path / "good/private/test").glob("*.npz"))
    target.write_bytes(target.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        unpack_archive(archive, tmp_path / "good", entry)
    with pytest.raises(ValueError, match="dataset_id"):
        verify_release(tmp_path / "good", {**entry, "dataset_id": "another-release"})


def test_existing_private_manifest_cannot_silently_drop_test_labels(frozen, tmp_path):
    _, archive, entry = frozen
    root = tmp_path / "data"
    unpack_archive(archive, root, entry)
    labels = read_json(root / "private/manifest.json")
    labels["test_targets"] = []
    write_json(root / "private/manifest.json", labels)
    with pytest.raises(ValueError, match="manifest differs"):
        unpack_archive(archive, root, entry)


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_unsafe_archive_never_writes_outside_staging(tmp_path, kind):
    archive = tmp_path / "unsafe.tar.gz"
    member = tarfile.TarInfo("../escaped" if kind == "traversal" else "public/link")
    if kind == "symlink":
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escaped"
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(member, io.BytesIO(b""))
    entry = {"archive": {"sha256": file_hash(archive), "file_count": 1, "expanded_file_bytes": 0}}
    with pytest.raises(ValueError):
        unpack_archive(archive, tmp_path / "output", entry)
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "output").exists()


def test_validation_alias_preserves_submission_protocol(frozen, tmp_path):
    root, _, _ = frozen
    public = root / "public"
    a = next(iter_observations(public, "validation"))
    b = next(iter_observations(public, "val"))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(load_targets(public, "validation")["theta"], load_targets(public, "val")["theta"])
    run(public, tmp_path / "prediction", "xrdbench.api:PriorMean", "point", split="validation")
    result, _, _ = validate_submission(public, tmp_path / "prediction")
    assert result["split"] == "val"
