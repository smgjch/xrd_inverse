from pathlib import Path
import shutil

import numpy as np
import pytest

from xrdbench.api import make_batch, sample_seeds, PriorSampler, make_context
from xrdbench.data import audit, generate
from xrdbench.evaluate import compare, score
from xrdbench.io import iter_observations, load_npz, manifest, read_json, save_npz, write_json, file_hash
from xrdbench.reference import grid_reference
from xrdbench.submission import SubmissionWriter, run, validate_submission
from xrdbench.protocol import digest, rng_seed, snapshot


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench") / "release"
    generate(root, "rutile", "calibrated", "smoke")
    return root


def test_data_audit_no_label_input_and_determinism(dataset, tmp_path):
    result = audit(dataset / "public", dataset / "private")
    assert result["passed"]
    assert result["splits"]["train"]["structures"] == 32
    assert result["splits"]["test"]["observations"] == 88
    assert not list((dataset / "public" / "test").glob("*targets*"))
    batch = make_batch(next(iter_observations(dataset / "public", "test")))
    assert not hasattr(batch, "theta") and not hasattr(batch, "condition")
    assert not hasattr(batch, "group_ids")
    # Rebuilding in a fresh directory must produce the exact same release fingerprint.
    rebuilt = generate(tmp_path / "same", "rutile", "calibrated", "smoke")
    assert rebuilt["dataset_id"] == result["dataset_id"]


def test_prior_full_pipeline_and_physics(dataset, tmp_path):
    public, private = dataset / "public", dataset / "private"
    run(public, tmp_path / "pred", "xrdbench.api:PriorMean", "point", batch_size=7)
    result = score(public, tmp_path / "pred", tmp_path / "score", private, physics_k=1, physics_limit=1)
    assert result["observations"] == 88
    assert result["physics_selection"]["n_scored"] == 11
    assert "energy_score" not in result["primary_id_macro"]
    self_comparison = compare([tmp_path / "score"], [tmp_path / "score"])
    assert self_comparison["mean"] == 0 and self_comparison["ci95"] == [0, 0]
    run(public, tmp_path / "samples", "xrdbench.api:PriorSampler", "posterior", batch_size=9)
    sample_score = score(public, tmp_path / "samples", tmp_path / "posterior_score", private)
    assert "energy_score" in sample_score["primary_id_macro"]
    assert np.isfinite(sample_score["primary_id_macro"]["energy_score"])
    report = read_json(tmp_path / "posterior_score" / "metrics.json")
    assert report["point_rule"] == "sample_marginal_median"


def test_prior_draws_batch_invariant(dataset):
    data = manifest(dataset / "public")
    arrays = next(iter_observations(dataset / "public", "test"))
    model = PriorSampler(make_context(data))
    full = make_batch(arrays)
    both = model.predict(full, 256, sample_seeds(full, 10))["samples"]
    single = make_batch(arrays, slice(2, 3))
    only = model.predict(single, 256, sample_seeds(single, 10))["samples"]
    np.testing.assert_array_equal(only[0], both[2])


def test_mutated_ids_wrong_dataset_and_missing_rows_rejected(dataset, tmp_path):
    public = dataset / "public"
    run(public, tmp_path / "pred", "xrdbench.api:PriorMean", "point")
    pred_meta = read_json(tmp_path / "pred" / "manifest.json")
    file = tmp_path / "pred" / pred_meta["files"][0]["path"]
    raw = load_npz(file)
    raw["ids"][1] = raw["ids"][0]
    save_npz(file, **raw)
    pred_meta["files"][0]["sha256"] = file_hash(file)
    write_json(tmp_path / "pred" / "manifest.json", pred_meta)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_submission(public, tmp_path / "pred")
    pred_meta["dataset_id"] = "wrong-dataset"
    write_json(tmp_path / "pred" / "manifest.json", pred_meta)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_submission(public, tmp_path / "pred")
    # A well-formed partial export is still invalid.
    writer = SubmissionWriter(tmp_path / "partial", manifest(public), "test", "point", {"method_name": "partial"})
    batch = next(iter_observations(public, "test"))
    writer.add(batch["ids"][:1], {"point": np.ones((1, 2))})
    writer.finish()
    with pytest.raises(ValueError, match="IDs"):
        validate_submission(public, tmp_path / "partial")


def test_byte_tampering_and_false_posterior_rejected(dataset, tmp_path):
    public = tmp_path / "public"
    shutil.copytree(dataset / "public", public)
    record = manifest(public)["splits"]["test"]["observations"][0]
    with open(public / record["path"], "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        list(iter_observations(public, "test"))
    with pytest.raises(ValueError, match="refined"):
        SubmissionWriter(tmp_path / "bad_posterior", manifest(dataset / "public"), "test", "posterior",
                         {"method_name": "bad", "postprocessing": "local_refinement"})


def test_no_signal_grid_is_prior_without_labels(dataset, tmp_path):
    batch = next(iter_observations(dataset / "public", "test"))
    sample_id = batch["ids"][batch["condition"] == "no_signal_stress"][0]
    grid = grid_reference(dataset / "public", tmp_path / "grid", sample_id=str(sample_id), points=17)
    ref = manifest(dataset / "public")["config"]["task"]["reference"]
    np.testing.assert_allclose(grid["mean"], ref, atol=1e-12)
    assert grid["effective_grid_cells"] == pytest.approx(17**2)
    assert grid["certified_reference"] is False


def test_blind_track_does_not_allow_oracle_physics(tmp_path):
    root = tmp_path / "blind"
    generate(root, "si", "blind", "smoke")
    assert audit(root / "public", root / "private")["passed"]
    run(root / "public", tmp_path / "pred", "xrdbench.api:PriorSampler", "posterior")
    with pytest.raises(ValueError, match="no oracle"):
        score(root / "public", tmp_path / "pred", tmp_path / "score", root / "private", physics_k=8)


def test_physics_view_selects_same_128_structures_across_conditions(tmp_path):
    # A synthetic metadata fixture tests selection beyond 128 without costly simulation.
    public = tmp_path / "public"
    groups = np.repeat([f"g{i:04d}" for i in range(130)], 2)
    ids = np.asarray([f"row{i:04d}" for i in range(260)])
    path = public / "test/observations.npz"
    save_npz(path, ids=ids, group_ids=groups, condition=np.tile(["full_high", "block30"], 130))
    meta = {"config": snapshot("si", "calibrated", "smoke"), "splits": {"test": {"observations": [
        {"path": "test/observations.npz", "sha256": file_hash(path), "rows": 260}]}}}
    meta["dataset_id"] = digest(meta)
    write_json(public / "manifest.json", meta)
    view = next(iter_observations(public, "test", "physics128"))
    expected = set(sorted(set(groups), key=lambda g: rng_seed("physics-view-v1", g))[:128])
    assert len(view["ids"]) == 256 and set(view["group_ids"]) == expected
    assert set(view["group_ids"][view["condition"] == "full_high"]) == set(view["group_ids"][view["condition"] == "block30"])
