import numpy as np
import pytest

from xrdbench.data import generate, simulate_observation
from xrdbench.protocol import conditions_for, observation_schedule, TASKS, snapshot, HARD_PRIOR, BLIND_PRIOR
from xrdbench.io import iter_observations, load_targets, manifest, read_json, write_json
from xrdbench.submission import run, SubmissionWriter
from xrdbench.evaluate import score, compare
from xrdbench.reporting import summarize_scores


def test_hard_conditions_pair_instrument_visibility():
    for task in ("rutile", "zno"):
        for condition, cfg in conditions_for(task, "v0.2").items():
            known = simulate_observation(task, TASKS[task]["reference"], "calibrated", condition, 123, cfg=cfg)
            blind = simulate_observation(task, TASKS[task]["reference"], "blind", condition, 123, cfg=cfg)
            np.testing.assert_array_equal(known["counts"], blind["counts"])
            np.testing.assert_array_equal(known["mask"], blind["mask"])
            assert np.isfinite(known["nuisance"]).all() and np.isnan(blind["nuisance"]).all()
            assert not known["mask"].all() and known["mask"].any()
            assert (blind["counts"][~blind["mask"]] == 0).all()
    with pytest.raises(ValueError, match="Si"):
        conditions_for("si", "v0.2")


def test_union_schedule_fixed_budget_and_component_identity():
    for split in ("train", "val", "test"):
        group = f"rutile:{split}:00000001"
        a = observation_schedule("rutile", "v0.1", split, group)
        b = observation_schedule("rutile", "v0.2", split, group)
        ab = observation_schedule("rutile", "v0.1+v0.2", split, group)
        assert ab == ([a[0], b[1]] if split == "train" else a + b)
        if split == "train":
            assert len(a) == len(b) == len(ab) == 2


def test_manifest_nuisance_priors_are_unambiguous():
    old = snapshot("rutile", "blind", "pilot", "v0.1")
    hard = snapshot("rutile", "blind", "pilot", "v0.2")
    union = snapshot("rutile", "blind", "pilot", "v0.1+v0.2")
    assert old["blind_nuisance_prior"] == BLIND_PRIOR
    assert hard["calibrated_nuisance"] is None
    assert hard["blind_nuisance_prior"] == HARD_PRIOR
    assert union["calibrated_nuisance"] is None and union["blind_nuisance_prior"] is None
    assert union["nuisance_priors_by_domain"] == {"v0.1": BLIND_PRIOR, "v0.2": HARD_PRIOR}


def test_hard_scoring_and_union_weights(tmp_path):
    root = tmp_path / "union"
    generate(root, "rutile", "blind", "smoke", suite="v0.1+v0.2")
    cfg = manifest(root / "public")["config"]
    assert len(cfg["training_conditions"]) == 12
    labels = load_targets(root / "public", "test", root / "private")
    assert len(labels["ids"]) == 8 * 17
    run(root / "public", tmp_path / "pred", "xrdbench.api:PriorMean", "point")
    result = score(root / "public", tmp_path / "pred", tmp_path / "scored", root / "private")
    assert result["primary_id_macro"]["n_observations"] == 8 * 12
    assert compare([tmp_path / "scored"], [tmp_path / "scored"])["mean"] == 0
    hard = tmp_path / "hard"
    generate(hard, "rutile", "blind", "smoke", suite="v0.2")
    union_rows = {str(sid): counts for b in iter_observations(root / "public", "test") for sid, counts in zip(b["ids"], b["counts"])}
    for batch in iter_observations(hard / "public", "test"):
        for sid, counts in zip(batch["ids"], batch["counts"]):
            np.testing.assert_array_equal(counts, union_rows[str(sid)])


def test_reporting_weights_exclude_stress_and_guard_duplicates(tmp_path):
    root = tmp_path / "union"
    generate(root, "rutile", "blind", "smoke", suite="v0.1+v0.2")
    meta = manifest(root / "public")
    labels = load_targets(root / "public", "test", root / "private")
    old = np.isin(labels["condition"], meta["config"]["evaluation_domains"]["v0.1"])
    hard = np.isin(labels["condition"], meta["config"]["evaluation_domains"]["v0.2"])
    width = np.asarray(meta["config"]["task"]["upper"]) - meta["config"]["task"]["lower"]
    # Deliberately label-dependent fixture, never exported as a scientific method.
    point = labels["theta"].copy()
    point[hard] += .05 * width
    point[~(old | hard)] += 100
    writer = SubmissionWriter(tmp_path / "prediction", meta, "test", "point",
                              {"method_name": "unit_test_only", "training_suite": "v0.2", "training_seed": 0})
    writer.add(labels["ids"], {"point": point}); writer.finish()
    score(root / "public", tmp_path / "prediction", tmp_path / "score", root / "private")
    report = read_json(tmp_path / "score/metrics.json")
    assert report["by_domain"]["v0.1"]["normalized_mae"] == 0
    assert report["by_domain"]["v0.2"]["normalized_mae"] == pytest.approx(.05)
    assert report["primary_id_macro"]["normalized_mae"] == pytest.approx(.025)
    result = summarize_scores([tmp_path / "score"], tmp_path / "report")
    assert result["matrix_rows"] == 3 and result["training_performed"] is False
    rows = read_json(tmp_path / "report/results.json")["rows"]
    assert rows[2]["mean_relative_error_percent"] == pytest.approx(
        .5 * (rows[0]["mean_relative_error_percent"] + rows[1]["mean_relative_error_percent"]))
    assert rows[2]["n_observations"] == 96
    for domain in ("v0.1", "v0.2", "v0.1+v0.2"):
        assert compare([tmp_path / "score"], [tmp_path / "score"], condition=domain)["mean"] == 0
    assert compare([tmp_path / "score"], [tmp_path / "score"], metric="success_rel_0.001")["lower_is_better"] is False
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_scores([tmp_path / "score", tmp_path / "score"], tmp_path / "duplicate")
    report["validation"]["split"] = "val"
    write_json(tmp_path / "score/metrics.json", report)
    with pytest.raises(ValueError, match="held-out test"):
        summarize_scores([tmp_path / "score"], tmp_path / "wrong_split")
