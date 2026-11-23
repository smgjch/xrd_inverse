import numpy as np
import pytest

from xrdbench.data import observation_mask, simulate_observation
from xrdbench.metrics import cluster_ci, oracle_metrics, point_metrics, posterior_metrics
from xrdbench.physics import candidate_scores, expected_counts, peaks, poisson_deviance, poisson_nll, structure
from xrdbench.protocol import AXIS, CALIBRATED, task_spec
from xrdbench.submission import validate_outputs


@pytest.mark.parametrize("task,n_atoms,first_peak", [("si", 8, 28.44), ("rutile", 6, 27.43), ("zno", 4, 31.77)])
def test_standard_prototypes(task, n_atoms, first_peak):
    theta = task_spec(task)["reference"]
    crystal = structure(task, theta)
    centers, intensity = peaks(task, theta)
    assert len(crystal) == n_atoms
    assert centers[0] == pytest.approx(first_peak, abs=0.07)
    assert intensity.max() == pytest.approx(1)
    # A uniform lattice expansion must move the same first reflection to lower angle.
    larger, _ = peaks(task, np.asarray(theta) * 1.01)
    assert larger[0] < centers[0]


def test_mask_independent_of_lattice_and_hidden_dropout_is_observed():
    lo, hi = task_spec("rutile")["lower"], task_spec("rutile")["upper"]
    a = simulate_observation("rutile", lo, "blind", "block30", 18)
    b = simulate_observation("rutile", hi, "blind", "block30", 18)
    assert np.array_equal(a["mask"], b["mask"])
    assert (~a["mask"]).sum() == round(0.3 * len(AXIS))
    assert np.isnan(a["nuisance"]).all() and not a["nuisance_known"]
    drop = simulate_observation("rutile", lo, "calibrated", "drop30_ood", 18)
    assert drop["mask"].all()
    assert (a["counts"][~a["mask"]] == 0).all()


def test_poisson_mask_and_scale():
    y, mu = np.array([2., 999.]), np.array([2., 0.1])
    assert poisson_deviance(y, mu, [True, False]) == pytest.approx(0)
    assert poisson_nll(y, mu, [True, False]) == pytest.approx(2 - 2 * np.log(2))
    assert poisson_nll(y, mu, [False, False]) == 0
    assert poisson_deviance(np.zeros(2), mu, [True, True]) == pytest.approx(2 * mu.sum())
    rate1 = expected_counts("si", [5.431], CALIBRATED, 1000)
    rate2 = expected_counts("si", [5.431], CALIBRATED, 2000)
    np.testing.assert_allclose(rate2, rate1 * 2)


def test_observation_only_physics_ranking_recovers_matching_lattice():
    theta = np.asarray(task_spec("rutile")["reference"])
    observed = np.round(expected_counts("rutile", theta, CALIBRATED, 10000)).astype(np.uint32)
    candidates = np.stack([theta * 1.01, theta, theta * 0.99])
    mask = observation_mask("window25_45", 0)
    losses = candidate_scores("rutile", candidates, observed, mask, CALIBRATED, 10000)
    assert int(np.argmin(losses)) == 1
    # Unobserved garbage cannot change which candidate the real scorer selects.
    observed[~mask] = 123456
    np.testing.assert_array_equal(candidate_scores("rutile", candidates, observed, mask, CALIBRATED, 10000), losses)


def test_energy_score_known_prior_predictive_expectation_and_calibration():
    # Uniform(0,1): E|X-Y|=1/3, E|X-X'|=1/3, expected energy score=1/6.
    rng = np.random.default_rng(7)
    samples = rng.uniform(size=(2000, 256, 1))
    truth = rng.uniform(size=(2000, 1))
    result = posterior_metrics(samples, truth, np.array([0]), np.array([1]), np.arange(2000).astype(str))
    assert result["energy_score"].mean() == pytest.approx(1 / 6, abs=0.01)
    assert result["marginal_coverage_0.9"].mean() == pytest.approx(0.9, abs=0.03)
    assert result["interval_width_normalized_0.9"].mean() == pytest.approx(0.9, abs=0.015)
    bins = result["sbc_rank"][:, 0] * 10 // 257
    assert np.max(np.abs(np.bincount(bins, minlength=10) - 200)) < 55


def test_point_errors_oracle_prefixes_and_cluster_bootstrap():
    truth = np.array([[5.0, 3.0], [5.0, 3.0]])
    point = np.array([[5.1, 3.0], [5.0, 3.0]])
    m = point_metrics(point, truth, [4.5, 2.5], [5.5, 3.5])
    np.testing.assert_allclose(m["mae_angstrom"], [[0.1, 0], [0, 0]], atol=1e-12)
    np.testing.assert_array_equal(m["success_rel_0.005"], [0, 1])
    candidates = np.repeat(point[:, None], 64, axis=1)
    candidates[:, 7] = truth
    metrics = oracle_metrics(candidates, truth)
    np.testing.assert_array_equal(metrics["oracle_success_rel_0.005_at_8"], [1, 1])
    assert np.all(metrics["oracle_success_rel_0.005_at_8"] >= metrics["oracle_success_rel_0.005_at_1"])
    # Replicating corruption rows of a structure must not change the statistical unit.
    ci1 = cluster_ci(np.array([0., 1.]), np.array(["a", "b"]), repeats=200)
    ci2 = cluster_ci(np.array([0., 0., 0., 1.]), np.array(["a", "a", "a", "b"]), repeats=200)
    assert ci1 == ci2


def test_nonfinite_and_fake_samples_rejected():
    with pytest.raises(ValueError):
        validate_outputs({"point": np.array([[np.nan]])}, 1, 1, "point")
    with pytest.raises(ValueError):
        validate_outputs({"samples": np.zeros((1, 64, 1))}, 1, 1, "posterior")
    with pytest.raises(ValueError):
        validate_outputs({"point": np.ones((1, 1)), "truth": np.ones((1, 1))}, 1, 1, "point")
    with pytest.raises(ValueError):
        validate_outputs({"point": np.ones((1, 1)), "candidates": np.zeros((1, 64, 1))}, 1, 1, "candidates")
