from __future__ import annotations

import numpy as np

from .protocol import KS, LEVELS, THRESHOLDS, rng_seed


def point_metrics(point, truth, lower, upper):
    error = np.asarray(point) - truth
    relative = np.abs(error) / np.abs(truth)
    width = np.asarray(upper) - lower
    out = {"mae_angstrom": np.abs(error), "squared_error_angstrom": error**2,
           "absolute_relative_error": relative,
           "normalized_mae": np.mean(np.abs(error) / width, axis=-1),
           "max_relative_error": relative.max(axis=-1),
           "out_of_prior": np.any((point < lower) | (point > upper), axis=-1).astype(float)}
    for threshold in THRESHOLDS:
        out[f"success_rel_{threshold}"] = (relative.max(axis=-1) <= threshold).astype(float)
    return out


def posterior_metrics(samples, truth, lower, upper, ids):
    n, s, d = samples.shape
    if s % 2:
        raise ValueError("Even sample count required for the independent-pair energy estimator")
    width = np.asarray(upper) - lower
    z = (samples - lower) / width
    target = (truth - lower) / width
    # O(S), unbiased Monte Carlo estimator for iid posterior samples; no biased V-statistic.
    energy = np.linalg.norm(z - target[:, None], axis=-1).mean(axis=1)
    energy -= 0.5 * np.linalg.norm(z[:, :s // 2] - z[:, s // 2:], axis=-1).mean(axis=1)
    out = {"energy_score": energy,
           "sample_out_of_prior_fraction": np.any((samples < lower) | (samples > upper), axis=-1).mean(axis=1)}
    for level in LEVELS:
        low, high = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=1)
        covered = (truth >= low) & (truth <= high)
        out[f"marginal_coverage_{level}"] = covered.astype(float)
        out[f"interval_width_angstrom_{level}"] = high - low
        out[f"interval_width_normalized_{level}"] = (high - low) / width
        out[f"rectangle_coverage_{level}"] = covered.all(axis=-1).astype(float)
    below = (samples < truth[:, None]).sum(axis=1)
    equal = (samples == truth[:, None]).sum(axis=1)
    ranks = below.copy()
    for i, sample_id in enumerate(ids):
        ranks[i] += np.random.default_rng(rng_seed("rank-ties", str(sample_id))).integers(0, equal[i] + 1)
    out["sbc_rank"] = ranks
    return out


def oracle_metrics(candidates, truth):
    out = {}
    relative = np.max(np.abs(candidates - truth[:, None]) / np.abs(truth[:, None]), axis=-1)
    for k in KS:
        if k > candidates.shape[1]:
            continue
        best = np.min(relative[:, :k], axis=1)
        out[f"oracle_best_max_relative_error_at_{k}"] = best
        for threshold in THRESHOLDS:
            out[f"oracle_success_rel_{threshold}_at_{k}"] = (best <= threshold).astype(float)
    return out


def grouped_means(values, groups):
    groups = np.asarray(groups)
    _, index = np.unique(groups, return_inverse=True)
    count = np.bincount(index)
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Bootstrap expects one scalar metric per observation")
    return np.bincount(index, weights=values) / count


def cluster_ci(values, groups, repeats=2000, seed=0):
    values = grouped_means(values, groups)
    rng = np.random.default_rng(seed)
    results = []
    for start in range(0, repeats, 100):
        indices = rng.integers(0, len(values), size=(min(100, repeats - start), len(values)))
        results.extend(values[indices].mean(axis=1))
    return {"mean": float(values.mean()), "ci95": np.quantile(results, [0.025, 0.975]).tolist(),
            "n_independent_structures": len(values), "bootstrap_repeats": repeats}


def summarize(metrics, selection, groups):
    summary = {"n_observations": int(np.sum(selection)), "n_structures": len(set(groups[selection].tolist()))}
    for key, values in metrics.items():
        selected = values[selection]
        if key == "sbc_rank":
            s = 256
            bins = np.minimum((selected * 10 // (s + 1)).astype(int), 9)
            summary["rank_histogram_10bins"] = [np.bincount(bins[:, j], minlength=10).tolist() for j in range(bins.shape[1])]
            reference_bins = np.arange(s + 1) * 10 // (s + 1)
            summary["rank_histogram_expected_counts"] = (np.bincount(reference_bins, minlength=10) / (s + 1) * len(selected)).tolist()
        elif key == "squared_error_angstrom":
            summary["rmse_angstrom"] = np.sqrt(selected.mean(axis=0)).tolist()
        else:
            mean = selected.mean(axis=0)
            summary[key] = mean.tolist()
    if "max_relative_error" in metrics:
        summary["p95_max_relative_error"] = float(np.quantile(metrics["max_relative_error"][selection], 0.95))
    for key in ("normalized_mae", "energy_score"):
        if key in metrics:
            summary[key + "_cluster_ci95"] = cluster_ci(metrics[key][selection], groups[selection])
    return summary
