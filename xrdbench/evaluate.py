from __future__ import annotations

import csv
from pathlib import Path
import time

import numpy as np

from .io import (file_hash, iter_observations, load_npz, load_targets, manifest,
                 read_json, save_npz, write_json)
from .metrics import cluster_ci, oracle_metrics, point_metrics, posterior_metrics, summarize
from .physics import candidate_scores, expected_counts, poisson_deviance
from .protocol import CONDITIONS, ID_CONDITIONS, VERSION, rng_seed
from .submission import validate_submission


def score(public, submission, output, private=None, physics_k=0, physics_limit=128):
    if physics_k not in (0, 1, 8, 32, 64) or physics_limit < 1:
        raise ValueError("physics_k must be 0/1/8/32/64 and limit must be positive")
    validation, prediction, pred_meta = validate_submission(public, submission)
    data = manifest(public)
    cfg = data["config"]
    if physics_k and cfg["track"] != "calibrated":
        raise ValueError("Blind nuisance integration/profiling is not implemented; no oracle nuisance fallback")
    labels = load_targets(public, pred_meta["split"], private)
    if pred_meta["view"] != "full":
        selected = np.isin(labels["ids"], prediction["ids"])
        labels = {k: v[selected] for k, v in labels.items()}
    if not np.array_equal(labels["ids"], prediction["ids"]):
        raise ValueError("Label/prediction ID mismatch")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    lo, hi = np.asarray(cfg["task"]["lower"]), np.asarray(cfg["task"]["upper"])
    posterior = pred_meta["kind"] == "posterior"
    point = np.median(prediction["samples"], axis=1) if posterior else prediction["point"]
    metrics = point_metrics(point, labels["theta"], lo, hi)
    metrics["failure_rate"] = prediction["failed"].astype(float)
    candidates = prediction.get("samples", prediction.get("candidates", point[:, None]))
    metrics.update(oracle_metrics(candidates, labels["theta"]))
    if posterior:
        metrics.update(posterior_metrics(prediction["samples"], labels["theta"], lo, hi, labels["ids"]))
    conditions, groups = labels["condition"], labels["group_ids"]
    by_condition = {condition: summarize(metrics, conditions == condition, groups) for condition in sorted(set(conditions.tolist()))}
    report = {
        "protocol": cfg["protocol"], "suite": cfg.get("suite", "v0.1"),
        "task": cfg["task"]["task"],
        "id_conditions": cfg["training_conditions"],
        "dataset_id": data["dataset_id"], "submission_sha256": file_hash(Path(submission) / "manifest.json"),
        "validation": validation, "track": cfg["track"], "preset": cfg["preset"],
        "method_metadata": pred_meta["metadata"], "parameter_order": cfg["task"]["parameters"],
        "point_rule": "sample_marginal_median" if posterior else pred_meta["metadata"].get("point_rule", "declared_method_point"),
        "primary_id_macro": summarize(metrics, np.isin(conditions, cfg["training_conditions"]), groups),
        "by_domain": {domain: summarize(metrics, np.isin(conditions, names), groups)
                      for domain, names in cfg.get("evaluation_domains", {"v0.1": ID_CONDITIONS}).items()},
        "by_condition": by_condition, "inference_timing": pred_meta["timing"],
        "interpretation": {
            "main_table_eligible_size": cfg["preset"] == "main",
            "training_and_external_data_eligibility": "Coordinator must audit method_metadata; runner cannot verify training provenance",
            "oracle": "Uses true labels to select the best candidate; not deployment top-1",
            "rank_histograms": "SBC interpretation for prior-predictive ID only; OOD/stress ranks are diagnostics (no_signal has exact prior reference)",
            "rectangle_coverage": "Intersection of marginal intervals; NOT a nominal-alpha joint credible region",
            "ci": "Structure-cluster bootstrap conditional on trained model; not training-seed uncertainty",
            "energy": "Normalized prior-width units; unbiased independent-pair sample estimator can be negative by Monte Carlo noise",
        },
        "physics_selection": {"enabled": False},
    }
    save_npz(out / "per_sample_metrics.npz", ids=labels["ids"], group_ids=groups, condition=conditions, **metrics)
    if physics_k:
        report["physics_selection"] = physics_evaluation(public, pred_meta["split"], prediction, labels,
                                                         cfg, physics_k, physics_limit, out)
    write_json(out / "metrics.json", report)
    columns = ["condition", "regime", "n_structures", "normalized_mae", "success_rel_0.005", "p95_max_relative_error",
               "energy_score", "marginal_coverage_0.9", "interval_width_normalized_0.9", "failure_rate"]
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for condition, values in by_condition.items():
            row = {k: values.get(k, "") for k in columns}
            row.update(condition=condition, regime=cfg["conditions"][condition]["regime"])
            writer.writerow(row)
    return {"output": str(out), "method": validation["method"], "kind": pred_meta["kind"],
            "observations": len(point), "primary_id_macro": report["primary_id_macro"],
            "physics_selection": report["physics_selection"]}


def physics_evaluation(public, split, prediction, labels, cfg, k, limit, output):
    points = prediction.get("samples", prediction.get("candidates"))
    if points is None:
        points = prediction["point"][:, None]
    k = min(k, points.shape[1])
    subset_groups = sorted(set(labels["group_ids"].tolist()), key=lambda g: rng_seed("physics-view-v1", g))[:limit]
    selected = set(labels["ids"][np.isin(labels["group_ids"], subset_groups)].tolist())
    lookup = {sample_id: index for index, sample_id in enumerate(labels["ids"].tolist())}
    lo, hi = np.asarray(cfg["task"]["lower"]), np.asarray(cfg["task"]["upper"])
    chosen, indices, failed, deviances = [], [], [], []
    start = time.perf_counter()
    forward_calls = 0
    for batch in iter_observations(public, split):
        for row, sample_id in enumerate(batch["ids"].tolist()):
            if sample_id not in selected:
                continue
            i = lookup[sample_id]
            candidates = points[i, :k]
            valid = np.all((candidates >= lo) & (candidates <= hi), axis=-1)
            losses = np.full(k, np.inf)
            # No access to labels or hidden nuisance when scoring/choosing candidates.
            losses[valid] = candidate_scores(cfg["task"]["task"], candidates[valid], batch["counts"][row],
                batch["mask"][row], batch["nuisance"][row], batch["exposure"][row])
            forward_calls += int(valid.sum())
            is_failed = not np.isfinite(losses).any()
            theta = (lo + hi) / 2 if is_failed else candidates[int(np.argmin(losses))]
            chosen.append(theta); indices.append(i); failed.append(is_failed)
            # Count every row: failed selection gets the preregistered prior-mean fallback.
            mean = expected_counts(cfg["task"]["task"], theta, batch["nuisance"][row], batch["exposure"][row])
            forward_calls += 1
            observed_bins = int(batch["mask"][row].sum())
            deviance = poisson_deviance(batch["counts"][row], mean, batch["mask"][row])
            deviances.append(deviance / observed_bins if observed_bins else 0.0)
    elapsed = time.perf_counter() - start
    indices, chosen = np.asarray(indices), np.asarray(chosen)
    measures = point_metrics(chosen, labels["theta"][indices], lo, hi)
    measures["selection_failure_rate"] = np.asarray(failed, dtype=float)
    measures["poisson_deviance_per_observed_bin"] = np.asarray(deviances)
    save_npz(output / "physics_selected_metrics.npz", ids=labels["ids"][indices], group_ids=labels["group_ids"][indices],
             condition=labels["condition"][indices], selected_point=chosen, **measures)
    groups, conditions = labels["group_ids"][indices], labels["condition"][indices]
    return {"enabled": True, "k": k, "limit_per_condition": limit, "selection_rule": "min_masked_Poisson_NLL_with_uniform_bounded_prior",
            "n_scored": len(indices), "forward_calls": forward_calls, "coordinator_wall_seconds": elapsed,
            "note": "Raw posterior metrics unchanged. Nominal Gaussian scorer is deliberately misspecified on Voigt/dropout OOD; no-signal deviance convention=0.",
            "by_condition": {c: summarize(measures, conditions == c, groups) for c in sorted(set(conditions.tolist()))}}


def compare(a_paths, b_paths, metric="normalized_mae", condition="id", repeats=2000):
    """Average independent training seeds first, then paired bootstrap by structure."""
    if metric not in ("normalized_mae", "energy_score", "max_relative_error", "success_rel_0.001", "success_rel_0.005"):
        raise ValueError("Choose a supported scalar metric")
    all_scores = []
    ref = None
    dataset_id, split, view, id_conditions = None, None, None, None
    for paths in (a_paths, b_paths):
        series = []
        for path in paths:
            report = read_json(Path(path) / "metrics.json")
            raw = load_npz(Path(path) / "per_sample_metrics.npz")
            if ref is None:
                ref = raw; dataset_id = report["dataset_id"]; split = report["validation"]["split"]; view = report["validation"]["view"]
                id_conditions = report.get("id_conditions", ID_CONDITIONS)
            if report["dataset_id"] != dataset_id or report["validation"]["split"] != split or report["validation"]["view"] != view:
                raise ValueError("Paired comparison requires identical dataset, split and view")
            if any(not np.array_equal(raw[key], ref[key]) for key in ("ids", "group_ids", "condition")):
                raise ValueError("Paired comparison requires identical sample/group/condition IDs")
            if metric not in raw:
                raise ValueError(f"{metric} is unavailable (point models have no posterior score)")
            series.append(raw[metric])
        all_scores.append(np.stack(series))
    if condition in ("id", "v0.1", "v0.2", "v0.1+v0.2"):
        names = list(id_conditions)
        if condition == "v0.1":
            names = [c for c in names if not c.startswith("hard_")]
        elif condition == "v0.2":
            names = [c for c in names if c.startswith("hard_")]
        elif condition == "v0.1+v0.2" and report.get("suite") != "v0.1+v0.2":
            raise ValueError("Combined comparison requires a union release")
        selection = np.isin(ref["condition"], names)
    else:
        selection = ref["condition"] == condition
    if not selection.any():
        raise ValueError("Empty condition selection")
    delta = all_scores[0].mean(axis=0) - all_scores[1].mean(axis=0)
    result = cluster_ci(delta[selection], ref["group_ids"][selection], repeats=repeats)
    result.update(dataset_id=dataset_id, split=split, view=view, metric=metric, condition=condition,
                  difference="A minus B", lower_is_better=not metric.startswith("success_"),
                  seed_counts=[len(x) for x in all_scores],
                  per_seed_means=[[float(v[selection].mean()) for v in x] for x in all_scores],
                  uncertainty="CI resamples structures after averaging runs; training variation is shown separately, not combined")
    return result
