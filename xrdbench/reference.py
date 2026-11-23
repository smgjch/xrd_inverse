"""Coarse numerical posterior diagnostic; certification requires grid convergence."""
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .io import iter_observations, manifest, save_npz, write_json
from .physics import candidate_scores


def grid_reference(public, output, sample_id=None, split="test", points=65):
    meta = manifest(public)
    cfg = meta["config"]
    if cfg["track"] != "calibrated":
        raise ValueError("Grid diagnostic only supports calibrated track; no oracle nuisance")
    d = len(cfg["task"]["parameters"])
    if points < 2 or points**d > 20000:
        raise ValueError("Choose 2 <= points with at most 20000 grid cells for this diagnostic")
    obs = None
    for batch in iter_observations(public, split):
        for i, sid in enumerate(batch["ids"]):
            if sample_id is None or sid == sample_id:
                obs = {k: v[i] for k, v in batch.items()}; break
        if obs is not None:
            break
    if obs is None:
        raise ValueError("Unknown sample ID")
    lo, hi = np.asarray(cfg["task"]["lower"]), np.asarray(cfg["task"]["upper"])
    axes = [a + (np.arange(points) + 0.5) * (b - a) / points for a, b in zip(lo, hi)]
    grid = np.asarray(list(product(*axes)))
    if not obs["mask"].any():
        logw = np.full(len(grid), -np.log(len(grid)))
    else:
        loss = candidate_scores(cfg["task"]["task"], grid, obs["counts"], obs["mask"], obs["nuisance"], obs["exposure"])
        logw = -loss - logsumexp(-loss)
    weights = np.exp(logw)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    save_npz(out / "grid.npz", theta=grid, log_weights=logw)
    result = {"dataset_id": meta["dataset_id"], "sample_id": str(obs["ids"]), "points_per_dimension": points,
              "mean": (weights[:, None] * grid).sum(axis=0).tolist(), "maximum_weight": float(weights.max()),
              "effective_grid_cells": float(1 / np.sum(weights**2)), "certified_reference": False,
              "warning": "Coarse-grid diagnostic only. Compare denser grids / adaptive integration and all modes before treating as reference; profile likelihood is not marginalized posterior."}
    write_json(out / "diagnostic.json", result)
    return result
