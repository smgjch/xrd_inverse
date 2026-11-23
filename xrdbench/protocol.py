"""Frozen protocol values. Any change requires a new protocol/dataset version."""
from __future__ import annotations

import hashlib
import json

import numpy as np

VERSION = "xrdbench-0.2.0"
SEED = 20261123
WAVELENGTH = 1.5406  # Angstrom; monochromatic, not a Ka1/Ka2 doublet.
AXIS = np.linspace(10.0, 90.0, 4096, dtype=np.float64)
TASKS = {
    "si": {"parameters": ["a"], "reference": [5.431], "crystal_system": "cubic", "space_group": 227},
    "rutile": {"parameters": ["a", "c"], "reference": [4.5937, 2.9587], "crystal_system": "tetragonal", "space_group": 136, "oxygen_u": 0.305},
    "zno": {"parameters": ["a", "c"], "reference": [3.2495, 5.2069], "crystal_system": "hexagonal", "space_group": 186, "oxygen_u": 0.3825},
}
SIZES = {"smoke": (32, 8, 8), "pilot": (5000, 512, 512), "main": (100000, 10000, 5000)}
CONDITIONS = {
    "full_high": {"regime": "id", "exposure": 100000.0, "mask": "full"},
    "full_standard": {"regime": "id", "exposure": 10000.0, "mask": "full"},
    "full_low": {"regime": "id", "exposure": 1000.0, "mask": "full"},
    "block30": {"regime": "id", "exposure": 10000.0, "mask": "block", "fraction": 0.3},
    "block50": {"regime": "id", "exposure": 10000.0, "mask": "block", "fraction": 0.5},
    "window25_45": {"regime": "id", "exposure": 10000.0, "mask": "window", "window": [25.0, 45.0]},
    "broad_ood": {"regime": "ood", "exposure": 10000.0, "mask": "full", "width_range": [0.4, 0.7]},
    "voigt_ood": {"regime": "ood", "exposure": 10000.0, "mask": "full", "profile": "pseudo_voigt", "mixing": 0.5},
    "drop30_ood": {"regime": "ood", "exposure": 10000.0, "mask": "full", "drop_probability": 0.3},
    "window26_30_stress": {"regime": "stress", "exposure": 10000.0, "mask": "window", "window": [26.0, 30.0]},
    "no_signal_stress": {"regime": "stress", "exposure": 10000.0, "mask": "none"},
}
ID_CONDITIONS = [name for name, cfg in CONDITIONS.items() if cfg["regime"] == "id"]
SUITES = ("v0.1", "v0.2", "v0.1+v0.2")
# Fixed absolute windows: never centered on a hidden true peak.
HARD_WINDOWS = {
    "rutile": {"mixed": [[33., 39.]], "paired": [[33., 39.], [52., 57.]]},
    "zno": {"mixed": [[45., 50.]], "paired": [[33., 37.], [45., 50.]]},
}
HARD_PRIOR = {
    "fwhm_deg": {"distribution": "uniform", "range": [0.15, 0.60]},
    "zero_shift_deg": {"distribution": "uniform", "range": [-0.10, 0.10]},
    "scale": {"distribution": "log_uniform", "range": [0.7, 1.3]},
    "background_density": {"distribution": "uniform", "range": [0.003, 0.02]},
    "background_slope": {"distribution": "uniform", "range": [-0.4, 0.4]},
}


def conditions_for(task, suite):
    if suite not in SUITES:
        raise ValueError(f"Unknown suite: {suite}")
    result = dict(CONDITIONS) if suite != "v0.2" else {}
    if suite != "v0.1":
        if task not in HARD_WINDOWS:
            raise ValueError("v0.2 supports rutile/zno; Si remains a v0.1 sanity task")
        for window_name, windows in HARD_WINDOWS[task].items():
            for exposure in (100, 300, 1000):
                result[f"hard_{window_name}_e{exposure}"] = {
                    "regime": "id", "domain": "v0.2", "exposure": float(exposure),
                    "mask": "windows", "windows": windows,
                    "nuisance_prior": HARD_PRIOR,
                    "nuisance_pairing": "same_draw_in_calibrated_and_blind",
                }
    return result


def observation_schedule(task, suite, split, group):
    """Union releases retain standalone observation IDs and realizations."""
    old = list(ID_CONDITIONS) if split != "test" else list(CONDITIONS)
    hard = [c for c in conditions_for(task, "v0.2")] if suite != "v0.1" else []
    if split == "train":
        old = np.random.default_rng(rng_seed(SEED, group, "conditions")).choice(ID_CONDITIONS, size=2).tolist()
        if hard:
            hard = np.random.default_rng(rng_seed(SEED, group, "conditions-v0.2")).choice(hard, size=2).tolist()
        if suite == "v0.1+v0.2":
            return [(old[0], 0), (hard[1], 1)]
    if suite == "v0.1":
        return list(enumerate_pairs(old))
    if suite == "v0.2":
        return list(enumerate_pairs(hard))
    return list(enumerate_pairs(old)) + list(enumerate_pairs(hard))


def enumerate_pairs(names):
    return ((name, i) for i, name in enumerate(names))
NUISANCE_NAMES = ["fwhm_deg", "zero_shift_deg", "scale", "background_density", "background_slope"]
CALIBRATED = [0.15, 0.0, 1.0, 0.01, 0.0]
BLIND_PRIOR = {
    "fwhm_deg": {"distribution": "uniform", "range": [0.1, 0.3]},
    "zero_shift_deg": {"distribution": "uniform", "range": [-0.04, 0.04]},
    "scale": {"distribution": "log_uniform", "range": [0.7, 1.3]},
    "background_density": {"distribution": "uniform", "range": [0.003, 0.02]},
    "background_slope": {"distribution": "uniform", "range": [-0.4, 0.4]},
}
SAMPLES = 256
KS = [1, 8, 32, 64]
LEVELS = [0.5, 0.8, 0.9, 0.95]
THRESHOLDS = [0.001, 0.005, 0.01]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def rng_seed(*keys):
    return int.from_bytes(hashlib.sha256(canonical(list(keys))).digest()[:8], "little")


def task_spec(task):
    spec = dict(TASKS[task])
    ref = np.asarray(spec["reference"])
    spec.update(task=task, lower=(ref * 0.95).tolist(), upper=(ref * 1.05).tolist(), units="angstrom")
    return spec


def snapshot(task, track, preset, suite="v0.1"):
    if track not in ("calibrated", "blind"):
        raise ValueError("track must be calibrated or blind")
    cfg = conditions_for(task, suite)
    training = [name for name, value in cfg.items() if value["regime"] == "id"]
    domains = {}
    if suite != "v0.2":
        domains["v0.1"] = ID_CONDITIONS
    if suite != "v0.1":
        domains["v0.2"] = [name for name in training if name.startswith("hard_")]
    result = {
        "protocol": "xrdbench-0.1.0" if suite == "v0.1" else VERSION,
        "suite": suite, "evaluation_domains": domains,
        "task": task_spec(task), "track": track, "preset": preset,
        "seed": SEED, "split_structure_counts": dict(zip(("train", "val", "test"), SIZES[preset])),
        "train_augmentations": 2, "axis": AXIS.tolist(), "wavelength": WAVELENGTH,
        "peak_calculation_range": [5.0, 95.0], "conditions": cfg,
        "training_conditions": training, "calibrated_nuisance": CALIBRATED,
        "blind_nuisance_prior": BLIND_PRIOR, "nuisance_names": NUISANCE_NAMES,
        "posterior_samples": SAMPLES, "candidate_ks": KS,
        "coverage_levels": LEVELS, "relative_success_thresholds": THRESHOLDS,
    }
    if suite != "v0.1":
        # There is no fixed calibrated nuisance in v0.2, nor one universal
        # blind prior in the union. Do not expose misleading legacy defaults.
        result["calibrated_nuisance"] = None
        result["blind_nuisance_prior"] = HARD_PRIOR if suite == "v0.2" else None
        result["nuisance_priors_by_domain"] = {
            domain: HARD_PRIOR if domain == "v0.2" else BLIND_PRIOR for domain in domains
        }
    return result
