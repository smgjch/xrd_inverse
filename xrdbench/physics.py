"""Authoritative CPU simulator/scorer. No torch, test labels, or hidden nuisance."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import xlogy
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

from .protocol import AXIS, TASKS, WAVELENGTH


def structure(task, theta):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (len(TASKS[task]["reference"]),) or not np.isfinite(theta).all() or (theta <= 0).any():
        raise ValueError("Invalid lattice parameters")
    if task == "si":
        return Structure.from_spacegroup(227, Lattice.cubic(theta[0]), ["Si"], [[0, 0, 0]])
    if task == "rutile":
        u = TASKS[task]["oxygen_u"]
        return Structure.from_spacegroup(136, Lattice.tetragonal(*theta), ["Ti", "O"], [[0, 0, 0], [u, u, 0]])
    if task == "zno":
        return Structure.from_spacegroup(186, Lattice.hexagonal(*theta), ["Zn", "O"], [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, TASKS[task]["oxygen_u"]]])
    raise KeyError(task)


@lru_cache(maxsize=16384)
def _peaks(task, theta):
    pattern = XRDCalculator(wavelength=WAVELENGTH, symprec=0).get_pattern(
        structure(task, theta), scaled=True, two_theta_range=(5, 95)
    )
    return np.asarray(pattern.x), np.asarray(pattern.y) / 100.0


def peaks(task, theta):
    return _peaks(task, tuple(float(x) for x in theta))


def expected_counts(task, theta, nuisance, exposure, *, profile="gaussian", mixing=0.5, gains=None, peak_data=None):
    """Exposure is integrated counts of a unit-intensity isolated reflection."""
    width, shift, scale, b0, b1 = np.asarray(nuisance, dtype=float)
    if width <= 0 or scale <= 0 or b0 < 0 or abs(b1) > 1 or exposure <= 0:
        raise ValueError("Invalid nuisance parameters")
    centers, intensities = peaks(task, theta) if peak_data is None else peak_data
    if gains is not None:
        intensities = intensities * gains
    delta = AXIS[:, None] - (centers[None, :] + shift)
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-0.5 * (delta / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    if profile == "pseudo_voigt":
        lorentz = (width / 2) / (np.pi * (delta**2 + (width / 2) ** 2))
        kernel = (1 - mixing) * gaussian + mixing * lorentz
    elif profile == "gaussian":
        kernel = gaussian
    else:
        raise ValueError(profile)
    density = scale * np.sum(kernel * intensities, axis=1) + b0 * (1 + b1 * (AXIS - 50) / 40)
    return np.maximum(exposure * (AXIS[1] - AXIS[0]) * density, 1e-12)


def poisson_nll(counts, mean, mask):
    """Sum on observed bins only; omit log(y!) constant for ranking."""
    counts, mean, mask = np.asarray(counts), np.asarray(mean), np.asarray(mask, dtype=bool)
    return float(np.sum(mean[mask] - xlogy(counts[mask], mean[mask])))


def poisson_deviance(counts, mean, mask):
    counts, mean, mask = np.asarray(counts), np.asarray(mean), np.asarray(mask, dtype=bool)
    y, mu = counts[mask], mean[mask]
    return float(2 * np.sum(xlogy(y, y / mu) - y + mu))


def candidate_scores(task, candidates, counts, mask, nuisance, exposure):
    """Observation-only scorer. Blind-track profiling/marginalization is NOT implemented."""
    if not np.isfinite(nuisance).all():
        raise ValueError("Shared scorer requires observable nuisance (calibrated track)")
    result = []
    for theta in candidates:
        if not np.isfinite(theta).all() or (np.asarray(theta) <= 0).any():
            result.append(float("inf"))
            continue
        mean = expected_counts(task, theta, nuisance, exposure)
        result.append(poisson_nll(counts, mean, mask))
    return np.asarray(result)
