"""Minimal adapter ABI. Inputs never include labels or hidden nuisance values."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import rng_seed


@dataclass(frozen=True)
class Context:
    dataset_id: str
    task: str
    track: str
    parameter_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    two_theta_deg: np.ndarray
    wavelength_angstrom: float


@dataclass(frozen=True)
class ObservationBatch:
    ids: np.ndarray
    counts: np.ndarray
    mask: np.ndarray
    exposure: np.ndarray
    nuisance: np.ndarray
    nuisance_known: np.ndarray


def make_context(meta):
    cfg = meta["config"]
    return Context(meta["dataset_id"], cfg["task"]["task"], cfg["track"], tuple(cfg["task"]["parameters"]),
                   np.asarray(cfg["task"]["lower"]), np.asarray(cfg["task"]["upper"]),
                   np.asarray(cfg["axis"]), cfg["wavelength"])


def make_batch(arrays, selection=slice(None)):
    # Explicit allowlist: no condition ID, group ID, label, clean spectrum, or latent nuisance.
    return ObservationBatch(**{key: arrays[key][selection] for key in ObservationBatch.__dataclass_fields__})


def encode(batch):
    """Suggested two-channel encoder input; optional, not a forced preprocessing rule."""
    intensity = np.log1p(batch.counts.astype(np.float32)) / np.log1p(batch.exposure[:, None])
    return np.stack([intensity * batch.mask, batch.mask.astype(np.float32)], axis=1).astype(np.float32)


def sample_seeds(batch, seed):
    return np.asarray([rng_seed("inference", int(seed), str(sample_id)) for sample_id in batch.ids], dtype=np.uint64)


class PriorMean:
    """No-spectrum negative control, NOT a strong scientific baseline."""
    def __init__(self, context, checkpoint=None):
        self.context = context

    def synchronize(self):
        pass

    def predict(self, batch, n_samples, seeds):
        return {"point": np.broadcast_to((self.context.lower + self.context.upper) / 2,
                                         (len(batch.ids), len(self.context.lower))).copy()}


class PriorSampler(PriorMean):
    def predict(self, batch, n_samples, seeds):
        return {"samples": np.stack([np.random.default_rng(int(seed)).uniform(
            self.context.lower, self.context.upper, size=(n_samples, len(self.context.lower))) for seed in seeds])}
