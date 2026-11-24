"""Conditional bounded neural spline flow with a shared ConvNeXt spectrum encoder.

Rational-quadratic equations adapted from bayesiains/nflows, commit
5d22344436a4a5f4c3b93a3b99322a6bfe374c08 (MIT; THIRD_PARTY_LICENSES/nflows.txt).
The flow maps the unit parameter box bijectively onto a uniform base. It uses
alternating scalar coupling transforms, exact likelihood and inverse sampling.
No clipping, rejection, test labels or physical postprocessing is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .convnext import (ConvNeXtRegressor, ModelConfig, features, set_device,
                       synchronize, validate_context)

UPSTREAM_COMMIT = "5d22344436a4a5f4c3b93a3b99322a6bfe374c08"


def rational_quadratic(inputs, parameters, inverse=False, min_bin=1e-3, min_derivative=1e-3):
    """Monotone [0,1] -> [0,1] spline; vectorized gather avoids dynamic masking.

    Inputs must be in the unit interval. Domain checks are done by the public
    density API and training loader, rather than synchronizing each NPU layer.
    """
    bins = (parameters.shape[-1] - 1) // 3
    if bins < 2 or parameters.shape[-1] != 3 * bins + 1 or bins * min_bin >= 1:
        raise ValueError("Invalid spline bins/parameters")
    widths_raw, heights_raw, derivatives_raw = torch.split(parameters, [bins, bins, bins + 1], dim=-1)

    def knots(logits):
        sizes = min_bin + (1 - min_bin * bins) * F.softmax(logits, dim=-1)
        cumulative = torch.cat((torch.zeros_like(sizes[..., :1]), torch.cumsum(sizes, dim=-1)[..., :-1],
                                torch.ones_like(sizes[..., :1])), dim=-1)
        return cumulative, cumulative[..., 1:] - cumulative[..., :-1]

    cumwidths, widths = knots(widths_raw)
    cumheights, heights = knots(heights_raw)
    # Zero logits initialize an exact identity transform, including derivatives.
    derivatives = min_derivative + F.softplus(derivatives_raw, beta=math.log(2) / (1 - min_derivative))
    boundaries = cumheights if inverse else cumwidths
    index = (inputs.unsqueeze(-1) >= boundaries).sum(-1).sub(1).clamp(0, bins - 1).unsqueeze(-1)
    def take(values):
        return values.gather(-1, index).squeeze(-1)
    x0, y0, width, height = map(take, (cumwidths, cumheights, widths, heights))
    delta = height / width
    d0, d1 = take(derivatives), take(derivatives[..., 1:])
    curvature = d0 + d1 - 2 * delta
    if inverse:
        offset = inputs - y0
        a = offset * curvature + height * (delta - d0)
        b = height * d0 - offset * curvature
        c = -delta * offset
        discriminant = b.square() - 4 * a * c
        position = 2 * c / (-b - discriminant.clamp_min(0).sqrt())
        outputs = x0 + position * width
    else:
        position = (inputs - x0) / width
        mixed = position * (1 - position)
        outputs = y0 + height * (delta * position.square() + d0 * mixed) / (delta + curvature * mixed)
    mixed = position * (1 - position)
    denominator = delta + curvature * mixed
    numerator = delta.square() * (d1 * position.square() + 2 * delta * mixed + d0 * (1 - position).square())
    logdet = numerator.log() - 2 * denominator.log()
    return outputs, -logdet if inverse else logdet


@dataclass(frozen=True)
class FlowConfig:
    layers: int = 6
    bins: int = 8
    hidden: int = 128
    context_dim: int = 128


class SplineCoupling(nn.Module):
    def __init__(self, config, target):
        super().__init__()
        self.target = target
        self.context_projection = nn.Linear(config.context_dim, config.hidden)
        self.coordinate_projection = nn.Linear(1, config.hidden, bias=False)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden, config.hidden), nn.SiLU(),
                                 nn.Linear(config.hidden, 3 * config.bins + 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, unit, context, inverse=False):
        other = unit[..., 1 - self.target:2 - self.target]
        projected = self.context_projection(context)
        if unit.ndim == 3:
            projected = projected[:, None, :]
        parameters = self.net(projected + self.coordinate_projection(other))
        transformed, logdet = rational_quadratic(unit[..., self.target], parameters, inverse)
        values = [unit[..., 0], unit[..., 1]]
        values[self.target] = transformed
        return torch.stack(values, dim=-1), logdet


class ConditionalSplineFlow(nn.Module):
    def __init__(self, config=FlowConfig()):
        super().__init__()
        if config.layers < 2 or config.bins < 2 or config.bins >= 1000:
            raise ValueError("At least two coupling layers and valid spline bins are required")
        self.config = config
        self.layers = nn.ModuleList(SplineCoupling(config, i % 2) for i in range(config.layers))

    def transform(self, unit, context, inverse=False):
        logdet = torch.zeros_like(unit[..., 0])
        for layer in reversed(self.layers) if inverse else self.layers:
            unit, increment = layer(unit, context, inverse)
            logdet = logdet + increment
        return unit, logdet

    def log_prob(self, unit, context):
        # Do not evaluate splines outside their domain. Return the correct -inf
        # support value, rather than assigning clipped points positive density.
        valid = ((unit >= 0) & (unit <= 1) & torch.isfinite(unit)).all(-1)
        safe = torch.where(valid.unsqueeze(-1), unit, torch.full_like(unit, 0.5))
        _, logdet = self.transform(safe, context)
        return torch.where(valid, logdet, torch.full_like(logdet, -float("inf")))

    def sample_from_base(self, base, context):
        return self.transform(base, context, inverse=True)[0]


class ConditionalNSF(nn.Module):
    def __init__(self, model_config=ModelConfig(), flow_config=FlowConfig()):
        super().__init__()
        if model_config.output_dim != 2 or flow_config.context_dim != 128:
            raise ValueError("Current NSF baseline supports the two-parameter rutile/ZnO tasks")
        self.encoder = ConvNeXtRegressor(model_config)
        # Exact same backbone, positional pool and 512 -> 128 projection as the
        # point baseline, replacing only the final 128 -> 2 regression layer.
        self.encoder.head = nn.Sequential(*list(self.encoder.head.children())[:-2])
        self.flow = ConditionalSplineFlow(flow_config)

    def log_prob(self, unit, spectrum, metadata):
        return self.flow.log_prob(unit, self.encoder(spectrum, metadata))

    def sample_from_base(self, base, spectrum, metadata):
        return self.flow.sample_from_base(base, self.encoder(spectrum, metadata))


def base_samples(seeds, n_samples):
    if n_samples < 1:
        raise ValueError("A positive number of posterior samples is required")
    return np.stack([np.random.default_rng(int(seed)).random((n_samples, 2)).astype(np.float32)
                     for seed in seeds])


class Adapter:
    def __init__(self, context, checkpoint):
        self.context = context
        self.device = set_device(os.environ.get("XRD_BENCH_DEVICE", "cpu"))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("model_kind") != "conditional_bounded_nsf":
            raise ValueError("Not a conditional NSF checkpoint")
        validate_context(context, saved)
        self.model = ConditionalNSF(ModelConfig(**saved["model_config"]), FlowConfig(**saved["flow_config"]))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()

    def synchronize(self):
        synchronize(self.device)

    @torch.inference_mode()
    def predict(self, batch, n_samples, seeds):
        spectrum, metadata = features(batch)
        unit = self.model.sample_from_base(torch.from_numpy(base_samples(seeds, n_samples)).to(self.device),
            torch.from_numpy(spectrum).to(self.device), torch.from_numpy(metadata).to(self.device))
        unit = unit.cpu().numpy()
        if not np.isfinite(unit).all() or (unit < -1e-6).any() or (unit > 1 + 1e-6).any():
            raise FloatingPointError("Invalid inverse-spline samples")
        return {"samples": self.context.lower + unit * (self.context.upper - self.context.lower)}
