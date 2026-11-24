"""Position-preserving ConvNeXt1D regression for the frozen lattice benchmark.

ConvNeXt block adapted from OpenAlphaDiffract, copyright 2026 UChicago
Argonne, LLC. See THIRD_PARTY_LICENSES/OpenAlphaDiffract.txt. This adaptation
adds channel LayerNorm and uses a new benchmark-specific stem and head.
It is not a reproduction of AlphaDiffract's original multi-task experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os

import numpy as np
import torch
from torch import nn

from .api import encode


@dataclass(frozen=True)
class ModelConfig:
    dims: tuple = (48, 96, 192)
    depths: tuple = (2, 2, 4)
    kernel_size: int = 7
    pool_bins: int = 16
    head_dim: int = 512
    dropout: float = 0.1
    drop_path: float = 0.05
    layer_scale: float = 1e-3
    input_bins: int = 4096
    output_dim: int = 2


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_path, layer_scale):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.expand = nn.Linear(dim, 4 * dim)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale))
        self.drop_path = drop_path

    def forward(self, x):
        residual = self.depthwise(x).transpose(1, 2)
        residual = self.project(self.activation(self.expand(self.norm(residual))))
        residual = (residual * self.gamma).transpose(1, 2)
        if self.training and self.drop_path:
            keep = 1 - self.drop_path
            gate = (torch.rand((x.shape[0], 1, 1), device=x.device) < keep).to(x.dtype)
            residual = residual * gate / keep
        return x + residual


class ConvNeXtRegressor(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        if len(config.dims) != len(config.depths) or config.kernel_size % 2 != 1:
            raise ValueError("Matching stages and an odd kernel size are required")
        self.stem = nn.Sequential(nn.Conv1d(3, config.dims[0], 9, stride=4, padding=4),
                                  ChannelNorm(config.dims[0]))
        stages = []
        block_index, blocks = 0, sum(config.depths)
        for i, (dim, depth) in enumerate(zip(config.dims, config.depths)):
            if i:
                stages.extend([ChannelNorm(config.dims[i - 1]),
                               nn.Conv1d(config.dims[i - 1], dim, 2, stride=2)])
            for _ in range(depth):
                stages.append(ConvNeXtBlock(dim, config.kernel_size,
                    config.drop_path * block_index / max(blocks - 1, 1), config.layer_scale))
                block_index += 1
        self.stages = nn.Sequential(*stages)
        self.norm = ChannelNorm(config.dims[-1])
        # Retain absolute angular location; global average pooling alone discards it.
        self.pool = nn.AdaptiveAvgPool1d(config.pool_bins)
        self.metadata = nn.Sequential(nn.Linear(7, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(config.dims[-1] * config.pool_bins + 32, config.head_dim),
            nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.head_dim, 128),
            nn.GELU(), nn.Linear(128, config.output_dim), nn.Sigmoid())

    def forward(self, spectrum, metadata):
        x = self.pool(self.norm(self.stages(self.stem(spectrum)))).flatten(1)
        return self.head(torch.cat((x, self.metadata(metadata)), dim=1))


def features(batch):
    """Only observable inputs; fixed transforms require no fitted/test statistics."""
    spectral = encode(batch)
    position = np.broadcast_to(np.linspace(-1, 1, spectral.shape[-1], dtype=np.float32),
                               (len(batch.ids), 1, spectral.shape[-1]))
    known = batch.nuisance_known.astype(np.float32)
    nuisance = np.nan_to_num(batch.nuisance, nan=0.0).astype(np.float32)
    nuisance = (nuisance - np.array([0.3, 0, 1, 0.01, 0], dtype=np.float32)) / np.array(
        [0.3, 0.1, 0.3, 0.01, 0.4], dtype=np.float32)
    nuisance *= known[:, None]
    metadata = np.column_stack(((np.log10(batch.exposure) - 3) / 2, known, nuisance)).astype(np.float32)
    return np.ascontiguousarray(np.concatenate((spectral, position), axis=1)), metadata


def set_device(name):
    if name == "cpu":
        return torch.device(name)
    if not name.startswith("npu:"):
        raise ValueError("Use CPU or an authorized Ascend910B2 Merlin job")
    import torch_npu  # noqa: F401
    torch.npu.set_device(name)
    if "910B2" not in torch.npu.get_device_name(torch.npu.current_device()):
        raise RuntimeError("This project only authorizes Ascend910B2")
    torch.npu.set_compile_mode(jit_compile=False)
    return torch.device(name)


def synchronize(device):
    if str(device).startswith("npu"):
        torch.npu.synchronize()


def validate_context(context, saved):
    expected = saved["context"]
    if context.dataset_id not in saved["allowed_evaluation_dataset_ids"]:
        raise ValueError("Checkpoint does not allow this evaluation dataset_id")
    if (context.task != expected["task"] or context.track != expected["track"] or
            list(context.parameter_names) != expected["parameter_names"]):
        raise ValueError("Checkpoint task, track or parameter convention mismatch")
    for name in ("lower", "upper", "two_theta_deg"):
        if not np.array_equal(getattr(context, name), expected[name]):
            raise ValueError(f"Checkpoint {name} mismatch")
    if context.wavelength_angstrom != expected["wavelength_angstrom"]:
        raise ValueError("Checkpoint wavelength mismatch")


class Adapter:
    def __init__(self, context, checkpoint):
        self.context = context
        self.device = set_device(os.environ.get("XRD_BENCH_DEVICE", "cpu"))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_context(context, saved)
        self.model = ConvNeXtRegressor(ModelConfig(**saved["model_config"]))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()

    def synchronize(self):
        synchronize(self.device)

    @torch.inference_mode()
    def predict(self, batch, n_samples, seeds):
        x, meta = features(batch)
        unit = self.model(torch.from_numpy(x).to(self.device), torch.from_numpy(meta).to(self.device))
        return {"point": self.context.lower + unit.cpu().numpy() * (self.context.upper - self.context.lower)}
