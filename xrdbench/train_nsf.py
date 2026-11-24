"""Single-round amortized NPE on frozen train/validation pairs, with no new simulations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .api import make_context
from .convnext import ModelConfig, set_device, synchronize
from .io import file_hash, manifest, write_json
from .nsf import ConditionalNSF, FlowConfig, UPSTREAM_COMMIT
from .train_convnext import load_split, source_hash


def sample_energy(samples, target):
    """Same independent-half Monte Carlo energy estimator as the common scorer."""
    half = samples.shape[1] // 2
    return (torch.linalg.vector_norm(samples - target[:, None, :], dim=-1).mean(1)
            - 0.5 * torch.linalg.vector_norm(samples[:, :half] - samples[:, half:2 * half], dim=-1).mean(1))


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    total, nll, energy, mae = 0, 0.0, 0.0, 0.0
    for spectrum, metadata, target, base in loader:
        spectrum, metadata, target = (x.to(device) for x in (spectrum, metadata, target))
        context = model.encoder(spectrum, metadata)
        log_prob = model.flow.log_prob(target, context)
        samples = model.flow.sample_from_base(base.to(device), context)
        if not torch.isfinite(log_prob).all().item() or not torch.isfinite(samples).all().item():
            raise FloatingPointError("Non-finite validation density or samples")
        n = len(target); total += n
        nll += -log_prob.sum().item()
        energy += sample_energy(samples, target).sum().item()
        # torch.median returns the lower middle for even S; use both middle values
        # to match NumPy's marginal median in the independent evaluator.
        ordered = samples.sort(dim=1).values
        middle = samples.shape[1] // 2
        point = (ordered[:, middle - 1] + ordered[:, middle]) / 2
        mae += (point - target).abs().mean(-1).sum().item()
    return {"val_nll_unit": nll / total, "val_energy_score": energy / total,
            "val_normalized_mae": mae / total}


def train(public, output, device="npu:0", epochs=100, batch_size=128, learning_rate=3e-4,
          weight_decay=1e-4, seed=0, patience=8, validate_every=5, max_seconds=3500,
          model_config=None, flow_config=None, export_test=False):
    if min(epochs, batch_size, patience, validate_every) < 1 or max_seconds <= 0:
        raise ValueError("Training limits must be positive")
    public, output = Path(public), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = set_device(device)
    if str(device).startswith("npu"):
        torch.npu.manual_seed_all(seed)
    started = time.perf_counter()
    meta = manifest(public); context = make_context(meta)
    model_config = model_config or ModelConfig(output_dim=len(context.lower))
    flow_config = flow_config or FlowConfig()
    if model_config.input_bins != len(context.two_theta_deg):
        raise ValueError("Spectrum axis does not match model configuration")
    train_data, train_groups = load_split(public, "train")
    validation_data, validation_groups = load_split(public, "val")
    if train_groups & validation_groups:
        raise ValueError("Training/validation structure leakage")
    for data in (train_data, validation_data):
        target = data.tensors[2]
        if not torch.isfinite(target).all() or not ((target >= 0) & (target <= 1)).all():
            raise ValueError("Target outside declared prior")
    # Common random numbers reduce MC variation in checkpoint comparisons. This
    # draws only base uniforms, independently of every validation target.
    validation_base = np.random.default_rng(37191 + seed).random((len(validation_data), 256, 2)).astype(np.float32)
    validation_data = TensorDataset(*validation_data.tensors, torch.from_numpy(validation_base))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model = ConditionalNSF(model_config, flow_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    context_dict = {"task": context.task, "track": context.track, "parameter_names": list(context.parameter_names),
        "lower": context.lower.tolist(), "upper": context.upper.tolist(), "two_theta_deg": context.two_theta_deg.tolist(),
        "wavelength_angstrom": context.wavelength_angstrom}
    hardware = torch.npu.get_device_name(0) if str(device).startswith("npu") else "CPU"
    method = {"method_name": "Conditional-NSF-NPE", "implementation": "bounded_rational_quadratic_coupling_flow_shared_convnext_encoder",
        "source_repository": "https://github.com/bayesiains/nflows", "upstream_commit": UPSTREAM_COMMIT,
        "encoder_source_repository": "https://github.com/AdvancedPhotonSource/OpenAlphaDiffract",
        "encoder_upstream_commit": "ec118b9d899f57e8971a7ad4d4aa23b3f38572e0", "code_commit": source_hash(),
        "training_dataset_id": meta["dataset_id"], "training_suite": meta["config"].get("suite", "v0.1"),
        "training_seed": seed, "unique_train_structures": len(train_groups), "train_observations": len(train_data),
        "extra_simulations": 0, "external_pretraining": False, "postprocessing": "none",
        "point_rule": "sample_marginal_median", "hardware": hardware, "precision": "float32",
        "hyperparameter_trials": 1, "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_selection": "minimum_validation_energy_score_256_fixed_base_samples",
        "hyperparameters": {"model": asdict(model_config), "flow": asdict(flow_config),
            "epochs_limit": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "loss": "exact_negative_log_density_in_prior_unit_coordinates",
            "base_distribution": "Uniform([0,1]^2)", "warmup_epochs": min(3, epochs), "schedule": "cosine",
            "patience_validation_checks": patience, "validate_every_epochs": validate_every,
            "validation_samples": 256, "max_seconds": max_seconds},
        "notes": "Single-round NPE, same fixed pairs and ConvNeXt encoder as point baseline; trained from scratch. No sbi runtime or original-paper XRD reproduction claim."}
    write_json(output / "method.json", method)
    write_json(output / "environment.json", {"torch": str(torch.__version__), "numpy": np.__version__,
        "device": str(device), "hardware": hardware,
        "torch_npu": str(__import__("torch_npu").__version__) if str(device).startswith("npu") else None})
    initial = validate(model, validation_loader, device)
    print({"event": "ready", "parameters": method["trainable_parameters"], "train_rows": len(train_data),
           "validation_rows": len(validation_data), **initial}, flush=True)
    best, best_epoch, stale, steps = float("inf"), 0, 0, 0
    history, stop_reason = [], "epoch_limit"
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        warmup = min(3, epochs)
        multiplier = (epoch / warmup if epoch <= warmup else
                      0.02 + 0.98 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup))) / 2)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * multiplier
        model.train(); loss_sum, seen = 0.0, 0
        for spectrum, metadata, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prob(target.to(device), spectrum.to(device), metadata.to(device)).mean()
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training density")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(target); seen += len(target); steps += 1
            if time.perf_counter() - started > max_seconds:
                stop_reason = "time_limit"; break
        row = {"epoch": epoch, "optimizer_steps": steps, "train_nll_unit": loss_sum / seen,
               "learning_rate": learning_rate * multiplier}
        if epoch == 1 or epoch % validate_every == 0 or epoch == epochs or stop_reason == "time_limit":
            values = validate(model, validation_loader, device)
            row.update(values)
            if values["val_energy_score"] < best:
                best, best_epoch, stale = values["val_energy_score"], epoch, 0
                checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "model_kind": "conditional_bounded_nsf", "model_config": asdict(model_config),
                    "flow_config": asdict(flow_config), "dataset_id": meta["dataset_id"],
                    "allowed_evaluation_dataset_ids": [meta["dataset_id"]], "context": context_dict,
                    "parameter_names": list(context.parameter_names), "epoch": epoch,
                    "validation": values, "training_seed": seed, "code_sha256": method["code_commit"]}
                torch.save(checkpoint, output / "best.tmp")
                (output / "best.tmp").replace(output / "best.pt")
            else:
                stale += 1
        synchronize(device)
        row.update(best_val_energy_score=best, epoch_seconds=time.perf_counter() - epoch_start,
                   elapsed_seconds=time.perf_counter() - started)
        history.append(row); write_json(output / "history.json", history)
        print(row, flush=True)
        if stop_reason == "time_limit":
            break
        if epoch >= 30 and stale >= patience:
            stop_reason = "validation_patience"; break
    method.update(training_wall_seconds=time.perf_counter() - started, selected_epoch=best_epoch,
                  optimizer_steps=steps, checkpoint_sha256=file_hash(output / "best.pt"))
    write_json(output / "method.json", method)
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    result = {"dataset_id": meta["dataset_id"], "task": context.task, "track": context.track,
        "best_epoch": best_epoch, "best_validation": selected["validation"], "initial_validation": initial,
        "epochs_completed": len(history), "optimizer_steps": steps, "stop_reason": stop_reason,
        "training_wall_seconds": method["training_wall_seconds"], "trainable_parameters": method["trainable_parameters"],
        "training_seed": seed, "test_labels_read": False}
    write_json(output / "training_summary.json", result)
    if export_test:
        from .submission import run
        os.environ["XRD_BENCH_DEVICE"] = str(device)
        del optimizer, model, loader, validation_loader, train_data, validation_data, selected
        if str(device).startswith("npu"):
            torch.npu.empty_cache()
        for split in ("val", "test"):
            run(public, output / f"prediction_{split}", "xrdbench.nsf:Adapter", "posterior", split=split,
                checkpoint=output / "best.pt", metadata=method, batch_size=batch_size)
    print({"event": "complete", **result}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=3500)
    parser.add_argument("--export-test", action="store_true")
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
