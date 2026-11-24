"""Train ConvNeXt on frozen train/validation data; optionally export predictions.

Training never opens test observations or private test targets. Test inference
is a separate post-training action using the validation-selected checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .api import make_batch, make_context
from .convnext import ConvNeXtRegressor, ModelConfig, features, set_device, synchronize
from .io import checked_load, file_hash, manifest, write_json


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def load_split(public, split):
    if split not in ("train", "val"):
        raise ValueError("Training loader only permits train and val")
    meta = manifest(public)
    records = meta["splits"][split]
    if len(records["observations"]) != len(records["targets"]):
        raise ValueError("Mismatched observation/target shards")
    spectra, metadata, targets, groups = [], [], [], set()
    lower, upper = np.asarray(meta["config"]["task"]["lower"]), np.asarray(meta["config"]["task"]["upper"])
    for obs_record, label_record in zip(records["observations"], records["targets"]):
        obs, labels = checked_load(public, obs_record), checked_load(public, label_record)
        if not np.array_equal(obs["ids"], labels["ids"]) or not np.array_equal(obs["group_ids"], labels["group_ids"]):
            raise ValueError("Training observation/label alignment mismatch")
        x, m = features(make_batch(obs))
        spectra.append(x); metadata.append(m)
        targets.append(((labels["theta"] - lower) / (upper - lower)).astype(np.float32))
        groups.update(obs["group_ids"].tolist())
    arrays = [np.concatenate(parts) for parts in (spectra, metadata, targets)]
    return TensorDataset(*(torch.from_numpy(a) for a in arrays)), groups


@torch.inference_mode()
def validation_error(model, loader, device):
    model.eval()
    absolute, n = 0.0, 0
    for x, m, target in loader:
        prediction = model(x.to(device), m.to(device))
        absolute += (prediction - target.to(device)).abs().sum().item()
        n += target.numel()
    return absolute / n


def train(public, output, device="npu:0", epochs=100, batch_size=128, learning_rate=5e-4,
          weight_decay=0.01, seed=0, patience=20, max_seconds=3500, config=None, export_test=False):
    public, output = Path(public), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = set_device(device)
    if str(device).startswith("npu"):
        torch.npu.manual_seed_all(seed)
    meta = manifest(public); context = make_context(meta)
    config = config or ModelConfig(output_dim=len(context.lower))
    if config.input_bins != len(context.two_theta_deg):
        raise ValueError("Spectrum axis does not match model configuration")
    started = time.perf_counter()
    train_data, train_groups = load_split(public, "train")
    val_data, val_groups = load_split(public, "val")
    if train_groups & val_groups:
        raise ValueError("Training/validation structure leakage")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model = ConvNeXtRegressor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.SmoothL1Loss(beta=0.02)
    context_dict = {"task": context.task, "track": context.track, "parameter_names": list(context.parameter_names),
        "lower": context.lower.tolist(), "upper": context.upper.tolist(), "two_theta_deg": context.two_theta_deg.tolist(),
        "wavelength_angstrom": context.wavelength_angstrom}
    hardware = torch.npu.get_device_name(0) if str(device).startswith("npu") else "CPU"
    method = {"method_name": "ConvNeXt1D-Lattice", "implementation": "adapted_convnext_block_with_benchmark_stem_and_position_preserving_head",
        "source_repository": "https://github.com/AdvancedPhotonSource/OpenAlphaDiffract",
        "upstream_commit": "ec118b9d899f57e8971a7ad4d4aa23b3f38572e0", "code_commit": source_hash(),
        "training_dataset_id": meta["dataset_id"], "training_suite": meta["config"].get("suite", "v0.1"),
        "training_seed": seed, "unique_train_structures": len(train_groups), "train_observations": len(train_data),
        "extra_simulations": 0, "external_pretraining": False, "postprocessing": "none", "point_rule": "direct_regression",
        "hardware": hardware, "precision": "float32", "hyperparameter_trials": 1,
        "trainable_parameters": sum(p.numel() for p in model.parameters()), "checkpoint_selection": "minimum_validation_normalized_mae",
        "hyperparameters": {"model": asdict(config), "epochs_limit": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay, "loss": "SmoothL1", "huber_beta": 0.02,
            "warmup_epochs": min(3, epochs), "schedule": "cosine", "patience": patience, "max_seconds": max_seconds},
        "notes": "Pilot seed-0 architecture adaptation; no original AlphaDiffract performance reproduction claim."}
    write_json(output / "method.json", method)
    write_json(output / "environment.json", {"torch": str(torch.__version__), "numpy": np.__version__,
        "device": str(device), "hardware": hardware, "torch_npu": str(__import__("torch_npu").__version__) if str(device).startswith("npu") else None})
    initial = validation_error(model, val_loader, device)
    print(json.dumps({"event": "ready", "parameters": method["trainable_parameters"], "train_rows": len(train_data),
                      "validation_rows": len(val_data), "initial_val_normalized_mae": initial}), flush=True)
    best, best_epoch, stale, steps = float("inf"), 0, 0, 0
    stop_reason = "epoch_limit"
    history = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        warmup = min(3, epochs)
        multiplier = epoch / warmup if epoch <= warmup else 0.02 + 0.98 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup))) / 2
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * multiplier
        model.train(); loss_sum, seen = 0.0, 0
        for x, m, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.to(device), m.to(device))
            loss = loss_function(prediction, target.to(device))
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(x); seen += len(x); steps += 1
            if time.perf_counter() - started > max_seconds:
                stop_reason = "time_limit"; break
        value = validation_error(model, val_loader, device)
        synchronize(device)
        improved = value < best
        if improved:
            best, best_epoch, stale = value, epoch, 0
            checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_config": asdict(config), "dataset_id": meta["dataset_id"],
                "allowed_evaluation_dataset_ids": [meta["dataset_id"]], "parameter_names": list(context.parameter_names),
                "context": context_dict, "epoch": epoch, "validation_normalized_mae": value,
                "training_seed": seed, "code_sha256": method["code_commit"]}
            torch.save(checkpoint, output / "best.tmp")
            (output / "best.tmp").replace(output / "best.pt")
        else:
            stale += 1
        row = {"epoch": epoch, "optimizer_steps": steps, "train_loss": loss_sum / seen,
            "val_normalized_mae": value, "best_val_normalized_mae": best, "learning_rate": learning_rate * multiplier,
            "epoch_seconds": time.perf_counter() - epoch_start, "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if stop_reason == "time_limit":
            break
        if epoch >= 30 and stale >= patience:
            stop_reason = "validation_patience"; break
    method.update(training_wall_seconds=time.perf_counter() - started, checkpoint_sha256=file_hash(output / "best.pt"),
                  selected_epoch=best_epoch, optimizer_steps=steps)
    write_json(output / "method.json", method)
    result = {"dataset_id": meta["dataset_id"], "task": context.task, "track": context.track,
        "best_epoch": best_epoch, "best_validation_normalized_mae": best, "initial_validation_normalized_mae": initial,
        "epochs_completed": len(history), "optimizer_steps": steps, "stop_reason": stop_reason,
        "training_wall_seconds": method["training_wall_seconds"], "trainable_parameters": method["trainable_parameters"],
        "training_seed": seed, "test_labels_read": False}
    write_json(output / "training_summary.json", result)
    if export_test:
        from .submission import run
        os.environ["XRD_BENCH_DEVICE"] = str(device)
        del optimizer, model, train_data, val_data, loader, val_loader
        if str(device).startswith("npu"):
            torch.npu.empty_cache()
        for split in ("val", "test"):
            run(public, output / f"prediction_{split}", "xrdbench.convnext:Adapter", "point", split=split,
                checkpoint=output / "best.pt", metadata=method, batch_size=batch_size)
    print(json.dumps({"event": "complete", **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=3500)
    parser.add_argument("--export-test", action="store_true")
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
