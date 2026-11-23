from __future__ import annotations

import importlib.metadata
from pathlib import Path
import platform

import numpy as np

from . import __version__
from .io import file_hash, save_npz, write_json, manifest, iter_observations, load_targets
from .physics import expected_counts, peaks, structure
from .protocol import (AXIS, BLIND_PRIOR, CALIBRATED, CONDITIONS, ID_CONDITIONS,
                       SEED, SIZES, digest, rng_seed, snapshot, task_spec, observation_schedule)


def observation_mask(condition, seed, cfg=None):
    cfg = CONDITIONS[condition] if cfg is None else cfg
    mask = np.ones(AXIS.size, dtype=bool)
    if cfg["mask"] == "block":
        length = round(AXIS.size * cfg["fraction"])
        start = np.random.default_rng(seed).integers(0, AXIS.size - length + 1)
        mask[start:start + length] = False
    elif cfg["mask"] == "window":
        lo, hi = cfg["window"]
        mask = (AXIS >= lo) & (AXIS <= hi)
    elif cfg["mask"] == "none":
        mask[:] = False
    elif cfg["mask"] == "windows":
        mask[:] = False
        for lo, hi in cfg["windows"]:
            mask |= (AXIS >= lo) & (AXIS <= hi)
    return mask


def draw_nuisance(track, condition, seed, cfg=None):
    cfg = CONDITIONS[condition] if cfg is None else cfg
    rng = np.random.default_rng(seed)
    if track == "calibrated" and "nuisance_prior" not in cfg:
        result = np.asarray(CALIBRATED, dtype=float)
    else:
        result = []
        for spec in cfg.get("nuisance_prior", BLIND_PRIOR).values():
            lo, hi = spec["range"]
            result.append(np.exp(rng.uniform(np.log(lo), np.log(hi))) if spec["distribution"] == "log_uniform" else rng.uniform(lo, hi))
        result = np.asarray(result)
    if "width_range" in cfg:
        result[0] = rng.uniform(*cfg["width_range"])
    return result


def simulate_observation(task, theta, track, condition, seed, peak_data=None, cfg=None):
    cfg = CONDITIONS[condition] if cfg is None else cfg
    nuisance = draw_nuisance(track, condition, rng_seed(seed, "nuisance"), cfg)
    peak_data = peaks(task, theta) if peak_data is None else peak_data
    gains = None
    if cfg.get("drop_probability"):
        gains = np.random.default_rng(rng_seed(seed, "peak_gains")).binomial(1, 1 - cfg["drop_probability"], len(peak_data[0]))
    mean = expected_counts(task, theta, nuisance, cfg["exposure"],
                           profile=cfg.get("profile", "gaussian"), mixing=cfg.get("mixing", 0.5),
                           gains=gains, peak_data=peak_data)
    counts = np.random.default_rng(rng_seed(seed, "poisson")).poisson(mean).astype(np.uint32)
    mask = observation_mask(condition, rng_seed(seed, "mask"), cfg)
    counts[~mask] = 0
    public_nuisance = nuisance if track == "calibrated" else np.full(5, np.nan)
    return {"counts": counts, "mask": mask, "exposure": cfg["exposure"],
            "nuisance": public_nuisance, "nuisance_known": track == "calibrated"}


def source_fingerprint():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


def generate(output, task="rutile", track="calibrated", preset="smoke", shard_size=256, suite="v0.1"):
    cfg = snapshot(task, track, preset, suite)
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {root}")
    root.mkdir(parents=True)
    public, private = root / "public", root / "private"
    public.mkdir(); private.mkdir()
    lo, hi = np.asarray(cfg["task"]["lower"]), np.asarray(cfg["task"]["upper"])
    cif = public / "prototype.cif"
    cif.write_text(structure(task, cfg["task"]["reference"]).to(fmt="cif"), encoding="utf-8")
    meta = {
        "config": cfg, "generator": {"package": __version__, "source_sha256": source_fingerprint(),
            "python": platform.python_version(), "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pymatgen", "spglib", "monty")}},
        "prototype": {"path": "prototype.cif", "sha256": file_hash(cif),
                      "provenance": "Explicit idealized standard prototype in xrdbench/physics.py; not an experimental CIF"},
        "splits": {},
    }
    test_targets = []
    for split, size in zip(("train", "val", "test"), SIZES[preset]):
        records = {"observations": [], "targets": [], "structure_count": size, "observation_count": 0}
        obs_buffer, target_buffer, shard = [], [], 0

        def flush():
            nonlocal shard
            if not obs_buffer:
                return
            obs_path = f"{split}/observations-{shard:05d}.npz"
            target_path = f"{split}/targets-{shard:05d}.npz"
            obs_arrays = {key: np.asarray([o[key] for o in obs_buffer]) for key in obs_buffer[0]}
            target_arrays = {key: np.asarray([o[key] for o in target_buffer]) for key in target_buffer[0]}
            save_npz(public / obs_path, **obs_arrays)
            label_root = private if split == "test" else public
            save_npz(label_root / target_path, **target_arrays)
            records["observations"].append({"path": obs_path, "sha256": file_hash(public / obs_path), "rows": len(obs_buffer)})
            label_record = {"path": target_path, "sha256": file_hash(label_root / target_path), "rows": len(obs_buffer)}
            (test_targets if split == "test" else records["targets"]).append(label_record)
            records["observation_count"] += len(obs_buffer)
            shard += 1
            obs_buffer.clear(); target_buffer.clear()

        for index in range(size):
            group = f"{task}:{split}:{index:08d}"
            # Preset-independent per-structure stream makes 1k/5k/20k/100k train prefixes nested.
            theta = np.random.default_rng(rng_seed(SEED, group, "theta")).uniform(lo, hi)
            peak_data = peaks(task, theta)
            for condition, replica in observation_schedule(task, suite, split, group):
                sample_id = f"{group}:{condition}:{replica}"
                obs = simulate_observation(task, theta, track, condition, rng_seed(SEED, sample_id), peak_data,
                                           cfg["conditions"][condition])
                identity = {"ids": sample_id, "group_ids": group, "condition": condition}
                obs.update(identity)
                obs_buffer.append(obs)
                target_buffer.append({**identity, "theta": theta})
                if len(obs_buffer) >= shard_size:
                    flush()
            if (index + 1) % 1000 == 0:
                print(f"{split}: {index + 1}/{size} structures", flush=True)
        flush()
        meta["splits"][split] = records
    meta["dataset_id"] = digest(meta)
    write_json(public / "manifest.json", meta)
    write_json(private / "manifest.json", {"dataset_id": meta["dataset_id"], "test_targets": test_targets})
    return {"dataset_id": meta["dataset_id"], "root": str(root), "sizes": {k: {f: v[f] for f in ("structure_count", "observation_count")} for k, v in meta["splits"].items()}}


def audit(public, private=None):
    meta = manifest(public)
    if file_hash(Path(public) / meta["prototype"]["path"]) != meta["prototype"]["sha256"]:
        raise ValueError("Prototype CIF checksum mismatch")
    d = len(meta["config"]["task"]["parameters"])
    split_groups, result = {}, {}
    theta_splits = {}
    for split, records in meta["splits"].items():
        ids, groups, conditions = [], [], []
        for batch in iter_observations(public, split):
            n = len(batch["ids"])
            assert batch["counts"].shape == batch["mask"].shape == (n, AXIS.size)
            assert batch["mask"].dtype == np.bool_
            assert np.issubdtype(batch["counts"].dtype, np.integer) and (batch["counts"] >= 0).all()
            assert (batch["counts"][~batch["mask"]] == 0).all()
            assert np.isfinite(batch["exposure"]).all() and (batch["exposure"] > 0).all()
            known = meta["config"]["track"] == "calibrated"
            assert np.all(batch["nuisance_known"] == known)
            assert np.isfinite(batch["nuisance"]).all() if known else np.isnan(batch["nuisance"]).all()
            ids.extend(batch["ids"].tolist()); groups.extend(batch["group_ids"].tolist()); conditions.extend(batch["condition"].tolist())
        assert len(ids) == len(set(ids)) == records["observation_count"]
        split_groups[split] = set(groups)
        assert len(split_groups[split]) == records["structure_count"]
        if split != "test" or private is not None:
            labels = load_targets(public, split, private)
            assert np.array_equal(labels["ids"], ids)
            assert np.array_equal(labels["group_ids"], groups)
            assert labels["theta"].shape == (len(ids), d)
            lo, hi = np.asarray(meta["config"]["task"]["lower"]), np.asarray(meta["config"]["task"]["upper"])
            assert np.isfinite(labels["theta"]).all() and (labels["theta"] >= lo).all() and (labels["theta"] <= hi).all()
            by_group = {}
            for group, theta in zip(groups, labels["theta"]):
                if group in by_group:
                    assert np.array_equal(by_group[group], theta)
                by_group[group] = theta
            theta_splits[split] = {tuple(theta) for theta in by_group.values()}
        result[split] = {"observations": len(ids), "structures": len(split_groups[split]), "conditions": sorted(set(conditions))}
    for left in split_groups:
        for right in split_groups:
            if left < right:
                assert split_groups[left].isdisjoint(split_groups[right])
                if left in theta_splits and right in theta_splits:
                    assert theta_splits[left].isdisjoint(theta_splits[right])
    return {"passed": True, "dataset_id": meta["dataset_id"], "splits": result}
