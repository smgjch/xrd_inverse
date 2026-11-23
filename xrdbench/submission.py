from __future__ import annotations

import importlib
from pathlib import Path
import platform
import time

import numpy as np

from .api import make_batch, make_context, sample_seeds
from .io import (checked_load, concatenate, file_hash, iter_observations, manifest,
                 read_json, save_npz, write_json)
from .protocol import SAMPLES, VERSION


def validate_outputs(output, n, d, kind):
    if kind not in ("point", "posterior", "candidates"):
        raise ValueError("Invalid output kind")
    allowed = {"point", "failed"} if kind == "point" else ({"samples", "failed"} if kind == "posterior" else {"point", "candidates", "failed"})
    if set(output) - allowed:
        raise ValueError(f"Unexpected prediction fields: {set(output) - allowed}")
    required = allowed - {"failed"}
    if not required <= set(output):
        raise ValueError(f"Missing fields: {required - set(output)}")
    result = {key: np.asarray(value) for key, value in output.items()}
    result.setdefault("failed", np.zeros(n, dtype=bool))
    for key, value in result.items():
        if key == "failed":
            if value.shape != (n,) or value.dtype != np.bool_:
                raise ValueError("failed must be bool[N]")
            continue
        if value.dtype.kind not in "fi" or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite numeric arrays; submit a declared fallback for failed rows")
        shape = (n, d) if key == "point" else (n, SAMPLES if key == "samples" else 64, d)
        if value.shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {value.shape}")
    if kind == "candidates":
        if not np.all(np.any(np.all(np.isclose(result["point"][:, None], result["candidates"], atol=1e-7, rtol=0), axis=-1), axis=1)):
            raise ValueError("Candidate point must be an actual submitted candidate")
    return result


class SubmissionWriter:
    """Use this writer to export predictions from any framework, including offline code."""
    def __init__(self, output, dataset, split, kind, metadata, view="full"):
        if kind not in ("point", "posterior", "candidates"):
            raise ValueError(kind)
        if split not in ("val", "test"):
            raise ValueError("Submissions are for val/test only")
        if kind == "posterior" and metadata.get("postprocessing", "none") != "none":
            raise ValueError("Export refined/reweighted solutions as candidates, not posterior samples")
        if "method_name" not in metadata:
            raise ValueError("metadata must include method_name")
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.kind, self.d = kind, len(dataset["config"]["task"]["parameters"])
        self.buffer, self.buffer_rows = [], 0
        if view not in ("full", "physics128"):
            raise ValueError("Unknown evaluation view")
        self.meta = {"protocol": dataset["config"]["protocol"], "dataset_id": dataset["dataset_id"], "split": split, "view": view,
                     "kind": kind, "metadata": metadata, "files": [], "timing": {}}

    def add(self, ids, prediction):
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in "US":
            raise ValueError("ids must be strings[N]")
        output = validate_outputs(prediction, len(ids), self.d, self.kind)
        self.buffer.append({key: np.array(value, copy=True) for key, value in {"ids": ids, **output}.items()})
        self.buffer_rows += len(ids)
        if self.buffer_rows >= 1024:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        arrays = concatenate(self.buffer)
        name = f"predictions-{len(self.meta['files']):05d}.npz"
        save_npz(self.root / name, **arrays)
        self.meta["files"].append({"path": name, "sha256": file_hash(self.root / name), "rows": self.buffer_rows})
        self.buffer, self.buffer_rows = [], 0

    def finish(self, timing=None):
        self._flush()
        if not self.meta["files"]:
            raise ValueError("Empty submission")
        self.meta["timing"] = timing or {"source": "external_export", "verified_by_runner": False}
        write_json(self.root / "manifest.json", self.meta)


def run(public, output, adapter, kind, split="test", seed=0, batch_size=32, checkpoint=None, metadata=None, view="full"):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data = manifest(public)
    ctx = make_context(data)
    method_meta = dict(metadata or {})
    if not method_meta and adapter not in ("xrdbench.api:PriorMean", "xrdbench.api:PriorSampler"):
        raise ValueError("Custom adapters require --metadata; use examples/method_metadata.json")
    if not method_meta:
        method_meta = {"method_name": adapter.split(":")[-1], "purpose": "negative_control",
                       "training_dataset_id": None, "unique_train_structures": 0, "extra_simulations": 0,
                       "external_pretraining": False, "postprocessing": "none", "hardware": "CPU"}
    method_meta.update(adapter=adapter, inference_seed=int(seed), batch_size=batch_size)
    if checkpoint is not None and Path(checkpoint).is_file():
        method_meta["checkpoint_sha256"] = file_hash(checkpoint)
    writer = SubmissionWriter(output, data, split, kind, method_meta, view=view)
    start = time.perf_counter()
    module, name = adapter.split(":", 1)
    model = getattr(importlib.import_module(module), name)(context=ctx, checkpoint=checkpoint)
    if not callable(getattr(model, "synchronize", None)):
        raise TypeError("Adapter must implement synchronize(), even if CPU no-op")
    model.synchronize()
    load_seconds = time.perf_counter() - start
    n_samples = SAMPLES if kind == "posterior" else 64
    first = make_batch(next(iter_observations(public, split, view)), slice(0, batch_size))
    warm_start = time.perf_counter()
    validate_outputs(model.predict(first, n_samples, sample_seeds(first, seed)), len(first.ids), writer.d, kind)
    model.synchronize()
    warm_seconds = time.perf_counter() - warm_start
    durations, rows, total_rows = [], [], 0
    pipeline_start = time.perf_counter()
    for arrays in iter_observations(public, split, view):
        for i in range(0, len(arrays["ids"]), batch_size):
            batch = make_batch(arrays, slice(i, i + batch_size))
            model.synchronize()
            tick = time.perf_counter()
            prediction = model.predict(batch, n_samples, sample_seeds(batch, seed))
            model.synchronize()
            durations.append(time.perf_counter() - tick)
            rows.append(len(batch.ids)); total_rows += len(batch.ids)
            writer.add(batch.ids, prediction)
    # Include the final buffered shard in pipeline time; model timing stays unchanged.
    writer._flush()
    pipeline_seconds = time.perf_counter() - pipeline_start
    timing = {"source": "runner_wall_clock", "verified_by_runner": True, "host_platform": platform.platform(),
              "adapter_load_seconds": load_seconds, "warmup_seconds": warm_seconds,
              "model_total_seconds": sum(durations), "model_seconds_per_observation": sum(durations) / total_rows,
              "pipeline_seconds_including_read_hash_and_write": pipeline_seconds,
              "batch_seconds": durations, "batch_rows": rows, "n_observations": total_rows,
              "excludes": ["training", "dataset_generation", "coordinator_physics_scoring"]}
    writer.finish(timing)
    return validate_submission(public, output)[0]


def validate_submission(public, submission):
    dataset = manifest(public)
    root = Path(submission)
    meta = read_json(root / "manifest.json")
    if meta["protocol"] != dataset["config"]["protocol"] or meta["dataset_id"] != dataset["dataset_id"]:
        raise ValueError("Submission protocol/dataset fingerprint mismatch")
    if meta["split"] not in ("val", "test"):
        raise ValueError("Invalid evaluation split")
    if meta["kind"] == "posterior" and meta["metadata"].get("postprocessing", "none") != "none":
        raise ValueError("Postprocessed candidates cannot be scored as raw posterior samples")
    prediction = concatenate(checked_load(root, record) for record in meta["files"])
    expected_ids = np.concatenate([a["ids"] for a in iter_observations(public, meta["split"], meta["view"])])
    if len(set(prediction["ids"].tolist())) != len(prediction["ids"]):
        raise ValueError("Duplicate prediction IDs")
    if not np.array_equal(prediction["ids"], expected_ids):
        raise ValueError("Prediction IDs must exactly match the release order; no dropped/extra/reordered rows")
    validate_outputs({k: v for k, v in prediction.items() if k != "ids"}, len(expected_ids),
                     len(dataset["config"]["task"]["parameters"]), meta["kind"])
    return {"passed": True, "dataset_id": dataset["dataset_id"], "kind": meta["kind"],
            "split": meta["split"], "view": meta["view"], "observations": len(expected_ids), "method": meta["metadata"]["method_name"]}, prediction, meta
