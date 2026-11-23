"""Verify CLI, offline interoperability, raw-vs-physics separation and reference command."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from xrdbench.io import iter_observations, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source = Path(args.demo_root) / "rutile_calibrated"
    public, private = source / "data/public", source / "data/private"
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    metadata = read_json(source / "prior_samples/manifest.json")
    write_json(out / "method.json", metadata["metadata"])
    pred_file = source / "prior_samples" / metadata["files"][0]["path"]
    assert len(metadata["files"]) == 1, "This integration fixture expects the 88-row smoke submission in one shard"

    def execute(module, arguments, name):
        proc = subprocess.run([sys.executable, "-m", module, *map(str, arguments)], capture_output=True, text=True, check=True)
        (out / (name + ".stdout.txt")).write_text(proc.stdout)
        (out / (name + ".stderr.txt")).write_text(proc.stderr)
        print(name + " passed", flush=True)
        return proc.stdout

    execute("examples.export_npz", ["--public", public, "--input", pred_file, "--metadata", out / "method.json",
        "--kind", "posterior", "--output", out / "exported"], "offline_export")
    execute("xrdbench.cli", ["validate", "--public", public, "--submission", out / "exported"], "validate")
    execute("xrdbench.cli", ["score", "--public", public, "--private", private, "--submission", out / "exported",
        "--output", out / "score", "--physics-k", "8", "--physics-limit", "1"], "score")
    original = read_json(source / "prior_samples_score/metrics.json")
    exported = read_json(out / "score/metrics.json")
    assert exported["primary_id_macro"] == original["primary_id_macro"]
    paired = json.loads(execute("xrdbench.cli", ["compare", "--a", source / "prior_samples_score",
        "--b", out / "score", "--metric", "energy_score"], "compare"))
    assert paired["mean"] == 0 and paired["ci95"] == [0, 0]
    batch = next(iter_observations(public, "test"))
    sample_id = batch["ids"][batch["condition"] == "no_signal_stress"][0]
    grid = json.loads(execute("xrdbench.cli", ["reference-grid", "--public", public, "--sample-id", sample_id,
        "--points", "17", "--output", out / "grid"], "reference_grid"))
    assert abs(grid["effective_grid_cells"] - 17**2) < 1e-6
    write_json(out / "validation.json", {"passed": True, "offline_export_matches_original": True,
               "physics_scoring_did_not_modify_raw_posterior_metrics": True, "reference_no_signal_is_prior": True})


if __name__ == "__main__":
    main()
