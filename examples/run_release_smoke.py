"""Generate and evaluate all six small releases on CPU; never submit GPU jobs."""
import argparse
from pathlib import Path

from xrdbench.data import audit, generate
from xrdbench.evaluate import compare, score
from xrdbench.io import write_json
from xrdbench.submission import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New directory on NAS")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for task in ("si", "rutile", "zno"):
        for track in ("calibrated", "blind"):
            path = root / f"{task}_{track}"
            print(f"Validating {task}/{track}", flush=True)
            dataset = generate(path / "data", task, track, "smoke")
            public, private = path / "data/public", path / "data/private"
            audit_result = audit(public, private)
            for name, adapter, kind in (("prior_mean", "PriorMean", "point"), ("prior_samples", "PriorSampler", "posterior")):
                run(public, path / name, f"xrdbench.api:{adapter}", kind, batch_size=16)
                score(public, path / name, path / (name + "_score"), private,
                      physics_k=1 if track == "calibrated" else 0, physics_limit=2)
            comparison = compare([path / "prior_mean_score"], [path / "prior_samples_score"])
            write_json(path / "comparison.json", comparison)
            results.append({"task": task, "track": track, "dataset": dataset, "audit": audit_result,
                            "point_and_posterior_pipeline_passed": True})
    write_json(root / "validation.json", {"purpose": "integration_smoke_not_baseline_performance", "releases": results})
    print("All six small releases passed; see validation.json", flush=True)


if __name__ == "__main__":
    main()
