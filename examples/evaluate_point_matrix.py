"""Score completed point-baseline runs and matched prior-mean controls on CPU."""
import argparse
from pathlib import Path

from xrdbench.evaluate import compare, score
from xrdbench.io import read_json, write_json
from xrdbench.reporting import summarize_scores
from xrdbench.submission import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=["rutile", "zno"])
    parser.add_argument("--tracks", nargs="+", default=["blind", "calibrated"])
    args = parser.parse_args()
    data_root, run_root, output = Path(args.data_root), Path(args.run_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    scores, comparisons, training = [], [], []
    for task in args.tasks:
        for track in args.tracks:
            release = data_root / task / track / "v0.1+v0.2"
            fitted = run_root / task / track / "seed0"
            training.append(read_json(fitted / "training_summary.json"))
            scope = output / task / track
            scope.mkdir(parents=True)
            score(release / "public", fitted / "prediction_val", scope / "validation")
            score(release / "public", fitted / "prediction_test", scope / "convnext", release / "private")
            run(release / "public", scope / "prior_prediction", "xrdbench.api:PriorMean", "point")
            score(release / "public", scope / "prior_prediction", scope / "prior", release / "private")
            scores.extend([scope / "convnext", scope / "prior"])
            for domain in ("v0.1", "v0.2", "v0.1+v0.2"):
                comparisons.append({"task": task, "track": track, "evaluation_suite": domain,
                    **compare([scope / "convnext"], [scope / "prior"], condition=domain)})
    result = summarize_scores(scores, output / "summary")
    write_json(output / "paired_vs_prior.json", comparisons)
    write_json(output / "training_summaries.json", training)
    print(result)


if __name__ == "__main__":
    main()
