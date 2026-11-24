"""Score frozen NSF predictions, prior controls, and optional point predictions."""
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
    parser.add_argument("--point-run-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=["rutile", "zno"])
    parser.add_argument("--tracks", nargs="+", default=["blind", "calibrated"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data_root, run_root, output = Path(args.data_root), Path(args.run_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    scores, comparisons, training = [], [], []
    for task in args.tasks:
        for track in args.tracks:
            release = data_root / task / track / "v0.1+v0.2"
            fitted = run_root / task / track / f"seed{args.seed}"
            training.append(read_json(fitted / "training_summary.json"))
            inference_seed = read_json(fitted / "prediction_test" / "manifest.json")["metadata"]["inference_seed"]
            scope = output / task / track
            scope.mkdir(parents=True)
            score(release / "public", fitted / "prediction_val", scope / "validation")
            score(release / "public", fitted / "prediction_test", scope / "nsf", release / "private")
            scores.append(scope / "nsf")
            for name, adapter, kind in (("prior_sampler", "PriorSampler", "posterior"), ("prior_mean", "PriorMean", "point")):
                run(release / "public", scope / f"{name}_prediction", f"xrdbench.api:{adapter}", kind,
                    seed=inference_seed)
                score(release / "public", scope / f"{name}_prediction", scope / name, release / "private")
                scores.append(scope / name)
            if args.point_run_root:
                point = Path(args.point_run_root) / task / track / f"seed{args.seed}" / "prediction_test"
                score(release / "public", point, scope / "convnext", release / "private")
                scores.append(scope / "convnext")
            comparisons_to_run = [("prior_sampler", "energy_score"), ("prior_mean", "normalized_mae")]
            if args.point_run_root:
                comparisons_to_run.append(("convnext", "normalized_mae"))
            for domain in ("v0.1", "v0.2", "v0.1+v0.2"):
                for baseline, metric in comparisons_to_run:
                    comparisons.append({"task": task, "track": track, "evaluation_suite": domain,
                        "baseline": baseline, "metric": metric,
                        **compare([scope / "nsf"], [scope / baseline], condition=domain, metric=metric)})
            print(f"Scored {task}/{track}: posterior metrics, calibration, all 17 conditions and matched controls", flush=True)
    result = summarize_scores(scores, output / "summary")
    write_json(output / "paired_comparisons.json", comparisons)
    write_json(output / "training_summaries.json", training)
    print(result, flush=True)


if __name__ == "__main__":
    main()
