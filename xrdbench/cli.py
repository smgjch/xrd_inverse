import argparse
import json

from .data import audit, generate
from .evaluate import compare, score
from .io import read_json
from .reference import grid_reference
from .submission import run, validate_submission
from .reporting import summarize_scores


def main():
    parser = argparse.ArgumentParser(description="Shared XRD lattice inverse benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--output", required=True)
    gen.add_argument("--task", choices=["si", "rutile", "zno"], default="rutile")
    gen.add_argument("--track", choices=["calibrated", "blind"], default="calibrated")
    gen.add_argument("--preset", choices=["smoke", "pilot", "main"], default="smoke")
    gen.add_argument("--shard-size", type=int, default=256)
    gen.add_argument("--suite", choices=["v0.1", "v0.2", "v0.1+v0.2"], default="v0.1")
    check = commands.add_parser("audit")
    check.add_argument("--public", required=True); check.add_argument("--private")
    infer = commands.add_parser("run")
    infer.add_argument("--public", required=True); infer.add_argument("--output", required=True)
    infer.add_argument("--adapter", required=True)
    infer.add_argument("--kind", choices=["point", "posterior", "candidates"], required=True)
    infer.add_argument("--split", choices=["val", "test"], default="test")
    infer.add_argument("--view", choices=["full", "physics128"], default="full")
    infer.add_argument("--seed", type=int, default=0); infer.add_argument("--batch-size", type=int, default=32)
    infer.add_argument("--checkpoint"); infer.add_argument("--metadata")
    validate = commands.add_parser("validate")
    validate.add_argument("--public", required=True); validate.add_argument("--submission", required=True)
    evaluation = commands.add_parser("score")
    evaluation.add_argument("--public", required=True); evaluation.add_argument("--private")
    evaluation.add_argument("--submission", required=True); evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--physics-k", type=int, default=0); evaluation.add_argument("--physics-limit", type=int, default=128)
    paired = commands.add_parser("compare")
    paired.add_argument("--a", nargs="+", required=True); paired.add_argument("--b", nargs="+", required=True)
    paired.add_argument("--metric", default="normalized_mae"); paired.add_argument("--condition", default="id")
    paired.add_argument("--repeats", type=int, default=2000)
    report = commands.add_parser("summarize", help="Aggregate scored union-test submissions; never trains")
    report.add_argument("--scores", nargs="+", required=True)
    report.add_argument("--output", required=True)
    grid = commands.add_parser("reference-grid")
    grid.add_argument("--public", required=True); grid.add_argument("--output", required=True)
    grid.add_argument("--sample-id"); grid.add_argument("--split", default="test", choices=["val", "test"])
    grid.add_argument("--points", type=int, default=65)
    args = vars(parser.parse_args()); command = args.pop("command")
    if command == "run" and args["metadata"]:
        args["metadata"] = read_json(args["metadata"])
    if command == "validate":
        result = validate_submission(**args)[0]
    elif command == "compare":
        args["a_paths"] = args.pop("a"); args["b_paths"] = args.pop("b")
        result = compare(**args)
    elif command == "summarize":
        result = summarize_scores(**args)
    else:
        result = {"generate": generate, "audit": audit, "run": run, "score": score, "reference-grid": grid_reference}[command](**args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
