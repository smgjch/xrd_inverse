"""Generate each task in one CPU process to reuse its cached reflection lists."""
import argparse
from pathlib import Path

from xrdbench.data import audit, generate
from xrdbench.io import manifest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["rutile", "zno"], required=True)
    parser.add_argument("--preset", choices=["smoke", "pilot", "main"], default="pilot")
    args = parser.parse_args()
    results = []
    for track in ("calibrated", "blind"):
        for suite in ("v0.1", "v0.2", "v0.1+v0.2"):
            root = args.output / args.task / track / suite
            if root.exists():
                cfg = manifest(root / "public")["config"]
                if cfg.get("suite") != suite or cfg["preset"] != args.preset:
                    raise ValueError("Existing release differs from requested configuration")
            else:
                print(generate(root, args.task, track, args.preset, suite=suite), flush=True)
            results.append({"task": args.task, "track": track, "suite": suite,
                            "audit": audit(root / "public", root / "private")})
            write_json(args.output / f"{args.task}_release_audit.json", results)


if __name__ == "__main__":
    main()
