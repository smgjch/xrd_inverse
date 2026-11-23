"""Export externally generated NumPy predictions; usable with TensorFlow/JAX/etc."""
import argparse

from xrdbench.io import load_npz, manifest, read_json
from xrdbench.submission import SubmissionWriter, validate_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--public", required=True)
    p.add_argument("--input", required=True, help="NPZ with ids plus point / samples / candidates")
    p.add_argument("--output", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--kind", required=True, choices=["point", "posterior", "candidates"])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--view", default="full", choices=["full", "physics128"])
    args = p.parse_args()
    data = load_npz(args.input)
    ids = data.pop("ids")
    writer = SubmissionWriter(args.output, manifest(args.public), args.split, args.kind, read_json(args.metadata), view=args.view)
    writer.add(ids, data)
    writer.finish()
    print(validate_submission(args.public, args.output)[0])


if __name__ == "__main__":
    main()
