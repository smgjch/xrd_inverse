"""Coordinator check: file integrity, fixed budgets, union reuse, paired observations."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from xrdbench.data import audit
from xrdbench.io import iter_observations, manifest, write_json


def row_hashes(public, split, include_nuisance=True):
    result = {}
    fields = ["counts", "mask", "exposure"]
    if include_nuisance:
        fields.extend(["nuisance", "nuisance_known"])
    for batch in iter_observations(public, split):
        for i, sid in enumerate(batch["ids"].tolist()):
            h = hashlib.sha256()
            for field in fields:
                h.update(np.asarray(batch[field][i]).tobytes())
            if sid in result:
                raise ValueError("Duplicate observation")
            result[sid] = h.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for task in ("rutile", "zno"):
        for track in ("calibrated", "blind"):
            roots = {suite: args.data / task / track / suite for suite in ("v0.1", "v0.2", "v0.1+v0.2")}
            for suite, root in roots.items():
                result = audit(root / "public", root / "private")
                info = manifest(root / "public")
                assert result["splits"]["train"]["observations"] == 2 * result["splits"]["train"]["structures"]
                reports.append({"task": task, "track": track, "suite": suite, **result})
            for split in ("train", "val", "test"):
                old = row_hashes(roots["v0.1"] / "public", split)
                hard = row_hashes(roots["v0.2"] / "public", split)
                union = row_hashes(roots["v0.1+v0.2"] / "public", split)
                merged = {**old, **hard}
                assert len(merged) == len(old) + len(hard), "Domains must have distinct observation IDs"
                assert all(merged.get(sid) == value for sid, value in union.items())
                if split == "train":
                    assert len(union) == len(old) == len(hard)
                    assert sum(sid in old for sid in union) * 2 == len(union)
                    assert all(sid.endswith(":0") for sid in union if sid in old)
                    assert all(sid.endswith(":1") for sid in union if sid in hard)
                else:
                    assert union == merged
            print(f"Verified {task}/{track}: file hashes, split isolation, fixed training budget, union identity", flush=True)
        for split in ("train", "val", "test"):
            cal = row_hashes(args.data / task / "calibrated/v0.2/public", split, False)
            blind = row_hashes(args.data / task / "blind/v0.2/public", split, False)
            assert cal == blind, "v0.2 calibrated/blind measured data must be identical"
        print(f"Verified {task}: matched calibrated/blind observations", flush=True)
    write_json(args.output, {"passed": True, "training_performed": False, "releases": reports,
        "checks": ["file_hashes", "split_isolation", "fixed_training_budget", "union_exact_observation_reuse", "v0.2_paired_instrument_information"]})


if __name__ == "__main__":
    main()
