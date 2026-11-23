"""CPU-only integration check with RANDOM weights. Not training or performance evidence."""
import argparse
import os
from pathlib import Path

import torch

from examples.torch_point_adapter import ExampleNet
from xrdbench.io import manifest
from xrdbench.submission import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    data = manifest(args.public)
    parameters = data["config"]["task"]["parameters"]
    torch.manual_seed(123)
    model = ExampleNet(len(parameters))
    checkpoint = root / "random_untrained.pt"
    torch.save({"state_dict": model.state_dict(), "dataset_id": data["dataset_id"], "parameter_names": parameters}, checkpoint)
    os.environ["XRD_BENCH_DEVICE"] = "cpu"
    result = run(args.public, root / "submission", "examples.torch_point_adapter:Adapter", "point", checkpoint=str(checkpoint),
                 metadata={"method_name": "random_untrained_adapter_check", "purpose": "integration_only_not_baseline",
                           "training_dataset_id": None, "unique_train_structures": 0, "external_pretraining": False,
                           "extra_simulations": 0, "postprocessing": "none", "hardware": "CPU", "point_rule": "untrained_network"})
    print(result)


if __name__ == "__main__":
    main()
