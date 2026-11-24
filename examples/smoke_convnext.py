"""Check full ConvNeXt forward/backward/checkpoint on CPU or authorized NPU."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from xrdbench.convnext import ConvNeXtRegressor, ModelConfig, set_device, synchronize
from xrdbench.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    device = set_device(args.device)
    config = ModelConfig(drop_path=0, dropout=0)
    model = ConvNeXtRegressor(config).to(device)
    x = torch.rand((args.batch_size, 3, 4096), device=device)
    m = torch.zeros((args.batch_size, 7), device=device)
    target = torch.full((args.batch_size, 2), 0.25, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    losses, times = [], []
    for _ in range(8):
        synchronize(device); start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x, m) - target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); synchronize(device)
        losses.append(loss.item()); times.append(time.perf_counter() - start)
    if not losses[-1] < losses[0] or not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError("Full-network gradient smoke failed")
    path = Path(args.output); path.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "model_config": asdict(config)}, path / "smoke.pt")
    result = {"passed": True, "device": str(device), "torch": str(torch.__version__),
        "parameters": sum(p.numel() for p in model.parameters()), "losses": losses, "step_seconds": times,
        "hardware": torch.npu.get_device_name(0) if str(device).startswith("npu") else "CPU"}
    write_json(path / "smoke.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
