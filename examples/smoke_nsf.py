"""Exercise full NSF forward/inverse, likelihood gradients and 256-sample inference."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch

from xrdbench.convnext import ModelConfig, set_device, synchronize
from xrdbench.io import write_json
from xrdbench.nsf import ConditionalNSF, FlowConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    device = set_device(args.device)
    model = ConditionalNSF(ModelConfig(dropout=0, drop_path=0), FlowConfig()).to(device)
    x = torch.rand((args.batch_size, 3, 4096), device=device)
    metadata = torch.zeros((args.batch_size, 7), device=device)
    target = 0.2 + 0.1 * torch.rand((args.batch_size, 2), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses, times = [], []
    for _ in range(8):
        synchronize(device); started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = -model.log_prob(target, x, metadata).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step(); synchronize(device)
        losses.append(loss.item()); times.append(time.perf_counter() - started)
    model.eval()
    with torch.inference_mode():
        context = model.encoder(x, metadata)
        base = torch.rand((args.batch_size, 256, 2), device=device)
        started = time.perf_counter()
        samples = model.flow.sample_from_base(base, context)
        restored, _ = model.flow.transform(samples, context)
        synchronize(device)
        sample_seconds = time.perf_counter() - started
        error = (restored - base).abs().max().item()
        support = ((samples >= 0) & (samples <= 1)).all().item()
    if not losses[-1] < losses[0] or not support or error > 3e-4:
        raise RuntimeError(f"NSF NPU smoke failed: loss={losses}, inverse_error={error}, support={support}")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    result = {"passed": True, "device": str(device), "torch": str(torch.__version__),
        "parameters": sum(p.numel() for p in model.parameters()), "losses": losses,
        "step_seconds": times, "sample_and_roundtrip_seconds": sample_seconds,
        "max_roundtrip_error": error, "sample_shape": list(samples.shape),
        "hardware": torch.npu.get_device_name(0) if str(device).startswith("npu") else "CPU"}
    write_json(output / "smoke.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
