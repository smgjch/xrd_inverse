"""Complete adapter pattern; this small network is a wiring example, not DeepLPnet."""
import os

import numpy as np
import torch

from xrdbench.api import encode


class ExampleNet(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(2, 16, 9, stride=4, padding=4), torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, 7, stride=4, padding=3), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(8), torch.nn.Flatten(),
        )
        self.head = torch.nn.Sequential(torch.nn.Linear(256 + 7, 64), torch.nn.ReLU(), torch.nn.Linear(64, output_dim), torch.nn.Sigmoid())

    def forward(self, spectrum, metadata):
        return self.head(torch.cat([self.encoder(spectrum), metadata], dim=-1))


def features(batch):
    # All values here are observable. For blind track, unknown nuisance is filled with 0
    # and the separate known flag tells the network that these are NOT measured zeros.
    meta = np.column_stack([np.log10(batch.exposure), batch.nuisance_known,
                            np.nan_to_num(batch.nuisance, nan=0.0)]).astype(np.float32)
    return encode(batch), meta


class Adapter:
    def __init__(self, context, checkpoint):
        self.context = context
        self.device = os.environ.get("XRD_BENCH_DEVICE", "npu:0")
        if self.device.startswith("npu"):
            import torch_npu  # noqa: F401; required to register NPU backend
            torch.npu.set_device(self.device)
            if "910B2" not in torch.npu.get_device_name(torch.npu.current_device()):
                raise RuntimeError("This project only authorizes Ascend910B2")
        elif self.device != "cpu":
            raise ValueError("Project permits CPU or authorized Ascend910B2; CUDA is not configured")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        allowed = set(saved.get("allowed_evaluation_dataset_ids", [saved["dataset_id"]]))
        if context.dataset_id not in allowed or tuple(saved["parameter_names"]) != context.parameter_names:
            raise ValueError("Checkpoint dataset or parameter convention mismatch")
        self.model = ExampleNet(len(context.lower))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to(self.device).eval()

    def synchronize(self):
        if self.device.startswith("npu"):
            torch.npu.synchronize()

    @torch.inference_mode()
    def predict(self, batch, n_samples, seeds):
        spectrum, meta = features(batch)
        unit = self.model(torch.from_numpy(spectrum).to(self.device), torch.from_numpy(meta).to(self.device))
        unit = unit.cpu().numpy()
        return {"point": self.context.lower + unit * (self.context.upper - self.context.lower)}
