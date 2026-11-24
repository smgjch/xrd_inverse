from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xrdbench.api import make_batch, make_context
from xrdbench.convnext import Adapter, ConvNeXtRegressor, ModelConfig, features, set_device
from xrdbench.data import generate
from xrdbench.io import iter_observations, manifest
from xrdbench.train_convnext import load_split


@pytest.fixture(scope="module")
def blind_data(tmp_path_factory):
    path = tmp_path_factory.mktemp("convnext") / "data"
    generate(path, "rutile", "blind", "smoke", suite="v0.1+v0.2")
    return path


def test_feature_inputs_and_training_split_guard(blind_data):
    public = blind_data / "public"
    raw = next(iter_observations(public, "train"))
    x, m = features(make_batch(raw))
    assert x.shape[1:] == (3, 4096) and m.shape[1] == 7
    assert np.isfinite(x).all() and np.isfinite(m).all()
    assert np.all(m[:, 1:] == 0)  # blind known flag and all nuisance features are zero
    alternate = {**raw, "theta": np.full((len(raw["ids"]), 2), 999),
                 "group_ids": np.full_like(raw["group_ids"], "modified")}
    for a, b in zip(features(make_batch(alternate)), (x, m)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="only permits"):
        load_split(public, "test")
    train, train_groups = load_split(public, "train")
    val, val_groups = load_split(public, "val")
    assert len(train) == 64 and len(val) == 96 and not train_groups & val_groups


def test_gradients_checkpoint_and_context_guard(blind_data, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    public = blind_data / "public"
    data = manifest(public); context = make_context(data)
    batch = make_batch(next(iter_observations(public, "val")), slice(0, 4))
    x, m = [torch.from_numpy(v) for v in features(batch)]
    config = ModelConfig(dims=(8, 16), depths=(1, 1), pool_bins=4, head_dim=32, drop_path=0, dropout=0)
    model = ConvNeXtRegressor(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    target = torch.tensor([[0.1, 0.9]]).expand(4, 2)
    initial = (model(x, m) - target).square().mean().item()
    for _ in range(8):
        optimizer.zero_grad()
        loss = (model(x, m) - target).square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert (model(x, m) - target).square().mean().item() < initial
    model.eval()
    saved = {"state_dict": model.state_dict(), "model_config": asdict(config),
        "allowed_evaluation_dataset_ids": [context.dataset_id],
        "context": {"task": context.task, "track": context.track,
            "parameter_names": list(context.parameter_names), "lower": context.lower.tolist(),
            "upper": context.upper.tolist(), "two_theta_deg": context.two_theta_deg.tolist(),
            "wavelength_angstrom": context.wavelength_angstrom}}
    path = tmp_path / "model.pt"; torch.save(saved, path)
    monkeypatch.setenv("XRD_BENCH_DEVICE", "cpu")
    adapter = Adapter(context, path)
    actual = adapter.predict(batch, 256, np.zeros(4))["point"]
    expected = context.lower + model(x, m).detach().numpy() * (context.upper - context.lower)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="task, track"):
        Adapter(replace(context, track="calibrated"), path)
    with pytest.raises(ValueError, match="dataset_id"):
        Adapter(replace(context, dataset_id="different"), path)
    with pytest.raises(ValueError, match="authorized"):
        set_device("cuda:0")
