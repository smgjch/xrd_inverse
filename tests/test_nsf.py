from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xrdbench.api import make_batch, make_context, sample_seeds
from xrdbench.convnext import ModelConfig
from xrdbench.data import generate
from xrdbench.evaluate import score
from xrdbench.io import iter_observations, manifest
from xrdbench.nsf import (Adapter, ConditionalNSF, ConditionalSplineFlow, FlowConfig,
                          base_samples, rational_quadratic)
from xrdbench.reporting import summarize_scores
from xrdbench.submission import run
from xrdbench.train_nsf import sample_energy, train

torch.set_num_threads(2)


def test_identity_prior_and_support():
    model = ConditionalSplineFlow(FlowConfig(layers=4, bins=8, context_dim=3)).double()
    unit = torch.rand(13, 2, dtype=torch.float64)
    context = torch.randn(13, 3, dtype=torch.float64)
    torch.testing.assert_close(model.log_prob(unit, context), torch.zeros(13, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(model.sample_from_base(unit[:, None, :], context)[:, 0], unit, atol=1e-12, rtol=0)
    outside = unit.clone(); outside[0, 0] = -0.01
    assert model.log_prob(outside, context)[0].isneginf()


def test_spline_inverse_jacobian_and_integrated_density():
    torch.manual_seed(19)
    parameters = torch.randn(9, 25, dtype=torch.float64) * 0.7
    x = torch.linspace(0.01, 0.99, 9, dtype=torch.float64).requires_grad_()
    y, ld = rational_quadratic(x, parameters)
    restored, inverse_ld = rational_quadratic(y, parameters, inverse=True)
    torch.testing.assert_close(restored, x, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(inverse_ld, -ld, atol=1e-10, rtol=1e-10)
    derivative = torch.autograd.grad(y.sum(), x)[0]
    torch.testing.assert_close(derivative.log(), ld, atol=1e-10, rtol=1e-10)
    grid = (torch.arange(20000, dtype=torch.float64) + 0.5) / 20000
    _, density_log = rational_quadratic(grid, parameters[0].expand(len(grid), -1))
    assert density_log.exp().mean().item() == pytest.approx(1.0, abs=1e-5)


def test_nontrivial_flow_inverse_and_full_jacobian():
    torch.manual_seed(8)
    model = ConditionalSplineFlow(FlowConfig(layers=4, bins=8, hidden=16, context_dim=3)).double()
    for layer in model.layers:
        torch.nn.init.normal_(layer.net[-1].weight, std=0.1)
    x = torch.tensor([[0.27, 0.61]], dtype=torch.float64, requires_grad=True)
    context = torch.randn(1, 3, dtype=torch.float64)
    y, ld = model.transform(x, context)
    restored, inverse_ld = model.transform(y, context, inverse=True)
    torch.testing.assert_close(restored, x, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(inverse_ld, -ld, atol=1e-10, rtol=1e-10)
    jac = torch.autograd.functional.jacobian(lambda value: model.transform(value, context)[0], x)
    torch.testing.assert_close(torch.linalg.slogdet(jac[0, :, 0, :])[1], ld[0], atol=1e-10, rtol=1e-10)


def test_flow_learns_conditional_density():
    torch.manual_seed(21)
    flow = ConditionalSplineFlow(FlowConfig(layers=4, bins=8, hidden=24, context_dim=1))
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.003)
    context = torch.linspace(-1, 1, 192)[:, None]
    target = torch.cat((0.5 + 0.25 * context, 0.5 - 0.2 * context), 1)
    target += 0.03 * torch.randn_like(target)
    before = flow.log_prob(target, context).mean().item()
    for _ in range(70):
        optimizer.zero_grad()
        loss = -flow.log_prob(target, context).mean()
        loss.backward(); optimizer.step()
    after = flow.log_prob(target, context).mean().item()
    assert after > before + 1.5
    assert all(torch.isfinite(p).all() for p in flow.parameters())


def test_energy_matches_independent_numpy_definition():
    samples = np.random.default_rng(0).uniform(size=(8, 256, 2))
    target = np.random.default_rng(1).uniform(size=(8, 2))
    expected = (np.linalg.norm(samples - target[:, None], axis=-1).mean(1)
                - 0.5 * np.linalg.norm(samples[:, :128] - samples[:, 128:], axis=-1).mean(1))
    actual = sample_energy(torch.from_numpy(samples), torch.from_numpy(target)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    seeds = np.array([17, 24, 192], dtype=np.uint64)
    np.testing.assert_array_equal(base_samples(seeds, 64), base_samples(seeds, 256)[:, :64])


def test_training_adapter_and_unified_posterior_scoring(tmp_path, monkeypatch):
    root = tmp_path / "data"
    generate(root, "rutile", "blind", "smoke", suite="v0.1+v0.2")
    # Removing private labels demonstrates that training never needs them.
    private = root / "private"
    moved = tmp_path / "evaluation_labels"
    private.rename(moved)
    config = ModelConfig(dims=(8, 16), depths=(1, 1), pool_bins=4, head_dim=32, dropout=0, drop_path=0)
    flow = FlowConfig(layers=2, bins=4, hidden=16)
    result = train(root / "public", tmp_path / "fit", device="cpu", epochs=2, batch_size=16,
                   model_config=config, flow_config=flow, max_seconds=120, validate_every=1)
    assert result["test_labels_read"] is False
    monkeypatch.setenv("XRD_BENCH_DEVICE", "cpu")
    context = make_context(manifest(root / "public"))
    adapter = Adapter(context, tmp_path / "fit/best.pt")
    arrays = next(iter_observations(root / "public", "test"))
    batch = make_batch(arrays, slice(0, 3))
    both = adapter.predict(batch, 256, sample_seeds(batch, 0))["samples"]
    one = make_batch(arrays, slice(1, 2))
    alone = adapter.predict(one, 256, sample_seeds(one, 0))["samples"]
    np.testing.assert_allclose(both[1], alone[0], atol=2e-6, rtol=0)
    assert both.shape == (3, 256, 2)
    assert (both >= context.lower).all() and (both <= context.upper).all()
    run(root / "public", tmp_path / "prediction", "xrdbench.nsf:Adapter", "posterior",
        checkpoint=tmp_path / "fit/best.pt", metadata={"method_name": "NSF-test"}, batch_size=11)
    scored = score(root / "public", tmp_path / "prediction", tmp_path / "score", moved)
    assert np.isfinite(scored["primary_id_macro"]["energy_score"])
    assert summarize_scores([tmp_path / "score"], tmp_path / "report")["matrix_rows"] == 3
