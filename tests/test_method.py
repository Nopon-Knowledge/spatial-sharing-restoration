import math

import numpy as np
import pytest
import torch
from scipy.optimize import lsq_linear

from spatial_sharing.modules import RGLDSE, DSPC, StrengthPredictor, gate_features, state_loss
from spatial_sharing.model import RestorationModel, SmokeBackbone
from spatial_sharing.projection import project_affine, oracle_strength
from spatial_sharing.analysis import color_matrix_fit, curve_fit, projected_shift, paired_bootstrap
from spatial_sharing.data import assert_disjoint, BalancedBatches
from spatial_sharing.metrics import luminance, image_metrics, aggregate

torch.set_num_threads(2)


def count(module):
    return sum(p.numel() for p in module.parameters())


def test_parameter_counts_match_supplement():
    state, correction = RGLDSE(), DSPC()
    assert count(state.global_state)+count(state.channel_gate)+count(state.severity) == 83009
    assert count(state.local_states)+count(state.local_fusion) == 93987
    assert count(state) == 176998
    assert count(correction)-count(correction.state_projection) == 3302
    assert count(correction.state_projection) == 1024
    assert count(state)+count(correction) == 181324
    assert count(StrengthPredictor()) == 1073


def test_identity_initialization_and_local_sizes():
    state = RGLDSE()
    features = torch.randn(2, 64, 17, 21)
    result = state(torch.randn(2, 256), features)
    assert torch.equal(result["guidance"], features)
    assert [m.shape[-2:] for m in result["states"]] == [(17, 21), (8, 10), (4, 5)]
    x, y = torch.rand(2, 3, 17, 21), torch.randn(2, 3, 17, 21)
    corrected = DSPC()(x, y)
    assert torch.equal(corrected["corrected"], y)
    assert torch.equal(corrected["gain"], torch.ones_like(y))


def test_bounds_and_interpolation_before_exponentiation():
    head = DSPC(downsample=4)
    torch.nn.init.normal_(head.gain_head.weight, std=2)
    torch.nn.init.normal_(head.bias_head.weight, std=2)
    x, y = torch.rand(1, 3, 15, 17), torch.rand(1, 3, 15, 17)
    out = head(x, y)
    pooled = torch.nn.functional.adaptive_avg_pool2d(torch.cat([x, y], 1), (4, 5))
    coarse = .4*head.gain_head(head.trunk(pooled)).tanh()
    expected = torch.nn.functional.interpolate(coarse, (15, 17), mode="bilinear", align_corners=False).exp()
    torch.testing.assert_close(out["gain"], expected)
    assert out["gain"].min() >= math.exp(-.4) and out["gain"].max() <= math.exp(.4)
    assert out["bias"].abs().max() <= .1
    assert not torch.allclose(expected, torch.nn.functional.interpolate(coarse.exp(), (15, 17), mode="bilinear", align_corners=False))


def test_state_loss_matches_pooled_residual():
    x, y = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    r = (x-y).abs().mean(1, keepdim=True)
    states = {"severity": r.mean((2, 3)), "states": [torch.nn.functional.adaptive_avg_pool2d(r, (n, n)) for n in (16, 8, 4)]}
    assert state_loss(states, x, y).item() == 0
    assert state_loss(states, x, y, mismatch=True).item() > 0


@pytest.mark.parametrize("variant", ["B0", "B1", "B2", "B3", "B4", "B5", "shared", "airnet", "dspc_only"])
def test_variant_training_backward(variant):
    model = RestorationModel(SmokeBackbone(), variant)
    x, y = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    out = model(x, x.flip(-1))
    loss = model.loss(out, x, y)["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.encoder.weight.grad is not None
    if model.correction is not None:
        assert model.correction.bias_head.weight.grad.abs().sum() > 0
    if variant == "B1":
        assert torch.equal(out["gain"], torch.ones_like(x))


@pytest.mark.parametrize("constant", [False, True])
def test_projection_orthogonality_identity_and_box_optimum(constant):
    rng = np.random.default_rng(1)
    x = rng.uniform(-.1, 1.1, (3, 16, 16))
    if constant:
        x[0] = .4
        x[1] = 0
    c = x*1.7+rng.normal(0, .1, x.shape)+.3
    y = rng.random(x.shape)
    s, _, _ = project_affine(x, c)
    d = c-s
    assert np.max(np.abs(d.mean((1, 2)))) < 1e-12
    assert np.max(np.abs((d*x).mean((1, 2)))) < 1e-12
    py, _, _ = project_affine(x, y)
    np.testing.assert_allclose(np.sum((c-y)**2), np.sum((s-py)**2)+np.sum((d-(y-py))**2), rtol=1e-12)
    bounded, a, b = project_affine(x, c, bounded=True)
    # NumPy exp and Python's libm exp can differ by one ULP across platforms.
    tolerance = 2 * np.finfo(a.dtype).eps
    assert np.all((a >= math.exp(-.4)-tolerance) & (a <= math.exp(.4)+tolerance))
    assert np.max(np.abs(b)) <= .1
    for ch in range(3):
        design = np.column_stack([x[ch].ravel(), np.ones(256)])
        fit = lsq_linear(design, c[ch].ravel(), bounds=([math.exp(-.4), -.1], [math.exp(.4), .1]), tol=1e-13)
        np.testing.assert_allclose(np.sum((bounded[ch]-c[ch])**2), np.sum((design@fit.x-c[ch].ravel())**2), atol=1e-10)


def test_energy_alignment_and_oracle():
    rng = np.random.default_rng(4)
    x, c, y = rng.random((3, 3, 16, 16))
    s, _, _ = project_affine(x, c)
    d = c-s
    lam = oracle_strength(s, c, y)
    actual = np.sum((s-y)**2)-np.sum((s+lam*d-y)**2)
    expected = 2*lam*np.sum((y-s)*d)-lam*lam*np.sum(d*d)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.mean((s+lam*d-y)**2) <= min(np.mean((s-y)**2), np.mean((c-y)**2))+1e-12
    assert oracle_strength(s, s, y) == 0
    moved = projected_shift(x, d, (4, 0))
    np.testing.assert_allclose(np.linalg.norm(moved), np.linalg.norm(d), atol=1e-12)
    np.testing.assert_allclose((moved*x).mean((1, 2)), 0, atol=1e-12)


def test_shared_curve_monotonicity_extrapolation_and_color_fit():
    rng = np.random.default_rng(5)
    x = rng.uniform(-.2, 1.2, (3, 16, 16))
    c = x*1.1+.03
    output, nodes = curve_fit(x, c)
    np.testing.assert_allclose(output, c, atol=1e-8)
    assert np.min(np.diff(nodes, axis=1)) >= -1e-10
    color = np.einsum("ij,jhw->ihw", np.eye(3)+.1, x)+.05
    np.testing.assert_allclose(color_matrix_fit(x, color)[0], color, atol=1e-12)


def test_gate_statistics_and_cached_loss():
    x = torch.rand(4, 3, 16, 16)
    gain, bias = torch.rand_like(x)*.5+.7, torch.rand_like(x)*.2-.1
    features = gate_features(x, gain, bias)
    assert features.shape == (4, 24)
    torch.testing.assert_close(features[:, 0:3], x.mean((2, 3)))
    torch.testing.assert_close(features[:, 12:15], gain.mean((2, 3)))
    gate = StrengthPredictor()
    gate.fit_normalizer(features)
    torch.testing.assert_close(gate(features), torch.full((4,), .5))
    y, s, d = torch.rand_like(x), torch.rand_like(x), torch.rand_like(x)
    strength = gate(features)
    r = y-s
    k, t, q = (v.mean((1, 2, 3)) for v in (r*r, r*d, d*d))
    cached = (k-2*strength*t+strength.square()*q).mean()
    direct = ((s+strength[:, None, None, None]*d-y)**2).mean()
    torch.testing.assert_close(cached, direct)


def test_task_budget_leakage_and_macro():
    rows = [{"task": task, "id": str(i), "scene": str(i), "input": f"x{i}", "target": f"y{i}"}
            for i, task in enumerate(("rain", "haze", "snow"))]
    samples = [i for batch in BalancedBatches(rows, 16, 3) for i, _ in batch]
    assert [samples.count(i) for i in range(3)] == [16, 16, 16]
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(rows, rows)
    records = [{"task": "rain", "metrics": {"y_psnr": 30.}}]*10 + [{"task": "haze", "metrics": {"y_psnr": 10.}}]
    assert aggregate(records)["macro"]["y_psnr"] == 20.


def test_metric_clipping_and_luminance():
    x = np.ones((3, 16, 16))
    np.testing.assert_allclose(luminance(x), 235/255)
    metrics = image_metrics(x*2, x)
    assert metrics["y_psnr"] == float("inf")
    assert metrics["rgb_mse_clip"] == 0
    assert metrics["rgb_mse_raw"] == 1
    assert metrics["out_of_range_percent"] == 100
    assert metrics["y_ssim"] == 1


def test_scene_bootstrap_keeps_pairing():
    rows = [{"task": t, "scene": str(i), "scores": {"a": 30+i, "b": 29+i}} for t in ("rain", "haze", "snow") for i in range(5)]
    result = paired_bootstrap(rows, "a", "b", draws=100)
    assert result["mean_difference"] == 1
    assert result["ci95"] == [1, 1]


def test_upstream_cbde_eval_interface():
    """Real upstream encoder on CPU; does not claim a CUDA DGRN integration test."""
    import importlib
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    source = Path(__file__).resolve().parents[1]/"third_party"/"AirNet"
    if not (source/"net"/"encoder.py").is_file():
        pytest.skip("Optional upstream CBDE: run python scripts/fetch_airnet.py first.")
    sys.path.insert(0, str(source))
    encoder = importlib.import_module("net.encoder").CBDE(SimpleNamespace(batch_size=2)).eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 32, 48)
        z, features = encoder(x, x)
        states = RGLDSE()(z, features)
    assert z.shape == (1, 256)
    assert features.shape == states["guidance"].shape == (1, 64, 32, 48)
    assert torch.equal(states["guidance"], features)
