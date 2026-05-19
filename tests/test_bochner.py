"""Tests for the Bochner time-encoding layer."""
from __future__ import annotations

import math

import pytest


@pytest.fixture
def bochner():
    pytest.importorskip("torch")
    try:
        from mambaguard.models.bochner_encoding import BochnerTimeEncoding
    except Exception as exc:
        pytest.skip(f"BochnerTimeEncoding unavailable: {exc}")
    return BochnerTimeEncoding(d_T=16)


def test_output_shape(bochner):
    import torch

    t = torch.linspace(0, 100, 32)
    out = bochner(t)
    assert out.shape == (32, 16)


def test_periodicity_per_frequency(bochner):
    """For each learnable ω, Φ(Δt + 2π/ω) ≈ Φ(Δt) on the matching column."""
    import torch

    omega = bochner.omega.detach().abs()
    assert omega.numel() > 0
    omega_val = float(omega[0].item())
    if omega_val <= 0.0:
        pytest.skip("non-positive omega init")
    period = 2.0 * math.pi / omega_val
    t = torch.tensor([0.0, 1.0, 2.0])
    out1 = bochner(t)
    out2 = bochner(t + period)
    # Column 0 corresponds to cos(omega[0]·Δt) up to scale; check it.
    assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-3)
