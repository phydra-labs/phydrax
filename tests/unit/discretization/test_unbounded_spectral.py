import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _fourier(count):
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))


def test_hilbert_multiplier_zero_mean_nyquist_and_square_law():
    space = _fourier(16)
    x = space.axes[0].nodes
    values = 0.4 + jnp.cos(2.0 * jnp.pi * x) + (-1.0) ** jnp.arange(16)
    coefficients = space.project(values)
    hilbert = phx.discretization.spectral_hilbert_operator(space, 0)
    transformed = hilbert(coefficients)
    twice = hilbert(transformed)
    supported = coefficients.at[space.axes[0].modes.zero_mask].set(0.0)
    supported = supported.at[space.axes[0].modes.nyquist_mask].set(0.0)

    np.testing.assert_allclose(
        space.reconstruct(transformed),
        jnp.sin(2.0 * jnp.pi * x),
        atol=2e-12,
    )
    np.testing.assert_allclose(twice, -supported, atol=2e-12)
    assert jnp.all(transformed[space.axes[0].modes.zero_mask] == 0.0)
    assert jnp.all(transformed[space.axes[0].modes.nyquist_mask] == 0.0)


def test_modal_transfer_preserves_fourier_modes_and_constrained_traces():
    coarse = _fourier(8)
    fine = _fourier(12)
    x = coarse.axes[0].nodes
    coefficients = coarse.project(jnp.sin(2.0 * jnp.pi * x))
    transfer = phx.discretization.prepare_spectral_modal_transfer(coarse, fine)
    restored = phx.discretization.prepare_spectral_modal_transfer(fine, coarse)(
        transfer(coefficients)
    )
    np.testing.assert_allclose(restored, coefficients, atol=2e-12)
    payload = jnp.stack((coefficients, 2.0 * coefficients), axis=-1)
    restored_payload = phx.discretization.prepare_spectral_modal_transfer(fine, coarse)(
        transfer(payload)
    )
    np.testing.assert_allclose(restored_payload, payload, atol=2e-12)
    assert transfer.report.lossless

    domain = phx.discretization.AxisDomain.interval(-1.0, 1.0)
    boundary = phx.discretization.SpectralBoundaryConditionPlan.dirichlet()
    constrained_coarse = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ConstrainedBasisPlan(
                phx.discretization.LegendreBasisPlan(6),
                boundary,
            ),
        )
    ).prepare((domain,))
    constrained_fine = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ConstrainedBasisPlan(
                phx.discretization.LegendreBasisPlan(8),
                boundary,
            ),
        )
    ).prepare((domain,))
    constrained_transfer = phx.discretization.prepare_spectral_modal_transfer(
        constrained_coarse,
        constrained_fine,
    )
    assert constrained_transfer.report.trace_residual < 1e-10


def test_modal_decay_uses_physical_norm_and_detects_empty_tail():
    space = _fourier(16)
    x = space.axes[0].nodes
    coefficients = space.project(jnp.sin(2.0 * jnp.pi * x))
    prepared = phx.discretization.SpectralModalDiagnosticsPlan(space).prepare()
    report = eqx.filter_jit(prepared.evaluate)(coefficients)
    zero = prepared.evaluate(jnp.zeros_like(coefficients))

    np.testing.assert_allclose(
        report.total_norm,
        1.0 / jnp.sqrt(2.0),
        atol=2e-12,
    )
    assert report.relative_tail_norms[0] < 1e-12
    assert bool(report.finite)
    assert bool(zero.zero_reference_norm)
    assert zero.relative_tail_norms[0] == 0.0


def test_rational_line_represents_algebraic_tail_and_projected_derivative():
    domain = phx.discretization.AxisDomain.real_line()
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.RationalChebyshevLineBasisPlan(8, 1.0),),
        axis_names=("y",),
    ).prepare((domain,))
    y = space.axes[0].nodes
    values = 1.0 / (1.0 + y**2)
    coefficients = space.project(values)
    derivative = jax.jit(lambda value: space.derivative_values(value, axis=0))(
        coefficients
    )

    np.testing.assert_allclose(space.reconstruct(coefficients), values, atol=2e-12)
    np.testing.assert_allclose(
        derivative,
        -2.0 * y / (1.0 + y**2) ** 2,
        atol=3e-11,
    )
    assert not space.axes[0].derivative_exact
    assert space.axes[0].derivative_residual > 0.0
    assert jnp.all(jnp.isfinite(space.axes[0].nodes))
    assert jnp.all(jnp.isfinite(space.quadrature_weights))


def test_rational_half_line_quadrature_and_scale_identity():
    domain = phx.discretization.AxisDomain.half_line(0.0)
    first = phx.discretization.TensorSpectralPlan(
        (phx.discretization.RationalChebyshevHalfLineBasisPlan(20, 2.0),)
    ).prepare((domain,))
    second = phx.discretization.TensorSpectralPlan(
        (phx.discretization.RationalChebyshevHalfLineBasisPlan(20, 3.0),)
    ).prepare((domain,))
    integral = jnp.sum(first.quadrature_weights * jnp.exp(-first.axes[0].nodes))

    np.testing.assert_allclose(integral, 1.0, atol=2e-8)
    assert first.prepared_id != second.prepared_id
    with pytest.raises(ValueError, match="same map family and scale"):
        phx.discretization.prepare_spectral_modal_transfer(first, second)


def test_linear_trace_constraints_support_robin_and_decay():
    domain = phx.discretization.AxisDomain.interval(-1.0, 1.0)
    robin = phx.discretization.SpectralBoundaryConditionPlan.robin(
        lower=(1.0, 1.0),
        upper=(1.0, -1.0),
    )
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ConstrainedBasisPlan(
                phx.discretization.LegendreBasisPlan(8),
                robin,
            ),
        )
    ).prepare((domain,))
    assert space.axes[0].boundary == "homogeneous_trace"

    line = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ConstrainedBasisPlan(
                phx.discretization.RationalChebyshevLineBasisPlan(8, 2.0),
                phx.discretization.SpectralBoundaryConditionPlan.decay(),
            ),
        )
    ).prepare((phx.discretization.AxisDomain.real_line(),))
    assert line.axes[0].boundary == "decay"
    assert line.axes[0].derivative_matrix is not None


def test_constrained_polynomial_physical_laplacian_rejects_modal_closure():
    domain = phx.discretization.AxisDomain.interval(-1.0, 1.0)
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ConstrainedBasisPlan(
                phx.discretization.LegendreBasisPlan(8),
                phx.discretization.SpectralBoundaryConditionPlan.dirichlet(),
            ),
        )
    ).prepare((domain,))
    nodes = space.axes[0].nodes
    values = 1.0 - nodes**2
    coefficients = space.project(values)

    np.testing.assert_allclose(space.laplacian(values), -2.0, atol=2e-10)
    with pytest.raises(ValueError, match="closed modal derivative"):
        space.modal_laplacian(coefficients)
    with pytest.raises(ValueError, match="closed modal derivative"):
        phx.discretization.spectral_laplacian_operator(space)
