import jax.numpy as jnp

import phydrax as phx
from phydrax.operators.interpolation import (
    fit_mixed_tensor,
    fourier_type1,
    fourier_type2,
    MixedTensorReconstructionPlan,
)


def test_cid09_type1_is_algebraic_transpose_of_type2():
    phases = jnp.asarray([[-1.1], [-0.2], [0.4], [1.7]])
    coefficients = jnp.asarray([1.0 + 0.2j, -0.3j, 0.7, 0.1 + 0.5j])
    values = jnp.asarray([0.5 - 0.1j, 0.2, -0.7j, 1.1])
    forward = fourier_type2(phases, coefficients, method="direct", tolerance=1e-12)
    transpose = fourier_type1(phases, values, (4,), method="direct", tolerance=1e-12)
    assert jnp.allclose(
        jnp.dot(forward, values), jnp.vdot(coefficients.conj(), transpose)
    )


def test_cid10_mixed_fourier_chebyshev_reconstruction_round_trip():
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(5),
            phx.discretization.ChebyshevBasisPlan(6),
        ),
        axis_names=("theta", "x"),
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
        )
    )
    plan = MixedTensorReconstructionPlan(space.axes, ("theta", "x"))
    theta, x = jnp.meshgrid(space.axes[0].nodes, space.axes[1].nodes, indexing="ij")
    values = jnp.cos(2.0 * theta) + 0.5 * x**3
    interpolant = fit_mixed_tensor(values, plan)
    paired = jnp.stack((theta.reshape(-1), x.reshape(-1)), axis=-1)
    assert jnp.allclose(interpolant(paired).real, values.reshape(-1), atol=1e-6)
