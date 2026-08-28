#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_spherical_space_composes_laplacian_noise_kernel_and_sfno():
    space = phx.discretization.SphericalSpectralPlan(4).prepare()
    plan = space.transform
    theta, phi = jnp.meshgrid(plan.theta, plan.phi, indexing="ij")
    values = jnp.cos(theta) + 0.2 * jnp.sin(theta) * jnp.cos(phi)

    reconstructed = space.reconstruct(space.project(values))
    laplacian = space.laplacian(values)
    noise = phx.stochastic.SpatialNoiseBasis.from_spectrum(
        space,
        lambda eigenvalues: jnp.exp(-0.1 * eigenvalues),
        rank=4,
    )
    kernel = phx.kernels.SphereSpectralKernel.from_discretization(
        space,
        phx.kernels.HeatSpectralMultiplier(0.1),
    )
    axes = (
        phx.nn.operator.OperatorAxis(
            "theta",
            plan.theta,
            quadrature_weights=plan.theta_quadrature_weights,
            basis="sphere",
        ),
        phx.nn.operator.OperatorAxis(
            "phi",
            plan.phi,
            quadrature_weights=plan.phi_quadrature_weights,
            basis="fourier",
            periodic=True,
        ),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"u": phx.nn.operator.FunctionSamples(values=values, axes=axes)},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=axes)},
    )
    model = phx.nn.operator.architectures.SFNO(
        space,
        width=4,
        depth=1,
        source_key="u",
        key=jr.key(2),
    )
    prediction = model(batch)
    kernel_matrix = kernel.matrix(space.points[:4], space.points[:4])

    assert jnp.allclose(reconstructed, values, rtol=1e-10, atol=1e-10)
    assert jnp.all(jnp.isfinite(laplacian))
    assert jnp.all(jnp.isfinite(noise.diffusion))
    assert jnp.all(jnp.isfinite(kernel_matrix))
    assert jnp.all(jnp.isfinite(prediction))
    assert prediction.shape == space.state_shape
    assert noise.field_space_id == space.physical_space.field_space_id
