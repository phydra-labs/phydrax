#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _circle_panelization():
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
    return phx.operators.BoundaryPanelization2D(
        geometry.boundary_atlas,
        panels_per_chart=8,
        quadrature_order=6,
        geometry=geometry,
    )


def test_modified_helmholtz_stokes_and_elasticity_2d_layers_are_finite_off_surface():
    panelization = _circle_panelization()
    scalar_density = jnp.ones((panelization.node_count,))
    vector_density = jnp.ones((panelization.node_count, 2))
    target = jnp.asarray((2.0, 0.25))

    modified = phx.operators.ModifiedHelmholtzLayerPotential2D(
        panelization,
        scalar_density,
        decay=1.3,
        minimum_clearance=0.1,
    )
    stokes = phx.operators.StokesLayerPotential2D(
        panelization,
        vector_density,
        viscosity=0.8,
        minimum_clearance=0.1,
    )
    elasticity = phx.operators.ElasticityLayerPotential2D(
        panelization,
        vector_density,
        young_modulus=10.0,
        poisson_ratio=0.25,
        reduction="plane_strain",
        minimum_clearance=0.1,
    )

    assert jnp.isfinite(modified(target))
    assert jnp.all(jnp.isfinite(stokes(target)))
    assert jnp.all(jnp.isfinite(elasticity(target)))
    assert jnp.allclose(
        stokes.kernel.value(target, jnp.zeros((2,))),
        stokes.kernel.value(jnp.zeros((2,)), target),
    )
    assert jnp.allclose(
        elasticity.kernel.value(target, jnp.zeros((2,))),
        elasticity.kernel.value(jnp.zeros((2,)), target),
    )

    cell = phx.discretization.PeriodicCell(
        jnp.eye(2),
        periodic_axes=(True, True),
    )
    kernel = phx.operators.PeriodicScalarSpectralKernel2D(
        cell,
        equation="modified_helmholtz",
        parameter=1.0,
        cutoff=3,
        maximum_modes=100,
    )
    panelization = _circle_panelization()
    potential = phx.operators.PeriodicScalarLayerPotential2D(
        panelization,
        kernel,
        jnp.ones((panelization.node_count,)),
    )

    assert bool(kernel.evidence.successful)
    assert kernel.evidence.mode_count == 49
    assert jnp.isfinite(potential(jnp.asarray((0.25, 0.1))))
