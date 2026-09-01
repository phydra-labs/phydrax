#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _periodic_spaces(dimension, count):
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(
                count,
                periodic=True,
                endpoint=False,
            )
            for _ in range(dimension)
        ),
        axis_names=names,
    ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))
    spectral = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(dimension)),
        axis_names=names,
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in range(dimension))
    )
    return grid, spectral


def test_periodic_vic_conserves_deposition_and_returns_divergence_free_velocity():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2),
        jnp.ones((2,)),
        ambient_dimension=2,
    ).prepare()
    grid, spectral = _periodic_spaces(2, 16)
    request = phx.discretization.VortexFieldRequest(
        velocity=True,
        velocity_gradient=True,
        vorticity=True,
    )
    prepared = phx.operators.PeriodicVortexInCellPlan(
        particles,
        grid,
        spectral,
        phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(
        source_capacity=2,
        target_capacity=2,
        request=request,
    )
    position = jnp.asarray(((0.25, 0.5), (0.75, 0.5)))
    source = phx.discretization.VortexSourceState(
        position,
        jnp.asarray((1.0, -1.0)),
        core_radius=jnp.full((2,), 0.1),
    )
    target = phx.discretization.VortexTargetState(
        position,
        source_indices=jnp.arange(2),
    )
    evaluation = prepared.evaluate(source, target, request=request)
    diagnostics = evaluation.diagnostics.backend_diagnostics

    np.testing.assert_allclose(diagnostics.deposited_strength, 0.0, atol=1e-14)
    assert diagnostics.balance_defect < 1e-12
    assert diagnostics.divergence_norm < 1e-10
    assert evaluation.velocity_gradient.shape == (2, 2, 2)
    assert evaluation.vorticity.shape == (2,)
    assert bool(evaluation.successful)


def test_periodic_vic_rejects_nonzero_total_vorticity():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2),
        jnp.ones((2,)),
        ambient_dimension=2,
    ).prepare()
    grid, spectral = _periodic_spaces(2, 8)
    prepared = phx.operators.PeriodicVortexInCellPlan(
        particles,
        grid,
        spectral,
        phx.discretization.TensorBSplineSplatAssignment(1),
    ).prepare(source_capacity=2, target_capacity=2)

    position = jnp.asarray(((0.25, 0.5), (0.75, 0.5)))
    source = phx.discretization.VortexSourceState(
        position,
        jnp.asarray((1.0, 0.5)),
        core_radius=jnp.full((2,), 0.1),
    )
    target = phx.discretization.VortexTargetState(
        position,
        source_indices=jnp.arange(2),
    )
    with pytest.raises(Exception, match="zero total integrated vorticity"):
        prepared.evaluate(source, target)


def test_periodic_three_dimensional_vic_returns_velocity_gradient():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2),
        jnp.ones((2,)),
        ambient_dimension=3,
    ).prepare()
    grid, spectral = _periodic_spaces(3, 6)
    request = phx.discretization.VortexFieldRequest(
        velocity=True,
        velocity_gradient=True,
    )
    prepared = phx.operators.PeriodicVortexInCellPlan(
        particles,
        grid,
        spectral,
        phx.discretization.TensorBSplineSplatAssignment(1),
    ).prepare(
        source_capacity=2,
        target_capacity=2,
        request=request,
    )
    position = jnp.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)))
    source = phx.discretization.VortexSourceState(
        position,
        jnp.asarray(((0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        core_radius=jnp.full((2,), 0.15),
    )
    target = phx.discretization.VortexTargetState(
        position,
        source_indices=jnp.arange(2),
    )
    evaluation = prepared.evaluate(source, target, request=request)
    diagnostics = evaluation.diagnostics.backend_diagnostics

    assert evaluation.velocity.shape == (2, 3)
    assert evaluation.velocity_gradient.shape == (2, 3, 3)
    assert diagnostics.divergence_norm < 1e-9
    assert bool(evaluation.successful)
