#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _compiled(material):
    dimension = material.dimension
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False)
            for _ in range(dimension)
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))
    position = jnp.asarray([[0.27, 0.31, 0.35], [0.43, 0.38, 0.44], [0.36, 0.52, 0.57]])[
        :, :dimension
    ]
    volume = jnp.full((3,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), volume, ambient_dimension=dimension
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("implicit", material),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))),
            periodic=(True,) * dimension,
            support_margin=0.0,
        ),
    )
    return compiled, position, volume


def test_implicit_hyperelastic_step_converges_and_has_implicit_gradient():
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    compiled, position, volume = _compiled(material)
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    velocity = jnp.broadcast_to(jnp.asarray((0.03, -0.01)), position.shape)
    initial = compiled.initialize_state(position, velocity, volume, arguments)
    implicit = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics)
    detail = implicit.step_detailed(initial, 0.001, arguments)

    assert bool(detail.successful)
    assert detail.diagnostics.residual_norm < 1e-8
    assert bool(detail.diagnostics.tangent_successful)
    assert detail.accepted_state.time == 0.001

    def objective(scale):
        state = compiled.initialize_state(position, scale * velocity, volume, arguments)
        result = implicit.step_detailed(state, 0.001, arguments)
        return jnp.sum(result.accepted_state.particles.position**2)

    derivative = jax.grad(objective)(jnp.asarray(1.0))
    assert jnp.isfinite(derivative)


def test_implicit_plane_stress_uses_condensed_material_tangent():
    material = phx.applications.solid_mechanics.IsotropicPlaneStressMPMConstitutivePlan(
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    )
    compiled, position, volume = _compiled(material)
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    initial = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    detail = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).step_detailed(
        initial, 0.001, arguments
    )

    assert bool(detail.successful)
    assert bool(detail.diagnostics.tangent_successful)
    assert jnp.all(detail.accepted_state.particles.material_state[:, -1] == 0.0)


def test_implicit_j2_accepts_elastic_step_and_preserves_history_shape():
    material = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    compiled, position, volume = _compiled(material)
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.FiniteStrainJ2Parameters(8.0, 25.0, 10.0, 1.0)
    )
    initial = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    detail = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).step_detailed(
        initial, 0.0005, arguments
    )

    assert bool(detail.successful)
    assert detail.accepted_state.particles.material_state.shape == (3, 10)
    np.testing.assert_allclose(
        detail.accepted_state.particles.material_state[:, 9], 0.0, atol=1e-12
    )
