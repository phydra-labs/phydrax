#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_isotropic_plane_stress_closes_p33_and_condenses_tangent():
    material = phx.applications.solid_mechanics.IsotropicPlaneStressMPMConstitutivePlan(
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    )
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        3.0, 11.0
    )
    deformation = jnp.asarray([[[1.12, 0.06], [0.02, 0.93]]])
    history = material.initialize_state((1,), jnp.float64)
    response = material.evaluate(
        deformation, history, jnp.asarray((2.0,)), parameters, 0.0, 0.01
    )
    linearized = material.evaluate_linearized(
        deformation, history, jnp.asarray((2.0,)), parameters, 0.0, 0.01
    )

    assert bool(response.successful[0])
    assert abs(response.diagnostics["plane_stress_residual"][0]) < 1e-9
    assert response.diagnostics["out_of_plane_stretch"][0] > 0.0
    assert bool(linearized.tangent_successful[0])
    assert jnp.all(jnp.isfinite(linearized.algorithmic_tangent))

    def reduced_energy(value):
        return material.evaluate(
            value[None], history, jnp.asarray((2.0,)), parameters, 0.0, 0.01
        ).reference_energy_density[0]

    gradient = jax.grad(reduced_energy)(deformation[0])
    np.testing.assert_allclose(gradient, response.first_piola[0], rtol=2e-8, atol=2e-8)


def test_plane_stress_material_initializes_and_advances_explicit_mpm():
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(10, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.3, 0.3], [0.45, 0.35], [0.35, 0.5]])
    volume = jnp.full((3,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    material = phx.applications.solid_mechanics.IsotropicPlaneStressMPMConstitutivePlan(
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("plane-stress", material),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert state.particles.material_state.shape == (3, 1)
    assert bool(detail.successful)
    assert jnp.all(detail.accepted_state.particles.material_state[:, -1] == 0.0)


def test_finite_strain_j2_yields_dissipates_and_preserves_plastic_volume():
    material = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.FiniteStrainJ2Parameters(
        10.0, 30.0, 0.15, 2.0
    )
    history = material.initialize_state((1,), jnp.float64)
    deformation = jnp.asarray([[[1.0, 0.18, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    response = material.evaluate(
        deformation, history, jnp.asarray((1.0,)), parameters, 0.0, 0.01
    )

    assert bool(response.successful[0])
    assert int(response.branch_code[0]) == 1
    assert response.dissipation_increment[0] > 0.0
    assert response.diagnostics["plastic_multiplier"][0] > 0.0
    assert abs(response.diagnostics["plastic_determinant"][0] - 1.0) < 1e-9
    assert response.trial_state[0, 9] > 0.0

    linearized = material.evaluate_linearized(
        deformation, history, jnp.asarray((1.0,)), parameters, 0.0, 0.01
    )
    assert bool(linearized.tangent_successful[0])


def test_coupled_j2_plane_stress_solves_thickness_and_plastic_branch():
    material = phx.applications.solid_mechanics.finite_strain_j2_plane_stress_plan()
    parameters = phx.applications.solid_mechanics.FiniteStrainJ2Parameters(
        8.0, 25.0, 0.1, 1.0
    )
    history = material.initialize_state((1,), jnp.float64)
    deformation = jnp.asarray([[[1.16, 0.12], [0.0, 0.92]]])
    response = material.evaluate(
        deformation, history, jnp.asarray((1.0,)), parameters, 0.0, 0.01
    )

    assert bool(response.successful[0])
    assert abs(response.diagnostics["plane_stress_residual"][0]) < 1e-8
    assert response.diagnostics["out_of_plane_stretch"][0] > 0.0
    assert response.dissipation_increment[0] >= 0.0
