#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _compiled(
    dimension=2,
    *,
    particle_count=4,
    points=12,
    periodic=True,
    boundary=None,
    clamp_x=False,
    external_acceleration=None,
    external_acceleration_id=None,
):
    axis = tuple(
        phx.discretization.UniformAxisSpec(
            points,
            periodic=periodic,
            endpoint=not periodic,
        )
        for _ in range(dimension)
    )
    bounds = (
        jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,))))
        if periodic
        else jnp.stack(
            (
                -0.25 * jnp.ones((dimension,)),
                1.25 * jnp.ones((dimension,)),
            )
        )
    )
    grid = phx.discretization.TensorGridPlan(
        axis, axis_names=tuple("xyz"[:dimension])
    ).prepare(bounds)
    if clamp_x:
        if boundary is not None:
            raise ValueError("clamp_x and boundary are mutually exclusive.")
        mask = jnp.zeros(grid.vertices().shape + (dimension,), dtype=bool)
        boundary = phx.discretization.PrescribedGridVelocityPlan(
            mask.at[..., 0].set(True)
        )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count),
        jnp.full((particle_count,), 0.01),
        ambient_dimension=dimension,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    domain = phx.discretization.MPMParticleDomainPlan(
        jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))),
        periodic=(periodic,) * dimension,
        support_margin=0.0 if periodic else 0.21,
    )
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(dimension)
    problem = phx.equations.MaterialPointProblemIR(
        "test-solid",
        material,
        external_acceleration=external_acceleration,
        external_acceleration_id=external_acceleration_id,
    )
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        domain,
        boundary=boundary,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    return compiled, arguments


def _positions(dimension):
    base = jnp.asarray(
        [
            [0.28, 0.31, 0.35],
            [0.42, 0.36, 0.44],
            [0.34, 0.49, 0.53],
            [0.48, 0.52, 0.61],
        ]
    )
    return base[:, :dimension]


def test_constant_translation_crosses_grid_without_force_or_state_loss():
    compiled, arguments = _compiled()
    position = _positions(2).at[0, 0].set(0.999)
    velocity = jnp.broadcast_to(jnp.asarray((0.2, -0.05)), position.shape)
    state = compiled.initialize_state(
        position,
        velocity,
        jnp.full((4,), 0.01),
        arguments,
    )
    step_size = jnp.asarray(0.006)

    detail = jax.jit(
        lambda value: compiled.dynamics.step_detailed(value, step_size, arguments)
    )(state)

    assert bool(detail.successful)
    np.testing.assert_allclose(
        detail.accepted_state.particles.position,
        position + step_size * velocity,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.velocity, velocity, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.deformation_gradient,
        jnp.broadcast_to(jnp.eye(2), (4, 2, 2)),
        rtol=2e-11,
        atol=2e-11,
    )
    assert detail.diagnostics.transfer.relative_mass_defect < 1e-12
    assert detail.diagnostics.transfer.relative_momentum_defect < 1e-10
    assert not bool(detail.diagnostics.transfer.angular_momentum_valid)
    assert detail.diagnostics.transfer.relative_angular_momentum_defect == 0.0
    assert int(detail.accepted_state.accepted_step) == 1


def test_nonperiodic_apic_transfer_certifies_angular_momentum():
    compiled, arguments = _compiled(periodic=False)
    position = _positions(2)
    velocity = jnp.broadcast_to(jnp.asarray((0.08, -0.03)), position.shape)
    state = compiled.initialize_state(
        position,
        velocity,
        jnp.full((4,), 0.01),
        arguments,
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    assert bool(detail.diagnostics.transfer.angular_momentum_valid)
    assert detail.diagnostics.transfer.relative_angular_momentum_defect < 1e-10


def test_prescribed_grid_velocity_reports_impulse_and_work():
    compiled, arguments = _compiled(clamp_x=True)
    position = _positions(2)
    velocity = jnp.broadcast_to(jnp.asarray((0.1, 0.0)), position.shape)
    state = compiled.initialize_state(
        position,
        velocity,
        jnp.full((4,), 0.01),
        arguments,
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    np.testing.assert_allclose(
        detail.accepted_state.particles.velocity[:, 0], 0.0, atol=1e-12
    )
    assert detail.diagnostics.energy.boundary_work < 0.0
    assert detail.diagnostics.energy.grid_kinetic_after == 0.0


def test_external_acceleration_updates_momentum_and_work():
    def acceleration(time, position, velocity, args):
        del time, velocity, args
        return jnp.broadcast_to(jnp.asarray((0.0, -0.2)), position.shape)

    compiled, arguments = _compiled(
        external_acceleration=acceleration,
        external_acceleration_id="constant-downward",
    )
    position = _positions(2)
    state = compiled.initialize_state(
        position,
        jnp.zeros_like(position),
        jnp.full((4,), 0.01),
        arguments,
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(detail.successful)
    np.testing.assert_allclose(
        detail.accepted_state.particles.velocity[:, 1],
        -2.0e-4,
        rtol=1e-10,
        atol=1e-12,
    )
    assert detail.diagnostics.energy.external_work > 0.0


def test_apic_affine_field_reproduces_velocity_gradient_and_deformation():
    compiled, arguments = _compiled()
    position = _positions(2)
    affine = jnp.asarray([[0.04, -0.02], [0.03, -0.01]])
    offset = jnp.asarray((0.2, -0.1))
    velocity = jax.vmap(lambda point: affine @ point + offset)(position)
    affine_state = jnp.broadcast_to(affine, (4, 2, 2))
    state = compiled.initialize_state(
        position,
        velocity,
        jnp.full((4,), 0.01),
        arguments,
        affine_velocity=affine_state,
    )
    dt = jnp.asarray(0.005)
    detail = compiled.dynamics.step_detailed(state, dt, arguments)

    assert bool(detail.successful)
    expected_deformation = jnp.broadcast_to(jnp.eye(2) + dt * affine, (4, 2, 2))
    np.testing.assert_allclose(
        detail.accepted_state.particles.affine_velocity,
        affine_state,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.deformation_gradient,
        expected_deformation,
        rtol=2e-10,
        atol=2e-10,
    )
    assert detail.diagnostics.transfer.maximum_apic_condition <= 1.0 + 1e-10


def test_oversized_step_rejects_without_mutating_accepted_particle_state():
    compiled, arguments = _compiled()
    state = compiled.initialize_state(
        _positions(2),
        jnp.zeros((4, 2)),
        jnp.full((4,), 0.01),
        arguments,
    )
    detail = compiled.dynamics.step_detailed(state, 10.0, arguments)

    assert not bool(detail.successful)
    assert int(detail.accepted_state.last_status) == int(
        phx.discretization.MPMRunStatus.STABILITY_LIMIT_EXCEEDED
    )
    assert int(detail.rejection_reasons) & int(
        phx.discretization.MPMRejectionReason.STABILITY
    )
    assert int(detail.accepted_state.accepted_step) == 0
    assert detail.accepted_state.time == state.time
    for accepted, initial in zip(
        jax.tree.leaves(detail.accepted_state.particles),
        jax.tree.leaves(state.particles),
        strict=True,
    ):
        np.testing.assert_array_equal(accepted, initial)


def test_particle_support_domain_rejection_is_transactional():
    compiled, arguments = _compiled(periodic=False)
    state = compiled.initialize_state(
        _positions(2),
        jnp.zeros((4, 2)),
        jnp.full((4,), 0.01),
        arguments,
    )
    outside = state.particles.position.at[0, 0].set(1.1)
    invalid_particles = phx.discretization.MPMParticleState(
        outside,
        state.particles.velocity,
        state.particles.deformation_gradient,
        state.particles.affine_velocity,
        state.particles.reference_volume,
        state.particles.first_piola,
        state.particles.reference_energy_density,
        state.particles.maximum_wave_speed,
        state.particles.material_state,
    )
    invalid = phx.discretization.MPMRuntimeState(
        invalid_particles,
        state.time,
        state.accepted_step,
        state.last_status,
    )
    detail = compiled.dynamics.step_detailed(invalid, 0.001, arguments)

    assert not bool(detail.successful)
    assert int(detail.accepted_state.last_status) == int(
        phx.discretization.MPMRunStatus.DOMAIN_REJECTED
    )
    np.testing.assert_array_equal(
        detail.accepted_state.particles.position, invalid.particles.position
    )


def test_three_dimensional_step_is_finite_and_conservative():
    compiled, arguments = _compiled(3, points=10)
    position = _positions(3)
    velocity = jnp.broadcast_to(jnp.asarray((0.04, -0.02, 0.03)), position.shape)
    state = compiled.initialize_state(
        position,
        velocity,
        jnp.full((4,), 0.01),
        arguments,
    )
    detail = jax.jit(
        lambda value: compiled.dynamics.step_detailed(value, 0.002, arguments)
    )(state)

    assert bool(detail.successful)
    assert jax.tree.all(
        jax.tree.map(
            lambda value: jnp.all(jnp.isfinite(value)), detail.accepted_state.particles
        )
    )
    assert detail.diagnostics.transfer.relative_mass_defect < 1e-12
    assert detail.diagnostics.transfer.relative_momentum_defect < 1e-10
