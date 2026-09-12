#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.discrete_velocity._quadrature import d2v17_quadrature
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial_boundary import (
    EquilibriumReservoirD2VBoundaryPlan,
    MaxwellThermalD2VBoundaryPlan,
    OutwardExtrapolationD2VBoundaryPlan,
    PeriodicD2VBoundaryPlan,
    SmoothCompressibleD2VBoundaryStatus,
    SmoothCompressibleD2VLinkOwner,
    SmoothCompressibleD2VReservoirParameters,
    SpecularAdiabaticD2VBoundaryPlan,
)


def _quadrature():
    return d2v17_quadrature(dtype=jnp.float64)


def _direction(quadrature, velocity):
    matches = np.all(np.asarray(quadrature.velocities) == np.asarray(velocity), axis=1)
    return int(np.flatnonzero(matches)[0])


def _unique_state(shape=(6, 6)):
    count = int(np.prod(shape) * 17)
    particles = jnp.arange(1, count + 1, dtype=jnp.float64).reshape(shape + (17,))
    energy = 2.0 * particles + 1.0
    return SmoothCompressibleKineticState(particles, energy)


def _uniform_state(particles, energy, shape=(6, 6)):
    return SmoothCompressibleKineticState(
        jnp.broadcast_to(jnp.asarray(particles), shape + (17,)),
        jnp.broadcast_to(jnp.asarray(energy), shape + (17,)),
    )


def _content(plan, state):
    volume = plan.topology.cell_volume
    velocities = plan.topology.quadrature.velocities
    mass = jnp.sum(state.particle_populations) * volume
    momentum = (
        jnp.sum(
            jnp.einsum("xyq,qd->xyd", state.particle_populations, velocities),
            axis=(0, 1),
        )
        * volume
    )
    energy = jnp.sum(state.total_energy_populations) * volume
    return jnp.concatenate((mass[None], momentum, energy[None]))


def test_d2v17_compiled_topology_owns_long_links_and_double_reflects_corners():
    quadrature = _quadrature()
    plan = SpecularAdiabaticD2VBoundaryPlan(
        quadrature, (6, 6), (1.0, 1.0), 1.0, retain_history=True
    )
    topology = plan.topology
    incoming = _direction(quadrature, (2, 2))
    reflected = _direction(quadrature, (-2, -2))

    assert topology.pull_offsets[incoming] == (2, 2)
    assert tuple(np.asarray(topology.maximum_reach)) == (2, 2)
    assert int(topology.owner[0, 0, incoming]) == int(
        SmoothCompressibleD2VLinkOwner.SPECULAR_ADIABATIC_WALL
    )
    assert (
        int(topology.source_x[0, 0, incoming]),
        int(topology.source_y[0, 0, incoming]),
    ) == (
        1,
        1,
    )
    assert int(topology.source_direction[0, 0, incoming]) == reflected
    assert bool(topology.physical_axis_mask[0, 0, incoming, 0])
    assert bool(topology.physical_axis_mask[0, 0, incoming, 1])
    assert sum(topology.owner_counts) == int(np.prod(topology.population_shape))
    assert np.all(
        np.isin(
            np.asarray(topology.owner),
            [int(owner) for owner in SmoothCompressibleD2VLinkOwner],
        )
    )

    state = _unique_state()
    result = plan.route(state)
    assert bool(result.successful)
    assert result.history is not None
    np.testing.assert_array_equal(
        result.candidate_state.particle_populations[0, 0, incoming],
        state.particle_populations[1, 1, reflected],
    )
    np.testing.assert_array_equal(
        result.candidate_state.total_energy_populations[0, 0, incoming],
        state.total_energy_populations[1, 1, reflected],
    )


def test_periodic_axis_wraps_before_the_other_axis_applies_specular_law():
    quadrature = _quadrature()
    plan = SpecularAdiabaticD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        periodic_axes=(True, False),
    )
    state = _unique_state()
    incoming = _direction(quadrature, (2, 2))
    y_reflected = _direction(quadrature, (2, -2))

    result = plan.route(state)

    assert bool(result.successful)
    assert result.history is None
    np.testing.assert_array_equal(
        result.candidate_state.particle_populations[0, 0, incoming],
        state.particle_populations[4, 1, y_reflected],
    )
    np.testing.assert_array_equal(
        result.candidate_state.total_energy_populations[0, 0, incoming],
        state.total_energy_populations[4, 1, y_reflected],
    )


def test_periodic_plan_matches_exact_coupled_pull_and_has_no_boundary_exchange():
    quadrature = _quadrature()
    plan = PeriodicD2VBoundaryPlan(quadrature, (6, 6), (1.0, 1.0), 1.0)
    state = _unique_state()

    result = plan.route(state)
    expected_particles = jnp.stack(
        tuple(
            jnp.roll(state.particle_populations[..., q], shift=offset, axis=(0, 1))
            for q, offset in enumerate(plan.topology.pull_offsets)
        ),
        axis=-1,
    )
    expected_energy = jnp.stack(
        tuple(
            jnp.roll(state.total_energy_populations[..., q], shift=offset, axis=(0, 1))
            for q, offset in enumerate(plan.topology.pull_offsets)
        ),
        axis=-1,
    )

    assert bool(result.successful)
    np.testing.assert_array_equal(
        result.candidate_state.particle_populations, expected_particles
    )
    np.testing.assert_array_equal(
        result.candidate_state.total_energy_populations, expected_energy
    )
    np.testing.assert_allclose(result.mass_momentum_energy_exchange, 0.0, atol=1.0e-10)


def test_matching_uniform_reservoir_is_stationary_without_model_evaluation():
    quadrature = _quadrature()
    particles = jnp.arange(1, 18, dtype=jnp.float64) / 32.0
    energy = jnp.arange(18, 35, dtype=jnp.float64) / 16.0
    state = _uniform_state(particles, energy)
    plan = EquilibriumReservoirD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        incoming_particle_populations=particles,
        incoming_total_energy_populations=energy,
    )

    result = plan.route(state)

    assert bool(result.successful)
    assert int(result.status) == int(SmoothCompressibleD2VBoundaryStatus.SUCCESS)
    assert result.history is None
    np.testing.assert_array_equal(
        result.candidate_state.particle_populations, state.particle_populations
    )
    np.testing.assert_array_equal(
        result.candidate_state.total_energy_populations,
        state.total_energy_populations,
    )
    np.testing.assert_array_equal(result.mass_momentum_energy_exchange, jnp.zeros((4,)))


def test_reservoir_replaces_incoming_links_only_under_exclusive_ownership():
    quadrature = _quadrature()
    state = _unique_state()
    particles = 1000.0 + jnp.arange(17, dtype=jnp.float64)
    energy = 2000.0 + jnp.arange(17, dtype=jnp.float64)
    plan = EquilibriumReservoirD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        incoming_particle_populations=particles,
        incoming_total_energy_populations=energy,
    )

    result = plan.route(state)
    reservoir = plan.topology.owner == int(
        SmoothCompressibleD2VLinkOwner.EQUILIBRIUM_RESERVOIR
    )
    directions = jnp.broadcast_to(jnp.arange(17)[None, None, :], reservoir.shape)
    gathered_particles = plan.topology.gather(state.particle_populations)
    gathered_energy = plan.topology.gather(state.total_energy_populations)

    assert bool(result.successful)
    np.testing.assert_array_equal(
        result.candidate_state.particle_populations[reservoir],
        particles[directions[reservoir]],
    )
    np.testing.assert_array_equal(
        result.candidate_state.total_energy_populations[reservoir],
        energy[directions[reservoir]],
    )
    np.testing.assert_array_equal(
        result.candidate_state.particle_populations[~reservoir],
        gathered_particles[~reservoir],
    )
    np.testing.assert_array_equal(
        result.candidate_state.total_energy_populations[~reservoir],
        gathered_energy[~reservoir],
    )


def test_stationary_specular_wall_has_zero_mass_energy_flux_and_normal_impulse():
    quadrature = _quadrature()
    plan = SpecularAdiabaticD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        periodic_axes=(True, False),
    )
    state = _unique_state()

    result = plan.route(state)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.mass_momentum_energy_exchange[jnp.asarray((0, 1, 3))],
        jnp.asarray((0.0, 0.0, 0.0)),
        atol=1.0e-10,
    )
    assert abs(float(result.mass_momentum_energy_exchange[2])) > 0.0
    np.testing.assert_allclose(
        result.wall_momentum_impulse + result.mass_momentum_energy_exchange[1:3],
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(result.heat_exchange, jnp.asarray(0.0))
    np.testing.assert_array_equal(result.wall_work, jnp.asarray(0.0))


def test_reservoir_ledger_is_actual_accepted_population_content_change():
    quadrature = _quadrature()
    state = _unique_state()
    face_particles = jnp.stack(
        tuple(10.0 * (face + 1) + jnp.arange(17) for face in range(4))
    ).astype(jnp.float64)
    face_energy = 3.0 * face_particles
    plan = EquilibriumReservoirD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (0.5, 0.5),
        0.5,
        periodic_axes=(False, True),
        incoming_particle_populations=face_particles,
        incoming_total_energy_populations=face_energy,
    )

    result = plan.route(state)
    actual = _content(plan, result.accepted_state) - _content(plan, state)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.mass_momentum_energy_exchange, actual, rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(result.wall_momentum_impulse, jnp.zeros((2,)))
    np.testing.assert_array_equal(result.heat_exchange, jnp.asarray(0.0))
    np.testing.assert_array_equal(result.wall_work, jnp.asarray(0.0))


def test_incompatible_reservoir_corner_refuses_without_explicit_corner_data():
    quadrature = _quadrature()
    state = _uniform_state(jnp.ones((17,)), 2.0 * jnp.ones((17,)))
    face_particles = jnp.stack(
        tuple(jnp.full((17,), float(face + 1)) for face in range(4))
    )
    face_energy = 2.0 * face_particles
    plan = EquilibriumReservoirD2VBoundaryPlan(quadrature, (6, 6), (1.0, 1.0), 1.0)
    incompatible = SmoothCompressibleD2VReservoirParameters(face_particles, face_energy)

    refused = plan.route(state, incompatible)

    assert not bool(refused.successful)
    assert int(refused.status) == int(
        SmoothCompressibleD2VBoundaryStatus.INCOMPATIBLE_RESERVOIR_CORNER
    )
    assert bool(refused.rollback_applied)
    np.testing.assert_array_equal(
        refused.accepted_state.particle_populations, state.particle_populations
    )
    np.testing.assert_array_equal(
        refused.accepted_state.total_energy_populations,
        state.total_energy_populations,
    )
    np.testing.assert_array_equal(refused.mass_momentum_energy_exchange, jnp.zeros((4,)))

    explicit = SmoothCompressibleD2VReservoirParameters(
        face_particles,
        face_energy,
        corner_particle_populations=jnp.full((4, 17), 8.0),
        corner_total_energy_populations=jnp.full((4, 17), 16.0),
    )
    accepted = plan.route(state, explicit)
    assert bool(accepted.successful)


def test_invalid_reservoir_target_rolls_back_both_population_fields_atomically():
    quadrature = _quadrature()
    state = _uniform_state(jnp.ones((17,)), 2.0 * jnp.ones((17,)))
    invalid_particles = jnp.ones((4, 17)).at[0, 1].set(-1.0)
    parameters = SmoothCompressibleD2VReservoirParameters(
        invalid_particles, 2.0 * jnp.ones((4, 17))
    )
    plan = EquilibriumReservoirD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        periodic_axes=(False, True),
    )

    result = plan.route(state, parameters)

    assert not bool(result.successful)
    assert int(result.status) == int(SmoothCompressibleD2VBoundaryStatus.INVALID_TARGET)
    assert bool(result.rollback_applied)
    np.testing.assert_array_equal(
        result.accepted_state.particle_populations, state.particle_populations
    )
    np.testing.assert_array_equal(
        result.accepted_state.total_energy_populations,
        state.total_energy_populations,
    )
    np.testing.assert_array_equal(result.mass_momentum_energy_exchange, jnp.zeros((4,)))


def test_zero_gradient_outflow_accepts_outward_flow_and_refuses_backflow_atomically():
    quadrature = _quadrature()
    positive_x = _direction(quadrature, (1, 0))
    negative_x = _direction(quadrature, (-1, 0))
    base_particles = jnp.ones((6, 6, 17), dtype=jnp.float64)
    base_energy = 2.0 * jnp.ones((6, 6, 17), dtype=jnp.float64)
    outward_particles = base_particles.at[:2, :, negative_x].add(1.0)
    outward_particles = outward_particles.at[-2:, :, positive_x].add(1.0)
    outward_state = SmoothCompressibleKineticState(outward_particles, base_energy)
    plan = OutwardExtrapolationD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        periodic_axes=(False, True),
    )

    accepted = plan.route(outward_state)
    assert bool(accepted.successful)
    assert int(accepted.status) == int(SmoothCompressibleD2VBoundaryStatus.SUCCESS)

    backflow_particles = base_particles.at[:2, :, positive_x].add(1.0)
    backflow_particles = backflow_particles.at[-2:, :, positive_x].add(1.0)
    backflow_state = SmoothCompressibleKineticState(backflow_particles, base_energy)
    refused = plan.route(backflow_state)

    assert not bool(refused.successful)
    assert int(refused.status) == int(SmoothCompressibleD2VBoundaryStatus.BACKFLOW)
    assert bool(refused.rollback_applied)
    np.testing.assert_array_equal(
        refused.accepted_state.particle_populations,
        backflow_state.particle_populations,
    )
    np.testing.assert_array_equal(
        refused.accepted_state.total_energy_populations,
        backflow_state.total_energy_populations,
    )
    np.testing.assert_array_equal(refused.mass_momentum_energy_exchange, jnp.zeros((4,)))


def test_maxwell_zero_accommodation_is_the_specular_limit():
    quadrature = _quadrature()
    diffuse_particles = quadrature.weights
    diffuse_energy = 3.0 * diffuse_particles
    maxwell = MaxwellThermalD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        diffuse_particle_populations=diffuse_particles,
        diffuse_total_energy_populations=diffuse_energy,
        accommodation=0.0,
        wall_velocity=jnp.zeros((2,)),
        periodic_axes=(False, True),
    )
    specular = SpecularAdiabaticD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        periodic_axes=(False, True),
    )
    state = _unique_state()

    maxwell_result = maxwell.route(state)
    specular_result = specular.route(state)

    assert bool(maxwell_result.successful)
    np.testing.assert_array_equal(
        maxwell_result.candidate_state.particle_populations,
        specular_result.candidate_state.particle_populations,
    )
    np.testing.assert_array_equal(
        maxwell_result.candidate_state.total_energy_populations,
        specular_result.candidate_state.total_energy_populations,
    )


def test_maxwell_diffuse_density_closes_mass_and_splits_heat_from_wall_work():
    quadrature = _quadrature()
    diffuse_particles = quadrature.weights
    diffuse_energy = 3.0 * diffuse_particles
    wall_velocity = jnp.asarray(((0.0, 0.2), (0.0, -0.1), (0.0, 0.0), (0.0, 0.0)))
    plan = MaxwellThermalD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        diffuse_particle_populations=diffuse_particles,
        diffuse_total_energy_populations=diffuse_energy,
        accommodation=0.75,
        wall_velocity=wall_velocity,
        periodic_axes=(False, True),
    )
    state = _unique_state()

    result = plan.route(state)

    assert bool(result.successful)
    assert jnp.all(result.candidate_state.particle_populations > 0.0)
    assert jnp.all(result.candidate_state.total_energy_populations > 0.0)
    np.testing.assert_allclose(
        result.mass_momentum_energy_exchange[0],
        0.0,
        atol=5.0e-10,
    )
    np.testing.assert_allclose(
        result.wall_momentum_impulse + result.mass_momentum_energy_exchange[1:3],
        0.0,
        atol=1.0e-10,
    )
    assert abs(float(result.wall_work)) > 0.0
    np.testing.assert_allclose(
        result.heat_exchange + result.wall_work,
        result.mass_momentum_energy_exchange[3],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_maxwell_refuses_invalid_diffuse_data_and_nonpositive_candidate():
    quadrature = _quadrature()
    diffuse_particles = quadrature.weights
    diffuse_energy = 3.0 * diffuse_particles
    with pytest.raises(ValueError, match="strictly positive"):
        MaxwellThermalD2VBoundaryPlan(
            quadrature,
            (6, 6),
            (1.0, 1.0),
            1.0,
            diffuse_particle_populations=diffuse_particles.at[1].set(0.0),
            diffuse_total_energy_populations=diffuse_energy,
            accommodation=1.0,
            wall_velocity=jnp.zeros((2,)),
            periodic_axes=(False, True),
        )
    with pytest.raises(ValueError, match="unit density"):
        MaxwellThermalD2VBoundaryPlan(
            quadrature,
            (6, 6),
            (1.0, 1.0),
            1.0,
            diffuse_particle_populations=2.0 * diffuse_particles,
            diffuse_total_energy_populations=diffuse_energy,
            accommodation=1.0,
            wall_velocity=jnp.zeros((2,)),
            periodic_axes=(False, True),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MaxwellThermalD2VBoundaryPlan(
            quadrature,
            (6, 6),
            (1.0, 1.0),
            1.0,
            diffuse_particle_populations=diffuse_particles,
            diffuse_total_energy_populations=diffuse_energy,
            accommodation=1.01,
            wall_velocity=jnp.zeros((2,)),
            periodic_axes=(False, True),
        )

    plan = MaxwellThermalD2VBoundaryPlan(
        quadrature,
        (6, 6),
        (1.0, 1.0),
        1.0,
        diffuse_particle_populations=diffuse_particles,
        diffuse_total_energy_populations=diffuse_energy,
        accommodation=1.0,
        wall_velocity=jnp.zeros((2,)),
        periodic_axes=(False, True),
    )
    zero_state = SmoothCompressibleKineticState(
        jnp.zeros((6, 6, 17), dtype=jnp.float64),
        jnp.zeros((6, 6, 17), dtype=jnp.float64),
    )
    result = plan.route(zero_state)

    assert not bool(result.successful)
    assert int(result.status) == int(
        SmoothCompressibleD2VBoundaryStatus.NONPOSITIVE_CANDIDATE
    )
    assert bool(result.rollback_applied)
    np.testing.assert_array_equal(
        result.accepted_state.particle_populations,
        zero_state.particle_populations,
    )
    np.testing.assert_array_equal(
        result.accepted_state.total_energy_populations,
        zero_state.total_energy_populations,
    )
    np.testing.assert_array_equal(result.mass_momentum_energy_exchange, jnp.zeros((4,)))
