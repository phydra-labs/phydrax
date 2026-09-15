#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.discrete_velocity._quadrature import d2v17_quadrature
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial_forcing import (
    SmoothCompressibleD2VBodyForcingPlan,
    SmoothCompressibleD2VForcingStatus,
    ZeroSmoothCompressibleD2VForcingPlan,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport


jax.config.update("jax_enable_x64", True)


def _method_and_state():
    method = SmoothCompressibleD2VKineticMethod(
        d2v17_quadrature(),
        IdealGasMaterial(1.4, 1.0),
        ConstantTransport(0.03, 0.04),
    )
    density = 1.2
    momentum = jnp.asarray((0.12, -0.06))
    pressure = 0.5 * density
    total_energy = pressure / (1.4 - 1.0) + 0.5 * jnp.dot(momentum, momentum) / density
    state = method.equilibrium(
        jnp.concatenate((jnp.asarray((density,)), momentum, total_energy[None]))
    )
    assert bool(method.realizability(state).realizable)
    return method, state


def test_zero_source_is_exact_and_has_zero_finite_evidence():
    method, state = _method_and_state()
    result = ZeroSmoothCompressibleD2VForcingPlan(method).apply(state, jnp.asarray(0.125))

    np.testing.assert_array_equal(
        result.candidate_state.particle_populations, state.particle_populations
    )
    np.testing.assert_array_equal(
        result.candidate_state.total_energy_populations,
        state.total_energy_populations,
    )
    np.testing.assert_array_equal(
        result.accepted_state.particle_populations, state.particle_populations
    )
    np.testing.assert_array_equal(
        result.accepted_state.total_energy_populations,
        state.total_energy_populations,
    )
    np.testing.assert_array_equal(
        result.particle_population_increment,
        jnp.zeros_like(state.particle_populations),
    )
    np.testing.assert_array_equal(
        result.total_energy_population_increment,
        jnp.zeros_like(state.total_energy_populations),
    )
    np.testing.assert_array_equal(result.mass_momentum_energy_increment, jnp.zeros((4,)))
    np.testing.assert_array_equal(result.evidence.target_source_moments, jnp.zeros((4,)))
    np.testing.assert_array_equal(
        result.evidence.recovered_source_moments, jnp.zeros((4,))
    )
    np.testing.assert_array_equal(result.evidence.source_moment_residual, jnp.zeros((4,)))
    np.testing.assert_array_equal(
        result.evidence.particle_source_second_moment, jnp.zeros((2, 2))
    )
    np.testing.assert_array_equal(
        result.evidence.energy_source_first_moment, jnp.zeros((2,))
    )
    assert bool(result.evidence.finite)
    assert bool(result.successful)
    assert not bool(result.rollback_applied)
    assert int(result.evidence.status) == int(SmoothCompressibleD2VForcingStatus.SUCCESS)


def test_body_force_has_zero_mass_exact_impulse_and_midpoint_work():
    method, state = _method_and_state()
    acceleration = jnp.asarray((0.02, -0.03))
    heating = 0.04
    time_step = jnp.asarray(0.01)
    result = SmoothCompressibleD2VBodyForcingPlan(
        method,
        acceleration=acceleration,
        volumetric_heating=heating,
    ).apply(state, time_step)

    before = method.moments(state)
    force = before.density * acceleration
    momentum_increment = time_step * force
    midpoint_velocity = (before.momentum + 0.5 * momentum_increment) / before.density
    acceleration_work = time_step * jnp.dot(midpoint_velocity, force)
    energy_increment = acceleration_work + time_step * heating
    expected_increment = jnp.concatenate(
        (jnp.zeros((1,)), momentum_increment, energy_increment[None])
    )
    candidate = method.moments(result.candidate_state)
    observed_increment = candidate.conserved - before.conserved

    assert bool(result.successful)
    np.testing.assert_allclose(result.evidence.body_force, force, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        result.evidence.midpoint_velocity,
        midpoint_velocity,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.acceleration_work_increment,
        acceleration_work,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.volumetric_heating_increment,
        time_step * heating,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.target_source_moments,
        expected_increment,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.recovered_source_moments,
        expected_increment,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        observed_increment, expected_increment, rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        result.mass_momentum_energy_increment,
        expected_increment,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.source_moment_residual, 0.0, rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        result.evidence.particle_source_second_moment,
        jnp.einsum(
            "q,qi,qj->ij",
            result.particle_population_increment,
            method.quadrature.velocities,
            method.quadrature.velocities,
        ),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.energy_source_first_moment,
        jnp.einsum(
            "q,qi->i",
            result.total_energy_population_increment,
            method.quadrature.velocities,
        ),
        rtol=0.0,
        atol=2.0e-14,
    )
    evidence_arrays = (
        result.evidence.body_force,
        result.evidence.midpoint_velocity,
        result.evidence.target_source_moments,
        result.evidence.recovered_source_moments,
        result.evidence.source_moment_residual,
        result.evidence.particle_source_second_moment,
        result.evidence.energy_source_first_moment,
        result.evidence.minimum_particle_population,
        result.evidence.minimum_total_energy_population,
    )
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in evidence_arrays)
    assert bool(result.evidence.finite)


def test_pure_heating_changes_only_the_energy_zeroth_source_moment():
    method, state = _method_and_state()
    time_step = jnp.asarray(0.2)
    heating = 0.3
    result = SmoothCompressibleD2VBodyForcingPlan(
        method,
        acceleration=(0.0, 0.0),
        volumetric_heating=heating,
    )(state, time_step)

    np.testing.assert_array_equal(
        result.candidate_state.particle_populations, state.particle_populations
    )
    np.testing.assert_allclose(
        jnp.sum(result.total_energy_population_increment),
        time_step * heating,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.energy_source_first_moment,
        jnp.zeros((2,)),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.target_source_moments,
        jnp.concatenate(
            (
                jnp.zeros((3,)),
                jnp.asarray(time_step * heating)[None],
            )
        ),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert bool(result.successful)


def test_negative_and_nonfinite_candidates_refuse_both_population_fields():
    method, state = _method_and_state()
    zero = ZeroSmoothCompressibleD2VForcingPlan(method)
    negative_state = SmoothCompressibleKineticState(
        state.particle_populations.at[0].set(-1.0e-3),
        state.total_energy_populations,
    )
    negative = zero(negative_state, jnp.asarray(0.1))

    assert not bool(negative.successful)
    assert bool(negative.rollback_applied)
    assert int(negative.evidence.status) == int(
        SmoothCompressibleD2VForcingStatus.NONPOSITIVE_CANDIDATE
    )
    assert float(negative.evidence.minimum_particle_population) < 0.0
    np.testing.assert_array_equal(
        negative.accepted_state.particle_populations,
        negative_state.particle_populations,
    )
    np.testing.assert_array_equal(
        negative.accepted_state.total_energy_populations,
        negative_state.total_energy_populations,
    )
    np.testing.assert_array_equal(
        negative.mass_momentum_energy_increment, jnp.zeros((4,))
    )

    nonfinite_state = SmoothCompressibleKineticState(
        state.particle_populations.at[0].set(jnp.nan),
        state.total_energy_populations,
    )
    nonfinite = zero(nonfinite_state, jnp.asarray(0.1))

    assert not bool(nonfinite.successful)
    assert bool(nonfinite.rollback_applied)
    assert not bool(nonfinite.evidence.finite)
    assert int(nonfinite.evidence.status) == int(
        SmoothCompressibleD2VForcingStatus.NONFINITE_INPUT
    )
    np.testing.assert_allclose(
        nonfinite.accepted_state.particle_populations,
        nonfinite_state.particle_populations,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        nonfinite.accepted_state.total_energy_populations,
        nonfinite_state.total_energy_populations,
    )
    np.testing.assert_array_equal(
        nonfinite.mass_momentum_energy_increment, jnp.zeros((4,))
    )
