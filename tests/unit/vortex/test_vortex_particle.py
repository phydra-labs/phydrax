#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _particles(dimension, count):
    return phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.ones((count,)),
        ambient_dimension=dimension,
    ).prepare()


def test_direct_2d_excludes_only_explicit_self_and_preserves_coincident_distinct_blob():
    plan = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=2,
        source_chunk_size=1,
        target_chunk_size=1,
    ).prepare(source_capacity=2, target_capacity=2)
    position = jnp.zeros((2, 2))
    circulation = jnp.asarray((1.0, -0.5))
    core = jnp.asarray((0.2, 0.3))
    result = plan.evaluate(
        position,
        circulation,
        core,
        request=phx.discretization.VortexFieldRequest(
            velocity=True,
            velocity_gradient=True,
            vorticity=True,
        ),
    )

    np.testing.assert_allclose(result.velocity, 0.0)
    assert int(result.diagnostics.excluded_interaction_count) == 2
    assert int(result.diagnostics.coincident_distinct_count) == 2
    assert jnp.all(jnp.isfinite(result.velocity_gradient))
    assert jnp.all(jnp.isfinite(result.vorticity))
    assert bool(result.successful)


def test_direct_2d_chunking_and_permutation_leave_fields_unchanged():
    position = jnp.asarray(((-0.3, 0.2), (0.5, -0.1), (0.1, 0.7)))
    circulation = jnp.asarray((0.7, -0.4, 0.9))
    core = jnp.asarray((0.2, 0.3, 0.25))
    target = jnp.asarray(((0.2, -0.4), (0.8, 0.1)))
    coarse = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=3,
        maximum_targets=2,
        source_chunk_size=3,
        target_chunk_size=2,
    ).prepare(source_capacity=3, target_capacity=2)
    fine = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=3,
        maximum_targets=2,
        source_chunk_size=1,
        target_chunk_size=1,
    ).prepare(source_capacity=3, target_capacity=2)
    expected = coarse.evaluate(position, circulation, core, targets=target).velocity
    actual = fine.evaluate(position, circulation, core, targets=target).velocity
    permutation = jnp.asarray((2, 0, 1))
    permuted = coarse.evaluate(
        position[permutation],
        circulation[permutation],
        core[permutation],
        targets=target,
    ).velocity

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(permuted, expected, rtol=1e-12, atol=1e-12)


def test_direct_plan_rejects_resource_overflow_before_execution():
    plan = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=4,
        maximum_targets=4,
        maximum_interactions=8,
    )
    with pytest.raises(ValueError, match="interactions"):
        plan.prepare(source_capacity=4, target_capacity=4)


def test_pse_is_exactly_conservative_for_unequal_particle_volumes():
    plan = phx.operators.GaussianParticleStrengthExchangePlan(
        2,
        0.5,
        active_mask=jnp.ones((3,), dtype=bool),
    ).prepare(capacity=3, dimension=2)
    evaluation = plan.evaluate(
        jnp.asarray(((-0.2, 0.0), (0.0, 0.0), (0.3, 0.0))),
        jnp.asarray((0.4, 1.2, -0.2)),
        jnp.asarray((0.2, 0.5, 0.3)),
        0.01,
    )

    np.testing.assert_allclose(jnp.sum(evaluation.rate), 0.0, atol=1e-14)
    assert bool(evaluation.diagnostics.conservative)
    assert bool(evaluation.successful)


def test_compiled_2d_pair_is_differentiable_and_keeps_mass_distinct_from_circulation():
    particles = _particles(2, 2)
    properties = phx.discretization.VortexParticleProperties(
        jnp.full((2,), 0.1),
        jnp.asarray((0.25, 0.75)),
    )
    method = phx.discretization.VortexParticleMethodPlan(
        phx.operators.GaussianDirectVortexPlan2D(maximum_sources=2)
    )
    compiled = phx.equations.compile_vortex_particle_flow(
        phx.equations.VortexParticleFlowProblem("pair", 2),
        particles,
        properties,
        method,
    )
    position = jnp.asarray(((-0.5, 0.0), (0.5, 0.0)))
    circulation = jnp.asarray((1.0, 1.0))
    state = compiled.initialize_state(position, circulation)
    rate = eqx.filter_jit(compiled.dynamics)(0.0, state, None)
    gradient = jax.grad(
        lambda values: jnp.sum(compiled.dynamics(0.0, values, None) ** 2)
    )(state)

    assert rate.shape == state.shape
    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(particles.masses, 1.0)
    np.testing.assert_allclose(
        compiled.dynamics.state_layout.unpack(state).strength, circulation
    )


def test_classic_3d_dynamics_adds_velocity_gradient_stretching():
    particles = _particles(3, 2)
    properties = phx.discretization.VortexParticleProperties(
        jnp.full((2,), 0.2),
        jnp.ones((2,)),
    )
    method = phx.discretization.VortexParticleMethodPlan(
        phx.operators.GaussianErfDirectVortexPlan3D(
            source_chunk_size=2,
            target_chunk_size=2,
            interaction_budget=4,
        )
    )
    compiled = phx.equations.compile_vortex_particle_flow(
        phx.equations.VortexParticleFlowProblem("stretching", 3),
        particles,
        properties,
        method,
    )
    state = compiled.initialize_state(
        jnp.asarray(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0))),
        jnp.asarray(((0.0, 1.0, 0.2), (0.0, -1.0, 0.2))),
    )
    evaluation = compiled.dynamics.evaluate(0.0, state)

    assert evaluation[3].shape == (2, 3)
    assert jnp.linalg.norm(evaluation[3]) > 0.0
    assert jnp.all(jnp.isfinite(evaluation[3]))
