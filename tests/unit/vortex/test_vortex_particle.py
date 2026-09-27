#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _particles(dimension: Any, count: Any) -> Any:
    return phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.ones((count,)),
        ambient_dimension=dimension,
    ).prepare()


def test_population_initialization_rejects_invalid_active_stable_ids() -> None:
    plan = phx.discretization.VortexPopulationPlan(2, 2)
    with pytest.raises(RuntimeError, match="stable IDs"):
        plan.initialize(
            jnp.zeros((2, 2)),
            jnp.ones((2,)),
            jnp.ones((2,)),
            jnp.ones((2,)),
            stable_ids=jnp.asarray((5, 5)),
        )
    with pytest.raises(RuntimeError, match="stable IDs"):
        plan.initialize(
            jnp.zeros((2, 2)),
            jnp.ones((2,)),
            jnp.ones((2,)),
            jnp.ones((2,)),
            stable_ids=jnp.asarray((-1, 6)),
        )


def test_direct_2d_excludes_only_explicit_self_and_preserves_coincident_distinct_blob() -> (
    None
):
    request = phx.discretization.VortexFieldRequest(
        velocity=True,
        velocity_gradient=True,
        vorticity=True,
    )
    plan = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=2,
        source_chunk_size=1,
        target_chunk_size=1,
    ).prepare(
        source_capacity=2,
        target_capacity=2,
        request=request,
    )
    position = jnp.zeros((2, 2))
    source = phx.discretization.VortexSourceState(
        position,
        jnp.asarray((1.0, -0.5)),
        core_radius=jnp.asarray((0.2, 0.3)),
    )
    target = phx.discretization.VortexTargetState(
        position,
        source_indices=jnp.arange(2),
    )
    result = plan.evaluate(source, target, request=request)

    np.testing.assert_allclose(result.velocity, 0.0)
    assert int(result.diagnostics.excluded_interaction_count) == 2
    assert int(result.diagnostics.coincident_distinct_count) == 2
    assert jnp.all(jnp.isfinite(result.velocity_gradient))
    assert jnp.all(jnp.isfinite(result.vorticity))
    assert bool(result.successful)


def test_direct_2d_chunking_and_permutation_leave_fields_unchanged() -> None:
    position = jnp.asarray(((-0.3, 0.2), (0.5, -0.1), (0.1, 0.7)))
    circulation = jnp.asarray((0.7, -0.4, 0.9))
    core = jnp.asarray((0.2, 0.3, 0.25))
    target = jnp.asarray(((0.2, -0.4), (0.8, 0.1)))
    source = phx.discretization.VortexSourceState(
        position,
        circulation,
        core_radius=core,
    )
    targets = phx.discretization.VortexTargetState(target)
    coarse = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=3,
        maximum_targets=2,
        source_chunk_size=3,
        target_chunk_size=2,
    ).prepare(
        source_capacity=3,
        target_capacity=2,
        target_topology="arbitrary-targets",
    )
    fine = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=3,
        maximum_targets=2,
        source_chunk_size=1,
        target_chunk_size=1,
    ).prepare(
        source_capacity=3,
        target_capacity=2,
        target_topology="arbitrary-targets",
    )
    expected = coarse.evaluate(source, targets).velocity
    actual = fine.evaluate(source, targets).velocity
    permutation = jnp.asarray((2, 0, 1))
    permuted_source = phx.discretization.VortexSourceState(
        position[permutation],
        circulation[permutation],
        core_radius=core[permutation],
    )
    permuted = coarse.evaluate(permuted_source, targets).velocity

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(permuted, expected, rtol=1e-12, atol=1e-12)


def test_direct_plan_rejects_resource_overflow_before_execution() -> None:
    plan = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=4,
        maximum_targets=4,
        maximum_interactions=8,
    )
    with pytest.raises(ValueError, match="interactions"):
        plan.prepare(source_capacity=4, target_capacity=4)


def test_pse_is_exactly_conservative_for_unequal_particle_volumes() -> None:
    plan = phx.operators.GaussianParticleStrengthExchangePlan(
        2,
        0.5,
    ).prepare(capacity=3, dimension=2)
    source = phx.discretization.VortexSourceState(
        jnp.asarray(((-0.2, 0.0), (0.0, 0.0), (0.3, 0.0))),
        jnp.asarray((0.4, 1.2, -0.2)),
        volume=jnp.asarray((0.2, 0.5, 0.3)),
    )
    evaluation = plan.evaluate(source, 0.01)

    np.testing.assert_allclose(jnp.sum(evaluation.rate), 0.0, atol=1e-14)
    assert bool(evaluation.diagnostics.conservative)
    assert bool(evaluation.successful)


def test_pse_particle_box_exchanges_only_through_periodic_axes() -> None:
    source = phx.discretization.VortexSourceState(
        jnp.asarray(((0.02, 0.5), (0.98, 0.5))),
        jnp.asarray((1.0, -1.0)),
        volume=jnp.asarray((0.1, 0.1)),
    )
    periodic = phx.operators.GaussianParticleStrengthExchangePlan(
        2,
        0.05,
        # ty: ignore[invalid-argument-type]
        box=phx.discretization.ParticleBox([0.0, 0.0], [1.0, 1.0]),
    )
    walled = phx.operators.GaussianParticleStrengthExchangePlan(
        2,
        0.05,
        box=phx.discretization.ParticleBox(
            # ty: ignore[invalid-argument-type]
            [0.0, 0.0],
            # ty: ignore[invalid-argument-type]
            [1.0, 1.0],
            periodic_axes=(False, False),
        ),
    )
    periodic_rate = periodic.prepare(capacity=2, dimension=2).evaluate(source, 0.01).rate
    walled_rate = walled.prepare(capacity=2, dimension=2).evaluate(source, 0.01).rate

    assert periodic.capabilities.domain == "periodic"
    assert walled.capabilities.domain == "free-space"
    assert float(periodic_rate[0]) < 0.0 < float(periodic_rate[1])
    np.testing.assert_allclose(jnp.sum(periodic_rate), 0.0, atol=1e-14)
    np.testing.assert_allclose(walled_rate, 0.0, atol=0.0)
    with pytest.raises(ValueError, match="less than half each period"):
        phx.operators.GaussianParticleStrengthExchangePlan(
            2,
            0.2,
            # ty: ignore[invalid-argument-type]
            box=phx.discretization.ParticleBox([0.0, 0.0], [1.0, 1.0]),
        )


def test_compiled_2d_pair_is_differentiable_and_keeps_mass_distinct_from_circulation() -> (
    None
):
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


def test_classic_3d_dynamics_adds_velocity_gradient_stretching() -> None:
    particles = _particles(3, 2)
    properties = phx.discretization.VortexParticleProperties(
        jnp.full((2,), 0.2),
        jnp.ones((2,)),
    )
    method = phx.discretization.VortexParticleMethodPlan(
        phx.operators.GaussianErfDirectVortexPlan3D(
            maximum_sources=2,
            maximum_targets=2,
            source_chunk_size=2,
            target_chunk_size=2,
            maximum_interactions=4,
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
