#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.metrix import (
    EuclideanStateGeometry,
    FlatTorusStateGeometry,
    LieGroupStateGeometry,
    PointwiseStateGeometry,
    SpecialUnitaryGroup,
)
from phydrax.sampling._split_group_dynamics import (
    adapt_split_group_metric,
    initialize_split_group_dynamics_state,
    prepare_split_group_dynamics,
    split_group_transition,
    split_integrator_trajectory,
    SplitGroupDynamicsPlan,
    SplitGroupTarget,
    SplitMetricAdaptationPlan,
    transported_group_u_turn,
)


def _torus_target():
    geometry = FlatTorusStateGeometry(2.0 * jnp.pi)

    def first(position):
        return -0.35 * jnp.sum(1.0 - jnp.cos(position))

    def second(position):
        return -0.65 * jnp.sum(1.0 - jnp.cos(position))

    return SplitGroupTarget(
        lambda position: first(position) + second(position),
        (first, second),
        geometry,
        configuration_shape=(2,),
        local_coordinate_shape=(2,),
        reference_measure="flat-torus",
        target_id="two-force-flat-torus",
    )


def _prepared(integrator="omelyan", dynamics="ghmc", persistence=0.4):
    plan = SplitGroupDynamicsPlan(
        step_size=0.04,
        trajectory_steps=5,
        maximum_tree_depth=4,
        momentum_persistence=persistence,
        integrator=integrator,
        dynamics=dynamics,
    )
    return prepare_split_group_dynamics(_torus_target(), plan)


def test_leapfrog_and_omelyan_coefficients_are_palindromic():
    leapfrog = _prepared(integrator="leapfrog")
    omelyan = _prepared(integrator="omelyan")

    assert jnp.array_equal(leapfrog.kick_coefficients, leapfrog.kick_coefficients[::-1])
    assert jnp.array_equal(leapfrog.drift_coefficients, leapfrog.drift_coefficients[::-1])
    assert jnp.allclose(omelyan.kick_coefficients, omelyan.kick_coefficients[::-1])
    assert jnp.allclose(omelyan.drift_coefficients, omelyan.drift_coefficients[::-1])
    assert leapfrog.frozen and omelyan.frozen


@pytest.mark.parametrize("integrator", ["leapfrog", "omelyan"])
def test_split_integrator_is_reversible_on_flat_torus(integrator):
    prepared = _prepared(integrator=integrator)
    state = initialize_split_group_dynamics_state(
        prepared,
        jnp.asarray([0.2, -0.3]),
        momentum=jnp.asarray([0.4, -0.1]),
    )
    forward = split_integrator_trajectory(
        prepared,
        state.position,
        state.momentum,
        state.force_gradients,
        steps=3,
    )
    reverse = split_integrator_trajectory(
        prepared,
        forward[0],
        forward[1],
        forward[3],
        steps=3,
        direction=-1,
    )

    assert jnp.allclose(reverse[0], state.position, atol=2e-5)
    assert jnp.allclose(reverse[1], state.momentum, atol=2e-5)
    assert forward[4] and reverse[4]


def test_generalized_hmc_uses_exact_target_correction_and_partial_refresh():
    prepared = _prepared(persistence=0.6)
    state = initialize_split_group_dynamics_state(
        prepared,
        jnp.asarray([0.1, 0.25]),
        momentum=jnp.asarray([0.2, -0.4]),
    )
    result = split_group_transition(prepared, state, key=jax.random.key(21))

    assert prepared.target.geometry.contains(result.state.position)
    assert result.evidence.membership_preserved
    assert result.evidence.exact_target_correction
    assert jnp.allclose(result.evidence.momentum_refresh_correlation, 0.6)
    assert jnp.allclose(result.evidence.detailed_balance_residual, 0.0)
    assert result.evidence.status == 0
    assert 0.0 <= result.evidence.acceptance_probability <= 1.0


def test_finite_metric_adaptation_returns_rebound_frozen_production_state():
    prepared = _prepared(integrator="leapfrog", persistence=0.0)
    state = initialize_split_group_dynamics_state(prepared, jnp.asarray([0.15, -0.2]))
    result = adapt_split_group_metric(
        prepared,
        state,
        SplitMetricAdaptationPlan(
            4,
            minimum_inverse_mass=0.02,
            maximum_inverse_mass=5.0,
        ),
        key=jax.random.key(22),
    )

    assert result.frozen
    assert result.prepared.frozen
    assert result.adaptation_steps == 4
    assert result.inverse_mass_history.shape == (4, 2)
    assert jnp.all(result.inverse_mass_history >= 0.02)
    assert jnp.all(result.inverse_mass_history <= 5.0)
    assert result.state.prepared_id == result.prepared.prepared_id


def test_transported_u_turn_uses_endpoint_velocities_in_one_tangent_space():
    prepared = _prepared()
    forward = transported_group_u_turn(
        prepared,
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.5, 0.0]),
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([1.0, 0.0]),
    )
    turning = transported_group_u_turn(
        prepared,
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.5, 0.0]),
        jnp.asarray([-1.0, 0.0]),
        jnp.asarray([1.0, 0.0]),
    )

    assert forward.finite and not forward.turning
    assert turning.finite and turning.turning
    assert jnp.allclose(forward.displacement, jnp.asarray([0.5, 0.0]))


def test_transported_group_nuts_has_finite_tree_and_detects_turning():
    prepared = _prepared(dynamics="nuts-reference", persistence=0.0)
    state = initialize_split_group_dynamics_state(prepared, jnp.asarray([2.6, -2.4]))
    result = split_group_transition(prepared, state, key=jax.random.key(23))

    assert result.evidence.integration_steps <= prepared.maximum_integration_steps
    assert prepared.target.geometry.contains(result.state.position)
    assert result.evidence.u_turn_detected | result.evidence.maximum_depth_reached
    assert result.reference_measure == "flat-torus"


def test_su_n_is_supported_and_noncompact_geometry_fails_before_execution():
    group = SpecialUnitaryGroup(2)
    geometry = PointwiseStateGeometry(
        LieGroupStateGeometry(group),
        group.point_shape,
        local_shape=group.algebra_shape,
    )
    target = SplitGroupTarget(
        lambda links: 0.4 * jnp.real(jnp.trace(links[0])),
        (lambda links: 0.4 * jnp.real(jnp.trace(links[0])),),
        geometry,
        configuration_shape=(1, 2, 2),
        local_coordinate_shape=(1, 3),
        reference_measure="product-haar",
        target_id="single-su2-link",
    )
    prepared = prepare_split_group_dynamics(
        target, SplitGroupDynamicsPlan(step_size=0.03)
    )
    state = initialize_split_group_dynamics_state(
        prepared, jnp.eye(2, dtype="complex128")[None]
    )
    assert state.valid

    unsupported = SplitGroupTarget(
        lambda value: -jnp.sum(value**2),
        (lambda value: -jnp.sum(value**2),),
        EuclideanStateGeometry(),
        configuration_shape=(2,),
        local_coordinate_shape=(2,),
        reference_measure="flat-torus",
        target_id="unsupported-noncompact",
    )
    with pytest.raises(TypeError, match="supports flat tori"):
        prepare_split_group_dynamics(unsupported, SplitGroupDynamicsPlan(step_size=0.03))
