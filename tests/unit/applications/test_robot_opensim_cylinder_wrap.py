#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.robotics._analytic_wrap import (
    AnalyticWrapStatus,
    PlanarCylinderRouteWrapPlan,
)
from phydrax.applications.robotics._opensim_cylinder_wrap import (
    OpenSimCylinderRouteWrapPlan,
    OpenSimCylinderWrapStatus,
)


_POINTS = jnp.asarray(((-2.0, 0.35, -0.8), (2.1, 0.65, 1.2)))
_VELOCITY = jnp.asarray(((0.12, -0.04, 0.17), (-0.03, 0.08, -0.11)))


def _prepared(*, side="shortest", length=8.0, samples=32):
    return OpenSimCylinderRouteWrapPlan(samples, side=side).prepare(
        jnp.zeros(3), jnp.asarray((0.0, 0.0, 1.0)), 1.0, length
    )


def _accepted(prepared, points=_POINTS):
    source = prepared.initial_state()
    candidate = prepared.propose(source, points)
    assert bool(candidate.successful)
    return prepared.commit(candidate, source)


def _stationary_source_oracle(points, sign):
    """Independent polar tangents + Newton solve of source axial stationarity.

    OpenSim's _make_spiral_path uses sqrt((r*theta)^2 + dz^2), and
    _adjust_tangent_point equates the straight/helix axial angles. This solves
    those equations without using the production common-slope elimination.
    It is an equation oracle, not an assertion of OpenSim executable parity.
    """
    points = np.asarray(points, dtype=float)
    distance = np.linalg.norm(points[:, :2], axis=-1)
    polar = np.arctan2(points[:, 1], points[:, 0])
    tangent_angle = polar + sign * np.asarray((1.0, -1.0)) * np.arccos(1.0 / distance)
    arc = np.mod(sign * (tangent_angle[1] - tangent_angle[0]), 2.0 * np.pi)
    spans = np.asarray(
        (np.sqrt(distance[0] ** 2 - 1.0), arc, np.sqrt(distance[1] ** 2 - 1.0))
    )
    z = points[0, 2] + (points[1, 2] - points[0, 2]) * np.asarray((1 / 3, 2 / 3))
    for _ in range(24):
        dz = np.asarray((z[0] - points[0, 2], z[1] - z[0], points[1, 2] - z[1]))
        segment_lengths = np.sqrt(spans**2 + dz**2)
        slope = dz / segment_lengths
        stiffness = spans**2 / segment_lengths**3
        residual = np.asarray((slope[0] - slope[1], slope[1] - slope[2]))
        hessian = np.asarray(
            (
                (stiffness[0] + stiffness[1], -stiffness[1]),
                (-stiffness[1], stiffness[1] + stiffness[2]),
            )
        )
        direction = np.linalg.solve(hessian, residual)
        step = 1.0
        for _ in range(24):
            trial = z - step * direction
            trial_dz = np.asarray(
                (trial[0] - points[0, 2], trial[1] - trial[0], points[1, 2] - trial[1])
            )
            if np.sum(np.sqrt(spans**2 + trial_dz**2)) <= np.sum(segment_lengths):
                break
            step *= 0.5
        z = trial
    dz = np.asarray((z[0] - points[0, 2], z[1] - z[0], points[1, 2] - z[1]))
    segment_lengths = np.sqrt(spans**2 + dz**2)
    tangents = np.column_stack((np.cos(tangent_angle), np.sin(tangent_angle), z))
    return np.sum(segment_lengths), segment_lengths[1], tangents, sign * arc


def test_unequal_axial_lateral_path_matches_source_stationarity_and_shortest_cost():
    prepared = _prepared()
    candidate = prepared.propose(prepared.initial_state(), _POINTS)
    result = candidate.evaluation
    positive = _stationary_source_oracle(_POINTS, 1)
    negative = _stationary_source_oracle(_POINTS, -1)
    expected = min((positive, negative), key=lambda row: row[0])
    assert bool(candidate.successful)
    np.testing.assert_allclose(result.total_length_m, expected[0], rtol=2e-6)
    np.testing.assert_allclose(result.surface_length_m, expected[1], rtol=2e-6)
    np.testing.assert_allclose(result.tangent_points_m, expected[2], atol=2e-6)
    np.testing.assert_allclose(result.signed_surface_angle_rad, expected[3], atol=2e-6)
    np.testing.assert_allclose(
        result.evidence.candidate_lengths_m[1:], (positive[0], negative[0]), rtol=2e-6
    )
    assert float(result.evidence.selected_excess_length_m) == 0.0
    assert float(result.evidence.shortest_lateral_gap_m) > 0.1
    assert float(result.evidence.tangent_direction_residual) < 2e-6
    samples = np.asarray(result.surface_points_m)
    np.testing.assert_allclose(np.linalg.norm(samples[:, :2], axis=-1), 1.0, atol=2e-6)
    np.testing.assert_allclose(samples[[0, -1]], result.tangent_points_m, atol=2e-6)
    # Display capacity never becomes the mechanical discretization.
    coarse = _prepared(samples=2).propose(_prepared(samples=2).initial_state(), _POINTS)
    np.testing.assert_allclose(coarse.evaluation.total_length_m, result.total_length_m)


def test_side_prescription_selects_complete_tangent_paths_not_reversed_surface_only():
    positive = _prepared(side="positive")
    negative = _prepared(side="negative")
    for prepared, sign in ((positive, 1), (negative, -1)):
        result = prepared.propose(prepared.initial_state(), _POINTS).evaluation
        expected = _stationary_source_oracle(_POINTS, sign)
        assert bool(result.evidence.successful)
        np.testing.assert_allclose(result.tangent_points_m, expected[2], atol=2e-6)
        np.testing.assert_allclose(result.total_length_m, expected[0], rtol=2e-6)
        assert float(result.evidence.tangent_direction_residual) < 2e-6
        assert np.sign(float(result.signed_surface_angle_rad)) == sign


def test_planar_source_fidelity_remains_separate_and_nonplanar_fails_there():
    planar = PlanarCylinderRouteWrapPlan().prepare(
        jnp.zeros(3), jnp.asarray((0.0, 0.0, 1.0)), 1.0, 8.0
    )
    previous = planar.evaluate(*_POINTS)
    assert not bool(previous.evidence.successful)
    assert int(previous.evidence.status) & int(
        AnalyticWrapStatus.NONPLANAR_CYLINDER_ROUTE
    )
    assert bool(_prepared().propose(_prepared().initial_state(), _POINTS).successful)


def test_fixed_branch_jit_vmap_jvp_vjp_and_virtual_power():
    prepared = _prepared()
    state = _accepted(prepared)
    result = eqx.filter_jit(prepared.evaluate_fixed_branch)(state, _POINTS)
    assert bool(result.evidence.fixed_branch_gradient_supported)
    operator = prepared.length_jacobian_operator(state, _POINTS)
    step = 2e-3
    upper = prepared.evaluate_fixed_branch(state, _POINTS + step * _VELOCITY)
    lower = prepared.evaluate_fixed_branch(state, _POINTS - step * _VELOCITY)
    finite_difference = (upper.total_length_m - lower.total_length_m) / (2 * step)
    np.testing.assert_allclose(operator.mv(_VELOCITY)[0], finite_difference, atol=2e-4)
    derivative = jax.grad(
        lambda points: prepared.evaluate_fixed_branch(state, points).total_length_m
    )(_POINTS)
    np.testing.assert_allclose(derivative, operator.transpose_mv(jnp.ones(1)), atol=2e-6)
    loads, power = eqx.filter_jit(prepared.tensile_force_pullback)(
        state, _POINTS, _VELOCITY, 120.0, force_owner="native-tension"
    )
    assert bool(power.successful)
    np.testing.assert_allclose(loads, -120.0 * derivative, atol=3e-5)
    np.testing.assert_allclose(
        jnp.sum(loads * _VELOCITY), -120.0 * operator.mv(_VELOCITY)[0], atol=1e-5
    )
    batch = eqx.filter_jit(jax.vmap(lambda points: prepared.propose(state, points)))(
        jnp.stack((_POINTS, _POINTS + 0.02 * _VELOCITY))
    )
    assert bool(jnp.all(batch.successful))


def test_rigid_frame_covariance_and_endpoint_action_reaction():
    prepared = _prepared()
    state = _accepted(prepared)
    angle = 0.71
    rotation = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(angle), -np.sin(angle)),
            (0.0, np.sin(angle), np.cos(angle)),
        )
    )
    translation = jnp.asarray((0.7, -0.4, 0.3))
    moved = prepared.plan.prepare(translation, rotation[:, 2], 1.0, 8.0)
    moved_points = _POINTS @ rotation.T + translation
    moved_state = _accepted(moved, moved_points)
    original = prepared.evaluate_fixed_branch(state, _POINTS)
    transformed = moved.evaluate_fixed_branch(moved_state, moved_points)
    np.testing.assert_allclose(
        transformed.total_length_m, original.total_length_m, rtol=2e-6
    )
    np.testing.assert_allclose(
        transformed.tangent_points_m,
        original.tangent_points_m @ rotation.T + translation,
        atol=2e-6,
    )
    original_loads, _ = prepared.tensile_force_pullback(
        state, _POINTS, _VELOCITY, 50.0, force_owner="native-tension"
    )
    moved_loads, _ = moved.tensile_force_pullback(
        moved_state,
        moved_points,
        _VELOCITY @ rotation.T,
        50.0,
        force_owner="native-tension",
    )
    np.testing.assert_allclose(moved_loads, original_loads @ rotation.T, atol=2e-5)
    # Axial force has no cylinder reaction; transverse reaction belongs to the
    # obstacle support, not to an additional muscle/contact force law.
    np.testing.assert_allclose(jnp.sum(original_loads[:, 2]), 0.0, atol=2e-5)


def test_candidate_commit_branch_transition_and_stale_or_foreign_rollback():
    prepared = _prepared()
    initial = prepared.initial_state()
    candidate = prepared.propose(initial, _POINTS)
    assert not bool(initial.initialized)
    assert not bool(candidate.evaluation.evidence.fixed_branch_gradient_supported)
    state = prepared.commit(candidate, initial)
    assert int(state.accepted_steps) == 1
    np.testing.assert_array_equal(
        prepared.commit(candidate, state).endpoints_m, state.endpoints_m
    )
    assert int(prepared.commit(candidate, state).accepted_steps) == 1
    switched_points = _POINTS.at[:, 1].multiply(-1.0)
    switch = prepared.propose(state, switched_points)
    assert bool(switch.successful)
    assert bool(switch.evaluation.evidence.mode_changed)
    assert not bool(switch.evaluation.evidence.fixed_branch_gradient_supported)
    loads, power = prepared.tensile_force_pullback(
        state, switched_points, _VELOCITY, 20.0, force_owner="native-tension"
    )
    np.testing.assert_array_equal(loads, jnp.zeros((2, 3)))
    assert not bool(power.successful)
    accepted = prepared.commit(switch, state)
    assert int(accepted.branch) != int(state.branch)
    assert bool(
        prepared.evaluate_fixed_branch(
            accepted, switched_points
        ).evidence.fixed_branch_gradient_supported
    )
    foreign = _prepared(length=9.0)
    np.testing.assert_array_equal(
        foreign.commit(switch, state).endpoints_m, state.endpoints_m
    )
    assert int(foreign.commit(switch, state).accepted_steps) == int(state.accepted_steps)
    modified = eqx.tree_at(lambda item: item.radius_m, prepared, jnp.asarray(1.1))
    assert not bool(modified.propose(state, _POINTS).successful)


@pytest.mark.parametrize(
    "points,flag",
    (
        (((-2.0, 0.0, -0.8), (2.0, 0.0, 1.2)), OpenSimCylinderWrapStatus.TOPOLOGY_TIE),
        (((-2.0, 1.0, -0.8), (2.0, 1.0, 1.2)), OpenSimCylinderWrapStatus.CONTACT_EVENT),
        (
            ((0.5, 0.0, 0.0), (2.0, 0.65, 1.2)),
            OpenSimCylinderWrapStatus.ENDPOINT_INSIDE_RADIUS,
        ),
        (
            ((float("nan"), 0.35, -0.8), (2.1, 0.65, 1.2)),
            OpenSimCylinderWrapStatus.NONFINITE,
        ),
    ),
)
def test_event_and_invalid_input_roll_back_all_state_and_load_zero(points, flag):
    prepared = _prepared()
    state = _accepted(prepared)
    points = jnp.asarray(points)
    candidate = prepared.propose(state, points)
    assert not bool(candidate.successful)
    assert int(candidate.evaluation.evidence.status) & int(flag)
    assert eqx.tree_equal(prepared.commit(candidate, state), state)
    loads, evidence = prepared.tensile_force_pullback(
        state, points, _VELOCITY, 100.0, force_owner="native-tension"
    )
    assert not bool(evidence.successful)
    np.testing.assert_array_equal(loads, jnp.zeros((2, 3)))


def test_cap_rim_gate_does_not_substitute_a_direct_chord_or_longer_branch():
    prepared = _prepared(length=0.1)
    source = prepared.initial_state()
    candidate = prepared.propose(source, _POINTS)
    assert not bool(candidate.successful)
    assert int(candidate.evaluation.evidence.status) & int(
        OpenSimCylinderWrapStatus.CAP_OR_RIM_UNSUPPORTED
    )
    assert eqx.tree_equal(prepared.commit(candidate, source), source)
    # A provably free chord wholly above the finite extent needs no cap model.
    above = _POINTS.at[:, 2].set(jnp.asarray((2.0, 3.0)))
    direct = prepared.propose(source, above)
    assert bool(direct.successful)
    assert not bool(direct.evaluation.evidence.applied)
    np.testing.assert_allclose(
        direct.evaluation.total_length_m, jnp.linalg.norm(above[1] - above[0])
    )


def test_clear_direct_branch_and_exclusive_force_owner():
    prepared = _prepared()
    points = _POINTS.at[:, 1].add(3.0)
    state = _accepted(prepared, points)
    assert int(state.branch) == 0
    loads, evidence = prepared.tensile_force_pullback(
        state, points, _VELOCITY, 17.0, force_owner="native-tension"
    )
    assert bool(evidence.successful)
    direction = (points[1] - points[0]) / jnp.linalg.norm(points[1] - points[0])
    np.testing.assert_allclose(
        loads, jnp.stack((17.0 * direction, -17.0 * direction)), atol=2e-6
    )
    with pytest.raises(ValueError, match="Provider-native"):
        prepared.tensile_force_pullback(
            state, points, _VELOCITY, 17.0, force_owner="provider-native"
        )
    loads, evidence = prepared.tensile_force_pullback(
        state, points, _VELOCITY, -17.0, force_owner="native-tension"
    )
    assert not bool(evidence.successful)
    np.testing.assert_array_equal(loads, jnp.zeros((2, 3)))
