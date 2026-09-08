#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import electrophysiology as ep
from phydrax.linalg import (
    LinearSolvePolicy,
    LinearSystem,
    MaterializationPolicy,
    materialize,
    solve,
    StructuredDirect,
    TreeLinearOperator,
    TreeTopology,
)


@pytest.fixture(autouse=True)
def _double_precision():
    with jax.enable_x64():
        yield


def _dense(diagonal, lower, upper, parents):
    matrix = np.diag(np.asarray(diagonal))
    for child, parent in enumerate(parents):
        if parent >= 0:
            matrix[child, parent] = lower[child]
            matrix[parent, child] = upper[child]
    return matrix


def _branched(scheme, mechanisms=None):
    # Deliberately non-topological numbering; root and junction can both clamp.
    morphology = ep.CellMorphologyPlan(
        "tree-test",
        (
            ep.CompartmentSpec("junction", "soma", 60.0, 3.0),
            ep.CompartmentSpec("left", "junction", 80.0, 2.0),
            ep.CompartmentSpec("soma", None, 20.0, 20.0),
            ep.CompartmentSpec("right", "junction", 100.0, 1.5),
        ),
    ).prepare()
    return ep.CableSolverPlan(0.1, scheme=scheme, residual_tolerance=1.0e-10).prepare(
        morphology,
        ep.MembraneProgram(mechanisms or (ep.PassiveLeak(0.1, -68.0),)),
    )


def test_nonsymmetric_tree_block_solve_and_implicit_actions():
    parents = (2, 2, -1, 0, 0)
    topology = TreeTopology(parents)
    diagonal = jnp.asarray([4.0, 3.0, 6.0, 2.0, 2.5])
    lower = jnp.asarray([-0.2, 0.4, 0.0, 0.3, -0.1])
    upper = jnp.asarray([-0.4, -0.2, 0.0, 0.1, 0.25])
    right = jnp.arange(10.0).reshape(5, 2) / 10.0
    weights = jnp.asarray([[0.1, 0.4], [0.3, -0.2], [-0.4, 0.7], [0.8, 0.2], [0.2, -0.5]])
    policy = LinearSolvePolicy(StructuredDirect())

    def solution(d, lo, up, b):
        return solve(
            LinearSystem(TreeLinearOperator(d, lo, up, topology)), b, policy=policy
        ).value

    operator = TreeLinearOperator(diagonal, lower, upper, topology)
    expected_matrix = _dense(diagonal, lower, upper, parents)
    np.testing.assert_allclose(
        materialize(operator, MaterializationPolicy()), expected_matrix
    )
    np.testing.assert_allclose(
        operator.transpose_mv(right[:, 0]), expected_matrix.T @ right[:, 0]
    )
    np.testing.assert_allclose(
        operator.adjoint_mv(right[:, 0]), expected_matrix.T @ right[:, 0]
    )
    np.testing.assert_allclose(
        jax.jit(solution)(diagonal, lower, upper, right),
        np.linalg.solve(expected_matrix, right),
        rtol=1e-12,
        atol=1e-12,
    )

    def objective(d, lo, up, b):
        return jnp.sum(weights * solution(d, lo, up, b))

    primals = (diagonal, lower, upper, right)
    directions = tuple(
        jnp.cos(jnp.arange(value.size).reshape(value.shape)) / 7.0 for value in primals
    )
    gradients = jax.grad(objective, argnums=(0, 1, 2, 3))(*primals)
    reverse = sum(
        jnp.sum(gradient * direction)
        for gradient, direction in zip(gradients, directions, strict=True)
    )
    _, forward = jax.jvp(objective, primals, directions)
    epsilon = 1e-5
    plus = tuple(
        value + epsilon * direction
        for value, direction in zip(primals, directions, strict=True)
    )
    minus = tuple(
        value - epsilon * direction
        for value, direction in zip(primals, directions, strict=True)
    )
    finite_difference = (objective(*plus) - objective(*minus)) / (2 * epsilon)
    np.testing.assert_allclose(forward, finite_difference, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(reverse, finite_difference, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "scheme,theta", [("backward-euler", 1.0), ("crank-nicolson", 0.5)]
)
@pytest.mark.parametrize(
    "mask", [(True, False, False, True), (False, False, True, False)]
)
def test_branched_theta_clamps_preserve_physical_currents(scheme, theta, mask):
    runtime = _branched(scheme)
    state = ep.initialize_cable_state(runtime, jnp.asarray([-64.0, -63.0, -65.0, -62.0]))
    inputs = ep.CableStepInputs(
        jnp.asarray([0.1, 0.0, 0.2, -0.05]),
        jnp.asarray([0.0, 0.001, 0.0, 0.002]),
        jnp.asarray([0.0, 0.02, 0.0, -0.03]),
        jnp.asarray(mask),
        jnp.asarray([-55.0, -60.0, -70.0, -45.0]),
    )
    elapsed = jnp.asarray(0.037)
    result = jax.jit(lambda dt: ep.step_cable(runtime, state, inputs, elapsed_ms=dt))(
        elapsed
    )
    morphology = runtime.morphology
    edge = np.asarray(morphology.edge_conductance_uS)
    conductance = (
        0.1e-5 * np.asarray(morphology.membrane_area_um2) + inputs.synaptic_conductance_uS
    )
    physical = _dense(
        morphology.axial_diagonal_uS + conductance,
        -edge,
        -edge,
        morphology.topology.parent_index,
    )
    offset = (
        68.0 * 0.1e-5 * morphology.membrane_area_um2 + inputs.synaptic_current_offset_nA
    )
    capacity_rate = np.asarray(morphology.capacitance_nF / elapsed)
    original_matrix = np.diag(capacity_rate) + theta * physical
    original_right = (
        capacity_rate * state.voltage_mV
        - (1 - theta) * physical @ state.voltage_mV
        + inputs.injected_current_nA
        - offset
    )
    constrained = original_matrix.copy()
    constrained[np.asarray(mask)] = np.eye(4)[np.asarray(mask)]
    constrained_right = np.where(mask, inputs.voltage_clamp_target_mV, original_right)
    expected = np.linalg.solve(constrained, constrained_right)
    expected_current = np.where(mask, original_matrix @ expected - original_right, 0.0)
    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.state.voltage_mV, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.evidence.clamp_current_nA, expected_current, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(result.evidence.kirchhoff_residual_nA, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.evidence.charge_balance_residual_nA, 0.0, atol=1e-12
    )
    np.testing.assert_allclose(result.state.time_ms, elapsed)


def test_dynamic_elapsed_and_clamp_target_derivatives_include_gates_and_current():
    runtime = _branched(
        "crank-nicolson", (ep.PassiveLeak(0.1, -65.0), ep.HodgkinHuxleyNaK())
    )
    state = ep.initialize_cable_state(runtime, jnp.asarray([-61.0, -66.0, -65.0, -62.0]))
    state = eqx.tree_at(
        lambda value: value.membrane.gates[1], state, state.membrane.gates[1] * 0.9
    )
    neutral = ep.zero_cable_inputs(runtime)

    def objective(parameters):
        inputs = eqx.tree_at(
            lambda value: (
                value.injected_current_nA,
                value.voltage_clamp_mask,
                value.voltage_clamp_target_mV,
            ),
            neutral,
            (
                jnp.asarray([parameters[0], 0.0, 0.0, 0.0]),
                jnp.asarray([True, False, False, False]),
                jnp.full((4,), parameters[1]),
            ),
        )
        result = ep.step_cable(runtime, state, inputs, elapsed_ms=parameters[2])
        return (
            jnp.sum(result.state.voltage_mV)
            + 0.1 * jnp.sum(result.state.membrane.gates[1])
            + jnp.sum(result.evidence.clamp_current_nA)
        )

    parameters = jnp.asarray([0.1, -58.0, 0.027])
    direction = jnp.asarray([0.03, 0.2, 0.004])
    _, forward = jax.jvp(objective, (parameters,), (direction,))
    reverse = jnp.dot(jax.grad(objective)(parameters), direction)
    epsilon = 1e-5
    finite_difference = (
        objective(parameters + epsilon * direction)
        - objective(parameters - epsilon * direction)
    ) / (2 * epsilon)
    np.testing.assert_allclose(forward, finite_difference, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(reverse, finite_difference, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize(
    "diagonal,lower,upper,parents",
    [
        ([0.0], [0.0], [0.0], [-1]),
        ([2.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1, 0]),
        ([1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [-1, 0]),
    ],
)
def test_invalid_leaf_or_root_pivot_fails_closed_even_for_zero_rhs(
    diagonal, lower, upper, parents
):
    operator = TreeLinearOperator(diagonal, lower, upper, TreeTopology(parents))
    result = solve(
        LinearSystem(operator),
        jnp.zeros((len(parents),)),
        policy=LinearSolvePolicy(StructuredDirect()),
    )
    assert not bool(result.successful)
    assert not bool(jnp.all(jnp.isfinite(result.value)))


class _SixGateCurrent(eqx.Module):
    mechanism_id: str = eqx.field(static=True, default="six-gate-current")
    gate_count: int = eqx.field(static=True, default=6)
    nonlinear: bool = eqx.field(static=True, default=False)

    def initial_gates(self, voltage_mV, /):
        return jnp.ones(voltage_mV.shape + (self.gate_count,))

    def affine_current(self, voltage_mV, gates, area, intracellular, extracellular, /):
        del intracellular, extracellular
        conductance = 1e-5 * area * jnp.mean(gates, axis=-1)
        return (
            conductance,
            65.0 * conductance,
            jnp.zeros_like(voltage_mV),
            jnp.zeros_like(voltage_mV, dtype=jnp.int32),
        )

    def update_gates(self, voltage_mV, gates, dt_ms, /):
        del voltage_mV
        return ep.exact_affine_gate_update(
            gates, jnp.ones_like(gates), jnp.arange(1.0, self.gate_count + 1), dt_ms
        )


def test_actual_gate_counts_dynamic_update_and_atomic_failure():
    runtime = _branched("backward-euler", (ep.PassiveLeak(0.1, -65.0), _SixGateCurrent()))
    state = ep.initialize_cable_state(runtime, jnp.full((4,), -65.0))
    state = eqx.tree_at(
        lambda value: value.membrane.gates[1], state, jnp.full((4, 6), 0.2)
    )
    neutral = ep.zero_cable_inputs(runtime)
    advanced = ep.step_cable(runtime, state, neutral, elapsed_ms=jnp.asarray(0.3))
    assert bool(advanced.evidence.successful)
    expected_gates = np.broadcast_to(
        1.0 - 0.8 * np.exp(-0.3 / np.arange(1.0, 7.0)), (4, 6)
    )
    np.testing.assert_allclose(
        advanced.state.membrane.gates[1], expected_gates, rtol=1e-13
    )
    assert advanced.state.membrane.gates[0].shape == (4, 0)
    for elapsed in (0.0, -0.1, np.nan):
        rejected = ep.step_cable(runtime, state, neutral, elapsed_ms=jnp.asarray(elapsed))
        assert not bool(rejected.evidence.successful)
        for prior, retained in zip(
            jax.tree.leaves(state), jax.tree.leaves(rejected.state), strict=True
        ):
            np.testing.assert_array_equal(retained, prior)


def test_singular_cable_retains_all_prior_state():
    morphology = ep.CellMorphologyPlan(
        "singular", (ep.CompartmentSpec("soma", None, 10.0, 10.0),)
    ).prepare()
    runtime = ep.CableSolverPlan(0.1).prepare(
        morphology, ep.MembraneProgram((ep.PassiveLeak(0.0, -65.0),))
    )
    state = ep.initialize_cable_state(runtime, jnp.asarray([-65.0]))
    inputs = ep.zero_cable_inputs(runtime)
    inputs = eqx.tree_at(
        lambda value: value.synaptic_conductance_uS,
        inputs,
        -morphology.capacitance_nF / runtime.plan.dt_ms,
    )
    rejected = ep.step_cable(runtime, state, inputs)
    assert not bool(rejected.evidence.successful)
    assert int(rejected.evidence.status) & int(ep.CableSolveStatus.LINEAR_FAILURE)
    for prior, retained in zip(
        jax.tree.leaves(state), jax.tree.leaves(rejected.state), strict=True
    ):
        np.testing.assert_array_equal(retained, prior)
