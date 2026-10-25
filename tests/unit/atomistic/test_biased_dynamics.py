"""Biased transactions: constraint projection, terminal exhaustion, continuation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic import PreparedAtomisticSystem
from phydrax.atomistic._constraints import DistanceConstraintPlan
from phydrax.atomistic._dynamics import AtomisticDynamicsPlan, VelocityVerletPlan
from phydrax.atomistic._thermal import BAOABLangevinPlan
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.atomistic.sampling import (
    AtomisticABFForcePort,
    AtomisticABFPlan,
    AtomisticABFState,
    AtomisticBiasHistoryPlan,
    AtomisticBiasHistoryState,
    BiasedDynamicsState,
    BiasedStepReason,
    CollectiveVariableKind,
    grow_bias_history_capacity,
    HarmonicBiasForm,
    HillHistoryBiasForm,
    PreparedBiasedDynamics,
    read_biased_dynamics_checkpoint,
    write_biased_dynamics_checkpoint,
)
from tests._support.atomistic_dynamics_fixtures import (
    atomistic_system,
    measure,
    prepare_potential,
    row,
    table,
)
from tests._support.atomistic_sampling_fixtures import (
    bias_term,
    biased_dynamics,
    cv_program,
    nve_table,
    start,
    zero_field,
)


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


POSITIONS = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 2.0, 0.0]])


def test_bias_forces_enter_the_kick_before_rattle_projection() -> None:
    # Baseline defect: the old wrapper kicked after projection (residual -0.02).
    topology = MolecularTopologyPlan(constraints=[[0, 1]], constraint_distances=[1.2])
    system = atomistic_system([0, 1, 2], [1.0] * 3, topology=topology)
    constraints = DistanceConstraintPlan(tolerance=1.0e-12).prepare(system)
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 2])
    term = bias_term(system, variables, HarmonicBiasForm(np.asarray([1.0]), 10.0))
    dynamics = biased_dynamics(
        system, term, VelocityVerletPlan(1.0e-2), constraints=constraints
    )
    thermodynamic = nve_table(dynamics)
    state = start(dynamics, thermodynamic, POSITIONS)
    for _ in range(3):
        result = dynamics.step_detailed(state, thermodynamic)
        assert bool(result.successful)
        state = result.accepted_state
    q = state.kinematics.positions
    v = state.kinematics.momenta / system.plan.masses[:, None]
    bond = q[0] - q[1]
    residual = float(jnp.dot(bond, v[0] - v[1]) / jnp.linalg.norm(bond))
    assert abs(residual) <= 1.0e-10
    np.testing.assert_allclose(
        float(state.force.potential_energy),
        0.5 * 10.0 * (float(jnp.linalg.norm(q[0] - q[2])) - 1.0) ** 2,
    )


def test_bias_on_the_constrained_distance_is_removed_by_the_projection() -> None:
    # Exact baseline scenario: a bias on the rigid bond itself acts only along
    # the constraint, so the projected step must keep every momentum at zero.
    topology = MolecularTopologyPlan(constraints=[[0, 1]], constraint_distances=[1.2])
    system = atomistic_system([0, 1, 2], [1.0] * 3, topology=topology)
    constraints = DistanceConstraintPlan(tolerance=1.0e-12).prepare(system)
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    term = bias_term(system, variables, HarmonicBiasForm(np.asarray([1.0]), 10.0))
    dynamics = biased_dynamics(
        system, term, VelocityVerletPlan(1.0e-2), constraints=constraints
    )
    thermodynamic = nve_table(dynamics)
    runtime = PreparedBiasedDynamics(dynamics, thermodynamic, physical_force_groups=(0,))
    stepped = runtime.step(runtime.initialize(start(dynamics, thermodynamic, POSITIONS)))
    assert bool(stepped.successful)
    state = stepped.accepted_state.dynamics
    assert int(state.step_index) == 1
    q = state.kinematics.positions
    v = state.kinematics.momenta / system.plan.masses[:, None]
    bond = q[0] - q[1]
    residual = float(jnp.dot(bond, v[0] - v[1]) / jnp.linalg.norm(bond))
    assert abs(residual) <= 1.0e-10
    np.testing.assert_allclose(float(jnp.linalg.norm(bond)), 1.2, rtol=0.0, atol=1.0e-10)
    assert float(jnp.max(jnp.abs(v))) <= 1.0e-10
    np.testing.assert_allclose(
        float(state.force.potential_energy), 0.5 * 10.0 * (1.2 - 1.0) ** 2, rtol=1.0e-10
    )


def _metadynamics(
    capacity: int,
) -> tuple[PreparedAtomisticSystem, PreparedBiasedDynamics, BiasedDynamicsState]:
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    history = AtomisticBiasHistoryPlan(1, height=0.1, widths=0.1, capacity=capacity)
    term = bias_term(system, variables, HillHistoryBiasForm(history))
    dynamics = biased_dynamics(system, term, VelocityVerletPlan(1.0e-3))
    thermodynamic = nve_table(dynamics)
    runtime = PreparedBiasedDynamics(dynamics, thermodynamic, physical_force_groups=(0,))
    return system, runtime, runtime.initialize(start(dynamics, thermodynamic, POSITIONS))


def _history(
    runtime: PreparedBiasedDynamics, state: BiasedDynamicsState
) -> AtomisticBiasHistoryState:
    """The history of the runtime's single bias term, validated by that term."""
    term = runtime.bias_terms[0]
    history = term.require_state(state.dynamics.term_states[term.name]).history
    if history is None:
        raise AssertionError(f"Bias term {term.name!r} binds no history.")
    return history


def _leaf_contract(
    state: BiasedDynamicsState,
) -> list[tuple[str, tuple[int, ...], str]]:
    """Path, shape and dtype of every state leaf (the checkpoint template contract)."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    return [
        (jax.tree_util.keystr(path), tuple(leaf.shape), str(leaf.dtype))
        for path, leaf in leaves
    ]


def test_accepted_and_rejected_steps_keep_the_initialized_leaf_contract() -> None:
    # Regression: the deposition counters (history ``written``, the
    # ``deposition_events`` ledger) were promoted from int32 to int64 by the
    # x64 integer reductions, so the state no longer matched its template.
    _, runtime, state = _metadynamics(4)
    contract = _leaf_contract(state)
    for _ in range(3):
        result = runtime.step(state)
        assert bool(result.successful)
        assert _leaf_contract(result.candidate_state) == contract
        assert _leaf_contract(result.accepted_state) == contract
        state = result.accepted_state
    assert int(state.deposition_events) == int(_history(runtime, state).written) == 3
    # Capacity one: one accepted deposit, then latched terminal rejections.
    _, runtime, state = _metadynamics(1)
    contract = _leaf_contract(state)
    accepted = []
    for _ in range(3):
        result = runtime.step(state)
        accepted.append(bool(result.successful))
        assert _leaf_contract(result.candidate_state) == contract
        assert _leaf_contract(result.accepted_state) == contract
        state = result.accepted_state
    assert accepted == [True, False, False]


def test_history_exhaustion_is_terminal_and_explicit_then_grows() -> None:
    # Baseline defect 7: accepted pattern [T, F, F, F] with a silent repeated retry.
    system, runtime, state = _metadynamics(1)
    first = runtime.step(state)
    assert bool(first.successful) and int(first.accepted_state.deposition_events) == 1
    work = float(first.history_work)
    assert work > 0.0
    second = runtime.step(first.accepted_state)
    assert not bool(second.successful)
    assert int(second.reasons) & int(BiasedStepReason.HISTORY_EXHAUSTED)
    assert int(second.accepted_state.dynamics.step_index) == int(
        first.accepted_state.dynamics.step_index
    )
    term_id = runtime.bias_terms[0].name
    history = _history(runtime, second.accepted_state)
    assert int(history.written) == 1
    # Terminal: the latched status refuses again without committing anything.
    repeated = runtime.step(second.accepted_state)
    assert not bool(repeated.successful)
    assert int(repeated.reasons) & int(BiasedStepReason.HISTORY_EXHAUSTED)
    assert int(repeated.accepted_state.deposition_events) == 1
    np.testing.assert_array_equal(
        np.asarray(repeated.accepted_state.dynamics.random_key),
        np.asarray(second.accepted_state.dynamics.random_key),
    )
    assert int(repeated.accepted_state.dynamics.step_index) == int(
        first.accepted_state.dynamics.step_index
    )
    grown, resumed = grow_bias_history_capacity(
        runtime,
        second.accepted_state,
        term_id,
        4,
        (row(measure(system), "nve"),),
    )
    third = grown.step(resumed)
    assert bool(third.successful)
    assert int(_history(grown, third.accepted_state).written) == 2


def test_checkpoint_restores_exact_history_in_a_fresh_runtime(tmp_path: Path) -> None:
    _, runtime, state = _metadynamics(4)
    for _ in range(2):
        state = runtime.step(state).accepted_state
    path = tmp_path / "biased.npz"
    written = write_biased_dynamics_checkpoint(path, runtime, state)
    _, fresh_runtime, template = _metadynamics(4)
    restored = read_biased_dynamics_checkpoint(path, fresh_runtime, template)
    assert restored.payload_id == written.payload_id
    # Every leaf (history, counters, ledgers, forces, random key) is exact.
    assert _leaf_contract(restored.state) == _leaf_contract(state)
    for value, reference in zip(
        jax.tree_util.tree_leaves(restored.state),
        jax.tree_util.tree_leaves(state),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(value), np.asarray(reference))
    np.testing.assert_array_equal(
        np.asarray(_history(fresh_runtime, restored.state).centers),
        np.asarray(_history(runtime, state).centers),
    )
    continued = fresh_runtime.step(restored.state)
    reference = runtime.step(state)
    np.testing.assert_allclose(
        np.asarray(continued.accepted_state.dynamics.kinematics.positions),
        np.asarray(reference.accepted_state.dynamics.kinematics.positions),
        rtol=0.0,
        atol=0.0,
    )


def test_fresh_forces_after_history_change_equal_a_new_evaluation() -> None:
    _, runtime, state = _metadynamics(4)
    state = runtime.step(state).accepted_state.dynamics
    reference = runtime.dynamics.evaluate_state(state, runtime.thermodynamic)
    np.testing.assert_allclose(
        np.asarray(state.force.forces),
        np.asarray(reference.forces),
        rtol=1e-13,
        atol=1e-14,
    )
    assert bool(
        runtime.dynamics.force_cache_valid(
            state,
            runtime.thermodynamic.state_at_replica(
                state.thermodynamic_state_index
            ).controls,
        )
    )


def _abf_runtime(
    min_samples: int,
) -> tuple[PreparedBiasedDynamics, BiasedDynamicsState]:
    # The adaptive-bias review control: BAOAB NVT at the ABF temperature.
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    abf = AtomisticABFPlan(
        variables,
        minimum=0.5,
        maximum=2.0,
        bins=8,
        temperature=0.5,
        min_samples=min_samples,
    ).prepare(system)
    dynamics = AtomisticDynamicsPlan(
        system,
        prepare_potential(system, (zero_field(),)),
        BAOABLangevinPlan(1.0e-3, 0.1),
        force_ports=(AtomisticABFForcePort(abf, name="abf"),),
    ).prepare()
    thermodynamic = table(dynamics, row(measure(system), "nvt", temperature=0.5))
    runtime = PreparedBiasedDynamics(dynamics, thermodynamic, physical_force_groups=(0,))
    return runtime, runtime.initialize(start(dynamics, thermodynamic, POSITIONS))


def _abf_state(state: BiasedDynamicsState) -> AtomisticABFState:
    value = state.dynamics.term_states["abf"]
    if not isinstance(value, AtomisticABFState):
        raise AssertionError("The ABF slot binds another state type.")
    return value


def test_abf_statistics_below_the_ramp_are_committed_without_a_force_change() -> None:
    # Review baseline: two reported samples, zero retained counts, revision 0.
    runtime, state = _abf_runtime(200)
    for _ in range(3):
        result = runtime.step(state)
        assert bool(result.successful)
        state = result.accepted_state
    retained = _abf_state(state)
    assert int(state.abf_samples) == 3
    assert int(jnp.sum(retained.counts)) == 3
    assert int(retained.revision) == 0
    assert int(retained.out_of_range) == 0 and int(retained.irregular) == 0
    # Below the ramp the applied field is unchanged: the cache stays valid
    # without a re-prime and equals a fresh evaluation.
    dynamics_state = state.dynamics
    controls = runtime.thermodynamic.state_at_replica(
        dynamics_state.thermodynamic_state_index
    ).controls
    assert bool(runtime.dynamics.force_cache_valid(dynamics_state, controls))
    reference = runtime.dynamics.evaluate_state(dynamics_state, runtime.thermodynamic)
    np.testing.assert_allclose(
        np.asarray(dynamics_state.force.forces),
        np.asarray(reference.forces),
        rtol=1e-13,
        atol=1e-14,
    )


def test_abf_ramp_reaches_the_applied_force_through_committed_steps() -> None:
    # min_samples=1, full_samples=2: the ramp is zero at one sample, full at two.
    runtime, state = _abf_runtime(1)
    revisions = []
    for _ in range(3):
        result = runtime.step(state)
        assert bool(result.successful)
        state = result.accepted_state
        revisions.append(int(_abf_state(state).revision))
    retained = _abf_state(state)
    assert int(jnp.sum(retained.counts)) == int(state.abf_samples) == 3
    # The pair stays in one bin over three short steps: its ramp opens at the
    # second sample, after which every sample changes the applied force.
    assert revisions == [0, 1, 2]
    dynamics_state = state.dynamics
    reference = runtime.dynamics.evaluate_state(dynamics_state, runtime.thermodynamic)
    np.testing.assert_allclose(
        np.asarray(dynamics_state.force.forces),
        np.asarray(reference.forces),
        rtol=1e-13,
        atol=1e-14,
    )
