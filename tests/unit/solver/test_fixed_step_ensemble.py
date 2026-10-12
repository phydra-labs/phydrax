#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


def _decay(time: Array, state: Array, args: Any, /) -> Array:
    del time, args
    return -state


def _problem() -> phx.solver.FixedStepProblem:
    return phx.solver.FixedStepProblem(
        phx.solver.SSPRK33FixedStepMethod(_decay),
        jnp.ones((2,), dtype=jnp.float64),
        t0=0.0,
        t1=0.05,
        step_size=0.01,
    )


def _states() -> Array:
    return jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float64)


def _plan(*, chunk: int = 2, cap: int = 2**24) -> phx.solver.FixedStepEnsemblePlan:
    return phx.solver.FixedStepEnsemblePlan(
        _problem(),
        _states(),
        ("first", "second", "third"),
        member_chunk_size=chunk,
        maximum_output_bytes=cap,
    )


@pytest.mark.parametrize(
    "stride", [1, 2, 9], ids=["trajectory", "checkpoints", "endpoints"]
)
def test_members_match_standalone_compiled_solutions(stride: int) -> None:
    template = _problem()
    problem = eqx.tree_at(
        lambda p: p.initial_state,
        template,
        template.initial_state.astype(jnp.complex128),
    )
    plan = phx.solver.FixedStepEnsemblePlan(
        problem,
        _states().astype(jnp.complex128),
        ("first", "second", "third"),
        member_chunk_size=2,
    )
    result = phx.solver.execute_fixed_step_ensemble(plan.prepare(save_every=stride))
    standalone = eqx.filter_jit(phx.solver.solve_fixed_step)
    for i in range(plan.member_count):
        problem = eqx.tree_at(
            lambda p: p.initial_state, plan.problem, plan.initial_states[i]
        )
        expected = standalone(problem, save_every=stride)
        tolerance = 8 * np.finfo(expected.states.dtype).eps
        state_scale = float(jnp.max(jnp.abs(plan.initial_states[i])))
        np.testing.assert_allclose(
            result.states[i],
            expected.states,
            rtol=tolerance,
            atol=tolerance * state_scale,
        )
        np.testing.assert_array_equal(result.valid[i], expected.valid)
        assert jnp.issubdtype(result.times.dtype, jnp.floating)
        np.testing.assert_array_equal(result.times, jnp.real(expected.times))
    assert bool(result.evidence.successful)


def test_chunk_policy_preserves_member_identity_and_numerics() -> None:
    first = _plan(chunk=1)
    second = _plan(chunk=4)
    assert first.path_ids == second.path_ids
    assert first.realization_ids == second.realization_ids
    assert first.plan_id != second.plan_id
    a = phx.solver.execute_fixed_step_ensemble(first.prepare())
    b = phx.solver.execute_fixed_step_ensemble(second.prepare())
    np.testing.assert_allclose(
        a.states, b.states, rtol=8 * np.finfo(np.float64).eps, atol=0
    )
    assert a.result_id != b.result_id


def test_padding_is_excluded_from_evidence_reductions_and_replay() -> None:
    prepared = _plan(chunk=4).prepare()
    result = phx.solver.execute_fixed_step_ensemble(prepared)
    replay = phx.solver.execute_fixed_step_ensemble(prepared)
    assert prepared.padded_member_count == 4
    assert result.states.shape == (3, 6, 2)
    assert result.status.shape == (3,)
    assert len(result.realization_ids) == len(result.path_ids) == 3
    assert result.solutions.successful.shape == (3,)
    assert float(result.evidence.effective_sample_size) == pytest.approx(3.0)
    np.testing.assert_allclose(result.mean(), np.asarray(result.states).mean(axis=0))
    np.testing.assert_array_equal(result.states, replay.states)
    assert result.result_id == replay.result_id


def _nan_step(
    step_index: Array, time: Array, state: Array, step_size: Array, args: Any, /
) -> phx.solver.FixedStepResult:
    del time, step_size
    fault = (state[0] > 2.0) & (step_index == 1)
    candidate = jnp.where(fault, jnp.full_like(state, jnp.nan), state)
    successful = ~fault if args else jnp.asarray(True)
    accepted = jnp.where(successful, candidate, state)
    return phx.solver.FixedStepResult(
        candidate,
        accepted,
        successful,
        jnp.zeros((), dtype=state.dtype),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.zeros((), dtype=state.dtype),
        evidence={"fault": fault},
    )


@pytest.mark.parametrize(
    ("refuse", "status"),
    [
        (True, phx.solver.DeterministicEnsembleStatus.SOLVER_FAILURE),
        (False, phx.solver.DeterministicEnsembleStatus.NONFINITE_TRAJECTORY),
    ],
    ids=["method-refusal", "nonfinite-accepted"],
)
def test_member_fault_is_exposed_without_poisoning_reducer(
    refuse: bool, status: phx.solver.DeterministicEnsembleStatus
) -> None:
    problem = phx.solver.FixedStepProblem(
        phx.solver.CallableFixedStepMethod(_nan_step, "nan-member"),
        _states()[0],
        t0=0.0,
        t1=0.03,
        step_size=0.01,
        args=refuse,
    )
    result = phx.solver.execute_fixed_step_ensemble(
        phx.solver.FixedStepEnsemblePlan(
            problem, _states()[:2], ("good", "fault"), member_chunk_size=3
        ).prepare()
    )
    np.testing.assert_array_equal(result.status, [0, int(status)])
    np.testing.assert_array_equal(result.reducer.valid, [True, False])
    np.testing.assert_array_equal(result.mean(), result.states[0])
    assert not bool(result.evidence.successful)
    assert float(result.evidence.effective_sample_size) == 1.0
    evidence = result.solutions.evidence
    assert evidence is not None
    if refuse:
        np.testing.assert_array_equal(evidence.refused_step, [-1, 1])
        np.testing.assert_array_equal(evidence.refused["fault"], [False, True])


def _tree_step(
    step_index: Array,
    time: Array,
    state: dict[str, Array],
    step_size: Array,
    args: Any,
    /,
) -> phx.solver.FixedStepResult:
    del step_index, time, step_size, args
    candidate = {"position": state["position"] + 1.0, "route": state["route"] + 1}
    return phx.solver.FixedStepResult(
        candidate,
        candidate,
        jnp.asarray(True),
        jnp.zeros((), dtype=jnp.float64),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.zeros((), dtype=jnp.float64),
    )


def test_structured_mixed_dtype_states_match_standalone() -> None:
    state = {
        "position": jnp.zeros((2,), dtype=jnp.float64),
        "route": jnp.asarray([4, 7], dtype=jnp.int32),
    }
    problem = phx.solver.FixedStepProblem(
        phx.solver.CallableFixedStepMethod(_tree_step, "tree-addition"),
        state,
        t0=0.0,
        t1=0.03,
        step_size=0.01,
        state_geometry=phx.discretization.DEMStateGeometry("ensemble-tree"),
    )
    states = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf + 1]), state)
    result = phx.solver.execute_fixed_step_ensemble(
        phx.solver.FixedStepEnsemblePlan(problem, states, ("a", "b")).prepare()
    )
    standalone = eqx.filter_jit(phx.solver.solve_fixed_step)
    expected = standalone(problem)
    np.testing.assert_array_equal(
        result.states["position"][0], expected.states["position"]
    )
    np.testing.assert_array_equal(result.states["route"][0], expected.states["route"])
    assert result.states["route"].dtype == jnp.int32
    assert bool(result.evidence.successful)


@pytest.mark.parametrize("cap", [1, None], ids=["retained-output", "compiler-memory"])
def test_resource_refusal_carries_required_and_limit_bytes(cap: int | None) -> None:
    admitted = _plan().prepare()
    limit = admitted.estimated_output_bytes if cap is None else cap
    assert admitted.required_bytes > admitted.estimated_output_bytes
    with pytest.raises(phx.execution.MemoryBudgetError) as refusal:
        _plan(cap=limit).prepare()
    assert refusal.value.required_bytes > refusal.value.limit_bytes
    assert refusal.value.limit_bytes == limit
    assert refusal.value.budget == "maximum_output_bytes"


def test_admission_accounts_for_unpadded_compiled_outputs() -> None:
    prepared = _plan(chunk=4).prepare()
    assert prepared.required_bytes >= prepared.compiler_memory.output_bytes
    solutions, finite, status = prepared.executable(
        prepared.plan.problem,
        prepared.plan.initial_states,
        prepared.plan.member_count,
        prepared.plan.member_chunk_size,
        prepared.save_every,
    )
    assert solutions.states.shape[0] == prepared.plan.member_count
    assert finite.shape == status.shape == (prepared.plan.member_count,)


@pytest.mark.parametrize(
    "policy", ["member_chunk_size", "maximum_members", "maximum_output_bytes"]
)
@pytest.mark.parametrize("value", [True, "2", 2.5])
def test_plan_refuses_wrong_policy_kinds(policy: str, value: object) -> None:
    with pytest.raises(TypeError):
        phx.solver.FixedStepEnsemblePlan(
            _problem(),
            _states(),
            ("a", "b", "c"),
            **{policy: value},  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("value", [True, "2", 2.5])
def test_prepare_refuses_wrong_save_every_kind(value: object) -> None:
    with pytest.raises(TypeError):
        _plan().prepare(save_every=value)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "weights",
    [np.asarray([True, False, True]), np.asarray([1.0, 1.0, 1.0j]), ("1", "1", "1")],
    ids=["bool", "complex", "string"],
)
def test_plan_refuses_wrong_weight_kinds(weights: object) -> None:
    with pytest.raises(TypeError):
        phx.solver.FixedStepEnsemblePlan(
            _problem(),
            _states(),
            ("a", "b", "c"),
            weights=weights,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("states", "labels"),
    [
        (_states(), ("same", " same ", "last")),
        (_states()[:, :1], ("a", "b", "c")),
        (_states().astype(jnp.float32), ("a", "b", "c")),
        (_states().at[1, 0].set(jnp.nan), ("a", "b", "c")),
    ],
    ids=["duplicate-label", "shape-mismatch", "dtype-mismatch", "nonfinite-initial"],
)
def test_plan_refuses_invalid_member_declarations(
    states: Array, labels: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        phx.solver.FixedStepEnsemblePlan(_problem(), states, labels)


def test_explicit_weights_control_only_valid_member_mass() -> None:
    plan = phx.solver.FixedStepEnsemblePlan(
        _problem(),
        _states(),
        ("a", "b", "c"),
        weights=jnp.asarray([0.0, 1.0, 3.0], dtype=jnp.float64),
    )
    result = phx.solver.execute_fixed_step_ensemble(plan.prepare())
    np.testing.assert_array_equal(result.reducer.normalized_weights, [0.0, 0.25, 0.75])
    np.testing.assert_allclose(
        result.mean(), 0.25 * result.states[1] + 0.75 * result.states[2]
    )
