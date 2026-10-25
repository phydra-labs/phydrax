"""Adaptive kernel histories: cadence, owners, simultaneous deposits and capacity."""

from __future__ import annotations

import math
from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic.sampling import (
    AtomisticBiasDeposition,
    AtomisticBiasHistoryPlan,
    AtomisticBiasHistoryState,
    AtomisticBiasTermState,
    BiasHistoryOwnership,
    BiasHistoryStorage,
    CollectiveVariableGeometry,
    CollectiveVariableKind,
    CollectiveVariableMetric,
    CollectiveVariableStatus,
    deposit_bias_history,
    FrozenBiasHistory,
    grow_bias_history,
    HillHistoryBiasForm,
    HillKernel,
    PreparedAtomisticBiasTerm,
    warm_start_bias_history,
    with_bias_history,
)
from phydrax.atomistic.sampling._bias import BiasFormEvaluation
from tests._support.atomistic_dynamics_fixtures import atomistic_system
from tests._support.atomistic_sampling_fixtures import bias_term, cv_program


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


PLAIN = (CollectiveVariableMetric(),)


def _plan(
    *,
    widths: float = 0.5,
    capacity: int = 4,
    interval: int = 1,
    ramp_depositions: int = 0,
    bias_factor: float | None = None,
    temperature: float | None = None,
    ownership: BiasHistoryOwnership = BiasHistoryOwnership.SHARED,
    owner_count: int = 1,
    storage: BiasHistoryStorage = BiasHistoryStorage.TERMINAL,
    kernel: HillKernel = HillKernel.NEAREST_IMAGE,
    wrapped_images: int = 0,
) -> AtomisticBiasHistoryPlan:
    return AtomisticBiasHistoryPlan(
        1,
        height=1.0,
        widths=widths,
        capacity=capacity,
        interval=interval,
        ramp_depositions=ramp_depositions,
        bias_factor=bias_factor,
        temperature=temperature,
        ownership=ownership,
        owner_count=owner_count,
        storage=storage,
        kernel=kernel,
        wrapped_images=wrapped_images,
    )


def _deposit(
    plan: AtomisticBiasHistoryPlan,
    state: AtomisticBiasHistoryState,
    values: list[float],
    owners: list[int],
    active: list[bool],
    metrics: tuple[CollectiveVariableMetric, ...] = PLAIN,
) -> AtomisticBiasDeposition:
    return deposit_bias_history(
        plan,
        state,
        jnp.asarray(values, dtype=jnp.float64).reshape((-1, 1)),
        jnp.asarray(owners, dtype=jnp.int32),
        jnp.asarray(active, dtype=jnp.bool_),
        metrics,
        boltzmann_constant=1.0,
    )


def test_cadence_is_on_committed_steps() -> None:
    plan = _plan(interval=3)
    assert [bool(plan.due(step)) for step in range(7)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_simultaneous_well_tempered_heights_use_the_pre_event_history() -> None:
    plan = _plan(bias_factor=5.0, temperature=2.0)
    state = plan.initialize()
    first = _deposit(plan, state, [0.0], [0], [True])
    np.testing.assert_allclose(np.asarray(first.heights), [1.0])
    event = _deposit(plan, first.state, [0.0, 0.0], [0, 0], [True, True])
    pre = 1.0  # one unit hill at the deposit point
    expected = math.exp(-pre / (2.0 * (5.0 - 1.0)))
    np.testing.assert_allclose(np.asarray(event.heights), [expected, expected])
    assert np.asarray(event.slots).tolist() == [1, 2]
    np.testing.assert_allclose(
        float(plan.evaluate(jnp.asarray([0.0]), PLAIN, event.state, 0).energy),
        1.0 + 2.0 * expected,
    )


def test_terminal_exhaustion_is_all_or_nothing_and_explicit() -> None:
    # Baseline defect 7: exhaustion previously retried and failed every step.
    plan = _plan(capacity=2)
    state = _deposit(plan, plan.initialize(), [0.0], [0], [True]).state
    refused = _deposit(plan, state, [1.0, 2.0], [0, 0], [True, True])
    assert not bool(refused.inserted) and bool(refused.exhausted)
    assert int(refused.state.written) == 1
    np.testing.assert_array_equal(np.asarray(refused.state.valid), [True, False])
    grown_plan, grown = grow_bias_history(plan, state, 4)
    assert grown_plan.scientific_id == plan.scientific_id
    assert grown_plan.history_id != plan.history_id
    accepted = _deposit(grown_plan, grown, [1.0, 2.0], [0, 0], [True, True])
    assert bool(accepted.inserted) and int(accepted.state.written) == 3


def test_fifo_overwrites_oldest_and_is_flagged_for_analysis() -> None:
    plan = _plan(capacity=2, storage=BiasHistoryStorage.FIFO)
    state = plan.initialize()
    for value in (0.0, 1.0, 2.0):
        state = _deposit(plan, state, [value], [0], [True]).state
    np.testing.assert_allclose(np.asarray(state.centers)[:, 0], [2.0, 1.0])
    assert FrozenBiasHistory(plan, state).fifo_overwritten


def test_owner_tags_select_walker_and_state_histories() -> None:
    plan = _plan(ownership=BiasHistoryOwnership.STATE, owner_count=2)
    state = _deposit(plan, plan.initialize(), [0.0, 0.0], [0, 1], [True, False]).state
    owned = plan.evaluate(jnp.asarray([0.0]), PLAIN, state, 0)
    assert float(owned.energy) == pytest.approx(1.0) and bool(owned.successful)
    assert float(plan.evaluate(jnp.asarray([0.0]), PLAIN, state, 1).energy) == 0.0


def test_ramp_scales_new_kernels_until_later_events() -> None:
    plan = _plan(ramp_depositions=2)
    state = _deposit(plan, plan.initialize(), [0.0], [0], [True]).state
    # Age one deposition event: half height (event counter semantics of the source).
    np.testing.assert_allclose(
        float(plan.evaluate(jnp.asarray([0.0]), PLAIN, state, 0).energy), 0.5
    )
    state = _deposit(plan, state, [5.0], [0], [True]).state
    np.testing.assert_allclose(
        float(plan.evaluate(jnp.asarray([0.0]), PLAIN, state, 0).energy),
        1.0,
        atol=1e-12,
    )


def test_wrapped_kernel_reports_its_truncation_tail() -> None:
    period = 2.0 * math.pi
    metrics = (CollectiveVariableMetric(period=period),)
    nearest = _plan(widths=2.0)
    wrapped = _plan(widths=2.0, kernel=HillKernel.WRAPPED, wrapped_images=2)
    assert wrapped.seam_kernel_bound(metrics) < nearest.seam_kernel_bound(metrics)
    state = _deposit(
        wrapped, wrapped.initialize(), [math.pi - 0.01], [0], [True], metrics
    ).state
    left = wrapped.evaluate(jnp.asarray([math.pi - 1e-6]), metrics, state, 0).energy
    right = wrapped.evaluate(jnp.asarray([-math.pi + 1e-6]), metrics, state, 0).energy
    np.testing.assert_allclose(float(left), float(right), rtol=1e-5)


def test_warm_start_is_a_new_initial_condition() -> None:
    plan = _plan()
    state = _deposit(plan, plan.initialize(), [0.3], [0], [True]).state
    warm = warm_start_bias_history(plan, FrozenBiasHistory(plan, state))
    np.testing.assert_allclose(np.asarray(warm.centers)[0, 0], 0.3)
    assert int(warm.written) == 1


TRIANGLE = np.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0]])
# Collinear points: rank-one correlation with any reference, so the top
# key-matrix eigenvalue is doubly degenerate and the superposition undefined.
COLLINEAR = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.2, 0.0, 0.0]])


def _aligned_term(
    *,
    ownership: BiasHistoryOwnership = BiasHistoryOwnership.SHARED,
    owner_count: int = 1,
    kernel_owner: int = 0,
) -> tuple[PreparedAtomisticBiasTerm, AtomisticBiasTermState]:
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = cv_program(system, CollectiveVariableKind.CONFIGURATION, [0, 1, 2])
    plan = AtomisticBiasHistoryPlan(
        9,
        height=1.0,
        widths=0.5,
        capacity=4,
        ownership=ownership,
        owner_count=owner_count,
        kernel=HillKernel.ALIGNED_MSD,
    )
    term = bias_term(system, variables, HillHistoryBiasForm(plan)).prepare(system)
    initial = term.initial_term_state(jnp.float64)
    if initial is None or initial.history is None:
        raise AssertionError("A hill term binds a history state.")
    # One kernel at the triangle; three padding slots stay invalid.
    event = deposit_bias_history(
        plan,
        initial.history,
        jnp.asarray(TRIANGLE.reshape((1, 9))),
        jnp.asarray([kernel_owner], dtype=jnp.int32),
        jnp.asarray([True]),
        variables.metrics,
        boltzmann_constant=1.0,
    )
    assert bool(event.inserted)
    return term, with_bias_history(initial, event.state)


def _form_evidence(
    term: PreparedAtomisticBiasTerm, state: AtomisticBiasTermState, query: np.ndarray
) -> BiasFormEvaluation:
    values = term.plan.variables.evaluate(
        CollectiveVariableGeometry(jnp.asarray(query))
    ).values
    return term.plan.form.evaluate(
        values,
        term.plan.variables.metrics,
        state.controls,
        state.history,
        state.owner_key,
    )


def test_active_degenerate_alignment_refuses_the_history_bias() -> None:
    term, state = _aligned_term()
    energy, _, successful = term.bias(
        CollectiveVariableGeometry(jnp.asarray(COLLINEAR)), state
    )
    assert np.isfinite(float(energy))
    assert not bool(successful)
    evidence = _form_evidence(term, state, COLLINEAR)
    assert not bool(evidence.successful)
    assert int(evidence.status) & int(CollectiveVariableStatus.ALIGNMENT_GAP)
    assert evidence.minimum_relative_gap is not None
    assert float(evidence.minimum_relative_gap) <= 1.0e-10


def test_exact_zero_msd_and_padding_slots_remain_regular() -> None:
    term, state = _aligned_term()
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = TRIANGLE @ rotation.T + np.asarray([0.3, -0.2, 0.5])

    def energy(positions: jax.Array) -> jax.Array:
        return term.bias(CollectiveVariableGeometry(positions), state)[0]

    value, _, successful = term.bias(
        CollectiveVariableGeometry(jnp.asarray(moved)), state
    )
    assert bool(successful)
    # Zero aligned MSD: full height, an existing MSD derivative (zero force).
    np.testing.assert_allclose(float(value), 1.0, rtol=1e-12)
    force = np.asarray(jax.grad(energy)(jnp.asarray(moved)))
    assert np.all(np.isfinite(force))
    np.testing.assert_allclose(force, 0.0, atol=1e-10)
    evidence = _form_evidence(term, state, moved)
    # The padding (zero-centre) slots are degenerate but inactive.
    assert bool(evidence.successful) and int(evidence.status) == 0
    assert evidence.minimum_relative_gap is not None
    assert float(evidence.minimum_relative_gap) > 1.0e-10


def test_unselected_and_zero_ramp_kernels_do_not_refuse() -> None:
    # A state-owned kernel of owner 1 is not felt by owner 0.
    term, state = _aligned_term(
        ownership=BiasHistoryOwnership.STATE, owner_count=2, kernel_owner=1
    )
    energy, _, successful = term.bias(
        CollectiveVariableGeometry(jnp.asarray(COLLINEAR)), state
    )
    assert bool(successful) and float(energy) == 0.0
    # A kernel with zero ramp weight contributes nothing and is not admitted.
    shared, shared_state = _aligned_term()
    history = shared_state.history
    assert history is not None
    ramped = AtomisticBiasHistoryPlan(
        9,
        height=1.0,
        widths=0.5,
        capacity=4,
        ramp_depositions=1,
        kernel=HillKernel.ALIGNED_MSD,
    )
    fresh = eqx.tree_at(
        lambda value: (value.deposited_at, value.history_identity),
        history,
        (jnp.full_like(history.deposited_at, history.deposits), ramped.history_identity),
    )
    evaluation = ramped.evaluate(
        jnp.asarray(COLLINEAR.reshape((-1,))),
        shared.plan.variables.metrics,
        fresh,
        jnp.int32(0),
    )
    assert bool(evaluation.successful)
    assert int(evaluation.active_kernels) == 0
    assert float(evaluation.energy) == 0.0
