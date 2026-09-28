#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
import phydrax.applications.two_phase_flow._bubbly_run as bubbly_run
from tests._support.assertions import assert_tree_equal


two_phase_api = phx.applications.two_phase_flow
P0 = 1.0e5


def _fixed_step(
    candidate: two_phase_api.TwoPhaseContinuationState,
    accepted: two_phase_api.TwoPhaseContinuationState,
    successful: bool,
) -> phx.solver.FixedStepResult:
    return phx.solver.FixedStepResult(
        candidate_state=candidate,
        accepted_state=accepted,
        successful=jnp.asarray(successful),
        residual=jnp.asarray(0.0, dtype=jnp.float64),
        iterations=jnp.asarray(0, dtype=jnp.int32),
        work=jnp.asarray(0, dtype=jnp.int32),
        transform_applied=jnp.asarray(False),
        transform_correction_norm=jnp.asarray(0.0, dtype=jnp.float64),
    )


def _transaction_case() -> tuple[
    two_phase_api.IncompressibleTwoPhaseVOFMethod,
    two_phase_api.TwoPhaseContinuationState,
    phx.solver.FixedStepResult,
]:
    cells = 8
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(cells, periodic=True),
            phx.discretization.UniformCellAxisSpec(cells, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=1000.0,
        gas_density=1.0,
        liquid_viscosity=0.0,
        gas_viscosity=0.0,
    )
    two_phase = two_phase_api.IncompressibleTwoPhaseVOFPlan(
        discretization, material, maximum_iterations=32
    ).prepare()
    identity = two_phase_api.BubbleComponentPlan(
        two_phase, component_capacity=4, maximum_rounds=32, pair_capacity=8
    )
    compartments = two_phase_api.BubbleCompartmentPlan(
        phx.bubble_dynamics.IsothermalIdealBubbleGasLaw(1.4),
        phx.bubble_dynamics.BubbleEnvironment(P0, 300.0),
        capacity=4,
        dimension=2,
    )
    plan = two_phase_api.BubblyFlowPlan(
        two_phase, identity, compartments=compartments, atmosphere_pressure=P0
    )
    alpha = jnp.ones((cells, cells), dtype=jnp.float64)
    bubbles = plan.initial_state(alpha, compartment_pressure=P0)
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase, bubbles=plan)
    source = method.initial_continuation(two_phase.initial_state(alpha), bubbles=bubbles)

    proposed_alpha = np.ones((cells, cells), dtype=np.float64)
    proposed_alpha[2:4, 2:4] = 0.0
    identity_proposal = identity.propose(bubbles.identity, jnp.asarray(proposed_alpha))
    proposed_bubbles = two_phase_api.BubblyFlowState(
        identity=identity_proposal.commit(bubbles.identity),
        event=identity_proposal.event,
        markers=None,
        compartments=bubbles.compartments,
        contacts=None,
        dilatation=bubbles.dilatation,
    )
    evidence = source.bubble_evidence
    if evidence is None:
        raise RuntimeError("Bubbly continuation did not initialize evidence.")
    mismatch_evidence = eqx.tree_at(
        lambda value: (
            value.topology,
            value.registry_mismatch,
            value.creation_pressure,
            value.host_transaction_required,
            value.derivative_available,
            value.successful,
        ),
        evidence,
        (
            identity_proposal.evidence,
            jnp.asarray(True),
            jnp.full((identity.component_capacity,), P0, dtype=jnp.float64),
            jnp.asarray(True),
            identity_proposal.evidence.derivative_available,
            jnp.asarray(False),
        ),
    )
    candidate = eqx.tree_at(
        lambda value: (value.bubbles, value.bubble_evidence),
        source,
        (proposed_bubbles, mismatch_evidence),
    )
    return method, source, _fixed_step(candidate, source, False)


def _epochs(
    state: two_phase_api.TwoPhaseContinuationState,
) -> tuple[int, int]:
    bubbles = state.bubbles
    if bubbles is None or bubbles.compartments is None:
        raise RuntimeError("Bubbly continuation lost its compartment registry.")
    return int(bubbles.identity.epoch), int(bubbles.compartments.epoch)


def test_pressure_failure_after_transaction_rolls_back_complete_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, source, mismatch = _transaction_case()
    attempts: list[two_phase_api.TwoPhaseContinuationState] = []

    def attempt(
        method_: two_phase_api.IncompressibleTwoPhaseVOFMethod,
        step_index: Array,
        time: Array,
        state: two_phase_api.TwoPhaseContinuationState,
        step_size: Array,
    ) -> phx.solver.FixedStepResult:
        del method_, step_index, time, step_size
        attempts.append(state)
        if len(attempts) == 1:
            return mismatch
        evidence = state.bubble_evidence
        if evidence is None:
            raise RuntimeError("Retry continuation lost its bubble evidence.")
        failed_evidence = eqx.tree_at(
            lambda value: (
                value.registry_mismatch,
                value.projection_status,
                value.successful,
            ),
            evidence,
            (
                jnp.asarray(False),
                jnp.asarray(
                    phx.solver.MACCompartmentProjectionStatus.PRESSURE_SOLVE_FAILED,
                    dtype=jnp.int32,
                ),
                jnp.asarray(False),
            ),
        )
        candidate = eqx.tree_at(
            lambda value: value.bubble_evidence, state, failed_evidence
        )
        return _fixed_step(candidate, state, False)

    monkeypatch.setattr(bubbly_run, "_attempt", attempt)
    result = two_phase_api.run_bubbly_flow(
        method, source, steps=1, step_size=1.0e-3, maximum_transactions=1
    )

    assert result.status is two_phase_api.BubblyFlowRunStatus.STEP_FAILED
    assert result.message == "step 0 failed: PRESSURE_SOLVE_FAILED"
    assert len(attempts) == 2
    assert _epochs(attempts[1]) == (0, 1)
    assert _epochs(result.continuation) == _epochs(source) == (0, 0)
    assert_tree_equal(result.continuation, source)
    assert_tree_equal(
        result.compartment_ledger, two_phase_api.BubbleCompartmentLedger.zeros()
    )


def test_transaction_budget_exhaustion_rolls_back_complete_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, source, mismatch = _transaction_case()
    attempts: list[two_phase_api.TwoPhaseContinuationState] = []

    def attempt(
        method_: two_phase_api.IncompressibleTwoPhaseVOFMethod,
        step_index: Array,
        time: Array,
        state: two_phase_api.TwoPhaseContinuationState,
        step_size: Array,
    ) -> phx.solver.FixedStepResult:
        del method_, step_index, time, step_size
        attempts.append(state)
        return mismatch

    monkeypatch.setattr(bubbly_run, "_attempt", attempt)
    result = two_phase_api.run_bubbly_flow(
        method, source, steps=1, step_size=1.0e-3, maximum_transactions=1
    )

    assert result.status is two_phase_api.BubblyFlowRunStatus.TRANSACTIONS_EXHAUSTED
    assert len(attempts) == 2
    assert _epochs(attempts[1]) == (0, 1)
    assert _epochs(result.continuation) == _epochs(source) == (0, 0)
    assert_tree_equal(result.continuation, source)
    assert_tree_equal(
        result.compartment_ledger, two_phase_api.BubbleCompartmentLedger.zeros()
    )


def test_accepted_retry_commits_transaction_and_deterministic_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, source, mismatch = _transaction_case()

    def run_once() -> two_phase_api.BubblyFlowRun:
        attempts = 0

        def attempt(
            method_: two_phase_api.IncompressibleTwoPhaseVOFMethod,
            step_index: Array,
            time: Array,
            state: two_phase_api.TwoPhaseContinuationState,
            step_size: Array,
        ) -> phx.solver.FixedStepResult:
            nonlocal attempts
            del method_, step_index, time, step_size
            attempts += 1
            if attempts == 1:
                return mismatch
            candidate: two_phase_api.TwoPhaseContinuationState = mismatch.candidate_state
            candidate_bubbles = candidate.bubbles
            state_bubbles = state.bubbles
            evidence = candidate.bubble_evidence
            if candidate_bubbles is None or state_bubbles is None or evidence is None:
                raise RuntimeError("Accepted retry lost bubbly transaction state.")
            accepted_bubbles = eqx.tree_at(
                lambda value: value.compartments,
                candidate_bubbles,
                state_bubbles.compartments,
            )
            accepted_evidence = eqx.tree_at(
                lambda value: (
                    value.registry_mismatch,
                    value.host_transaction_required,
                    value.successful,
                ),
                evidence,
                (jnp.asarray(False), jnp.asarray(False), jnp.asarray(True)),
            )
            accepted = eqx.tree_at(
                lambda value: (value.bubbles, value.bubble_evidence),
                candidate,
                (accepted_bubbles, accepted_evidence),
            )
            return _fixed_step(accepted, accepted, True)

        monkeypatch.setattr(bubbly_run, "_attempt", attempt)
        return two_phase_api.run_bubbly_flow(
            method, source, steps=1, step_size=1.0e-3, maximum_transactions=1
        )

    first = run_once()
    replay = run_once()

    assert first.status is two_phase_api.BubblyFlowRunStatus.COMPLETED
    assert _epochs(first.continuation) == (1, 1)
    assert len(first.journal.records) == 1
    assert first.journal.records[0].kind == "create"
    assert float(first.compartment_ledger.created_amount) > 0.0
    assert_tree_equal(first, replay)
