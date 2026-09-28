#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.solver._balance_law_composition import AdditiveIMEXTableau
from phydrax.solver._conservation_temporal import (
    ConservationIMEXMethod,
    ImplicitConservationStageResult,
    prepare_element_block_preconditioner,
)
from phydrax.solver._fem_multirate import (
    ConservativeLocalTimeStepPlan,
    DGMultirateTracePlan,
    TimeSlabFluxLedger,
)


def test_conservation_imex_commits_converged_implicit_stage() -> None:
    tableau = AdditiveIMEXTableau(
        jnp.asarray(((0.0,),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
    )

    def implicit_solver(provisional: Any, time: Any, coefficient: Any, args: Any) -> Any:
        del time, args
        state = provisional / (1.0 + 10.0 * coefficient)
        return ImplicitConservationStageResult(
            state,
            jnp.asarray(True),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0.0),
        )

    method = ConservationIMEXMethod(
        tableau,
        lambda time, state, args: -state,
        lambda time, state, args: -10.0 * state,
        implicit_solver,
        method_id="linear-imex",
    )
    # ty: ignore[invalid-argument-type]
    result = method.step(0.0, jnp.asarray((1.0,)), 0.1)
    assert result.successful
    np.testing.assert_allclose(result.accepted_state, (0.45,), atol=2.0e-12)
    assert result.implicit_iterations == 1


def test_conservation_imex_calls_custom_validator_and_retains_rejected_input() -> None:
    # ty: ignore[invalid-argument-type]
    tableau = AdditiveIMEXTableau([[0.0]], [[1.0]], [1.0], [1.0])

    def solve(provisional: Any, time: Any, coefficient: Any, args: Any) -> Any:
        del time, args
        return ImplicitConservationStageResult(
            provisional / (1.0 + coefficient),
            # ty: ignore[invalid-argument-type]
            True,
            # ty: ignore[invalid-argument-type]
            1,
            jnp.asarray(0.0),
        )

    method = ConservationIMEXMethod(
        tableau,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: -state,
        solve,
        validator=lambda candidate: jnp.all(candidate >= 0.75),
        method_id="validator-regression",
    )
    step = eqx.filter_jit(method.step)
    # ty: ignore[invalid-argument-type]
    accepted = step(0.0, jnp.asarray([1.0]), 0.1)
    # ty: ignore[invalid-argument-type]
    rejected = step(0.0, jnp.asarray([0.5]), 0.1)
    assert accepted.successful
    np.testing.assert_allclose(accepted.accepted_state, [1.0 / 1.1])
    assert not rejected.successful
    np.testing.assert_allclose(rejected.candidate_state, [0.5 / 1.1])
    np.testing.assert_array_equal(rejected.accepted_state, [0.5])
    assert rejected.implicit_iterations == 1
    with pytest.raises(ValueError, match="validator"):
        ConservationIMEXMethod(
            tableau,
            method.explicit_rhs,
            method.implicit_rhs,
            solve,
            # ty: ignore[invalid-argument-type]
            validator=False,
            method_id="invalid-validator",
        )


def test_zero_implicit_diagonal_and_zero_step_preserve_tableau_and_derivatives() -> None:
    tableau = AdditiveIMEXTableau(
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0], [1.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0], [0.5, 0.5]],
        # ty: ignore[invalid-argument-type]
        [0.5, 0.5],
        # ty: ignore[invalid-argument-type]
        [0.0, 1.0],
    )

    def solve(provisional: Any, time: Any, coefficient: Any, args: Any) -> Any:
        del time, args
        # A diagonal solver need not be defined for an explicit stage.
        return jnp.where(
            coefficient != 0.0, provisional / (1.0 + 2.0 * coefficient), jnp.nan
        )

    def solve_with_evidence(
        provisional: Any, time: Any, coefficient: Any, args: Any
    ) -> Any:
        return ImplicitConservationStageResult(
            solve(provisional, time, coefficient, args),
            coefficient != 0.0,
            # ty: ignore[invalid-argument-type]
            1,
            jnp.asarray(0.0),
        )

    method = ConservationIMEXMethod(
        tableau,
        lambda time, state, args: -state,
        lambda time, state, args: -2.0 * state,
        solve_with_evidence,
        method_id="explicit-first-imex",
    )
    # ty: ignore[invalid-argument-type]
    result = eqx.filter_jit(method.step)(0.0, jnp.asarray(1.0), 0.1)
    expected = 1.0 - 0.15 * (1.0 + 0.8 / 1.1)
    assert result.successful
    assert result.implicit_iterations == 1
    np.testing.assert_allclose(result.accepted_state, expected, atol=1e-12)

    def tableau_step(state: Any, step: Any) -> Any:
        return tableau.step(
            state,
            # ty: ignore[invalid-argument-type]
            0.0,
            step,
            lambda state, time, args: -state,
            solve,
            implicit_rhs=lambda state, time, args: -2.0 * state,
        )

    def method_step(state: Any, step: Any) -> Any:
        # ty: ignore[invalid-argument-type]
        return method.step(0.0, state, step).accepted_state

    for advance in (tableau_step, method_step):
        np.testing.assert_allclose(jax.jit(advance)(1.0, 0.1), expected, atol=1e-12)
        np.testing.assert_allclose(jax.grad(advance, argnums=0)(1.0, 0.1), expected)
        np.testing.assert_array_equal(jax.jit(advance)(2.0, 0.0), 2.0)
        np.testing.assert_allclose(jax.grad(advance, argnums=0)(2.0, 0.0), 1.0)
        np.testing.assert_allclose(jax.grad(advance, argnums=1)(2.0, 0.0), -6.0)
    # ty: ignore[invalid-argument-type]
    zero = method.step(0.0, jnp.asarray(2.0), 0.0)
    assert zero.successful
    assert zero.implicit_iterations == 0
    np.testing.assert_array_equal(zero.maximum_implicit_residual, 0.0)


def test_complex_imex_reports_real_residual_and_retains_failed_solve_evidence() -> None:
    # ty: ignore[invalid-argument-type]
    tableau = AdditiveIMEXTableau([[0.0]], [[1.0]], [1.0], [1.0])
    rate = 2.0 + 3.0j

    def solve(provisional: Any, time: Any, coefficient: Any, converged: Any) -> Any:
        del time
        state = provisional / (1.0 + rate * coefficient)
        state = state + jnp.where(converged, 0.0, 0.01j)
        residual = jnp.max(jnp.abs(state - provisional + coefficient * rate * state))
        # ty: ignore[invalid-argument-type]
        return ImplicitConservationStageResult(state, converged, 3, residual)

    method = ConservationIMEXMethod(
        tableau,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: -rate * state,
        solve,
        method_id="complex-modal-decay",
    )
    state = jnp.asarray([1.0 + 2.0j, -0.5j])
    step = eqx.filter_jit(method.step)
    # ty: ignore[invalid-argument-type]
    accepted = step(0.0, state, 0.1, jnp.asarray(True))
    # ty: ignore[invalid-argument-type]
    rejected = step(0.0, state, 0.1, jnp.asarray(False))
    assert accepted.successful
    np.testing.assert_allclose(accepted.accepted_state, state / (1.0 + 0.1 * rate))
    assert jnp.issubdtype(accepted.maximum_implicit_residual.dtype, jnp.floating)
    assert accepted.maximum_implicit_residual < 1e-12
    assert not rejected.successful
    assert rejected.implicit_iterations == 3
    assert rejected.maximum_implicit_residual > 1e-3
    np.testing.assert_array_equal(rejected.accepted_state, state)


def test_real_initial_imex_state_promotes_for_complex_evolution() -> None:
    # ty: ignore[invalid-argument-type]
    tableau = AdditiveIMEXTableau([[0.0]], [[1.0]], [1.0], [1.0])

    def solve(provisional: Any, time: Any, coefficient: Any, args: Any) -> Any:
        del time, args
        return provisional / (1.0 + 1j * coefficient)

    def solve_with_evidence(
        provisional: Any, time: Any, coefficient: Any, args: Any
    ) -> Any:
        state = solve(provisional, time, coefficient, args)
        residual = jnp.max(jnp.abs(state - provisional + 1j * coefficient * state))
        # ty: ignore[invalid-argument-type]
        return ImplicitConservationStageResult(state, True, 1, residual)

    method = ConservationIMEXMethod(
        tableau,
        lambda time, state, args: jnp.zeros_like(state),
        lambda time, state, args: -1j * state,
        solve_with_evidence,
        method_id="real-initial-complex-evolution",
    )

    def tableau_step(state: Any, step: Any) -> Any:
        return tableau.step(
            state,
            # ty: ignore[invalid-argument-type]
            0.0,
            step,
            lambda state, time, args: jnp.zeros_like(state),
            solve,
            implicit_rhs=lambda state, time, args: -1j * state,
        )

    def method_step(state: Any, step: Any) -> Any:
        # ty: ignore[invalid-argument-type]
        return method.step(0.0, state, step).accepted_state

    initial = jnp.asarray([1.0])
    for advance in (tableau_step, method_step):
        result = jax.jit(advance)(initial, 0.1)
        assert jnp.issubdtype(result.dtype, jnp.complexfloating)
        np.testing.assert_allclose(result, [1.0 / (1.0 + 0.1j)], atol=1e-12)
        imaginary_response = lambda step: jnp.imag(advance(initial, step)[0])
        np.testing.assert_allclose(
            jax.grad(imaginary_response)(0.1),
            -(1.0 - 0.1**2) / (1.0 + 0.1**2) ** 2,
        )
        np.testing.assert_array_equal(jax.jit(advance)(initial, 0.0), initial)
        np.testing.assert_allclose(jax.grad(imaginary_response)(0.0), -1.0)


def _forward_backward_parts() -> AdditiveIMEXTableau:
    """Explicit stage, then implicit part 0, then implicit part 1 (Lie order)."""
    return AdditiveIMEXTableau(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        np.asarray([0.0, 1.0, 1.0]),
        np.asarray([0.0, 1.0, 1.0]),
        explicit_weights=np.asarray([1.0, 0.0, 0.0]),
        implicit_parts=(None, 0, 1),
    )


def _held_first_part(provisional: Any, time: Any, coefficient: Any, args: Any) -> Any:
    """Part 0: y' = -a y on free components; component 0 is held at zero."""
    del time
    state = (provisional / (1.0 + args["a"] * coefficient)).at[0].set(0.0)
    return ImplicitConservationStageResult(
        state,
        jnp.asarray(True),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(1e-14),
        jnp.asarray(7, dtype=jnp.int32),
        {"coefficient": coefficient, "scale": jnp.max(state)},
    )


def _second_part(provisional: Any, time: Any, coefficient: Any, args: Any) -> Any:
    """Part 1: y' = -b y; its convergence flag comes from ``args``."""
    del time
    return ImplicitConservationStageResult(
        provisional / (1.0 + args["b"] * coefficient),
        args["converged"],
        jnp.asarray(3, dtype=jnp.int32),
        jnp.asarray(2e-14),
        jnp.asarray(4, dtype=jnp.int32),
    )


def _block_method() -> ConservationIMEXMethod:
    return ConservationIMEXMethod(
        _forward_backward_parts(),
        lambda time, state, args: args["e"] * state,
        (
            lambda time, state, args: (-args["a"] * state).at[0].set(0.0),
            lambda time, state, args: -args["b"] * state,
        ),
        (_held_first_part, _second_part),
        method_id="forward-backward-block-sequential",
    )


def test_block_sequential_parts_give_the_forward_backward_lie_product() -> None:
    args = {
        "e": jnp.asarray(0.5),
        "a": jnp.asarray(3.0),
        "b": jnp.asarray(5.0),
        "converged": jnp.asarray(True),
    }
    state = jnp.asarray([2.0, 1.0, -0.5])
    step = jnp.asarray(0.1)
    result = eqx.filter_jit(_block_method().step)(jnp.asarray(0.0), state, step, args)
    expected = state * (1.0 + 0.1 * 0.5) / (1.0 + 0.1 * 3.0) / (1.0 + 0.1 * 5.0)
    assert result.successful
    np.testing.assert_allclose(result.accepted_state[1:], expected[1:], rtol=1e-14)
    # The held component is committed exactly: stage 3 starts from the stage-2
    # value and the stiffly accurate result is the stage-3 value.
    assert float(result.accepted_state[0]) == 0.0
    np.testing.assert_array_equal(result.stage_successful, [True, True, True])
    np.testing.assert_array_equal(result.stage_iterations, [0, 2, 3])
    np.testing.assert_array_equal(result.stage_status, [0, 7, 4])
    assert result.implicit_iterations == 5
    np.testing.assert_allclose(result.maximum_implicit_residual, 2e-14)
    assert result.stage_evidence[0] is None
    np.testing.assert_allclose(result.stage_evidence[1]["coefficient"], step)
    assert result.stage_evidence[2] is None

    tableau_result = _forward_backward_parts().step(
        state,
        jnp.asarray(0.0),
        step,
        lambda state, time, args: args["e"] * state,
        (
            lambda provisional, time, coefficient, args: (
                _held_first_part(provisional, time, coefficient, args).state
            ),
            lambda provisional, time, coefficient, args: (
                _second_part(provisional, time, coefficient, args).state
            ),
        ),
        args,
        implicit_rhs=(
            lambda state, time, args: (-args["a"] * state).at[0].set(0.0),
            lambda state, time, args: -args["b"] * state,
        ),
    )
    np.testing.assert_allclose(tableau_result, result.accepted_state, rtol=1e-15)
    assert float(tableau_result[0]) == 0.0


def test_failed_implicit_part_rejects_the_step_and_keeps_stage_evidence() -> None:
    args = {
        "e": jnp.asarray(0.5),
        "a": jnp.asarray(3.0),
        "b": jnp.asarray(5.0),
        "converged": jnp.asarray(False),
    }
    state = jnp.asarray([2.0, 1.0, -0.5])
    result = _block_method().step(jnp.asarray(0.0), state, jnp.asarray(0.1), args)
    assert not result.successful
    np.testing.assert_array_equal(result.accepted_state, state)
    np.testing.assert_array_equal(result.stage_successful, [True, True, False])
    np.testing.assert_array_equal(result.stage_status, [0, 7, 4])
    zero = _block_method().step(
        jnp.asarray(0.0), state, jnp.asarray(0.0), {**args, "converged": True}
    )
    assert zero.successful
    np.testing.assert_array_equal(zero.accepted_state, state)
    np.testing.assert_array_equal(zero.stage_iterations, [0, 0, 0])
    np.testing.assert_array_equal(zero.stage_status, [0, 0, 0])
    # A vanished diagonal step zero-fills the solver's evidence structure.
    np.testing.assert_array_equal(zero.stage_evidence[1]["scale"], 0.0)


def test_partitioned_tableau_and_callbacks_fail_closed() -> None:
    explicit = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="explicit-only"):
        AdditiveIMEXTableau(
            explicit,
            np.asarray([[0.0, 0.0], [0.5, 1.0]]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            explicit_weights=np.asarray([1.0, 0.0]),
            implicit_parts=(None, 0),
        )
    with pytest.raises(ValueError, match="consecutively"):
        AdditiveIMEXTableau(
            explicit,
            np.asarray([[0.0, 0.0], [0.0, 1.0]]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            explicit_weights=np.asarray([1.0, 0.0]),
            implicit_parts=(None, 1),
        )
    with pytest.raises(ValueError, match="every implicit part"):
        AdditiveIMEXTableau(
            explicit,
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([0.5, 0.5]),
            np.asarray([0.0, 1.0]),
            explicit_weights=np.asarray([1.0, 0.0]),
            implicit_parts=(0, 1),
        )
    with pytest.raises(TypeError, match="one callable per implicit part"):
        ConservationIMEXMethod(
            _forward_backward_parts(),
            lambda time, state, args: state,
            lambda time, state, args: state,
            (_held_first_part, _second_part),
            method_id="missing-part-rate",
        )


def test_conservation_temporal_scenario_1() -> None:
    state = jnp.asarray(((1.0,), (2.0,)))
    preconditioner = prepare_element_block_preconditioner(
        state,
        (jnp.asarray((0,)), jnp.asarray((1,))),
        lambda time, value, args: -2.0 * value,
        time=0.0,
        step_coefficient=0.1,
    )
    residual = jnp.asarray(((1.2,), (2.4,)))
    np.testing.assert_allclose(preconditioner.apply(residual), ((1.0,), (2.0,)))
    trace_plan = DGMultirateTracePlan(jnp.asarray(((0, 1),)), history_depth=2)
    plan = ConservativeLocalTimeStepPlan(
        jnp.asarray((0, 1), dtype=jnp.int32), 0.2, trace_plan
    )
    np.testing.assert_allclose(plan.cell_step_sizes(), (0.2, 0.1))
    np.testing.assert_array_equal(plan.active_cells(0), (True, True))
    np.testing.assert_array_equal(plan.active_cells(1), (False, True))
    ledger = TimeSlabFluxLedger.zeros(1, (1,), 0.2, ledger_id="shared-interface")
    ledger = ledger.add_substep(jnp.asarray(((3.0,),)), 0.1)
    ledger = ledger.add_substep(jnp.asarray(((5.0,),)), 0.1)
    assert ledger.complete
    contribution = ledger.equal_opposite_contributions()
    np.testing.assert_allclose(contribution.plus, ((0.8,),))
    np.testing.assert_allclose(contribution.minus, ((-0.8,),))
    np.testing.assert_allclose(contribution.conservation_defect, 0.0)
