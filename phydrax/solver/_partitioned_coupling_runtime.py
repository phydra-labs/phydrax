#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import ArraySpace
from ..nonlinear import (
    FixedPointIteration,
    FixedPointProblem,
    implicit_root_result,
    NonlinearStatus,
    NonlinearSystemProblem,
)
from ._partitioned_coupling_graph import PreparedCoupling
from ._partitioned_coupling_types import (
    CouplingProvenance,
    CouplingState,
    CouplingStatus,
    CouplingWindow,
    CouplingWindowDiagnostics,
    CouplingWindowResult,
    ExplicitCouplingPolicy,
    ImplicitCouplingPolicy,
)
from ._partitioned_coupling_waveform import (
    coupling_signal_finite,
    coupling_signal_norm,
    flatten_coupling_signal,
    subtract_coupling_signals,
    transfer_coupling_signal,
    unflatten_coupling_signal,
    validate_coupling_signal,
)


class _CouplingEvaluation(StrictModule):
    candidate_states: tuple[Any, ...]
    exchange_values: tuple[Any, ...]
    residuals: tuple[Any, ...]
    participant_statuses: Array
    participant_residual_norms: Array
    participant_error_norms: Array
    participant_error_reference_norms: Array
    participant_error_orders: Array
    participant_error_reliable: Array
    participant_iterations: Array
    participant_work: Array
    successful: Array
    finite: Array


def _tree_stop(value: Any, /) -> Any:
    return jax.tree.map(jax.lax.stop_gradient, value)


def _target_port(prepared: PreparedCoupling, exchange_index: int, /):
    subsystem_index = prepared.exchange_target_subsystems[exchange_index]
    input_index = prepared.exchange_target_input_indices[exchange_index]
    return prepared.subsystems[subsystem_index].input_ports[input_index]


def _source_port(prepared: PreparedCoupling, exchange_index: int, /):
    subsystem_index = prepared.exchange_source_subsystems[exchange_index]
    output_index = prepared.exchange_source_output_indices[exchange_index]
    return prepared.subsystems[subsystem_index].output_ports[output_index]


def _apply_exchange(
    prepared: PreparedCoupling,
    exchange_index: int,
    output: Any,
    /,
) -> Any:
    exchange = prepared.exchanges[exchange_index]
    source_port = _source_port(prepared, exchange_index)
    target_port = _target_port(prepared, exchange_index)
    if exchange.transfer is None:
        action = lambda value: value
    elif exchange.use_adjoint:
        operator = exchange.transfer.adjoint_operator
        if operator is None:
            raise RuntimeError("Prepared adjoint coupling transfer is unavailable.")
        action = operator.mv
    else:
        action = exchange.transfer.operator.mv
    return transfer_coupling_signal(source_port, target_port, output, action)


def _participant_finite(result, /) -> Array:
    outputs_finite = jnp.asarray(True)
    for output in result.outputs:
        outputs_finite = outputs_finite & coupling_signal_finite(output)
    estimate = result.error_estimate
    return (
        tree_allfinite(result.candidate_state)
        & outputs_finite
        & jnp.isfinite(result.residual_norm)
        & jnp.isfinite(estimate.error_norm)
        & jnp.isfinite(estimate.reference_norm)
    )


def _evaluate_participant(
    prepared: PreparedCoupling,
    subsystem_index: int,
    window: CouplingWindow,
    start_state: CouplingState,
    input_values: tuple[Any, ...],
    args: Any,
    /,
):
    subsystem = prepared.subsystems[subsystem_index]
    result = subsystem.advance_window(
        window,
        start_state.participant_states[subsystem_index],
        input_values,
        args,
    )
    if len(result.outputs) != len(subsystem.output_ports):
        raise ValueError("Participant output cardinality changed after preparation.")
    candidate = jax.tree.map(
        lambda value, reference: jnp.asarray(value, dtype=reference.dtype),
        result.candidate_state,
        start_state.participant_states[subsystem_index],
    )
    outputs = tuple(
        validate_coupling_signal(port, value)
        for port, value in zip(subsystem.output_ports, result.outputs, strict=True)
    )
    return eqx.tree_at(
        lambda value: (value.candidate_state, value.outputs),
        result,
        (candidate, outputs),
    )


def _outgoing_exchange_indices(
    prepared: PreparedCoupling,
    subsystem_index: int,
    /,
) -> tuple[int, ...]:
    return tuple(
        exchange_index
        for exchange_index, source in enumerate(prepared.exchange_source_subsystems)
        if source == subsystem_index
    )


def _empty_evidence(prepared: PreparedCoupling, start_state: CouplingState, /):
    count = len(prepared.subsystems)
    dtype = start_state.time.dtype
    return (
        list(start_state.participant_states),
        [jnp.asarray(-1, dtype=jnp.int32) for _ in range(count)],
        [jnp.asarray(jnp.inf, dtype=dtype) for _ in range(count)],
        [jnp.asarray(0, dtype=jnp.int32) for _ in range(count)],
        [jnp.asarray(0, dtype=jnp.int32) for _ in range(count)],
        [jnp.asarray(False) for _ in range(count)],
        [jnp.asarray(False) for _ in range(count)],
        [None for _ in range(count)],
        [() for _ in range(count)],
    )


def _record_result(
    subsystem_index: int,
    result,
    candidate_states,
    statuses,
    residual_norms,
    iterations,
    work,
    error_estimates,
    successful,
    finite,
    outputs,
    /,
) -> None:
    candidate_states[subsystem_index] = result.candidate_state
    statuses[subsystem_index] = result.status
    residual_norms[subsystem_index] = result.residual_norm
    iterations[subsystem_index] = result.iterations
    work[subsystem_index] = result.work
    error_estimates[subsystem_index] = result.error_estimate
    successful[subsystem_index] = result.successful
    finite[subsystem_index] = _participant_finite(result)
    outputs[subsystem_index] = result.outputs


def _apply_subsystem_outputs(
    prepared: PreparedCoupling,
    subsystem_index: int,
    outputs: tuple[Any, ...],
    working_values: list[Any],
    /,
) -> None:
    for exchange_index in _outgoing_exchange_indices(prepared, subsystem_index):
        output_index = prepared.exchange_source_output_indices[exchange_index]
        working_values[exchange_index] = _apply_exchange(
            prepared, exchange_index, outputs[output_index]
        )


def _finalize_evaluation(
    prepared: PreparedCoupling,
    candidate_states: list[Any],
    working_values: list[Any],
    used_inputs: list[Any],
    statuses: list[Array],
    residual_norms: list[Array],
    iterations: list[Array],
    work: list[Array],
    error_estimates: list[Any],
    successful: list[Array],
    finite: list[Array],
    /,
) -> _CouplingEvaluation:
    residuals = tuple(
        subtract_coupling_signals(_target_port(prepared, exchange_index), used, mapped)
        for exchange_index, (used, mapped) in enumerate(
            zip(used_inputs, working_values, strict=True)
        )
    )
    exchange_finite = jnp.asarray(True)
    for value, residual in zip(working_values, residuals, strict=True):
        exchange_finite = (
            exchange_finite
            & coupling_signal_finite(value)
            & coupling_signal_finite(residual)
        )
    participant_success = jnp.all(jnp.stack(successful))
    participant_finite = jnp.all(jnp.stack(finite))
    return _CouplingEvaluation(
        candidate_states=tuple(candidate_states),
        exchange_values=tuple(working_values),
        residuals=residuals,
        participant_statuses=jnp.stack(statuses),
        participant_residual_norms=jnp.stack(residual_norms),
        participant_error_norms=jnp.stack(
            tuple(value.error_norm for value in error_estimates)
        ),
        participant_error_reference_norms=jnp.stack(
            tuple(value.reference_norm for value in error_estimates)
        ),
        participant_error_orders=jnp.stack(
            tuple(value.order for value in error_estimates)
        ),
        participant_error_reliable=jnp.stack(
            tuple(value.reliable for value in error_estimates)
        ),
        participant_iterations=jnp.stack(iterations),
        participant_work=jnp.stack(work),
        successful=participant_success,
        finite=participant_finite & exchange_finite,
    )


def _global_jacobi_evaluation(
    prepared: PreparedCoupling,
    window: CouplingWindow,
    start_state: CouplingState,
    exchange_values: tuple[Any, ...],
    args: Any,
    /,
) -> _CouplingEvaluation:
    working_values = list(exchange_values)
    used_inputs = list(exchange_values)
    (
        candidate_states,
        statuses,
        residual_norms,
        iterations,
        work,
        error_estimates,
        successful,
        finite,
        outputs,
    ) = _empty_evidence(prepared, start_state)
    for subsystem_index in range(len(prepared.subsystems)):
        input_values = tuple(
            exchange_values[exchange_index]
            for exchange_index in prepared.input_exchange_indices[subsystem_index]
        )
        result = _evaluate_participant(
            prepared, subsystem_index, window, start_state, input_values, args
        )
        _record_result(
            subsystem_index,
            result,
            candidate_states,
            statuses,
            residual_norms,
            iterations,
            work,
            error_estimates,
            successful,
            finite,
            outputs,
        )
    for subsystem_index, subsystem_outputs in enumerate(outputs):
        _apply_subsystem_outputs(
            prepared, subsystem_index, subsystem_outputs, working_values
        )
    return _finalize_evaluation(
        prepared,
        candidate_states,
        working_values,
        used_inputs,
        statuses,
        residual_norms,
        iterations,
        work,
        error_estimates,
        successful,
        finite,
    )


def _global_gauss_seidel_evaluation(
    prepared: PreparedCoupling,
    window: CouplingWindow,
    start_state: CouplingState,
    exchange_values: tuple[Any, ...],
    subsystem_order: tuple[str, ...],
    args: Any,
    /,
) -> _CouplingEvaluation:
    working_values = list(exchange_values)
    used_inputs = list(exchange_values)
    (
        candidate_states,
        statuses,
        residual_norms,
        iterations,
        work,
        error_estimates,
        successful,
        finite,
        outputs,
    ) = _empty_evidence(prepared, start_state)
    index_by_id = {
        subsystem.subsystem_id: index
        for index, subsystem in enumerate(prepared.subsystems)
    }
    for subsystem_id in subsystem_order:
        subsystem_index = index_by_id[subsystem_id]
        input_indices = prepared.input_exchange_indices[subsystem_index]
        input_values = tuple(working_values[index] for index in input_indices)
        for exchange_index, value in zip(input_indices, input_values, strict=True):
            used_inputs[exchange_index] = value
        result = _evaluate_participant(
            prepared, subsystem_index, window, start_state, input_values, args
        )
        _record_result(
            subsystem_index,
            result,
            candidate_states,
            statuses,
            residual_norms,
            iterations,
            work,
            error_estimates,
            successful,
            finite,
            outputs,
        )
        _apply_subsystem_outputs(
            prepared, subsystem_index, result.outputs, working_values
        )
    return _finalize_evaluation(
        prepared,
        candidate_states,
        working_values,
        used_inputs,
        statuses,
        residual_norms,
        iterations,
        work,
        error_estimates,
        successful,
        finite,
    )


def _stagewise_evaluation(
    prepared: PreparedCoupling,
    window: CouplingWindow,
    start_state: CouplingState,
    exchange_values: tuple[Any, ...],
    args: Any,
    /,
    *,
    gauss_seidel_order: tuple[str, ...] | None = None,
) -> _CouplingEvaluation:
    working_values = list(exchange_values)
    used_inputs = list(exchange_values)
    (
        candidate_states,
        statuses,
        residual_norms,
        iterations,
        work,
        error_estimates,
        successful,
        finite,
        outputs,
    ) = _empty_evidence(prepared, start_state)
    index_by_id = {
        subsystem.subsystem_id: index
        for index, subsystem in enumerate(prepared.subsystems)
    }
    for stage in prepared.stages:
        if stage.cyclic and gauss_seidel_order is not None:
            stage_members = set(stage.subsystem_indices)
            order = tuple(
                index_by_id[subsystem_id]
                for subsystem_id in gauss_seidel_order
                if index_by_id[subsystem_id] in stage_members
            )
            for subsystem_index in order:
                input_indices = prepared.input_exchange_indices[subsystem_index]
                input_values = tuple(working_values[index] for index in input_indices)
                for exchange_index, value in zip(
                    input_indices, input_values, strict=True
                ):
                    used_inputs[exchange_index] = value
                result = _evaluate_participant(
                    prepared,
                    subsystem_index,
                    window,
                    start_state,
                    input_values,
                    args,
                )
                _record_result(
                    subsystem_index,
                    result,
                    candidate_states,
                    statuses,
                    residual_norms,
                    iterations,
                    work,
                    error_estimates,
                    successful,
                    finite,
                    outputs,
                )
                _apply_subsystem_outputs(
                    prepared, subsystem_index, result.outputs, working_values
                )
            continue

        snapshot = tuple(working_values)
        for subsystem_index in stage.subsystem_indices:
            input_indices = prepared.input_exchange_indices[subsystem_index]
            input_values = tuple(snapshot[index] for index in input_indices)
            for exchange_index, value in zip(input_indices, input_values, strict=True):
                used_inputs[exchange_index] = value
            result = _evaluate_participant(
                prepared, subsystem_index, window, start_state, input_values, args
            )
            _record_result(
                subsystem_index,
                result,
                candidate_states,
                statuses,
                residual_norms,
                iterations,
                work,
                error_estimates,
                successful,
                finite,
                outputs,
            )
        for subsystem_index in stage.subsystem_indices:
            _apply_subsystem_outputs(
                prepared, subsystem_index, outputs[subsystem_index], working_values
            )
    return _finalize_evaluation(
        prepared,
        candidate_states,
        working_values,
        used_inputs,
        statuses,
        residual_norms,
        iterations,
        work,
        error_estimates,
        successful,
        finite,
    )


def _pack_interface(
    prepared: PreparedCoupling,
    exchange_values: tuple[Any, ...],
    /,
) -> Array:
    coordinates: list[Array] = []
    for exchange_index in prepared.implicit_exchange_indices:
        port = _target_port(prepared, exchange_index)
        flattened = flatten_coupling_signal(port, exchange_values[exchange_index])
        coordinates.append(
            jnp.asarray(flattened / port.reference_scale, dtype=prepared.coordinate_dtype)
        )
    if not coordinates:
        return jnp.zeros((0,), dtype=prepared.coordinate_dtype)
    return coordinates[0] if len(coordinates) == 1 else jnp.concatenate(coordinates)


def _unpack_interface(
    prepared: PreparedCoupling,
    coordinates: Array,
    base_values: tuple[Any, ...],
    /,
) -> tuple[Any, ...]:
    value = jnp.asarray(coordinates, dtype=prepared.coordinate_dtype)
    if value.shape != (prepared.report.resources.interface_size,):
        raise ValueError("Coupling interface coordinates have the wrong shape.")
    unpacked = list(base_values)
    for local_index, exchange_index in enumerate(prepared.implicit_exchange_indices):
        port = _target_port(prepared, exchange_index)
        offset = prepared.interface_offsets[local_index]
        size = prepared.interface_sizes[local_index]
        reference_dtype = flatten_coupling_signal(port, base_values[exchange_index]).dtype
        physical = value[offset : offset + size].astype(reference_dtype)
        unpacked[exchange_index] = unflatten_coupling_signal(
            port,
            physical * jnp.asarray(port.reference_scale, dtype=reference_dtype),
        )
    return tuple(unpacked)


def _pack_residual(
    prepared: PreparedCoupling,
    residuals: tuple[Any, ...],
    /,
) -> Array:
    coordinates: list[Array] = []
    for exchange_index in prepared.implicit_exchange_indices:
        port = _target_port(prepared, exchange_index)
        flattened = flatten_coupling_signal(port, residuals[exchange_index])
        safe = jnp.where(jnp.isfinite(flattened), flattened, jnp.zeros_like(flattened))
        coordinates.append(
            jnp.asarray(safe / port.reference_scale, dtype=prepared.coordinate_dtype)
        )
    return coordinates[0] if len(coordinates) == 1 else jnp.concatenate(coordinates)


def _exchange_diagnostics(
    prepared: PreparedCoupling,
    evaluation: _CouplingEvaluation,
    /,
):
    physical_norms: list[Array] = []
    normalized_norms: list[Array] = []
    thresholds: list[Array] = []
    certified: list[Array] = []
    tolerance_by_port = (
        {}
        if not isinstance(prepared.policy, ImplicitCouplingPolicy)
        else {value.port_id: value for value in prepared.policy.tolerances}
    )
    implicit_set = (
        set()
        if not isinstance(prepared.policy, ImplicitCouplingPolicy)
        else set(prepared.implicit_exchange_indices)
    )
    for exchange_index, residual in enumerate(evaluation.residuals):
        port = _target_port(prepared, exchange_index)
        physical = coupling_signal_norm(port, residual)
        normalized = jnp.linalg.norm(
            flatten_coupling_signal(port, residual) / port.reference_scale
        )
        if exchange_index in implicit_set:
            tolerance = tolerance_by_port[port.port_id]
            threshold = jnp.asarray(
                tolerance.absolute + tolerance.relative * port.reference_scale,
                dtype=physical.dtype,
            )
            accepted = physical <= threshold
        else:
            threshold = jnp.asarray(jnp.inf, dtype=physical.dtype)
            accepted = jnp.asarray(True)
        physical_norms.append(physical)
        normalized_norms.append(normalized)
        thresholds.append(threshold)
        certified.append(accepted)
    return (
        jnp.stack(physical_norms),
        jnp.stack(normalized_norms),
        jnp.stack(thresholds),
        jnp.stack(certified),
    )


def _accepted_state(
    successful: Array,
    candidate: CouplingState,
    original: CouplingState,
    /,
) -> CouplingState:
    participant_states = tuple(
        jax.tree.map(
            lambda candidate_value, original_value: jnp.where(
                successful, candidate_value, original_value
            ),
            candidate_value,
            original_value,
        )
        for candidate_value, original_value in zip(
            candidate.participant_states, original.participant_states, strict=True
        )
    )
    exchange_values = tuple(
        jax.tree.map(
            lambda candidate_value, original_value: jnp.where(
                successful, candidate_value, original_value
            ),
            candidate_value,
            original_value,
        )
        for candidate_value, original_value in zip(
            candidate.exchange_values, original.exchange_values, strict=True
        )
    )
    return CouplingState(
        participant_states,
        exchange_values,
        jnp.where(successful, candidate.time, original.time),
        jnp.where(successful, candidate.window_index, original.window_index),
        subsystem_ids=original.subsystem_ids,
        exchange_ids=original.exchange_ids,
    )


def _stop_state(state: CouplingState, /) -> CouplingState:
    return CouplingState(
        tuple(_tree_stop(value) for value in state.participant_states),
        tuple(_tree_stop(value) for value in state.exchange_values),
        jax.lax.stop_gradient(state.time),
        jax.lax.stop_gradient(state.window_index),
        subsystem_ids=state.subsystem_ids,
        exchange_ids=state.exchange_ids,
    )


def _status_from_nonlinear(
    nonlinear_status: Array,
    evaluation: _CouplingEvaluation,
    certified: Array,
    /,
) -> Array:
    exhausted = (
        (nonlinear_status == int(NonlinearStatus.MAXIMUM_STEPS_REACHED))
        | (nonlinear_status == int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED))
        | (nonlinear_status == int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED))
    )
    return jnp.where(
        ~evaluation.successful,
        int(CouplingStatus.PARTICIPANT_FAILURE),
        jnp.where(
            ~evaluation.finite,
            int(CouplingStatus.NONFINITE_EVALUATION),
            jnp.where(
                exhausted,
                int(CouplingStatus.WORK_EXHAUSTED),
                jnp.where(
                    nonlinear_status != int(NonlinearStatus.SUCCESS),
                    int(CouplingStatus.NONLINEAR_FAILURE),
                    jnp.where(
                        ~certified,
                        int(CouplingStatus.CERTIFICATION_FAILURE),
                        int(CouplingStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _window_result(
    prepared: PreparedCoupling,
    start_state: CouplingState,
    window: CouplingWindow,
    evaluation: _CouplingEvaluation,
    /,
    *,
    nonlinear_status: Array,
    coupling_iterations: Array,
    nonlinear_residual_evaluations: Array,
    implicit: bool,
) -> CouplingWindowResult:
    (
        physical_norms,
        normalized_norms,
        thresholds,
        exchange_certified,
    ) = _exchange_diagnostics(prepared, evaluation)
    certified = jnp.all(exchange_certified)
    if implicit:
        status = _status_from_nonlinear(nonlinear_status, evaluation, certified)
        successful = status == int(CouplingStatus.SUCCESS)
        converged = successful
    else:
        status = jnp.where(
            ~evaluation.successful,
            int(CouplingStatus.PARTICIPANT_FAILURE),
            jnp.where(
                ~evaluation.finite,
                int(CouplingStatus.NONFINITE_EVALUATION),
                int(CouplingStatus.SUCCESS),
            ),
        ).astype(jnp.int32)
        successful = status == int(CouplingStatus.SUCCESS)
        converged = jnp.asarray(False)
    candidate = CouplingState(
        evaluation.candidate_states,
        evaluation.exchange_values,
        window.end,
        start_state.window_index + 1,
        subsystem_ids=start_state.subsystem_ids,
        exchange_ids=start_state.exchange_ids,
    )
    accepted = _accepted_state(successful, candidate, start_state)
    participant_evaluations = jnp.full(
        (len(prepared.subsystems),),
        nonlinear_residual_evaluations + 1 if implicit else 1,
        dtype=jnp.int32,
    )
    transfer_applications = jnp.full(
        (len(prepared.exchanges),),
        nonlinear_residual_evaluations + 1 if implicit else 1,
        dtype=jnp.int32,
    )
    diagnostics = CouplingWindowDiagnostics(
        exchange_residual_norms=physical_norms,
        normalized_exchange_residual_norms=normalized_norms,
        exchange_thresholds=thresholds,
        exchange_certified=exchange_certified,
        participant_statuses=evaluation.participant_statuses,
        participant_residual_norms=evaluation.participant_residual_norms,
        participant_error_norms=evaluation.participant_error_norms,
        participant_error_reference_norms=(evaluation.participant_error_reference_norms),
        participant_error_orders=evaluation.participant_error_orders,
        participant_error_reliable=evaluation.participant_error_reliable,
        participant_iterations=evaluation.participant_iterations,
        participant_work=evaluation.participant_work,
        participant_evaluations=participant_evaluations,
        transfer_applications=transfer_applications,
        coupling_iterations=jnp.asarray(coupling_iterations, dtype=jnp.int32),
        nonlinear_residual_evaluations=jnp.asarray(
            nonlinear_residual_evaluations, dtype=jnp.int32
        ),
        counts_complete=prepared.report.resources.complete and not implicit,
    )
    method_id = (
        f"explicit-{prepared.policy.sweep.kind}"
        if isinstance(prepared.policy, ExplicitCouplingPolicy)
        else prepared.policy.method.method_id
    )
    provenance = CouplingProvenance(
        problem_id=prepared.problem_id,
        graph_id=prepared.graph_id,
        plan_id=prepared.plan_id,
        policy_id=prepared.policy.policy_id,
        method_id=method_id,
        differentiation_policy_id=prepared.differentiation.policy_id,
        numeric_version=prepared.numeric_version,
    )
    if prepared.differentiation.mode == "none":
        candidate = _stop_state(candidate)
        accepted = _stop_state(accepted)
    return CouplingWindowResult(
        candidate_state=candidate,
        accepted_state=accepted,
        successful=successful,
        converged=converged,
        status=status,
        nonlinear_status=jnp.asarray(nonlinear_status, dtype=jnp.int32),
        diagnostics=diagnostics,
        provenance=provenance,
    )


def advance_coupling_window(
    prepared: PreparedCoupling,
    state: CouplingState,
    window_size: Any,
    args: Any = None,
    /,
) -> CouplingWindowResult:
    """Advance one fixed coupling window and atomically commit only valid work."""

    if not isinstance(prepared, PreparedCoupling):
        raise TypeError("prepared must be PreparedCoupling.")
    if not isinstance(state, CouplingState):
        raise TypeError("state must be CouplingState.")
    if state.subsystem_ids != prepared.report.subsystem_ids:
        raise ValueError("Coupling state subsystem identity does not match its plan.")
    if state.exchange_ids != prepared.report.exchange_ids:
        raise ValueError("Coupling state exchange identity does not match its plan.")
    size = jnp.asarray(window_size, dtype=state.time.dtype)
    if size.shape != ():
        raise ValueError("Coupling window_size must be scalar.")
    size = eqx.error_if(
        size,
        ~jnp.isfinite(size) | (size <= 0.0),
        "Coupling window_size must be finite and positive.",
    )
    window = CouplingWindow(
        state.window_index,
        state.time,
        state.time + size,
    )
    policy = prepared.policy
    if isinstance(policy, ExplicitCouplingPolicy):
        if policy.sweep.kind == "jacobi":
            evaluation = _global_jacobi_evaluation(
                prepared, window, state, state.exchange_values, args
            )
        else:
            evaluation = _global_gauss_seidel_evaluation(
                prepared,
                window,
                state,
                state.exchange_values,
                policy.sweep.subsystem_order,
                args,
            )
        return _window_result(
            prepared,
            state,
            window,
            evaluation,
            nonlinear_status=jnp.asarray(-1, dtype=jnp.int32),
            coupling_iterations=jnp.asarray(1, dtype=jnp.int32),
            nonlinear_residual_evaluations=jnp.asarray(0, dtype=jnp.int32),
            implicit=False,
        )

    if not isinstance(policy, ImplicitCouplingPolicy):
        raise TypeError("Unsupported prepared coupling policy.")
    initial_coordinates = _pack_interface(prepared, state.exchange_values)

    if isinstance(policy.method, FixedPointIteration):
        sweep = policy.fixed_point_sweep
        if sweep is None:
            raise RuntimeError("Prepared fixed-point coupling sweep is missing.")

        def mapping(coordinates, runtime_args):
            current_values = _unpack_interface(
                prepared, coordinates, state.exchange_values
            )
            evaluation = _stagewise_evaluation(
                prepared,
                window,
                state,
                current_values,
                runtime_args,
                gauss_seidel_order=(
                    None if sweep.kind == "jacobi" else sweep.subsystem_order
                ),
            )
            return _pack_interface(prepared, evaluation.exchange_values)

        fixed_problem = FixedPointProblem(
            mapping,
            problem_id=f"{prepared.problem_id}/interface-fixed-point",
        )
        nonlinear_result = policy.method.solve(
            fixed_problem,
            initial_coordinates,
            termination=policy.termination,
            args=args,
        )
    else:
        coordinate_space = ArraySpace(
            (prepared.report.resources.interface_size,),
            dtype=prepared.coordinate_dtype,
            space_id=f"{prepared.plan_id}/interface-coordinates",
        )

        def residual(coordinates, runtime_args):
            current_values = _unpack_interface(
                prepared, coordinates, state.exchange_values
            )
            evaluation = _stagewise_evaluation(
                prepared, window, state, current_values, runtime_args
            )
            return _pack_residual(prepared, evaluation.residuals), evaluation

        problem = NonlinearSystemProblem(
            residual,
            state_space=coordinate_space,
            residual_space=coordinate_space,
            has_aux=True,
            validity=lambda coordinates, current_residual, evaluation, runtime_args: (
                evaluation.successful & evaluation.finite
            ),
            problem_id=f"{prepared.problem_id}/interface-root",
        )
        if prepared.differentiation.mode == "implicit":
            nonlinear_result = implicit_root_result(
                problem,
                initial_coordinates,
                method=policy.method,
                termination=policy.termination,
                derivative_policy=policy.derivative_policy,
                args=args,
            )
        else:
            nonlinear_result = policy.method.solve(
                problem,
                initial_coordinates,
                termination=policy.termination,
                args=args,
            )

    final_values = _unpack_interface(
        prepared, nonlinear_result.state, state.exchange_values
    )
    final_evaluation = _stagewise_evaluation(prepared, window, state, final_values, args)
    diagnostics = nonlinear_result.diagnostics
    return _window_result(
        prepared,
        state,
        window,
        final_evaluation,
        nonlinear_status=nonlinear_result.status,
        coupling_iterations=diagnostics.iterations,
        nonlinear_residual_evaluations=diagnostics.residual_evaluations,
        implicit=True,
    )


__all__ = ["advance_coupling_window"]
