#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import ArraySpace
from ..nonlinear import (
    AbstractNonlinearMethod,
    FixedPointIteration,
    FixedPointProblem,
    implicit_root_result,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._partitioned_coupling_graph import PreparedCoupling
from ._partitioned_coupling_types import (
    CouplingExchange,
    CouplingPort,
    CouplingProvenance,
    CouplingState,
    CouplingStatus,
    CouplingSubsystemResult,
    CouplingWindow,
    CouplingWindowDiagnostics,
    CouplingWindowErrorEstimate,
    CouplingWindowResult,
    ExplicitCouplingPolicy,
    ImplicitCouplingPolicy,
)
from ._partitioned_coupling_waveform import (
    coupling_signal_finite,
    coupling_signal_norm,
    flatten_coupling_signal,
    integrate_coupling_waveform,
    subtract_coupling_signals,
    transfer_coupling_signal,
    unflatten_coupling_signal,
    validate_coupling_signal,
)


_ParticipantEvidence: TypeAlias = tuple[
    list[Any],
    list[Array],
    list[Array],
    list[Array],
    list[Array],
    list[CouplingWindowErrorEstimate | None],
    list[Array],
    list[Array],
    list[tuple[Any, ...]],
    list[Any],
]


class _CouplingEvaluation(StrictModule):
    candidate_states: tuple[Any, ...]
    exchange_values: tuple[Any, ...]
    residuals: tuple[Any, ...]
    source_values: tuple[Any, ...]
    used_inputs: tuple[Any, ...]
    participant_statuses: Array
    participant_residual_norms: Array
    participant_error_norms: Array
    participant_error_reference_norms: Array
    participant_error_orders: Array
    participant_error_reliable: Array
    participant_iterations: Array
    participant_work: Array
    participant_evidence: tuple[Any, ...]
    successful: Array
    finite: Array


class _ParticipantWork(StrictModule):
    """Per-participant work and iterations spent by interface evaluations."""

    work: Array
    iterations: Array


def _tree_stop(value: Any, /) -> Any:
    return jax.tree.map(jax.lax.stop_gradient, value)


def _target_port(prepared: PreparedCoupling, exchange_index: int, /) -> CouplingPort:
    subsystem_index = prepared.exchange_target_subsystems[exchange_index]
    input_index = prepared.exchange_target_input_indices[exchange_index]
    return prepared.subsystems[subsystem_index].input_ports[input_index]


def _source_port(prepared: PreparedCoupling, exchange_index: int, /) -> CouplingPort:
    subsystem_index = prepared.exchange_source_subsystems[exchange_index]
    output_index = prepared.exchange_source_output_indices[exchange_index]
    return prepared.subsystems[subsystem_index].output_ports[output_index]


def _integrated(exchange: CouplingExchange, /) -> bool:
    return exchange.temporal is not None and exchange.temporal.kind == "integrate"


def _time_scale(prepared: PreparedCoupling, exchange: CouplingExchange, /) -> float:
    """Reference scale of the coupling clock unit a waveform integration multiplies."""
    if not _integrated(exchange):
        return 1.0
    time_unit = prepared.time_unit
    if time_unit is None:
        raise RuntimeError("Prepared waveform integration lacks its clock time unit.")
    return float(time_unit.scale_to_reference)


def _unit_factor(
    prepared: PreparedCoupling,
    exchange: CouplingExchange,
    source: CouplingPort,
    target: CouplingPort,
    /,
) -> float | None:
    """Exact reference-scale ratio applied after the spatial transfer.

    Preparation proved one physical quantity with matching inventory dimensions;
    any storage change between density and extensive coordinates is carried by
    the certified transfer, so only the quantity scales (and a waveform
    integration's clock unit) remain to convert.
    """
    source_quantity = source.quantity
    target_quantity = target.quantity
    # Preparation requires physical descriptors at both ends of a typed exchange.
    if source_quantity is None or target_quantity is None:
        return None
    return float(
        source_quantity.unit.scale_to_reference / target_quantity.unit.scale_to_reference
    ) * _time_scale(prepared, exchange)


def _apply_exchange(
    prepared: PreparedCoupling,
    exchange_index: int,
    output: Any,
    window: CouplingWindow,
    /,
) -> Any:
    exchange = prepared.exchanges[exchange_index]
    source_port = _source_port(prepared, exchange_index)
    target_port = _target_port(prepared, exchange_index)
    if exchange.transfer is None:
        action = lambda value: value
    elif exchange.use_adjoint:
        operator = exchange.transfer.hilbert_adjoint_operator
        if operator is None:
            raise RuntimeError("Prepared adjoint coupling transfer is unavailable.")
        action = operator.mv
    else:
        action = exchange.transfer.primal_operator.mv
    factor = _unit_factor(prepared, exchange, source_port, target_port)
    if factor is not None:
        spatial_action = action
        action = lambda value: jax.tree.map(
            lambda leaf: factor * leaf, spatial_action(value)
        )
    return transfer_coupling_signal(
        source_port, target_port, exchange.temporal, output, action, window.size
    )


def _participant_finite(result: CouplingSubsystemResult, /) -> Array:
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
) -> CouplingSubsystemResult:
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


def _empty_evidence(
    prepared: PreparedCoupling, start_state: CouplingState, /
) -> _ParticipantEvidence:
    count = len(prepared.subsystems)
    dtype = start_state.time.dtype
    return (
        list(start_state.participant_states),
        [jnp.asarray(-1, dtype=jnp.int32) for _ in range(count)],
        [jnp.asarray(jnp.inf, dtype=dtype) for _ in range(count)],
        [jnp.asarray(0, dtype=jnp.int32) for _ in range(count)],
        [jnp.asarray(0, dtype=jnp.int32) for _ in range(count)],
        [None for _ in range(count)],
        [jnp.asarray(False) for _ in range(count)],
        [jnp.asarray(False) for _ in range(count)],
        [() for _ in range(count)],
        [None for _ in range(count)],
    )


def _record_result(
    subsystem_index: int,
    result: CouplingSubsystemResult,
    candidate_states: list[Any],
    statuses: list[Array],
    residual_norms: list[Array],
    iterations: list[Array],
    work: list[Array],
    error_estimates: list[CouplingWindowErrorEstimate | None],
    successful: list[Array],
    finite: list[Array],
    outputs: list[tuple[Any, ...]],
    evidence: list[Any],
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
    evidence[subsystem_index] = result.evidence


def _apply_subsystem_outputs(
    prepared: PreparedCoupling,
    subsystem_index: int,
    outputs: tuple[Any, ...],
    working_values: list[Any],
    window: CouplingWindow,
    /,
) -> None:
    for exchange_index in _outgoing_exchange_indices(prepared, subsystem_index):
        output_index = prepared.exchange_source_output_indices[exchange_index]
        working_values[exchange_index] = _apply_exchange(
            prepared, exchange_index, outputs[output_index], window
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
    outputs: list[tuple[Any, ...]],
    evidence: list[Any],
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
        source_values=tuple(
            outputs[subsystem_index][output_index]
            for subsystem_index, output_index in zip(
                prepared.exchange_source_subsystems,
                prepared.exchange_source_output_indices,
                strict=True,
            )
        ),
        used_inputs=tuple(used_inputs),
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
        participant_evidence=tuple(evidence),
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
        evidence,
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
            evidence,
        )
    for subsystem_index, subsystem_outputs in enumerate(outputs):
        _apply_subsystem_outputs(
            prepared, subsystem_index, subsystem_outputs, working_values, window
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
        outputs,
        evidence,
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
        evidence,
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
            evidence,
        )
        _apply_subsystem_outputs(
            prepared, subsystem_index, result.outputs, working_values, window
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
        outputs,
        evidence,
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
        evidence,
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
                    evidence,
                )
                _apply_subsystem_outputs(
                    prepared, subsystem_index, result.outputs, working_values, window
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
                evidence,
            )
        for subsystem_index in stage.subsystem_indices:
            _apply_subsystem_outputs(
                prepared,
                subsystem_index,
                outputs[subsystem_index],
                working_values,
                window,
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
        outputs,
        evidence,
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
) -> tuple[Array, Array, Array, Array]:
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
        cumulative_exchange_budget=jnp.where(
            successful,
            candidate.cumulative_exchange_budget,
            original.cumulative_exchange_budget,
        ),
        budget_row_ids=original.budget_row_ids,
        graph_id=original.graph_id,
    )


def _stop_state(state: CouplingState, /) -> CouplingState:
    return CouplingState(
        tuple(_tree_stop(value) for value in state.participant_states),
        tuple(_tree_stop(value) for value in state.exchange_values),
        jax.lax.stop_gradient(state.time),
        jax.lax.stop_gradient(state.window_index),
        subsystem_ids=state.subsystem_ids,
        exchange_ids=state.exchange_ids,
        cumulative_exchange_budget=jax.lax.stop_gradient(
            state.cumulative_exchange_budget
        ),
        budget_row_ids=state.budget_row_ids,
        graph_id=state.graph_id,
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


def _amount_scale(port: CouplingPort, /) -> float:
    """Reference scale of `quantity × measurement` inventories at one port."""
    quantity = port.quantity
    measurement = port.measurement
    # Whole-window targets and their budgeted sources are typed and measured.
    if quantity is None or measurement is None:
        raise RuntimeError("A budgeted exchange lacks its prepared inventory semantics.")
    return float(quantity.unit.scale_to_reference * measurement.unit.scale_to_reference)


def _proposed_amount(
    prepared: PreparedCoupling,
    exchange_index: int,
    evaluation: _CouplingEvaluation,
    window: CouplingWindow,
    /,
) -> Any:
    """Source amount spent by one budgeted exchange over the complete window."""
    exchange = prepared.exchanges[exchange_index]
    value = evaluation.source_values[exchange_index]
    if not _integrated(exchange):
        return value
    rate = integrate_coupling_waveform(_source_port(prepared, exchange_index), value)
    return jax.tree.map(lambda leaf: window.size.astype(leaf.dtype) * leaf, rate)


def _exchange_budget_rows(
    prepared: PreparedCoupling,
    exchange_index: int,
    evaluation: _CouplingEvaluation,
    window: CouplingWindow,
    dtype: Any,
    /,
) -> tuple[Array, Array]:
    """Debit/credit rows per inventory component and their certification."""
    exchange = prepared.exchanges[exchange_index]
    source = _source_port(prepared, exchange_index)
    target = _target_port(prepared, exchange_index)
    source_measurement = source.measurement
    target_measurement = target.measurement
    if source_measurement is None or target_measurement is None:
        raise RuntimeError("A budgeted exchange lacks its prepared measurements.")
    amount = _proposed_amount(prepared, exchange_index, evaluation, window)
    received = evaluation.used_inputs[exchange_index]
    debit_scale = _amount_scale(source) * _time_scale(prepared, exchange)
    debit = -source_measurement.inventory(amount) * debit_scale
    credit = target_measurement.inventory(received) * _amount_scale(target)
    rows = jnp.stack((debit, credit), axis=-1).astype(dtype)
    # Rounding is bounded relative to the booked amounts, floored by the inventory
    # of a signal at the ports' declared reference scales, never by one SI unit.
    source_floor = jnp.asarray(source_measurement.covector_norms, dtype=dtype) * (
        source.reference_scale * debit_scale
    )
    if _integrated(exchange):
        source_floor = source_floor * window.size.astype(dtype)
    target_floor = jnp.asarray(target_measurement.covector_norms, dtype=dtype) * (
        target.reference_scale * _amount_scale(target)
    )
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(rows[:, 0]), jnp.abs(rows[:, 1])),
        jnp.maximum(source_floor, target_floor),
    )
    tolerance = 64 * jnp.finfo(rows.dtype).eps * scale
    # Certification uses the actual consumed proposal, not independently rounded
    # participant diagnostics or a forced equal-and-opposite ledger.
    received_ = target.space.flatten(received)
    mapped = target.space.flatten(evaluation.exchange_values[exchange_index])
    local_scale = jnp.maximum(jnp.abs(received_), jnp.abs(mapped))
    local_tolerance = (
        64
        * jnp.finfo(received_.dtype).eps
        * jnp.maximum(local_scale, target.reference_scale)
    )
    certified = (
        jnp.all(jnp.isfinite(rows))
        & jnp.all(jnp.abs(rows[:, 0] + rows[:, 1]) <= tolerance)
        & jnp.all(jnp.abs(received_ - mapped) <= local_tolerance)
    )
    return rows, certified


def _physical_window_budget(
    prepared: PreparedCoupling,
    evaluation: _CouplingEvaluation,
    window: CouplingWindow,
    dtype: Any,
    /,
) -> tuple[Array, Array]:
    rows: list[Array] = []
    certified = jnp.asarray(True)
    for index in range(len(prepared.exchanges)):
        if _target_port(prepared, index).temporal_kind != "interval_integral":
            rows.append(jnp.zeros((1, 2), dtype=dtype))
            continue
        exchange_rows, exchange_certified = _exchange_budget_rows(
            prepared, index, evaluation, window, dtype
        )
        rows.append(exchange_rows)
        certified = certified & exchange_certified
    return jnp.concatenate(rows, axis=0), certified


def coupling_counts_complete(prepared: PreparedCoupling, /) -> bool:
    """Whether window evidence counts the work of every executed evaluation.

    Explicit sweeps evaluate each participant once, and fixed-point interface
    iterations account every iterate through the problem's evaluation work.
    General root methods do not report per-evaluation work, so their windows
    report only the final evaluation and stay incomplete.
    """
    policy = prepared.policy
    if isinstance(policy, ExplicitCouplingPolicy):
        exact = True
    elif isinstance(policy, ImplicitCouplingPolicy):
        exact = isinstance(policy.method, FixedPointIteration)
    else:
        raise TypeError("Unsupported prepared coupling policy.")
    return prepared.report.resources.complete and exact


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
    iterate_work: _ParticipantWork | None = None,
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
    proposed_budget, budget_certified = _physical_window_budget(
        prepared,
        evaluation,
        window,
        start_state.cumulative_exchange_budget.dtype,
    )
    successful = successful & budget_certified
    status = jnp.where(
        (status == int(CouplingStatus.SUCCESS)) & ~budget_certified,
        int(CouplingStatus.CERTIFICATION_FAILURE),
        status,
    ).astype(jnp.int32)
    converged = converged & successful
    candidate = CouplingState(
        evaluation.candidate_states,
        evaluation.exchange_values,
        window.end,
        start_state.window_index + 1,
        subsystem_ids=start_state.subsystem_ids,
        exchange_ids=start_state.exchange_ids,
        cumulative_exchange_budget=start_state.cumulative_exchange_budget
        + proposed_budget,
        budget_row_ids=start_state.budget_row_ids,
        graph_id=start_state.graph_id,
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
    participant_work = evaluation.participant_work
    participant_iterations = evaluation.participant_iterations
    if iterate_work is not None:
        participant_work = participant_work + iterate_work.work
        participant_iterations = participant_iterations + iterate_work.iterations
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
        participant_iterations=participant_iterations,
        participant_work=participant_work,
        participant_evaluations=participant_evaluations,
        transfer_applications=transfer_applications,
        coupling_iterations=jnp.asarray(coupling_iterations, dtype=jnp.int32),
        nonlinear_residual_evaluations=jnp.asarray(
            nonlinear_residual_evaluations, dtype=jnp.int32
        ),
        counts_complete=coupling_counts_complete(prepared),
    )
    policy = prepared.policy
    if isinstance(policy, ExplicitCouplingPolicy):
        method_id = f"explicit-{policy.sweep.kind}"
    else:
        # Preparation admits only explicit and implicit coupling policies.
        if not (isinstance(policy, ImplicitCouplingPolicy)):
            raise RuntimeError(
                "Internal invariant failed: isinstance(policy, ImplicitCouplingPolicy)."
            )
        method_id = policy.method.method_id
    provenance = CouplingProvenance(
        problem_id=prepared.problem_id,
        graph_id=prepared.graph_id,
        plan_id=prepared.plan_id,
        policy_id=prepared.policy.policy_id,
        method_id=method_id,
        differentiation_policy_id=prepared.differentiation.policy_id,
        numeric_version=prepared.numeric_version,
    )
    participant_evidence = evaluation.participant_evidence
    if prepared.differentiation.mode == "none":
        candidate = _stop_state(candidate)
        accepted = _stop_state(accepted)
        participant_evidence = tuple(_tree_stop(value) for value in participant_evidence)
    return CouplingWindowResult(
        candidate_state=candidate,
        accepted_state=accepted,
        successful=successful,
        converged=converged,
        status=status,
        nonlinear_status=jnp.asarray(nonlinear_status, dtype=jnp.int32),
        diagnostics=diagnostics,
        provenance=provenance,
        proposed_exchange_budget=proposed_budget,
        accepted_exchange_budget=jnp.where(successful, proposed_budget, 0.0),
        participant_evidence=participant_evidence,
    )


def _checked_window(
    prepared: PreparedCoupling, state: CouplingState, window_size: Any, /
) -> CouplingWindow:
    """Refuse host plans, then bind the next native window."""
    if not isinstance(prepared, PreparedCoupling):
        raise TypeError("prepared must be PreparedCoupling.")
    if not prepared.report.jit_eligible:
        raise ValueError(
            "This coupling plan contains host participants; advance it only with "
            "advance_host_coupling_window."
        )
    return _bind_window(prepared, state, window_size)


def _require_state_identity(prepared: PreparedCoupling, state: CouplingState, /) -> None:
    """Refuse a state whose participant, exchange, ledger, or graph identity differs."""
    if not isinstance(state, CouplingState):
        raise TypeError("state must be CouplingState.")
    if state.subsystem_ids != prepared.report.subsystem_ids:
        raise ValueError("Coupling state subsystem identity does not match its plan.")
    if state.exchange_ids != prepared.report.exchange_ids:
        raise ValueError("Coupling state exchange identity does not match its plan.")
    if state.budget_row_ids != prepared.reference_state.budget_row_ids:
        raise ValueError("Coupling state ledger rows do not match its plan.")
    if state.graph_id != prepared.graph_id:
        raise ValueError(
            "Coupling state physical graph identity does not match its plan."
        )


def _bind_window(
    prepared: PreparedCoupling, state: CouplingState, window_size: Any, /
) -> CouplingWindow:
    """Validate the state's identity against its plan and bind the next window."""
    _require_state_identity(prepared, state)
    size = jnp.asarray(window_size, dtype=state.time.dtype)
    if size.shape != ():
        raise ValueError("Coupling window_size must be scalar.")
    size = eqx.error_if(
        size,
        ~jnp.isfinite(size) | (size <= 0.0),
        "Coupling window_size must be finite and positive.",
    )
    return CouplingWindow(
        state.window_index,
        state.time,
        state.time + size,
    )


def _fixed_point_interface_solve(
    prepared: PreparedCoupling,
    method: FixedPointIteration,
    termination: NonlinearTermination,
    state: CouplingState,
    window: CouplingWindow,
    initial_coordinates: Array,
    args: Any,
    /,
    *,
    gauss_seidel_order: tuple[str, ...] | None,
) -> tuple[NonlinearResult, _ParticipantWork]:
    """Iterate the interface sweep map, accounting the work of every iterate."""

    def mapping(coordinates: Array, runtime_args: Any) -> tuple[Array, _ParticipantWork]:
        current_values = _unpack_interface(prepared, coordinates, state.exchange_values)
        evaluation = _stagewise_evaluation(
            prepared,
            window,
            state,
            current_values,
            runtime_args,
            gauss_seidel_order=gauss_seidel_order,
        )
        return _pack_interface(prepared, evaluation.exchange_values), _ParticipantWork(
            evaluation.participant_work, evaluation.participant_iterations
        )

    problem = FixedPointProblem(
        mapping,
        problem_id=f"{prepared.problem_id}/interface-fixed-point",
        evaluation_work=True,
    )
    result = method.solve(
        problem,
        initial_coordinates,
        termination=termination,
        args=args,
    )
    iterate_work = result.evaluation_work
    if not isinstance(iterate_work, _ParticipantWork):
        raise RuntimeError("Fixed-point interface solve did not report its iterate work.")
    return result, iterate_work


def _root_interface_solve(
    prepared: PreparedCoupling,
    policy: ImplicitCouplingPolicy,
    method: AbstractNonlinearMethod,
    state: CouplingState,
    window: CouplingWindow,
    initial_coordinates: Array,
    args: Any,
    /,
) -> NonlinearResult:
    """Solve the interface residual with a general nonlinear root method."""
    coordinate_space = ArraySpace(
        (prepared.report.resources.interface_size,),
        dtype=prepared.coordinate_dtype,
        space_id=f"{prepared.plan_id}/interface-coordinates",
    )

    def residual(
        coordinates: Array, runtime_args: Any
    ) -> tuple[Array, _CouplingEvaluation]:
        current_values = _unpack_interface(prepared, coordinates, state.exchange_values)
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
        return implicit_root_result(
            problem,
            initial_coordinates,
            method=method,
            termination=policy.termination,
            derivative_policy=policy.derivative_policy,
            args=args,
        )
    return method.solve(
        problem,
        initial_coordinates,
        termination=policy.termination,
        args=args,
    )


def advance_coupling_window(
    prepared: PreparedCoupling,
    state: CouplingState,
    window_size: Any,
    args: Any = None,
    /,
) -> CouplingWindowResult:
    """Advance one fixed coupling window and atomically commit only valid work."""

    window = _checked_window(prepared, state, window_size)
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
    method = policy.method
    if isinstance(method, FixedPointIteration):
        sweep = policy.fixed_point_sweep
        if sweep is None:
            raise RuntimeError("Prepared fixed-point coupling sweep is missing.")
        # The final re-evaluation replays the same sweep as the iterates, so a
        # Gauss-Seidel consumer receives the amount its source spent this sweep.
        gauss_seidel_order = None if sweep.kind == "jacobi" else sweep.subsystem_order
        nonlinear_result, iterate_work = _fixed_point_interface_solve(
            prepared,
            method,
            policy.termination,
            state,
            window,
            initial_coordinates,
            args,
            gauss_seidel_order=gauss_seidel_order,
        )
    else:
        nonlinear_result = _root_interface_solve(
            prepared, policy, method, state, window, initial_coordinates, args
        )
        iterate_work = None
        gauss_seidel_order = None

    final_values = _unpack_interface(
        prepared, nonlinear_result.state, state.exchange_values
    )
    final_evaluation = _stagewise_evaluation(
        prepared,
        window,
        state,
        final_values,
        args,
        gauss_seidel_order=gauss_seidel_order,
    )
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
        iterate_work=iterate_work,
    )


__all__ = ["advance_coupling_window"]
