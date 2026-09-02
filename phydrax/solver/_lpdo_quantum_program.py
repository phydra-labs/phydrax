#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..operators.quantum._operations import (
    LocalKrausChannelOperation,
    LocalUnitaryOperation,
    QuantumProgram,
)
from ..operators.quantum._propagation import (
    kraus_trace_preservation_residual,
    unitarity_residual,
)
from ..tensor_network import LocallyPurifiedDensity
from ..tensor_network._split import TensorTruncationEvidence, truncated_svd


class LPDOQuantumProgramStatus(IntEnum):
    SUCCESS = 0
    INVALID_OPERATION = 1
    INVALID_INITIAL_STATE = 2
    NONFINITE_RESULT = 3
    TRUNCATION_BUDGET_EXCEEDED = 4
    TRACE_DRIFT = 5


class LPDOQuantumProgramPolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_purification_dimension: int = eqx.field(static=True)
    maximum_window_sites: int = eqx.field(static=True)
    maximum_workspace_elements: int = eqx.field(static=True)
    unitarity_tolerance: float = eqx.field(static=True)
    trace_preservation_tolerance: float = eqx.field(static=True)
    trace_tolerance: float = eqx.field(static=True)
    maximum_discarded_weight: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_bond_dimension: int,
        maximum_purification_dimension: int,
        maximum_window_sites: int = 6,
        maximum_workspace_elements: int = 10_000_000,
        unitarity_tolerance: float = 1e-6,
        trace_preservation_tolerance: float = 1e-6,
        trace_tolerance: float = 1e-6,
        maximum_discarded_weight: float = 1e-6,
    ):
        integers = (
            int(maximum_bond_dimension),
            int(maximum_purification_dimension),
            int(maximum_window_sites),
            int(maximum_workspace_elements),
        )
        tolerances = tuple(
            float(value)
            for value in (
                unitarity_tolerance,
                trace_preservation_tolerance,
                trace_tolerance,
                maximum_discarded_weight,
            )
        )
        if any(value < 1 for value in integers):
            raise ValueError("LPDO program capacities must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("LPDO program tolerances must be finite and nonnegative.")
        self.maximum_bond_dimension = integers[0]
        self.maximum_purification_dimension = integers[1]
        self.maximum_window_sites = integers[2]
        self.maximum_workspace_elements = integers[3]
        self.unitarity_tolerance = tolerances[0]
        self.trace_preservation_tolerance = tolerances[1]
        self.trace_tolerance = tolerances[2]
        self.maximum_discarded_weight = tolerances[3]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "lpdo-quantum-program-policy",
                "capacities": integers,
                "tolerances": tolerances,
            }
        )


class LPDOQuantumProgramRoute(StrictModule):
    target_indices: tuple[int, ...] = eqx.field(static=True)
    window_start: int = eqx.field(static=True)
    window_stop: int = eqx.field(static=True)
    target_positions: tuple[int, ...] = eqx.field(static=True)
    kraus_anchor_position: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)


class LPDOQuantumProgramCostEstimate(StrictModule):
    operation_elements: int = eqx.field(static=True)
    maximum_window_elements: int = eqx.field(static=True)
    split_count: int = eqx.field(static=True)
    maximum_kraus_capacity: int = eqx.field(static=True)


class LPDOQuantumProgramPlan(StrictModule):
    policy: LPDOQuantumProgramPolicy
    routes: tuple[LPDOQuantumProgramRoute, ...]
    cost: LPDOQuantumProgramCostEstimate
    program_id: str = eqx.field(static=True)
    state_structure_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class LPDOQuantumOperationEvidence(StrictModule):
    finite: Array
    structure_residual: Array
    valid: Array
    operation_kind: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    cp_by_construction: bool = eqx.field(static=True)


class PreparedLPDOQuantumProgram(StrictModule):
    plan: LPDOQuantumProgramPlan
    program: QuantumProgram
    operation_evidence: tuple[LPDOQuantumOperationEvidence, ...]
    operations_valid: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class LPDOQuantumProgramDiagnostics(StrictModule):
    status: Array
    initial_state_valid: Array
    operations_valid: Array
    final_finite: Array
    initial_trace_residual: Array
    final_trace_residual: Array
    accumulated_bond_discarded_weight: Array
    accumulated_purification_discarded_weight: Array
    completely_positive_by_construction: Array
    trace_preserving_operations: Array
    positive_semidefinite_by_construction: Array
    trace_within_tolerance: Array
    truncation_within_budget: Array
    operation_evidence: tuple[LPDOQuantumOperationEvidence, ...]
    bond_truncations: tuple[TensorTruncationEvidence, ...]
    purification_truncations: tuple[TensorTruncationEvidence, ...]

    @property
    def successful(self) -> Array:
        return self.status == int(LPDOQuantumProgramStatus.SUCCESS)


class LPDOQuantumProgramResult(StrictModule):
    final_state: LocallyPurifiedDensity
    diagnostics: LPDOQuantumProgramDiagnostics
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


def _route(program, operation):
    indices = program.layout.target_indices(operation.target_wire_ids)
    start = min(indices)
    stop = max(indices)
    positions = tuple(index - start for index in indices)
    anchor = min(indices) - start
    identifier = canonical_fingerprint(
        {
            "kind": "lpdo-quantum-program-route",
            "targets": indices,
            "window": (start, stop),
            "positions": positions,
            "kraus_anchor": anchor,
            "schema": operation.schema_id,
        }
    )
    return LPDOQuantumProgramRoute(indices, start, stop, positions, anchor, identifier)


def plan_lpdo_quantum_program(
    program: QuantumProgram,
    template_state: LocallyPurifiedDensity,
    policy: LPDOQuantumProgramPolicy,
    /,
) -> LPDOQuantumProgramPlan:
    if not isinstance(program, QuantumProgram):
        raise TypeError("program must be a QuantumProgram.")
    if not isinstance(template_state, LocallyPurifiedDensity):
        raise TypeError("template_state must be LocallyPurifiedDensity.")
    if not isinstance(policy, LPDOQuantumProgramPolicy):
        raise TypeError("policy must be LPDOQuantumProgramPolicy.")
    if program.state_kind != "density-matrix":
        raise ValueError("LPDO execution requires a density-matrix program.")
    if template_state.site_count != len(program.layout.wire_ids):
        raise ValueError("LPDO site count must match the program layout.")
    if template_state.physical_dimensions != program.layout.local_dimensions:
        raise ValueError("LPDO physical dimensions must match the program layout.")
    routes = tuple(_route(program, operation) for operation in program.operations)
    for operation, route in zip(program.operations, routes, strict=True):
        if len(route.target_indices) > 2:
            raise ValueError(
                "LPDO execution supports only one-site channels and one/two-site "
                "unitaries; decompose larger operations before planning."
            )
        if (
            len(route.target_indices) == 2
            and abs(route.target_indices[1] - route.target_indices[0]) != 1
        ):
            raise ValueError(
                "Non-nearest-neighbor LPDO operations require an explicit "
                "caller-visible SWAP compilation."
            )
        if (
            isinstance(operation, LocalKrausChannelOperation)
            and len(route.target_indices) != 1
        ):
            raise ValueError("LPDO Kraus channels must act on exactly one site.")
    maximum_window = 0
    split_count = 0
    maximum_kraus = 0
    for operation, route in zip(program.operations, routes, strict=True):
        span = route.window_stop - route.window_start + 1
        if span > policy.maximum_window_sites:
            raise MemoryError("LPDO operation exceeds maximum_window_sites.")
        left = 1 if route.window_start == 0 else policy.maximum_bond_dimension
        right = (
            1
            if route.window_stop == template_state.site_count - 1
            else policy.maximum_bond_dimension
        )
        local = 1
        for site in range(route.window_start, route.window_stop + 1):
            local *= (
                template_state.physical_dimensions[site]
                * policy.maximum_purification_dimension
            )
        kraus = (
            operation.kraus.shape[0]
            if isinstance(operation, LocalKrausChannelOperation)
            else 1
        )
        maximum_kraus = max(maximum_kraus, int(kraus))
        maximum_window = max(maximum_window, left * local * right * int(kraus))
        split_count += span - 1
    if maximum_window > policy.maximum_workspace_elements:
        raise MemoryError("LPDO operation exceeds maximum_workspace_elements.")
    operation_elements = sum(
        int(
            operation.unitary.size
            if isinstance(operation, LocalUnitaryOperation)
            else operation.kraus.size
        )
        for operation in program.operations
    )
    cost = LPDOQuantumProgramCostEstimate(
        operation_elements, maximum_window, split_count, maximum_kraus
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "lpdo-quantum-program-plan",
            "program": program.program_id,
            "state": template_state.structure_id,
            "precision": template_state.precision.policy_id,
            "policy": policy.policy_id,
            "routes": tuple(route.route_id for route in routes),
        }
    )
    return LPDOQuantumProgramPlan(
        policy,
        routes,
        cost,
        program.program_id,
        template_state.structure_id,
        template_state.precision.policy_id,
        plan_id,
    )


def _validate_schema(program, plan):
    if program.program_id != plan.program_id:
        raise ValueError("Quantum-program structure changed; replan is required.")
    routes = tuple(
        _route(program, operation).route_id for operation in program.operations
    )
    if routes != tuple(route.route_id for route in plan.routes):
        raise ValueError("Quantum-program routes changed; replan is required.")


def _operation_evidence(program, policy):
    records = []
    for operation in program.operations:
        if isinstance(operation, LocalUnitaryOperation):
            finite = jnp.all(jnp.isfinite(operation.unitary))
            residual = unitarity_residual(operation.unitary)
            valid = finite & (residual <= policy.unitarity_tolerance)
            records.append(
                LPDOQuantumOperationEvidence(
                    finite, residual, valid, "unitary", operation.schema_id, True
                )
            )
        else:
            finite = jnp.all(jnp.isfinite(operation.kraus))
            residual = kraus_trace_preservation_residual(operation.kraus)
            valid = finite & (residual <= policy.trace_preservation_tolerance)
            records.append(
                LPDOQuantumOperationEvidence(
                    finite, residual, valid, "kraus", operation.schema_id, True
                )
            )
    return tuple(records)


def prepare_lpdo_quantum_program(program, plan, /):
    if not isinstance(plan, LPDOQuantumProgramPlan):
        raise TypeError("plan must be LPDOQuantumProgramPlan.")
    _validate_schema(program, plan)
    evidence = _operation_evidence(program, plan.policy)
    valid = (
        jnp.all(jnp.stack([record.valid for record in evidence]))
        if evidence
        else jnp.asarray(True)
    )
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-lpdo-quantum-program", "plan": plan.plan_id}
    )
    return PreparedLPDOQuantumProgram(
        plan,
        program,
        evidence,
        valid,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_lpdo_quantum_program(prepared, program, /):
    if not isinstance(prepared, PreparedLPDOQuantumProgram):
        raise TypeError("prepared must be PreparedLPDOQuantumProgram.")
    _validate_schema(program, prepared.plan)
    evidence = _operation_evidence(program, prepared.plan.policy)
    valid = (
        jnp.all(jnp.stack([record.valid for record in evidence]))
        if evidence
        else jnp.asarray(True)
    )
    return PreparedLPDOQuantumProgram(
        prepared.plan,
        program,
        evidence,
        valid,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def _contract_window(state, route):
    tensors = state.precision.contraction(
        state.tensors[route.window_start : route.window_stop + 1]
    )
    window = tensors[0]
    for tensor in tensors[1:]:
        window = oe.contract("l...a,apkb->l...pkb", window, tensor)
    return window


def _window_labels(span):
    labels = [0]
    for position in range(span):
        labels.extend((1 + 2 * position, 2 + 2 * position))
    labels.append(2 * span + 1)
    return labels


def _apply_unitary(window, route, physical_dimensions, unitary):
    span = route.window_stop - route.window_start + 1
    labels = _window_labels(span)
    physical_labels = [1 + 2 * position for position in range(span)]
    output_labels = list(range(2 * span + 2, 2 * span + 2 + len(route.target_positions)))
    gate_labels = output_labels + [
        physical_labels[position] for position in route.target_positions
    ]
    target_to_output = dict(zip(route.target_positions, output_labels, strict=True))
    result_labels = list(labels)
    for position, output in target_to_output.items():
        result_labels[1 + 2 * position] = output
    target_dimensions = tuple(
        physical_dimensions[route.window_start + position]
        for position in route.target_positions
    )
    return oe.contract(
        window,
        labels,
        unitary.reshape(target_dimensions + target_dimensions),
        gate_labels,
        result_labels,
    )


def _apply_kraus(window, route, physical_dimensions, kraus):
    span = route.window_stop - route.window_start + 1
    labels = _window_labels(span)
    physical_labels = [1 + 2 * position for position in range(span)]
    global_kraus_label = 2 * span + 2
    output_labels = list(range(2 * span + 3, 2 * span + 3 + len(route.target_positions)))
    gate_labels = (
        [global_kraus_label]
        + output_labels
        + [physical_labels[position] for position in route.target_positions]
    )
    target_to_output = dict(zip(route.target_positions, output_labels, strict=True))
    result_labels = [0]
    for position in range(span):
        result_labels.append(target_to_output.get(position, physical_labels[position]))
        result_labels.append(2 + 2 * position)
        if position == route.kraus_anchor_position:
            result_labels.append(global_kraus_label)
    result_labels.append(2 * span + 1)
    target_dimensions = tuple(
        physical_dimensions[route.window_start + position]
        for position in route.target_positions
    )
    result = oe.contract(
        window,
        labels,
        kraus.reshape((kraus.shape[0],) + target_dimensions + target_dimensions),
        gate_labels,
        result_labels,
    )
    shape = list(result.shape)
    anchor_axis = 2 + 2 * route.kraus_anchor_position
    shape[anchor_axis : anchor_axis + 2] = [shape[anchor_axis] * shape[anchor_axis + 1]]
    return result.reshape(tuple(shape))


def _split_window(state, route, window, policy):
    precision = state.precision
    dimensions = state.physical_dimensions[route.window_start : route.window_stop + 1]
    current = window
    tensors = []
    records = []
    left_dimension = int(current.shape[0])
    for dimension in dimensions[:-1]:
        kraus_dimension = int(current.shape[2])
        matrix = current.reshape((left_dimension * dimension * kraus_dimension, -1))
        left, right, evidence = truncated_svd(
            matrix,
            maximum_rank=policy.maximum_bond_dimension,
            absorb="right",
            precision=precision,
            evidence_source=state.tensors,
            evidence_children={"input-state": state.precision_evidence},
        )
        retained = evidence.retained_rank
        tensors.append(
            left.reshape((left_dimension, dimension, kraus_dimension, retained))
        )
        current = right.reshape((retained,) + current.shape[3:])
        left_dimension = retained
        records.append(evidence)
    tensors.append(
        current.reshape(
            (left_dimension, dimensions[-1], current.shape[2], current.shape[-1])
        )
    )
    return tuple(tensors), tuple(records)


def _compress_purification(state, site, maximum_dimension):
    precision = state.precision
    tensor = precision.contraction(state.tensors[site])
    matrix = jnp.transpose(tensor, (0, 1, 3, 2)).reshape((-1, tensor.shape[2]))
    compressed, _, evidence = truncated_svd(
        matrix,
        maximum_rank=maximum_dimension,
        absorb="left",
        precision=precision,
        evidence_source=state.tensors,
        evidence_children={"input-state": state.precision_evidence},
    )
    retained = evidence.retained_rank
    compressed = compressed.reshape(
        (tensor.shape[0], tensor.shape[1], tensor.shape[-1], retained)
    )
    tensors = list(state.tensors)
    tensors[site] = jnp.transpose(compressed, (0, 1, 3, 2))
    return LocallyPurifiedDensity(tuple(tensors), precision=precision), evidence


def execute_lpdo_quantum_program(prepared, state, /):
    if not isinstance(prepared, PreparedLPDOQuantumProgram):
        raise TypeError("prepared must be PreparedLPDOQuantumProgram.")
    if not isinstance(state, LocallyPurifiedDensity):
        raise TypeError("state must be LocallyPurifiedDensity.")
    if state.structure_id != prepared.plan.state_structure_id:
        raise ValueError("LPDO structure differs from the prepared template.")
    if state.precision.policy_id != prepared.plan.precision_policy_id:
        raise ValueError("LPDO precision differs from the prepared plan.")
    initial_finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in state.tensors])
    )
    initial_trace = state.raw_trace()
    initial_trace_residual = jnp.abs(initial_trace - 1.0)
    initial_valid = (
        initial_finite
        & jnp.isfinite(initial_trace)
        & (initial_trace > 0.0)
        & (initial_trace_residual <= prepared.plan.policy.trace_tolerance)
    )
    current = state
    bond_truncations = []
    purification_truncations = []
    for operation, route, evidence in zip(
        prepared.program.operations,
        prepared.plan.routes,
        prepared.operation_evidence,
        strict=True,
    ):
        window = _contract_window(current, route)
        if isinstance(operation, LocalUnitaryOperation):
            transformed = _apply_unitary(
                window, route, current.physical_dimensions, operation.unitary
            )
        else:
            transformed = _apply_kraus(
                window, route, current.physical_dimensions, operation.kraus
            )
        replacement, records = _split_window(
            current, route, transformed, prepared.plan.policy
        )
        tensors = list(current.tensors)
        tensors[route.window_start : route.window_stop + 1] = replacement
        current = LocallyPurifiedDensity(tuple(tensors), precision=current.precision)
        bond_truncations.extend(records)
        if isinstance(operation, LocalKrausChannelOperation):
            anchor = route.window_start + route.kraus_anchor_position
            current, purification = _compress_purification(
                current,
                anchor,
                prepared.plan.policy.maximum_purification_dimension,
            )
            purification_truncations.append(purification)

    final_finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in current.tensors])
    )
    final_trace = current.raw_trace()
    final_trace_residual = jnp.abs(final_trace - 1.0)
    bond_weights = (
        jnp.stack([record.discarded_weight for record in bond_truncations])
        if bond_truncations
        else jnp.zeros((0,), dtype=state.tensors[0].real.dtype)
    )
    purification_weights = (
        jnp.stack([record.discarded_weight for record in purification_truncations])
        if purification_truncations
        else jnp.zeros((0,), dtype=state.tensors[0].real.dtype)
    )
    bond_total = jnp.sum(bond_weights)
    purification_total = jnp.sum(purification_weights)
    total = bond_total + purification_total
    cp_by_construction = jnp.asarray(
        all(record.cp_by_construction for record in prepared.operation_evidence)
    )
    trace_preserving = prepared.operations_valid
    psd_by_construction = jnp.asarray(True)
    trace_within_tolerance = jnp.isfinite(final_trace_residual) & (
        final_trace_residual <= prepared.plan.policy.trace_tolerance
    )
    truncation_within_budget = jnp.isfinite(total) & (
        total <= prepared.plan.policy.maximum_discarded_weight
    )
    status = jnp.asarray(int(LPDOQuantumProgramStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~prepared.operations_valid,
        int(LPDOQuantumProgramStatus.INVALID_OPERATION),
        status,
    )
    status = jnp.where(
        prepared.operations_valid & ~initial_valid,
        int(LPDOQuantumProgramStatus.INVALID_INITIAL_STATE),
        status,
    )
    status = jnp.where(
        prepared.operations_valid & initial_valid & ~final_finite,
        int(LPDOQuantumProgramStatus.NONFINITE_RESULT),
        status,
    )
    status = jnp.where(
        prepared.operations_valid
        & initial_valid
        & final_finite
        & (total > prepared.plan.policy.maximum_discarded_weight),
        int(LPDOQuantumProgramStatus.TRUNCATION_BUDGET_EXCEEDED),
        status,
    )
    status = jnp.where(
        prepared.operations_valid
        & initial_valid
        & final_finite
        & (total <= prepared.plan.policy.maximum_discarded_weight)
        & (final_trace_residual > prepared.plan.policy.trace_tolerance),
        int(LPDOQuantumProgramStatus.TRACE_DRIFT),
        status,
    )
    diagnostics = LPDOQuantumProgramDiagnostics(
        status,
        initial_valid,
        prepared.operations_valid,
        final_finite,
        initial_trace_residual,
        final_trace_residual,
        bond_total,
        purification_total,
        cp_by_construction,
        trace_preserving,
        psd_by_construction,
        trace_within_tolerance,
        truncation_within_budget,
        prepared.operation_evidence,
        tuple(bond_truncations),
        tuple(purification_truncations),
    )
    return LPDOQuantumProgramResult(
        current,
        diagnostics,
        prepared.numeric_version,
        prepared.prepared_id,
    )


__all__ = [
    "LPDOQuantumOperationEvidence",
    "LPDOQuantumProgramCostEstimate",
    "LPDOQuantumProgramDiagnostics",
    "LPDOQuantumProgramPlan",
    "LPDOQuantumProgramPolicy",
    "LPDOQuantumProgramResult",
    "LPDOQuantumProgramRoute",
    "LPDOQuantumProgramStatus",
    "PreparedLPDOQuantumProgram",
    "execute_lpdo_quantum_program",
    "plan_lpdo_quantum_program",
    "prepare_lpdo_quantum_program",
    "refresh_lpdo_quantum_program",
]
