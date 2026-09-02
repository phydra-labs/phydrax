#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite, prod

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..operators.quantum._operations import LocalUnitaryOperation, QuantumProgram
from ..operators.quantum._propagation import unitarity_residual
from ..tensor_network import MatrixProductState
from ..tensor_network._split import TensorTruncationEvidence, truncated_svd


class MPSQuantumProgramStatus(IntEnum):
    SUCCESS = 0
    INVALID_OPERATION = 1
    INVALID_INITIAL_STATE = 2
    NONFINITE_RESULT = 3
    TRUNCATION_BUDGET_EXCEEDED = 4
    NORM_DRIFT = 5


class MPSQuantumProgramPolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_window_sites: int = eqx.field(static=True)
    maximum_workspace_elements: int = eqx.field(static=True)
    unitarity_tolerance: float = eqx.field(static=True)
    norm_tolerance: float = eqx.field(static=True)
    maximum_discarded_weight: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_bond_dimension: int,
        maximum_window_sites: int = 8,
        maximum_workspace_elements: int = 10_000_000,
        unitarity_tolerance: float = 1e-6,
        norm_tolerance: float = 1e-6,
        maximum_discarded_weight: float = 1e-6,
    ):
        integers = (
            int(maximum_bond_dimension),
            int(maximum_window_sites),
            int(maximum_workspace_elements),
        )
        tolerances = (
            float(unitarity_tolerance),
            float(norm_tolerance),
            float(maximum_discarded_weight),
        )
        if any(value < 1 for value in integers):
            raise ValueError("MPS program capacities must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("MPS program tolerances must be finite and nonnegative.")
        self.maximum_bond_dimension = integers[0]
        self.maximum_window_sites = integers[1]
        self.maximum_workspace_elements = integers[2]
        self.unitarity_tolerance = tolerances[0]
        self.norm_tolerance = tolerances[1]
        self.maximum_discarded_weight = tolerances[2]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mps-quantum-program-policy",
                "maximum_bond_dimension": integers[0],
                "maximum_window_sites": integers[1],
                "maximum_workspace_elements": integers[2],
                "unitarity_tolerance": tolerances[0],
                "norm_tolerance": tolerances[1],
                "maximum_discarded_weight": tolerances[2],
            }
        )


class MPSQuantumProgramRoute(StrictModule):
    target_indices: tuple[int, ...] = eqx.field(static=True)
    window_start: int = eqx.field(static=True)
    window_stop: int = eqx.field(static=True)
    target_positions: tuple[int, ...] = eqx.field(static=True)
    route_id: str = eqx.field(static=True)


class MPSQuantumProgramCostEstimate(StrictModule):
    operation_elements: int = eqx.field(static=True)
    maximum_window_elements: int = eqx.field(static=True)
    split_count: int = eqx.field(static=True)


class MPSQuantumProgramPlan(StrictModule):
    policy: MPSQuantumProgramPolicy
    routes: tuple[MPSQuantumProgramRoute, ...]
    cost: MPSQuantumProgramCostEstimate
    program_id: str = eqx.field(static=True)
    state_structure_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class MPSQuantumOperationEvidence(StrictModule):
    finite: Array
    unitarity_residual: Array
    valid: Array
    schema_id: str = eqx.field(static=True)


class PreparedMPSQuantumProgram(StrictModule):
    plan: MPSQuantumProgramPlan
    program: QuantumProgram
    operation_evidence: tuple[MPSQuantumOperationEvidence, ...]
    operations_valid: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class MPSQuantumProgramDiagnostics(StrictModule):
    status: Array
    initial_state_valid: Array
    operations_valid: Array
    final_finite: Array
    initial_norm_residual: Array
    final_norm_residual: Array
    accumulated_discarded_weight: Array
    maximum_discarded_weight: Array
    operation_evidence: tuple[MPSQuantumOperationEvidence, ...]
    truncations: tuple[TensorTruncationEvidence, ...]

    @property
    def successful(self) -> Array:
        return self.status == int(MPSQuantumProgramStatus.SUCCESS)


class MPSQuantumProgramResult(StrictModule):
    final_state: MatrixProductState
    diagnostics: MPSQuantumProgramDiagnostics
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


def _route(program: QuantumProgram, operation: LocalUnitaryOperation, /):
    indices = program.layout.target_indices(operation.target_wire_ids)
    start = min(indices)
    stop = max(indices)
    positions = tuple(index - start for index in indices)
    identifier = canonical_fingerprint(
        {
            "kind": "mps-quantum-program-route",
            "targets": indices,
            "window": (start, stop),
            "positions": positions,
            "schema": operation.schema_id,
        }
    )
    return MPSQuantumProgramRoute(indices, start, stop, positions, identifier)


def plan_mps_quantum_program(
    program: QuantumProgram,
    template_state: MatrixProductState,
    policy: MPSQuantumProgramPolicy,
    /,
) -> MPSQuantumProgramPlan:
    if not isinstance(program, QuantumProgram):
        raise TypeError("program must be a QuantumProgram.")
    if not isinstance(template_state, MatrixProductState):
        raise TypeError("template_state must be a MatrixProductState.")
    if not isinstance(policy, MPSQuantumProgramPolicy):
        raise TypeError("policy must be MPSQuantumProgramPolicy.")
    if program.state_kind != "state-vector":
        raise ValueError("MPS execution requires a state-vector program.")
    if template_state.site_count != len(program.layout.wire_ids):
        raise ValueError("MPS site count must match the program layout.")
    if template_state.physical_dimensions != program.layout.local_dimensions:
        raise ValueError("MPS physical dimensions must match the program layout.")
    routes = tuple(_route(program, operation) for operation in program.operations)
    maximum_window = 0
    split_count = 0
    for route in routes:
        span = route.window_stop - route.window_start + 1
        if span > policy.maximum_window_sites:
            raise MemoryError("MPS operation exceeds maximum_window_sites.")
        left = 1 if route.window_start == 0 else policy.maximum_bond_dimension
        right = (
            1
            if route.window_stop == template_state.site_count - 1
            else policy.maximum_bond_dimension
        )
        elements = (
            left
            * prod(
                template_state.physical_dimensions[
                    route.window_start : route.window_stop + 1
                ]
            )
            * right
        )
        maximum_window = max(maximum_window, elements)
        split_count += span - 1
    if maximum_window > policy.maximum_workspace_elements:
        raise MemoryError("MPS operation exceeds maximum_workspace_elements.")
    operation_elements = sum(
        int(operation.unitary.size) for operation in program.operations
    )
    cost = MPSQuantumProgramCostEstimate(operation_elements, maximum_window, split_count)
    plan_id = canonical_fingerprint(
        {
            "kind": "mps-quantum-program-plan",
            "program": program.program_id,
            "state": template_state.structure_id,
            "precision": template_state.precision.policy_id,
            "policy": policy.policy_id,
            "routes": tuple(route.route_id for route in routes),
        }
    )
    return MPSQuantumProgramPlan(
        policy,
        routes,
        cost,
        program.program_id,
        template_state.structure_id,
        template_state.precision.policy_id,
        plan_id,
    )


def _validate_schema(program: QuantumProgram, plan: MPSQuantumProgramPlan, /) -> None:
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
        finite = jnp.all(jnp.isfinite(operation.unitary))
        residual = unitarity_residual(operation.unitary)
        records.append(
            MPSQuantumOperationEvidence(
                finite,
                residual,
                finite & (residual <= policy.unitarity_tolerance),
                operation.schema_id,
            )
        )
    return tuple(records)


def prepare_mps_quantum_program(
    program: QuantumProgram,
    plan: MPSQuantumProgramPlan,
    /,
) -> PreparedMPSQuantumProgram:
    if not isinstance(plan, MPSQuantumProgramPlan):
        raise TypeError("plan must be MPSQuantumProgramPlan.")
    _validate_schema(program, plan)
    evidence = _operation_evidence(program, plan.policy)
    valid = (
        jnp.all(jnp.stack([record.valid for record in evidence]))
        if evidence
        else jnp.asarray(True)
    )
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-mps-quantum-program", "plan": plan.plan_id}
    )
    return PreparedMPSQuantumProgram(
        plan,
        program,
        evidence,
        valid,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_mps_quantum_program(
    prepared: PreparedMPSQuantumProgram,
    program: QuantumProgram,
    /,
) -> PreparedMPSQuantumProgram:
    if not isinstance(prepared, PreparedMPSQuantumProgram):
        raise TypeError("prepared must be PreparedMPSQuantumProgram.")
    _validate_schema(program, prepared.plan)
    evidence = _operation_evidence(program, prepared.plan.policy)
    valid = (
        jnp.all(jnp.stack([record.valid for record in evidence]))
        if evidence
        else jnp.asarray(True)
    )
    return PreparedMPSQuantumProgram(
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
        window = oe.contract("l...a,apb->l...pb", window, tensor)
    return window


def _apply_unitary(window, route, physical_dimensions, unitary):
    span = route.window_stop - route.window_start + 1
    physical_labels = list(range(1, span + 1))
    output_labels = list(range(span + 2, span + 2 + len(route.target_positions)))
    unitary_labels = output_labels + [
        physical_labels[position] for position in route.target_positions
    ]
    target_to_output = dict(zip(route.target_positions, output_labels, strict=True))
    result_labels = (
        [0]
        + [
            target_to_output.get(position, physical_labels[position])
            for position in range(span)
        ]
        + [span + 1]
    )
    target_dimensions = tuple(
        physical_dimensions[route.window_start + position]
        for position in route.target_positions
    )
    gate = unitary.reshape(target_dimensions + target_dimensions)
    return oe.contract(
        window,
        [0] + physical_labels + [span + 1],
        gate,
        unitary_labels,
        result_labels,
    )


def _split_window(state, route, window, policy):
    precision = state.precision
    dimensions = state.physical_dimensions[route.window_start : route.window_stop + 1]
    current = window
    tensors = []
    records = []
    left_dimension = int(current.shape[0])
    for dimension in dimensions[:-1]:
        matrix = current.reshape((left_dimension * dimension, -1))
        left, right, evidence = truncated_svd(
            matrix,
            maximum_rank=policy.maximum_bond_dimension,
            absorb="right",
            precision=precision,
            evidence_source=state.tensors,
            evidence_children={"input-state": state.precision_evidence},
        )
        retained = evidence.retained_rank
        tensors.append(left.reshape((left_dimension, dimension, retained)))
        remaining = current.shape[2:]
        current = right.reshape((retained,) + remaining)
        left_dimension = retained
        records.append(evidence)
    tensors.append(current.reshape((left_dimension, dimensions[-1], current.shape[-1])))
    return tuple(tensors), tuple(records)


def execute_mps_quantum_program(
    prepared: PreparedMPSQuantumProgram,
    state: MatrixProductState,
    /,
) -> MPSQuantumProgramResult:
    if not isinstance(prepared, PreparedMPSQuantumProgram):
        raise TypeError("prepared must be PreparedMPSQuantumProgram.")
    if not isinstance(state, MatrixProductState):
        raise TypeError("state must be a MatrixProductState.")
    if state.structure_id != prepared.plan.state_structure_id:
        raise ValueError("MPS structure differs from the prepared template.")
    if state.precision.policy_id != prepared.plan.precision_policy_id:
        raise ValueError("MPS precision differs from the prepared plan.")

    initial_finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in state.tensors])
    )
    initial_norm = state.norm()
    initial_norm_residual = jnp.abs(initial_norm - 1.0)
    initial_valid = (
        initial_finite
        & jnp.isfinite(initial_norm)
        & (initial_norm_residual <= prepared.plan.policy.norm_tolerance)
    )
    current = state
    truncations = []
    for operation, route, evidence in zip(
        prepared.program.operations,
        prepared.plan.routes,
        prepared.operation_evidence,
        strict=True,
    ):
        dimension = operation.unitary.shape[0]
        safe_unitary = jnp.where(
            evidence.valid,
            operation.unitary,
            jnp.eye(dimension, dtype=operation.unitary.dtype),
        )
        window = _contract_window(current, route)
        transformed = _apply_unitary(
            window, route, current.physical_dimensions, safe_unitary
        )
        replacement, records = _split_window(
            current, route, transformed, prepared.plan.policy
        )
        tensors = list(current.tensors)
        tensors[route.window_start : route.window_stop + 1] = replacement
        current = MatrixProductState(tuple(tensors), precision=current.precision)
        truncations.extend(records)

    final_finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(tensor)) for tensor in current.tensors])
    )
    final_norm = current.norm()
    final_norm_residual = jnp.abs(final_norm - 1.0)
    weights = (
        jnp.stack([record.discarded_weight for record in truncations])
        if truncations
        else jnp.zeros((0,), dtype=state.tensors[0].real.dtype)
    )
    accumulated = jnp.sum(weights)
    maximum = jnp.max(weights) if truncations else jnp.asarray(0.0, dtype=weights.dtype)
    status = jnp.asarray(int(MPSQuantumProgramStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~prepared.operations_valid,
        int(MPSQuantumProgramStatus.INVALID_OPERATION),
        status,
    )
    status = jnp.where(
        prepared.operations_valid & ~initial_valid,
        int(MPSQuantumProgramStatus.INVALID_INITIAL_STATE),
        status,
    )
    status = jnp.where(
        prepared.operations_valid & initial_valid & ~final_finite,
        int(MPSQuantumProgramStatus.NONFINITE_RESULT),
        status,
    )
    status = jnp.where(
        prepared.operations_valid
        & initial_valid
        & final_finite
        & (accumulated > prepared.plan.policy.maximum_discarded_weight),
        int(MPSQuantumProgramStatus.TRUNCATION_BUDGET_EXCEEDED),
        status,
    )
    status = jnp.where(
        prepared.operations_valid
        & initial_valid
        & final_finite
        & (accumulated <= prepared.plan.policy.maximum_discarded_weight)
        & (final_norm_residual > prepared.plan.policy.norm_tolerance),
        int(MPSQuantumProgramStatus.NORM_DRIFT),
        status,
    )
    diagnostics = MPSQuantumProgramDiagnostics(
        status,
        initial_valid,
        prepared.operations_valid,
        final_finite,
        initial_norm_residual,
        final_norm_residual,
        accumulated,
        maximum,
        prepared.operation_evidence,
        tuple(truncations),
    )
    return MPSQuantumProgramResult(
        current,
        diagnostics,
        prepared.numeric_version,
        prepared.prepared_id,
    )


__all__ = [
    "MPSQuantumOperationEvidence",
    "MPSQuantumProgramCostEstimate",
    "MPSQuantumProgramDiagnostics",
    "MPSQuantumProgramPlan",
    "MPSQuantumProgramPolicy",
    "MPSQuantumProgramResult",
    "MPSQuantumProgramRoute",
    "MPSQuantumProgramStatus",
    "PreparedMPSQuantumProgram",
    "execute_mps_quantum_program",
    "plan_mps_quantum_program",
    "prepare_mps_quantum_program",
    "refresh_mps_quantum_program",
]
