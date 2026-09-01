#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..operators.quantum._operations import LocalUnitaryOperation, QuantumProgram
from ..operators.quantum._propagation import (
    apply_local_kraus_to_density,
    apply_local_unitary_to_state,
    conjugate_local_density,
    density_invariant_residuals,
    kraus_trace_preservation_residual,
    unitarity_residual,
)
from ..operators.quantum._register import HilbertRegisterLayout


DensityPositivityAudit: TypeAlias = Literal["full", "construction"]


class DenseQuantumProgramStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    INVALID_OPERATION = 2
    NONFINITE_RESULT = 3
    PHYSICALITY_FAILED = 4


class DenseQuantumProgramPolicy(StrictModule):
    """Precision and resource envelope for deterministic dense execution."""

    maximum_state_bytes: int = eqx.field(static=True)
    maximum_operation_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    unitarity_tolerance: float = eqx.field(static=True)
    trace_preservation_tolerance: float = eqx.field(static=True)
    norm_tolerance: float = eqx.field(static=True)
    trace_tolerance: float = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    positivity_tolerance: float = eqx.field(static=True)
    density_positivity_audit: DensityPositivityAudit = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_state_bytes: int = 2**30,
        maximum_operation_bytes: int = 2**30,
        maximum_workspace_bytes: int = 2**30,
        compute_dtype: str = "complex128",
        unitarity_tolerance: float = 1e-6,
        trace_preservation_tolerance: float = 1e-6,
        norm_tolerance: float = 1e-6,
        trace_tolerance: float = 1e-6,
        hermiticity_tolerance: float = 1e-6,
        positivity_tolerance: float = 1e-8,
        density_positivity_audit: DensityPositivityAudit = "full",
    ):
        limits = (
            int(maximum_state_bytes),
            int(maximum_operation_bytes),
            int(maximum_workspace_bytes),
        )
        if any(limit <= 0 for limit in limits):
            raise ValueError("Dense quantum-program resource limits must be positive.")
        dtype = jnp.dtype(compute_dtype)
        if not jnp.issubdtype(dtype, jnp.complexfloating):
            raise TypeError("compute_dtype must name a complex floating-point dtype.")
        tolerances = (
            float(unitarity_tolerance),
            float(trace_preservation_tolerance),
            float(norm_tolerance),
            float(trace_tolerance),
            float(hermiticity_tolerance),
            float(positivity_tolerance),
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "Dense quantum-program tolerances must be finite and non-negative."
            )
        if density_positivity_audit not in ("full", "construction"):
            raise ValueError("Unknown density positivity-audit policy.")
        self.maximum_state_bytes = limits[0]
        self.maximum_operation_bytes = limits[1]
        self.maximum_workspace_bytes = limits[2]
        self.compute_dtype = dtype.name
        self.unitarity_tolerance = tolerances[0]
        self.trace_preservation_tolerance = tolerances[1]
        self.norm_tolerance = tolerances[2]
        self.trace_tolerance = tolerances[3]
        self.hermiticity_tolerance = tolerances[4]
        self.positivity_tolerance = tolerances[5]
        self.density_positivity_audit = density_positivity_audit
        self.policy_id = canonical_fingerprint(
            {
                "kind": "dense-quantum-program-policy",
                "maximum_state_bytes": limits[0],
                "maximum_operation_bytes": limits[1],
                "maximum_workspace_bytes": limits[2],
                "compute_dtype": dtype.name,
                "unitarity_tolerance": tolerances[0],
                "trace_preservation_tolerance": tolerances[1],
                "norm_tolerance": tolerances[2],
                "trace_tolerance": tolerances[3],
                "hermiticity_tolerance": tolerances[4],
                "positivity_tolerance": tolerances[5],
                "density_positivity_audit": density_positivity_audit,
            }
        )


class DenseQuantumProgramCostEstimate(StrictModule):
    total_dimension: int = eqx.field(static=True)
    state_elements_per_case: int = eqx.field(static=True)
    state_bytes_per_case: int = eqx.field(static=True)
    operation_elements: int = eqx.field(static=True)
    operation_bytes: int = eqx.field(static=True)
    workspace_bytes_per_case: int = eqx.field(static=True)
    maximum_kraus_capacity: int = eqx.field(static=True)
    operation_count: int = eqx.field(static=True)


class DenseQuantumProgramPlan(StrictModule):
    layout: HilbertRegisterLayout
    policy: DenseQuantumProgramPolicy
    cost: DenseQuantumProgramCostEstimate
    target_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    operation_kinds: tuple[str, ...] = eqx.field(static=True)
    operation_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    operation_dtypes: tuple[str, ...] = eqx.field(static=True)
    state_kind: str = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class DenseQuantumOperationEvidence(StrictModule):
    finite: Array
    structure_residual: Array
    valid: Array
    operation_kind: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    cp_by_construction: bool = eqx.field(static=True)


class PreparedDenseQuantumProgram(StrictModule):
    plan: DenseQuantumProgramPlan
    program: QuantumProgram
    operation_evidence: tuple[DenseQuantumOperationEvidence, ...]
    operations_valid: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class DenseQuantumProgramDiagnostics(StrictModule):
    status: Array
    initial_state_valid: Array
    operations_valid: Array
    final_finite: Array
    final_norm_residual: Array
    final_trace_residual: Array
    final_hermiticity_residual: Array
    final_minimum_eigenvalue: Array
    operation_evidence: tuple[DenseQuantumOperationEvidence, ...]
    positivity_audited: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(DenseQuantumProgramStatus.SUCCESS)


class DenseQuantumProgramResult(StrictModule):
    final_state: Array
    diagnostics: DenseQuantumProgramDiagnostics
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


def _operation_schema(
    program: QuantumProgram,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
    tuple[str, ...],
]:
    targets: list[tuple[int, ...]] = []
    kinds: list[str] = []
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    for operation in program.operations:
        targets.append(program.layout.target_indices(operation.target_wire_ids))
        if isinstance(operation, LocalUnitaryOperation):
            kinds.append("unitary")
            shapes.append(tuple(operation.unitary.shape))
            dtypes.append(str(operation.unitary.dtype))
        else:
            kinds.append("kraus")
            shapes.append(tuple(operation.kraus.shape))
            dtypes.append(str(operation.kraus.dtype))
    return tuple(targets), tuple(kinds), tuple(shapes), tuple(dtypes)


def plan_dense_quantum_program(
    program: QuantumProgram,
    policy: DenseQuantumProgramPolicy | None = None,
    /,
) -> DenseQuantumProgramPlan:
    """Resolve static local-operation routes and reject oversized dense work."""
    if not isinstance(program, QuantumProgram):
        raise TypeError("program must be a QuantumProgram.")
    selected = DenseQuantumProgramPolicy() if policy is None else policy
    if not isinstance(selected, DenseQuantumProgramPolicy):
        raise TypeError("policy must be DenseQuantumProgramPolicy or None.")
    targets, kinds, shapes, dtypes = _operation_schema(program)
    if any(dtype != selected.compute_dtype for dtype in dtypes):
        raise TypeError("Every operation dtype must match policy.compute_dtype.")
    dimension = program.layout.dimension
    state_elements = dimension if program.state_kind == "state-vector" else dimension**2
    itemsize = jnp.dtype(selected.compute_dtype).itemsize
    state_bytes = state_elements * itemsize
    operation_elements = sum(prod(shape) for shape in shapes)
    operation_bytes = operation_elements * itemsize
    workspace_multiplier = 2 if program.state_kind == "state-vector" else 3
    workspace_bytes = workspace_multiplier * state_bytes
    maximum_kraus = max(
        (shape[0] for kind, shape in zip(kinds, shapes, strict=True) if kind == "kraus"),
        default=0,
    )
    if state_bytes > selected.maximum_state_bytes:
        raise MemoryError("Dense quantum-program state exceeds maximum_state_bytes.")
    if operation_bytes > selected.maximum_operation_bytes:
        raise MemoryError(
            "Dense quantum-program operations exceed maximum_operation_bytes."
        )
    if workspace_bytes > selected.maximum_workspace_bytes:
        raise MemoryError(
            "Dense quantum-program workspace exceeds maximum_workspace_bytes."
        )
    cost = DenseQuantumProgramCostEstimate(
        dimension,
        state_elements,
        state_bytes,
        operation_elements,
        operation_bytes,
        workspace_bytes,
        maximum_kraus,
        len(program.operations),
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "dense-quantum-program-plan",
            "program": program.program_id,
            "policy": selected.policy_id,
        }
    )
    return DenseQuantumProgramPlan(
        program.layout,
        selected,
        cost,
        targets,
        kinds,
        shapes,
        dtypes,
        program.state_kind,
        program.program_id,
        plan_id,
    )


def _operation_evidence(
    program: QuantumProgram,
    policy: DenseQuantumProgramPolicy,
    /,
) -> tuple[DenseQuantumOperationEvidence, ...]:
    evidence: list[DenseQuantumOperationEvidence] = []
    for operation in program.operations:
        if isinstance(operation, LocalUnitaryOperation):
            finite = jnp.all(jnp.isfinite(operation.unitary))
            residual = unitarity_residual(operation.unitary)
            valid = finite & (residual <= policy.unitarity_tolerance)
            evidence.append(
                DenseQuantumOperationEvidence(
                    finite,
                    residual,
                    valid,
                    "unitary",
                    operation.schema_id,
                    False,
                )
            )
        else:
            finite = jnp.all(jnp.isfinite(operation.kraus))
            residual = kraus_trace_preservation_residual(operation.kraus)
            valid = finite & (residual <= policy.trace_preservation_tolerance)
            evidence.append(
                DenseQuantumOperationEvidence(
                    finite,
                    residual,
                    valid,
                    "kraus",
                    operation.schema_id,
                    True,
                )
            )
    return tuple(evidence)


def _validate_program_schema(
    program: QuantumProgram,
    plan: DenseQuantumProgramPlan,
    /,
) -> None:
    if program.program_id != plan.program_id:
        raise ValueError("Quantum-program structure changed; replan is required.")
    schema = _operation_schema(program)
    if schema != (
        plan.target_indices,
        plan.operation_kinds,
        plan.operation_shapes,
        plan.operation_dtypes,
    ):
        raise ValueError("Quantum-operation structure changed; replan is required.")


def prepare_dense_quantum_program(
    program: QuantumProgram,
    plan_or_policy: DenseQuantumProgramPlan | DenseQuantumProgramPolicy | None = None,
    /,
) -> PreparedDenseQuantumProgram:
    """Bind numerical local operators to one reusable dense execution plan."""
    if not isinstance(program, QuantumProgram):
        raise TypeError("program must be a QuantumProgram.")
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, DenseQuantumProgramPlan)
        else plan_dense_quantum_program(program, plan_or_policy)
    )
    _validate_program_schema(program, plan)
    evidence = _operation_evidence(program, plan.policy)
    valid = (
        jnp.all(jnp.stack([item.valid for item in evidence]))
        if evidence
        else jnp.asarray(True)
    )
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-dense-quantum-program", "plan": plan.plan_id}
    )
    return PreparedDenseQuantumProgram(
        plan,
        program,
        evidence,
        valid,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_dense_quantum_program(
    prepared: PreparedDenseQuantumProgram,
    program: QuantumProgram,
    /,
) -> PreparedDenseQuantumProgram:
    """Refresh local matrices while preserving exact program structure."""
    if not isinstance(prepared, PreparedDenseQuantumProgram):
        raise TypeError("prepared must be PreparedDenseQuantumProgram.")
    if not isinstance(program, QuantumProgram):
        raise TypeError("program must be a QuantumProgram.")
    _validate_program_schema(program, prepared.plan)
    evidence = _operation_evidence(program, prepared.plan.policy)
    valid = (
        jnp.all(jnp.stack([item.valid for item in evidence]))
        if evidence
        else jnp.asarray(True)
    )
    return PreparedDenseQuantumProgram(
        prepared.plan,
        program,
        evidence,
        valid,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def _batch_finite(value: Array, trailing_axes: int, /) -> Array:
    axes = tuple(range(value.ndim - trailing_axes, value.ndim))
    return jnp.all(jnp.isfinite(value), axis=axes)


def _density_trace_and_hermiticity(density: Array, /) -> tuple[Array, Array]:
    adjoint = jnp.swapaxes(jnp.conj(density), -1, -2)
    hermiticity = jnp.max(jnp.abs(density - adjoint), axis=(-2, -1))
    trace = jnp.abs(jnp.trace(density, axis1=-2, axis2=-1) - 1.0)
    return trace, hermiticity


def _execute_operations(prepared: PreparedDenseQuantumProgram, state: Array, /) -> Array:
    value = state
    for operation in prepared.program.operations:
        if isinstance(operation, LocalUnitaryOperation):
            value = (
                apply_local_unitary_to_state(
                    prepared.plan.layout,
                    operation.unitary,
                    operation.target_wire_ids,
                    value,
                )
                if prepared.plan.state_kind == "state-vector"
                else conjugate_local_density(
                    prepared.plan.layout,
                    operation.unitary,
                    operation.target_wire_ids,
                    value,
                )
            )
        else:
            value = apply_local_kraus_to_density(
                prepared.plan.layout,
                operation.kraus,
                operation.target_wire_ids,
                value,
            )
    return value


def execute_dense_quantum_program(
    prepared: PreparedDenseQuantumProgram,
    initial_state: ArrayLike,
    /,
) -> DenseQuantumProgramResult:
    """Execute one deterministic dense program and return physicality evidence."""
    if not isinstance(prepared, PreparedDenseQuantumProgram):
        raise TypeError("prepared must be PreparedDenseQuantumProgram.")
    state = jnp.asarray(initial_state)
    dimension = prepared.plan.cost.total_dimension
    if prepared.plan.state_kind == "state-vector":
        if state.ndim < 1 or state.shape[-1] != dimension:
            raise ValueError("Initial state must have shape (..., layout.dimension).")
        batch_shape = state.shape[:-1]
        trailing_axes = 1
    else:
        if state.ndim < 2 or state.shape[-2:] != (dimension, dimension):
            raise ValueError(
                "Initial density must have shape (..., layout.dimension, layout.dimension)."
            )
        batch_shape = state.shape[:-2]
        trailing_axes = 2
    if not jnp.issubdtype(state.dtype, jnp.complexfloating):
        raise TypeError("Initial quantum state must use complex floating coordinates.")
    if str(state.dtype) != prepared.plan.policy.compute_dtype:
        raise TypeError("Initial-state dtype must match policy.compute_dtype.")
    cases = prod(batch_shape) if batch_shape else 1
    if (
        cases * prepared.plan.cost.state_bytes_per_case
        > prepared.plan.policy.maximum_state_bytes
    ):
        raise MemoryError("Batched quantum state exceeds maximum_state_bytes.")
    if (
        cases * prepared.plan.cost.workspace_bytes_per_case
        > prepared.plan.policy.maximum_workspace_bytes
    ):
        raise MemoryError("Batched quantum workspace exceeds maximum_workspace_bytes.")

    initial_finite = _batch_finite(state, trailing_axes)
    nan_metric = jnp.full(batch_shape, jnp.nan)
    if prepared.plan.state_kind == "state-vector":
        initial_norm = jnp.abs(jnp.sum(jnp.abs(state) ** 2, axis=-1) - 1.0)
        initial_valid = initial_finite & (
            initial_norm <= prepared.plan.policy.norm_tolerance
        )
    else:
        initial_hermiticity, initial_trace, initial_minimum = density_invariant_residuals(
            state
        )
        initial_valid = (
            initial_finite
            & (initial_trace <= prepared.plan.policy.trace_tolerance)
            & (initial_hermiticity <= prepared.plan.policy.hermiticity_tolerance)
            & (initial_minimum >= -prepared.plan.policy.positivity_tolerance)
        )

    candidate = jax.lax.cond(
        prepared.operations_valid,
        lambda value: _execute_operations(prepared, value),
        lambda value: value,
        state,
    )
    final_finite = _batch_finite(candidate, trailing_axes)
    if prepared.plan.state_kind == "state-vector":
        final_norm = jnp.abs(jnp.sum(jnp.abs(candidate) ** 2, axis=-1) - 1.0)
        final_trace = nan_metric
        final_hermiticity = nan_metric
        final_minimum = nan_metric
        final_physical = final_norm <= prepared.plan.policy.norm_tolerance
        positivity_audited = False
    else:
        final_trace, final_hermiticity = _density_trace_and_hermiticity(candidate)
        if prepared.plan.policy.density_positivity_audit == "full":
            _, _, final_minimum = density_invariant_residuals(candidate)
            positivity_valid = final_minimum >= -prepared.plan.policy.positivity_tolerance
            positivity_audited = True
        else:
            final_minimum = nan_metric
            positivity_valid = initial_valid & prepared.operations_valid
            positivity_audited = False
        final_norm = nan_metric
        final_physical = (
            (final_trace <= prepared.plan.policy.trace_tolerance)
            & (final_hermiticity <= prepared.plan.policy.hermiticity_tolerance)
            & positivity_valid
        )
    status = jnp.full(
        batch_shape, int(DenseQuantumProgramStatus.SUCCESS), dtype=jnp.int32
    )
    status = jnp.where(
        ~final_physical,
        int(DenseQuantumProgramStatus.PHYSICALITY_FAILED),
        status,
    )
    status = jnp.where(
        ~final_finite,
        int(DenseQuantumProgramStatus.NONFINITE_RESULT),
        status,
    )
    status = jnp.where(
        ~initial_valid,
        int(DenseQuantumProgramStatus.INVALID_INITIAL_STATE),
        status,
    )
    status = jnp.where(
        ~prepared.operations_valid,
        int(DenseQuantumProgramStatus.INVALID_OPERATION),
        status,
    )
    diagnostics = DenseQuantumProgramDiagnostics(
        status,
        initial_valid,
        prepared.operations_valid,
        final_finite,
        final_norm,
        final_trace,
        final_hermiticity,
        final_minimum,
        prepared.operation_evidence,
        positivity_audited,
    )
    return DenseQuantumProgramResult(
        candidate,
        diagnostics,
        prepared.numeric_version,
        prepared.prepared_id,
    )


__all__ = [
    "DenseQuantumOperationEvidence",
    "DenseQuantumProgramCostEstimate",
    "DenseQuantumProgramDiagnostics",
    "DenseQuantumProgramPlan",
    "DenseQuantumProgramPolicy",
    "DenseQuantumProgramResult",
    "DenseQuantumProgramStatus",
    "DensityPositivityAudit",
    "PreparedDenseQuantumProgram",
    "execute_dense_quantum_program",
    "plan_dense_quantum_program",
    "prepare_dense_quantum_program",
    "refresh_dense_quantum_program",
]
