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
from ..operators.quantum._observables import LocalObservable
from ..operators.quantum._register import HilbertRegisterLayout
from ._quantum_program import DenseQuantumProgramResult, PreparedDenseQuantumProgram


class DenseQuantumExpectationStatus(IntEnum):
    SUCCESS = 0
    INVALID_PROGRAM = 1
    INVALID_OBSERVABLE = 2
    NONFINITE_RESULT = 3
    NONREAL_RESULT = 4


class DenseQuantumObservablePolicy(StrictModule):
    """Resource and real-output envelope for dense local-observable evaluation."""

    maximum_observable_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    imaginary_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_observable_bytes: int = 2**30,
        maximum_workspace_bytes: int = 2**30,
        imaginary_tolerance: float = 1e-6,
    ):
        observable_limit = int(maximum_observable_bytes)
        workspace_limit = int(maximum_workspace_bytes)
        tolerance = float(imaginary_tolerance)
        if observable_limit <= 0 or workspace_limit <= 0:
            raise ValueError("Dense observable resource limits must be positive.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("imaginary_tolerance must be finite and nonnegative.")
        self.maximum_observable_bytes = observable_limit
        self.maximum_workspace_bytes = workspace_limit
        self.imaginary_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "dense-quantum-observable-policy",
                "maximum_observable_bytes": observable_limit,
                "maximum_workspace_bytes": workspace_limit,
                "imaginary_tolerance": tolerance,
            }
        )


class DenseQuantumObservableCostEstimate(StrictModule):
    observable_count: int = eqx.field(static=True)
    target_group_count: int = eqx.field(static=True)
    observable_bytes: int = eqx.field(static=True)
    workspace_bytes_per_case: int = eqx.field(static=True)
    output_bytes_per_case: int = eqx.field(static=True)


class DenseQuantumObservablePlan(StrictModule):
    layout: HilbertRegisterLayout
    prepared_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    state_kind: str = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    observables: tuple[LocalObservable, ...]
    target_groups: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    group_observable_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    policy: DenseQuantumObservablePolicy
    cost: DenseQuantumObservableCostEstimate
    plan_id: str = eqx.field(static=True)


class DenseQuantumExpectationDiagnostics(StrictModule):
    status: Array
    program_successful: Array
    observables_valid: Array
    finite: Array
    maximum_imaginary_residual: Array

    @property
    def successful(self) -> Array:
        return self.status == int(DenseQuantumExpectationStatus.SUCCESS)


class DenseQuantumExpectationResult(StrictModule):
    complex_values: Array
    imaginary_residuals: Array
    diagnostics: DenseQuantumExpectationDiagnostics
    numeric_version: Array
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def real_values(self) -> Array:
        values = jnp.real(self.complex_values)
        return eqx.error_if(
            values,
            ~jnp.all(self.diagnostics.successful),
            "Quantum expectations are not valid certified real values.",
        )


def plan_dense_quantum_observables(
    prepared: PreparedDenseQuantumProgram,
    observables: tuple[LocalObservable, ...] | list[LocalObservable],
    policy: DenseQuantumObservablePolicy | None = None,
    /,
) -> DenseQuantumObservablePlan:
    """Group ordered local observables and admit bounded dense evaluation."""
    if not isinstance(prepared, PreparedDenseQuantumProgram):
        raise TypeError("prepared must be PreparedDenseQuantumProgram.")
    selected = DenseQuantumObservablePolicy() if policy is None else policy
    if not isinstance(selected, DenseQuantumObservablePolicy):
        raise TypeError("policy must be DenseQuantumObservablePolicy or None.")
    values = tuple(observables)
    if not values:
        raise ValueError("At least one local observable is required.")
    groups: dict[tuple[str, ...], list[int]] = {}
    observable_bytes = 0
    maximum_target_dimension = 0
    itemsize = jnp.dtype(prepared.plan.policy.compute_dtype).itemsize
    for index, observable in enumerate(values):
        if not isinstance(observable, LocalObservable):
            raise TypeError("observables must contain only LocalObservable values.")
        target_dimension = prepared.plan.layout.target_dimension(
            observable.target_wire_ids
        )
        if observable.matrix.shape != (target_dimension, target_dimension):
            raise ValueError(
                "Observable matrix dimension does not match its ordered targets."
            )
        if str(observable.matrix.dtype) != prepared.plan.policy.compute_dtype:
            raise TypeError("Observable and dense-program dtypes must match exactly.")
        groups.setdefault(observable.target_wire_ids, []).append(index)
        observable_bytes += observable.matrix.size * itemsize
        maximum_target_dimension = max(maximum_target_dimension, target_dimension)
    output_bytes = len(values) * itemsize
    workspace_bytes = maximum_target_dimension**2 * itemsize + output_bytes
    if observable_bytes > selected.maximum_observable_bytes:
        raise MemoryError("Local observables exceed maximum_observable_bytes.")
    if workspace_bytes > selected.maximum_workspace_bytes:
        raise MemoryError("Observable workspace exceeds maximum_workspace_bytes.")
    target_groups = tuple(groups)
    group_indices = tuple(tuple(groups[target]) for target in target_groups)
    cost = DenseQuantumObservableCostEstimate(
        len(values),
        len(target_groups),
        observable_bytes,
        workspace_bytes,
        output_bytes,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "dense-quantum-observable-plan",
            "prepared": prepared.prepared_id,
            "observables": [observable.observable_id for observable in values],
            "policy": selected.policy_id,
        }
    )
    return DenseQuantumObservablePlan(
        prepared.plan.layout,
        prepared.prepared_id,
        prepared.plan.layout.layout_id,
        prepared.plan.state_kind,
        prepared.plan.policy.compute_dtype,
        values,
        target_groups,
        group_indices,
        selected,
        cost,
        plan_id,
    )


def _state_reduced_density(
    state: Array,
    local_dimensions: tuple[int, ...],
    target_indices: tuple[int, ...],
    /,
) -> Array:
    batch_shape = state.shape[:-1]
    batch_ndim = len(batch_shape)
    remaining = tuple(
        index for index in range(len(local_dimensions)) if index not in target_indices
    )
    permutation = (
        tuple(range(batch_ndim))
        + tuple(batch_ndim + index for index in remaining)
        + tuple(batch_ndim + index for index in target_indices)
    )
    ordered = jnp.transpose(state.reshape(batch_shape + local_dimensions), permutation)
    target_dimension = prod(local_dimensions[index] for index in target_indices)
    grouped = ordered.reshape(batch_shape + (-1, target_dimension))
    return oe.contract("...ri,...rj->...ij", grouped, jnp.conj(grouped))


def _density_reduced_density(
    density: Array,
    local_dimensions: tuple[int, ...],
    target_indices: tuple[int, ...],
    /,
) -> Array:
    batch_shape = density.shape[:-2]
    batch_ndim = len(batch_shape)
    wire_count = len(local_dimensions)
    batch_labels = list(range(batch_ndim))
    next_label = batch_ndim
    ket_labels = list(range(next_label, next_label + wire_count))
    next_label += wire_count
    bra_labels: list[int] = []
    for index in range(wire_count):
        if index in target_indices:
            bra_labels.append(next_label)
            next_label += 1
        else:
            bra_labels.append(ket_labels[index])
    output_labels = (
        batch_labels
        + [ket_labels[index] for index in target_indices]
        + [bra_labels[index] for index in target_indices]
    )
    tensor = density.reshape(batch_shape + local_dimensions + local_dimensions)
    reduced = oe.contract(
        tensor,
        batch_labels + ket_labels + bra_labels,
        output_labels,
    )
    target_dimension = prod(local_dimensions[index] for index in target_indices)
    return reduced.reshape(batch_shape + (target_dimension, target_dimension))


def evaluate_dense_quantum_observables(
    plan: DenseQuantumObservablePlan,
    result: DenseQuantumProgramResult,
    /,
) -> DenseQuantumExpectationResult:
    """Evaluate a prepared ordered observable plan on one matching program result."""
    if not isinstance(plan, DenseQuantumObservablePlan):
        raise TypeError("plan must be DenseQuantumObservablePlan.")
    if not isinstance(result, DenseQuantumProgramResult):
        raise TypeError("result must be DenseQuantumProgramResult.")
    if result.prepared_id != plan.prepared_id:
        raise ValueError("Program result and observable plan prepared IDs must match.")
    batch_shape = (
        result.final_state.shape[:-1]
        if plan.state_kind == "state-vector"
        else result.final_state.shape[:-2]
    )
    cases = prod(batch_shape) if batch_shape else 1
    if cases * plan.cost.workspace_bytes_per_case > plan.policy.maximum_workspace_bytes:
        raise MemoryError("Batched observable workspace exceeds maximum_workspace_bytes.")
    outputs = [
        jnp.zeros(batch_shape, dtype=jnp.dtype(plan.compute_dtype))
        for _ in range(plan.cost.observable_count)
    ]
    for targets, indices in zip(
        plan.target_groups, plan.group_observable_indices, strict=True
    ):
        target_indices = plan.layout.target_indices(targets)
        reduced = (
            _state_reduced_density(
                result.final_state,
                plan.layout.local_dimensions,
                target_indices,
            )
            if plan.state_kind == "state-vector"
            else _density_reduced_density(
                result.final_state,
                plan.layout.local_dimensions,
                target_indices,
            )
        )
        matrices = jnp.stack(
            [plan.observables[index].matrix for index in indices],
            axis=0,
        )
        group_values = oe.contract("...ij,kji->...k", reduced, matrices)
        for local_index, output_index in enumerate(indices):
            outputs[output_index] = group_values[..., local_index]
    values = jnp.stack(outputs, axis=-1)
    imaginary_residuals = jnp.abs(jnp.imag(values))
    finite = jnp.all(jnp.isfinite(values), axis=-1)
    maximum_imaginary = jnp.max(imaginary_residuals, axis=-1)
    observables_valid = jnp.all(
        jnp.stack([observable.valid for observable in plan.observables])
    )
    program_successful = result.diagnostics.successful
    status = jnp.full(
        batch_shape,
        int(DenseQuantumExpectationStatus.SUCCESS),
        dtype=jnp.int32,
    )
    status = jnp.where(
        maximum_imaginary > plan.policy.imaginary_tolerance,
        int(DenseQuantumExpectationStatus.NONREAL_RESULT),
        status,
    )
    status = jnp.where(
        ~finite,
        int(DenseQuantumExpectationStatus.NONFINITE_RESULT),
        status,
    )
    status = jnp.where(
        ~observables_valid,
        int(DenseQuantumExpectationStatus.INVALID_OBSERVABLE),
        status,
    )
    status = jnp.where(
        ~program_successful,
        int(DenseQuantumExpectationStatus.INVALID_PROGRAM),
        status,
    )
    diagnostics = DenseQuantumExpectationDiagnostics(
        status,
        program_successful,
        observables_valid,
        finite,
        maximum_imaginary,
    )
    return DenseQuantumExpectationResult(
        values,
        imaginary_residuals,
        diagnostics,
        result.numeric_version,
        plan.plan_id,
        plan.prepared_id,
    )


__all__ = [
    "DenseQuantumExpectationDiagnostics",
    "DenseQuantumExpectationResult",
    "DenseQuantumExpectationStatus",
    "DenseQuantumObservableCostEstimate",
    "DenseQuantumObservablePlan",
    "DenseQuantumObservablePolicy",
    "evaluate_dense_quantum_observables",
    "plan_dense_quantum_observables",
]
