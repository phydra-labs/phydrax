#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ._channels import FiniteCPTPMap
from ._propagation import unitarity_residual
from ._subspaces import (
    BasisStateSubspace,
    DenseQuantumSubspace,
    project_quantum_operator,
    QuantumSubspace,
)


class GateQualityDiagnostics(StrictModule):
    """Independent target, subspace, and effective-operation evidence."""

    target_unitarity_residual: Array
    effective_isometry_residual: Array
    effective_isometry_available: Array
    input_subspace_valid: Array
    output_subspace_valid: Array
    source_operation_valid: Array
    metric_bounds_valid: Array
    finite: Array
    valid: Array


class GateQualityResult(StrictModule):
    """Leakage-sensitive logical gate quality without hidden compensation."""

    effective_operation: Array
    survival: Array
    leakage: Array
    average_fidelity: Array
    conditional_fidelity: Array
    conditional_fidelity_valid: Array
    diagnostics: GateQualityDiagnostics
    representation: str = eqx.field(static=True)
    logical_dimension: int = eqx.field(static=True)


class CoherentPauliExpansion(StrictModule):
    """Complex Pauli coefficients for a coherent qubit-space operator."""

    coefficients: Array
    weights: Array
    reconstruction_residual: Array
    finite: Array
    valid: Array
    qubit_count: int = eqx.field(static=True)


def _subspace_valid(subspace: QuantumSubspace, /) -> Array:
    if isinstance(subspace, BasisStateSubspace):
        return subspace.evidence.valid
    if isinstance(subspace, DenseQuantumSubspace):
        return subspace.evidence.valid
    raise TypeError("subspace must be a BasisStateSubspace or DenseQuantumSubspace.")


def _dense_isometry(subspace: QuantumSubspace, dtype, /) -> Array:
    if isinstance(subspace, DenseQuantumSubspace):
        return subspace.isometry.astype(dtype)
    if isinstance(subspace, BasisStateSubspace):
        return jnp.eye(subspace.physical_dimension, dtype=dtype)[
            :, subspace.basis_indices
        ]
    raise TypeError("subspace must be a BasisStateSubspace or DenseQuantumSubspace.")


def _target(
    target: ArrayLike, dimension: int, tolerance: float, /
) -> tuple[Array, Array]:
    value = jnp.asarray(target)
    if value.shape != (dimension, dimension):
        raise ValueError(f"target must have shape {(dimension, dimension)}.")
    if not jnp.issubdtype(value.dtype, jnp.complexfloating):
        value = value.astype(jnp.result_type(value.dtype, 1j))
    residual = unitarity_residual(value)
    return value, residual <= tolerance


def _quality_from_choi(
    effective_choi: Array,
    target: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    dimension = target.shape[0]
    choi4 = effective_choi.reshape(dimension, dimension, dimension, dimension)
    trace_sum = jnp.real(jnp.trace(effective_choi))
    target_overlap = jnp.real(
        ein.contract("ol,olpm,pm->", jnp.conj(target), choi4, target)
    )
    survival = trace_sum / dimension
    average = (trace_sum + target_overlap) / (dimension * (dimension + 1))
    conditional_valid = survival > jnp.finfo(survival.dtype).tiny
    conditional = jnp.where(conditional_valid, average / survival, jnp.nan)
    return survival, 1.0 - survival, average, conditional, conditional_valid


def _metric_bounds(
    survival: Array,
    average: Array,
    conditional: Array,
    conditional_valid: Array,
    tolerance: float,
    /,
) -> Array:
    return (
        (survival >= -tolerance)
        & (survival <= 1.0 + tolerance)
        & (average >= -tolerance)
        & (average <= 1.0 + tolerance)
        & (
            ~conditional_valid
            | ((conditional >= -tolerance) & (conditional <= 1.0 + tolerance))
        )
    )


def unitary_gate_quality(
    unitary: ArrayLike,
    target: ArrayLike,
    input_subspace: QuantumSubspace,
    output_subspace: QuantumSubspace | None = None,
    /,
    *,
    tolerance: float = 1e-8,
) -> GateQualityResult:
    """Evaluate one physical unitary on explicit logical input/output subspaces."""

    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    output = input_subspace if output_subspace is None else output_subspace
    if input_subspace.logical_dimension != output.logical_dimension:
        raise ValueError("Input and output logical dimensions must match.")
    value = jnp.asarray(unitary)
    expected = (output.physical_dimension, input_subspace.physical_dimension)
    if input_subspace.physical_dimension != output.physical_dimension:
        raise ValueError(
            "A unitary gate requires equal physical input/output dimensions."
        )
    if value.shape != expected:
        raise ValueError(f"unitary must have shape {expected}.")
    dimension = input_subspace.logical_dimension
    target_, target_valid = _target(target, dimension, tolerance_)
    effective = project_quantum_operator(value, input_subspace, output)
    effective_choi = jnp.outer(
        effective.reshape((-1,)), jnp.conj(effective.reshape((-1,)))
    )
    survival, leakage, average, conditional, conditional_valid = _quality_from_choi(
        effective_choi,
        target_,
    )
    isometry_residual = unitarity_residual(effective)
    input_valid = _subspace_valid(input_subspace)
    output_valid = _subspace_valid(output)
    source_valid = unitarity_residual(value) <= tolerance_
    finite = (
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(target_))
        & jnp.all(jnp.isfinite(effective))
        & jnp.isfinite(survival)
        & jnp.isfinite(average)
    )
    metric_bounds_valid = _metric_bounds(
        survival,
        average,
        conditional,
        conditional_valid,
        tolerance_,
    )
    valid = (
        finite
        & target_valid
        & input_valid
        & output_valid
        & source_valid
        & metric_bounds_valid
    )
    diagnostics = GateQualityDiagnostics(
        unitarity_residual(target_),
        isometry_residual,
        jnp.asarray(True),
        input_valid,
        output_valid,
        source_valid,
        metric_bounds_valid,
        finite,
        valid,
    )
    return GateQualityResult(
        effective,
        survival,
        leakage,
        average,
        conditional,
        conditional_valid,
        diagnostics,
        "effective-operator",
        dimension,
    )


def finite_channel_gate_quality(
    channel: FiniteCPTPMap,
    target: ArrayLike,
    input_subspace: QuantumSubspace,
    output_subspace: QuantumSubspace | None = None,
    /,
    *,
    tolerance: float = 1e-8,
) -> GateQualityResult:
    """Evaluate a physical CPTP map after leakage-producing subspace projection."""

    if not isinstance(channel, FiniteCPTPMap):
        raise TypeError("channel must be a FiniteCPTPMap.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    output = input_subspace if output_subspace is None else output_subspace
    if input_subspace.logical_dimension != output.logical_dimension:
        raise ValueError("Input and output logical dimensions must match.")
    if channel.input_dimension != input_subspace.physical_dimension:
        raise ValueError("Channel input dimension does not match input_subspace.")
    if channel.output_dimension != output.physical_dimension:
        raise ValueError("Channel output dimension does not match output_subspace.")
    dimension = input_subspace.logical_dimension
    target_, target_valid = _target(target, dimension, tolerance_)
    dtype = jnp.result_type(channel.choi_matrix.dtype, target_.dtype)
    input_isometry = _dense_isometry(input_subspace, dtype)
    output_isometry = _dense_isometry(output, dtype)
    physical_choi = channel.choi_matrix.reshape(
        channel.output_dimension,
        channel.input_dimension,
        channel.output_dimension,
        channel.input_dimension,
    )
    effective4 = ein.contract(
        "ao,il,aibj,bp,jm->olpm",
        jnp.conj(output_isometry),
        input_isometry,
        physical_choi,
        output_isometry,
        jnp.conj(input_isometry),
    )
    effective_choi = effective4.reshape(dimension * dimension, dimension * dimension)
    survival, leakage, average, conditional, conditional_valid = _quality_from_choi(
        effective_choi,
        target_,
    )
    input_valid = _subspace_valid(input_subspace)
    output_valid = _subspace_valid(output)
    finite = (
        jnp.all(jnp.isfinite(effective_choi))
        & jnp.isfinite(survival)
        & jnp.isfinite(average)
    )
    metric_bounds_valid = _metric_bounds(
        survival,
        average,
        conditional,
        conditional_valid,
        tolerance_,
    )
    valid = (
        finite
        & target_valid
        & input_valid
        & output_valid
        & channel.valid
        & metric_bounds_valid
    )
    diagnostics = GateQualityDiagnostics(
        unitarity_residual(target_),
        jnp.asarray(jnp.nan, dtype=survival.dtype),
        jnp.asarray(False),
        input_valid,
        output_valid,
        channel.valid,
        metric_bounds_valid,
        finite,
        valid,
    )
    return GateQualityResult(
        effective_choi,
        survival,
        leakage,
        average,
        conditional,
        conditional_valid,
        diagnostics,
        "effective-choi",
        dimension,
    )


def coherent_pauli_expansion(
    operator: ArrayLike,
    /,
    *,
    maximum_qubits: int = 6,
    tolerance: float = 1e-8,
) -> CoherentPauliExpansion:
    """Expand a coherent qubit-space operator without stochastic interpretation."""

    if isinstance(maximum_qubits, bool) or not isinstance(maximum_qubits, Integral):
        raise TypeError("maximum_qubits must be a positive integer.")
    if int(maximum_qubits) <= 0:
        raise ValueError("maximum_qubits must be positive.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    value = jnp.asarray(operator)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError("operator must be one nonempty square matrix.")
    dimension = int(value.shape[0])
    qubits = dimension.bit_length() - 1
    if 2**qubits != dimension:
        raise ValueError("operator dimension must be a power of two.")
    if qubits > int(maximum_qubits):
        raise ValueError("operator exceeds maximum_qubits.")
    if not jnp.issubdtype(value.dtype, jnp.complexfloating):
        value = value.astype(jnp.result_type(value.dtype, 1j))
    one_qubit = (
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=value.dtype),
        jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=value.dtype),
        jnp.asarray([[0.0, -1j], [1j, 0.0]], dtype=value.dtype),
        jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=value.dtype),
    )
    basis = [jnp.asarray([[1.0]], dtype=value.dtype)]
    for _ in range(qubits):
        basis = [jnp.kron(left, right) for left in basis for right in one_qubit]
    basis_array = jnp.stack(basis)
    coefficients = (
        ein.contract(
            "kab,ab->k",
            jnp.conj(basis_array),
            value,
        )
        / dimension
    )
    reconstruction = ein.contract("k,kab->ab", coefficients, basis_array)
    residual = jnp.max(jnp.abs(reconstruction - value))
    weights = jnp.real(coefficients * jnp.conj(coefficients))
    finite = jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(residual)
    valid = finite & (residual <= tolerance_)
    return CoherentPauliExpansion(
        coefficients,
        weights,
        residual,
        finite,
        valid,
        qubits,
    )


__all__ = [
    "CoherentPauliExpansion",
    "GateQualityDiagnostics",
    "GateQualityResult",
    "coherent_pauli_expansion",
    "finite_channel_gate_quality",
    "unitary_gate_quality",
]
