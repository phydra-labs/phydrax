#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    solve,
)
from ._conversions import abcd_to_scattering, scattering_to_abcd
from ._ports import ElectricalWaveReference


class CalibrationStandard(StrictModule):
    scattering: Array
    standard_id: str = eqx.field(static=True)

    def __init__(self, scattering: ArrayLike, /, *, standard_id: str):
        value = jnp.asarray(scattering, dtype=jnp.complex128)
        if (
            value.ndim < 2
            or value.shape[-2] != value.shape[-1]
            or bool(jnp.any(~jnp.isfinite(value)))
        ):
            raise ValueError("Calibration standard scattering must be finite and square.")
        identifier = str(standard_id)
        if not identifier:
            raise ValueError("standard_id must be non-empty.")
        self.scattering, self.standard_id = value, identifier


class VNAErrorModel(StrictModule):
    left_abcd: Array
    right_abcd: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_abcd: ArrayLike,
        right_abcd: ArrayLike,
        /,
        *,
        model_id: str | None = None,
    ):
        left = jnp.asarray(left_abcd, dtype=jnp.complex128)
        right = jnp.asarray(right_abcd, dtype=jnp.complex128)
        if (
            left.shape[-2:] != (2, 2)
            or right.shape != left.shape
            or bool(jnp.any(~jnp.isfinite(left)))
            or bool(jnp.any(~jnp.isfinite(right)))
        ):
            raise ValueError("VNA error boxes must be aligned finite ABCD matrices.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "vna-error-model",
                    "shape": left.shape,
                }
            )
            if model_id is None
            else str(model_id)
        )
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.left_abcd, self.right_abcd, self.model_id = left, right, identifier


class DeembeddingEvidence(StrictModule):
    reconstruction_residual: Array
    relative_residual: Array
    left_status: Array
    right_status: Array
    finite: Array


class DeembeddingResult(StrictModule):
    scattering: Array
    abcd: Array
    evidence: DeembeddingEvidence
    model_id: str = eqx.field(static=True)


class IdentifiabilityReport(StrictModule):
    singular_values: Array
    effective_rank: Array
    condition_number: Array
    identifiable: Array
    jacobian: Array
    problem_id: str = eqx.field(static=True)


def _dense_solve(matrix: Array, right: Array, problem_id: str, /):
    return solve(
        LinearSystem(
            DenseLinearOperator(matrix, operator_id=f"{problem_id}/matrix"),
            problem_id=problem_id,
        ),
        right,
        policy=LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status")),
    )


def apply_vna_error_model(
    dut_scattering: ArrayLike,
    error_model: VNAErrorModel,
    references: tuple[ElectricalWaveReference, ElectricalWaveReference],
    /,
) -> Array:
    if not isinstance(error_model, VNAErrorModel):
        raise TypeError("error_model must be VNAErrorModel.")
    dut = scattering_to_abcd(dut_scattering, references).matrix
    measured = error_model.left_abcd @ dut @ error_model.right_abcd
    return abcd_to_scattering(measured, references).matrix


def deembed_two_port(
    measured_scattering: ArrayLike,
    error_model: VNAErrorModel,
    references: tuple[ElectricalWaveReference, ElectricalWaveReference],
    /,
) -> DeembeddingResult:
    if not isinstance(error_model, VNAErrorModel):
        raise TypeError("error_model must be VNAErrorModel.")
    measured = scattering_to_abcd(measured_scattering, references).matrix
    left = _dense_solve(
        error_model.left_abcd, measured, f"{error_model.model_id}/left-deembed"
    )
    right = _dense_solve(
        jnp.swapaxes(error_model.right_abcd, -1, -2),
        jnp.swapaxes(jnp.asarray(left.value), -1, -2),
        f"{error_model.model_id}/right-deembed",
    )
    dut = jnp.swapaxes(jnp.asarray(right.value), -1, -2)
    reconstructed = error_model.left_abcd @ dut @ error_model.right_abcd
    residual = reconstructed - measured
    norm = jnp.linalg.norm(residual, axis=(-2, -1))
    relative = norm / jnp.maximum(jnp.linalg.norm(measured, axis=(-2, -1)), 1.0)
    scattering = abcd_to_scattering(dut, references).matrix
    finite = jnp.all(jnp.isfinite(scattering), axis=(-2, -1))
    evidence = DeembeddingEvidence(
        norm,
        relative,
        jnp.asarray(left.status, dtype=jnp.int32),
        jnp.asarray(right.status, dtype=jnp.int32),
        finite
        & jnp.all(left.status == int(LinearSolveStatus.SUCCESS))
        & jnp.all(right.status == int(LinearSolveStatus.SUCCESS)),
    )
    return DeembeddingResult(scattering, dut, evidence, error_model.model_id)


def parameter_identifiability(
    prediction: Callable[[Array, Any], ArrayLike],
    parameters: ArrayLike,
    /,
    *,
    args: Any = None,
    relative_threshold: float = 1e-8,
    problem_id: str = "circuit-identifiability",
) -> IdentifiabilityReport:
    if not callable(prediction):
        raise TypeError("prediction must be callable.")
    value = jnp.asarray(parameters, dtype=float)
    threshold = float(relative_threshold)
    if value.ndim != 1 or value.size == 0 or threshold < 0.0:
        raise ValueError("Identifiability parameters or threshold are invalid.")
    output = jnp.asarray(prediction(value, args))
    if not jnp.issubdtype(output.dtype, jnp.number):
        raise TypeError("prediction must return numeric values.")

    def real_coordinates(current):
        predicted = jnp.asarray(prediction(current, args)).reshape((-1,))
        return jnp.concatenate((jnp.real(predicted), jnp.imag(predicted)))

    jacobian = jax.jacfwd(real_coordinates)(value)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    cutoff = threshold * jnp.maximum(singular_values[0], 1.0)
    rank = jnp.sum(singular_values > cutoff)
    smallest = jnp.where(rank > 0, singular_values[jnp.maximum(rank - 1, 0)], 0.0)
    condition = jnp.where(smallest > 0.0, singular_values[0] / smallest, jnp.inf)
    identifier = str(problem_id)
    if not identifier:
        raise ValueError("problem_id must be non-empty.")
    return IdentifiabilityReport(
        singular_values,
        rank,
        condition,
        rank == value.size,
        jacobian,
        identifier,
    )


__all__ = [
    "CalibrationStandard",
    "DeembeddingEvidence",
    "DeembeddingResult",
    "IdentifiabilityReport",
    "VNAErrorModel",
    "apply_vna_error_model",
    "deembed_two_port",
    "parameter_identifiability",
]
