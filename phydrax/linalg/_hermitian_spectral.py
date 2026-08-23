#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._hermitian_precision import HermitianPrecisionPolicy


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


class HermitianSpectrum(StrictModule):
    """Hermitian eigendecomposition and rank/conditioning evidence."""

    matrix: Array
    eigenvalues: Array
    eigenvectors: Array
    hermiticity_residual: Array
    minimum_eigenvalue: Array
    minimum_gap: Array
    numerical_rank: Array
    condition_number: Array
    valid: Array
    precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
        precision: HermitianPrecisionPolicy | None = None,
    ):
        original = jnp.asarray(matrix)
        precision_ = HermitianPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, HermitianPrecisionPolicy):
            raise TypeError("precision must be a HermitianPrecisionPolicy or None.")
        value = precision_.compute(original)
        if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
            raise ValueError("Hermitian spectrum requires square trailing matrix axes.")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")
        hermitian = precision_.factorization(0.5 * (value + _adjoint(value)))
        eigenvalues, eigenvectors = jnp.linalg.eigh(hermitian)
        differences = jnp.abs(eigenvalues[..., 1:] - eigenvalues[..., :-1])
        minimum_gap = (
            jnp.min(differences, axis=-1)
            if value.shape[-1] > 1
            else jnp.full(value.shape[:-2], jnp.inf, dtype=eigenvalues.dtype)
        )
        magnitude = precision_.decision(jnp.max(jnp.abs(eigenvalues), axis=-1))
        threshold = precision_.decision(tolerance) * jnp.maximum(magnitude, 1.0)
        rank = jnp.sum(jnp.abs(eigenvalues) > threshold[..., None], axis=-1)
        minimum_absolute = precision_.decision(jnp.min(jnp.abs(eigenvalues), axis=-1))
        condition = magnitude / jnp.maximum(
            minimum_absolute, jnp.finfo(eigenvalues.dtype).tiny
        )
        residual = precision_.decision(
            jnp.max(
                jnp.abs(precision_.accumulation(value - _adjoint(value))),
                axis=(-2, -1),
            )
        )
        self.matrix = hermitian
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.hermiticity_residual = residual
        self.minimum_eigenvalue = precision_.decision(jnp.min(eigenvalues, axis=-1))
        self.minimum_gap = precision_.decision(minimum_gap)
        self.numerical_rank = rank
        self.condition_number = precision_.decision(condition)
        self.precision = precision_
        self.precision_evidence = precision_.evidence_for(original)
        self.valid = (
            jnp.all(jnp.isfinite(value), axis=(-2, -1))
            & (residual <= tolerance)
            & jnp.all(jnp.isfinite(eigenvalues), axis=-1)
        )
        self.tolerance = float(tolerance)

    def reconstruct(self) -> Array:
        return (self.eigenvectors * self.eigenvalues[..., None, :]) @ _adjoint(
            self.eigenvectors
        )


class HermitianFunctionResult(StrictModule):
    value: Array
    spectrum: HermitianSpectrum
    valid: Array
    function_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        spectrum: HermitianSpectrum,
        /,
        *,
        function_id: str,
        valid: ArrayLike,
    ):
        self.value = jnp.asarray(value)
        self.spectrum = spectrum
        self.valid = jnp.asarray(valid, dtype=bool)
        self.function_id = str(function_id)


def _spectral_result(
    matrix: ArrayLike,
    function,
    /,
    *,
    function_id: str,
    tolerance: float,
    positive: bool,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    spectrum = HermitianSpectrum(
        matrix,
        tolerance=tolerance,
        precision=precision,
    )
    transformed = function(spectrum.eigenvalues)
    value = (spectrum.eigenvectors * transformed[..., None, :]) @ _adjoint(
        spectrum.eigenvectors
    )
    valid = spectrum.valid & jnp.all(jnp.isfinite(transformed), axis=-1)
    if positive:
        valid = valid & (spectrum.minimum_eigenvalue > tolerance)
    return HermitianFunctionResult(
        spectrum.precision.output(0.5 * (value + _adjoint(value))),
        spectrum,
        function_id=function_id,
        valid=valid,
    )


def hermitian_sqrt(
    matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    return _spectral_result(
        matrix,
        lambda values: jnp.sqrt(jnp.maximum(values, 0.0)),
        function_id="hermitian-sqrt",
        tolerance=tolerance,
        positive=False,
        precision=precision,
    )


def hermitian_inverse_sqrt(
    matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    return _spectral_result(
        matrix,
        lambda values: 1.0 / jnp.sqrt(values),
        function_id="hermitian-inverse-sqrt",
        tolerance=tolerance,
        positive=True,
        precision=precision,
    )


def hermitian_log(
    matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    return _spectral_result(
        matrix,
        jnp.log,
        function_id="hermitian-log",
        tolerance=tolerance,
        positive=True,
        precision=precision,
    )


def hermitian_exp(
    matrix: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> HermitianFunctionResult:
    return _spectral_result(
        matrix,
        jnp.exp,
        function_id="hermitian-exp",
        tolerance=tolerance,
        positive=False,
        precision=precision,
    )


class SylvesterSolveResult(StrictModule):
    value: Array
    residual_norm: Array
    minimum_denominator: Array
    valid: Array
    precision_evidence: PrecisionEvidenceEnvelope

    def __init__(
        self,
        value: ArrayLike,
        residual_norm: ArrayLike,
        minimum_denominator: ArrayLike,
        valid: ArrayLike,
        precision_evidence: PrecisionEvidenceEnvelope,
        /,
    ):
        self.value = jnp.asarray(value)
        self.residual_norm = jnp.asarray(residual_norm)
        self.minimum_denominator = jnp.asarray(minimum_denominator)
        self.valid = jnp.asarray(valid, dtype=bool)
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        self.precision_evidence = precision_evidence


class HermitianSylvesterOperator(StrictModule):
    """Matrix-free ``X -> rho X + X rho`` and spectral inverse action."""

    matrix: Array
    spectrum: HermitianSpectrum
    tolerance: float = eqx.field(static=True)
    precision: HermitianPrecisionPolicy

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
        precision: HermitianPrecisionPolicy | None = None,
    ):
        precision_ = HermitianPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, HermitianPrecisionPolicy):
            raise TypeError("precision must be a HermitianPrecisionPolicy or None.")
        spectrum = HermitianSpectrum(
            matrix,
            tolerance=tolerance,
            precision=precision_,
        )
        self.matrix = spectrum.reconstruct()
        self.spectrum = spectrum
        self.precision = precision_
        self.tolerance = float(tolerance)

    def mv(self, value: ArrayLike, /) -> Array:
        operand = jnp.asarray(value)
        if operand.shape != self.matrix.shape:
            raise ValueError("Sylvester operand must match the matrix shape.")
        return self.matrix @ operand + operand @ self.matrix

    def solve(self, right_hand_side: ArrayLike, /) -> SylvesterSolveResult:
        right = self.precision.factorization(right_hand_side)
        if right.shape != self.matrix.shape:
            raise ValueError("Sylvester right-hand side must match the matrix shape.")
        vectors = self.spectrum.eigenvectors
        local = _adjoint(vectors) @ right @ vectors
        denominators = (
            self.spectrum.eigenvalues[..., :, None]
            + self.spectrum.eigenvalues[..., None, :]
        )
        minimum = jnp.min(jnp.abs(denominators), axis=(-2, -1))
        safe = jnp.where(jnp.abs(denominators) > self.tolerance, denominators, jnp.inf)
        solution = self.precision.output(vectors @ (local / safe) @ _adjoint(vectors))
        residual = self.precision.accumulation(self.mv(solution) - right)
        residual_norm = self.precision.decision(jnp.linalg.norm(residual, axis=(-2, -1)))
        valid = (
            self.spectrum.valid & (minimum > self.tolerance) & jnp.isfinite(residual_norm)
        )
        return SylvesterSolveResult(
            solution,
            residual_norm,
            self.precision.decision(minimum),
            valid,
            self.precision.evidence_for(right),
        )


class TracelessHermitianSpace(StrictModule):
    dimension: int = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        dimension_ = int(dimension)
        if dimension_ < 2:
            raise ValueError("Density tangent dimension must be at least two.")
        self.dimension = dimension_
        self.space_id = f"traceless-hermitian:{dimension_}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.dimension, self.dimension

    def project(self, value: ArrayLike, /) -> Array:
        matrix = jnp.asarray(value)
        if matrix.shape != self.shape:
            raise ValueError(f"Matrix must have shape {self.shape}.")
        hermitian = 0.5 * (matrix + _adjoint(matrix))
        trace = jnp.trace(hermitian) / float(self.dimension)
        return hermitian - trace * jnp.eye(self.dimension, dtype=hermitian.dtype)

    def inner(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return jnp.real(jnp.vdot(self.project(left), self.project(right)))


__all__ = [
    "HermitianFunctionResult",
    "HermitianSpectrum",
    "HermitianSylvesterOperator",
    "SylvesterSolveResult",
    "TracelessHermitianSpace",
    "hermitian_exp",
    "hermitian_inverse_sqrt",
    "hermitian_log",
    "hermitian_sqrt",
]
