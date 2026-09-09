#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule


SpectrumReplacement: TypeAlias = Literal["bulk-mean", "upper-edge", "hard-floor"]


class MarchenkoPasturDiagnostics(StrictModule):
    """Finite-sample spectral evidence relative to Marchenko--Pastur edges."""

    eigenvalues: Array
    noise_eigenvalue_mask: Array
    signal_eigenvalue_mask: Array
    aspect_ratio: Array
    noise_variance: Array
    lower_edge: Array
    upper_edge: Array
    effective_rank: Array
    condition_number: Array
    trace: Array
    feature_count: int = eqx.field(static=True)
    observation_count: int = eqx.field(static=True)


class RandomMatrixCleaningResult(StrictModule):
    """PSD covariance cleaning with trace and conditioning evidence retained."""

    covariance: Array
    raw_covariance: Array
    raw_eigenvalues: Array
    cleaned_eigenvalues: Array
    eigenvectors: Array
    diagnostics: MarchenkoPasturDiagnostics
    raw_trace: Array
    cleaned_trace: Array
    trace_error: Array
    minimum_eigenvalue: Array
    condition_number: Array
    effective_rank: Array
    psd: Array
    finite: Array
    replacement: SpectrumReplacement = eqx.field(static=True)
    preserve_trace: bool = eqx.field(static=True)


def _validate_covariance(covariance: ArrayLike) -> Array:
    matrix = jnp.asarray(covariance)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("covariance must have shape (..., features, features).")
    if matrix.shape[-1] < 1:
        raise ValueError("covariance must contain at least one feature.")
    if not jnp.issubdtype(matrix.dtype, jnp.inexact):
        matrix = matrix.astype(float)
    matrix = eqx.error_if(
        matrix,
        jnp.any(~jnp.isfinite(matrix)),
        "covariance entries must be finite.",
    )
    hermitian_error = jnp.max(
        jnp.abs(matrix - jnp.conj(jnp.swapaxes(matrix, -1, -2))), axis=(-2, -1)
    )
    tolerance = (
        64.0
        * jnp.finfo(matrix.real.dtype).eps
        * jnp.maximum(jnp.max(jnp.abs(matrix), axis=(-2, -1)), 1.0)
    )
    matrix = eqx.error_if(
        matrix,
        jnp.any(hermitian_error > tolerance),
        "covariance must be Hermitian within numerical tolerance.",
    )
    return 0.5 * (matrix + jnp.conj(jnp.swapaxes(matrix, -1, -2)))


def marchenko_pastur_diagnostics(
    covariance: ArrayLike,
    observation_count: int,
    /,
    *,
    noise_variance: ArrayLike | None = None,
    edge_tolerance: float = 0.0,
) -> MarchenkoPasturDiagnostics:
    """Diagnose a sample covariance against its null random-matrix bulk."""

    matrix = _validate_covariance(covariance)
    count = int(observation_count)
    if count < 2:
        raise ValueError("observation_count must be at least two.")
    tolerance = float(edge_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("edge_tolerance must be finite and nonnegative.")
    features = int(matrix.shape[-1])
    values = jnp.linalg.eigvalsh(matrix)
    values = eqx.error_if(
        values,
        jnp.any(values < -64.0 * jnp.finfo(values.dtype).eps),
        "covariance must be positive semidefinite.",
    )
    values = jnp.maximum(values, 0.0)
    trace = jnp.sum(values, axis=-1)
    if noise_variance is None:
        variance = trace / features
    else:
        variance = jnp.asarray(noise_variance, dtype=values.dtype)
        if variance.shape not in ((), values.shape[:-1]):
            raise ValueError("noise_variance must be scalar or match covariance cases.")
        variance = eqx.error_if(
            variance,
            jnp.any(~jnp.isfinite(variance) | (variance <= 0.0)),
            "noise_variance must be finite and positive.",
        )
        variance = jnp.broadcast_to(variance, values.shape[:-1])
    ratio = jnp.asarray(features / count, dtype=values.dtype)
    root = jnp.sqrt(ratio)
    lower = variance * jnp.square(jnp.maximum(1.0 - root, 0.0))
    upper = variance * jnp.square(1.0 + root)
    upper_limit = upper * (1.0 + tolerance)
    noise = values <= upper_limit[..., None]
    signal = ~noise
    scale = jnp.maximum(values[..., -1], jnp.finfo(values.dtype).tiny)
    numerical_floor = jnp.finfo(values.dtype).eps * scale
    positive = values > numerical_floor[..., None]
    rank = jnp.sum(positive, axis=-1).astype(jnp.int32)
    smallest = jnp.min(jnp.where(positive, values, jnp.inf), axis=-1)
    condition = jnp.where(rank == features, values[..., -1] / smallest, jnp.inf)
    return MarchenkoPasturDiagnostics(
        eigenvalues=values,
        noise_eigenvalue_mask=noise,
        signal_eigenvalue_mask=signal,
        aspect_ratio=jnp.broadcast_to(ratio, values.shape[:-1]),
        noise_variance=variance,
        lower_edge=lower,
        upper_edge=upper,
        effective_rank=rank,
        condition_number=condition,
        trace=trace,
        feature_count=features,
        observation_count=count,
    )


def clean_covariance_spectrum(
    covariance: ArrayLike,
    observation_count: int,
    /,
    *,
    noise_variance: ArrayLike | None = None,
    replacement: SpectrumReplacement = "bulk-mean",
    preserve_trace: bool = True,
    eigenvalue_floor: float = 0.0,
    edge_tolerance: float = 0.0,
) -> RandomMatrixCleaningResult:
    """Replace null-bulk eigenvalues and reconstruct a Hermitian PSD covariance."""

    if replacement not in ("bulk-mean", "upper-edge", "hard-floor"):
        raise ValueError(
            "replacement must be 'bulk-mean', 'upper-edge', or 'hard-floor'."
        )
    floor = float(eigenvalue_floor)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("eigenvalue_floor must be finite and nonnegative.")
    matrix = _validate_covariance(covariance)
    diagnostics = marchenko_pastur_diagnostics(
        matrix,
        observation_count,
        noise_variance=noise_variance,
        edge_tolerance=edge_tolerance,
    )
    raw_values, vectors = jnp.linalg.eigh(matrix)
    raw_values = jnp.maximum(raw_values, 0.0)
    noise = diagnostics.noise_eigenvalue_mask
    noise_count = jnp.sum(noise, axis=-1)
    noise_sum = jnp.sum(jnp.where(noise, raw_values, 0.0), axis=-1)
    noise_mean = noise_sum / jnp.maximum(noise_count, 1)
    if replacement == "bulk-mean":
        replacement_value = noise_mean
    elif replacement == "upper-edge":
        replacement_value = diagnostics.upper_edge
    else:
        replacement_value = jnp.asarray(floor, dtype=raw_values.dtype)
    cleaned = jnp.where(noise, replacement_value[..., None], raw_values)
    scale = jnp.maximum(raw_values[..., -1], 1.0)
    numerical_floor = jnp.maximum(
        jnp.asarray(floor, dtype=raw_values.dtype),
        jnp.finfo(raw_values.dtype).eps * scale,
    )
    cleaned = jnp.maximum(cleaned, numerical_floor[..., None])
    raw_trace = jnp.sum(raw_values, axis=-1)
    if preserve_trace:
        cleaned_trace_before = jnp.sum(cleaned, axis=-1)
        factor = raw_trace / jnp.maximum(
            cleaned_trace_before, jnp.finfo(raw_values.dtype).tiny
        )
        cleaned = cleaned * factor[..., None]
    cleaned_matrix = ein.contract(
        "...ik,...k,...jk->...ij", vectors, cleaned, jnp.conj(vectors)
    )
    cleaned_matrix = 0.5 * (
        cleaned_matrix + jnp.conj(jnp.swapaxes(cleaned_matrix, -1, -2))
    )
    cleaned_trace = jnp.real(jnp.trace(cleaned_matrix, axis1=-2, axis2=-1))
    minimum = jnp.min(cleaned, axis=-1)
    maximum = jnp.max(cleaned, axis=-1)
    condition = maximum / jnp.maximum(minimum, jnp.finfo(cleaned.dtype).tiny)
    effective_rank = jnp.sum(
        cleaned > jnp.finfo(cleaned.dtype).eps * maximum[..., None], axis=-1
    ).astype(jnp.int32)
    finite = jnp.all(jnp.isfinite(cleaned_matrix), axis=(-2, -1))
    psd = minimum >= -64.0 * jnp.finfo(cleaned.dtype).eps * jnp.maximum(maximum, 1.0)
    return RandomMatrixCleaningResult(
        covariance=cleaned_matrix,
        raw_covariance=matrix,
        raw_eigenvalues=raw_values,
        cleaned_eigenvalues=cleaned,
        eigenvectors=vectors,
        diagnostics=diagnostics,
        raw_trace=raw_trace,
        cleaned_trace=cleaned_trace,
        trace_error=cleaned_trace - raw_trace,
        minimum_eigenvalue=minimum,
        condition_number=condition,
        effective_rank=effective_rank,
        psd=psd,
        finite=finite,
        replacement=replacement,
        preserve_trace=bool(preserve_trace),
    )


__all__ = [
    "MarchenkoPasturDiagnostics",
    "RandomMatrixCleaningResult",
    "SpectrumReplacement",
    "clean_covariance_spectrum",
    "marchenko_pastur_diagnostics",
]
