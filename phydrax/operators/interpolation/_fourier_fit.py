#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, prod, sqrt
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from nufftax import (
    nufft1d1,
    nufft1d2,
    nufft2d1,
    nufft2d2,
    nufft3d1,
    nufft3d2,
)

import phydrax.ein as ein

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    FunctionLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LSMR,
    solve,
    TolerancePolicy,
)


FourierFitMethod = Literal["direct", "nufft"]
FourierWeightPolicy = Literal["uniform", "explicit"]


def _modes(mode_shape: tuple[int, ...], dtype) -> tuple[Array, ...]:
    return tuple(jnp.fft.fftfreq(size).astype(dtype) * size for size in mode_shape)


def _direct_type2(phases: Array, coefficients: Array, /) -> Array:
    result = coefficients
    for position, (axis_modes, coordinate) in enumerate(
        zip(
            _modes(tuple(coefficients.shape), phases.dtype),
            tuple(phases[:, axis] for axis in range(phases.shape[1])),
            strict=True,
        )
    ):
        phase = jnp.exp(1j * coordinate[:, None] * axis_modes[None, :]).astype(
            coefficients.dtype
        )
        result = (
            ein.contract("mk,k...->m...", phase, result)
            if position == 0
            else ein.contract("mk,mk...->m...", phase, result)
        )
    return result


def _direct_type1(phases: Array, values: Array, mode_shape: tuple[int, ...], /) -> Array:
    result = values
    for axis, axis_modes in enumerate(_modes(mode_shape, phases.dtype)):
        phase = jnp.exp(1j * phases[:, axis, None] * axis_modes[None, :]).astype(
            values.dtype
        )
        result = ein.contract("m...,mk->m...k", result, phase)
    return ein.contract("m...->...", result)


def fourier_type2(
    phases: Array,
    coefficients: Array,
    /,
    *,
    method: FourierFitMethod,
    tolerance: float,
) -> Array:
    """Apply the current-convention finite Fourier synthesis operator."""
    if method == "direct":
        return _direct_type2(phases, coefficients)
    dimension = int(phases.shape[-1])
    centered = jnp.fft.fftshift(coefficients, axes=tuple(range(dimension)))
    if dimension == 1:
        return nufft1d2(phases[:, 0], centered, eps=tolerance, isign=1, upsampfac=2.0)
    if dimension == 2:
        return nufft2d2(
            phases[:, 1],
            phases[:, 0],
            centered,
            eps=tolerance,
            isign=1,
            upsampfac=2.0,
        )
    if dimension == 3:
        return nufft3d2(
            phases[:, 2],
            phases[:, 1],
            phases[:, 0],
            centered,
            eps=tolerance,
            isign=1,
            upsampfac=2.0,
        )
    raise ValueError("NUFFT Fourier fitting supports one through three dimensions.")


def fourier_type1(
    phases: Array,
    values: Array,
    mode_shape: tuple[int, ...],
    /,
    *,
    method: FourierFitMethod,
    tolerance: float,
) -> Array:
    """Apply the normalization-paired algebraic transpose of Type-2 synthesis."""
    if method == "direct":
        return _direct_type1(phases, values, mode_shape)
    dimension = len(mode_shape)
    if dimension == 1:
        centered = nufft1d1(
            phases[:, 0],
            values,
            mode_shape[0],
            eps=tolerance,
            isign=1,
            upsampfac=2.0,
        )
    elif dimension == 2:
        centered = nufft2d1(
            phases[:, 1],
            phases[:, 0],
            values,
            mode_shape,
            eps=tolerance,
            isign=1,
            upsampfac=2.0,
        )
    elif dimension == 3:
        centered = nufft3d1(
            phases[:, 2],
            phases[:, 1],
            phases[:, 0],
            values,
            mode_shape,
            eps=tolerance,
            isign=1,
            upsampfac=2.0,
        )
    else:
        raise ValueError("NUFFT Fourier fitting supports one through three dimensions.")
    return jnp.fft.ifftshift(centered, axes=tuple(range(dimension)))


class FourierScatteredFitPlan(StrictModule):
    """Static matrix-free scattered Fourier least-squares specification."""

    mode_shape: tuple[int, ...] = eqx.field(static=True)
    periods: tuple[float, ...] = eqx.field(static=True)
    origins: tuple[float, ...] = eqx.field(static=True)
    method: FourierFitMethod = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: FourierWeightPolicy = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy | None
    query_chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        mode_shape: Sequence[int],
        periods: Sequence[float],
        /,
        *,
        origins: Sequence[float] | None = None,
        method: FourierFitMethod = "direct",
        tolerance: float = 1.0e-10,
        weight_policy: FourierWeightPolicy = "uniform",
        regularization: float = 0.0,
        linear_policy: LinearSolvePolicy | None = None,
        query_chunk_size: int | None = None,
    ):
        shape = tuple(int(size) for size in mode_shape)
        period_values = tuple(float(value) for value in periods)
        origin_values = (
            (0.0,) * len(shape)
            if origins is None
            else tuple(float(value) for value in origins)
        )
        if not shape or any(size < 1 for size in shape):
            raise ValueError("mode_shape must contain positive mode counts.")
        if len(period_values) != len(shape) or len(origin_values) != len(shape):
            raise ValueError("periods and origins must match mode_shape dimension.")
        if any(not isfinite(value) or value <= 0.0 for value in period_values):
            raise ValueError("Fourier periods must be finite and positive.")
        if any(not isfinite(value) for value in origin_values):
            raise ValueError("Fourier origins must be finite.")
        if method not in ("direct", "nufft"):
            raise ValueError("method must be 'direct' or 'nufft'.")
        if method == "nufft" and len(shape) > 3:
            raise ValueError("NUFFT fitting supports one through three dimensions.")
        tolerance_ = float(tolerance)
        regularization_ = float(regularization)
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Fourier tolerance must be finite and positive.")
        if not isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("regularization must be finite and nonnegative.")
        if weight_policy not in ("uniform", "explicit"):
            raise ValueError("weight_policy must be 'uniform' or 'explicit'.")
        if linear_policy is not None and not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        chunk = None if query_chunk_size is None else int(query_chunk_size)
        if chunk is not None and chunk < 1:
            raise ValueError("query_chunk_size must be positive or None.")
        self.mode_shape = shape
        self.periods = period_values
        self.origins = origin_values
        self.method = method
        self.tolerance = tolerance_
        self.weight_policy = weight_policy
        self.regularization = regularization_
        self.linear_policy = linear_policy
        self.query_chunk_size = chunk


class FourierFitDiagnostics(StrictModule, NonTrainableState):
    status: Array
    residual: Array
    normal_residual: Array
    condition_estimate: Array
    iterations: Array
    sample_count: int = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    method: FourierFitMethod = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


class FourierInterpolant(StrictModule):
    coefficients: Array
    origins: Array
    periods: Array
    source_mode_shape: tuple[int, ...] = eqx.field(static=True)
    method: FourierFitMethod = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    query_chunk_size: int | None = eqx.field(static=True)

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        points = jnp.asarray(coordinates)
        if not jnp.issubdtype(points.dtype, jnp.inexact) or jnp.issubdtype(
            points.dtype, jnp.complexfloating
        ):
            raise TypeError("Fourier query coordinates must be real inexact arrays.")
        if points.ndim < 2 or points.shape[-1] != len(self.source_mode_shape):
            raise ValueError(
                "Fourier coordinates must have shape (..., queries, dimension)."
            )
        query_shape = points.shape[:-1]
        flat = points.reshape((-1, len(self.source_mode_shape)))
        phases = 2.0 * jnp.pi * (flat - self.origins) / self.periods
        payload_shape = self.coefficients.shape[len(self.source_mode_shape) :]
        payload_count = prod(payload_shape) if payload_shape else 1
        coefficient_cases = self.coefficients.reshape(
            (*self.source_mode_shape, payload_count)
        )
        values = jax.vmap(
            lambda coefficients: fourier_type2(
                phases,
                coefficients,
                method=self.method,
                tolerance=self.tolerance,
            ),
            in_axes=-1,
            out_axes=-1,
        )(coefficient_cases)
        return values.reshape((*query_shape, *payload_shape))


def fit_fourier_scattered(
    coordinates: ArrayLike,
    values: ArrayLike,
    plan: FourierScatteredFitPlan,
    /,
    *,
    weights: ArrayLike | None = None,
) -> tuple[FourierInterpolant, FourierFitDiagnostics]:
    """Fit reusable finite Fourier coefficients through native matrix-free LSMR."""
    if not isinstance(plan, FourierScatteredFitPlan):
        raise TypeError("plan must be a FourierScatteredFitPlan.")
    points = jnp.asarray(coordinates)
    data = jnp.asarray(values)
    if points.ndim != 2 or points.shape[1] != len(plan.mode_shape):
        raise ValueError("coordinates must have shape (samples, dimension).")
    if data.ndim < 1 or data.shape[0] != points.shape[0]:
        raise ValueError("values must preserve the coordinate sample axis.")
    if not jnp.issubdtype(points.dtype, jnp.inexact) or jnp.issubdtype(
        points.dtype, jnp.complexfloating
    ):
        raise TypeError("Scattered Fourier coordinates must be real inexact arrays.")
    if bool(jnp.any(~jnp.isfinite(points))) or bool(jnp.any(~jnp.isfinite(data))):
        raise ValueError("Scattered Fourier coordinates and values must be finite.")
    sample_count = int(points.shape[0])
    mode_count = prod(plan.mode_shape)
    if sample_count == 0:
        raise ValueError("Scattered Fourier fitting requires at least one sample.")
    if plan.regularization == 0.0 and sample_count < mode_count:
        raise ValueError(
            "Underdetermined scattered Fourier fits require positive regularization."
        )
    if plan.weight_policy == "explicit" and weights is None:
        raise ValueError("weight_policy='explicit' requires weights=.")
    if plan.weight_policy == "uniform" and weights is not None:
        raise ValueError("weights= requires weight_policy='explicit'.")
    weight_array = None if weights is None else jnp.asarray(weights, dtype=points.dtype)
    if weight_array is not None:
        if weight_array.shape != (sample_count,):
            raise ValueError("weights must contain one value per sample.")
        if bool(jnp.any(~jnp.isfinite(weight_array))) or bool(
            jnp.any(weight_array <= 0.0)
        ):
            raise ValueError("Fourier fit weights must be finite and positive.")

    origins = jnp.asarray(plan.origins, dtype=points.dtype)
    periods = jnp.asarray(plan.periods, dtype=points.dtype)
    phases = 2.0 * jnp.pi * (points - origins) / periods
    complex_dtype = jnp.result_type(data, 1j)
    phases = phases.astype(jnp.real(jnp.zeros((), dtype=complex_dtype)).dtype)
    source = ArraySpace(plan.mode_shape, dtype=complex_dtype)
    target = ArraySpace((sample_count,), dtype=complex_dtype)
    operator = FunctionLinearOperator(
        lambda coefficients: fourier_type2(
            phases,
            coefficients,
            method=plan.method,
            tolerance=plan.tolerance,
        ),
        source=source,
        target=target,
        transpose_action=lambda sample_values: fourier_type1(
            phases,
            sample_values,
            plan.mode_shape,
            method=plan.method,
            tolerance=plan.tolerance,
        ),
        operator_id=f"fourier-scattered-{plan.method}-{'x'.join(map(str, plan.mode_shape))}",
    )
    regularizer = None
    if plan.regularization > 0.0:
        scale = jnp.asarray(sqrt(plan.regularization), dtype=complex_dtype)
        regularizer = FunctionLinearOperator(
            lambda coefficients: scale * coefficients,
            source=source,
            target=source,
            transpose_action=lambda coefficients: scale * coefficients,
            operator_id=f"fourier-tikhonov-{plan.regularization.hex()}",
        )
    problem = LeastSquaresProblem(
        operator,
        weights=weight_array,
        regularizer=regularizer,
    )
    policy = plan.linear_policy or LinearSolvePolicy(
        LSMR(damping=0.0),
        tolerance=TolerancePolicy(
            relative=plan.tolerance,
            absolute=plan.tolerance,
            max_steps=max(32, 4 * mode_count),
        ),
    )
    payload_shape = tuple(int(size) for size in data.shape[1:])
    flattened = data.reshape((sample_count, -1)).astype(complex_dtype)
    results = tuple(
        solve(problem, flattened[:, index], policy=policy)
        for index in range(int(flattened.shape[1]))
    )
    coefficients = jnp.stack(tuple(result.value for result in results), axis=-1).reshape(
        (*plan.mode_shape, *payload_shape)
    )
    diagnostics = FourierFitDiagnostics(
        status=jnp.stack(tuple(result.status for result in results)),
        residual=jnp.stack(tuple(result.diagnostics.residual_norm for result in results)),
        normal_residual=jnp.stack(
            tuple(result.diagnostics.normal_residual_norm for result in results)
        ),
        condition_estimate=jnp.stack(
            tuple(result.diagnostics.condition_estimate for result in results)
        ),
        iterations=jnp.stack(tuple(result.diagnostics.iterations for result in results)),
        sample_count=sample_count,
        mode_count=mode_count,
        method=plan.method,
        tolerance=plan.tolerance,
    )
    interpolant = FourierInterpolant(
        coefficients=coefficients,
        origins=origins,
        periods=periods,
        source_mode_shape=plan.mode_shape,
        method=plan.method,
        tolerance=plan.tolerance,
        query_chunk_size=plan.query_chunk_size,
    )
    return interpolant, diagnostics


__all__ = [
    "FourierFitDiagnostics",
    "FourierFitMethod",
    "FourierInterpolant",
    "FourierScatteredFitPlan",
    "FourierWeightPolicy",
    "fit_fourier_scattered",
    "fourier_type1",
    "fourier_type2",
]
