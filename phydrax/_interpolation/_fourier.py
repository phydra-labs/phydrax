#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from nufftax import nufft1d2, nufft2d2, nufft3d2

from .._spectral._fourier import resize_fourier_axis as _resize_fourier_axis
from ._types import InterpolationCapabilities, InterpolationResult


FourierEvaluationMethod: TypeAlias = Literal["direct", "nufft"]


FOURIER_CAPABILITIES = InterpolationCapabilities(
    partition_of_unity=True,
    nonnegative_value_weights=False,
    local_support=False,
    mask_renormalizable=False,
    tensor_product_composable=True,
    maximum_explicit_derivative_order=0,
)


def _as_inexact(values: ArrayLike, /) -> Array:
    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return array.astype(float)
    return array


def _validate_period(value: ArrayLike, dtype: jnp.dtype, name: str, /) -> Array:
    raw = jnp.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be scalar; got {raw.shape}.")
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    period = jnp.asarray(raw, dtype=dtype)
    return eqx.error_if(
        period,
        ~jnp.isfinite(period) | (period <= 0.0),
        f"{name} must be finite and positive.",
    )


def _axis_geometry(
    source_shape: tuple[int, ...],
    axis_nodes: Sequence[ArrayLike] | None,
    periods: Sequence[ArrayLike] | None,
    dtype: jnp.dtype,
    /,
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    dimensions = len(source_shape)
    periods_ = None if periods is None else tuple(periods)
    if periods_ is not None and len(periods_) != dimensions:
        raise ValueError("periods must provide one scalar per Fourier axis.")

    if axis_nodes is None:
        origins = tuple(jnp.asarray(0.0, dtype=dtype) for _ in source_shape)
        resolved_periods = tuple(
            _validate_period(
                1.0 if periods_ is None else periods_[axis],
                dtype,
                f"periods[{axis}]",
            )
            for axis in range(dimensions)
        )
        return origins, resolved_periods

    nodes_ = tuple(axis_nodes)
    if len(nodes_) != dimensions:
        raise ValueError("axis_nodes must provide one array per Fourier axis.")

    origins: list[Array] = []
    resolved_periods: list[Array] = []
    for axis, (nodes, size) in enumerate(zip(nodes_, source_shape, strict=True)):
        raw_nodes = jnp.asarray(nodes)
        if raw_nodes.ndim != 1:
            raise ValueError(f"axis_nodes[{axis}] must be rank one.")
        if raw_nodes.shape != (size,):
            raise ValueError(
                f"axis_nodes[{axis}] must have shape {(size,)}; got {raw_nodes.shape}."
            )
        if jnp.issubdtype(raw_nodes.dtype, jnp.complexfloating):
            raise TypeError("Fourier axis nodes must be real-valued.")
        values = jnp.asarray(raw_nodes, dtype=dtype)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            f"axis_nodes[{axis}] must be finite.",
        )

        if size == 1:
            if periods_ is None:
                raise ValueError(
                    "A one-node custom Fourier axis requires an explicit period."
                )
            period = _validate_period(periods_[axis], dtype, f"periods[{axis}]")
        else:
            spacing = jnp.diff(values)
            values = eqx.error_if(
                values,
                jnp.any(spacing <= 0.0),
                f"axis_nodes[{axis}] must be strictly increasing.",
            )
            inferred = jnp.asarray(float(size), dtype=dtype) * jnp.mean(spacing)
            period = (
                _validate_period(inferred, dtype, f"periods[{axis}]")
                if periods_ is None
                else _validate_period(periods_[axis], dtype, f"periods[{axis}]")
            )
            expected_spacing = period / float(size)
            values = eqx.error_if(
                values,
                jnp.logical_not(
                    jnp.allclose(
                        spacing,
                        expected_spacing,
                        rtol=1e-5,
                        atol=1e-8,
                    )
                ),
                f"axis_nodes[{axis}] must be uniformly spaced over its period.",
            )

        origins.append(values[0])
        resolved_periods.append(period)
    return tuple(origins), tuple(resolved_periods)


def _expanded_coefficients(
    values: Array,
    batch_ndim: int,
    spatial_ndim: int,
    /,
) -> tuple[Array, tuple[int, ...]]:
    axes = tuple(range(batch_ndim, batch_ndim + spatial_ndim))
    source_shape = tuple(int(values.shape[axis]) for axis in axes)
    coefficients = jnp.fft.fftn(values, axes=axes, norm="forward")
    for axis, size in zip(axes, source_shape, strict=True):
        if size % 2 == 0:
            coefficients = _resize_fourier_axis(coefficients, axis, size + 1)
    return coefficients, source_shape


def _normalize_queries(
    coordinates: ArrayLike,
    *,
    batch_shape: tuple[int, ...],
    spatial_ndim: int,
    origins: tuple[Array, ...],
    periods: tuple[Array, ...],
    dtype: jnp.dtype,
) -> tuple[Array, tuple[int, ...]]:
    raw = jnp.asarray(coordinates)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError("Fourier query coordinates must be real-valued.")
    query = jnp.asarray(raw, dtype=dtype)
    if query.ndim < len(batch_shape) + 1 or query.shape[-1:] != (spatial_ndim,):
        raise ValueError(
            "coordinates must have shape batch_shape + query_shape + "
            f"({spatial_ndim},); got {query.shape}."
        )
    if tuple(int(size) for size in query.shape[: len(batch_shape)]) != batch_shape:
        raise ValueError(
            f"Coordinate batch shape must be {batch_shape}; got "
            f"{query.shape[: len(batch_shape)]}."
        )
    query_shape = tuple(int(size) for size in query.shape[len(batch_shape) : -1])
    if any(size <= 0 for size in query_shape):
        raise ValueError("Fourier query axes must be nonempty.")
    query = eqx.error_if(
        query,
        jnp.any(~jnp.isfinite(query)),
        "Fourier query coordinates must be finite.",
    )

    two_pi = 2.0 * jnp.asarray(jnp.pi, dtype=dtype)
    normalized = []
    for axis, (origin, period) in enumerate(zip(origins, periods, strict=True)):
        angle = two_pi * (query[..., axis] - origin) / period
        normalized.append(jnp.mod(angle + jnp.pi, two_pi) - jnp.pi)
    return jnp.stack(normalized, axis=-1), query_shape


def _canonical_coefficients(
    coefficients: Array,
    *,
    batch_shape: tuple[int, ...],
    payload_shape: tuple[int, ...],
    spatial_ndim: int,
) -> Array:
    batch_ndim = len(batch_shape)
    spatial_axes = tuple(range(batch_ndim, batch_ndim + spatial_ndim))
    payload_axes = tuple(range(batch_ndim + spatial_ndim, coefficients.ndim))
    order = tuple(range(batch_ndim)) + payload_axes + spatial_axes
    arranged = jnp.transpose(coefficients, order)
    batch_count = prod(batch_shape) if batch_shape else 1
    payload_count = prod(payload_shape) if payload_shape else 1
    mode_shape = tuple(int(coefficients.shape[axis]) for axis in spatial_axes)
    return arranged.reshape((batch_count, payload_count, *mode_shape))


def _direct_fourier_evaluate(coordinates: Array, coefficients: Array, /) -> Array:
    dimensions = int(coordinates.shape[-1])
    mode_shape = tuple(int(size) for size in coefficients.shape[-dimensions:])
    modes = tuple(
        jnp.fft.fftfreq(size).astype(coordinates.dtype) * size for size in mode_shape
    )

    def evaluate_case(points: Array, case_coefficients: Array) -> Array:
        def evaluate_point(point: Array) -> Array:
            result = case_coefficients
            for axis_modes, coordinate in zip(modes, point, strict=True):
                phase = jnp.exp(1j * axis_modes * coordinate).astype(
                    case_coefficients.dtype
                )
                result = jnp.tensordot(result, phase, axes=((1,), (0,)))
            return result

        return jnp.swapaxes(jax.vmap(evaluate_point)(points), 0, 1)

    return jax.vmap(evaluate_case)(coordinates, coefficients)


def _nufft_fourier_evaluate(
    coordinates: Array,
    coefficients: Array,
    /,
    *,
    tolerance: float,
) -> Array:
    dimensions = int(coordinates.shape[-1])
    centered = jnp.fft.fftshift(
        coefficients,
        axes=tuple(range(coefficients.ndim - dimensions, coefficients.ndim)),
    )

    def evaluate_case(points: Array, case_coefficients: Array) -> Array:
        if dimensions == 1:
            return nufft1d2(
                points[:, 0],
                case_coefficients,
                eps=tolerance,
                isign=1,
                upsampfac=2.0,
            )
        if dimensions == 2:
            return nufft2d2(
                points[:, 1],
                points[:, 0],
                case_coefficients,
                eps=tolerance,
                isign=1,
                upsampfac=2.0,
            )
        return nufft3d2(
            points[:, 2],
            points[:, 1],
            points[:, 0],
            case_coefficients,
            eps=tolerance,
            isign=1,
            upsampfac=2.0,
        )

    return jax.vmap(evaluate_case)(coordinates, centered)


def _evaluate_query_chunks(
    evaluator: Callable[[Array, Array], Array],
    coordinates: Array,
    coefficients: Array,
    chunk_size: int | None,
    /,
) -> Array:
    if chunk_size is None:
        return evaluator(coordinates, coefficients)

    query_count = int(coordinates.shape[1])
    chunk_count = (query_count + chunk_size - 1) // chunk_size
    padded_count = chunk_count * chunk_size
    if padded_count != query_count:
        padding = jnp.broadcast_to(
            coordinates[:, :1, :],
            (coordinates.shape[0], padded_count - query_count, coordinates.shape[-1]),
        )
        coordinates = jnp.concatenate((coordinates, padding), axis=1)

    chunks = coordinates.reshape(
        (coordinates.shape[0], chunk_count, chunk_size, coordinates.shape[-1])
    )
    chunks = jnp.moveaxis(chunks, 1, 0)
    evaluated = jax.lax.map(lambda chunk: evaluator(chunk, coefficients), chunks)
    evaluated = jnp.transpose(evaluated, (1, 2, 0, 3)).reshape(
        (coordinates.shape[0], coefficients.shape[1], padded_count)
    )
    return evaluated[..., :query_count]


def fourier_interpolate(
    values: ArrayLike,
    coordinates: ArrayLike,
    /,
    *,
    spatial_ndim: int,
    payload_ndim: int = 1,
    axis_nodes: Sequence[ArrayLike] | None = None,
    periods: Sequence[ArrayLike] | None = None,
    method: FourierEvaluationMethod = "direct",
    tolerance: float | None = None,
    query_chunk_size: int | None = None,
) -> InterpolationResult:
    """Evaluate a periodic tensor-grid field at paired arbitrary coordinates.

    Values have shape ``batch_shape + source_shape + payload_shape`` and queries
    have shape ``batch_shape + query_shape + (spatial_ndim,)``. The direct method
    evaluates the finite Fourier series exactly up to floating-point roundoff;
    ``method="nufft"`` uses NUFFTAX Type 2 at an explicit tolerance.
    """
    dimensions = int(spatial_ndim)
    payload_dimensions = int(payload_ndim)
    if dimensions <= 0:
        raise ValueError("spatial_ndim must be positive.")
    if payload_dimensions < 0:
        raise ValueError("payload_ndim must be nonnegative.")
    if method not in ("direct", "nufft"):
        raise ValueError("method must be 'direct' or 'nufft'.")
    if method == "direct":
        if tolerance is not None:
            raise ValueError("The direct Fourier method does not accept a tolerance.")
        tolerance_ = None
    else:
        if dimensions not in (1, 2, 3):
            raise ValueError("NUFFT Fourier interpolation supports one to three axes.")
        if tolerance is None:
            raise ValueError("NUFFT Fourier interpolation requires a tolerance.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or not 0.0 < tolerance_ < 1.0:
            raise ValueError("NUFFT tolerance must be finite and lie in (0, 1).")
    if query_chunk_size is not None:
        query_chunk_size = int(query_chunk_size)
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive.")

    array = _as_inexact(values)
    minimum_rank = dimensions + payload_dimensions
    if array.ndim < minimum_rank:
        raise ValueError(
            "values must contain the declared spatial and payload dimensions."
        )
    batch_ndim = array.ndim - minimum_rank
    batch_shape = tuple(int(size) for size in array.shape[:batch_ndim])
    source_shape = tuple(
        int(size) for size in array.shape[batch_ndim : batch_ndim + dimensions]
    )
    if any(size <= 0 for size in source_shape):
        raise ValueError("Fourier source axes must be nonempty.")
    payload_shape = tuple(int(size) for size in array.shape[batch_ndim + dimensions :])

    coefficients, actual_source_shape = _expanded_coefficients(
        array,
        batch_ndim,
        dimensions,
    )
    if actual_source_shape != source_shape:
        raise RuntimeError("Fourier source layout changed during coefficient assembly.")
    dtype = coefficients.real.dtype
    origins, resolved_periods = _axis_geometry(
        source_shape,
        axis_nodes,
        periods,
        dtype,
    )
    normalized, query_shape = _normalize_queries(
        coordinates,
        batch_shape=batch_shape,
        spatial_ndim=dimensions,
        origins=origins,
        periods=resolved_periods,
        dtype=dtype,
    )

    batch_count = prod(batch_shape) if batch_shape else 1
    query_count = prod(query_shape) if query_shape else 1
    normalized = normalized.reshape((batch_count, query_count, dimensions))
    canonical = _canonical_coefficients(
        coefficients,
        batch_shape=batch_shape,
        payload_shape=payload_shape,
        spatial_ndim=dimensions,
    )

    if method == "direct":
        evaluator = _direct_fourier_evaluate
    else:
        assert tolerance_ is not None
        evaluator = lambda query, coeff: _nufft_fourier_evaluate(
            query,
            coeff,
            tolerance=tolerance_,
        )
    evaluated = _evaluate_query_chunks(
        evaluator,
        normalized,
        canonical,
        query_chunk_size,
    )
    if not jnp.issubdtype(array.dtype, jnp.complexfloating):
        evaluated = evaluated.real

    grouped = evaluated.reshape(batch_shape + payload_shape + query_shape)
    batch_axes = tuple(range(len(batch_shape)))
    payload_axes = tuple(range(len(batch_shape), len(batch_shape) + len(payload_shape)))
    query_axes = tuple(range(len(batch_shape) + len(payload_shape), grouped.ndim))
    output = jnp.transpose(grouped, batch_axes + query_axes + payload_axes)
    support = jnp.ones(batch_shape + query_shape, dtype=bool)
    return InterpolationResult(values=output, support=support)


__all__ = [
    "FOURIER_CAPABILITIES",
    "FourierEvaluationMethod",
    "fourier_interpolate",
]
