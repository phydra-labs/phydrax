#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Nonuniform Fourier transforms of types 1, 2 and 3.

The exact routes evaluate the defining sums. The gridded routes follow
Barnett, Magland and af Klinteberg, "A parallel nonuniform fast Fourier
transform library based on an 'exponential of semicircle' kernel", SIAM J. Sci.
Comput. 41 (2019): spread with the exponential-of-semicircle (ES) kernel onto a
twofold oversampled grid, apply an FFT, and deconvolve by the kernel Fourier
transform. Type 3 rescales both point sets, spreads onto a nonperiodic grid and
applies a gridded Type-2 transform at rescaled targets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from .._polynomial._orthogonal import legendre_rule_data
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar, positive_finite_float
from ..typing import parse


NonuniformFourierType: TypeAlias = Literal[1, 2]
NonuniformFourierRoute: TypeAlias = Literal["direct", "chunked", "gridded"]

_OVERSAMPLING = 2.0
# Width 16 is the widest ES kernel; it reaches double-precision roundoff.
_MINIMUM_AXIS_TOLERANCE = 1.0e-15
_MINIMUM_TOLERANCE = 1.0e-13
# The width rule attains its tolerance only approximately; a factor-two margin
# keeps the delivered relative error below the request.
_TOLERANCE_MARGIN = 0.5
_TWO_PI = 2.0 * math.pi


class NonuniformFourierResourceError(ValueError):
    """A gridded nonuniform Fourier plan exceeds its declared grid capacity."""


def _static_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _complex_dtype(*arrays: Array) -> jnp.dtype:
    """Return the complex dtype resolving every operand, at least ``complex64``."""
    itemsize = max(jnp.dtype(array.real.dtype).itemsize for array in arrays)
    return jnp.dtype(jnp.complex64 if itemsize <= 4 else jnp.complex128)


def _real_floating_dtype(dtype: DTypeLike, /) -> jnp.dtype:
    resolved = jnp.dtype(dtype)
    if not jnp.issubdtype(resolved, jnp.floating):
        raise TypeError("Nonuniform Fourier preparation requires a real floating dtype.")
    return resolved


def _smooth_even_size(minimum: float, /) -> int:
    """Return the smallest even 2-3-5-smooth integer at least ``minimum``."""
    size = max(2, math.ceil(minimum))
    size += size % 2
    while True:
        remainder = size
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return size
        size += 2


def _chunks(array: Array, chunk_size: int, /) -> Array:
    count = array.shape[0]
    chunk = max(1, min(chunk_size, count))
    chunk_count = -(-count // chunk)
    padding = chunk_count * chunk - count
    if padding:
        array = jnp.concatenate(
            (array, jnp.zeros((padding, *array.shape[1:]), dtype=array.dtype)),
            axis=0,
        )
    return array.reshape((chunk_count, chunk, *array.shape[1:]))


class NonuniformFourierKernel(StrictModule):
    """Exponential-of-semicircle kernel selected from a per-axis tolerance.

    ``phi(z) = exp(beta * (sqrt(1 - z**2) - 1))`` on ``|z| < 1`` spans
    ``width`` fine-grid cells with twofold oversampling. Following Barnett,
    Magland and af Klinteberg (2019),
    ``width = ceil(log10(1 / axis_tolerance)) + 1`` and ``beta = 2.30 * width``
    (2.20, 2.26 and 2.38 for widths 2, 3 and 4). Plans split the requested
    relative error evenly over tensor axes and gridding stages, because the
    aliasing errors of each factor add, and keep a factor-two margin.
    """

    axis_tolerance: float = eqx.field(static=True)
    width: int = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    oversampling: float = eqx.field(static=True)

    def __init__(self, axis_tolerance: float) -> None:
        requested = finite_real_scalar(axis_tolerance, "Nonuniform Fourier tolerance")
        if not _MINIMUM_AXIS_TOLERANCE <= requested < 1.0:
            raise ValueError(
                "Nonuniform Fourier axis tolerance must lie in [1e-15, 1); the widest "
                "exponential-of-semicircle kernel reaches double-precision roundoff."
            )
        width = max(2, math.ceil(math.log10(1.0 / requested)) + 1)
        beta_per_width = {2: 2.20, 3: 2.26, 4: 2.38}.get(width, 2.30)
        self.axis_tolerance = requested
        self.width = width
        self.beta = beta_per_width * width
        self.oversampling = _OVERSAMPLING

    def evaluate(self, z: Array, /) -> Array:
        """Evaluate the kernel at normalized offsets ``z`` (support ``|z| < 1``)."""
        inside = jnp.abs(z) < 1.0
        # The double select keeps the derivative finite at the support edge.
        radicand = jnp.where(inside, 1.0 - z * z, 1.0)
        return jnp.where(inside, jnp.exp(self.beta * (jnp.sqrt(radicand) - 1.0)), 0.0)

    def periodic_fine_size(self, mode_count: int, /) -> int:
        return _smooth_even_size(max(self.oversampling * mode_count, 2 * self.width))


def _requested_tolerance(value: float, /) -> float:
    tolerance = finite_real_scalar(value, "Nonuniform Fourier tolerance")
    if not _MINIMUM_TOLERANCE <= tolerance < 1.0:
        raise ValueError("Nonuniform Fourier tolerance must lie in [1e-13, 1).")
    return tolerance


def _validate_precision(tolerance: float, dtype: jnp.dtype, /) -> None:
    # Spreading, FFT and deconvolution accumulate roundoff of about 100 ulps.
    if tolerance < 100.0 * float(jnp.finfo(dtype).eps):
        raise ValueError(
            f"Nonuniform Fourier tolerance {tolerance:g} is below the resolution "
            f"of prepared dtype {dtype.name}."
        )


class _KernelTransformRule(StrictModule, NonTrainableState):
    """Native Gauss--Legendre rule for the ES kernel Fourier transform."""

    nodes: Array
    weighted_kernel: Array
    width: int = eqx.field(static=True)

    def __init__(self, kernel: NonuniformFourierKernel, dtype: jnp.dtype, /) -> None:
        # The half-interval rule integrates the even kernel; its endpoint
        # derivative singularity carries weight exp(-beta), below tolerance.
        rule = legendre_rule_data(2 * kernel.width + 4, "gauss", dtype=dtype)
        nodes = 0.5 * (rule.nodes + 1.0)
        self.nodes = nodes
        self.weighted_kernel = 0.5 * rule.weights * kernel.evaluate(nodes)
        self.width = kernel.width

    def transform(self, frequencies: Array, spacing: float, /) -> Array:
        """Return ``int phi(2u / (width * spacing)) exp(i k u) du`` at ``k``."""
        half_support = 0.5 * self.width * spacing
        nodes = self.nodes.astype(frequencies.dtype)
        phases = jnp.cos(frequencies[:, None] * (half_support * nodes)[None, :])
        return (2.0 * half_support) * ein.contract(
            "kq,q->k", phases, self.weighted_kernel.astype(frequencies.dtype)
        )


class NonuniformFourierGridEvidence(StrictModule):
    """Accuracy request, kernel, and resource estimate of a gridded transform.

    ``grid_points`` counts every resident fine-grid point per payload channel,
    ``grid_bytes`` their complex storage in the prepared precision, and
    ``working_entries`` the kernel-stencil entries of one point chunk.
    """

    requested_tolerance: float = eqx.field(static=True)
    kernel_width: int = eqx.field(static=True)
    kernel_beta: float = eqx.field(static=True)
    oversampling: float = eqx.field(static=True)
    fine_shape: tuple[int, ...] = eqx.field(static=True)
    grid_points: int = eqx.field(static=True)
    maximum_grid_points: int = eqx.field(static=True)
    working_entries: int = eqx.field(static=True)
    grid_bytes: int = eqx.field(static=True)


def _grid_evidence(
    tolerance: float,
    kernel: NonuniformFourierKernel,
    fine_shape: tuple[int, ...],
    grid_points: int,
    maximum_grid_points: int,
    chunk_size: int,
    dtype: jnp.dtype,
    /,
) -> NonuniformFourierGridEvidence:
    complex_itemsize = 2 * dtype.itemsize
    return NonuniformFourierGridEvidence(
        tolerance,
        kernel.width,
        kernel.beta,
        kernel.oversampling,
        fine_shape,
        grid_points,
        maximum_grid_points,
        chunk_size * kernel.width ** len(fine_shape),
        grid_points * complex_itemsize,
    )


def _refuse_grid(grid_points: int, maximum_grid_points: int, /) -> None:
    if grid_points > maximum_grid_points:
        raise NonuniformFourierResourceError(
            f"Nonuniform Fourier fine grids need {grid_points} points, above "
            f"maximum_grid_points={maximum_grid_points}."
        )


def _stencil(
    coordinates: Array,
    scales: tuple[float, ...],
    shifts: tuple[float, ...],
    fine_shape: tuple[int, ...],
    kernel: NonuniformFourierKernel,
    /,
) -> tuple[Array, Array]:
    """Return row-major fine-grid indices and tensor kernel weights per point."""
    count = coordinates.shape[0]
    half_width = 0.5 * kernel.width
    offsets = jnp.arange(kernel.width, dtype=coordinates.dtype)
    linear = jnp.zeros((count, 1), dtype=jnp.int64)
    weights = jnp.ones((count, 1), dtype=coordinates.dtype)
    for axis, size in enumerate(fine_shape):
        position = coordinates[:, axis] * scales[axis] + shifts[axis]
        nodes = jnp.ceil(position - half_width)[:, None] + offsets[None, :]
        axis_weights = kernel.evaluate((nodes - position[:, None]) / half_width)
        axis_index = jnp.mod(nodes.astype(jnp.int64), size)
        linear = (linear[:, :, None] * size + axis_index[:, None, :]).reshape((count, -1))
        weights = (weights[:, :, None] * axis_weights[:, None, :]).reshape((count, -1))
    return linear, weights


def _spread(
    coordinates: Array,
    strengths: Array,
    scales: tuple[float, ...],
    shifts: tuple[float, ...],
    fine_shape: tuple[int, ...],
    kernel: NonuniformFourierKernel,
    chunk_size: int,
    /,
) -> Array:
    """Spread point strengths onto the flattened fine grid in bounded chunks."""
    payload = strengths.shape[1:]

    def accumulate(grid: Array, chunk: tuple[Array, Array]) -> tuple[Array, None]:
        points, values = chunk
        linear, weights = _stencil(points, scales, shifts, fine_shape, kernel)
        contributions = ein.contract(
            "mq,m...->mq...", weights.astype(values.dtype), values
        )
        # The stencil depends on per-call coordinates, so there is no reusable
        # sparse relation to prepare; one scatter-add per chunk owns the update.
        grid = grid.at[linear.reshape(-1)].add(contributions.reshape((-1, *payload)))
        return grid, None

    grid, _ = jax.lax.scan(
        accumulate,
        jnp.zeros((math.prod(fine_shape), *payload), dtype=strengths.dtype),
        (_chunks(coordinates, chunk_size), _chunks(strengths, chunk_size)),
    )
    return grid


def _interpolate(
    grid: Array,
    coordinates: Array,
    scales: tuple[float, ...],
    shifts: tuple[float, ...],
    fine_shape: tuple[int, ...],
    kernel: NonuniformFourierKernel,
    chunk_size: int,
    /,
) -> Array:
    """Interpolate the flattened fine grid at points in bounded chunks."""

    def evaluate(points: Array) -> Array:
        linear, weights = _stencil(points, scales, shifts, fine_shape, kernel)
        return ein.contract("mq,mq...->m...", weights.astype(grid.dtype), grid[linear])

    outputs = jax.lax.map(evaluate, _chunks(coordinates, chunk_size))
    rows = outputs.shape[0] * outputs.shape[1]
    return outputs.reshape((rows, *outputs.shape[2:]))[: coordinates.shape[0]]


def _fine_fourier_sum(grid: Array, sign: int, dimension: int, /) -> Array:
    """Return ``sum_l grid[l] exp(i sign k l 2 pi / n)`` over the leading axes."""
    axes = tuple(range(dimension))
    match sign:
        case 1:
            return jnp.fft.ifftn(grid, axes=axes, norm="forward")
        case -1:
            return jnp.fft.fftn(grid, axes=axes)
        case _:
            raise ValueError("Nonuniform Fourier sign must be -1 or 1.")


def _scale_modes(values: Array, corrections: tuple[Array, ...], /) -> Array:
    letters = "ijk"[: len(corrections)]
    return ein.contract(
        f"{letters}...,{','.join(letters)}->{letters}...",
        values,
        *(correction.astype(values.dtype) for correction in corrections),
    )


class NonuniformFourierPlan(StrictModule):
    """Static Type-1/2 nonuniform Fourier convention, route, and resource policy.

    ``"direct"`` and ``"chunked"`` evaluate the exact sums. ``"gridded"``
    requires ``tolerance`` and executes the ES-kernel spread/FFT/deconvolution
    route on a twofold oversampled grid of at most ``maximum_grid_points``.
    """

    mode_shape: tuple[int, ...] = eqx.field(static=True)
    transform_type: NonuniformFourierType = eqx.field(static=True)
    sign: int = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    route: NonuniformFourierRoute = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    tolerance: float | None = eqx.field(static=True)
    kernel: NonuniformFourierKernel | None = eqx.field(static=True)
    fine_shape: tuple[int, ...] | None = eqx.field(static=True)
    maximum_grid_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_shape: Sequence[int],
        transform_type: NonuniformFourierType,
        /,
        *,
        sign: int = 1,
        centered: bool = False,
        route: NonuniformFourierRoute = "direct",
        chunk_size: int = 256,
        tolerance: float | None = None,
        maximum_grid_points: int = 2**24,
    ) -> None:
        supplied_shape = tuple(mode_shape)
        if not supplied_shape or len(supplied_shape) > 3:
            raise ValueError(
                "Nonuniform Fourier mode_shape must contain one to three positive sizes."
            )
        shape = tuple(
            _static_int(size, "Nonuniform Fourier mode size") for size in supplied_shape
        )
        if any(size <= 0 for size in shape):
            raise ValueError(
                "Nonuniform Fourier mode_shape must contain one to three positive sizes."
            )
        transform = parse(
            _static_int(transform_type, "Nonuniform Fourier transform_type"),
            NonuniformFourierType,
            "transform_type",
        )
        exponent_sign = _static_int(sign, "Nonuniform Fourier sign")
        if exponent_sign not in (-1, 1):
            raise ValueError("Nonuniform Fourier sign must be -1 or 1.")
        if not isinstance(centered, bool):
            raise TypeError("Nonuniform Fourier centered must be a bool.")
        route = parse(route, NonuniformFourierRoute, "route")
        chunk = _static_int(chunk_size, "Nonuniform Fourier chunk_size")
        if chunk < 1:
            raise ValueError("Nonuniform Fourier chunk_size must be positive.")
        capacity = _static_int(
            maximum_grid_points, "Nonuniform Fourier maximum_grid_points"
        )
        if capacity < 1:
            raise ValueError("Nonuniform Fourier maximum_grid_points must be positive.")
        match route:
            case "direct" | "chunked":
                if tolerance is not None:
                    raise ValueError(
                        "Exact nonuniform Fourier routes do not accept a tolerance."
                    )
                requested = None
                kernel = None
                fine_shape = None
            case "gridded":
                if tolerance is None:
                    raise ValueError(
                        "The gridded nonuniform Fourier route needs tolerance."
                    )
                requested = _requested_tolerance(tolerance)
                kernel = NonuniformFourierKernel(
                    _TOLERANCE_MARGIN * requested / len(shape)
                )
                fine_shape = tuple(kernel.periodic_fine_size(size) for size in shape)
                _refuse_grid(math.prod(fine_shape), capacity)
            case _:
                raise ValueError(f"Unsupported nonuniform Fourier route {route!r}.")
        self.mode_shape = shape
        self.transform_type = transform
        self.sign = exponent_sign
        self.centered = centered
        self.route = route
        self.chunk_size = chunk
        self.tolerance = requested
        self.kernel = kernel
        self.fine_shape = fine_shape
        self.maximum_grid_points = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nonuniform-fourier-plan",
                "mode_shape": shape,
                "type": transform,
                "sign": exponent_sign,
                "centered": centered,
                "route": route,
                "chunk_size": chunk,
                "tolerance": requested,
                "fine_shape": fine_shape,
            }
        )


class PreparedNonuniformFourier(StrictModule, NonTrainableState):
    """Prepared Type-1/2 mode arrays and gridded deconvolution factors."""

    modes: tuple[Array, ...]
    fine_indices: tuple[Array, ...]
    corrections: tuple[Array, ...]
    plan: NonuniformFourierPlan = eqx.field(static=True)
    evidence: NonuniformFourierGridEvidence | None = eqx.field(static=True)

    def __init__(
        self,
        plan: NonuniformFourierPlan,
        /,
        *,
        dtype: DTypeLike,
    ) -> None:
        if not isinstance(plan, NonuniformFourierPlan):
            raise TypeError("plan must be a NonuniformFourierPlan.")
        resolved_dtype = _real_floating_dtype(dtype)
        integer_modes = tuple(
            np.arange(size, dtype=np.int64) - size // 2
            if plan.centered
            else (np.arange(size, dtype=np.int64) + size // 2) % size - size // 2
            for size in plan.mode_shape
        )
        self.modes = tuple(
            jnp.asarray(mode, dtype=resolved_dtype) for mode in integer_modes
        )
        kernel = plan.kernel
        fine_shape = plan.fine_shape
        requested = plan.tolerance
        if kernel is None or fine_shape is None or requested is None:
            self.fine_indices = ()
            self.corrections = ()
            self.evidence = None
        else:
            _validate_precision(requested, resolved_dtype)
            rule = _KernelTransformRule(kernel, resolved_dtype)
            self.fine_indices = tuple(
                jnp.asarray(mode % size, dtype=jnp.int64)
                for mode, size in zip(integer_modes, fine_shape, strict=True)
            )
            self.corrections = tuple(
                (_TWO_PI / size) / rule.transform(mode, _TWO_PI / size)
                for mode, size in zip(self.modes, fine_shape, strict=True)
            )
            self.evidence = _grid_evidence(
                requested,
                kernel,
                fine_shape,
                math.prod(fine_shape),
                plan.maximum_grid_points,
                plan.chunk_size,
                resolved_dtype,
            )
        self.plan = plan

    def _gridding(self) -> tuple[NonuniformFourierKernel, tuple[int, ...]]:
        kernel = self.plan.kernel
        fine_shape = self.plan.fine_shape
        if kernel is None or fine_shape is None:
            raise ValueError("Prepared nonuniform Fourier plan is not gridded.")
        return kernel, fine_shape

    def _type2_direct(self, points: Array, values: Array, /) -> Array:
        dtype = _complex_dtype(points, values, *self.modes)
        result = values.astype(dtype)
        imaginary_unit = jnp.asarray(1j, dtype=dtype)
        for position, (mode, coordinate) in enumerate(
            zip(self.modes, points.T, strict=True)
        ):
            angle = self.plan.sign * coordinate[:, None] * mode[None, :]
            phase = jnp.exp(imaginary_unit * angle)
            result = (
                ein.contract("mk,k...->m...", phase, result)
                if position == 0
                else ein.contract("mk,mk...->m...", phase, result)
            )
        return result

    def _type2_gridded(self, points: Array, values: Array, /) -> Array:
        kernel, fine_shape = self._gridding()
        dtype = _complex_dtype(points, values, *self.modes)
        dimension = len(fine_shape)
        payload = values.shape[dimension:]
        scaled = _scale_modes(values.astype(dtype), self.corrections)
        grid = (
            jnp.zeros((*fine_shape, *payload), dtype=dtype)
            .at[jnp.ix_(*self.fine_indices)]
            .set(scaled)
        )
        grid = _fine_fourier_sum(grid, self.plan.sign, dimension)
        periodic = jnp.mod(points.astype(jnp.finfo(dtype).dtype), _TWO_PI)
        return _interpolate(
            grid.reshape((-1, *payload)),
            periodic,
            tuple(size / _TWO_PI for size in fine_shape),
            (0.0,) * dimension,
            fine_shape,
            kernel,
            self.plan.chunk_size,
        )

    def type2(self, coordinates: ArrayLike, coefficients: ArrayLike, /) -> Array:
        if self.plan.transform_type != 2:
            raise ValueError("Prepared plan is not Type 2.")
        points = jnp.asarray(coordinates)
        values = jnp.asarray(coefficients)
        if points.ndim != 2 or points.shape[1] != len(self.modes):
            raise ValueError(
                "Nonuniform Fourier coordinates must have shape (points, dimension)."
            )
        if jnp.issubdtype(points.dtype, jnp.complexfloating):
            raise TypeError("Nonuniform Fourier coordinates must be real-valued.")
        if values.shape[: len(self.modes)] != self.plan.mode_shape:
            raise ValueError("Nonuniform Fourier coefficients do not match mode_shape.")
        point_count = points.shape[0]
        match self.plan.route:
            case "direct":
                return self._type2_direct(points, values)
            case "gridded":
                return self._type2_gridded(points, values)
            case "chunked":
                if point_count == 0:
                    return self._type2_direct(points, values)
                outputs = jax.lax.map(
                    lambda chunk: self._type2_direct(chunk, values),
                    _chunks(points, self.plan.chunk_size),
                )
                rows = outputs.shape[0] * outputs.shape[1]
                return outputs.reshape((rows, *outputs.shape[2:]))[:point_count]
            case _:
                raise ValueError(
                    f"Unsupported nonuniform Fourier route {self.plan.route!r}."
                )

    def _type1_direct(self, points: Array, values: Array, /) -> Array:
        dtype = _complex_dtype(points, values, *self.modes)
        result = values.astype(dtype)
        imaginary_unit = jnp.asarray(1j, dtype=dtype)
        for axis, mode in enumerate(self.modes):
            angle = self.plan.sign * points[:, axis, None] * mode[None, :]
            phase = jnp.exp(imaginary_unit * angle)
            result = ein.contract("m...,mk->m...k", result, phase)
        return ein.contract("m...->...", result)

    def _type1_gridded(self, points: Array, values: Array, /) -> Array:
        kernel, fine_shape = self._gridding()
        dtype = _complex_dtype(points, values, *self.modes)
        dimension = len(fine_shape)
        payload = values.shape[1:]
        periodic = jnp.mod(points.astype(jnp.finfo(dtype).dtype), _TWO_PI)
        grid = _spread(
            periodic,
            values.astype(dtype),
            tuple(size / _TWO_PI for size in fine_shape),
            (0.0,) * dimension,
            fine_shape,
            kernel,
            self.plan.chunk_size,
        )
        grid = _fine_fourier_sum(
            grid.reshape((*fine_shape, *payload)), self.plan.sign, dimension
        )
        return _scale_modes(grid[jnp.ix_(*self.fine_indices)], self.corrections)

    def type1(self, coordinates: ArrayLike, strengths: ArrayLike, /) -> Array:
        if self.plan.transform_type != 1:
            raise ValueError("Prepared plan is not Type 1.")
        points = jnp.asarray(coordinates)
        values = jnp.asarray(strengths)
        if points.ndim != 2 or points.shape[1] != len(self.modes):
            raise ValueError(
                "Nonuniform Fourier coordinates must have shape (points, dimension)."
            )
        if jnp.issubdtype(points.dtype, jnp.complexfloating):
            raise TypeError("Nonuniform Fourier coordinates must be real-valued.")
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Nonuniform Fourier strengths need one leading value per point."
            )
        match self.plan.route:
            case "direct":
                return self._type1_direct(points, values)
            case "gridded":
                return self._type1_gridded(points, values)
            case "chunked":
                if points.shape[0] == 0:
                    return self._type1_direct(points, values)
                outputs = jax.lax.map(
                    lambda data: self._type1_direct(data[0], data[1]),
                    (
                        _chunks(points, self.plan.chunk_size),
                        _chunks(values, self.plan.chunk_size),
                    ),
                )
                return jnp.sum(outputs, axis=0)
            case _:
                raise ValueError(
                    f"Unsupported nonuniform Fourier route {self.plan.route!r}."
                )


def _finite_tuple(values: Sequence[float], name: str, /) -> tuple[float, ...]:
    return tuple(finite_real_scalar(value, name) for value in values)


class NonuniformFourierType3Plan(StrictModule):
    """Static Type-3 transform ``f_k = sum_j c_j exp(i sign s_k . x_j)``.

    Sources lie in the box ``source_center +/- source_half_width`` and targets
    in ``target_center +/- target_half_width``. The ES kernel spreads rescaled
    sources onto a nonperiodic grid of ``2 sigma S X / pi + width + 1`` points
    per axis; a gridded Type-2 plan with the same kernel evaluates that grid at
    rescaled targets. Each of the two gridding stages receives half of
    ``tolerance``, split over axes like a Type-1/2 plan.
    """

    source_center: tuple[float, ...] = eqx.field(static=True)
    source_half_width: tuple[float, ...] = eqx.field(static=True)
    target_center: tuple[float, ...] = eqx.field(static=True)
    target_half_width: tuple[float, ...] = eqx.field(static=True)
    sign: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    kernel: NonuniformFourierKernel = eqx.field(static=True)
    fine_shape: tuple[int, ...] = eqx.field(static=True)
    inner: NonuniformFourierPlan = eqx.field(static=True)
    maximum_grid_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_center: Sequence[float],
        source_half_width: Sequence[float],
        target_center: Sequence[float],
        target_half_width: Sequence[float],
        /,
        *,
        tolerance: float,
        sign: int = 1,
        chunk_size: int = 256,
        maximum_grid_points: int = 2**24,
    ) -> None:
        sources = _finite_tuple(source_center, "Type-3 source_center")
        targets = _finite_tuple(target_center, "Type-3 target_center")
        source_widths = tuple(
            positive_finite_float(value, "Type-3 source_half_width")
            for value in source_half_width
        )
        target_widths = tuple(
            positive_finite_float(value, "Type-3 target_half_width")
            for value in target_half_width
        )
        dimension = len(sources)
        if not 1 <= dimension <= 3 or any(
            len(values) != dimension for values in (targets, source_widths, target_widths)
        ):
            raise ValueError(
                "Type-3 centers and half widths need one to three matching axes."
            )
        exponent_sign = _static_int(sign, "Nonuniform Fourier sign")
        if exponent_sign not in (-1, 1):
            raise ValueError("Nonuniform Fourier sign must be -1 or 1.")
        chunk = _static_int(chunk_size, "Nonuniform Fourier chunk_size")
        if chunk < 1:
            raise ValueError("Nonuniform Fourier chunk_size must be positive.")
        capacity = _static_int(
            maximum_grid_points, "Nonuniform Fourier maximum_grid_points"
        )
        if capacity < 1:
            raise ValueError("Nonuniform Fourier maximum_grid_points must be positive.")
        requested = _requested_tolerance(tolerance)
        # The spreading and inner Type-2 stages each receive half the budget;
        # this kernel equals the one the inner plan selects.
        kernel = NonuniformFourierKernel(
            _TOLERANCE_MARGIN * (0.5 * requested) / dimension
        )
        margin = kernel.width + 1
        fine_shape = tuple(
            _smooth_even_size(
                max(
                    2.0 * kernel.oversampling * source * target / math.pi + margin,
                    2 * margin,
                )
            )
            for source, target in zip(source_widths, target_widths, strict=True)
        )
        _refuse_grid(math.prod(fine_shape), capacity)
        inner = NonuniformFourierPlan(
            fine_shape,
            2,
            sign=exponent_sign,
            centered=True,
            route="gridded",
            chunk_size=chunk,
            tolerance=0.5 * requested,
            maximum_grid_points=capacity,
        )
        inner_shape = inner.fine_shape
        if inner_shape is None:
            raise ValueError("Type-3 inner transform must be gridded.")
        _refuse_grid(math.prod(fine_shape) + math.prod(inner_shape), capacity)
        self.source_center = sources
        self.source_half_width = source_widths
        self.target_center = targets
        self.target_half_width = target_widths
        self.sign = exponent_sign
        self.chunk_size = chunk
        self.tolerance = requested
        self.kernel = kernel
        self.fine_shape = fine_shape
        self.inner = inner
        self.maximum_grid_points = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nonuniform-fourier-type3-plan",
                "source_center": sources,
                "source_half_width": source_widths,
                "target_center": targets,
                "target_half_width": target_widths,
                "sign": exponent_sign,
                "chunk_size": chunk,
                "tolerance": requested,
                "fine_shape": fine_shape,
            }
        )

    @property
    def spacings(self) -> tuple[float, ...]:
        return tuple(_TWO_PI / size for size in self.fine_shape)

    @property
    def source_scales(self) -> tuple[float, ...]:
        """Fine-grid cells per source unit, ``1 / (gamma h)``."""
        return tuple(
            self.kernel.oversampling * width / math.pi for width in self.target_half_width
        )

    @property
    def target_scales(self) -> tuple[float, ...]:
        """Kernel frequency per target unit, ``gamma = n / (2 sigma S)``."""
        return tuple(
            size / (2.0 * self.kernel.oversampling * width)
            for size, width in zip(self.fine_shape, self.target_half_width, strict=True)
        )


class NonuniformFourierType3Result(StrictModule):
    """Type-3 values and per-target support.

    ``supported`` is false for a target outside its declared box, and for every
    target when any source lies outside its declared box. Unsupported values
    are returned as computed, not replaced.
    """

    values: Array
    supported: Array


class PreparedNonuniformFourierType3(StrictModule, NonTrainableState):
    """Prepared Type-3 kernel quadrature and inner gridded Type-2 transform."""

    source_center: Array
    target_center: Array
    kernel_rule: _KernelTransformRule
    inner: PreparedNonuniformFourier
    plan: NonuniformFourierType3Plan = eqx.field(static=True)
    evidence: NonuniformFourierGridEvidence = eqx.field(static=True)

    def __init__(self, plan: NonuniformFourierType3Plan, /, *, dtype: DTypeLike) -> None:
        if not isinstance(plan, NonuniformFourierType3Plan):
            raise TypeError("plan must be a NonuniformFourierType3Plan.")
        resolved_dtype = _real_floating_dtype(dtype)
        _validate_precision(plan.tolerance, resolved_dtype)
        self.source_center = jnp.asarray(plan.source_center, dtype=resolved_dtype)
        self.target_center = jnp.asarray(plan.target_center, dtype=resolved_dtype)
        self.kernel_rule = _KernelTransformRule(plan.kernel, resolved_dtype)
        self.inner = PreparedNonuniformFourier(plan.inner, dtype=resolved_dtype)
        inner_evidence = self.inner.evidence
        if inner_evidence is None:
            raise ValueError("Type-3 inner transform must be gridded.")
        self.plan = plan
        self.evidence = _grid_evidence(
            plan.tolerance,
            plan.kernel,
            plan.fine_shape,
            math.prod(plan.fine_shape) + inner_evidence.grid_points,
            plan.maximum_grid_points,
            plan.chunk_size,
            resolved_dtype,
        )

    def apply(
        self,
        sources: ArrayLike,
        strengths: ArrayLike,
        targets: ArrayLike,
        /,
    ) -> NonuniformFourierType3Result:
        plan = self.plan
        dimension = len(plan.fine_shape)
        source_points = jnp.asarray(sources)
        values = jnp.asarray(strengths)
        target_points = jnp.asarray(targets)
        for points, name in ((source_points, "sources"), (target_points, "targets")):
            if points.ndim != 2 or points.shape[1] != dimension:
                raise ValueError(f"Type-3 {name} must have shape (points, {dimension}).")
            if jnp.issubdtype(points.dtype, jnp.complexfloating):
                raise TypeError(f"Type-3 {name} must be real-valued.")
        if values.shape[:1] != source_points.shape[:1]:
            raise ValueError("Type-3 strengths need one leading value per source.")
        dtype = _complex_dtype(source_points, target_points, values, self.source_center)
        real_dtype = jnp.finfo(dtype).dtype
        source_offsets = (
            source_points.astype(real_dtype)
            - self.source_center.astype(real_dtype)[None, :]
        )
        target_offsets = (
            target_points.astype(real_dtype)
            - self.target_center.astype(real_dtype)[None, :]
        )
        # s.x = D.C + D.(x - C) + (s - D).x splits the target-center phase
        # onto sources and the source-center phase onto targets.
        source_phase = jnp.exp(
            1j
            * (
                plan.sign * (source_offsets @ self.target_center.astype(real_dtype))
            ).astype(dtype)
        )
        grid = _spread(
            source_offsets,
            ein.contract("j,j...->j...", source_phase, values.astype(dtype)),
            plan.source_scales,
            tuple(0.5 * size for size in plan.fine_shape),
            plan.fine_shape,
            plan.kernel,
            plan.chunk_size,
        )
        spacings = plan.spacings
        kernel_frequencies = (
            target_offsets * jnp.asarray(plan.target_scales, dtype=real_dtype)[None, :]
        )
        grid_values = self.inner.type2(
            kernel_frequencies * jnp.asarray(spacings, dtype=real_dtype)[None, :],
            grid.reshape((*plan.fine_shape, *values.shape[1:])),
        )
        deconvolution = jnp.ones(target_points.shape[:1], dtype=real_dtype)
        for axis, spacing in enumerate(spacings):
            deconvolution = deconvolution * (
                spacing / self.kernel_rule.transform(kernel_frequencies[:, axis], spacing)
            )
        target_phase = jnp.exp(
            1j
            * (
                plan.sign
                * (
                    target_points.astype(real_dtype)
                    @ self.source_center.astype(real_dtype)
                )
            ).astype(dtype)
        )
        sources_supported = jnp.all(
            jnp.abs(source_offsets)
            <= jnp.asarray(plan.source_half_width, dtype=real_dtype)[None, :]
        )
        targets_supported = jnp.all(
            jnp.abs(target_offsets)
            <= jnp.asarray(plan.target_half_width, dtype=real_dtype)[None, :],
            axis=1,
        )
        return NonuniformFourierType3Result(
            ein.contract(
                "k,k...->k...",
                target_phase * deconvolution.astype(dtype),
                grid_values,
            ),
            targets_supported & sources_supported,
        )


__all__ = [
    "NonuniformFourierGridEvidence",
    "NonuniformFourierKernel",
    "NonuniformFourierPlan",
    "NonuniformFourierResourceError",
    "NonuniformFourierRoute",
    "NonuniformFourierType",
    "NonuniformFourierType3Plan",
    "NonuniformFourierType3Result",
    "PreparedNonuniformFourier",
    "PreparedNonuniformFourierType3",
]
