#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import ChartTransition, CoordinateChart
from ._map import DifferentiableMap, Immersion
from ._utils import _pointwise_array


class ComplexCoordinateConvention(StrictModule):
    """Pair real chart axes into ordered complex coordinates."""

    chart: CoordinateChart
    pairs: tuple[tuple[int, int], ...]

    def __init__(
        self,
        chart: CoordinateChart,
        pairs: Sequence[tuple[int, int]] | None = None,
        /,
    ):
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Complex coordinates require a CoordinateChart.")
        if chart.dimension % 2:
            raise ValueError("Complex coordinates require even real dimension.")
        half = chart.dimension // 2
        pairs_ = (
            tuple((index, half + index) for index in range(half))
            if pairs is None
            else tuple((int(real), int(imag)) for real, imag in pairs)
        )
        axes = tuple(axis for pair in pairs_ for axis in pair)
        if len(pairs_) != half or tuple(sorted(axes)) != tuple(range(chart.dimension)):
            raise ValueError("Complex coordinate pairs must partition all chart axes.")
        self.chart = chart
        self.pairs = pairs_

    @property
    def complex_dimension(self) -> int:
        return len(self.pairs)

    @property
    def real_axes(self) -> tuple[int, ...]:
        return tuple(real for real, _ in self.pairs)

    @property
    def imaginary_axes(self) -> tuple[int, ...]:
        return tuple(imaginary for _, imaginary in self.pairs)

    def to_complex(self, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinates)
        if values.shape[-1:] != (self.chart.dimension,):
            raise ValueError("Real coordinates must match the convention chart.")
        return (
            values[..., jnp.asarray(self.real_axes)]
            + 1j * values[..., jnp.asarray(self.imaginary_axes)]
        )

    def to_real(self, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinates)
        if values.shape[-1:] != (self.complex_dimension,):
            raise ValueError("Complex coordinates must match the convention dimension.")
        result = jnp.zeros(
            values.shape[:-1] + (self.chart.dimension,), dtype=values.real.dtype
        )
        result = result.at[..., jnp.asarray(self.real_axes)].set(jnp.real(values))
        return result.at[..., jnp.asarray(self.imaginary_axes)].set(jnp.imag(values))

    def standard_matrix(self, *, dtype=jnp.float64) -> Array:
        matrix = jnp.zeros((self.chart.dimension, self.chart.dimension), dtype=dtype)
        for real, imaginary in self.pairs:
            matrix = matrix.at[real, imaginary].set(-1.0)
            matrix = matrix.at[imaginary, real].set(1.0)
        return matrix


class _ConstantComplexStructure(StrictModule):
    matrix: Array

    def __init__(self, matrix: ArrayLike, /):
        self.matrix = jnp.asarray(matrix)

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.asarray(self.matrix, dtype=coordinates.dtype)


class AlmostComplexStructure(StrictModule):
    """A real tangent endomorphism candidate satisfying ``J² = -I``."""

    matrix_function: Callable[[Array], Array]
    chart: CoordinateChart

    def __init__(
        self,
        matrix: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
    ):
        if not callable(matrix):
            raise TypeError("Almost-complex matrix must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Almost-complex structure requires a CoordinateChart.")
        if chart.dimension % 2:
            raise ValueError("Almost-complex structures require even real dimension.")
        self.matrix_function = matrix
        self.chart = chart

    def _matrix_point(self, coordinates: Array, /) -> Array:
        matrix = jnp.asarray(self.matrix_function(coordinates))
        expected = (self.chart.dimension, self.chart.dimension)
        if matrix.shape != expected:
            raise ValueError(
                f"Almost-complex matrix must have shape {expected}; got {matrix.shape}."
            )
        return matrix

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._matrix_point,
            coordinates,
            self.chart.dimension,
        )

    def apply(self, vector: ArrayLike, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(vector)
        if values.shape[-1:] != (self.chart.dimension,):
            raise ValueError("Almost-complex vectors must match the chart dimension.")
        return oe.contract("...ij,...j->...i", self(coordinates), values)

    def apply_covector(self, covector: ArrayLike, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(covector)
        if values.shape[-1:] != (self.chart.dimension,):
            raise ValueError("Almost-complex covectors must match the chart dimension.")
        return oe.contract("...i,...ij->...j", values, self(coordinates))


class AlmostComplexValidationReport(StrictModule):
    valid: Array
    finite: Array
    algebra_residual: Array
    nijenhuis_residual: Array
    integrable: Array

    def __init__(
        self,
        *,
        valid: ArrayLike,
        finite: ArrayLike,
        algebra_residual: ArrayLike,
        nijenhuis_residual: ArrayLike,
        integrable: ArrayLike,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.algebra_residual = jnp.asarray(algebra_residual)
        self.nijenhuis_residual = jnp.asarray(nijenhuis_residual)
        self.integrable = jnp.asarray(integrable, dtype=bool)


def standard_complex_structure(
    convention: ComplexCoordinateConvention,
    /,
) -> AlmostComplexStructure:
    if not isinstance(convention, ComplexCoordinateConvention):
        raise TypeError("standard_complex_structure requires a coordinate convention.")
    return AlmostComplexStructure(
        _ConstantComplexStructure(convention.standard_matrix()),
        chart=convention.chart,
    )


def nijenhuis_tensor(
    structure: AlmostComplexStructure,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``N[..., k, i, j]`` for an almost-complex structure."""
    if not isinstance(structure, AlmostComplexStructure):
        raise TypeError("nijenhuis_tensor requires an AlmostComplexStructure.")

    def evaluate(point: Array) -> Array:
        matrix = structure._matrix_point(point)
        derivative = jax.jacfwd(structure._matrix_point)(point)
        first = oe.contract("li,kjl->kij", matrix, derivative)
        second = oe.contract("lj,kil->kij", matrix, derivative)
        third = oe.contract("kl,lji->kij", matrix, derivative)
        fourth = oe.contract("kl,lij->kij", matrix, derivative)
        return first - second - third + fourth

    return _pointwise_array(evaluate, coordinates, structure.chart.dimension)


def validate_almost_complex_structure(
    structure: AlmostComplexStructure,
    points: ArrayLike,
    /,
    *,
    algebra_tolerance: float = 1e-9,
    integrability_tolerance: float = 1e-8,
    require_integrable: bool = False,
    raise_on_error: bool = True,
) -> AlmostComplexValidationReport:
    if not isinstance(structure, AlmostComplexStructure):
        raise TypeError("structure must be an AlmostComplexStructure.")
    if algebra_tolerance < 0.0 or integrability_tolerance < 0.0:
        raise ValueError("Almost-complex tolerances must be non-negative.")
    matrix = structure(points)
    identity = jnp.eye(structure.chart.dimension, dtype=matrix.dtype)
    algebra_residual = jnp.max(jnp.abs(matrix @ matrix + identity))
    nijenhuis_residual = jnp.max(jnp.abs(nijenhuis_tensor(structure, points)))
    finite = jnp.all(jnp.isfinite(matrix))
    integrable = nijenhuis_residual <= integrability_tolerance
    valid = finite & (algebra_residual <= algebra_tolerance)
    if require_integrable:
        valid = valid & integrable
    report = AlmostComplexValidationReport(
        valid=valid,
        finite=finite,
        algebra_residual=algebra_residual,
        nijenhuis_residual=nijenhuis_residual,
        integrable=integrable,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Almost-complex validation failed: "
            f"algebra_residual={float(jax.device_get(algebra_residual))}, "
            f"nijenhuis_residual={float(jax.device_get(nijenhuis_residual))}."
        )
    return report


def holomorphicity_residual(
    map: DifferentiableMap | Immersion | ChartTransition,
    source: AlmostComplexStructure,
    target: AlmostComplexStructure,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the maximum residual of ``Dφ J_source = J_target Dφ``."""
    if not isinstance(map, (DifferentiableMap, Immersion, ChartTransition)):
        raise TypeError("map must be a differentiable coordinate map.")
    if not map.source.compatible_with(source.chart) or not map.target.compatible_with(
        target.chart
    ):
        raise ValueError("Map and almost-complex charts must match.")
    jacobian = map.jacobian(coordinates)
    difference = jacobian @ source(coordinates) - target(map(coordinates)) @ jacobian
    return jnp.max(jnp.abs(difference), axis=(-2, -1))


def wirtinger_derivatives(
    field: Callable[[Array], Array],
    convention: ComplexCoordinateConvention,
    coordinates: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Return pointwise ``∂field`` and ``∂̄field`` in a real chart."""
    if not callable(field):
        raise TypeError("field must be callable.")
    if not isinstance(convention, ComplexCoordinateConvention):
        raise TypeError("convention must be a ComplexCoordinateConvention.")

    def evaluate(point: Array) -> tuple[Array, Array]:
        derivative = jax.jacfwd(field)(point)
        real = derivative[..., jnp.asarray(convention.real_axes)]
        imaginary = derivative[..., jnp.asarray(convention.imaginary_axes)]
        return 0.5 * (real - 1j * imaginary), 0.5 * (real + 1j * imaginary)

    points = jnp.asarray(coordinates)
    if points.shape[-1:] != (convention.chart.dimension,):
        raise ValueError("Wirtinger coordinates must match the convention chart.")
    if points.ndim == 1:
        return evaluate(points)
    leading = points.shape[:-1]
    flat = points.reshape((-1, convention.chart.dimension))
    first, second = jax.vmap(evaluate)(flat)
    return (
        first.reshape(leading + first.shape[1:]),
        second.reshape(leading + second.shape[1:]),
    )


__all__ = [
    "AlmostComplexStructure",
    "AlmostComplexValidationReport",
    "ComplexCoordinateConvention",
    "holomorphicity_residual",
    "nijenhuis_tensor",
    "standard_complex_structure",
    "validate_almost_complex_structure",
    "wirtinger_derivatives",
]
