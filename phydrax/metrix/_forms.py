#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import comb

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chart import CoordinateChart
from ._map import DifferentiableMap
from ._metric import AbstractSemiRiemannianMetric
from ._utils import _pointwise_array


def _multi_indices(dimension: int, degree: int, /) -> tuple[tuple[int, ...], ...]:
    return tuple(combinations(range(dimension), degree))


def _wedge_sign(left: tuple[int, ...], right: tuple[int, ...], /) -> int:
    inversions = sum(left_axis > right_axis for left_axis in left for right_axis in right)
    return -1 if inversions % 2 else 1


def _require_same_chart(
    left: CoordinateChart,
    right: CoordinateChart,
    /,
) -> None:
    if not left.compatible_with(right):
        raise ValueError(
            f"Differential-form charts do not match: {left.name!r} and {right.name!r}."
        )


class DifferentialForm(StrictModule):
    """A coordinate differential form stored on increasing multi-indices."""

    coefficient_function: Callable[[Array], Array]
    chart: CoordinateChart
    degree: int
    indices: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        coefficients: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
        degree: int,
    ):
        degree_value = int(degree)
        if not callable(coefficients):
            raise TypeError("Differential-form coefficients must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Differential-form chart must be a CoordinateChart.")
        if degree_value < 0 or degree_value > chart.dimension:
            raise ValueError(
                f"Form degree must lie in [0, {chart.dimension}]; got {degree_value}."
            )
        self.coefficient_function = coefficients
        self.chart = chart
        self.degree = degree_value
        self.indices = _multi_indices(chart.dimension, degree_value)

    @property
    def coefficient_count(self) -> int:
        return comb(self.chart.dimension, self.degree)

    def _coefficients_point(self, coordinates: Array, /) -> Array:
        values = jnp.asarray(self.coefficient_function(coordinates))
        if self.degree == 0 and values.shape == ():
            values = values[None]
        expected = (self.coefficient_count,)
        if values.shape != expected:
            raise ValueError(
                f"Degree-{self.degree} form coefficients must have shape {expected}; "
                f"got {values.shape}."
            )
        return values

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        return _pointwise_array(
            self._coefficients_point,
            coordinates,
            self.chart.dimension,
        )


class _WedgeCoefficient(StrictModule):
    left: DifferentialForm
    right: DifferentialForm
    left_terms: tuple[int, ...]
    right_terms: tuple[int, ...]
    output_terms: tuple[int, ...]
    signs: tuple[int, ...]
    output_count: int

    def __init__(self, left: DifferentialForm, right: DifferentialForm, /):
        output_indices = _multi_indices(left.chart.dimension, left.degree + right.degree)
        output_lookup = {index: position for position, index in enumerate(output_indices)}
        left_terms: list[int] = []
        right_terms: list[int] = []
        result_terms: list[int] = []
        signs: list[int] = []
        for left_position, left_index in enumerate(left.indices):
            left_set = set(left_index)
            for right_position, right_index in enumerate(right.indices):
                if left_set.isdisjoint(right_index):
                    left_terms.append(left_position)
                    right_terms.append(right_position)
                    result_terms.append(
                        output_lookup[tuple(sorted(left_index + right_index))]
                    )
                    signs.append(_wedge_sign(left_index, right_index))
        self.left = left
        self.right = right
        self.left_terms = tuple(left_terms)
        self.right_terms = tuple(right_terms)
        self.output_terms = tuple(result_terms)
        self.signs = tuple(signs)
        self.output_count = len(output_indices)

    def __call__(self, coordinates: Array, /) -> Array:
        left = self.left._coefficients_point(coordinates)
        right = self.right._coefficients_point(coordinates)
        left_terms = jnp.asarray(self.left_terms, dtype=jnp.int32)
        right_terms = jnp.asarray(self.right_terms, dtype=jnp.int32)
        output_terms = jnp.asarray(self.output_terms, dtype=jnp.int32)
        signs = jnp.asarray(self.signs, dtype=left.dtype)
        terms = signs * left[left_terms] * right[right_terms]
        return (
            jnp.zeros((self.output_count,), dtype=terms.dtype).at[output_terms].add(terms)
        )


class _ExteriorDerivativeCoefficient(StrictModule):
    form: DifferentialForm
    source_terms: tuple[int, ...]
    derivative_axes: tuple[int, ...]
    output_terms: tuple[int, ...]
    signs: tuple[int, ...]
    output_count: int

    def __init__(self, form: DifferentialForm, /):
        output_indices = _multi_indices(form.chart.dimension, form.degree + 1)
        source_lookup = {index: position for position, index in enumerate(form.indices)}
        source_terms: list[int] = []
        derivative_axes: list[int] = []
        result_terms: list[int] = []
        signs: list[int] = []
        for output_position, output_index in enumerate(output_indices):
            for position, axis in enumerate(output_index):
                source_index = output_index[:position] + output_index[position + 1 :]
                source_terms.append(source_lookup[source_index])
                derivative_axes.append(axis)
                result_terms.append(output_position)
                signs.append(-1 if position % 2 else 1)
        self.form = form
        self.source_terms = tuple(source_terms)
        self.derivative_axes = tuple(derivative_axes)
        self.output_terms = tuple(result_terms)
        self.signs = tuple(signs)
        self.output_count = len(output_indices)

    def __call__(self, coordinates: Array, /) -> Array:
        derivative = jax.jacfwd(self.form._coefficients_point)(coordinates)
        source_terms = jnp.asarray(self.source_terms, dtype=jnp.int32)
        derivative_axes = jnp.asarray(self.derivative_axes, dtype=jnp.int32)
        output_terms = jnp.asarray(self.output_terms, dtype=jnp.int32)
        signs = jnp.asarray(self.signs, dtype=derivative.dtype)
        terms = signs * derivative[source_terms, derivative_axes]
        return (
            jnp.zeros((self.output_count,), dtype=terms.dtype).at[output_terms].add(terms)
        )


class _PullbackFormCoefficient(StrictModule):
    form: DifferentialForm
    map: DifferentiableMap
    target_indices: tuple[tuple[int, ...], ...]
    source_indices: tuple[tuple[int, ...], ...]

    def __init__(self, form: DifferentialForm, map: DifferentiableMap, /):
        self.form = form
        self.map = map
        self.target_indices = form.indices
        self.source_indices = _multi_indices(map.source.dimension, form.degree)

    def __call__(self, coordinates: Array, /) -> Array:
        target_coordinates = self.map.map_function(coordinates)
        coefficients = self.form._coefficients_point(target_coordinates)
        if self.form.degree == 0:
            return coefficients
        jacobian = self.map.jacobian(coordinates)
        target_indices = jnp.asarray(self.target_indices, dtype=jnp.int32)
        source_indices = jnp.asarray(self.source_indices, dtype=jnp.int32)
        rows = target_indices[:, None, :, None]
        columns = source_indices[None, :, None, :]
        minors = jacobian[rows, columns]
        determinants = jnp.linalg.det(minors)
        return jnp.sum(coefficients[:, None] * determinants, axis=0)


class _InteriorProductCoefficient(StrictModule):
    vector_field: Callable[[Array], Array]
    form: DifferentialForm
    vector_terms: tuple[int, ...]
    source_terms: tuple[int, ...]
    output_terms: tuple[int, ...]
    signs: tuple[int, ...]
    output_count: int

    def __init__(
        self,
        vector_field: Callable[[Array], Array],
        form: DifferentialForm,
        /,
    ):
        output_indices = _multi_indices(form.chart.dimension, form.degree - 1)
        source_lookup = {index: position for position, index in enumerate(form.indices)}
        vector_terms: list[int] = []
        source_terms: list[int] = []
        result_terms: list[int] = []
        signs: list[int] = []
        for output_position, output_index in enumerate(output_indices):
            output_set = set(output_index)
            for axis in range(form.chart.dimension):
                if axis in output_set:
                    continue
                source_index = tuple(sorted((axis,) + output_index))
                insertion_position = source_index.index(axis)
                vector_terms.append(axis)
                source_terms.append(source_lookup[source_index])
                result_terms.append(output_position)
                signs.append(-1 if insertion_position % 2 else 1)
        self.vector_field = vector_field
        self.form = form
        self.vector_terms = tuple(vector_terms)
        self.source_terms = tuple(source_terms)
        self.output_terms = tuple(result_terms)
        self.signs = tuple(signs)
        self.output_count = len(output_indices)

    def __call__(self, coordinates: Array, /) -> Array:
        vector = jnp.asarray(self.vector_field(coordinates))
        expected = (self.form.chart.dimension,)
        if vector.shape != expected:
            raise ValueError(
                f"Interior-product vector field must have shape {expected}; "
                f"got {vector.shape}."
            )
        coefficients = self.form._coefficients_point(coordinates)
        vector_terms = jnp.asarray(self.vector_terms, dtype=jnp.int32)
        source_terms = jnp.asarray(self.source_terms, dtype=jnp.int32)
        output_terms = jnp.asarray(self.output_terms, dtype=jnp.int32)
        signs = jnp.asarray(self.signs, dtype=coefficients.dtype)
        terms = signs * vector[vector_terms] * coefficients[source_terms]
        return (
            jnp.zeros((self.output_count,), dtype=terms.dtype).at[output_terms].add(terms)
        )


class _SumFormCoefficient(StrictModule):
    left: DifferentialForm
    right: DifferentialForm

    def __init__(self, left: DifferentialForm, right: DifferentialForm, /):
        self.left = left
        self.right = right

    def __call__(self, coordinates: Array, /) -> Array:
        return self.left._coefficients_point(
            coordinates
        ) + self.right._coefficients_point(coordinates)


class _ScaledFormCoefficient(StrictModule):
    form: DifferentialForm
    scale: float

    def __init__(self, form: DifferentialForm, scale: float, /):
        self.form = form
        self.scale = float(scale)

    def __call__(self, coordinates: Array, /) -> Array:
        return self.scale * self.form._coefficients_point(coordinates)


class _HodgeStarCoefficient(StrictModule):
    form: DifferentialForm
    metric: AbstractSemiRiemannianMetric
    source_indices: tuple[tuple[int, ...], ...]
    output_indices: tuple[tuple[int, ...], ...]
    output_terms: tuple[int, ...]
    signs: tuple[int, ...]
    orientation: int

    def __init__(
        self,
        form: DifferentialForm,
        metric: AbstractSemiRiemannianMetric,
        orientation: int,
        /,
    ):
        source = form.indices
        output = _multi_indices(form.chart.dimension, form.chart.dimension - form.degree)
        output_lookup = {index: position for position, index in enumerate(output)}
        complements: list[tuple[int, ...]] = []
        output_terms: list[int] = []
        signs: list[int] = []
        full = set(range(form.chart.dimension))
        for source_index in source:
            complement = tuple(sorted(full.difference(source_index)))
            complements.append(complement)
            output_terms.append(output_lookup[complement])
            signs.append(_wedge_sign(source_index, complement))
        self.form = form
        self.metric = metric
        self.source_indices = source
        self.output_indices = output
        self.output_terms = tuple(output_terms)
        self.signs = tuple(signs)
        self.orientation = int(orientation)

    def __call__(self, coordinates: Array, /) -> Array:
        coefficients = self.form._coefficients_point(coordinates)
        source_indices = jnp.asarray(self.source_indices, dtype=jnp.int32)
        if self.form.degree == 0:
            paired = coefficients
        else:
            inverse = self.metric.inverse(coordinates)
            rows = source_indices[:, None, :, None]
            columns = source_indices[None, :, None, :]
            induced_inverse = jnp.linalg.det(inverse[rows, columns])
            paired = induced_inverse @ coefficients
        signs = jnp.asarray(self.signs, dtype=coefficients.dtype)
        output_terms = jnp.asarray(self.output_terms, dtype=jnp.int32)
        values = (
            self.orientation * self.metric.volume_density(coordinates) * signs * paired
        )
        return (
            jnp.zeros((len(self.output_indices),), dtype=values.dtype)
            .at[output_terms]
            .set(values)
        )


def wedge(left: DifferentialForm, right: DifferentialForm, /) -> DifferentialForm:
    """Return the graded-antisymmetric wedge product."""
    if not isinstance(left, DifferentialForm) or not isinstance(right, DifferentialForm):
        raise TypeError("wedge requires two DifferentialForm instances.")
    _require_same_chart(left.chart, right.chart)
    degree = left.degree + right.degree
    if degree > left.chart.dimension:
        raise ValueError("Wedge-product degree exceeds the chart dimension.")
    return DifferentialForm(
        _WedgeCoefficient(left, right),
        chart=left.chart,
        degree=degree,
    )


def exterior_derivative(form: DifferentialForm, /) -> DifferentialForm:
    """Return the metric-independent exterior derivative ``d form``."""
    if not isinstance(form, DifferentialForm):
        raise TypeError("exterior_derivative requires a DifferentialForm.")
    if form.degree == form.chart.dimension:
        raise ValueError("The exterior derivative of a top form is identically zero.")
    return DifferentialForm(
        _ExteriorDerivativeCoefficient(form),
        chart=form.chart,
        degree=form.degree + 1,
    )


def pullback_form(
    form: DifferentialForm,
    map: DifferentiableMap,
    /,
) -> DifferentialForm:
    """Pull a form through a differentiable map using Jacobian minors."""
    if not isinstance(form, DifferentialForm):
        raise TypeError("pullback_form requires a DifferentialForm.")
    if not isinstance(map, DifferentiableMap):
        raise TypeError("pullback_form requires a DifferentiableMap.")
    if not map.target.compatible_with(form.chart):
        raise ValueError("Differentiable-map target chart must match the form chart.")
    if form.degree > map.source.dimension:
        raise ValueError("Form degree exceeds the source chart dimension.")
    return DifferentialForm(
        _PullbackFormCoefficient(form, map),
        chart=map.source,
        degree=form.degree,
    )


def interior_product(
    vector_field: Callable[[Array], Array],
    form: DifferentialForm,
    /,
) -> DifferentialForm:
    """Contract a vector field into the first slot of a positive-degree form."""
    if not callable(vector_field):
        raise TypeError("vector_field must be callable.")
    if not isinstance(form, DifferentialForm):
        raise TypeError("interior_product requires a DifferentialForm.")
    if form.degree == 0:
        raise ValueError("Interior product of a zero-form is identically zero.")
    return DifferentialForm(
        _InteriorProductCoefficient(vector_field, form),
        chart=form.chart,
        degree=form.degree - 1,
    )


def _add_forms(left: DifferentialForm, right: DifferentialForm, /) -> DifferentialForm:
    _require_same_chart(left.chart, right.chart)
    if left.degree != right.degree:
        raise ValueError("Only equal-degree forms can be added.")
    return DifferentialForm(
        _SumFormCoefficient(left, right),
        chart=left.chart,
        degree=left.degree,
    )


def lie_derivative(
    vector_field: Callable[[Array], Array],
    form: DifferentialForm,
    /,
) -> DifferentialForm:
    """Return Cartan's ``L_X form = d i_X form + i_X d form``."""
    if form.degree == 0:
        return interior_product(vector_field, exterior_derivative(form))
    first = exterior_derivative(interior_product(vector_field, form))
    if form.degree == form.chart.dimension:
        return first
    second = interior_product(vector_field, exterior_derivative(form))
    return _add_forms(first, second)


def hodge_star(
    form: DifferentialForm,
    metric: AbstractSemiRiemannianMetric,
    /,
    *,
    orientation: int = 1,
) -> DifferentialForm:
    """Return the Hodge dual for a nondegenerate metric and orientation."""
    if not isinstance(form, DifferentialForm):
        raise TypeError("hodge_star requires a DifferentialForm.")
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("hodge_star requires a nondegenerate metric.")
    _require_same_chart(form.chart, metric.chart)
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1.")
    return DifferentialForm(
        _HodgeStarCoefficient(form, metric, orientation),
        chart=form.chart,
        degree=form.chart.dimension - form.degree,
    )


def codifferential(
    form: DifferentialForm,
    metric: AbstractSemiRiemannianMetric,
    /,
    *,
    orientation: int = 1,
) -> DifferentialForm:
    """Return the metric codifferential under the declared orientation."""
    if not isinstance(form, DifferentialForm):
        raise TypeError("codifferential requires a DifferentialForm.")
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("codifferential requires a nondegenerate metric.")
    _require_same_chart(form.chart, metric.chart)
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1.")
    if form.degree == 0:
        return DifferentialForm(
            lambda coordinates: jnp.zeros((1,), dtype=coordinates.dtype),
            chart=form.chart,
            degree=0,
        )
    first = hodge_star(form, metric, orientation=orientation)
    derivative = exterior_derivative(first)
    result = hodge_star(derivative, metric, orientation=orientation)
    exponent = form.chart.dimension * (form.degree + 1) + metric.signature.index + 1
    sign = -1 if exponent % 2 else 1
    return DifferentialForm(
        _ScaledFormCoefficient(result, sign),
        chart=form.chart,
        degree=form.degree - 1,
    )


def hodge_laplacian(
    form: DifferentialForm,
    metric: AbstractSemiRiemannianMetric,
    /,
    *,
    orientation: int = 1,
) -> DifferentialForm:
    """Return ``d δ form + δ d form`` for a nondegenerate metric."""
    if not isinstance(form, DifferentialForm):
        raise TypeError("hodge_laplacian requires a DifferentialForm.")
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("hodge_laplacian requires a nondegenerate metric.")
    _require_same_chart(form.chart, metric.chart)
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1.")
    if form.degree == 0:
        return codifferential(exterior_derivative(form), metric, orientation=orientation)
    first = exterior_derivative(codifferential(form, metric, orientation=orientation))
    if form.degree == form.chart.dimension:
        return first
    second = codifferential(exterior_derivative(form), metric, orientation=orientation)
    return _add_forms(first, second)


__all__ = [
    "DifferentialForm",
    "codifferential",
    "exterior_derivative",
    "hodge_laplacian",
    "hodge_star",
    "interior_product",
    "lie_derivative",
    "pullback_form",
    "wedge",
]
