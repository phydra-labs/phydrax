#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import comb
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein
from phydrax.domain import AbstractGeometry, DomainFunction

from ..._strict import StrictModule
from ...metrix import (
    AbstractSemiRiemannianMetric,
    CoordinateChart,
    LorentzianMetric,
)
from ...metrix._exterior_basis import exterior_indices, wedge_sign
from ._domain_ops import _factor_and_dim, _resolve_var, grad


def _positions(deps: tuple[str, ...], function: DomainFunction, /) -> tuple[int, ...]:
    lookup = {label: position for position, label in enumerate(deps)}
    return tuple(lookup[label] for label in function.deps)


def _dependencies(
    domain_labels: tuple[str, ...],
    functions: tuple[DomainFunction, ...],
    var: str,
    /,
) -> tuple[str, ...]:
    return tuple(
        label
        for label in domain_labels
        if label == var or any(label in function.deps for function in functions)
    )


def _evaluate(
    function: DomainFunction,
    positions: tuple[int, ...],
    args: tuple[Any, ...],
    degree: int,
    coefficient_count: int,
    /,
    *,
    key: Any,
    kwargs: dict[str, Any],
) -> Array:
    values = jnp.asarray(
        function.func(
            *[args[position] for position in positions],
            key=key,
            **kwargs,
        )
    )
    if degree == 0 and values.shape[-1:] != (1,):
        values = values[..., None]
    if values.shape[-1:] != (coefficient_count,):
        raise ValueError(
            f"Degree-{degree} form coefficients require trailing size "
            f"{coefficient_count}; got {values.shape}."
        )
    return values


class DomainDifferentialForm(StrictModule):
    """Differential-form coefficients carried by a labeled DomainFunction."""

    coefficients: DomainFunction
    chart: CoordinateChart
    var: str
    degree: int
    indices: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        coefficients: DomainFunction,
        /,
        *,
        chart: CoordinateChart,
        degree: int,
        var: str | None = None,
    ):
        if not isinstance(coefficients, DomainFunction):
            raise TypeError("coefficients must be a DomainFunction.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        degree_value = int(degree)
        if degree_value < 0 or degree_value > chart.dimension:
            raise ValueError(
                f"Form degree must lie in [0, {chart.dimension}]; got {degree_value}."
            )
        variable = _resolve_var(coefficients, var)
        _, dimension = _factor_and_dim(coefficients, variable)
        if not isinstance(coefficients.domain.factor(variable), AbstractGeometry):
            raise ValueError("Domain differential forms require a geometry variable.")
        if dimension != chart.dimension:
            raise ValueError(
                f"Chart dimension {chart.dimension} does not match domain variable "
                f"{variable!r} dimension {dimension}."
            )
        self.coefficients = coefficients
        self.chart = chart
        self.var = variable
        self.degree = degree_value
        self.indices = exterior_indices(chart.dimension, degree_value)

    @property
    def coefficient_count(self) -> int:
        return comb(self.chart.dimension, self.degree)


class DomainMaxwellResiduals(StrictModule):
    """Covariant Maxwell field strength and its two form-valued residuals."""

    field_strength: DomainDifferentialForm
    homogeneous: DomainDifferentialForm
    inhomogeneous: DomainDifferentialForm

    def __init__(
        self,
        field_strength: DomainDifferentialForm,
        homogeneous: DomainDifferentialForm,
        inhomogeneous: DomainDifferentialForm,
        /,
    ):
        self.field_strength = field_strength
        self.homogeneous = homogeneous
        self.inhomogeneous = inhomogeneous


class _DomainWedgeCallable(StrictModule):
    left: DomainDifferentialForm
    right: DomainDifferentialForm
    left_positions: tuple[int, ...]
    right_positions: tuple[int, ...]
    left_terms: Array
    right_terms: Array
    output_terms: Array
    signs: Array
    output_count: int

    def __init__(
        self,
        left: DomainDifferentialForm,
        right: DomainDifferentialForm,
        deps: tuple[str, ...],
        /,
    ):
        output = exterior_indices(left.chart.dimension, left.degree + right.degree)
        lookup = {index: position for position, index in enumerate(output)}
        left_terms: list[int] = []
        right_terms: list[int] = []
        output_terms: list[int] = []
        signs: list[int] = []
        for left_position, left_index in enumerate(left.indices):
            left_set = set(left_index)
            for right_position, right_index in enumerate(right.indices):
                if left_set.isdisjoint(right_index):
                    left_terms.append(left_position)
                    right_terms.append(right_position)
                    output_terms.append(lookup[tuple(sorted(left_index + right_index))])
                    signs.append(wedge_sign(left_index, right_index))
        self.left = left
        self.right = right
        self.left_positions = _positions(deps, left.coefficients)
        self.right_positions = _positions(deps, right.coefficients)
        self.left_terms = jnp.asarray(left_terms, dtype=jnp.int32)
        self.right_terms = jnp.asarray(right_terms, dtype=jnp.int32)
        self.output_terms = jnp.asarray(output_terms, dtype=jnp.int32)
        self.signs = jnp.asarray(signs)
        self.output_count = len(output)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        args_tuple = tuple(args)
        left = _evaluate(
            self.left.coefficients,
            self.left_positions,
            args_tuple,
            self.left.degree,
            self.left.coefficient_count,
            key=key,
            kwargs=kwargs,
        )
        right = _evaluate(
            self.right.coefficients,
            self.right_positions,
            args_tuple,
            self.right.degree,
            self.right.coefficient_count,
            key=key,
            kwargs=kwargs,
        )
        terms = self.signs * left[..., self.left_terms] * right[..., self.right_terms]
        return (
            jnp.zeros(terms.shape[:-1] + (self.output_count,), dtype=terms.dtype)
            .at[..., self.output_terms]
            .add(terms)
        )


class _DomainExteriorCallable(StrictModule):
    form: DomainDifferentialForm
    derivative: DomainFunction
    derivative_positions: tuple[int, ...]
    source_terms: Array
    derivative_axes: Array
    output_terms: Array
    signs: Array
    output_count: int

    def __init__(
        self,
        form: DomainDifferentialForm,
        derivative: DomainFunction,
        deps: tuple[str, ...],
        /,
    ):
        output = exterior_indices(form.chart.dimension, form.degree + 1)
        lookup = {index: position for position, index in enumerate(form.indices)}
        source_terms: list[int] = []
        derivative_axes: list[int] = []
        output_terms: list[int] = []
        signs: list[int] = []
        for output_position, output_index in enumerate(output):
            for position, axis in enumerate(output_index):
                source_terms.append(
                    lookup[output_index[:position] + output_index[position + 1 :]]
                )
                derivative_axes.append(axis)
                output_terms.append(output_position)
                signs.append(-1 if position % 2 else 1)
        self.form = form
        self.derivative = derivative
        self.derivative_positions = _positions(deps, derivative)
        self.source_terms = jnp.asarray(source_terms, dtype=jnp.int32)
        self.derivative_axes = jnp.asarray(derivative_axes, dtype=jnp.int32)
        self.output_terms = jnp.asarray(output_terms, dtype=jnp.int32)
        self.signs = jnp.asarray(signs)
        self.output_count = len(output)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        derivative = jnp.asarray(
            self.derivative.func(
                *[args[position] for position in self.derivative_positions],
                key=key,
                **kwargs,
            )
        )
        if self.form.degree == 0 and derivative.shape[-1:] == (
            self.form.chart.dimension,
        ):
            derivative = derivative[..., None, :]
        expected = (
            self.form.coefficient_count,
            self.form.chart.dimension,
        )
        if derivative.shape[-2:] != expected:
            raise ValueError(
                f"Form coefficient derivative requires trailing shape {expected}; "
                f"got {derivative.shape}."
            )
        terms = self.signs * derivative[..., self.source_terms, self.derivative_axes]
        return (
            jnp.zeros(terms.shape[:-1] + (self.output_count,), dtype=terms.dtype)
            .at[..., self.output_terms]
            .add(terms)
        )


class _DomainInteriorCallable(StrictModule):
    vector: DomainFunction
    form: DomainDifferentialForm
    vector_positions: tuple[int, ...]
    form_positions: tuple[int, ...]
    vector_terms: Array
    source_terms: Array
    output_terms: Array
    signs: Array
    output_count: int

    def __init__(
        self,
        vector: DomainFunction,
        form: DomainDifferentialForm,
        deps: tuple[str, ...],
        /,
    ):
        output = exterior_indices(form.chart.dimension, form.degree - 1)
        lookup = {index: position for position, index in enumerate(form.indices)}
        vector_terms: list[int] = []
        source_terms: list[int] = []
        output_terms: list[int] = []
        signs: list[int] = []
        for output_position, output_index in enumerate(output):
            output_set = set(output_index)
            for axis in range(form.chart.dimension):
                if axis in output_set:
                    continue
                source_index = tuple(sorted((axis,) + output_index))
                vector_terms.append(axis)
                source_terms.append(lookup[source_index])
                output_terms.append(output_position)
                signs.append(-1 if source_index.index(axis) % 2 else 1)
        self.vector = vector
        self.form = form
        self.vector_positions = _positions(deps, vector)
        self.form_positions = _positions(deps, form.coefficients)
        self.vector_terms = jnp.asarray(vector_terms, dtype=jnp.int32)
        self.source_terms = jnp.asarray(source_terms, dtype=jnp.int32)
        self.output_terms = jnp.asarray(output_terms, dtype=jnp.int32)
        self.signs = jnp.asarray(signs)
        self.output_count = len(output)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        vector = jnp.asarray(
            self.vector.func(
                *[args[position] for position in self.vector_positions],
                key=key,
                **kwargs,
            )
        )
        if vector.shape[-1:] != (self.form.chart.dimension,):
            raise ValueError("Interior-product field has the wrong trailing dimension.")
        coefficients = _evaluate(
            self.form.coefficients,
            self.form_positions,
            tuple(args),
            self.form.degree,
            self.form.coefficient_count,
            key=key,
            kwargs=kwargs,
        )
        terms = (
            self.signs
            * vector[..., self.vector_terms]
            * coefficients[..., self.source_terms]
        )
        return (
            jnp.zeros(terms.shape[:-1] + (self.output_count,), dtype=terms.dtype)
            .at[..., self.output_terms]
            .add(terms)
        )


class _DomainHodgeCallable(StrictModule):
    form: DomainDifferentialForm
    metric: AbstractSemiRiemannianMetric
    form_positions: tuple[int, ...]
    coordinate_position: int
    source_indices: Array
    output_terms: Array
    signs: Array
    output_count: int
    orientation: int

    def __init__(
        self,
        form: DomainDifferentialForm,
        metric: AbstractSemiRiemannianMetric,
        deps: tuple[str, ...],
        orientation: int,
        /,
    ):
        output = exterior_indices(
            form.chart.dimension, form.chart.dimension - form.degree
        )
        output_lookup = {index: position for position, index in enumerate(output)}
        full = set(range(form.chart.dimension))
        output_terms: list[int] = []
        signs: list[int] = []
        for source in form.indices:
            complement = tuple(sorted(full.difference(source)))
            output_terms.append(output_lookup[complement])
            signs.append(wedge_sign(source, complement))
        self.form = form
        self.metric = metric
        self.form_positions = _positions(deps, form.coefficients)
        self.coordinate_position = deps.index(form.var)
        self.source_indices = jnp.asarray(form.indices, dtype=jnp.int32)
        self.output_terms = jnp.asarray(output_terms, dtype=jnp.int32)
        self.signs = jnp.asarray(signs)
        self.output_count = len(output)
        self.orientation = int(orientation)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        coefficients = _evaluate(
            self.form.coefficients,
            self.form_positions,
            tuple(args),
            self.form.degree,
            self.form.coefficient_count,
            key=key,
            kwargs=kwargs,
        )
        coordinates = args[self.coordinate_position]
        if self.form.degree == 0:
            paired = coefficients
        else:
            inverse = self.metric.inverse(coordinates)
            rows = self.source_indices[:, None, :, None]
            columns = self.source_indices[None, :, None, :]
            induced_inverse = jnp.linalg.det(inverse[..., rows, columns])
            paired = ein.contract("...ij,...j->...i", induced_inverse, coefficients)
        values = (
            self.orientation
            * self.metric.volume_density(coordinates)[..., None]
            * self.signs
            * paired
        )
        return (
            jnp.zeros(
                values.shape[:-1] + (self.output_count,),
                dtype=values.dtype,
            )
            .at[..., self.output_terms]
            .set(values)
        )


class _ZeroFormCallable(StrictModule):
    def __call__(self, coordinates: Array, /, *, key=None, **kwargs: Any) -> Array:
        del key, kwargs
        return jnp.zeros(coordinates.shape[:-1] + (1,), dtype=coordinates.dtype)


class _ScaleCallable(StrictModule):
    function: DomainFunction
    scale: float

    def __init__(self, function: DomainFunction, scale: float, /):
        self.function = function
        self.scale = float(scale)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        return self.scale * self.function.func(*args, key=key, **kwargs)


def domain_differential_form(
    coefficients: DomainFunction,
    /,
    *,
    chart: CoordinateChart,
    degree: int,
    var: str | None = None,
) -> DomainDifferentialForm:
    return DomainDifferentialForm(coefficients, chart=chart, degree=degree, var=var)


def domain_wedge(
    left: DomainDifferentialForm,
    right: DomainDifferentialForm,
    /,
) -> DomainDifferentialForm:
    if not isinstance(left, DomainDifferentialForm) or not isinstance(
        right, DomainDifferentialForm
    ):
        raise TypeError("domain_wedge requires two DomainDifferentialForm instances.")
    if left.coefficients.domain is not right.coefficients.domain:
        raise ValueError("domain_wedge currently requires one shared Domain instance.")
    if left.var != right.var or not left.chart.compatible_with(right.chart):
        raise ValueError("Domain forms must use the same variable and chart.")
    degree = left.degree + right.degree
    if degree > left.chart.dimension:
        raise ValueError("Wedge-product degree exceeds the chart dimension.")
    deps = _dependencies(
        left.coefficients.domain.labels,
        (left.coefficients, right.coefficients),
        left.var,
    )
    coefficients = DomainFunction(
        domain=left.coefficients.domain,
        deps=deps,
        func=_DomainWedgeCallable(left, right, deps),
        metadata=left.coefficients.metadata,
    )
    return DomainDifferentialForm(
        coefficients,
        chart=left.chart,
        degree=degree,
        var=left.var,
    )


def domain_exterior_derivative(
    form: DomainDifferentialForm,
    /,
    *,
    mode: Literal["reverse", "forward"] = "forward",
) -> DomainDifferentialForm:
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("domain_exterior_derivative requires a DomainDifferentialForm.")
    if form.degree == form.chart.dimension:
        raise ValueError("The exterior derivative of a top form is identically zero.")
    derivative = grad(form.coefficients, var=form.var, mode=mode)
    deps = _dependencies(
        form.coefficients.domain.labels,
        (derivative,),
        form.var,
    )
    coefficients = DomainFunction(
        domain=form.coefficients.domain,
        deps=deps,
        func=_DomainExteriorCallable(form, derivative, deps),
        metadata=derivative.metadata,
    )
    return DomainDifferentialForm(
        coefficients,
        chart=form.chart,
        degree=form.degree + 1,
        var=form.var,
    )


def domain_interior_product(
    vector: DomainFunction,
    form: DomainDifferentialForm,
    /,
) -> DomainDifferentialForm:
    if not isinstance(vector, DomainFunction):
        raise TypeError("vector must be a DomainFunction.")
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("form must be a DomainDifferentialForm.")
    if vector.domain is not form.coefficients.domain:
        raise ValueError("Domain interior product requires one shared Domain instance.")
    if form.degree == 0:
        raise ValueError("Interior product of a zero-form is identically zero.")
    deps = _dependencies(
        vector.domain.labels,
        (vector, form.coefficients),
        form.var,
    )
    coefficients = DomainFunction(
        domain=vector.domain,
        deps=deps,
        func=_DomainInteriorCallable(vector, form, deps),
        metadata=form.coefficients.metadata,
    )
    return DomainDifferentialForm(
        coefficients,
        chart=form.chart,
        degree=form.degree - 1,
        var=form.var,
    )


def _add_domain_forms(
    left: DomainDifferentialForm,
    right: DomainDifferentialForm,
    /,
) -> DomainDifferentialForm:
    if left.degree != right.degree:
        raise ValueError("Only equal-degree domain forms can be added.")
    if (
        left.coefficients.domain is not right.coefficients.domain
        or left.var != right.var
        or not left.chart.compatible_with(right.chart)
    ):
        raise ValueError("Domain forms must share one domain, variable, and chart.")
    return DomainDifferentialForm(
        left.coefficients + right.coefficients,
        chart=left.chart,
        degree=left.degree,
        var=left.var,
    )


def _scale_domain_form(
    form: DomainDifferentialForm,
    scale: float,
    /,
) -> DomainDifferentialForm:
    coefficients = DomainFunction(
        domain=form.coefficients.domain,
        deps=form.coefficients.deps,
        func=_ScaleCallable(form.coefficients, scale),
        metadata=form.coefficients.metadata,
    )
    return DomainDifferentialForm(
        coefficients,
        chart=form.chart,
        degree=form.degree,
        var=form.var,
    )


def domain_lie_derivative(
    vector: DomainFunction,
    form: DomainDifferentialForm,
    /,
) -> DomainDifferentialForm:
    if not isinstance(vector, DomainFunction):
        raise TypeError("vector must be a DomainFunction.")
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("form must be a DomainDifferentialForm.")
    if form.degree == 0:
        return domain_interior_product(vector, domain_exterior_derivative(form))
    first = domain_exterior_derivative(domain_interior_product(vector, form))
    if form.degree == form.chart.dimension:
        return first
    second = domain_interior_product(vector, domain_exterior_derivative(form))
    return _add_domain_forms(first, second)


def domain_hodge_star(
    form: DomainDifferentialForm,
    metric: AbstractSemiRiemannianMetric,
    /,
    *,
    orientation: int = 1,
) -> DomainDifferentialForm:
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("domain_hodge_star requires a DomainDifferentialForm.")
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("domain_hodge_star requires a nondegenerate metric.")
    if not form.chart.compatible_with(metric.chart):
        raise ValueError("Domain form and metric charts must match.")
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1.")
    deps = _dependencies(
        form.coefficients.domain.labels,
        (form.coefficients,),
        form.var,
    )
    coefficients = DomainFunction(
        domain=form.coefficients.domain,
        deps=deps,
        func=_DomainHodgeCallable(form, metric, deps, orientation),
        metadata=form.coefficients.metadata,
    )
    return DomainDifferentialForm(
        coefficients,
        chart=form.chart,
        degree=form.chart.dimension - form.degree,
        var=form.var,
    )


def domain_codifferential(
    form: DomainDifferentialForm,
    metric: AbstractSemiRiemannianMetric,
    /,
    *,
    orientation: int = 1,
) -> DomainDifferentialForm:
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("domain_codifferential requires a DomainDifferentialForm.")
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("domain_codifferential requires a nondegenerate metric.")
    if not form.chart.compatible_with(metric.chart):
        raise ValueError("Domain form and metric charts must match.")
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1.")
    if form.degree == 0:
        coefficients = DomainFunction(
            domain=form.coefficients.domain,
            deps=(form.var,),
            func=_ZeroFormCallable(),
            metadata=form.coefficients.metadata,
        )
        return DomainDifferentialForm(
            coefficients, chart=form.chart, degree=0, var=form.var
        )
    first = domain_hodge_star(form, metric, orientation=orientation)
    derivative = domain_exterior_derivative(first)
    result = domain_hodge_star(derivative, metric, orientation=orientation)
    exponent = form.chart.dimension * (form.degree + 1) + metric.signature.index + 1
    sign = -1 if exponent % 2 else 1
    coefficients = DomainFunction(
        domain=result.coefficients.domain,
        deps=result.coefficients.deps,
        func=_ScaleCallable(result.coefficients, sign),
        metadata=result.coefficients.metadata,
    )
    return DomainDifferentialForm(
        coefficients,
        chart=form.chart,
        degree=form.degree - 1,
        var=form.var,
    )


def domain_hodge_laplacian(
    form: DomainDifferentialForm,
    metric: AbstractSemiRiemannianMetric,
    /,
    *,
    orientation: int = 1,
) -> DomainDifferentialForm:
    if not isinstance(form, DomainDifferentialForm):
        raise TypeError("domain_hodge_laplacian requires a DomainDifferentialForm.")
    if not isinstance(metric, AbstractSemiRiemannianMetric):
        raise TypeError("domain_hodge_laplacian requires a nondegenerate metric.")
    if not form.chart.compatible_with(metric.chart):
        raise ValueError("Domain form and metric charts must match.")
    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1.")
    if form.degree == 0:
        return domain_codifferential(
            domain_exterior_derivative(form),
            metric,
            orientation=orientation,
        )
    first = domain_exterior_derivative(
        domain_codifferential(form, metric, orientation=orientation)
    )
    if form.degree == form.chart.dimension:
        return first
    second = domain_codifferential(
        domain_exterior_derivative(form), metric, orientation=orientation
    )
    return _add_domain_forms(first, second)


def domain_maxwell_residuals(
    field_strength: DomainDifferentialForm,
    metric: LorentzianMetric,
    /,
    *,
    electric_current: DomainDifferentialForm | None = None,
    magnetic_current: DomainDifferentialForm | None = None,
    orientation: int = 1,
) -> DomainMaxwellResiduals:
    r"""Compose ``dF - M`` and ``delta F + J_flat`` on four-dimensional spacetime."""
    if not isinstance(field_strength, DomainDifferentialForm):
        raise TypeError("domain_maxwell_residuals requires a DomainDifferentialForm.")
    if field_strength.chart.dimension != 4 or field_strength.degree != 2:
        raise ValueError(
            "Maxwell field strength must be a degree-2 form on a four-dimensional chart."
        )
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("domain_maxwell_residuals requires a LorentzianMetric.")
    if not field_strength.chart.compatible_with(metric.chart):
        raise ValueError("Maxwell field strength and metric charts must match.")
    if electric_current is not None:
        if not isinstance(electric_current, DomainDifferentialForm):
            raise TypeError("electric_current must be a DomainDifferentialForm.")
        if electric_current.degree != 1:
            raise ValueError("electric_current must be a degree-1 covector form.")
    if magnetic_current is not None:
        if not isinstance(magnetic_current, DomainDifferentialForm):
            raise TypeError("magnetic_current must be a DomainDifferentialForm.")
        if magnetic_current.degree != 3:
            raise ValueError("magnetic_current must be a degree-3 form.")

    homogeneous = domain_exterior_derivative(field_strength)
    if magnetic_current is not None:
        homogeneous = _add_domain_forms(
            homogeneous,
            _scale_domain_form(magnetic_current, -1.0),
        )
    inhomogeneous = domain_codifferential(
        field_strength,
        metric,
        orientation=orientation,
    )
    if electric_current is not None:
        inhomogeneous = _add_domain_forms(inhomogeneous, electric_current)
    return DomainMaxwellResiduals(
        field_strength,
        homogeneous,
        inhomogeneous,
    )


__all__ = [
    "DomainMaxwellResiduals",
    "DomainDifferentialForm",
    "domain_codifferential",
    "domain_differential_form",
    "domain_exterior_derivative",
    "domain_hodge_laplacian",
    "domain_maxwell_residuals",
    "domain_hodge_star",
    "domain_interior_product",
    "domain_lie_derivative",
    "domain_wedge",
]
