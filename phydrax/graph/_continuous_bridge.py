#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import CoordinateChart, DifferentialForm, exterior_derivative
from ._cochain import CochainComplexIR, CochainFieldSpec
from ._cochain_ops import cochain_exterior_derivative


class OrientedCellParameterization(StrictModule):
    """Batched reference-cell maps and quadrature for one cochain degree."""

    degree: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    map_function: Callable[[Array, Array], Array]
    jacobian_function: Callable[[Array, Array], Array]
    reference_points: Array
    quadrature_weights: Array
    orientation_signs: Array

    def __init__(
        self,
        degree: int,
        cell_count: int,
        ambient_dimension: int,
        map_function: Callable[[Array, Array], Array],
        jacobian_function: Callable[[Array, Array], Array],
        reference_points: ArrayLike,
        quadrature_weights: ArrayLike,
        orientation_signs: ArrayLike,
        /,
    ):
        degree_value = int(degree)
        count = int(cell_count)
        dimension = int(ambient_dimension)
        if degree_value < 0 or count <= 0 or dimension <= 0:
            raise ValueError("Cell degree, count, and ambient dimension are invalid.")
        if degree_value > dimension:
            raise ValueError("Cell degree cannot exceed the ambient dimension.")
        if not callable(map_function) or not callable(jacobian_function):
            raise TypeError("Cell map and Jacobian functions must be callable.")
        points = jnp.asarray(reference_points)
        weights = jnp.asarray(quadrature_weights)
        signs = jnp.asarray(orientation_signs)
        if points.ndim != 2 or points.shape[1] != degree_value:
            raise ValueError(
                f"Reference points must have shape (quadrature, {degree_value})."
            )
        if weights.shape != (points.shape[0],):
            raise ValueError("Quadrature weights must match the reference-point count.")
        if signs.shape != (count,):
            raise ValueError(f"orientation_signs must have shape {(count,)}.")
        if bool(jnp.any(~jnp.isfinite(points))) or bool(jnp.any(~jnp.isfinite(weights))):
            raise ValueError("Reference quadrature must be finite.")
        if degree_value == 0 and (
            points.shape[0] != 1
            or not bool(jnp.array_equal(weights, jnp.ones((1,), dtype=weights.dtype)))
            or not bool(jnp.all(signs == 1.0))
        ):
            raise ValueError(
                "Zero-cell sampling requires one unit-weight point and invariant signs."
            )
        if bool(jnp.any(jnp.abs(signs) != 1.0)):
            raise ValueError("Cell orientation signs must be ±1.")
        self.degree = degree_value
        self.cell_count = count
        self.ambient_dimension = dimension
        self.map_function = map_function
        self.jacobian_function = jacobian_function
        self.reference_points = points
        self.quadrature_weights = weights
        self.orientation_signs = signs


class ContinuousCochainBridge(StrictModule):
    """Oriented cell parameterizations connecting smooth forms to one complex."""

    complex: CochainComplexIR
    chart: CoordinateChart
    parameterizations: tuple[OrientedCellParameterization, ...]

    def __init__(
        self,
        complex: CochainComplexIR,
        chart: CoordinateChart,
        parameterizations: Sequence[OrientedCellParameterization],
        /,
    ):
        if not isinstance(complex, CochainComplexIR):
            raise TypeError("complex must be a CochainComplexIR.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        parameters = tuple(parameterizations)
        if len(parameters) != complex.max_degree + 1:
            raise ValueError("One cell parameterization is required for every degree.")
        for degree, parameterization in enumerate(parameters):
            if not isinstance(parameterization, OrientedCellParameterization):
                raise TypeError(
                    "parameterizations must contain OrientedCellParameterization objects."
                )
            if (
                parameterization.degree != degree
                or parameterization.cell_count != complex.cell_counts[degree]
                or parameterization.ambient_dimension != chart.dimension
            ):
                raise ValueError(
                    "Cell parameterization degree, count, and dimension must match "
                    "the cochain complex and chart."
                )
        self.complex = complex
        self.chart = chart
        self.parameterizations = parameters


class ContinuousCochainProjection(StrictModule):
    """Full graph-node cochain values with explicit sampling semantics."""

    values: Array
    spec: CochainFieldSpec
    complex_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        spec: CochainFieldSpec,
        complex_fingerprint: str,
        /,
    ):
        self.values = jnp.asarray(values)
        self.spec = spec
        self.complex_fingerprint = str(complex_fingerprint)


class StokesValidationReport(StrictModule):
    valid: Array
    maximum_residual: Array
    relative_residual: Array

    def __init__(
        self,
        *,
        valid: Array,
        maximum_residual: Array,
        relative_residual: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.maximum_residual = jnp.asarray(maximum_residual)
        self.relative_residual = jnp.asarray(relative_residual)


def _cell_integral(
    form: DifferentialForm,
    parameterization: OrientedCellParameterization,
    cell: Array,
    /,
) -> Array:
    degree = form.degree
    target_indices = jnp.asarray(form.indices, dtype=jnp.int32)

    def integrand(reference: Array) -> Array:
        coordinates = jnp.asarray(parameterization.map_function(cell, reference))
        expected_coordinate_shape = (parameterization.ambient_dimension,)
        if coordinates.shape != expected_coordinate_shape:
            raise ValueError(
                f"Cell map must return shape {expected_coordinate_shape}; "
                f"got {coordinates.shape}."
            )
        coefficients = form._coefficients_point(coordinates)
        if degree == 0:
            return coefficients[0]
        jacobian = jnp.asarray(parameterization.jacobian_function(cell, reference))
        expected_jacobian_shape = (
            parameterization.ambient_dimension,
            degree,
        )
        if jacobian.shape != expected_jacobian_shape:
            raise ValueError(
                f"Cell Jacobian must return shape {expected_jacobian_shape}; "
                f"got {jacobian.shape}."
            )
        minors = jacobian[target_indices, :]
        return jnp.sum(coefficients * jnp.linalg.det(minors))

    values = jax.vmap(integrand)(parameterization.reference_points)
    integral = jnp.sum(parameterization.quadrature_weights * values)
    return parameterization.orientation_signs[cell] * integral


def integrate_form_to_cochain(
    form: DifferentialForm,
    bridge: ContinuousCochainBridge,
    /,
) -> ContinuousCochainProjection:
    """Integrate a smooth form over explicitly parameterized oriented cells."""
    if not isinstance(form, DifferentialForm):
        raise TypeError("form must be a DifferentialForm.")
    if not isinstance(bridge, ContinuousCochainBridge):
        raise TypeError("bridge must be a ContinuousCochainBridge.")
    if not form.chart.compatible_with(bridge.chart):
        raise ValueError("Form and cochain-bridge charts must match.")
    if form.degree > bridge.complex.max_degree:
        raise ValueError("Form degree exceeds the cochain-complex dimension.")
    parameterization = bridge.parameterizations[form.degree]
    local_values = jax.vmap(lambda cell: _cell_integral(form, parameterization, cell))(
        jnp.arange(parameterization.cell_count, dtype=jnp.int32)
    )
    values = jnp.zeros((bridge.complex.num_cells,), dtype=local_values.dtype)
    values = values.at[bridge.complex.cell_entities(form.degree)].set(local_values)
    spec = CochainFieldSpec(
        form.degree,
        complex_side="primal",
        cell_orientation="invariant" if form.degree == 0 else "signed",
        sampling="point_value" if form.degree == 0 else "cell_integral",
    )
    return ContinuousCochainProjection(
        values,
        spec,
        bridge.complex.fingerprint,
    )


def validate_stokes_bridge(
    form: DifferentialForm,
    bridge: ContinuousCochainBridge,
    /,
    *,
    tolerance: float = 1e-7,
) -> StokesValidationReport:
    """Check that smooth and discrete exterior derivatives commute for one form."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    if form.degree >= min(form.chart.dimension, bridge.complex.max_degree):
        raise ValueError("Stokes validation requires a representable higher degree.")
    source = integrate_form_to_cochain(form, bridge)
    smooth = integrate_form_to_cochain(exterior_derivative(form), bridge)
    discrete_values = cochain_exterior_derivative(
        bridge.complex.graph,
        source.values,
        form.degree,
    )
    target = bridge.complex.cell_entities(form.degree + 1)
    difference = discrete_values[target] - smooth.values[target]
    maximum = jnp.max(jnp.abs(difference), initial=0.0)
    scale = jnp.maximum(
        jnp.max(jnp.abs(smooth.values[target]), initial=0.0),
        jnp.asarray(1.0, dtype=maximum.dtype),
    )
    relative = maximum / scale
    return StokesValidationReport(
        valid=relative <= tolerance,
        maximum_residual=maximum,
        relative_residual=relative,
    )


__all__ = [
    "ContinuousCochainBridge",
    "ContinuousCochainProjection",
    "OrientedCellParameterization",
    "StokesValidationReport",
    "integrate_form_to_cochain",
    "validate_stokes_bridge",
]
