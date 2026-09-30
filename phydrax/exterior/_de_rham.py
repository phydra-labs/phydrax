#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._polynomial._cubature import simplex_rule_data, tensor_product_rule_data
from .._strict import StrictModule
from ..discretization._cochain_hodge import DiagonalHodge
from ..discretization._structured_cochain import StructuredCochainBridge
from ..discretization._topology import CellComplexTopology
from ..linalg import compound_matrix
from ..metrix import (
    CoordinateChart,
    DifferentialForm,
    exterior_derivative,
    RiemannianMetric,
)
from ..operators.differential import domain_exterior_derivative, DomainDifferentialForm
from ..typing import Dim, Float64
from ._algebra import pullback
from ._complex import AbstractDeRhamComplex, DiscreteForm


class CellDim(Dim):
    pass


class AmbientDim(Dim):
    pass


class ReferenceDim(Dim):
    pass


class QuadratureDim(Dim):
    pass


def _validate_cell_dimensions(
    degree: int, cell_count: int, ambient_dimension: int, /
) -> None:
    for name, value in (
        ("degree", degree),
        ("cell_count", cell_count),
        ("ambient_dimension", ambient_dimension),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")
    if (
        degree < 0
        or cell_count < 0
        or ambient_dimension <= 0
        or degree > ambient_dimension
    ):
        raise ValueError("Cell degree, count, and ambient dimension are invalid.")


def _validate_reference_quadrature(points: Array, weights: Array, degree: int, /) -> None:
    if points.ndim != 2 or points.shape[1] != degree or points.shape[0] == 0:
        raise ValueError("Reference points must have shape (quadrature, degree).")
    if weights.shape != (points.shape[0],):
        raise ValueError("Quadrature weights must match the reference-point axis.")
    if bool(jnp.any(~jnp.isfinite(points))) or bool(jnp.any(~jnp.isfinite(weights))):
        raise ValueError("Reference quadrature must be finite.")
    if degree == 0 and (points.shape[0] != 1 or not bool(jnp.all(weights == 1))):
        raise ValueError(
            "Zero-cell sampling requires one unit-weight point and invariant signs."
        )


def _validate_cell_orientations(
    signs: Array, coorientations: Array | None, degree: int, cell_count: int, /
) -> None:
    if signs.shape != (cell_count,) or bool(jnp.any(jnp.abs(signs) != 1)):
        raise ValueError("Cell orientation signs must contain one ±1 value per cell.")
    if coorientations is not None and (
        coorientations.shape != (cell_count,)
        or bool(jnp.any(jnp.abs(coorientations) != 1))
    ):
        raise ValueError("Coorientation signs must contain one ±1 value per cell.")
    if degree == 0 and not bool(jnp.all(signs == 1)):
        raise ValueError(
            "Zero-cell sampling requires one unit-weight point and invariant signs."
        )


@final
class CellParameterization(StrictModule):
    """Reference maps for one cell degree, with explicit ambient coorientation.

    The map Jacobian includes its own signed orientation. ``orientation_signs``
    changes the orientation relative to that map. Twisted forms on embedded
    cells require ``coorientation_signs``; these orient the ambient normal bundle,
    not the cell tangent bundle.
    """

    __strict_contract__ = True
    degree: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    metric_compatible_quadrature: bool = eqx.field(static=True)
    map_function: Callable[[Array, Array], Array]
    jacobian_function: Callable[[Array, Array], Array]
    reference_points: Float64[QuadratureDim, ReferenceDim]
    quadrature_weights: Float64[QuadratureDim]
    orientation_signs: Float64[CellDim]
    coorientation_signs: Float64[CellDim] | None

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
        *,
        coorientation_signs: ArrayLike | None = None,
    ) -> None:
        _validate_cell_dimensions(degree, cell_count, ambient_dimension)
        if not callable(map_function) or not callable(jacobian_function):
            raise TypeError("Cell map and Jacobian functions must be callable.")
        points = jnp.asarray(reference_points, dtype=jnp.float64)
        weights = jnp.asarray(quadrature_weights, dtype=jnp.float64)
        signs = jnp.asarray(orientation_signs, dtype=jnp.float64)
        coorientations = (
            None
            if coorientation_signs is None
            else jnp.asarray(coorientation_signs, dtype=jnp.float64)
        )
        _validate_reference_quadrature(points, weights, degree)
        _validate_cell_orientations(signs, coorientations, degree, cell_count)
        self.degree = degree
        self.cell_count = cell_count
        self.ambient_dimension = ambient_dimension
        self.metric_compatible_quadrature = bool(jnp.all(weights >= 0))
        self.map_function = map_function
        self.jacobian_function = jacobian_function
        self.reference_points = points
        self.quadrature_weights = weights
        self.orientation_signs = signs
        self.coorientation_signs = coorientations


@final
class DeRhamBridge(StrictModule):
    """The de Rham integration map into a canonical cell realization."""

    complex: AbstractDeRhamComplex
    chart: CoordinateChart
    parameterizations: tuple[CellParameterization, ...]

    def __init__(
        self,
        complex: AbstractDeRhamComplex,
        chart: CoordinateChart,
        parameterizations: Sequence[CellParameterization],
        /,
    ) -> None:
        if not isinstance(complex, AbstractDeRhamComplex):
            raise TypeError("complex must be an AbstractDeRhamComplex.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        parameters = tuple(parameterizations)
        if len(parameters) != complex.dimension + 1:
            raise ValueError("One cell parameterization is required for every degree.")
        for degree, parameterization in enumerate(parameters):
            if not isinstance(parameterization, CellParameterization):
                raise TypeError(
                    "parameterizations must contain CellParameterization objects."
                )
            if (
                parameterization.degree != degree
                or parameterization.cell_count != complex.cell_counts[degree]
                or parameterization.ambient_dimension != chart.dimension
            ):
                raise ValueError(
                    "Cell parameterization degree, count, and dimension must match the complex and chart."
                )
        self.complex = complex
        self.chart = chart
        self.parameterizations = parameters


@final
class DeRhamCommutationEvidence(StrictModule):
    """Measured defect of R(dω) = d(Rω), including its canonical realization."""

    realization_id: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    valid: Array
    maximum_residual: Array
    relative_residual: Array

    def __init__(
        self,
        realization_id: str,
        degree: int,
        /,
        *,
        valid: Array,
        maximum_residual: Array,
        relative_residual: Array,
    ) -> None:
        self.realization_id = realization_id
        self.degree = degree
        self.valid = valid
        self.maximum_residual = maximum_residual
        self.relative_residual = relative_residual


@final
class _AffineCellMap(StrictModule):
    __strict_contract__ = True
    origins: Float64[CellDim, AmbientDim]
    jacobians: Float64[CellDim, AmbientDim, ReferenceDim]

    def __init__(self, origins: Array, jacobians: Array, /) -> None:
        if (
            origins.ndim != 2
            or jacobians.ndim != 3
            or origins.shape != jacobians.shape[:2]
        ):
            raise ValueError(
                "Affine cell origins and Jacobians must share cell and ambient axes."
            )
        self.origins = origins
        self.jacobians = jacobians

    def __call__(self, cell: Array, reference: Array, /) -> Array:
        return self.origins[cell] + self.jacobians[cell] @ reference


@final
class _AffineCellJacobian(StrictModule):
    mapping: _AffineCellMap

    def __init__(self, mapping: _AffineCellMap, /) -> None:
        self.mapping = mapping

    def __call__(self, cell: Array, reference: Array, /) -> Array:
        return self.mapping.jacobians[cell]


def _reference_quadrature(
    degree: int, order: int, /, *, simplex: bool
) -> tuple[Array, Array]:
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
        raise ValueError("order must be a nonnegative polynomial exactness degree.")
    if degree == 0:
        return jnp.zeros((1, 0), dtype=jnp.float64), jnp.ones((1,), dtype=jnp.float64)
    if simplex:
        rule = simplex_rule_data(degree, order)
    else:
        count = max(1, (order + 2) // 2)
        rule = tensor_product_rule_data(degree, count, family="gauss")
    return rule.points, rule.weights


def simplicial_parameterizations(
    topology: CellComplexTopology, vertices: ArrayLike, /, *, order: int
) -> tuple[CellParameterization, ...]:
    """Parameterize affine simplices in topology order with incidence orientation."""
    from ..discretization._cell_complex import simplicial_cell_geometry

    if not isinstance(topology, CellComplexTopology):
        raise TypeError("topology must be a CellComplexTopology.")
    coordinates = jnp.asarray(vertices, dtype=jnp.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != topology.entity_sets[0].count
        or coordinates.shape[1] < topology.dimension
    ):
        raise ValueError(
            "vertices must have one ambient coordinate vector per topology vertex."
        )
    rows, orientations = simplicial_cell_geometry(topology)
    parameters: list[CellParameterization] = []
    for degree, (cells, signs) in enumerate(zip(rows, orientations, strict=True)):
        corners = coordinates[jnp.asarray(cells, dtype=jnp.int32)]
        mapping = _AffineCellMap(
            corners[:, 0], jnp.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
        )
        points, weights = _reference_quadrature(degree, order, simplex=True)
        parameters.append(
            CellParameterization(
                degree,
                cells.shape[0],
                coordinates.shape[1],
                mapping,
                _AffineCellJacobian(mapping),
                points,
                weights,
                signs,
            )
        )
    return tuple(parameters)


def structured_parameterizations(
    bridge: StructuredCochainBridge, /, *, order: int
) -> tuple[CellParameterization, ...]:
    """Parameterize tensor cells, preserving component-major/C-order packing."""
    if not isinstance(bridge, StructuredCochainBridge):
        raise TypeError("bridge must be a StructuredCochainBridge.")
    parameters: list[CellParameterization] = []
    for degree, orientations in enumerate(bridge.orientations):
        origins: list[Array] = []
        jacobians: list[Array] = []
        for orientation, shape in zip(
            orientations, bridge.orientation_shapes[degree], strict=True
        ):
            indices = np.indices(shape, dtype=np.int32).reshape((bridge.dimension, -1)).T
            origin = jnp.stack(
                tuple(
                    axis.point_coordinates[jnp.asarray(indices[:, index])]
                    for index, axis in enumerate(bridge.grid.structured_axes)
                ),
                axis=-1,
            )
            jacobian = jnp.zeros(
                (indices.shape[0], bridge.dimension, degree), dtype=jnp.float64
            )
            for column, axis_index in enumerate(orientation):
                widths = bridge.grid.structured_axes[axis_index].interval_widths[
                    jnp.asarray(indices[:, axis_index])
                ]
                jacobian = jacobian.at[:, axis_index, column].set(widths)
            origins.append(origin)
            jacobians.append(jacobian)
        mapping = _AffineCellMap(jnp.concatenate(origins), jnp.concatenate(jacobians))
        points, weights = _reference_quadrature(degree, order, simplex=False)
        parameters.append(
            CellParameterization(
                degree,
                bridge.cochain.cell_counts[degree],
                bridge.dimension,
                mapping,
                _AffineCellJacobian(mapping),
                points,
                weights,
                jnp.ones((bridge.cochain.cell_counts[degree],), dtype=jnp.float64),
            )
        )
    return tuple(parameters)


def _form_evaluator(
    form: DifferentialForm | DomainDifferentialForm, /
) -> Callable[[Array], Array]:
    match form:
        case DifferentialForm():
            return form._coefficients_point
        case DomainDifferentialForm():
            if form.coefficients.deps != (form.var,):
                raise ValueError(
                    "Integration requires domain coefficients bound to their single geometry variable."
                )

            def evaluate(point: Array) -> Array:
                return jnp.asarray(form.coefficients.func(point, key=None))

            return evaluate
        case _:
            raise TypeError("form must be a DifferentialForm or DomainDifferentialForm.")


def integrate_form(
    form: DifferentialForm | DomainDifferentialForm, bridge: DeRhamBridge, /
) -> DiscreteForm:
    """Integrate scalar-fiber smooth coefficients into degree-local primal values.

    Twisted embedded forms require explicit cell coorientations. This map does
    not reinterpret twisted primal cell integrals as complementary dual cells.
    """
    evaluate = _form_evaluator(form)
    if not isinstance(bridge, DeRhamBridge):
        raise TypeError("bridge must be a DeRhamBridge.")
    if not form.chart.compatible_with(bridge.chart):
        raise ValueError("Form and bridge charts must match.")
    if form.degree > bridge.complex.dimension:
        raise ValueError("Form degree exceeds the complex dimension.")
    if form.form_type.fiber_shape:
        raise ValueError("DiscreteForm integration requires scalar-fiber coefficients.")
    if form.form_type.twist != bridge.complex.primal_twist:
        raise ValueError("Form twist must match the realization's primal twist.")
    parameters = bridge.parameterizations[form.degree]
    if (
        form.form_type.twist == "twisted"
        and form.degree != bridge.chart.dimension
        and parameters.coorientation_signs is None
    ):
        raise ValueError(
            "Twisted embedded integration requires explicit coorientation signs."
        )

    def cell_integral(cell: Array) -> Array:
        def integrand(reference: Array) -> Array:
            point = parameters.map_function(cell, reference)
            jacobian = parameters.jacobian_function(cell, reference)
            if point.shape != (parameters.ambient_dimension,) or jacobian.shape != (
                parameters.ambient_dimension,
                parameters.degree,
            ):
                raise ValueError(
                    "Cell maps and Jacobians must match declared ambient and reference dimensions."
                )
            # The algebra owner validates the static normal-bundle choice. The
            # per-cell coorientation stays a device leaf under vmap/JIT.
            embedded_twist = (
                form.form_type.twist == "twisted"
                and parameters.degree != parameters.ambient_dimension
            )
            value = pullback(
                evaluate(point),
                form.form_type,
                jacobian,
                coorientation=1 if embedded_twist else None,
            )[0]
            if embedded_twist and parameters.coorientation_signs is not None:
                value = value * parameters.coorientation_signs[cell]
            return value

        values = jax.vmap(integrand)(parameters.reference_points)
        integral = parameters.quadrature_weights @ values
        return (
            integral
            if form.form_type.twist == "twisted"
            else parameters.orientation_signs[cell] * integral
        )

    values = jax.vmap(cell_integral)(jnp.arange(parameters.cell_count, dtype=jnp.int32))
    return DiscreteForm(
        bridge.complex.realization_id, bridge.complex.form_type(form.degree), values
    )


def validate_de_rham_commutation(
    form: DifferentialForm | DomainDifferentialForm,
    bridge: DeRhamBridge,
    /,
    *,
    tolerance: float,
) -> DeRhamCommutationEvidence:
    """Measure the actual smooth/discrete d-commutation residual."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    source = integrate_form(form, bridge)
    if form.degree >= bridge.complex.dimension:
        raise ValueError("Commutation validation requires a representable higher degree.")
    match form:
        case DifferentialForm():
            derivative = exterior_derivative(form)
        case DomainDifferentialForm():
            derivative = domain_exterior_derivative(form)
        case _:
            raise TypeError("form must be a DifferentialForm or DomainDifferentialForm.")
    smooth = integrate_form(derivative, bridge)
    discrete = bridge.complex.exterior_derivative(form.degree, source.values)
    maximum = jnp.max(jnp.abs(discrete - smooth.values), initial=0.0)
    relative = maximum / jnp.maximum(jnp.max(jnp.abs(smooth.values), initial=0.0), 1.0)
    return DeRhamCommutationEvidence(
        bridge.complex.realization_id,
        form.degree,
        valid=relative <= tolerance,
        maximum_residual=maximum,
        relative_residual=relative,
    )


def _parameterized_measures(
    parameters: CellParameterization, metric: RiemannianMetric, /
) -> Array:
    if not parameters.metric_compatible_quadrature:
        raise ValueError("Metric cell assembly requires nonnegative quadrature weights.")

    def cell_measure(cell: Array) -> Array:
        def density(reference: Array) -> Array:
            point = parameters.map_function(cell, reference)
            jacobian = parameters.jacobian_function(cell, reference)
            if point.shape != (parameters.ambient_dimension,) or jacobian.shape != (
                parameters.ambient_dimension,
                parameters.degree,
            ):
                raise ValueError(
                    "Cell maps and Jacobians must match their declared dimensions."
                )
            induced = jacobian.T @ metric(point) @ jacobian
            return jnp.sqrt(compound_matrix(induced, parameters.degree)[0, 0])

        return parameters.quadrature_weights @ jax.vmap(density)(
            parameters.reference_points
        )

    return jax.vmap(cell_measure)(jnp.arange(parameters.cell_count, dtype=jnp.int32))


def metric_dual_hodges(
    bridge: DeRhamBridge,
    metric: RiemannianMetric,
    /,
    *,
    dual: Sequence[CellParameterization],
) -> tuple[DiagonalHodge, ...]:
    """Metric-only DEC Hodges from paired primal/dual parameterized cells.

    ``dual[k]`` parameterizes dimension ``n-k`` cells, one for each primal
    degree-k cell. Material/constitutive coefficients do not belong here.
    """
    if not isinstance(bridge, DeRhamBridge) or not isinstance(metric, RiemannianMetric):
        raise TypeError("bridge and metric must be a DeRhamBridge and RiemannianMetric.")
    if not bridge.chart.compatible_with(metric.chart):
        raise ValueError("Bridge and metric charts must match.")
    parameters = tuple(dual)
    if len(parameters) != len(bridge.parameterizations):
        raise ValueError("One dual parameterization is required per primal degree.")
    hodges: list[DiagonalHodge] = []
    for degree, (primal, complementary) in enumerate(
        zip(bridge.parameterizations, parameters, strict=True)
    ):
        if not isinstance(complementary, CellParameterization):
            raise TypeError("dual must contain CellParameterization objects.")
        if (
            complementary.degree != bridge.chart.dimension - degree
            or complementary.cell_count != primal.cell_count
            or complementary.ambient_dimension != bridge.chart.dimension
        ):
            raise ValueError(
                "Dual cell degree, count, and ambient dimension are incompatible."
            )
        hodges.append(
            DiagonalHodge(
                _parameterized_measures(complementary, metric)
                / _parameterized_measures(primal, metric)
            )
        )
    return tuple(hodges)


__all__ = [
    "CellParameterization",
    "DeRhamBridge",
    "DeRhamCommutationEvidence",
    "simplicial_parameterizations",
    "structured_parameterizations",
    "integrate_form",
    "validate_de_rham_commutation",
    "metric_dual_hodges",
]
