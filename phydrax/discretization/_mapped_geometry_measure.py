#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Mapped coordinate laws, certified physical moments, and nested-map equality proofs."""

from __future__ import annotations

import fractions as imported_fractions
import math
from fractions import Fraction
from typing import (
    assert_never,
    NamedTuple,
)

import numpy as np
from numpy.typing import NDArray

import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.discretization._coordinate_budget import reserve_polynomial
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.geometry._rational import fraction_endpoint

from .._host_storage import declared_integer_storage_bytes
from .._strict import StrictModule
from .._validation import nonnegative_finite_float
from ..geometry._rational import rational_vector
from ..linalg import _hermitian_spectral as imported_linalg__hermitian_spectral
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import checked
from . import (
    _cell_geometry_transition_types as transition_types,
    _coordinate_enclosure as imported_coordinate_enclosure,
    _exact_algebraic_geometry as exact_algebraic_geometry_owner,
    _exact_power_consumers as imported_exact_power_consumers,
    _exact_rational_geometry as exact_rational_geometry_owner,
)
from ._cell_geometry import (
    _cardinal_corner_nodes,
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    require_scalar_coordinate_element,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
    SplineCellGeometryElement,
)
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from ._coordinate_enclosure import (
    add,
    affine_arguments,
    bernstein_coefficients,
    compose,
    coordinate_corner_images,
    coordinate_expressions,
    CoordinateCoefficients,
    CoordinateSourceBank,
    Expression,
    expression_add,
    expression_bounds,
    expression_compose,
    expression_derivative,
    expression_determinant,
    expression_multiply,
    expression_scale,
    Polynomial,
    RationalPolynomial,
    restrict_chart_expressions,
    scale,
    source_basis,
)
from ._exact_algebraic_geometry import coefficient_endpoint, rational_coefficient
from ._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ._nested_reference_types import (
    NestedReferencePair,
    rational_affine_bits,
    rational_pair_affine,
    rational_pair_arrays,
)
from ._reference_cell import reference_cell_topology


_EPSILON = float(np.finfo(np.float64).eps)


def _rational_polynomial(value: Polynomial, /) -> Polynomial:
    if all(isinstance(coefficient, Fraction) for coefficient in value.values()):
        return value
    return {
        index: rational_coefficient(
            coefficient, role="mapped_geometry.rational_integration.coefficients"
        )
        for index, coefficient in value.items()
    }


def _rational_expression(value: Expression, /) -> Expression:
    if isinstance(value, RationalPolynomial):
        numerator = _rational_polynomial(value.numerator)
        denominator = _rational_polynomial(value.denominator)
        if numerator is value.numerator and denominator is value.denominator:
            return value
        return RationalPolynomial(numerator, denominator)
    return _rational_polynomial(value)


def _rational_bernstein_coefficients(
    polynomial: Polynomial,
    domain: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, ...]:
    """Validate rational controls for rational-only square-root integration."""
    return tuple(
        rational_coefficient(
            value, role="mapped_geometry.rational_integration.coefficients"
        )
        for value in bernstein_coefficients(polynomial, domain, dimension, budget=budget)
    )


def _rational_expression_bernstein_coefficients(
    value: Expression,
    domain: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, ...]:
    """Validate quotient controls for rational-only square-root integration."""
    return tuple(
        rational_coefficient(
            coefficient, role="mapped_geometry.rational_integration.coefficients"
        )
        for coefficient in imported_coordinate_enclosure.expression_bernstein_coefficients(
            value, domain, dimension, budget=budget
        )
    )


def _publish_exact_integral(
    total: imported_coordinate_enclosure.ExactScalar,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    if isinstance(total, exact_algebraic_geometry_owner.RadicalElement):
        value, error = total.rounded(budget=budget)
        return value, fraction_endpoint(
            error, upper=True, saturate=True
        ) if error else 0.0
    value = float(total)
    return value, max(
        value - fraction_endpoint(total, upper=False, saturate=True),
        fraction_endpoint(total, upper=True, saturate=True) - value,
    )


@checked
def mapped_expression_restriction(
    polynomials: tuple[Expression, ...],
    source_kind: str,
    target_kind: str,
    matrix: NDArray[np.float64] | NDArray[np.object_],
    offset: NDArray[np.float64] | NDArray[np.object_],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:
    """Restrict the algebraic expression, retaining nonremovable quotients."""
    return restrict_chart_expressions(
        polynomials,
        source_kind,
        target_kind,
        offset,
        matrix,
        budget=budget,
    )


@checked
def mapped_coordinate_expressions(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:
    scalar = require_scalar_coordinate_element(element, "Mapped coordinate expressions")
    coordinates = coordinate_expressions(scalar, local, budget=budget)
    if coordinates is None:
        raise ValueError(
            "mapped_geometry.source_expressions must provide the coordinate source law; got absent source expressions"
        )
    return coordinates


def _restricted_mapped_density(
    element: RestrictedCellGeometryElement,
    coefficients: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    parent = element.source_element
    parent_density = mapped_density_expression(parent, coefficients, budget=budget)
    dimension = element.topological_dimension
    if parent.cell_kind == "pyramid":
        collapse = imported_coordinate_enclosure.power(
            add(
                imported_coordinate_enclosure.constant(1, dimension, budget=budget),
                scale(
                    imported_coordinate_enclosure.axes(dimension, budget=budget)[2],
                    -1,
                    budget=budget,
                ),
                budget=budget,
            ),
            2,
            dimension,
            budget=budget,
        )
        if isinstance(parent_density, RationalPolynomial):
            parent_density = imported_coordinate_enclosure.rational_expression(
                parent_density.numerator,
                imported_coordinate_enclosure.multiply(
                    parent_density.denominator, collapse, budget=budget
                ),
                budget=budget,
            )
        else:
            parent_density = imported_coordinate_enclosure.rational_expression(
                parent_density, collapse, budget=budget
            )
    matrix = np.asarray(element.matrix, dtype=np.float64)
    action = tuple(
        tuple(
            imported_coordinate_enclosure.constant(
                Fraction(float(value)), dimension, budget=budget
            )
            for value in row
        )
        for row in matrix
    )
    action_measure = imported_coordinate_enclosure.evaluate(
        imported_coordinate_enclosure.determinant(action, budget=budget),
        (Fraction(0),) * dimension,
        budget=budget,
    )
    if action_measure <= 0:
        raise ValueError(
            "mapped_geometry.jacobian must have a strictly positive certified bound; got nonpositive Jacobian bound"
        )
    density = expression_scale(
        mapped_expression_restriction(
            (parent_density,),
            parent.cell_kind,
            element.cell_kind,
            matrix,
            np.asarray(element.offset, dtype=np.float64),
            budget=budget,
        )[0],
        action_measure,
        budget=budget,
    )
    if element.cell_kind == "pyramid":
        density = expression_multiply(
            density,
            imported_coordinate_enclosure.power(
                add(
                    imported_coordinate_enclosure.constant(1, dimension, budget=budget),
                    scale(
                        imported_coordinate_enclosure.axes(dimension, budget=budget)[2],
                        -1,
                        budget=budget,
                    ),
                    budget=budget,
                ),
                2,
                dimension,
                budget=budget,
            ),
            budget=budget,
        )
    return density


def _affine_mapped_density(
    polynomial_coordinates: tuple[Polynomial, ...],
    dimension: int,
    axis_plane: bool,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    units = tuple(
        tuple(int(axis == column) for axis in range(dimension))
        for column in range(dimension)
    )
    jacobian_values = tuple(
        tuple(value.get(unit, Fraction(0)) for unit in units)
        for value in polynomial_coordinates
    )
    numerator, denominator = imported_coordinate_enclosure.coefficient_profile(
        polynomial_coordinates, budget=budget
    )
    match dimension:
        case 1:
            arithmetic, terms = 0, 1
        case 2:
            arithmetic, terms = 3, 2
        case 3:
            arithmetic, terms = 17, 6
        case _:
            raise ValueError(
                "mapped_geometry.dimension must be one through three; got unsupported dimension"
            )
    reserve_polynomial(
        dimension * dimension + arithmetic,
        terms,
        0,
        dimension * (numerator + denominator) + terms.bit_length(),
        budget=budget,
    )
    if any(
        isinstance(value, exact_algebraic_geometry_owner.RadicalElement)
        for row in jacobian_values
        for value in row
    ):
        exact_algebraic_geometry_owner.reserve_algebraic_work(
            jacobian_values, arithmetic, terms, dimension, budget=budget
        )
    if dimension == 1:
        determinant_value = jacobian_values[0][0]
    elif dimension == 2:
        determinant_value = (
            jacobian_values[0][0] * jacobian_values[1][1]
            - jacobian_values[0][1] * jacobian_values[1][0]
        )
    else:
        determinant_value = (
            jacobian_values[0][0]
            * (
                jacobian_values[1][1] * jacobian_values[2][2]
                - jacobian_values[1][2] * jacobian_values[2][1]
            )
            - jacobian_values[0][1]
            * (
                jacobian_values[1][0] * jacobian_values[2][2]
                - jacobian_values[1][2] * jacobian_values[2][0]
            )
            + jacobian_values[0][2]
            * (
                jacobian_values[1][0] * jacobian_values[2][1]
                - jacobian_values[1][1] * jacobian_values[2][0]
            )
        )
    if axis_plane:
        determinant_value = abs(determinant_value)
    if determinant_value <= 0:
        raise ValueError(
            "mapped_geometry.jacobian must have a strictly positive certified bound; got nonpositive Jacobian bound"
        )
    density = imported_coordinate_enclosure.constant(
        determinant_value, dimension, budget=budget
    )
    return density


def _mapped_density_coordinates(
    element: ScalarCoordinateElement,
    coordinates: tuple[Expression, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Expression, ...], bool]:
    """Admit the full-dimensional or single constant-axis density chart."""
    dimension = element.topological_dimension
    axis_plane = False
    if len(coordinates) != dimension:
        if (
            element.cell_kind != "quadrilateral"
            or dimension != 2
            or len(coordinates) != 3
        ):
            raise ValueError(
                "mapped_geometry.dimension must be full-dimensional or an axis-plane quadrilateral; got unsupported embedded geometry"
            )
        constant_axes = tuple(
            axis
            for axis, polynomial in enumerate(coordinates)
            if all(
                not expression_derivative(polynomial, reference_axis, budget=budget)
                for reference_axis in range(dimension)
            )
        )
        if len(constant_axes) != 1:
            raise ValueError(
                "mapped_geometry.axis_plane must have one constant coordinate expression; got different number of constant axes"
            )
        coordinates = tuple(
            polynomial
            for axis, polynomial in enumerate(coordinates)
            if axis != constant_axes[0]
        )
        axis_plane = True
    return coordinates, axis_plane


def _certify_mapped_density(
    element: ScalarCoordinateElement,
    density: Expression,
    axis_plane: bool,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    """Certify physical positivity before retaining the chart density."""
    dimension = element.topological_dimension
    domain = (
        "simplex"
        if element.cell_kind in ("interval", "triangle", "tetrahedron")
        else "prism"
        if element.cell_kind == "prism"
        else "box"
    )
    if element.cell_kind == "pyramid":
        # The collapsed chart determinant already includes (1-w)^2. Prove the
        # physical Jacobian positive before integrating that chart density.
        collapse = imported_coordinate_enclosure.power(
            add(
                imported_coordinate_enclosure.constant(1, dimension, budget=budget),
                scale(
                    imported_coordinate_enclosure.axes(dimension, budget=budget)[2],
                    -1,
                    budget=budget,
                ),
                budget=budget,
            ),
            2,
            dimension,
            budget=budget,
        )
        if isinstance(density, RationalPolynomial):
            physical_density = imported_coordinate_enclosure.rational_expression(
                density.numerator,
                imported_coordinate_enclosure.multiply(
                    density.denominator, collapse, budget=budget
                ),
                budget=budget,
            )
        else:
            physical_density = imported_coordinate_enclosure.rational_expression(
                density, collapse, budget=budget
            )
    else:
        physical_density = density
    lower, upper = expression_bounds(physical_density, domain, dimension, budget=budget)
    if axis_plane and upper < 0:
        density, lower = expression_scale(density, -1, budget=budget), -upper
    if lower <= 0:
        raise ValueError(
            "mapped_geometry.jacobian must have a strictly positive certified bound; got nonpositive Jacobian bound"
        )
    return density


def _numeric_mapped_density(
    coordinates: tuple[Expression, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    """Compute a signed determinant, without any domain or positivity verdict."""
    return expression_determinant(
        tuple(
            tuple(
                expression_derivative(value, axis, budget=budget)
                for axis in range(dimension)
            )
            for value in coordinates
        ),
        budget=budget,
    )


def _composed_mapped_density(
    element: PolynomialComposedCellGeometryElement,
    coefficients: CoordinateSourceBank,
    *,
    budget: CoordinateEnclosureBudget,
    identity: tuple[str, str],
) -> Expression:
    """Pull back numerical determinants; certify only the final target chart."""
    source = element.source_element
    dimension = element.topological_dimension
    coefficients, source_key = imported_coordinate_enclosure.coordinate_preparation_key(
        source, coefficients, budget=budget
    )
    from ._coordinate_operands import _signed_determinant_operands_for

    # The source determinant remains signed: only the final composed target
    # chart is certified below, never the source's whole physical domain.
    source_density = (
        _signed_determinant_operands_for(
            source_key, source.cell_kind, source.topological_dimension, budget,
            element=source,
        )
        if source.cell_kind != "pyramid"
        else None
    )
    if source_density is None:
        source_density = budget.composed_source_density_cache.get(source_key)
    if source_density is None:
        with budget.temporary_scope():
            coordinates = imported_coordinate_enclosure._coordinate_expressions_from_prepared(
                source, coefficients, source_key, budget=budget
            )
            if coordinates is None:
                raise ValueError("mapped_geometry.source_expressions must provide the coordinate source law; got absent source expressions")
            source_density = _numeric_mapped_density(
                coordinates,
                dimension,
                budget=budget,
            )
            budget.retain_basis((source_key, source_density))
            budget.composed_source_density_cache[source_key] = source_density
    # The composed element identity binds the actual chart coefficients as well
    # as its chart/source elements; the chart element alone is not a binding.
    chart_key = identity
    arguments = imported_coordinate_enclosure._reference_composition_from_identity(
        element, chart_key, budget=budget
    )
    chart_density = budget.composed_chart_density_cache.get(chart_key)
    if chart_density is None:
        with budget.temporary_scope():
            chart_density = _numeric_mapped_density(arguments, dimension, budget=budget)
            budget.retain_basis((chart_key, chart_density))
            budget.composed_chart_density_cache[chart_key] = chart_density
    composition = imported_coordinate_enclosure.ExpressionComposition(
        arguments, budget=budget
    )
    return expression_multiply(
        composition(source_density, budget=budget), chart_density, budget=budget
    )


@checked
def mapped_density_expression(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:

    ledger = budget
    element = require_scalar_coordinate_element(element, "Mapped density expressions")
    if isinstance(element, RestrictedCellGeometryElement):
        coefficients = imported_coordinate_enclosure.coordinate_source_rows(local)
        if len(coefficients[0]) == element.topological_dimension:
            return _restricted_mapped_density(element, coefficients, budget=budget)
    coefficients, cache_key = imported_coordinate_enclosure.coordinate_preparation_key(
        element, local, budget=budget
    )
    signed_density = None
    if (
        element.cell_kind != "pyramid"
        and not isinstance(element, RationalComposedCellGeometryElement)
        and len(coefficients[0]) == element.topological_dimension
    ):
        from ._coordinate_operands import _signed_determinant_operands_for

        # Only the receipt owner can admit an earned signed numerical operand.
        # In particular, an arbitrary public working-cache entry is not a
        # source, physical-frame, or paid-current-request binding.
        signed_density = _signed_determinant_operands_for(
            cache_key, element.cell_kind, element.topological_dimension, budget,
            element=element,
        )
    if cache_key in ledger.density_cache:
        return ledger.density_cache[cache_key][0]
    if signed_density is not None:
        # No positivity, orientation, tolerance, or measure verdict crosses the
        # earning boundary. Certify this consumer's chart exactly as on a miss.
        density = _certify_mapped_density(
            element, signed_density, False, budget=budget,
        )
        retained = (density,)
        ledger.retain_basis(retained)
        ledger.density_cache[cache_key] = retained
        return density
    if (
        isinstance(element, PolynomialComposedCellGeometryElement)
        and element.source_element.topological_dimension == element.topological_dimension
        and len(coefficients[0]) == element.topological_dimension
    ):
        # det(D(X o A)) = det(DX)(A) det(DA), including both signs.
        # A folded source can be valid on this target, so neither numerical
        # operand may pass through the source-domain positivity certificate.
        density = _certify_mapped_density(
            element,
            _composed_mapped_density(
                element, coefficients, identity=(cache_key[0], cache_key[1]), budget=budget
            ),
            False,
            budget=budget,
        )
        retained = (density,)
        ledger.retain_basis(retained)
        ledger.density_cache[cache_key] = retained
        return density
    coordinates = imported_coordinate_enclosure._coordinate_expressions_from_prepared(
        element, coefficients, cache_key, budget=budget
    )
    if coordinates is None:
        raise ValueError("mapped_geometry.source_expressions must provide the coordinate source law; got absent source expressions")
    dimension = element.topological_dimension
    coordinates, axis_plane = _mapped_density_coordinates(
        element, coordinates, budget=budget
    )
    polynomial_coordinates = tuple(
        value for value in coordinates if not isinstance(value, RationalPolynomial)
    )
    affine = len(polynomial_coordinates) == len(coordinates) and all(
        sum(index) <= 1 for value in polynomial_coordinates for index in value
    )
    if affine and element.cell_kind != "pyramid":
        density = _affine_mapped_density(
            polynomial_coordinates, dimension, axis_plane, budget=budget
        )
        retained = (density,)
        ledger.retain_basis(retained)
        ledger.density_cache[cache_key] = retained
        return density
    if affine:
        # Retained affine coordinate coefficients already prove a constant
        # Jacobian. Do not differentiate and reserve each discarded constant term.
        units = tuple(
            tuple(int(axis == column) for axis in range(dimension))
            for column in range(dimension)
        )
        jacobian = tuple(
            tuple(
                imported_coordinate_enclosure.constant(
                    value.get(unit, Fraction(0)), dimension, budget=budget
                )
                for unit in units
            )
            for value in polynomial_coordinates
        )
    else:
        jacobian = tuple(
            tuple(
                expression_derivative(value, axis, budget=budget)
                for axis in range(dimension)
            )
            for value in coordinates
        )
    density = expression_determinant(jacobian, budget=budget)
    density = _certify_mapped_density(element, density, axis_plane, budget=budget)
    retained = (density,)
    ledger.retain_basis(retained)
    ledger.density_cache[cache_key] = retained
    return density


@checked
def mapped_polynomial_integral_fraction(
    polynomial: Polynomial, kind: str, *, budget: CoordinateEnclosureBudget
) -> imported_coordinate_enclosure.ExactScalar:
    total: imported_coordinate_enclosure.ExactScalar = imported_fractions.Fraction(0)

    ledger = budget
    ledger.reserve(len(polynomial))
    for index, coefficient in polynomial.items():
        if isinstance(coefficient, exact_algebraic_geometry_owner.RadicalElement):
            exact_algebraic_geometry_owner.reserve_algebraic_work(
                (coefficient, total), 2, 2, 1, budget=budget
            )
        if kind in ("interval", "triangle", "tetrahedron"):
            moment = imported_fractions.Fraction(
                math.prod(math.factorial(i) for i in index),
                math.factorial(sum(index) + len(index)),
            )
        elif kind == "prism":
            moment = imported_fractions.Fraction(
                math.factorial(index[0]) * math.factorial(index[1]),
                math.factorial(index[0] + index[1] + 2) * (index[2] + 1),
            )
        else:
            moment = imported_fractions.Fraction(1, math.prod(i + 1 for i in index))
        total += coefficient * moment
    return total


@checked
def integrate_mapped_polynomial(
    polynomial: Expression, kind: str, *, budget: CoordinateEnclosureBudget
) -> tuple[float, float]:
    """algebraic source-expression integral and outward binary64 publication error."""

    if isinstance(polynomial, RationalPolynomial):
        value, error, _ = _integrate_mapped_rational(polynomial, kind, budget=budget)
        return value, error
    total = mapped_polynomial_integral_fraction(polynomial, kind, budget=budget)
    return _publish_exact_integral(total, budget=budget)


def _integrate_mapped_rational(
    value: RationalPolynomial,
    kind: str,
    *,
    absolute_tolerance: Fraction = Fraction(1, 10**12),
    relative_tolerance: float = 0.0,
    maximum_terms: int = 32,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float, bool]:
    """Integrate the algebraic quotient with a certified geometric-series remainder.

    An apex singularity of a tetrahedron is first removed by its algebraic cone
    chart. This is a change of variables of the expression, not a
    replacement by the corner interpolant.
    """

    dimension = len(next(iter(value.denominator)))
    if kind == "tetrahedron":
        vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
        singular = [
            index
            for index, vertex in enumerate(vertices)
            if not imported_coordinate_enclosure.evaluate(
                value.denominator,
                rational_vector(vertex),
                budget=budget,
            )
        ]
        if len(singular) == 1:
            apex = vertices[singular[0]]
            base = vertices[[index for index in range(4) if index != singular[0]]]
            u, v, w = imported_coordinate_enclosure.axes(3, budget=budget)
            collapse = add(
                imported_coordinate_enclosure.constant(1, 3, budget=budget),
                scale(w, -1, budget=budget),
                budget=budget,
            )
            base_chart = affine_arguments(base[0], (base[1:] - base[0]).T, budget=budget)
            triangle_chart = tuple(
                compose(term, (u, v), budget=budget) for term in base_chart
            )
            arguments = tuple(
                add(
                    imported_coordinate_enclosure.multiply(term, collapse, budget=budget),
                    scale(w, Fraction(float(x)), budget=budget),
                    budget=budget,
                )
                for term, x in zip(triangle_chart, apex, strict=True)
            )
            mapped = expression_compose(value, arguments, budget=budget)
            frame = np.column_stack(
                (base[1] - base[0], base[2] - base[0], apex - base[0])
            )
            factor = abs(
                imported_coordinate_enclosure.evaluate(
                    imported_coordinate_enclosure.determinant(
                        tuple(
                            tuple(
                                imported_coordinate_enclosure.constant(
                                    Fraction(float(entry)), 3, budget=budget
                                )
                                for entry in row
                            )
                            for row in frame
                        ),
                        budget=budget,
                    ),
                    (Fraction(0),) * 3,
                    budget=budget,
                )
            )
            mapped = expression_multiply(
                mapped,
                scale(
                    imported_coordinate_enclosure.power(collapse, 2, 3, budget=budget),
                    factor,
                    budget=budget,
                ),
                budget=budget,
            )
            if isinstance(mapped, RationalPolynomial):
                return _integrate_mapped_rational(
                    mapped,
                    "prism",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                    maximum_terms=maximum_terms,
                    budget=budget,
                )
            integral, error = integrate_mapped_polynomial(mapped, "prism", budget=budget)
            return integral, error, True
    domain = (
        "simplex"
        if kind in ("interval", "triangle", "tetrahedron")
        else "prism"
        if kind == "prism"
        else "box"
    )
    controls = bernstein_coefficients(value.denominator, domain, dimension, budget=budget)
    if any(
        isinstance(coefficient, exact_algebraic_geometry_owner.RadicalElement)
        for coefficient in controls
    ):
        exact_algebraic_geometry_owner.reserve_algebraic_work(
            controls, 4 * len(controls) + 8, 8, 2, budget=budget
        )
    lower, upper = min(controls), max(controls)
    if upper < 0:
        value = RationalPolynomial(
            scale(value.numerator, -1, budget=budget),
            scale(value.denominator, -1, budget=budget),
        )
        lower, upper = -upper, -lower
    if lower <= 0:
        raise ValueError(
            "mapped_geometry.denominator must have a strictly positive enclosure; got nonpositive lower bound"
        )
    center = (lower + upper) / 2
    ratio = add(
        imported_coordinate_enclosure.constant(1, dimension, budget=budget),
        scale(value.denominator, -1 / center, budget=budget),
        budget=budget,
    )
    rho = (upper - lower) / (upper + lower)
    numerator_bound = max(
        abs(coefficient)
        for coefficient in bernstein_coefficients(
            value.numerator, domain, dimension, budget=budget
        )
    )
    volume = mapped_polynomial_integral_fraction(
        imported_coordinate_enclosure.constant(1, dimension, budget=budget),
        kind,
        budget=budget,
    )
    with budget.live_storage() as term_owner, budget.live_storage() as total_owner:
        with budget.temporary_scope():
            before_term = budget.temporary_bytes_upper
            term = imported_coordinate_enclosure.constant(1, dimension, budget=budget)
            term_storage = budget.temporary_bytes_upper - before_term
        term_owner.set_bound(term_storage)
        total: imported_coordinate_enclosure.ExactScalar = Fraction(0)
        for order in range(maximum_terms):
            # Only the current series term and rational sum survive an order.
            # Products, integration scratch and the replaced term do not.
            with budget.temporary_scope():
                if isinstance(center, exact_algebraic_geometry_owner.RadicalElement):
                    exact_algebraic_geometry_owner.reserve_algebraic_work(
                        (center, rho, numerator_bound, volume, total),
                        32,
                        16,
                        order + 2,
                        budget=budget,
                    )
                total += (
                    mapped_polynomial_integral_fraction(
                        imported_coordinate_enclosure.multiply(
                            value.numerator, term, budget=budget
                        ),
                        kind,
                        budget=budget,
                    )
                    / center
                )
                if isinstance(total, Fraction):
                    total_storage = (
                        256
                        + declared_integer_storage_bytes(
                            abs(total.numerator).bit_length()
                        )
                        + declared_integer_storage_bytes(total.denominator.bit_length())
                    )
                else:
                    total_storage = 512 + len(total.terms) * (
                        512 + 16 * ((total.coefficient_bits + 7) // 8)
                    )
                total_owner.set_bound(total_storage)
                remainder = (
                    volume * numerator_bound * rho ** (order + 1) / (center * (1 - rho))
                )
                allowed = max(
                    absolute_tolerance,
                    Fraction(relative_tolerance)
                    * max(abs(total) - remainder, Fraction(0)),
                )
                if remainder <= allowed:
                    lower_result, upper_result = (
                        coefficient_endpoint(
                            total - remainder, upper=False, saturate=True
                        ),
                        coefficient_endpoint(
                            total + remainder, upper=True, saturate=True
                        ),
                    )
                    published = float(total)
                    return (
                        published,
                        max(published - lower_result, upper_result - published),
                        remainder == 0,
                    )
                before_term = budget.temporary_bytes_upper
                term = imported_coordinate_enclosure.multiply(term, ratio, budget=budget)
                term_storage = budget.temporary_bytes_upper - before_term
            term_owner.set_bound(term_storage)
    raise transition_types.CellGeometryTransitionError(
        "resource_limit",
        "mapped_geometry.remainder must fit the rational integration error budget; got series term limit",
        measured=maximum_terms,
        limit=maximum_terms,
    )


@checked
def mapped_geometry_cells(
    mesh: CellMesh, geometry: CellGeometrySpec, *, budget: CoordinateEnclosureBudget
) -> tuple[
    tuple[
        reference_types_owner.FiniteElementSpec
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement
        | SplineCellGeometryElement
        | LayerColumnCellGeometryElement,
        CoordinateSourceBank,
    ],
    ...,
]:

    # Resolve on the caller's ledger: its validation, charges and retained
    # source banks are shared with the bank read below.
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    local = imported_coordinate_enclosure.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    # The actual routed gathers and their tuple/host-route storage remain paid
    # even when the source numeric bank came from a completed predecessor.
    count = sum(route.shape[0] for route in routes)
    entries = sum(route.size for route in routes)
    budget.reserve(
        count + entries,
        256 + 144 * count + 16 * entries + 8 * max(
            (route.size for route in routes), default=0
        ),
    )
    return tuple(
        (
            require_scalar_coordinate_element(element, "Mapped geometry cells"),
            tuple(local[index] for index in row),
        )
        for element, route in zip(elements, routes, strict=True)
        for row in np.asarray(route, dtype=np.int64)
    )


def _dense_polynomial_interval(
    polynomial: Polynomial,
    *,
    monomial_bounds: dict[tuple[int, ...], imported_fractions.Fraction] | None = None,
    reference_radius: imported_fractions.Fraction | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], imported_fractions.Fraction]:

    shape = tuple(
        max((index[axis] for index in polynomial), default=0) + 1 for axis in range(2)
    )
    values = np.zeros(shape, dtype=np.float64)
    error = imported_fractions.Fraction(0)
    for index, coefficient in polynomial.items():
        coefficient = rational_coefficient(
            coefficient, role="mapped_geometry.rational_integration.coefficients"
        )
        values[index] = float(coefficient)
        bound = (
            imported_fractions.Fraction(1)
            if monomial_bounds is None
            else _monomial_uniform_bound(
                index, monomial_bounds, reference_radius, budget=budget
            )
        )
        error += (
            abs(coefficient - imported_fractions.Fraction(float(values[index]))) * bound
        )
    return values, error


def _dense_polynomial_norm(values: NDArray[np.float64]) -> imported_fractions.Fraction:

    return sum(
        (
            abs(imported_fractions.Fraction(float(value)))
            for value in values.flat
            if value
        ),
        imported_fractions.Fraction(0),
    )


def _monomial_uniform_bound(
    index: tuple[int, ...],
    cache: dict[tuple[int, ...], Fraction],
    radius: Fraction | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    value = cache.get(index)
    if value is not None:
        return value
    a, b = index

    reserve_polynomial(1, 1, 2, (a + b) * (a + b + 1).bit_length() + 2, budget=budget)
    # Centered coordinates lie in [-radius, radius]; source simplex coordinates
    # use their constrained monomial maximizer. Both norms are submultiplicative.
    value = (
        radius ** (a + b)
        if radius is not None
        else Fraction(a**a * b**b, (a + b) ** (a + b))
        if a and b
        else Fraction(1)
    )
    cache[index] = value
    return value


def _dense_domain_polynomial_norm(
    values: NDArray[np.float64],
    simplex_bounds: dict[tuple[int, ...], Fraction] | None,
    reference_radius: Fraction | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    if simplex_bounds is None:
        return _dense_polynomial_norm(values)
    return sum(
        (
            abs(Fraction(float(value)))
            * _monomial_uniform_bound(
                index, simplex_bounds, reference_radius, budget=budget
            )
            for index, value in np.ndenumerate(values)
            if value
        ),
        Fraction(0),
    )


def _dense_polynomial_product(
    first: NDArray[np.float64],
    first_error: imported_fractions.Fraction,
    second: NDArray[np.float64],
    second_error: imported_fractions.Fraction,
    work: list[int],
    *,
    simplex_bounds: dict[tuple[int, ...], imported_fractions.Fraction] | None = None,
    reference_radius: imported_fractions.Fraction | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], imported_fractions.Fraction]:
    """Dense coefficient product with a rigorous coefficient l1 error enclosure."""

    entries = [(index, float(value)) for index, value in np.ndenumerate(second) if value]
    products = first.size * len(entries)
    if work[0] + products > work[1]:
        raise transition_types.CellGeometryTransitionError(
            "resource_limit",
            "embedded_measure.work_units must fit the coefficient-product budget; got excess proposed work",
            measured=work[0] + products,
            limit=work[1],
        )

    ledger = budget
    ledger.reserve(products)
    work[0] += products
    result = np.zeros(
        tuple(a + b - 1 for a, b in zip(first.shape, second.shape, strict=True)),
        dtype=np.float64,
    )
    for (i, j), value in entries:
        result[i : i + first.shape[0], j : j + first.shape[1]] += value * first
    first_norm, second_norm = (
        _dense_domain_polynomial_norm(
            value, simplex_bounds, reference_radius, budget=budget
        )
        for value in (first, second)
    )
    rounding = imported_fractions.Fraction(float(np.finfo(np.float64).eps)) * (
        2 * len(entries) + 1
    )
    if rounding >= 1:
        raise ValueError(
            "embedded_measure.coefficient_error must fit the arithmetic range; got unrepresentable error bound"
        )
    error = (
        first_error * second_norm
        + second_error * first_norm
        + first_error * second_error
        + rounding / (1 - rounding) * first_norm * second_norm
        + products * imported_fractions.Fraction(float(np.nextafter(0.0, math.inf)))
    )
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "embedded_measure.coefficients must remain finite; got polynomial arithmetic overflow"
        )
    return result, error


def _dense_polynomial_integral_fraction(
    values: NDArray[np.float64],
    kind: str,
    weight: Polynomial | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> imported_fractions.Fraction:

    polynomial: Polynomial = {
        index: imported_fractions.Fraction(float(value))
        for index, value in np.ndenumerate(values)
        if value
    }
    result = mapped_polynomial_integral_fraction(
        polynomial
        if weight is None
        else imported_coordinate_enclosure.multiply(polynomial, weight, budget=budget),
        kind,
        budget=budget,
    )
    return rational_coefficient(
        result, role="mapped_geometry.rational_integration.coefficients"
    )


def _validate_embedded_denominators(
    coordinates: tuple[Expression, ...],
    cell_kind: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:

    domain = "simplex" if cell_kind == "triangle" else "box"
    denominators: list[tuple[Polynomial, RationalPolynomial]] = []
    budget = budget
    for coordinate in coordinates:
        if not isinstance(coordinate, RationalPolynomial):
            continue
        for denominator, _ in denominators:
            if budget is not None:
                budget.reserve(len(coordinate.denominator) + len(denominator))
            if len(coordinate.denominator) != len(denominator):
                continue
            pivot = next(iter(denominator))
            if pivot not in coordinate.denominator:
                continue
            ratio = coordinate.denominator[pivot] / denominator[pivot]
            if all(
                coordinate.denominator.get(index) == ratio * value
                for index, value in denominator.items()
            ):
                break
        else:
            denominators.append((coordinate.denominator, coordinate))
    for denominator, coordinate in denominators:
        for vertex in reference_cell_topology(cell_kind).vertices:
            point = rational_vector(vertex)
            if not imported_coordinate_enclosure.evaluate(
                denominator, point, budget=budget
            ):
                imported_coordinate_enclosure.expression_reference_evaluate(
                    coordinate, point, domain, budget=budget
                )


def _embedded_affine_triangle_squared(
    corners: CoordinateSourceBank, *, budget: CoordinateEnclosureBudget
) -> imported_coordinate_enclosure.ExactScalar:
    """Exact constant Gram determinant of a proved affine embedded triangle."""
    bits = max(
        abs(value.numerator).bit_length() + value.denominator.bit_length()
        if isinstance(value, Fraction)
        else 2 * value.coefficient_bits
        for point in corners
        for value in point
    )
    reserve_polynomial(20, 12, 0, 4 * bits + 4, budget=budget)
    if any(
        isinstance(value, exact_algebraic_geometry_owner.RadicalElement)
        for point in corners
        for value in point
    ):
        exact_algebraic_geometry_owner.reserve_algebraic_work(
            corners, 20, 12, 4, budget=budget
        )
    first = tuple(corners[1][axis] - corners[0][axis] for axis in range(3))
    second = tuple(corners[2][axis] - corners[0][axis] for axis in range(3))
    cross = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    squared = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]
    return squared


@checked
def embedded_squared_density(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    prepared_coordinates: tuple[Expression, ...] | None = None,
    denominators_certified: bool = False,
    budget: CoordinateEnclosureBudget,
) -> Expression:

    element = require_scalar_coordinate_element(element, "Embedded squared density")
    if (
        prepared_coordinates is None
        and element.cell_kind == "triangle"
        and element.topological_dimension == 2
        and len(local[0]) == 3
    ):
        corners = imported_coordinate_enclosure.affine_simplex_corner_images(
            element, local, budget=budget
        )
        if corners is not None:
            return imported_coordinate_enclosure.constant(
                _embedded_affine_triangle_squared(corners, budget=budget),
                2,
                budget=budget,
            )
    coordinates = (
        mapped_coordinate_expressions(element, local, budget=budget)
        if prepared_coordinates is None
        else prepared_coordinates
    )
    if (
        element.topological_dimension != 2
        or len(coordinates) != 3
        or element.cell_kind not in ("triangle", "quadrilateral")
    ):
        raise ValueError(
            "embedded_measure.element must map triangles or quadrilaterals into three dimensions; got unsupported coordinate map"
        )
    if not denominators_certified:
        _validate_embedded_denominators(coordinates, element.cell_kind, budget=budget)
    first = tuple(expression_derivative(value, 0, budget=budget) for value in coordinates)
    second = tuple(
        expression_derivative(value, 1, budget=budget) for value in coordinates
    )
    cross = tuple(
        expression_add(
            expression_multiply(first[i], second[j], budget=budget),
            expression_scale(
                expression_multiply(first[j], second[i], budget=budget), -1, budget=budget
            ),
            budget=budget,
        )
        for i, j in ((1, 2), (2, 0), (0, 1))
    )
    return imported_coordinate_enclosure.expression_sum(
        tuple(expression_multiply(value, value, budget=budget) for value in cross),
        budget=budget,
    )


def _compress_dense_interval(
    values: NDArray[np.float64],
    publication_error: Fraction,
    omitted_budget: Fraction,
    /,
    *,
    simplex_bounds: dict[tuple[int, ...], Fraction] | None = None,
    reference_radius: Fraction | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], Fraction]:
    """Discard coherent degree tails only under an algebraic uniform-error bound."""
    degree = sum(size - 1 for size in values.shape)
    if degree <= 4 or omitted_budget <= 0:
        return values, publication_error
    by_degree = [Fraction(0) for _ in range(degree + 1)]
    for index, value in np.ndenumerate(values):
        if value:
            bound = (
                Fraction(1)
                if simplex_bounds is None
                else _monomial_uniform_bound(
                    index, simplex_bounds, reference_radius, budget=budget
                )
            )
            by_degree[sum(index)] += abs(Fraction(float(value))) * bound
    discarded, retained_degree = Fraction(0), degree
    for candidate in range(degree, 4, -1):
        if discarded + by_degree[candidate] > omitted_budget:
            break
        discarded += by_degree[candidate]
        retained_degree = candidate - 1
    if retained_degree == degree:
        return values, publication_error
    shape = tuple(min(size, retained_degree + 1) for size in values.shape)
    compressed = np.array(values[: shape[0], : shape[1]], dtype=np.float64, copy=True)
    for index in np.ndindex(compressed.shape):
        if sum(index) > retained_degree:
            compressed[index] = 0.0
    return compressed, publication_error + discarded


def _compressed_dense_polynomial_interval(
    polynomial: Polynomial,
    omitted_budget: Fraction,
    /,
    *,
    simplex_bounds: dict[tuple[int, ...], Fraction] | None = None,
    reference_radius: Fraction | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], Fraction]:
    values, error = _dense_polynomial_interval(
        polynomial,
        monomial_bounds=simplex_bounds,
        reference_radius=reference_radius,
        budget=budget,
    )
    return _compress_dense_interval(
        values,
        error,
        omitted_budget,
        simplex_bounds=simplex_bounds,
        reference_radius=reference_radius,
        budget=budget,
    )


def _centered_weighted_integral(
    values: NDArray[np.float64],
    kind: str,
    local_weight: Polynomial | None,
    moment_cache: dict[tuple[int, int], Fraction],
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:

    ledger = budget
    terms = tuple(
        (index, float(value)) for index, value in np.ndenumerate(values) if value
    )
    ledger.reserve(len(terms))
    result = imported_fractions.Fraction(0)
    weights = (
        {(0, 0): imported_fractions.Fraction(1)} if local_weight is None else local_weight
    )
    for index, value in terms:
        moment = moment_cache.get(index)
        if moment is None:
            moment = imported_fractions.Fraction(0)
            for power, coefficient_ in weights.items():
                coefficient_ = rational_coefficient(
                    coefficient_, role="mapped_geometry.rational_integration.coefficients"
                )
                exponents = tuple(a + b for a, b in zip(index, power, strict=True))
                # algebraic moments of the source domain in coordinates
                # (u - 1/2, v - 1/2), not a box surrogate for the triangle.
                a, b = exponents

                def interval_moment(order: int) -> imported_fractions.Fraction:
                    return (
                        imported_fractions.Fraction(0)
                        if order % 2
                        else imported_fractions.Fraction(1, (order + 1) * 2**order)
                    )

                functional = (
                    (
                        (-1) ** (b + 1) * interval_moment(a + b + 1)
                        - imported_fractions.Fraction(-1, 2) ** (b + 1)
                        * interval_moment(a)
                    )
                    / (b + 1)
                    if kind == "triangle"
                    else interval_moment(a) * interval_moment(b)
                )
                moment += coefficient_ * functional
            moment_cache[index] = moment
        result += imported_fractions.Fraction(value) * moment
    return result


@checked
def sqrt_polynomial_integral_piece(
    polynomial: Polynomial,
    kind: str,
    absolute_tolerance: imported_fractions.Fraction | float,
    relative_tolerance: float,
    maximum_terms: int,
    work: list[int],
    *,
    weight: Polynomial | None = None,
    budget: CoordinateEnclosureBudget,
) -> (
    tuple[
        imported_fractions.Fraction,
        imported_fractions.Fraction,
        imported_fractions.Fraction,
    ]
    | None
):
    """Validated binomial approximation with enclosed coefficient compression."""

    polynomial = _rational_polynomial(polynomial)
    if weight is not None:
        weight = _rational_polynomial(weight)
    domain = "simplex" if kind == "triangle" else "box"
    coefficients = _rational_bernstein_coefficients(polynomial, domain, 2, budget=budget)
    lower, upper = min(coefficients), max(coefficients)
    if upper <= 0:
        raise ValueError(
            "embedded_measure.gram must have a positive upper measure bound; got nonpositive Gram measure"
        )
    if lower <= 0:
        return None
    center = (lower + upper) / 2
    ratio = add(
        scale(polynomial, 1 / center, budget=budget),
        imported_coordinate_enclosure.constant(-1, 2, budget=budget),
        budget=budget,
    )
    # Bernstein positivity gives rho < 1. High-degree conditioning is decided
    # by its propagated arithmetic enclosure, not a coefficient-norm heuristic.
    high_degree = max((sum(index) for index in ratio), default=0) > 4
    simplex_bounds: dict[tuple[int, ...], imported_fractions.Fraction] | None = (
        {} if high_degree and kind == "triangle" else None
    )
    if not high_degree and sum(
        (abs(value) for value in ratio.values()), imported_fractions.Fraction(0)
    ) > imported_fractions.Fraction(1, 2):
        return None
    rho = (upper - lower) / (upper + lower)
    root_lower, root_upper = fraction_sqrt_interval(center, budget=budget)
    root_midpoint, root_error = (
        (root_lower + root_upper) / 2,
        (root_upper - root_lower) / 2,
    )
    reference_volume = (
        imported_fractions.Fraction(1, 2)
        if kind == "triangle"
        else imported_fractions.Fraction(1)
    )
    weight_coefficients = (
        (imported_fractions.Fraction(1),)
        if weight is None
        else _rational_bernstein_coefficients(weight, domain, 2, budget=budget)
    )
    weight_sup = max(abs(value) for value in weight_coefficients)
    weight_integral = (
        reference_volume
        if weight is None
        else rational_coefficient(
            mapped_polynomial_integral_fraction(weight, kind, budget=budget),
            role="mapped_geometry.rational_integration.coefficients",
        )
    )
    lower_integral = (
        fraction_sqrt_interval(lower, budget=budget)[0] * abs(weight_integral)
        if min(weight_coefficients) >= 0 or max(weight_coefficients) <= 0
        else imported_fractions.Fraction(0)
    )
    tolerance = max(
        imported_fractions.Fraction(absolute_tolerance),
        imported_fractions.Fraction(relative_tolerance) * lower_integral,
    )
    compression_budget = (
        imported_fractions.Fraction(0)
        if weight_sup == 0
        else min(
            tolerance / (16 * root_upper * reference_volume * weight_sup), (1 - rho) / 16
        )
    )
    reference_radius = imported_fractions.Fraction(1, 2) if high_degree else None
    local_weight = weight
    if high_degree:
        centered = affine_arguments(np.asarray((0.5, 0.5)), np.eye(2), budget=budget)
        ratio = compose(ratio, centered, budget=budget)
        local_weight = (
            None if weight is None else compose(weight, centered, budget=budget)
        )
        simplex_bounds = {}
    ratio_values, ratio_error = _compressed_dense_polynomial_interval(
        ratio,
        compression_budget,
        simplex_bounds=simplex_bounds,
        reference_radius=reference_radius,
        budget=budget,
    )
    power_values, power_error = (
        np.ones((1, 1), dtype=np.float64),
        imported_fractions.Fraction(0),
    )
    series_integral, integral_error = (
        imported_fractions.Fraction(0),
        imported_fractions.Fraction(0),
    )
    coefficient = imported_fractions.Fraction(1)
    moment_cache: dict[tuple[int, int], imported_fractions.Fraction] = {}

    for order in range(maximum_terms):
        integral = (
            _centered_weighted_integral(
                power_values, kind, local_weight, moment_cache, budget=budget
            )
            if high_degree
            else _dense_polynomial_integral_fraction(
                power_values, kind, weight, budget=budget
            )
        )
        series_integral += coefficient * integral
        integral_error += abs(coefficient) * power_error * reference_volume * weight_sup
        next_coefficient = (
            coefficient * (imported_fractions.Fraction(1, 2) - order) / (order + 1)
        )
        remainder = (
            abs(next_coefficient)
            * rho ** (order + 1)
            / (1 - rho)
            * reference_volume
            * weight_sup
        )
        error = root_upper * (integral_error + remainder) + root_error * abs(
            series_integral
        )
        if error <= tolerance / 2:
            return root_midpoint * series_integral, error, tolerance
        if high_degree and root_upper * integral_error > tolerance / 2:
            return None
        if order + 1 < maximum_terms:
            power_values, power_error = _dense_polynomial_product(
                power_values,
                power_error,
                ratio_values,
                ratio_error,
                work,
                simplex_bounds=simplex_bounds,
                reference_radius=reference_radius,
                budget=budget,
            )
            if high_degree and compression_budget:
                power_values, power_error = _compress_dense_interval(
                    power_values,
                    power_error,
                    compression_budget / (maximum_terms * abs(next_coefficient)),
                    simplex_bounds=simplex_bounds,
                    reference_radius=reference_radius,
                    budget=budget,
                )
            coefficient = next_coefficient
    return None


def embedded_measure_submaps(
    kind: str,
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
    half = 0.5 * np.eye(2)
    if kind == "quadrilateral":
        return tuple((half, np.asarray([x, y])) for x in (0.0, 0.5) for y in (0.0, 0.5))
    return (
        (half, np.asarray([0.0, 0.0])),
        (half, np.asarray([0.5, 0.0])),
        (half, np.asarray([0.0, 0.5])),
        (np.asarray([[0.0, -0.5], [0.5, 0.5]]), np.asarray([0.5, 0.0])),
    )


def _sqrt_weighted_integral_pass(
    gram: Polynomial,
    weight: Polynomial,
    kind: str,
    tolerance: imported_fractions.Fraction,
    maximum_terms: int,
    work: list[int],
    subcells: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:

    ledger = budget
    pending = [(gram, weight, imported_fractions.Fraction(1))]
    value, error = imported_fractions.Fraction(0), imported_fractions.Fraction(0)
    while pending:
        piece, local_weight, fraction = pending.pop()
        subcells[0] += 1
        if subcells[0] > subcells[1]:
            raise transition_types.CellGeometryTransitionError(
                "resource_limit",
                "embedded_measure.subcells must fit the reference-subdivision budget; got excess visited pieces",
                measured=subcells[0],
                limit=subcells[1],
            )
        with ledger.temporary_scope():
            integral = sqrt_polynomial_integral_piece(
                piece,
                kind,
                tolerance,
                0.0,
                maximum_terms,
                work,
                weight=local_weight,
                budget=budget,
            )
        if integral is None:
            for matrix, offset in embedded_measure_submaps(kind):
                arguments = affine_arguments(offset, matrix, budget=budget)
                with ledger.temporary_scope():
                    child_gram, child_weight = (
                        compose(piece, arguments, budget=budget),
                        compose(local_weight, arguments, budget=budget),
                    )
                _certified_sqrt_retain_expressions(
                    (child_gram, child_weight), budget=budget
                )
                pending.append((child_gram, child_weight, fraction / 4))
        else:
            central, bound, _ = integral
            value += fraction * central
            error += fraction * bound
    published = float(value)
    error += abs(imported_fractions.Fraction(published) - value)
    return published, (
        fraction_endpoint(error, upper=True, saturate=True) if error else 0.0
    )


def _certified_sqrt_exact_polynomial_root(
    value: Polynomial, work: list[int], *, budget: CoordinateEnclosureBudget
) -> Polynomial | None:
    """Extract an algebraic polynomial square by leading-term division, or decline."""

    if not value:
        return {}
    leading = max(value)
    coefficient = value[leading]
    coefficient = rational_coefficient(
        coefficient, role="mapped_geometry.rational_integration.coefficients"
    )
    if coefficient < 0 or any(index % 2 for index in leading):
        return None
    a, b = math.isqrt(coefficient.numerator), math.isqrt(coefficient.denominator)
    if a * a != coefficient.numerator or b * b != coefficient.denominator:
        return None
    root_index = tuple(index // 2 for index in leading)
    root_coefficient = Fraction(a, b)
    root = {root_index: root_coefficient}
    while True:
        proposed = work[0] + len(root) ** 2
        if proposed > work[1]:
            raise transition_types.CellGeometryTransitionError(
                "resource_limit",
                "embedded_measure.work_units must fit the square-extraction budget; got excess proposed work",
                measured=proposed,
                limit=work[1],
            )
        work[0] = proposed
        residual = add(
            value,
            scale(
                imported_coordinate_enclosure.multiply(root, root, budget=budget),
                -1,
                budget=budget,
            ),
            budget=budget,
        )
        if not residual:
            return root
        residual_index = max(residual)
        next_index = tuple(i - j for i, j in zip(residual_index, root_index, strict=True))
        if any(index < 0 for index in next_index) or next_index >= min(root):
            return None
        root[next_index] = residual[residual_index] / (2 * root_coefficient)


def _certified_sqrt_dense_sum(
    first: NDArray[np.float64],
    first_error: Fraction,
    second: NDArray[np.float64],
    second_error: Fraction,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], Fraction]:
    """Enclose coefficient addition exactly, including cancellation and rounding."""
    shape = tuple(max(a, b) for a, b in zip(first.shape, second.shape, strict=True))
    entries = math.prod(shape)
    proposed = work[0] + entries
    if proposed > work[1]:
        raise transition_types.CellGeometryTransitionError(
            "resource_limit",
            "embedded_measure.work_units must fit the coefficient-sum budget; got excess proposed work",
            measured=proposed,
            limit=work[1],
        )
    work[0] = proposed

    ledger = budget
    ledger.reserve(entries, 3 * entries * np.dtype(np.float64).itemsize)
    result = np.zeros(shape, dtype=np.float64)
    result[: first.shape[0], : first.shape[1]] = first
    result[: second.shape[0], : second.shape[1]] += second
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "embedded_measure.coefficients must remain finite; got reciprocal coefficient overflow"
        )
    rounding = Fraction(0)
    for index, value in np.ndenumerate(result):
        a = (
            first[index]
            if all(i < n for i, n in zip(index, first.shape, strict=True))
            else 0.0
        )
        b = (
            second[index]
            if all(i < n for i, n in zip(index, second.shape, strict=True))
            else 0.0
        )
        rounding += abs(Fraction(float(value)) - Fraction(float(a)) - Fraction(float(b)))
    return result, first_error + second_error + rounding


def _certified_sqrt_dense_product(
    first: NDArray[np.float64],
    first_error: Fraction,
    second: NDArray[np.float64],
    second_error: Fraction,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], Fraction]:
    """Reserve dense scratch and coefficient-entry storage before convolution."""

    ledger = budget
    shape = tuple(a + b - 1 for a, b in zip(first.shape, second.shape, strict=True))
    ledger.reserve(
        0, math.prod(shape) * np.dtype(np.float64).itemsize + second.size * 256
    )
    return _dense_polynomial_product(
        first, first_error, second, second_error, work, budget=budget
    )


def _certified_sqrt_dense_weight_piece(
    polynomial: Polynomial,
    weight: NDArray[np.float64],
    weight_error: Fraction,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, Fraction] | None:
    """Polynomial moments with enclosed dense products, without a Fraction expansion."""

    domain = "simplex" if kind == "triangle" else "box"
    controls = _rational_bernstein_coefficients(polynomial, domain, 2, budget=budget)
    lower, upper = min(controls), max(controls)
    if upper <= 0:
        raise ValueError(
            "embedded_measure.gram must have a positive upper measure bound; got nonpositive Gram measure"
        )
    if lower <= 0:
        return None
    center = (lower + upper) / 2
    ratio = add(
        scale(polynomial, 1 / center, budget=budget),
        imported_coordinate_enclosure.constant(-1, 2, budget=budget),
        budget=budget,
    )
    if sum((abs(value) for value in ratio.values()), Fraction(0)) > Fraction(1, 8):
        return None
    rho = (upper - lower) / (upper + lower)
    root_lower, root_upper = fraction_sqrt_interval(center, budget=budget)
    root_midpoint, root_error = (
        (root_lower + root_upper) / 2,
        (root_upper - root_lower) / 2,
    )
    volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
    weight_sup = _dense_polynomial_norm(weight) + weight_error
    ratio_values, ratio_error = _dense_polynomial_interval(ratio, budget=budget)
    power_values, power_error = np.ones((1, 1), dtype=np.float64), Fraction(0)
    series_values, series_error = np.zeros((1, 1), dtype=np.float64), Fraction(0)
    coefficient = Fraction(1)
    for order in range(maximum_terms):
        coefficient_values, coefficient_error = _dense_polynomial_interval(
            imported_coordinate_enclosure.constant(coefficient, 2, budget=budget),
            budget=budget,
        )
        term_values, term_error = _certified_sqrt_dense_product(
            power_values,
            power_error,
            coefficient_values,
            coefficient_error,
            work,
            budget=budget,
        )
        series_values, series_error = _certified_sqrt_dense_sum(
            series_values,
            series_error,
            term_values,
            term_error,
            work,
            budget=budget,
        )
        next_coefficient = coefficient * (Fraction(1, 2) - order) / (order + 1)
        remainder = abs(next_coefficient) * rho ** (order + 1) / (1 - rho)
        approximation_error = (
            root_upper * (series_error + remainder) * volume * weight_sup
        )
        if approximation_error <= tolerance / 8:
            break
        if order + 1 < maximum_terms:
            power_values, power_error = _certified_sqrt_dense_product(
                power_values,
                power_error,
                ratio_values,
                ratio_error,
                work,
                budget=budget,
            )
            coefficient = next_coefficient
    else:
        return None
    factor_values, factor_error = _dense_polynomial_interval(
        imported_coordinate_enclosure.constant(root_midpoint, 2, budget=budget),
        budget=budget,
    )
    density_values, density_error = _certified_sqrt_dense_product(
        series_values,
        series_error + remainder,
        factor_values,
        factor_error + root_error,
        work,
        budget=budget,
    )
    weighted_values, weighted_error = _certified_sqrt_dense_product(
        density_values,
        density_error,
        weight,
        weight_error,
        work,
        budget=budget,
    )
    error = weighted_error * volume
    if error > tolerance / 2:
        return None
    return _dense_polynomial_integral_fraction(
        weighted_values, kind, budget=budget
    ), error


def _certified_sqrt_rational_piece(
    gram: Expression,
    weight: Expression,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, Fraction] | None:
    """Reduce sqrt(N/D)*A/B to sqrt(N*D)*A/(D*B), with an enclosed reciprocal."""

    domain = "simplex" if kind == "triangle" else "box"
    numerator, denominator = imported_coordinate_enclosure.expression_parts(
        gram, 2, budget=budget
    )
    weighted_numerator, weighted_denominator = (
        imported_coordinate_enclosure.expression_parts(weight, 2, budget=budget)
    )

    def product(first: Polynomial, second: Polynomial) -> Polynomial:
        proposed = work[0] + len(first) * len(second)
        if proposed > work[1]:
            raise transition_types.CellGeometryTransitionError(
                "resource_limit",
                "embedded_measure.work_units must fit the coefficient-product budget; got excess proposed work",
                measured=proposed,
                limit=work[1],
            )
        work[0] = proposed
        return imported_coordinate_enclosure.multiply(first, second, budget=budget)

    for divisor in (denominator, weighted_denominator):
        if min(_rational_bernstein_coefficients(divisor, domain, 2, budget=budget)) <= 0:
            raise ValueError(
                "embedded_measure.denominator must have a strictly positive enclosure; got nonpositive denominator bound"
            )
    denominator_root = _certified_sqrt_exact_polynomial_root(
        denominator, work, budget=budget
    )
    if denominator_root is not None:
        root_controls = _rational_bernstein_coefficients(
            denominator_root, domain, 2, budget=budget
        )
        if max(root_controls) < 0:
            denominator_root = scale(denominator_root, -1, budget=budget)
        elif min(root_controls) <= 0:
            denominator_root = None
    polynomial = (
        numerator if denominator_root is not None else product(numerator, denominator)
    )
    divisor = product(
        denominator_root if denominator_root is not None else denominator,
        weighted_denominator,
    )
    if set(polynomial) == {(0, 0)} and set(divisor) == {(0, 0)}:
        root_lower, root_upper = fraction_sqrt_interval(
            rational_coefficient(
                polynomial[(0, 0)],
                role="mapped_geometry.rational_integration.coefficients",
            ),
            budget=budget,
        )
        moment = rational_coefficient(
            mapped_polynomial_integral_fraction(weighted_numerator, kind, budget=budget),
            role="mapped_geometry.rational_integration.coefficients",
        ) / rational_coefficient(
            divisor[(0, 0)], role="mapped_geometry.rational_integration.coefficients"
        )
        return (root_lower + root_upper) * moment / 2, (root_upper - root_lower) * abs(
            moment
        ) / 2
    exponent = 1
    while len(divisor) > 1:
        base = _certified_sqrt_exact_polynomial_root(divisor, work, budget=budget)
        if base is None:
            break
        base_controls = _rational_bernstein_coefficients(base, domain, 2, budget=budget)
        if max(base_controls) < 0:
            base = scale(base, -1, budget=budget)
        elif min(base_controls) <= 0:
            break
        divisor, exponent = base, 2 * exponent
    controls = _rational_bernstein_coefficients(divisor, domain, 2, budget=budget)
    lower, upper = min(controls), max(controls)
    if lower <= 0:
        return None
    center = (lower + upper) / 2
    ratio = add(
        imported_coordinate_enclosure.constant(1, 2, budget=budget),
        scale(divisor, -1 / center, budget=budget),
        budget=budget,
    )
    rho = (upper - lower) / (upper + lower)
    if sum((abs(value) for value in ratio.values()), Fraction(0)) > Fraction(1, 8):
        return None
    density_controls = _rational_bernstein_coefficients(
        polynomial, domain, 2, budget=budget
    )
    density_lower, density_upper = min(density_controls), max(density_controls)
    if density_upper <= 0:
        raise ValueError(
            "embedded_measure.gram must have a positive upper measure bound; got nonpositive Gram measure"
        )
    if density_lower <= 0:
        return None
    root_ratio = add(
        scale(polynomial, 2 / (density_lower + density_upper), budget=budget),
        imported_coordinate_enclosure.constant(-1, 2, budget=budget),
        budget=budget,
    )
    if sum((abs(value) for value in root_ratio.values()), Fraction(0)) > Fraction(1, 8):
        return None
    weight_upper = max(
        abs(value)
        for value in _rational_bernstein_coefficients(
            weighted_numerator, domain, 2, budget=budget
        )
    )
    volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
    ratio_values, ratio_error = _dense_polynomial_interval(ratio, budget=budget)
    series_values, series_error = np.zeros((1, 1), dtype=np.float64), Fraction(0)
    power_values, power_error = np.ones((1, 1), dtype=np.float64), Fraction(0)
    remainder = Fraction(0)
    coefficient = Fraction(1)
    for order in range(maximum_terms):
        coefficient_values, coefficient_error = _dense_polynomial_interval(
            imported_coordinate_enclosure.constant(coefficient, 2, budget=budget),
            budget=budget,
        )
        term_values, term_error = _certified_sqrt_dense_product(
            power_values,
            power_error,
            coefficient_values,
            coefficient_error,
            work,
            budget=budget,
        )
        series_values, series_error = _certified_sqrt_dense_sum(
            series_values,
            series_error,
            term_values,
            term_error,
            work,
            budget=budget,
        )
        next_coefficient = coefficient * Fraction(exponent + order, order + 1)
        tail_ratio = rho * Fraction(exponent + order + 1, order + 2)
        if tail_ratio < 1:
            remainder = (
                fraction_sqrt_interval(density_upper, budget=budget)[1]
                * weight_upper
                * volume
                * next_coefficient
                * rho ** (order + 1)
                / (center**exponent * (1 - tail_ratio))
            )
            if remainder <= tolerance / 4:
                break
        if order + 1 < maximum_terms:
            power_values, power_error = _certified_sqrt_dense_product(
                power_values,
                power_error,
                ratio_values,
                ratio_error,
                work,
                budget=budget,
            )
            coefficient = next_coefficient
    else:
        return None
    numerator_values, numerator_error = _dense_polynomial_interval(
        scale(weighted_numerator, 1 / center**exponent, budget=budget),
        budget=budget,
    )
    weight_values, weight_error = _certified_sqrt_dense_product(
        numerator_values,
        numerator_error,
        series_values,
        series_error,
        work,
        budget=budget,
    )
    integral = _certified_sqrt_dense_weight_piece(
        polynomial,
        weight_values,
        weight_error,
        kind,
        tolerance / 2,
        maximum_terms,
        work,
        budget=budget,
    )
    if integral is None:
        return None
    value, error = integral
    return value, error + remainder


def _certified_sqrt_retain_expressions(
    values: tuple[Expression, ...], *, budget: CoordinateEnclosureBudget
) -> None:
    """Account for live coefficient data, not released expansion intermediates."""

    parts: tuple[Polynomial, ...] = tuple(
        polynomial
        for value in values
        for polynomial in (
            (value.numerator, value.denominator)
            if isinstance(value, RationalPolynomial)
            else (value,)
        )
    )
    numerator_bits, denominator_bits = imported_coordinate_enclosure.coefficient_profile(
        parts, budget=budget
    )
    reserve_polynomial(
        0,
        sum(len(part) for part in parts),
        2,
        numerator_bits + denominator_bits,
        budget=budget,
    )


def _certified_sqrt_expression_child(
    gram: Expression,
    weight: Expression,
    arguments: tuple[Polynomial, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, Expression]:
    """Release expansion scratch while retaining the queued source data."""
    with budget.temporary_scope():
        child_gram = expression_compose(gram, arguments, budget=budget)
        child_weight = expression_compose(weight, arguments, budget=budget)
    _certified_sqrt_retain_expressions((child_gram, child_weight), budget=budget)
    return child_gram, child_weight


def _certified_sqrt_expression_pass(
    gram: Expression,
    weight: Expression,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
    subcells: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:

    ledger = budget

    submaps = (
        embedded_measure_submaps(kind)
        if kind != "interval"
        else tuple(
            (
                np.asarray(((0.5, 0.0), (0.0, 1.0)), dtype=np.float64),
                np.asarray((offset, 0.0), dtype=np.float64),
            )
            for offset in (0.0, 0.5)
        )
    )
    pending = [(gram, weight, Fraction(1))]
    value, error = Fraction(0), Fraction(0)
    while pending:
        piece, local_weight, fraction = pending.pop()
        subcells[0] += 1
        if subcells[0] > subcells[1]:
            raise transition_types.CellGeometryTransitionError(
                "resource_limit",
                "embedded_measure.subcells must fit the reference-subdivision budget; got excess visited pieces",
                measured=subcells[0],
                limit=subcells[1],
            )
        with ledger.temporary_scope():
            integral = _certified_sqrt_rational_piece(
                piece,
                local_weight,
                "quadrilateral" if kind == "interval" else kind,
                tolerance,
                maximum_terms,
                work,
                budget=budget,
            )
        if integral is None:
            for matrix, offset in submaps:
                arguments = affine_arguments(offset, matrix, budget=budget)
                child_gram, child_weight = _certified_sqrt_expression_child(
                    piece,
                    local_weight,
                    arguments,
                    budget=ledger,
                )
                pending.append(
                    (
                        child_gram,
                        child_weight,
                        fraction / (2 if kind == "interval" else 4),
                    )
                )
        else:
            central, bound = integral
            value += fraction * central
            error += fraction * bound
    published = float(value)
    error += abs(Fraction(published) - value)
    return published, fraction_endpoint(
        error, upper=True, saturate=True
    ) if error else 0.0


def _certified_sqrt_cone_transform(
    gram: Expression, weight: Expression, kind: str, *, budget: CoordinateEnclosureBudget
) -> tuple[Expression, Expression, str]:
    """Integrate a bounded rational apex through its algebraic measure-preserving cone.

    Coordinate corner values belong to the source-limit owner. Tangent Gram
    may have a bounded directional limit; no value is assigned at that null set.
    """

    if kind != "triangle":
        return gram, weight, kind
    vertices = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    singular = {
        vertex
        for value in (gram, weight)
        if isinstance(value, RationalPolynomial)
        for vertex in vertices
        if not imported_coordinate_enclosure.evaluate(
            value.denominator, vertex, budget=budget
        )
    }
    if not singular:
        return gram, weight, kind
    if len(singular) != 1:
        raise ValueError(
            "embedded_measure.cone must have one removable source corner; got unsupported corner singularity"
        )
    apex = next(iter(singular))
    for value in (gram, weight):
        _rational_expression_bernstein_coefficients(value, "simplex", 2, budget=budget)
    base = tuple(vertex for vertex in vertices if vertex != apex)
    u, collapse = imported_coordinate_enclosure.axes(2, budget=budget)
    arguments = tuple(
        add(
            imported_coordinate_enclosure.constant(a, 2, budget=budget),
            imported_coordinate_enclosure.multiply(
                collapse,
                add(
                    imported_coordinate_enclosure.constant(b - a, 2, budget=budget),
                    scale(u, c - b, budget=budget),
                    budget=budget,
                ),
                budget=budget,
            ),
            budget=budget,
        )
        for a, b, c in zip(apex, base[0], base[1], strict=True)
    )
    return (
        expression_compose(gram, arguments, budget=budget),
        expression_multiply(
            expression_compose(weight, arguments, budget=budget), collapse, budget=budget
        ),
        "quadrilateral",
    )


@checked
def certified_sqrt_polynomial_integral_state(
    gram: Expression,
    weight: Expression,
    kind: str,
    absolute_tolerance: imported_fractions.Fraction | float,
    relative_tolerance: float,
    maximum_subcells: int,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:

    gram = _rational_expression(gram)
    weight = _rational_expression(weight)
    if (
        kind == "interval"
        or isinstance(gram, RationalPolynomial)
        or isinstance(weight, RationalPolynomial)
    ):
        gram, weight, kind = _certified_sqrt_cone_transform(
            gram, weight, kind, budget=budget
        )
        domain = "simplex" if kind == "triangle" else "box"
        coefficients = _rational_expression_bernstein_coefficients(
            gram, domain, 2, budget=budget
        )
        weight_coefficients = _rational_expression_bernstein_coefficients(
            weight, domain, 2, budget=budget
        )
        upper = max(coefficients)
        if upper <= 0:
            raise ValueError(
                "embedded_measure.gram must have a positive upper measure bound; got nonpositive Gram measure"
            )
        volume = (
            imported_fractions.Fraction(1, 2)
            if kind == "triangle"
            else imported_fractions.Fraction(1)
        )
        characteristic = (
            fraction_sqrt_interval(upper, budget=budget)[1]
            * max(abs(x) for x in weight_coefficients)
            * volume
        )
        absolute, relative = (
            imported_fractions.Fraction(absolute_tolerance),
            imported_fractions.Fraction(relative_tolerance),
        )
        tolerance = (
            max(absolute, characteristic / 8)
            if relative * characteristic > absolute
            else absolute
        )
        subcells = [0, int(maximum_subcells)]
        while True:
            value, error = _certified_sqrt_expression_pass(
                gram,
                weight,
                kind,
                tolerance,
                maximum_terms,
                work,
                subcells,
                budget=budget,
            )
            lower_magnitude = max(
                abs(imported_fractions.Fraction(value))
                - imported_fractions.Fraction(error),
                imported_fractions.Fraction(0),
            )
            allowed = max(absolute, relative * lower_magnitude)
            if imported_fractions.Fraction(error) <= allowed:
                return value, error
            tolerance = (
                max(absolute, allowed)
                if lower_magnitude
                else max(absolute, tolerance / 4)
            )
    domain = "simplex" if kind == "triangle" else "box"
    gram_coefficients = _rational_bernstein_coefficients(gram, domain, 2, budget=budget)
    weight_coefficients = _rational_bernstein_coefficients(
        weight, domain, 2, budget=budget
    )
    upper = max(gram_coefficients)
    if upper <= 0:
        raise ValueError(
            "embedded_measure.gram must have a positive upper measure bound; got nonpositive Gram measure"
        )
    weight_sup = max(abs(value) for value in weight_coefficients)
    volume = (
        imported_fractions.Fraction(1, 2)
        if kind == "triangle"
        else imported_fractions.Fraction(1)
    )
    characteristic = fraction_sqrt_interval(upper, budget=budget)[1] * weight_sup * volume
    lower_integral = imported_fractions.Fraction(0)
    lower = min(gram_coefficients)
    if lower > 0 and (min(weight_coefficients) >= 0 or max(weight_coefficients) <= 0):
        lower_integral = fraction_sqrt_interval(lower, budget=budget)[0] * abs(
            rational_coefficient(
                mapped_polynomial_integral_fraction(weight, kind, budget=budget),
                role="mapped_geometry.rational_integration.coefficients",
            )
        )
    absolute, relative = (
        imported_fractions.Fraction(absolute_tolerance),
        imported_fractions.Fraction(relative_tolerance),
    )
    if relative * characteristic <= absolute:
        tolerance = absolute
    elif lower_integral:
        tolerance = max(absolute, relative * lower_integral)
    else:
        tolerance = max(absolute, characteristic / 8)
    subcells = [0, int(maximum_subcells)]
    while True:
        value, error = _sqrt_weighted_integral_pass(
            gram,
            weight,
            kind,
            tolerance,
            maximum_terms,
            work,
            subcells,
            budget=budget,
        )
        lower_magnitude = max(
            abs(imported_fractions.Fraction(value)) - imported_fractions.Fraction(error),
            imported_fractions.Fraction(0),
        )
        allowed = max(absolute, relative * lower_magnitude)
        if imported_fractions.Fraction(error) <= allowed:
            return value, error
        tolerance = (
            max(absolute, relative * lower_magnitude)
            if lower_magnitude
            else max(absolute, tolerance / 4)
        )


@checked
def certified_sqrt_polynomial_integral(
    gram: Expression,
    weight: Expression,
    kind: str,
    /,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-10,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    """Enclose a signed algebraic source weight times sqrt(Gram).

    Polynomial reduction order is unchanged. Rational expressions use a positive
    Bernstein denominator proof and polynomial reciprocal/binomial moments with
    absolute remainder bounds. Relative acceptance uses the proved magnitude of
    the whole integral, including cancellation; all passes share resource limits.
    """

    gram = _rational_expression(gram)
    weight = _rational_expression(weight)
    absolute = nonnegative_finite_float(absolute_tolerance, "absolute_tolerance")
    relative = nonnegative_finite_float(relative_tolerance, "relative_tolerance")
    if kind not in ("interval", "triangle", "quadrilateral") or absolute == relative == 0:
        raise ValueError(
            "embedded_measure.error_budget must be positive on supported interval, triangle, or quadrilateral cells; got unsupported kind or nonpositive tolerance"
        )
    if min(maximum_work, maximum_subcells, maximum_binomial_terms) <= 0:
        raise ValueError(
            "embedded_measure.work_limits must be positive; got nonpositive work limit"
        )
    if (
        kind != "interval"
        and not isinstance(gram, RationalPolynomial)
        and not isinstance(weight, RationalPolynomial)
    ):
        if any(len(index) != 2 for polynomial in (gram, weight) for index in polynomial):
            raise ValueError(
                "embedded_measure.polynomial must have two reference dimensions; got incompatible polynomial indices"
            )
        rational_gram: Polynomial = {
            index: rational_coefficient(
                value, role="mapped_geometry.rational_integration.coefficients"
            )
            for index, value in gram.items()
            if value
        }
        rational_weight: Polynomial = {
            index: rational_coefficient(
                value, role="mapped_geometry.rational_integration.coefficients"
            )
            for index, value in weight.items()
            if value
        }
        return certified_sqrt_polynomial_integral_state(
            rational_gram,
            rational_weight,
            kind,
            absolute,
            relative,
            int(maximum_subcells),
            int(maximum_binomial_terms),
            [0, int(maximum_work)],
            budget=budget,
        )

    dimension = 1 if kind == "interval" else 2
    expressions: list[Expression] = []
    for value in (gram, weight):
        numerator, denominator = imported_coordinate_enclosure.expression_parts(
            value, dimension, budget=budget
        )
        if any(
            len(index) != dimension
            for polynomial in (numerator, denominator)
            for index in polynomial
        ):
            raise ValueError(
                "embedded_measure.expressions must share the reference dimension; got incompatible expression indices"
            )
        rational_numerator: Polynomial = {
            index: rational_coefficient(
                coefficient, role="mapped_geometry.rational_integration.coefficients"
            )
            for index, coefficient in numerator.items()
            if coefficient
        }
        rational_denominator: Polynomial = {
            index: rational_coefficient(
                coefficient, role="mapped_geometry.rational_integration.coefficients"
            )
            for index, coefficient in denominator.items()
            if coefficient
        }
        if kind == "interval":
            lifted_numerator: Polynomial = {
                (index[0], 0): coefficient
                for index, coefficient in rational_numerator.items()
            }
            lifted_denominator: Polynomial = {
                (index[0], 0): coefficient
                for index, coefficient in rational_denominator.items()
            }
            rational_numerator, rational_denominator = (
                lifted_numerator,
                lifted_denominator,
            )
        expressions.append(
            imported_coordinate_enclosure.rational_expression(
                rational_numerator, rational_denominator, budget=budget
            )
            if isinstance(value, RationalPolynomial)
            else rational_numerator
        )
    gram, weight = expressions
    return certified_sqrt_polynomial_integral_state(
        gram,
        weight,
        kind,
        absolute,
        relative,
        int(maximum_subcells),
        int(maximum_binomial_terms),
        [0, int(maximum_work)],
        budget=budget,
    )


class MappedEdgeArcLengthEnclosure(NamedTuple):
    value: float
    error: float
    lower: float
    upper: float


class PreparedMappedEdgeArcLength(NamedTuple):
    gram: Expression

    @checked
    def integrate(
        self,
        /,
        *,
        absolute_tolerance: float = 1e-10,
        relative_tolerance: float = 1e-10,
        maximum_work: int = 100_000_000,
        maximum_subcells: int = 10000,
        maximum_binomial_terms: int = 32,
        budget: CoordinateEnclosureBudget,
    ) -> MappedEdgeArcLengthEnclosure:

        value, error = certified_sqrt_polynomial_integral(
            self.gram,
            imported_coordinate_enclosure.constant(1, 1, budget=budget),
            "interval",
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_work=maximum_work,
            maximum_subcells=maximum_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
            budget=budget,
        )
        lower = (
            value
            if error == 0
            else fraction_endpoint(
                Fraction(value) - Fraction(error), upper=False, saturate=True
            )
        )
        upper = (
            value
            if error == 0
            else fraction_endpoint(
                Fraction(value) + Fraction(error), upper=True, saturate=True
            )
        )
        if lower <= 0:
            raise ValueError(
                "mapped_edge.length must have a strictly positive enclosure; got nonpositive length bound"
            )
        return MappedEdgeArcLengthEnclosure(value, error, lower, upper)


@checked
def prepare_mapped_edge_arc_length(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    start: tuple[Fraction, ...],
    end: tuple[Fraction, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PreparedMappedEdgeArcLength:
    """Restrict the coordinate expressions to the full edge."""

    element = require_scalar_coordinate_element(element, "Mapped edge arc length")
    dimension = element.topological_dimension
    if len(start) != dimension or len(end) != dimension or start == end:
        raise ValueError(
            "mapped_edge.endpoints must be distinct points of the source reference dimension; got incompatible endpoints"
        )
    variable = imported_coordinate_enclosure.axes(1, budget=budget)[0]
    arguments = tuple(
        add(
            imported_coordinate_enclosure.constant(Fraction(a), 1, budget=budget),
            scale(variable, Fraction(b) - Fraction(a), budget=budget),
            budget=budget,
        )
        for a, b in zip(start, end, strict=True)
    )
    edge = tuple(
        expression_compose(value, arguments, budget=budget)
        for value in mapped_coordinate_expressions(element, local, budget=budget)
    )
    velocity = tuple(expression_derivative(value, 0, budget=budget) for value in edge)
    gram = imported_coordinate_enclosure.expression_sum(
        tuple(expression_multiply(value, value, budget=budget) for value in velocity),
        budget=budget,
    )
    return PreparedMappedEdgeArcLength(gram)


def _certified_sqrt_dyadic_interval(
    value: Fraction, work: list[int], *, budget: CoordinateEnclosureBudget
) -> tuple[Fraction, Fraction]:
    """Prove a 128-bit dyadic radical enclosure with integer arithmetic."""
    if value <= 0:
        raise ValueError(
            "embedded_measure.squared_scale must be positive; got nonpositive radical scale"
        )

    proposed = work[0] + 1
    if proposed > work[1]:
        raise transition_types.CellGeometryTransitionError(
            "resource_limit",
            "embedded_measure.work_units must fit the radical work budget; got excess proposed work",
            measured=proposed,
            limit=work[1],
        )
    ledger = budget
    bits = value.numerator.bit_length() + value.denominator.bit_length() + 512
    ledger.reserve(1, 256 + bits)
    work[0] = proposed
    denominator = 1 << 128
    scaled = value.numerator << 256
    numerator = math.isqrt(scaled // value.denominator)
    lower = Fraction(numerator, denominator)
    return (
        lower,
        lower
        if numerator * numerator * value.denominator == scaled
        else Fraction(numerator + 1, denominator),
    )


def _certified_sqrt_exact_product(
    first: Polynomial,
    second: Polynomial,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:

    proposed = work[0] + len(first) * len(second)
    if proposed > work[1]:
        raise transition_types.CellGeometryTransitionError(
            "resource_limit",
            "embedded_measure.work_units must fit the polynomial-product budget; got excess proposed work",
            measured=proposed,
            limit=work[1],
        )
    work[0] = proposed
    return imported_coordinate_enclosure.multiply(first, second, budget=budget)


def _certified_sqrt_denominator_power(
    divisor: Polynomial,
    domain: str,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, int]:

    exponent = 1
    while len(divisor) > 1:
        base = _certified_sqrt_exact_polynomial_root(divisor, work, budget=budget)
        if base is None:
            break
        controls = _rational_bernstein_coefficients(base, domain, 2, budget=budget)
        if max(controls) < 0:
            base = scale(base, -1, budget=budget)
        elif min(controls) <= 0:
            break
        divisor, exponent = base, 2 * exponent
    return divisor, exponent


def _certified_sqrt_exact_reciprocal(
    base: Polynomial,
    exponent: int,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, Fraction] | None:

    if exponent == 0:
        return imported_coordinate_enclosure.constant(1, 2, budget=budget), Fraction(0)
    domain = "simplex" if kind == "triangle" else "box"
    controls = _rational_bernstein_coefficients(base, domain, 2, budget=budget)
    lower, upper = min(controls), max(controls)
    if lower <= 0:
        raise ValueError(
            "embedded_measure.denominator must have a positive lower bound; got nonpositive reciprocal denominator"
        )
    center = (lower + upper) / 2
    rho = (upper - lower) / (upper + lower)
    if rho > Fraction(1, 8):
        return None
    ratio = add(
        imported_coordinate_enclosure.constant(1, 2, budget=budget),
        scale(base, -1 / center, budget=budget),
        budget=budget,
    )
    series: Polynomial = {}
    term = imported_coordinate_enclosure.constant(1, 2, budget=budget)
    coefficient = Fraction(1)
    for order in range(maximum_terms):
        series = add(
            series,
            scale(term, coefficient / center**exponent, budget=budget),
            budget=budget,
        )
        next_coefficient = coefficient * Fraction(exponent + order, order + 1)
        tail_ratio = rho * Fraction(exponent + order + 1, order + 2)
        if tail_ratio < 1:
            error = (
                next_coefficient
                * rho ** (order + 1)
                / (center**exponent * (1 - tail_ratio))
            )
            if error <= tolerance:
                return series, error
        if order + 1 < maximum_terms:
            term = _certified_sqrt_exact_product(term, ratio, work, budget=budget)
            coefficient = next_coefficient
    return None


def _certified_sqrt_exact_density_piece(
    gram: Expression,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, Fraction] | None:
    """algebraic polynomial approximation of a positive rational radical, with a uniform tail."""

    domain = "simplex" if kind == "triangle" else "box"
    numerator, divisor = imported_coordinate_enclosure.expression_parts(
        gram, 2, budget=budget
    )
    if min(_rational_bernstein_coefficients(divisor, domain, 2, budget=budget)) <= 0:
        raise ValueError(
            "embedded_measure.denominator must have a positive lower bound; got nonpositive approximation denominator"
        )
    root = _certified_sqrt_exact_polynomial_root(divisor, work, budget=budget)
    if (
        root is not None
        and len(root) == 1
        and all(index == 0 for index in next(iter(root)))
    ):
        numerator_root = _certified_sqrt_exact_polynomial_root(
            numerator, work, budget=budget
        )
        if numerator_root:
            root_controls = _rational_bernstein_coefficients(
                numerator_root, domain, 2, budget=budget
            )
            sign = 1 if min(root_controls) >= 0 else -1 if max(root_controls) <= 0 else 0
            if sign:
                # A sign-certified polynomial square needs no binomial expansion.
                return (
                    scale(
                        numerator_root,
                        Fraction(sign) / next(iter(root.values())),
                        budget=budget,
                    ),
                    Fraction(0),
                )
    if root is None:
        numerator = _certified_sqrt_exact_product(numerator, divisor, work, budget=budget)
    else:
        root_controls = _rational_bernstein_coefficients(root, domain, 2, budget=budget)
        if max(root_controls) < 0:
            root = scale(root, -1, budget=budget)
        elif min(root_controls) <= 0:
            return None
        divisor = root
    base, exponent = _certified_sqrt_denominator_power(
        divisor, domain, work, budget=budget
    )
    controls = _rational_bernstein_coefficients(base, domain, 2, budget=budget)
    base_lower, base_upper = min(controls), max(controls)
    if base_lower <= 0 or (base_upper - base_lower) / (
        base_upper + base_lower
    ) > Fraction(1, 8):
        return None
    square = _certified_sqrt_exact_product(base, base, work, budget=budget)
    leading = max(numerator)
    factor = rational_coefficient(
        numerator[leading] / square[leading] if leading == max(square) else Fraction(0),
        role="mapped_geometry.rational_integration.coefficients",
    )
    residual = add(numerator, scale(square, -factor, budget=budget), budget=budget)
    rho = (
        max(
            abs(value)
            for value in _rational_bernstein_coefficients(
                residual, domain, 2, budget=budget
            )
        )
        / (factor * base_lower**2)
        if factor > 0
        else Fraction(1)
    )
    if rho <= Fraction(1, 8):
        ratio = scale(residual, 1 / factor, budget=budget)
        initial_exponent, exponent_step = exponent - 1, 2
    else:
        values = _rational_bernstein_coefficients(numerator, domain, 2, budget=budget)
        lower, upper = min(values), max(values)
        if lower <= 0:
            return None
        factor = (lower + upper) / 2
        rho = (upper - lower) / (upper + lower)
        if rho > Fraction(1, 8):
            return None
        ratio = add(
            scale(numerator, 1 / factor, budget=budget),
            imported_coordinate_enclosure.constant(-1, 2, budget=budget),
            budget=budget,
        )
        initial_exponent, exponent_step = exponent, 0
    root_lower, root_upper = _certified_sqrt_dyadic_interval(factor, work, budget=budget)
    root_midpoint, root_error = (
        (root_lower + root_upper) / 2,
        (root_upper - root_lower) / 2,
    )
    series: Polynomial = {}
    power = imported_coordinate_enclosure.constant(1, 2, budget=budget)
    coefficient, reciprocal_error = Fraction(1), Fraction(0)
    for order in range(maximum_terms):
        power_bound = max(
            abs(value)
            for value in _rational_bernstein_coefficients(power, domain, 2, budget=budget)
        )
        reciprocal_goal = (
            tolerance / (8 * maximum_terms * root_upper * abs(coefficient) * power_bound)
            if power_bound
            else tolerance
        )
        reciprocal = _certified_sqrt_exact_reciprocal(
            base,
            initial_exponent + exponent_step * order,
            kind,
            reciprocal_goal,
            maximum_terms,
            work,
            budget=budget,
        )
        if reciprocal is None:
            return None
        polynomial, error = reciprocal
        series = add(
            series,
            scale(
                _certified_sqrt_exact_product(power, polynomial, work, budget=budget),
                coefficient,
                budget=budget,
            ),
            budget=budget,
        )
        reciprocal_error += abs(coefficient) * power_bound * error
        next_coefficient = coefficient * (Fraction(1, 2) - order) / (order + 1)
        tail = (
            abs(next_coefficient)
            * rho ** (order + 1)
            / ((1 - rho) * base_lower**initial_exponent)
        )
        series_bound = (
            1 + rho / (2 * (1 - rho))
        ) / base_lower**initial_exponent + reciprocal_error
        error = root_upper * (reciprocal_error + tail) + root_error * series_bound
        if error <= tolerance / 2:
            return scale(series, root_midpoint, budget=budget), error
        if order + 1 < maximum_terms:
            power = _certified_sqrt_exact_product(power, ratio, work, budget=budget)
            coefficient = next_coefficient
    return None


class _PreparedSqrtIntegralPiece(NamedTuple):
    arguments: tuple[Polynomial, ...]
    polynomial: Polynomial
    error: Fraction
    fraction: Fraction
    kind: str
    moments: dict[tuple[int, ...], Fraction]

    def moment(
        self,
        index: tuple[int, ...],
        work: list[int],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> Fraction:
        if index in self.moments:
            return self.moments[index]

        proposed = work[0] + len(self.polynomial)
        if proposed > work[1]:
            raise transition_types.CellGeometryTransitionError(
                "resource_limit",
                "embedded_measure.work_units must fit the moment work budget; got excess proposed work",
                measured=proposed,
                limit=work[1],
            )
        numerator_bits, denominator_bits = (
            imported_coordinate_enclosure.coefficient_profile(
                (self.polynomial,), budget=budget
            )
        )
        degree = (
            sum(
                max((term[axis] for term in self.polynomial), default=0) + index[axis]
                for axis in range(2)
            )
            + 2
        )
        reserve_polynomial(
            len(self.polynomial),
            1,
            2,
            numerator_bits
            + denominator_bits
            + degree * max(degree, 1).bit_length()
            + len(self.polynomial).bit_length(),
            budget=budget,
        )
        work[0] = proposed
        total = Fraction(0)
        for exponent, coefficient in self.polynomial.items():
            coefficient = rational_coefficient(
                coefficient, role="mapped_geometry.rational_integration.coefficients"
            )
            first, second = exponent[0] + index[0], exponent[1] + index[1]
            moment = (
                Fraction(
                    math.factorial(first) * math.factorial(second),
                    math.factorial(first + second + 2),
                )
                if self.kind == "triangle"
                else Fraction(1, (first + 1) * (second + 1))
            )
            total += coefficient * moment
        self.moments[index] = total
        return total


class PreparedMappedSqrtIntegral(NamedTuple):
    pieces: tuple[_PreparedSqrtIntegralPiece, ...]
    work: list[int]

    def integral(
        self, weight: Polynomial, *, budget: CoordinateEnclosureBudget
    ) -> tuple[float, float]:

        weight = _rational_polynomial(weight)
        value, error = Fraction(0), Fraction(0)
        with budget.activate():
            for piece in self.pieces:
                local_weight = compose(weight, piece.arguments, budget=budget)
                coefficients = _rational_bernstein_coefficients(
                    local_weight,
                    "simplex" if piece.kind == "triangle" else "box",
                    2,
                    budget=budget,
                )
                central = sum(
                    (
                        rational_coefficient(
                            coefficient,
                            role="mapped_geometry.rational_integration.coefficients",
                        )
                        * piece.moment(index, self.work, budget=budget)
                        for index, coefficient in local_weight.items()
                    ),
                    Fraction(0),
                )
                volume = Fraction(1, 2) if piece.kind == "triangle" else Fraction(1)
                value += piece.fraction * central
                error += (
                    piece.fraction
                    * volume
                    * piece.error
                    * max(abs(coefficient) for coefficient in coefficients)
                )
        published = float(value)
        error += abs(Fraction(published) - value)
        return published, fraction_endpoint(
            error, upper=True, saturate=True
        ) if error else 0.0


def _certified_sqrt_exact_weighted_density_piece(
    gram: Expression,
    denominator: Polynomial,
    exponent: int,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, Fraction] | None:
    """Keep a proved positive reciprocal factor independent of the Gram radical."""
    if exponent == 0:
        return _certified_sqrt_exact_density_piece(
            gram,
            kind,
            tolerance,
            maximum_terms,
            work,
            budget=budget,
        )

    domain = "simplex" if kind == "triangle" else "box"
    controls = _rational_bernstein_coefficients(denominator, domain, 2, budget=budget)
    lower, upper = min(controls), max(controls)
    if lower <= 0:
        raise ValueError(
            "embedded_measure.denominator must have a positive lower bound; got nonpositive weighted denominator"
        )
    if (upper - lower) / (upper + lower) > Fraction(1, 8):
        return None
    density = _certified_sqrt_exact_density_piece(
        gram,
        kind,
        tolerance * lower**exponent / 4,
        maximum_terms,
        work,
        budget=budget,
    )
    if density is None:
        return None
    polynomial, density_error = density
    magnitude = max(
        abs(value)
        for value in _rational_bernstein_coefficients(
            polynomial, domain, 2, budget=budget
        )
    )
    reciprocal = _certified_sqrt_exact_reciprocal(
        denominator,
        exponent,
        kind,
        tolerance / (4 * max(magnitude, Fraction(1))),
        maximum_terms,
        work,
        budget=budget,
    )
    if reciprocal is None:
        return None
    factor, reciprocal_error = reciprocal
    result = _certified_sqrt_exact_product(polynomial, factor, work, budget=budget)
    return result, density_error / lower**exponent + magnitude * reciprocal_error


@checked
def certified_sqrt_prepare_expression_integral(
    gram: Expression,
    kind: str,
    uniform_tolerance: Fraction,
    work: list[int],
    maximum_subcells: int = 10000,
    maximum_terms: int = 32,
    *,
    jacobian: Polynomial | None = None,
    arguments: tuple[Polynomial, ...] | None = None,
    denominator: Polynomial | None = None,
    denominator_exponent: int = 0,
    budget: CoordinateEnclosureBudget,
) -> PreparedMappedSqrtIntegral:
    """Prepare algebraic moment functionals for an already-owned reference density."""

    gram = _rational_expression(gram)
    if jacobian is not None:
        jacobian = _rational_polynomial(jacobian)
    if denominator is not None:
        denominator = _rational_polynomial(denominator)
    ledger = budget

    if uniform_tolerance <= 0 or maximum_subcells <= 0 or maximum_terms <= 0:
        raise ValueError(
            "embedded_measure.limits must give positive precision and resources; got nonpositive limit"
        )
    pieces: list[_PreparedSqrtIntegralPiece] = []
    with ledger.activate():
        pending = [
            (
                gram,
                imported_coordinate_enclosure.constant(1, 2, budget=budget)
                if jacobian is None
                else jacobian,
                imported_coordinate_enclosure.axes(2, budget=budget)
                if arguments is None
                else arguments,
                Fraction(1),
                imported_coordinate_enclosure.constant(1, 2, budget=budget)
                if denominator is None
                else denominator,
            )
        ]
        visited = 0
        while pending:
            local_gram, local_jacobian, local_arguments, fraction, local_denominator = (
                pending.pop()
            )
            visited += 1
            if visited > maximum_subcells:
                raise transition_types.CellGeometryTransitionError(
                    "resource_limit",
                    "embedded_measure.subcells must fit the reference-subdivision budget; got excess visited pieces",
                    measured=visited,
                    limit=maximum_subcells,
                )
            domain = "simplex" if kind == "triangle" else "box"
            jacobian_sup = max(
                abs(value)
                for value in _rational_bernstein_coefficients(
                    local_jacobian, domain, 2, budget=budget
                )
            )
            with ledger.temporary_scope():
                approximation = _certified_sqrt_exact_weighted_density_piece(
                    local_gram,
                    local_denominator,
                    denominator_exponent,
                    kind,
                    uniform_tolerance / max(jacobian_sup, Fraction(1)),
                    maximum_terms,
                    work,
                    budget=budget,
                )
                if approximation is not None:
                    polynomial, error = approximation
                    polynomial = _certified_sqrt_exact_product(
                        polynomial,
                        local_jacobian,
                        work,
                        budget=budget,
                    )
            if approximation is not None:
                _certified_sqrt_retain_expressions((polynomial,), budget=budget)
                pieces.append(
                    _PreparedSqrtIntegralPiece(
                        local_arguments,
                        polynomial,
                        error * jacobian_sup,
                        fraction,
                        kind,
                        {},
                    )
                )
            else:
                for matrix, offset in embedded_measure_submaps(kind):
                    child_arguments = affine_arguments(offset, matrix, budget=budget)
                    child_gram, child_jacobian = _certified_sqrt_expression_child(
                        local_gram,
                        local_jacobian,
                        child_arguments,
                        budget=ledger,
                    )
                    if isinstance(child_jacobian, RationalPolynomial):
                        raise ValueError(
                            "embedded_measure.reference_jacobian must be polynomial; got rational expression"
                        )
                    mapped_arguments = tuple(
                        compose(value, child_arguments, budget=budget)
                        for value in local_arguments
                    )
                    child_denominator = (
                        compose(local_denominator, child_arguments, budget=budget)
                        if denominator_exponent
                        else local_denominator
                    )
                    pending.append(
                        (
                            child_gram,
                            child_jacobian,
                            mapped_arguments,
                            fraction / 4,
                            child_denominator,
                        )
                    )
    return PreparedMappedSqrtIntegral(tuple(pieces), work)


@checked
def certified_sqrt_prepare_mapped_integral(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    uniform_tolerance: Fraction,
    work: list[int],
    maximum_subcells: int = 10000,
    maximum_terms: int = 32,
    *,
    budget: CoordinateEnclosureBudget,
) -> PreparedMappedSqrtIntegral:
    """Prepare one density for many algebraic weighted physical moment functionals."""

    ledger = budget

    with ledger.activate():
        gram = embedded_squared_density(element, local, budget=budget)
        gram = _rational_expression(gram)
        jacobian = imported_coordinate_enclosure.constant(1, 2, budget=budget)
        if isinstance(gram, RationalPolynomial) and isinstance(
            element,
            (
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
            ),
        ):
            root, arguments = imported_coordinate_enclosure.coordinate_reference_chain(
                element, budget=budget
            )
            # Rational reference pullbacks remain in the Gram; extracting
            # their Jacobian into the polynomial moment path would lose its denominator.
            polynomial_arguments = tuple(
                value for value in arguments if not isinstance(value, RationalPolynomial)
            )
            if root.topological_dimension == 2 and len(polynomial_arguments) == len(
                arguments
            ):
                jacobian = imported_coordinate_enclosure.determinant(
                    tuple(
                        tuple(
                            imported_coordinate_enclosure.derivative(
                                value, axis, budget=budget
                            )
                            for axis in range(2)
                        )
                        for value in polynomial_arguments
                    ),
                    budget=budget,
                )
                controls = _rational_bernstein_coefficients(
                    jacobian,
                    "simplex" if element.cell_kind == "triangle" else "box",
                    2,
                    budget=budget,
                )
                if max(controls) < 0:
                    jacobian = scale(jacobian, -1, budget=budget)
                elif min(controls) <= 0:
                    raise ValueError(
                        "embedded_measure.reference_jacobian must have a positive lower bound; got nonpositive reference-map bound"
                    )
                gram = expression_compose(
                    embedded_squared_density(root, local, budget=budget),
                    arguments,
                    budget=budget,
                )
        kind = element.cell_kind
        arguments = imported_coordinate_enclosure.axes(2, budget=budget)
        if kind == "triangle" and any(
            isinstance(value, RationalPolynomial) for value in (gram, jacobian)
        ):
            vertices = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
            singular = {
                vertex
                for value in (gram, jacobian)
                if isinstance(value, RationalPolynomial)
                for vertex in vertices
                if not imported_coordinate_enclosure.evaluate(
                    value.denominator, vertex, budget=budget
                )
            }
            if singular:
                apex = next(iter(singular))
                base = tuple(vertex for vertex in vertices if vertex != apex)
                u, collapse = imported_coordinate_enclosure.axes(2, budget=budget)
                arguments = tuple(
                    add(
                        imported_coordinate_enclosure.constant(a, 2, budget=budget),
                        imported_coordinate_enclosure.multiply(
                            collapse,
                            add(
                                imported_coordinate_enclosure.constant(
                                    b - a, 2, budget=budget
                                ),
                                scale(u, c - b, budget=budget),
                                budget=budget,
                            ),
                            budget=budget,
                        ),
                        budget=budget,
                    )
                    for a, b, c in zip(apex, base[0], base[1], strict=True)
                )
                gram, jacobian, kind = _certified_sqrt_cone_transform(
                    gram,
                    jacobian,
                    kind,
                    budget=budget,
                )
                if isinstance(jacobian, RationalPolynomial):
                    raise ValueError(
                        "embedded_measure.reference_jacobian must be polynomial after the cone chart; got rational expression"
                    )
        return certified_sqrt_prepare_expression_integral(
            gram,
            kind,
            uniform_tolerance,
            work,
            maximum_subcells,
            maximum_terms,
            jacobian=jacobian,
            arguments=arguments,
            budget=budget,
        )


@checked
def certified_sqrt_mapped_integral_state(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    weight: Expression,
    absolute_tolerance: Fraction | float,
    relative_tolerance: float,
    maximum_subcells: int,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    """Keep algebraic pullback determinants outside the root of a rational Gram."""

    ledger = budget

    with ledger.temporary_scope():
        gram = embedded_squared_density(element, local, budget=budget)
        gram = _rational_expression(gram)
        weight = _rational_expression(weight)
        if isinstance(gram, RationalPolynomial) and isinstance(
            element,
            (
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
            ),
        ):
            root, arguments = imported_coordinate_enclosure.coordinate_reference_chain(
                element, budget=budget
            )
            # Keep a rational reference Jacobian inside the Gram expression.
            polynomial_arguments = tuple(
                value for value in arguments if not isinstance(value, RationalPolynomial)
            )
            if root.topological_dimension == 2 and len(polynomial_arguments) == len(
                arguments
            ):
                jacobian = imported_coordinate_enclosure.determinant(
                    tuple(
                        tuple(
                            imported_coordinate_enclosure.derivative(
                                value, axis, budget=budget
                            )
                            for axis in range(2)
                        )
                        for value in polynomial_arguments
                    ),
                    budget=budget,
                )
                domain = "simplex" if element.cell_kind == "triangle" else "box"
                controls = _rational_bernstein_coefficients(
                    jacobian, domain, 2, budget=budget
                )
                if max(controls) < 0:
                    jacobian = scale(jacobian, -1, budget=budget)
                elif min(controls) <= 0:
                    raise ValueError(
                        "embedded_measure.reference_jacobian must be sign-definite; got zero-crossing Jacobian enclosure"
                    )
                gram = imported_coordinate_enclosure.expression_compose(
                    embedded_squared_density(root, local, budget=budget),
                    arguments,
                    budget=budget,
                )
                weight = imported_coordinate_enclosure.expression_multiply(
                    weight, jacobian, budget=budget
                )
        return certified_sqrt_polynomial_integral_state(
            gram,
            weight,
            element.cell_kind,
            absolute_tolerance,
            relative_tolerance,
            maximum_subcells,
            maximum_terms,
            work,
            budget=budget,
        )


def _certified_embedded_cell_measure(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    absolute_tolerance: Fraction,
    relative_tolerance: float,
    maximum_subcells: int,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:

    element = require_scalar_coordinate_element(element, "Certified embedded measure")
    return certified_sqrt_mapped_integral_state(
        element,
        local,
        imported_coordinate_enclosure.constant(1, 2, budget=budget),
        absolute_tolerance,
        relative_tolerance,
        maximum_subcells,
        maximum_terms,
        work,
        budget=budget,
    )


@checked
def reuse_embedded_polynomial_area(
    gram: Polynomial,
    kind: str,
    references: list[tuple[Polynomial, Fraction, float, float]],
    absolute: Fraction,
    relative: float,
    maximum_subcells: int,
    maximum_terms: int,
    work: list[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[float, float], tuple[Polynomial, Fraction, float, float] | None]:
    """Reuse an area enclosure only after proving the Gram difference."""

    gram = _rational_polynomial(gram)
    if all(not any(index) for index in gram):
        # Constant Grams have their own direct certified square-root integral.
        # Comparing distinct scalar areas against every earlier cell is quadratic
        # work and cannot improve that proof.
        return certified_sqrt_polynomial_integral_state(
            gram,
            imported_coordinate_enclosure.constant(1, 2, budget=budget),
            kind,
            absolute,
            relative,
            maximum_subcells,
            maximum_terms,
            work,
            budget=budget,
        ), None
    domain = "simplex" if kind == "triangle" else "box"
    volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
    for reference, lower, value, error in reversed(references):
        if gram == reference:
            allowed = max(
                absolute,
                Fraction(relative) * max(Fraction(value) - Fraction(error), Fraction(0)),
            )
            if Fraction(error) <= allowed:
                return (value, error), None
        difference = add(gram, scale(reference, -1, budget=budget), budget=budget)
        controls = _rational_bernstein_coefficients(difference, domain, 2, budget=budget)
        delta = max(abs(coefficient) for coefficient in controls)
        if delta >= lower:
            continue
        denominator = (
            fraction_sqrt_interval(lower, budget=budget)[0]
            + fraction_sqrt_interval(lower - delta, budget=budget)[0]
        )
        if denominator <= 0:
            continue
        # Nonnegative Bernstein bases have equal reference integrals. Their
        # absolute-control envelope encloses integral(abs(Gram delta)).
        delta_integral = (
            volume
            * sum((abs(coefficient) for coefficient in controls), Fraction(0))
            / len(controls)
        )
        bound = Fraction(error) + delta_integral / denominator
        published = fraction_endpoint(bound, upper=True, saturate=True) if bound else 0.0
        allowed = max(
            absolute,
            Fraction(relative) * max(Fraction(value) - Fraction(published), Fraction(0)),
        )
        if Fraction(published) <= allowed:
            return (value, published), None
    value, error = certified_sqrt_polynomial_integral_state(
        gram,
        imported_coordinate_enclosure.constant(1, 2, budget=budget),
        kind,
        absolute,
        relative,
        maximum_subcells,
        maximum_terms,
        work,
        budget=budget,
    )
    lower = min(_rational_bernstein_coefficients(gram, domain, 2, budget=budget))
    return (value, error), ((gram, lower, value, error) if lower > 0 else None)


def _exact_affine_simplex_measure_from_corners(
    corners: tuple[tuple[imported_coordinate_enclosure.ExactScalar, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float, bool]:
    """algebraic affine simplex corners to a bounded binary64 measure."""

    dimension = len(corners) - 1
    ambient = len(corners[0])
    embedded_triangle = dimension == 2 and ambient == 3
    if dimension not in (1, 2, 3) or any(len(point) != ambient for point in corners):
        raise ValueError(
            "affine_measure.corners must align in reference and ambient dimensions; got incompatible corner bank"
        )
    if ambient != dimension and not embedded_triangle:
        raise ValueError(
            "affine_measure.dimension must be full-dimensional or an embedded triangle; got unsupported dimensions"
        )
    ledger = budget

    ledger.reserve(sum(len(point) for point in corners))
    bits = max(
        abs(value.numerator).bit_length() + value.denominator.bit_length()
        if isinstance(value, Fraction)
        else 2 * value.coefficient_bits
        for point in corners
        for value in point
    )
    if embedded_triangle:
        squared = _embedded_affine_triangle_squared(corners, budget=budget)
        if squared <= 0:
            raise ValueError(
                "mapped_geometry.gram must have a positive lower bound; got nonpositive embedded Jacobian"
            )
        lower_root, upper_root = fraction_sqrt_interval(
            rational_coefficient(
                squared, role="mapped_geometry.rational_integration.coefficients"
            ),
            budget=budget,
        )
        lower_measure, upper_measure = lower_root / 2, upper_root / 2
        value = math.sqrt(float(squared)) / 2
        return (
            value,
            max(
                value - fraction_endpoint(lower_measure, upper=False, saturate=True),
                fraction_endpoint(upper_measure, upper=True, saturate=True) - value,
            ),
            lower_root == upper_root,
        )
    match dimension:
        case 1:
            arithmetic, terms = 1, 1
        case 2:
            arithmetic, terms = 7, 2
        case 3:
            arithmetic, terms = 26, 6
        case _:
            raise ValueError(
                "affine_measure.dimension must be one through three; got unsupported dimension"
            )
    reserve_polynomial(
        arithmetic,
        dimension * dimension + terms,
        0,
        dimension * (2 * bits + 1) + terms.bit_length(),
        budget=budget,
    )
    if any(
        isinstance(value, exact_algebraic_geometry_owner.RadicalElement)
        for point in corners
        for value in point
    ):
        exact_algebraic_geometry_owner.reserve_algebraic_work(
            corners, arithmetic, dimension * dimension + terms, dimension, budget=budget
        )
    columns = tuple(
        tuple(corners[column + 1][axis] - corners[0][axis] for column in range(dimension))
        for axis in range(dimension)
    )
    if dimension == 1:
        determinant_value = columns[0][0]
    elif dimension == 2:
        determinant_value = columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0]
    else:
        determinant_value = (
            columns[0][0]
            * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
            - columns[0][1]
            * (columns[1][0] * columns[2][2] - columns[1][2] * columns[2][0])
            + columns[0][2]
            * (columns[1][0] * columns[2][1] - columns[1][1] * columns[2][0])
        )
    if determinant_value <= 0:
        raise ValueError(
            "mapped_geometry.jacobian must have a strictly positive certified bound; got nonpositive Jacobian bound"
        )
    measure = determinant_value / math.factorial(dimension)
    value, error = _publish_exact_integral(measure, budget=budget)
    return value, error, True


def _exact_affine_simplex_measure(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparation: tuple[_NestedCornerProofs, CoordinateLiveStorage] | None = None,
) -> tuple[float, float, bool] | None:
    """algebraic degree-one simplex measure without materializing its polynomials.

    The target kind and degree do not prove the physical map affine: a
    restriction keeps its source's coefficients and degree. The shortcut applies
    only when the unrestricted source is a degree-one simplex whose basis has
    total degree at most one; other maps return ``None`` for expression
    integration.

    A closed measure batch may retain the same actual element's completed
    cardinal/source proof, as nested-map proofs do. Only fully admitted owners
    with cardinal corners are retained; other sources take the original route.
    Cell coefficient banks, corner/component work and determinants are never
    retained here.
    """

    scalar = require_scalar_coordinate_element(element, "Exact affine measure")
    dimension = scalar.topological_dimension
    ambient = len(local[0])
    if (
        scalar.cell_kind not in ("interval", "triangle", "tetrahedron")
        or scalar.degree != 1
        or (
            ambient != dimension and not (scalar.cell_kind == "triangle" and ambient == 3)
        )
    ):
        return None
    proof = None if preparation is None else preparation[0].get(id(scalar))
    if proof is not None and proof[0] is scalar:
        _, _, nodes = proof
    else:
        root = scalar
        while isinstance(root, RestrictedCellGeometryElement):
            root = root.source_element
        if (
            not isinstance(
                root,
                (reference_types_owner.FiniteElementSpec, BarycentricCellGeometryElement),
            )
            or root.cell_kind not in ("interval", "triangle", "tetrahedron")
            or root.degree != 1
        ):
            return None
        # Admit the dictionary entry, strong owner and complete corner-node
        # tuple before preparing or retaining any operation-local metadata.
        previous_bound = 0 if preparation is None else preparation[1].bound
        if preparation is not None:
            preparation[1].set_bound(previous_bound + 256 + 8 * (dimension + 1))
            # A new source must itself have a closed, actual-owner-bound input
            # graph before any of its preparation can span an earlier proof.
            # Pay the complete structure/NumPy-byte/JAX check, not an ambient
            # admission flag; custom protocols retire earlier proofs first.
            if imported_coordinate_enclosure.coordinate_identity_inputs(
                scalar, budget
            ) is None:
                preparation[0].clear()
                previous_bound = 0
                preparation[1].set_bound(256 + 8 * (dimension + 1))
        basis = source_basis(root, budget=budget)
        if basis is None or any(sum(index) > 1 for term in basis for index in term):
            if preparation is not None:
                preparation[1].set_bound(previous_bound)
            return None
        nodes = _cardinal_corner_nodes(scalar, budget=budget)
        if preparation is not None:
            admission = budget.element_identity_admission_cache.get(id(scalar))
            if nodes is not None and admission is not None and admission.owner is scalar:
                preparation[0][id(scalar)] = (scalar, True, nodes)
            else:
                # Protocol/custom owners and weighted corners must keep their
                # complete original authentication/evaluation route. Retire
                # earlier proofs as well, so none spans unchecked custom reentry.
                preparation[0].clear()
                preparation[1].set_bound(0)
    if nodes is None:
        corners = coordinate_corner_images(scalar, local, budget=budget)
    else:
        # The shared exact cardinal proof selects source-owned coefficients,
        # avoiding a per-cell map key and retained weighted-image graph.
        budget.reserve(len(nodes) * (1 + ambient), 128 + 8 * len(nodes))
        corners = tuple(local[node] for node in nodes)
    if corners is None or len(corners) != dimension + 1:
        return None
    return _exact_affine_simplex_measure_from_corners(corners, budget=budget)


def _require_measure_publication(
    integral: tuple[float, float],
    absolute_per_cell: Fraction,
    relative: float,
) -> None:
    allowed = max(
        absolute_per_cell,
        Fraction(relative)
        * max(Fraction(integral[0]) - Fraction(integral[1]), Fraction(0)),
    )
    if Fraction(integral[1]) > allowed:
        raise transition_types.CellGeometryTransitionError(
            "resource_limit",
            "cell_measure.publication_error must fit the integration error budget; got excess binary64 publication error",
            measured=integral[1],
            limit=float(allowed),
        )


class _CellMeasureControls(StrictModule):
    absolute_per_cell: Fraction
    relative: float
    maximum_subcells: int
    maximum_binomial_terms: int
    work: list[int]


class _CoordinateCellMeasure(StrictModule):
    integral: tuple[float, float]
    exact: bool
    reference: tuple[Polynomial, Fraction, float, float] | None


def _measure_coordinate_cell(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    controls: _CellMeasureControls,
    references: dict[str, list[tuple[Polynomial, Fraction, float, float]]],
    *,
    budget: CoordinateEnclosureBudget,
) -> _CoordinateCellMeasure:
    exact = True
    reference = None
    work = controls.work
    affine = _exact_affine_simplex_measure(element, local, budget=budget)
    if affine is not None:
        affine_integral = affine[:2]
        exact = exact and affine[2]
        _require_measure_publication(
            affine_integral, controls.absolute_per_cell, controls.relative
        )
        return _CoordinateCellMeasure(affine_integral, exact, None)
    embedded: bool = (
        reference_cell_topology(element.cell_kind).dimension == 2 and len(local[0]) == 3
    )
    coordinates: tuple[Expression, ...] = ()
    constant_axes = []
    if embedded:
        coordinates = mapped_coordinate_expressions(element, local, budget=budget)
        if element.cell_kind == "quadrilateral":
            constant_axes = [
                axis
                for axis, polynomial in enumerate(coordinates)
                if all(
                    not imported_coordinate_enclosure.expression_derivative(
                        polynomial, i, budget=budget
                    )
                    for i in range(2)
                )
            ]
    match embedded:
        case False | True if not embedded or (
            element.cell_kind == "quadrilateral" and len(constant_axes) == 1
        ):
            density = mapped_density_expression(element, local, budget=budget)
            if isinstance(density, RationalPolynomial):
                value, error, rational_exact = _integrate_mapped_rational(
                    density,
                    element.cell_kind,
                    absolute_tolerance=controls.absolute_per_cell,
                    relative_tolerance=controls.relative,
                    maximum_terms=controls.maximum_binomial_terms,
                    budget=budget,
                )
                integral = value, error
                exact = exact and rational_exact
            else:
                integral = integrate_mapped_polynomial(
                    density,
                    element.cell_kind,
                    budget=budget,
                )
            _require_measure_publication(
                integral, controls.absolute_per_cell, controls.relative
            )
        case True if any(isinstance(value, RationalPolynomial) for value in coordinates):
            exact = False
            integral = _certified_embedded_cell_measure(
                element,
                local,
                controls.absolute_per_cell,
                controls.relative,
                int(controls.maximum_subcells),
                int(controls.maximum_binomial_terms),
                work,
                budget=budget,
            )
        case True:
            exact = False
            gram = embedded_squared_density(
                element,
                local,
                prepared_coordinates=coordinates,
                budget=budget,
            )
            if isinstance(gram, RationalPolynomial):
                raise TypeError(
                    "cell_measure.gram must be polynomial for polynomial coordinates; got rational expression"
                )
            integral, reference = reuse_embedded_polynomial_area(
                gram,
                element.cell_kind,
                references.get(element.cell_kind, []),
                controls.absolute_per_cell,
                controls.relative,
                int(controls.maximum_subcells),
                int(controls.maximum_binomial_terms),
                work,
                budget=budget,
            )
        case _:
            assert_never(embedded)

    return _CoordinateCellMeasure(integral, exact, reference)


def _certified_cell_measures(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-10,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Physical cell measures with quantitative absolute integration errors.

    Volume/axis-plane densities retain algebraic rational integration. General
    embedded triangle/quad sqrt-Gram densities use Bernstein-positive dyadic
    subdivisions and validated binomial polynomial integration: remainder,
    source conversion, coefficient arithmetic and publication errors are enclosed.
    No quadrature samples certify this measure. The equality flag is false on that route.

    Proved affine simplices publish their exact corner determinant directly;
    they do not materialize per-cell polynomial preparation/cache-key graphs.

    Absolute tolerance is allocated across cells; relative budgets use certified
    lower areas, so summed errors are at most absolute plus relative total area.
    ``maximum_work`` bounds dense coefficient products across all cells;
    ``maximum_subcells`` bounds visited reference pieces per cell. Exhaustion
    raises structured ``CellGeometryTransitionError`` without partial acceptance.
    """

    absolute = nonnegative_finite_float(absolute_tolerance, "absolute_tolerance")
    relative = nonnegative_finite_float(relative_tolerance, "relative_tolerance")
    if (
        absolute == relative == 0
        or min(maximum_work, maximum_subcells, maximum_binomial_terms) <= 0
    ):
        raise ValueError(
            "cell_measure.limits must give positive error and work budgets; got nonpositive limit"
        )

    match geometry.exact_source:
        case (
            ExactPowerCellGeometryLinearActionSource()
            | ExactPowerCellGeometryRestrictionSource()
        ) if all(block.cell_kind == "hexahedron" for block in mesh.blocks):
            # The retained algebraic vertex bank defines the Q1 polynomials.
            pass
        case (
            ExactPowerCellGeometrySource()
            | ExactPowerCellGeometryRestrictionSource()
            | ExactPowerCellGeometryLinearActionSource()
        ):
            return imported_exact_power_consumers.exact_power_cell_measures(
                mesh, geometry, maximum_work=maximum_work, budget=budget
            )
        case None:
            pass
        case (
            ExactPlcCellGeometrySource()
            | ExactPlcCellGeometryConvexSource()
            | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
            | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
        ):
            pass
        case invalid:
            assert_never(invalid)

    ledger = budget

    cells = mapped_geometry_cells(mesh, geometry, budget=budget)
    integrals: list[tuple[float, float]] = []
    exact, work = True, [0, int(maximum_work)]
    absolute_per_cell = imported_fractions.Fraction(absolute) / len(cells)
    controls = _CellMeasureControls(
        absolute_per_cell, relative, maximum_subcells, maximum_binomial_terms, work
    )
    from ._coordinate_operands import _signed_determinant_operand_batch

    with (
        ledger.temporary_scope(),
        ledger.live_storage() as affine_storage,
        _signed_determinant_operand_batch(mesh, geometry, budget) as signed_batch,
    ):
        references: dict[
            str, list[tuple[Polynomial, imported_fractions.Fraction, float, float]]
        ] = {}
        affine_proofs: _NestedCornerProofs = {}

        def affine_measure_key(
            element: CellGeometryElement, local: CoordinateSourceBank
        ) -> imported_coordinate_enclosure.CoordinatePreparationKey | None:
            # Only the operation's completed actual-owner/cardinal proof admits
            # reuse. Its first source authentication still takes the original
            # route below; neither a cached verdict nor a Fraction hash earns it.
            proof = affine_proofs.get(id(element))
            admission = ledger.element_identity_admission_cache.get(id(element))
            if (
                proof is None
                or proof[0] is not element
                or admission is None
                or admission.owner is not element
                or type(local) is not tuple
            ):
                return None
            payload_size = 8
            for row in local:
                ledger.reserve(1)
                if type(row) is not tuple:
                    return None
                payload_size += 8
                for value in row:
                    ledger.reserve(1)
                    if (
                        type(value) is not Fraction
                        or type(value.numerator) is not int
                        or type(value.denominator) is not int
                        or value.denominator <= 0
                    ):
                        return None
                    payload_size += (
                        16
                        + max(1, (value.numerator.bit_length() + 8) // 8)
                        + max(1, (value.denominator.bit_length() + 7) // 8)
                    )
            # Admit the growing bytearray, largest transient integer encoding,
            # final immutable bytes, namespace and key before allocating them.
            # This lives alongside the ledger's entire retained cache union.
            identity = admission.identity
            ledger.reserve(0, 1024 + 3 * payload_size + 4 * len(identity[0]))
            payload = bytearray()
            payload.extend(len(local).to_bytes(8, "big"))
            for row in local:
                ledger.reserve(1)
                payload.extend(len(row).to_bytes(8, "big"))
                for value in row:
                    ledger.reserve(1)
                    assert isinstance(value, Fraction)
                    numerator, denominator = value.numerator, value.denominator
                    size = max(1, (numerator.bit_length() + 8) // 8)
                    payload.extend(size.to_bytes(8, "big"))
                    payload.extend(numerator.to_bytes(size, "big", signed=True))
                    size = max(1, (denominator.bit_length() + 7) // 8)
                    payload.extend(size.to_bytes(8, "big"))
                    payload.extend(denominator.to_bytes(size, "big"))
            # Complete current values, not digests or mutable Fraction owners,
            # are numerical authority. The empty bank avoids retaining the
            # per-cell coefficient graph and separates this key's namespace.
            return (
                "affine-rational-measure:" + identity[0],
                identity[1],
                (),
                bytes(payload),
            )

        for element, local in cells:
            if signed_batch is not None:
                signed_batch.for_element(element)
            # Always earn the first actual affine/cardinal proof afresh in this
            # operation. Later closed rational cells can skip only numerical
            # selection, determinant arithmetic and rounding already earned.
            with ledger.temporary_scope():
                key = affine_measure_key(element, local)
                retained = None
                if key is not None:
                    ledger.reserve(1)
                    retained = ledger.exact_cell_measure_cache.get(key)
                    if (
                        retained is not None
                        and math.isfinite(retained[0])
                        and math.isfinite(retained[1])
                        and retained[0] > retained[1] >= 0
                    ):
                        _require_measure_publication(
                            retained, controls.absolute_per_cell, controls.relative
                        )
                        integrals.append(retained)
                        continue
                affine = _exact_affine_simplex_measure(
                    element,
                    local,
                    preparation=(affine_proofs, affine_storage),
                    budget=budget,
                )
                if affine is not None:
                    integral = affine[:2]
                    _require_measure_publication(
                        integral, controls.absolute_per_cell, controls.relative
                    )
                    if key is None:
                        key = affine_measure_key(element, local)
                        if key is not None and affine[2]:
                            ledger.reserve(1)
                            retained = ledger.exact_cell_measure_cache.get(key)
                    if affine[2]:
                        if (
                            key is not None
                            and math.isfinite(integral[0])
                            and math.isfinite(integral[1])
                            and integral[0] > integral[1] >= 0
                        ):
                            if retained is None:
                                ledger.retain_basis((key, integral))
                                ledger.exact_cell_measure_cache[key] = integral
                            if (
                                ledger.capture_coordinate_operands
                                and ledger.capture_measure_operands
                            ):
                                from ._coordinate_operands import (
                                    _remember_exact_cell_measure_operand,
                                )

                                # Only this freshly executed exact kernel may
                                # issue numerical operands. The cache-hit
                                # shortcut above never crosses this boundary.
                                _remember_exact_cell_measure_operand(
                                    key, integral, ledger
                                )
                    if key is None:
                        # No proof may span unsupported scalar callbacks before
                        # another cell attempts the closed rational reuse path.
                        affine_proofs.clear()
                        affine_storage.set_bound(0)
                    integrals.append(integral)
                    exact = exact and affine[2]
                    continue
            # No operation-local proof may span the general source route:
            # unsupported/custom protocols must not introduce unchecked reentry
            # between this proof's authentication and its next use.
            affine_proofs.clear()
            affine_storage.set_bound(0)
            # An algebraic integral and its outward publication error depend
            # only on the immutable element/source bank, not on the caller's
            # requested tolerance. Recheck that tolerance on every import.
            local, key = imported_coordinate_enclosure.coordinate_preparation_key(
                element, local, budget=budget
            )
            budget.reserve(1)
            retained = ledger.exact_cell_measure_cache.get(key)
            if retained is not None:
                _require_measure_publication(
                    retained, controls.absolute_per_cell, controls.relative
                )
                integrals.append(retained)
                continue
            reference = None
            with ledger.temporary_scope():
                measured = _measure_coordinate_cell(
                    element, local, controls, references, budget=budget
                )
                exact = exact and measured.exact
                integral, reference = measured.integral, measured.reference
                integrals.append(integral)
            if measured.exact:
                ledger.retain_basis((key, integral))
                ledger.exact_cell_measure_cache[key] = integral
            if reference is not None:
                gram = reference[0]
                numerator, denominator = (
                    imported_coordinate_enclosure.coefficient_profile(
                        (gram,), budget=budget
                    )
                )
                reserve_polynomial(
                    1, len(gram), 2, numerator + denominator, budget=budget
                )
                references.setdefault(element.cell_kind, []).append(reference)
    return (
        np.asarray([value for value, _ in integrals], dtype=np.float64),
        np.asarray([error for _, error in integrals], dtype=np.float64),
        exact,
    )


@checked
def certified_cell_measures(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-10,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Measure cells with a work bound covering source preparation and integration."""
    starting_work = budget.work_units
    try:
        with (
            budget.measurement(),
            budget.bound_stage(maximum_work, budget.maximum_memory_bytes),
        ):
            return _certified_cell_measures(
                mesh,
                geometry,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                maximum_work=maximum_work,
                maximum_subcells=maximum_subcells,
                maximum_binomial_terms=maximum_binomial_terms,
                budget=budget,
            )
    except CoordinateEnclosureResourceError as error:
        measured = error.requested
        limit = error.limit
        # Only this operation's cap is phase-relative; inherited caps remain cumulative.
        if (
            error.resource == "coefficient_work"
            and error.limit == starting_work + maximum_work
        ):
            measured -= starting_work
            limit -= starting_work
        raise transition_types.CellGeometryTransitionError(
            "resource_limit",
            "cell_measure.resources must fit source preparation and integration budgets; got work or storage refusal",
            measured=float(measured),
            limit=float(limit),
        ) from error


@checked
def mapped_root_identities(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
) -> tuple[tuple[str, str, int, tuple[int, ...]], ...]:
    origin = geometry.restriction_source
    result: list[tuple[str, str, int, tuple[int, ...]]] = []
    for block in mesh.blocks:
        if origin is None:
            cells = np.asarray(block.global_ids)
            corners = np.asarray(mesh.vertex_global_ids)[np.asarray(block.vertices)]
            geometry_id, topology_id = cell_geometry_id(geometry), mesh.topology_id
        else:
            cells = np.asarray(origin.block_parent_cell_ids[block.name])
            corners = np.asarray(origin.block_parent_vertex_ids[block.name])
            geometry_id, topology_id = (
                origin.source_geometry_id,
                origin.source_topology_id,
            )
        result.extend(
            (
                geometry_id,
                topology_id,
                int(cell),
                tuple(int(v) for v in vertices if v >= 0),
            )
            for cell, vertices in zip(cells, corners, strict=True)
        )
    return tuple(result)


type _AffinePairLaw = tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
    tuple[tuple[Fraction, ...], ...],
]


def _affine_pair_bound(
    fine_kind: str,
    fine: CoordinateSourceBank,
    coarse_kind: str,
    coarse: CoordinateSourceBank,
    pair: NestedReferencePair,
    laws: dict[tuple[object, ...], _AffinePairLaw | None],
    law_storage: CoordinateLiveStorage | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Bound one proved affine simplex difference at its extreme vertices.

    The exact child law and the coarse reference parameters of its fine
    corners depend only on the pair's binary64 view and retained rational rows.
    With ``law_storage``, a law seen once keeps only its admitted key and a
    repeated law is retained, so later equal laws charge only their difference
    work and unique laws never accumulate parameter banks.
    """
    fine_dimension = reference_cell_topology(fine_kind).dimension
    coarse_dimension = reference_cell_topology(coarse_kind).dimension
    ambient = len(fine[0])
    key: tuple[object, ...] = ()
    law: _AffinePairLaw | None = None
    seen = False
    if law_storage is not None:
        law_storage.set_bound(
            law_storage.bound + 256 + 16 * (pair.matrix.size + pair.offset.size)
        )
        key = (
            fine_kind,
            pair.matrix.shape,
            np.asarray(pair.matrix, dtype=np.float64).tobytes(),
            np.asarray(pair.offset, dtype=np.float64).tobytes(),
            pair.exact_matrix,
            pair.exact_offset,
        )
        seen = key in laws
        law = laws.get(key)
        if seen:
            # Only a first sighting retains its key.
            law_storage.set_bound(
                law_storage.bound - 256 - 16 * (pair.matrix.size + pair.offset.size)
            )
        else:
            laws[key] = None
    if law is None:
        # Binary64 pairs stay below 1076 bits; retained rational source-owned
        # pairs charge their full rational size.
        matrix, offset = rational_pair_affine(pair)
        imported_linalg__hermitian_spectral.reserve_fraction_work(
            budget,
            pair.matrix.size + pair.offset.size,
            pair.matrix.size + pair.offset.size,
            max(1076, rational_affine_bits((matrix, offset))),
        )
        parameters = None
    else:
        matrix, offset, parameters = law
    profile_rows = (*fine, *coarse, *matrix, offset)
    budget.reserve(sum(len(row) for row in profile_rows))
    numerator = max(
        (
            abs(value.numerator).bit_length()
            if isinstance(value, Fraction)
            else value.coefficient_bits
            for row in profile_rows
            for value in row
        ),
        default=1,
    )
    denominator = max(
        (
            value.denominator.bit_length()
            if isinstance(value, Fraction)
            else sum(
                coefficient.denominator.bit_length() for _, coefficient in value.terms
            )
            for row in profile_rows
            for value in row
        ),
        default=1,
    )
    bits = (
        4 * (numerator + denominator + 1) * (coarse_dimension + 1) * (fine_dimension + 3)
        + 64
    )
    parameter_work = 0 if parameters is not None else coarse_dimension * (
        2 * fine_dimension + 1
    )
    work = len(fine) * (parameter_work + ambient * (3 * coarse_dimension + 2))
    imported_linalg__hermitian_spectral.reserve_fraction_work(
        budget,
        work,
        len(fine) * (coarse_dimension + ambient) + 2 * coarse_dimension * ambient,
        bits,
    )
    if any(
        isinstance(value, exact_algebraic_geometry_owner.RadicalElement)
        for row in profile_rows
        for value in row
    ):
        exact_algebraic_geometry_owner.reserve_algebraic_work(
            profile_rows,
            work,
            len(fine) * (coarse_dimension + ambient),
            coarse_dimension + 1,
            budget=budget,
        )
    if parameters is None:
        vertices = reference_cell_topology(fine_kind).vertices
        retain = law_storage is not None and seen
        if retain and law_storage is not None:
            # Admit the repeated law before building it: every exact scalar of
            # the map and of the (dimension + 1)-term parameter sums.
            scalar_bits = (len(offset) + 1) * (
                rational_affine_bits((matrix, offset)) + 2
            )
            scalars = (
                sum(len(row) for row in matrix) + len(offset) + len(vertices) * len(offset)
            )
            law_storage.set_bound(
                law_storage.bound
                + 512
                + scalars * (128 + 2 * ((scalar_bits + 7) // 8))
            )
        parameters = tuple(
            tuple(
                start
                + sum(
                    (
                        coefficient * Fraction(float(value))
                        for coefficient, value in zip(row, reference, strict=True)
                    ),
                    Fraction(0),
                )
                for start, row in zip(offset, matrix, strict=True)
            )
            for reference in vertices
        )
        if retain:
            laws[key] = (matrix, offset, parameters)
    bound = 0.0
    origin = coarse[0]
    for axis in range(ambient):
        # Coarse edge components are shared by every fine corner of this axis.
        start = origin[axis]
        edges = tuple(corner[axis] - start for corner in coarse[1:])
        differences = tuple(
            actual[axis]
            - start
            - sum(
                (
                    parameter * edge
                    for parameter, edge in zip(point, edges, strict=True)
                ),
                Fraction(0),
            )
            for actual, point in zip(fine, parameters, strict=True)
        )
        lower, upper = min(differences), max(differences)
        if lower or upper:
            bound = max(
                bound,
                abs(coefficient_endpoint(lower, upper=False, saturate=True)),
                abs(coefficient_endpoint(upper, upper=True, saturate=True)),
            )
    return bound


@checked
def affine_nested_geometry_pair_bound(
    fine_element: CellGeometryElement,
    fine_local: CoordinateCoefficients,
    coarse_element: CellGeometryElement,
    coarse_local: CoordinateCoefficients,
    pair: NestedReferencePair,
    *,
    budget: CoordinateEnclosureBudget,
) -> float | None:
    """Bound a proved affine simplex difference exactly at its extreme vertices."""

    fine = imported_coordinate_enclosure.affine_simplex_corner_images(
        fine_element, fine_local, budget=budget
    )
    coarse = imported_coordinate_enclosure.affine_simplex_corner_images(
        coarse_element, coarse_local, budget=budget
    )
    if fine is None or coarse is None:
        return None
    return _affine_pair_bound(
        fine_element.cell_kind,
        fine,
        coarse_element.cell_kind,
        coarse,
        pair,
        {},
        None,
        budget=budget,
    )


@checked
def polynomial_nested_geometry_pair_bound(
    fine_element: CellGeometryElement,
    fine_local: CoordinateCoefficients,
    coarse_element: CellGeometryElement,
    coarse_local: CoordinateCoefficients,
    pair: NestedReferencePair,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Bound the full source-polynomial difference on the fine reference cell."""
    if isinstance(
        fine_element, (PolynomialComposedCellGeometryElement, RestrictedCellGeometryElement)
    ):
        # This is an equality of definitions, not a fitted corner map or a
        # positivity verdict. Authenticate the current complete laws and ordered
        # banks once, including the scalar-representation part of each bank key.
        count = (
            fine_local.size
            if isinstance(fine_local, np.ndarray)
            else sum(len(row) for row in fine_local)
        ) + (
            coarse_local.size
            if isinstance(coarse_local, np.ndarray)
            else sum(len(row) for row in coarse_local)
        )
        budget.reserve(count + 4, 512 + count)
        _, fine_key = imported_coordinate_enclosure.coordinate_preparation_key(
            fine_element, fine_local, budget=budget
        )
        _, coarse_key = imported_coordinate_enclosure.coordinate_preparation_key(
            coarse_element, coarse_local, budget=budget
        )
        coarse_identity = coarse_key[0], coarse_key[1]
        source_identity = (
            coarse_identity
            if fine_element.source_element is coarse_element
            else imported_coordinate_enclosure.element_identity(
                fine_element.source_element, budget=budget
            )
        )
        if (
            source_identity == coarse_identity
            and len(fine_key) == len(coarse_key)
            and fine_key[2] == coarse_key[2]
            and fine_key[3:] == coarse_key[3:]
        ):
            # Admit the rational_pair_arrays validation scratch, exact atoms and
            # object arrays before allocating them. Binary64 rationals need at
            # most 1075 bits; explicit rational sources may need more.
            pair_count = pair.matrix.size + pair.offset.size
            bits = 1075
            if pair.exact_matrix is not None and pair.exact_offset is not None:
                bits = max(
                    bits,
                    max(
                        (
                            max(
                                abs(value.numerator).bit_length(),
                                value.denominator.bit_length(),
                            )
                            for row in pair.exact_matrix
                            for value in row
                            if isinstance(value, Fraction)
                        ),
                        default=1,
                    ),
                    max(
                        (
                            max(
                                abs(value.numerator).bit_length(),
                                value.denominator.bit_length(),
                            )
                            for value in pair.exact_offset
                            if isinstance(value, Fraction)
                        ),
                        default=1,
                    ),
                )
            budget.reserve(
                3 * pair_count + 2,
                512 + pair_count * (128 + 2 * declared_integer_storage_bytes(bits)),
            )
            exact_matrix, exact_offset = rational_pair_arrays(pair)
            if isinstance(fine_element, RestrictedCellGeometryElement):
                budget.reserve(pair_count, 512 + 16 * pair_count)
                actual_matrix = np.asarray(fine_element.matrix, dtype=np.float64)
                actual_offset = np.asarray(fine_element.offset, dtype=np.float64)
                if (
                    actual_matrix.shape == exact_matrix.shape
                    and actual_offset.shape == exact_offset.shape
                    and all(
                        Fraction(float(actual)) == expected
                        for actual, expected in zip(
                            actual_matrix.flat, exact_matrix.flat, strict=True
                        )
                    )
                    and all(
                        Fraction(float(actual)) == expected
                        for actual, expected in zip(
                            actual_offset.flat, exact_offset.flat, strict=True
                        )
                    )
                ):
                    # Both maps are the same complete source law evaluated at
                    # exactly the same affine reference argument, not just at
                    # its corners or within a floating-point fitting tolerance.
                    return 0.0
            else:
                arguments = imported_coordinate_enclosure._reference_composition_from_identity(
                    fine_element, (fine_key[0], fine_key[1]), budget=budget
                )
                expected = imported_coordinate_enclosure.restriction_chart_arguments(
                    fine_element.cell_kind, exact_offset, exact_matrix, budget=budget
                )
                expected_arguments = (
                    expected.arguments
                    if isinstance(expected, imported_coordinate_enclosure.PreparedPolynomialArguments)
                    else expected
                )
                budget.reserve(
                    len(arguments)
                    + sum(len(argument) for argument in arguments)
                    + sum(len(argument) for argument in expected_arguments)
                )
                if arguments == expected_arguments:
                    # For every reference point, sum_i c_i (b_i o A) is exactly
                    # (sum_i c_i b_i) o A, for arbitrary nonlinear/nonunit sources.
                    return 0.0
    fine_coordinates = mapped_coordinate_expressions(
        fine_element, fine_local, budget=budget
    )
    coarse_coordinates = mapped_coordinate_expressions(
        coarse_element, coarse_local, budget=budget
    )
    exact_matrix, exact_offset = rational_pair_arrays(pair)
    restricted = mapped_expression_restriction(
        coarse_coordinates,
        coarse_element.cell_kind,
        fine_element.cell_kind,
        exact_matrix,
        exact_offset,
        budget=budget,
    )
    domain = (
        "simplex"
        if fine_element.cell_kind in ("interval", "triangle", "tetrahedron")
        else "prism"
        if fine_element.cell_kind == "prism"
        else "box"
    )
    bound = 0.0
    for actual, expected in zip(fine_coordinates, restricted, strict=True):
        difference = expression_add(
            actual, expression_scale(expected, -1, budget=budget), budget=budget
        )
        if difference:
            lower, upper = expression_bounds(
                difference,
                domain,
                reference_cell_topology(fine_element.cell_kind).dimension,
                budget=budget,
            )
            bound = max(bound, abs(lower), abs(upper))
    return bound


# Element identity -> (element, affine simplex source, cardinal corner rows).
type _NestedCornerProofs = dict[
    int, tuple[ScalarCoordinateElement, bool, tuple[int, ...] | None]
]


def _nested_affine_corner_images(
    element: ScalarCoordinateElement,
    local: CoordinateSourceBank,
    *,
    proofs: _NestedCornerProofs,
    storage: CoordinateLiveStorage,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank | None:
    """Select proved cardinal affine images from the admitted exact source bank.

    ``proofs`` keeps each live element's completed affine-basis and cardinal
    corner verdicts for one proof, so cells sharing an element object do not
    rebuild its source identity; both verdicts depend on the element alone.
    """
    proof = proofs.get(id(element))
    if proof is None or proof[0] is not element:
        affine = imported_coordinate_enclosure.affine_simplex_source(
            element, budget=budget
        )
        nodes = _cardinal_corner_nodes(element, budget=budget) if affine else None
        storage.set_bound(storage.bound + 256 + (0 if nodes is None else 8 * len(nodes)))
        proofs[id(element)] = (element, affine, nodes)
    else:
        _, affine, nodes = proof
    if not affine:
        return None
    if nodes is None:
        return imported_coordinate_enclosure.coordinate_corner_images(
            element, local, budget=budget
        )
    # Preserve the canonical corner/component evaluation charges; map-key
    # preparation and intermediate weighted operations are no longer needed.
    budget.reserve(len(nodes) * (1 + len(local[0])), 128 + 8 * len(nodes))
    return tuple(local[node] for node in nodes)


@checked
def certify_nested_geometry_pairs(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    pairs: tuple[NestedReferencePair, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Prove paired coordinate equality from specified algebraic source expressions."""
    source = mapped_geometry_cells(source_mesh, source_geometry, budget=budget)
    target = mapped_geometry_cells(target_mesh, target_geometry, budget=budget)
    if (
        source_geometry.restriction_source is not None
        or target_geometry.restriction_source is not None
    ):
        source_roots = mapped_root_identities(source_mesh, source_geometry)
        target_roots = mapped_root_identities(target_mesh, target_geometry)
        if any(
            source_roots[pair.source_cell] != target_roots[pair.target_cell]
            for pair in pairs
        ):
            raise ValueError(
                "nested_geometry.source_root must retain the scientific root identity; got stale or mismatched root identities"
            )
    bound = 0.0
    # Children of one coarse cell share its corner images, and equal child laws
    # share their exact reference parameters; both live for this proof only.
    laws: dict[tuple[object, ...], _AffinePairLaw | None] = {}
    coarse_images: dict[tuple[bool, int], CoordinateSourceBank | None] = {}
    proofs: _NestedCornerProofs = {}
    with budget.temporary_scope(), budget.live_storage() as reuse_storage:
        for pair in pairs:
            # Coordinate banks remain request-owned; this pair's restrictions and
            # difference enclosures end after its scalar bound has been recorded.
            with budget.temporary_scope():
                fine, coarse = (
                    (source[pair.source_cell], target[pair.target_cell])
                    if pair.fine_is_source
                    else (target[pair.target_cell], source[pair.source_cell])
                )
                fine_element, fine_local = fine
                coarse_element, coarse_local = coarse
                fine_corners = _nested_affine_corner_images(
                    fine_element,
                    fine_local,
                    proofs=proofs,
                    storage=reuse_storage,
                    budget=budget,
                )
                coarse_key = (
                    pair.fine_is_source,
                    pair.target_cell if pair.fine_is_source else pair.source_cell,
                )
                if coarse_key in coarse_images:
                    coarse_corners = coarse_images[coarse_key]
                else:
                    coarse_corners = (
                        _nested_affine_corner_images(
                            coarse_element,
                            coarse_local,
                            proofs=proofs,
                            storage=reuse_storage,
                            budget=budget,
                        )
                    )
                    reuse_storage.set_bound(
                        reuse_storage.bound
                        + 256
                        + (0 if coarse_corners is None else 8 * len(coarse_corners))
                    )
                    coarse_images[coarse_key] = coarse_corners
                if fine_corners is None or coarse_corners is None:
                    pair_bound = polynomial_nested_geometry_pair_bound(
                        fine_element,
                        fine_local,
                        coarse_element,
                        coarse_local,
                        pair,
                        budget=budget,
                    )
                else:
                    pair_bound = _affine_pair_bound(
                        fine_element.cell_kind,
                        fine_corners,
                        coarse_element.cell_kind,
                        coarse_corners,
                        pair,
                        laws,
                        reuse_storage,
                        budget=budget,
                    )
                bound = max(bound, pair_bound)
    scale_ = max(
        float(np.max(np.abs(np.asarray(source_geometry.coordinates)))),
        float(np.max(np.abs(np.asarray(target_geometry.coordinates)))),
        1.0,
    )
    if bound > 64 * _EPSILON * scale_:
        raise ValueError(
            "nested_geometry.physical_map must agree under the reference action within rounding slack; got excess physical-map deviation"
        )
    return bound
