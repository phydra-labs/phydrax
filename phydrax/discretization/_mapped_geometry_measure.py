from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from .._validation import finite_real_scalar, positive_finite_float
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import checked
from ._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
    ScalarCoordinateElement,
    coordinate_effective_basis,
)
from ._cell_geometry_validity import certify_cell_geometry_validity
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import (
    add,
    axes,
    bernstein_coefficients,
    compose,
    compose_expression,
    constant,
    derivative,
    ExactScalar,
    Expression,
    linear_combinations,
    multiply,
    normalize_pyramid_reference_expression,
    Polynomial,
    pyramid_reference_to_box_arguments,
    RationalPolynomial,
    scale,
    sum_polynomials,
)
from ._exact_algebraic_geometry import coefficient_endpoint, coefficient_interval
from ._reference_cell import reference_cell_topology


class MappedCellMeasures(NamedTuple):
    values: np.ndarray
    absolute_errors: np.ndarray
    exact: bool


MappedGeometryCell = tuple[
    CellGeometryElement, tuple[tuple[ExactScalar, ...], ...], int
]
_BoxBounds = tuple[tuple[Fraction, Fraction], ...]


class _PositiveMeasurePatch(NamedTuple):
    dimension: int
    numerator: Polynomial
    denominator: Polynomial
    jacobian: Polynomial
    volume_scale: Fraction
    jacobian_integral: Fraction
    numerator_center: ExactScalar
    denominator_center: ExactScalar
    numerator_ratio: Fraction
    denominator_ratio: Fraction


def _reference_domain(cell_kind: str, /) -> str:
    if cell_kind in ("interval", "triangle", "tetrahedron") or cell_kind.startswith(
        "simplex:"
    ):
        return "simplex"
    if cell_kind == "prism":
        return "prism"
    if cell_kind == "pyramid":
        return "pyramid"
    if cell_kind in (
        "quadrilateral",
        "hexahedron",
        "tensor:1",
        "tensor:2",
        "tensor:3",
    ):
        return "box"
    raise ValueError(f"Mapped measures do not support reference cell {cell_kind!r}.")


def _reference_box_map(
    cell_kind: str, /
) -> tuple[tuple[Expression, ...], Polynomial]:
    dimension = reference_cell_topology(cell_kind).dimension
    unit = axes(dimension)
    one = constant(1, dimension)
    family = _reference_domain(cell_kind)
    if family == "box":
        return unit, one
    if family == "pyramid":
        if dimension != 3:
            raise ValueError("Pyramid mapped measures require three reference dimensions.")
        width = add(one, scale(unit[2], -1))
        return pyramid_reference_to_box_arguments(), multiply(width, width)
    if family == "prism":
        if dimension != 3:
            raise ValueError("Prism mapped measures require three reference dimensions.")
        width = add(one, scale(unit[0], -1))
        return (unit[0], multiply(width, unit[1]), unit[2]), width
    if dimension == 1:
        return unit, one
    if dimension == 2:
        width = add(one, scale(unit[0], -1))
        return (unit[0], multiply(width, unit[1])), width
    if dimension == 3:
        first_width = add(one, scale(unit[0], -1))
        second_width = add(one, scale(unit[1], -1))
        return (
            unit[0],
            multiply(first_width, unit[1]),
            multiply(multiply(first_width, second_width), unit[2]),
        ), multiply(multiply(first_width, first_width), second_width)
    raise ValueError(f"Simplex mapped measures do not support dimension {dimension}.")


def _pyramid_weighted_density(value: Expression, /) -> Expression:
    height = axes(3)[2]
    width = add(constant(1, 3), scale(height, -1))
    normalized = normalize_pyramid_reference_expression(
        multiply(value, multiply(width, width))
    )
    return normalized.expression


def _determinant(
    matrix: tuple[tuple[Expression, ...], ...], dimension: int, /
) -> Expression:
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        raise ValueError("Mapped Jacobian must be square for volume integration.")
    terms: list[Expression] = []
    for permutation in itertools.permutations(range(dimension)):
        inversions = sum(
            permutation[first] > permutation[second]
            for first in range(dimension)
            for second in range(first + 1, dimension)
        )
        term: Expression = constant(1, dimension)
        for row, column in enumerate(permutation):
            term = multiply(term, matrix[row][column])
        terms.append(scale(term, -1 if inversions % 2 else 1))
    return sum_polynomials(terms, dimension)


def _mapped_jacobian(
    element: CellGeometryElement,
    local: tuple[tuple[ExactScalar, ...], ...],
    /,
    *,
    cell_row: int | None = None,
) -> tuple[tuple[Expression, ...], ...]:
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError("Mapped coordinate Jacobians require a scalar coordinate element.")
    coordinates = mapped_coordinate_expressions(element, local, cell_row=cell_row)
    return tuple(
        tuple(
            derivative(coordinate, axis)
            for axis in range(element.topological_dimension)
        )
        for coordinate in coordinates
    )


def _gram_density(
    jacobian: tuple[tuple[Expression, ...], ...], dimension: int, /
) -> Expression:
    if len(jacobian) < dimension or any(len(row) != dimension for row in jacobian):
        raise ValueError("Embedded mapped Jacobian has incompatible dimensions.")
    gram = tuple(
        tuple(
            sum_polynomials(
                tuple(
                    multiply(jacobian[ambient][first], jacobian[ambient][second])
                    for ambient in range(len(jacobian))
                ),
                dimension,
            )
            for second in range(dimension)
        )
        for first in range(dimension)
    )
    return _determinant(gram, dimension)


@checked
def mapped_coordinate_expressions(
    element: CellGeometryElement,
    local: tuple[tuple[ExactScalar, ...], ...],
    /,
    *,
    cell_row: int | None = None,
) -> tuple[Expression, ...]:
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError("Mapped coordinate expressions require a scalar coordinate element.")
    scalar = element
    basis = coordinate_effective_basis(scalar, cell_row=cell_row)
    if not basis or len(basis) != len(local):
        raise ValueError("Mapped coordinate basis and local coefficient rows differ.")
    ambient = len(local[0])
    if any(len(row) != ambient for row in local):
        raise ValueError("Mapped coordinate coefficients must have one ambient width.")
    coefficients = tuple(
        tuple(local[dof][axis] for dof in range(len(local)))
        for axis in range(ambient)
    )
    return linear_combinations(basis, coefficients)


@checked
def mapped_density_expression(
    element: CellGeometryElement,
    local: tuple[tuple[ExactScalar, ...], ...],
    /,
    *,
    cell_row: int | None = None,
) -> Expression:
    """Return the exact signed Jacobian determinant of a mapped volume cell."""
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError("Mapped volume density requires a scalar coordinate element.")
    dimension = element.topological_dimension
    jacobian = _mapped_jacobian(element, local, cell_row=cell_row)
    if len(jacobian) != dimension:
        raise ValueError("Mapped volume integration requires a square physical map.")
    return _determinant(jacobian, dimension)


def _polynomial_integral(
    polynomial: Polynomial, cell_kind: str, /
) -> ExactScalar:
    topology = reference_cell_topology(cell_kind)
    dimension = topology.dimension
    family = _reference_domain(cell_kind)
    if family == "pyramid":
        transformed = _pyramid_weighted_density(polynomial)
        if isinstance(transformed, RationalPolynomial):
            raise RuntimeError("Polynomial pyramid measure normalization became rational.")
        polynomial = dict(transformed)
        family = "box"
    total: ExactScalar = Fraction(0)
    for exponent, coefficient in polynomial.items():
        if family == "simplex":
            moment = Fraction(
                math.prod(math.factorial(power) for power in exponent),
                math.factorial(sum(exponent) + dimension),
            )
        elif family == "prism":
            moment = Fraction(
                math.factorial(exponent[0]) * math.factorial(exponent[1]),
                math.factorial(exponent[0] + exponent[1] + 2)
                * (exponent[2] + 1),
            )
        else:
            moment = Fraction(1, math.prod(power + 1 for power in exponent))
        total += coefficient * moment
    return total


def _integrate_rational(
    value: RationalPolynomial,
    cell_kind: str,
    absolute_tolerance: Fraction,
    relative_tolerance: Fraction,
    /,
) -> tuple[float, float, bool]:
    if cell_kind == "pyramid":
        transformed = _pyramid_weighted_density(value)
        if not isinstance(transformed, RationalPolynomial):
            exact_value = _polynomial_integral(dict(transformed), "hexahedron")
            published, error = _publish_integral(exact_value)
            return published, _positive_outward_error(error), True
        value = transformed
        cell_kind = "hexahedron"
    dimension = reference_cell_topology(cell_kind).dimension
    domain = _reference_domain(cell_kind)
    numerator_controls = bernstein_coefficients(value.numerator, domain, dimension)
    denominator_controls = bernstein_coefficients(value.denominator, domain, dimension)
    if any(not isinstance(coefficient, Fraction) for coefficient in (*numerator_controls, *denominator_controls)):
        raise ValueError("Rational mapped measure integration requires rational coefficients.")
    lower = min(denominator_controls)
    upper = max(denominator_controls)
    if upper < 0:
        value = RationalPolynomial(
            scale(value.numerator, -1), scale(value.denominator, -1)
        )
        numerator_controls = tuple(-coefficient for coefficient in numerator_controls)
        lower, upper = -upper, -lower
    if lower <= 0:
        raise ValueError(
            "Rational mapped measure requires a strictly positive denominator enclosure."
        )
    center = (lower + upper) / 2
    ratio = add(constant(1, dimension), scale(value.denominator, -1 / center))
    contraction = (upper - lower) / (upper + lower)
    numerator_bound = max(
        (abs(coefficient) for coefficient in numerator_controls), default=Fraction(0)
    )
    reference_volume = _polynomial_integral(constant(1, dimension), cell_kind)
    term: Expression = constant(1, dimension)
    total = Fraction(0)
    order = 0
    while True:
        integral_term = _polynomial_integral(
            multiply(value.numerator, term), cell_kind
        )
        if not isinstance(integral_term, Fraction):
            raise ValueError("Rational mapped measure series requires rational moments.")
        total += integral_term / center
        remainder = (
            reference_volume
            * numerator_bound
            * contraction ** (order + 1)
            / (center * (1 - contraction))
        )
        permitted = max(
            absolute_tolerance,
            relative_tolerance * max(abs(total) - remainder, Fraction(0)),
        )
        if remainder <= permitted:
            lower_value = coefficient_endpoint(total - remainder, upper=False, saturate=True)
            upper_value = coefficient_endpoint(total + remainder, upper=True, saturate=True)
            published = float(total)
            if not math.isfinite(published):
                raise ValueError("Mapped cell measure is not representable in binary64.")
            error = max(published - lower_value, upper_value - published)
            return published, _positive_outward_error(error), remainder == 0
        term = multiply(term, ratio)
        order += 1


def _positive_outward_error(error: float, /) -> float:
    if not math.isfinite(error) or error < 0.0:
        raise ValueError("Mapped measure error must be finite and nonnegative.")
    positive = float(np.nextafter(np.float64(error), np.float64(math.inf)))
    if not math.isfinite(positive):
        raise ValueError("Mapped measure error is not representable in binary64.")
    return positive


def _box_cell_kind(dimension: int, /) -> str:
    match dimension:
        case 1:
            return "interval"
        case 2:
            return "quadrilateral"
        case 3:
            return "hexahedron"
        case _:
            raise ValueError(f"Mapped measure box dimension {dimension} is unsupported.")


def _box_arguments(bounds: _BoxBounds, /) -> tuple[Polynomial, ...]:
    dimension = len(bounds)
    unit = axes(dimension)
    return tuple(
        add(constant(lower, dimension), scale(unit[axis], upper - lower))
        for axis, (lower, upper) in enumerate(bounds)
    )


def _restricted_polynomial(
    polynomial: Polynomial, bounds: _BoxBounds, /
) -> Polynomial:
    dimension = len(bounds)
    restricted = compose(
        polynomial, _box_arguments(bounds), dimension=dimension
    )
    if isinstance(restricted, RationalPolynomial):
        raise RuntimeError("Polynomial Bernstein restriction became rational.")
    return restricted


def _positive_measure_patch(
    numerator: Polynomial,
    denominator: Polynomial,
    jacobian: Polynomial,
    bounds: _BoxBounds,
    /,
) -> _PositiveMeasurePatch | None:
    dimension = len(bounds)
    local_numerator = _restricted_polynomial(numerator, bounds)
    local_denominator = _restricted_polynomial(denominator, bounds)
    local_jacobian = _restricted_polynomial(jacobian, bounds)
    numerator_controls = bernstein_coefficients(local_numerator, "box", dimension)
    denominator_controls = bernstein_coefficients(local_denominator, "box", dimension)
    if not numerator_controls or not denominator_controls:
        return None
    if all(value > 0 for value in denominator_controls):
        orientation = 1
    elif all(value < 0 for value in denominator_controls):
        orientation = -1
    else:
        return None
    normalized_numerator = tuple(orientation * value for value in numerator_controls)
    normalized_denominator = tuple(orientation * value for value in denominator_controls)
    if not all(value > 0 for value in normalized_numerator):
        return None
    numerator_minimum = min(normalized_numerator)
    numerator_maximum = max(normalized_numerator)
    denominator_minimum = min(normalized_denominator)
    denominator_maximum = max(normalized_denominator)
    numerator_center = (numerator_minimum + numerator_maximum) / 2
    denominator_center = (denominator_minimum + denominator_maximum) / 2
    numerator_variation = (
        numerator_maximum - numerator_minimum
    ) / (numerator_maximum + numerator_minimum)
    denominator_variation = (
        denominator_maximum - denominator_minimum
    ) / (denominator_maximum + denominator_minimum)
    numerator_ratio = min(
        coefficient_interval(numerator_variation)[1], Fraction(1)
    )
    denominator_ratio = min(
        coefficient_interval(denominator_variation)[1], Fraction(1)
    )
    normalized_numerator_polynomial = scale(local_numerator, orientation)
    normalized_denominator_polynomial = scale(local_denominator, orientation)
    if isinstance(normalized_numerator_polynomial, RationalPolynomial) or isinstance(
        normalized_denominator_polynomial, RationalPolynomial
    ):
        raise RuntimeError("Polynomial positivity normalization became rational.")
    volume_scale = Fraction(1)
    for lower, upper in bounds:
        volume_scale *= upper - lower
    jacobian_integral = volume_scale * _polynomial_integral(
        local_jacobian, _box_cell_kind(dimension)
    )
    if not isinstance(jacobian_integral, Fraction) or jacobian_integral <= 0:
        return None
    return _PositiveMeasurePatch(
        dimension,
        normalized_numerator_polynomial,
        normalized_denominator_polynomial,
        local_jacobian,
        volume_scale,
        jacobian_integral,
        numerator_center,
        denominator_center,
        numerator_ratio,
        denominator_ratio,
    )


def _split_box(bounds: _BoxBounds, depth: int, /) -> tuple[_BoxBounds, _BoxBounds]:
    dimension = len(bounds)
    widths = tuple(upper - lower for lower, upper in bounds)
    widest = max(widths)
    axis = next(
        (depth + offset) % dimension
        for offset in range(dimension)
        if widths[(depth + offset) % dimension] == widest
    )
    lower, upper = bounds[axis]
    midpoint = (lower + upper) / 2
    first = list(bounds)
    second = list(bounds)
    first[axis] = (lower, midpoint)
    second[axis] = (midpoint, upper)
    return tuple(first), tuple(second)


def _positive_measure_patches(
    numerator: Polynomial,
    denominator: Polynomial,
    jacobian: Polynomial,
    dimension: int,
    /,
) -> tuple[_PositiveMeasurePatch, ...]:
    unit_bounds: _BoxBounds = tuple(
        (Fraction(0), Fraction(1)) for _ in range(dimension)
    )
    pending: list[tuple[_BoxBounds, int]] = [(unit_bounds, 0)]
    patches: list[_PositiveMeasurePatch] = []
    while pending:
        bounds, depth = pending.pop()
        patch = _positive_measure_patch(
            numerator, denominator, jacobian, bounds
        )
        if patch is None or max(
            patch.numerator_ratio, patch.denominator_ratio
        ) > Fraction(1, 2):
            first, second = _split_box(bounds, depth)
            pending.append((second, depth + 1))
            pending.append((first, depth + 1))
        else:
            patches.append(patch)
    return tuple(patches)


def _binomial_polynomial(
    value: Polynomial,
    center: ExactScalar,
    order: int,
    dimension: int,
    /,
    *,
    inverse_square_root: bool,
) -> Polynomial:
    relative = scale(
        add(value, scale(constant(center, dimension), -1)),
        1 / center,
    )
    if isinstance(relative, RationalPolynomial):
        raise RuntimeError("Polynomial binomial expansion became rational.")
    total = constant(0, dimension)
    term = constant(1, dimension)
    coefficient = Fraction(1)
    for index in range(order + 1):
        total = add(total, scale(term, coefficient))
        if index < order:
            step = (
                Fraction(-1, 2) - index
                if inverse_square_root
                else Fraction(1, 2) - index
            )
            coefficient = coefficient * step / (index + 1)
            term = multiply(term, relative)
            if isinstance(term, RationalPolynomial):
                raise RuntimeError("Polynomial binomial term became rational.")
    if isinstance(total, RationalPolynomial):
        raise RuntimeError("Polynomial binomial sum became rational.")
    return total


def _positive_square_root_bounds(value: ExactScalar, /) -> tuple[Fraction, Fraction]:
    if value <= 0:
        raise ValueError("Square-root enclosure requires a positive exact value.")
    bits = 80
    while True:
        lower, upper = coefficient_interval(value, bits=bits)
        if lower > 0:
            return (
                fraction_sqrt_interval(lower)[0],
                fraction_sqrt_interval(upper)[1],
            )
        bits *= 2


def _integrate_positive_measure_patch(
    patch: _PositiveMeasurePatch,
    absolute_tolerance: Fraction,
    box_kind: str,
    /,
) -> tuple[Fraction, Fraction, bool]:
    factor_squared = patch.numerator_center / patch.denominator_center
    factor_lower, factor_upper = _positive_square_root_bounds(factor_squared)
    numerator_ratio = patch.numerator_ratio
    denominator_ratio = patch.denominator_ratio
    order = 0
    while True:
        numerator_tail = numerator_ratio ** (order + 1) / (1 - numerator_ratio)
        denominator_tail = denominator_ratio ** (order + 1) / (
            1 - denominator_ratio
        )
        remainder = factor_upper * patch.jacobian_integral * (
            numerator_tail / (1 - denominator_ratio)
            + denominator_tail / (1 - numerator_ratio)
        )
        if remainder <= absolute_tolerance:
            break
        order += 1
    dimension = patch.dimension
    numerator_series = _binomial_polynomial(
        patch.numerator,
        patch.numerator_center,
        order,
        dimension,
        inverse_square_root=False,
    )
    denominator_series = _binomial_polynomial(
        patch.denominator,
        patch.denominator_center,
        order,
        dimension,
        inverse_square_root=True,
    )
    approximation_expression = multiply(
        multiply(numerator_series, denominator_series),
        patch.jacobian,
    )
    if isinstance(approximation_expression, RationalPolynomial):
        raise RuntimeError("Polynomial measure approximation became rational.")
    approximation = _polynomial_integral(
        approximation_expression, box_kind
    ) * patch.volume_scale
    approximation_lower, approximation_upper = coefficient_interval(approximation)
    products = (
        factor_lower * approximation_lower,
        factor_lower * approximation_upper,
        factor_upper * approximation_lower,
        factor_upper * approximation_upper,
    )
    lower = min(products) - remainder
    upper = max(products) + remainder
    exact = (
        remainder == 0
        and factor_lower == factor_upper
        and approximation_lower == approximation_upper
    )
    return lower, upper, exact


def _integrate_embedded_measure(
    value: Expression,
    cell_kind: str,
    absolute_tolerance: Fraction,
    /,
) -> tuple[float, float, bool]:
    dimension = reference_cell_topology(cell_kind).dimension
    box_arguments, box_jacobian = _reference_box_map(cell_kind)
    mapped = compose_expression(value, box_arguments)
    if isinstance(mapped, RationalPolynomial):
        numerator = dict(mapped.numerator)
        denominator = dict(mapped.denominator)
    else:
        numerator = dict(mapped)
        denominator = constant(1, dimension)
    patches = _positive_measure_patches(
        numerator, denominator, box_jacobian, dimension
    )
    box_kind = _box_cell_kind(dimension)
    reference_volume = _polynomial_integral(box_jacobian, box_kind)
    if not isinstance(reference_volume, Fraction) or reference_volume <= 0:
        raise ValueError("Embedded reference-cell measure must be positive.")
    lower = Fraction(0)
    upper = Fraction(0)
    exact = True
    for patch in patches:
        local_tolerance = (
            absolute_tolerance * patch.jacobian_integral / reference_volume
        )
        patch_lower, patch_upper, patch_exact = _integrate_positive_measure_patch(
            patch, local_tolerance, box_kind
        )
        lower += patch_lower
        upper += patch_upper
        exact = exact and patch_exact
    midpoint = (lower + upper) / 2
    published = float(midpoint)
    if not math.isfinite(published):
        raise ValueError("Mapped cell measure is not representable in binary64.")
    lower_value = coefficient_endpoint(lower, upper=False, saturate=True)
    upper_value = coefficient_endpoint(upper, upper=True, saturate=True)
    error = max(published - lower_value, upper_value - published)
    return published, _positive_outward_error(error), exact


def _publish_integral(value: ExactScalar, /) -> tuple[float, float]:
    published = float(value)
    if not math.isfinite(published):
        raise ValueError("Mapped cell measure is not representable in binary64.")
    lower = coefficient_endpoint(value, upper=False, saturate=True)
    upper = coefficient_endpoint(value, upper=True, saturate=True)
    return published, max(published - lower, upper - published)


@checked
def mapped_geometry_cells(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[MappedGeometryCell, ...]:
    elements, routes, _ = geometry.resolve(mesh)
    source_bank = geometry.source_coordinates()
    cells: list[MappedGeometryCell] = []
    for element, route in zip(elements, routes, strict=True):
        for cell_row, row in enumerate(np.asarray(route, dtype=np.int64)):
            cells.append(
                (
                    element,
                    tuple(source_bank[int(index)] for index in row),
                    cell_row,
                )
            )
    return tuple(cells)


@checked
def certified_cell_measures(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> MappedCellMeasures:
    """Integrate certified mapped volume and embedded measures per cell.

    Polynomial volume determinants use exact reference moments. Rational volume
    determinants use a Bernstein-positive geometric series. Embedded measures
    integrate the square root of an exactly formed Gram determinant with
    positive Bernstein enclosures and a binomial-series remainder.
    """
    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Mapped cell measures require a CellMesh and CellGeometrySpec.")
    absolute = positive_finite_float(absolute_tolerance, "absolute_tolerance")
    relative = finite_real_scalar(relative_tolerance, "relative_tolerance")
    if relative < 0.0:
        raise ValueError("relative_tolerance must be nonnegative.")
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    if not validity.all_certified:
        raise ValueError("Mapped cell measures require every cell to be certified valid.")
    cells = mapped_geometry_cells(mesh, geometry)
    values = np.empty((len(cells),), dtype=np.float64)
    errors = np.empty_like(values)
    exact = True
    absolute_per_cell = Fraction(absolute) / max(len(cells), 1)
    relative_fraction = Fraction(relative)
    for flattened_row, (element, local, cell_row) in enumerate(cells):
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError("Mapped cell measures require scalar coordinate elements.")
        scalar = element
        dimension = scalar.topological_dimension
        jacobian = _mapped_jacobian(element, local, cell_row=cell_row)
        ambient_dimension = len(jacobian)
        if ambient_dimension == dimension:
            integral = _determinant(jacobian, dimension)
            integration_cell_kind = scalar.cell_kind
            if scalar.cell_kind == "pyramid":
                integral = _pyramid_weighted_density(integral)
                integration_cell_kind = "hexahedron"
            if isinstance(integral, RationalPolynomial):
                (
                    values[flattened_row],
                    errors[flattened_row],
                    cell_exact,
                ) = _integrate_rational(
                    integral,
                    integration_cell_kind,
                    absolute_per_cell,
                    relative_fraction,
                )
                exact = exact and cell_exact
            else:
                exact_value = _polynomial_integral(integral, integration_cell_kind)
                (
                    values[flattened_row],
                    errors[flattened_row],
                ) = _publish_integral(exact_value)
        elif ambient_dimension > dimension and dimension in (1, 2):
            gram = _gram_density(jacobian, dimension)
            (
                values[flattened_row],
                errors[flattened_row],
                cell_exact,
            ) = _integrate_embedded_measure(
                gram, scalar.cell_kind, absolute_per_cell
            )
            exact = exact and cell_exact
        else:
            raise ValueError(
                "Mapped cell measures require volume maps or embedded curve/surface maps."
            )
    if (
        np.any(values <= errors)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(errors))
    ):
        raise ValueError("Mapped cell measures must have positive finite enclosures.")
    values.setflags(write=False)
    errors.setflags(write=False)
    return MappedCellMeasures(values, errors, exact)


__all__ = [
    "MappedCellMeasures",
    "MappedGeometryCell",
    "certified_cell_measures",
    "mapped_coordinate_expressions",
    "mapped_density_expression",
    "mapped_geometry_cells",
]
