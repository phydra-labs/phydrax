"""Exact reference-form moments for polynomial and rational cell charts.

The transfer action is built in the target reference chart: source forms are
composed with the exact target-to-source map, pulled back by its exterior
minors, paired with the target entity test forms, and integrated on the actual
reference entity. Rational integrals use signed Bernstein enclosures and retain
an absolute error bound for every local matrix entry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations, product
from math import comb, factorial, fsum, prod
from typing import Literal, Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ...linalg._exact import solve_exact
from ...typing import ConvertibleToArray
from .. import _coordinate_enclosure as algebra
from .._exact_algebraic_geometry import coefficient_endpoint
from .._nested_reference import NestedCellReferenceMap, NestedSimplexMap
from .._nested_reference_types import reference_chart_arguments_identity
from ._form_elements import (
    _exact_generators,
    _form_entity_chart,
    FormBasis,
)
from ._reference import FiniteElementSpec


_ExactScalar = Fraction | algebra.RadicalElement
_Polynomial = Mapping[tuple[int, ...], _ExactScalar]
_Expression = _Polynomial | algebra.RationalPolynomial
type _CellOrientationMatrix = (
    ConvertibleToArray | Sequence[Sequence[Fraction | int | float]]
)
type _Tetrahedron = tuple[
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
]


@dataclass(frozen=True, slots=True)
class _PreparedReferenceChartOrientation:
    family: str
    argument_identity: tuple[object, ...]
    sign: int

    def require(
        self,
        arguments: tuple[_Expression, ...],
        family: str,
        /,
    ) -> int:
        identity = reference_chart_arguments_identity((arguments,))
        if (
            identity is None
            or self.family != family
            or self.argument_identity != identity
        ):
            raise ValueError(
                "Prepared chart orientation is bound to a different exact map."
            )
        return self.sign


_OrientationCache = dict[
    tuple[str, tuple[object, ...]], _PreparedReferenceChartOrientation
]


def _expression_for_algebra(value: _Expression, /) -> algebra.Expression:
    if isinstance(value, algebra.RationalPolynomial) or isinstance(value, dict):
        return value
    return dict(value)


def _evaluate_expression(
    value: _Expression, point: tuple[Fraction | algebra.RadicalElement, ...], /
) -> _ExactScalar:
    return algebra.evaluate_expression(_expression_for_algebra(value), point)


@runtime_checkable
class _IntegrationPolicy(Protocol):
    relative_tolerance: float
    absolute_tolerance: float
    policy_id: str


if TYPE_CHECKING:
    from ...geometry._common_root_volume_overlap import CurvedOverlapIntegrationPolicy
else:
    CurvedOverlapIntegrationPolicy = _IntegrationPolicy


@dataclass(frozen=True, slots=True)
class MomentIntegralMatrix:
    """A local moment action and certified entrywise absolute errors."""

    values: npt.NDArray[np.float64]
    errors: npt.NDArray[np.float64]
    exact: npt.NDArray[np.bool_]
    enclosed: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        errors = np.array(self.errors, dtype=np.float64, copy=True)
        exact = np.array(self.exact, dtype=np.bool_, copy=True)
        enclosed = np.array(self.enclosed, dtype=np.bool_, copy=True)
        if (
            values.ndim != 2
            or errors.shape != values.shape
            or exact.shape != values.shape
            or enclosed.shape != values.shape
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(errors))
            or np.any(errors < 0.0)
            or np.any(exact == enclosed)
        ):
            raise ValueError(
                "Moment integral matrices need aligned finite values and evidence."
            )
        for array in (values, errors, exact, enclosed):
            array.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "exact", exact)
        object.__setattr__(self, "enclosed", enclosed)

    @property
    def exact_count(self) -> int:
        return int(np.count_nonzero(self.exact))

    @property
    def enclosed_count(self) -> int:
        return int(np.count_nonzero(self.enclosed))


def _fraction_matrix(values: _CellOrientationMatrix, /) -> list[list[Fraction]]:
    array = (
        np.asarray(values)
        if isinstance(values, np.ndarray) or hasattr(values, "dtype")
        else np.asarray(values, dtype=object)
    )
    if array.ndim != 2:
        raise ValueError("Cell orientation matrices must be finite real matrices.")
    result = []
    for row in array:
        converted = []
        for value in row:
            if isinstance(value, Fraction):
                converted.append(value)
            elif isinstance(value, (int, np.integer, bool, np.bool_)):
                converted.append(Fraction(int(value)))
            elif isinstance(value, (float, np.floating)):
                scalar = float(value)
                if not np.isfinite(scalar):
                    raise ValueError(
                        "Cell orientation matrices must be finite real matrices."
                    )
                converted.append(Fraction(scalar))
            else:
                raise TypeError(
                    "Cell orientation matrices require exact rational entries."
                )
        result.append(converted)
    return result


def _affine_arguments(
    origin: npt.ArrayLike, jacobian: npt.ArrayLike, variables: int, /
) -> tuple[algebra.Polynomial, ...]:
    offset = np.asarray(origin, dtype=np.float64)
    matrix = np.asarray(jacobian, dtype=np.float64)
    if (
        offset.ndim != 1
        or matrix.shape != (offset.size, variables)
        or not np.all(np.isfinite(offset))
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("Reference entity chart has inconsistent finite extents.")
    variables_ = algebra.axes(variables) if variables else ()
    result = []
    for axis, value in enumerate(offset):
        terms = tuple(
            algebra.scale(variables_[column], Fraction(float(matrix[axis, column])))
            for column in range(variables)
        )
        result.append(
            algebra.add(
                algebra.constant(Fraction(float(value)), variables),
                algebra.sum_polynomials(terms, variables),
            )
        )
    return tuple(result)


def _reference_arguments(
    reference_map: NestedSimplexMap | NestedCellReferenceMap | None,
    dimension: int,
    /,
) -> tuple[_Expression, ...]:
    if reference_map is None:
        return algebra.axes(dimension)
    if isinstance(reference_map, NestedCellReferenceMap):
        if reference_map.arguments is None:
            raise ValueError("Exact reference-cell chart arguments are unavailable.")
        if len(reference_map.arguments) != dimension:
            raise ValueError("Exact reference-cell chart has the wrong source dimension.")
        return tuple(_expression_for_algebra(value) for value in reference_map.arguments)
    if not isinstance(reference_map, NestedSimplexMap):
        raise TypeError("Reference moments require an exact nested reference map.")
    vertices = reference_map.vertices
    if (
        reference_map.dimension != dimension
        or len(vertices) != dimension + 1
        or any(len(vertex) != dimension for vertex in vertices)
    ):
        raise ValueError("Nested simplex reference chart has inconsistent corners.")
    variables = algebra.axes(dimension)
    return tuple(
        algebra.add(
            algebra.constant(vertices[0][axis], dimension),
            algebra.sum_polynomials(
                tuple(
                    algebra.scale(
                        variables[column],
                        vertices[column + 1][axis] - vertices[0][axis],
                    )
                    for column in range(dimension)
                ),
                dimension,
            ),
        )
        for axis in range(dimension)
    )


def _compose(
    expression: _Expression, arguments: tuple[_Expression, ...], /
) -> _Expression:
    return algebra.compose_expression(
        _expression_for_algebra(expression),
        tuple(_expression_for_algebra(argument) for argument in arguments),
    )


def _determinant(
    matrix: tuple[tuple[_Expression, ...], ...], variables: int, /
) -> _Expression:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Exterior chart minors require square submatrices.")
    if size == 0:
        return algebra.constant(1, variables)
    row_numerators: list[tuple[Mapping[tuple[int, ...], _ExactScalar], ...]] = []
    determinant_denominator: Mapping[tuple[int, ...], _ExactScalar] = algebra.constant(
        1, variables
    )
    unit = algebra.constant(1, variables)
    for row in matrix:
        denominators = tuple(
            value.denominator if isinstance(value, algebra.RationalPolynomial) else unit
            for value in row
        )
        if all(value == denominators[0] for value in denominators[1:]):
            row_denominator = denominators[0]
            numerator_row = tuple(
                value.numerator
                if isinstance(value, algebra.RationalPolynomial)
                else value
                for value in row
            )
        else:
            row_denominator = unit
            for denominator in denominators:
                row_denominator = algebra.multiply(row_denominator, denominator)
            numerator_values = []
            for column, value in enumerate(row):
                numerator = (
                    value.numerator
                    if isinstance(value, algebra.RationalPolynomial)
                    else value
                )
                other_denominators = unit
                for other_column, denominator in enumerate(denominators):
                    if other_column != column:
                        other_denominators = algebra.multiply(
                            other_denominators, denominator
                        )
                numerator_values.append(algebra.multiply(numerator, other_denominators))
            numerator_row = tuple(numerator_values)
        row_numerators.append(numerator_row)
        determinant_denominator = algebra.multiply(
            determinant_denominator, row_denominator
        )
    terms: list[Mapping[tuple[int, ...], _ExactScalar]] = []
    for permutation in permutations(range(size)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(size)
            for right in range(left + 1, size)
        )
        product: Mapping[tuple[int, ...], _ExactScalar] = algebra.constant(
            -1 if inversions % 2 else 1, variables
        )
        for row, column in enumerate(permutation):
            product = algebra.multiply(product, row_numerators[row][column])
        terms.append(product)
    numerator = algebra.sum_polynomials(terms, variables)
    return (
        numerator
        if determinant_denominator == unit
        else algebra.RationalPolynomial(numerator, determinant_denominator)
    )


def _wedge_sign(first: tuple[int, ...], second: tuple[int, ...], /) -> int:
    return -1 if sum(a > b for a in first for b in second) % 2 else 1


def _basis_components(basis: FormBasis, /) -> tuple[tuple[_Expression, ...], ...]:
    coefficients = basis.exact_coefficients
    blades = comb(basis.dimension, basis.form_degree)
    if (
        len(coefficients) != len(basis.exponents)
        or any(len(row) != blades for row in coefficients)
        or any(
            len(dofs) != basis.local_dof_count
            for monomial in coefficients
            for dofs in monomial
        )
    ):
        raise ValueError(
            "Finite-element form basis coefficients have inconsistent extents."
        )
    result = []
    for dof in range(basis.local_dof_count):
        components = []
        for component in range(blades):
            polynomial = {
                exponent: coefficients[monomial][component][dof]
                for monomial, exponent in enumerate(basis.exponents)
                if coefficients[monomial][component][dof]
            }
            components.append(polynomial)
        result.append(tuple(components))
    return tuple(result)


def _simplex_test_forms(
    basis: FormBasis,
    entity: tuple[int, ...],
    entity_dimension: int,
    entity_dofs: tuple[int, ...],
    /,
) -> dict[int, tuple[_Expression, ...]]:
    degree = (
        basis.order
        + basis.form_degree
        - entity_dimension
        - int(basis.family == "trimmed")
    )
    if degree < 0:
        raise ValueError("A declared simplex moment has negative test polynomial degree.")
    exponents, coefficients, labels = _exact_generators(
        entity_dimension,
        entity_dimension - basis.form_degree,
        degree,
        basis.family == "full",
    )
    if len(coefficients[0][0]) != len(labels):
        raise ValueError("Simplex moment generators and labels disagree.")
    label_rows = {label: index for index, label in enumerate(labels)}
    test_blades = tuple(
        combinations(range(entity_dimension), entity_dimension - basis.form_degree)
    )
    result = {}
    for dof in entity_dofs:
        _, alpha, sigma = basis.dof_labels[dof]
        if any(vertex not in entity for vertex in sigma):
            raise ValueError(
                "Simplex DOF test chart references a vertex outside its entity."
            )
        local_sigma = tuple(entity.index(vertex) for vertex in sigma)
        test = label_rows.get((alpha, local_sigma))
        if test is None:
            raise ValueError("A simplex DOF has no matching complementary test form.")
        result[dof] = tuple(
            {
                exponent: coefficients[monomial][component][test]
                for monomial, exponent in enumerate(exponents)
                if coefficients[monomial][component][test]
            }
            for component in range(len(test_blades))
        )
    return result


def _target_test_forms(
    basis: FormBasis,
    entity: tuple[int, ...],
    entity_dimension: int,
    entity_dofs: tuple[int, ...],
    free_axes: tuple[int, ...],
    /,
) -> tuple[tuple[tuple[int, ...], ...], dict[int, tuple[_Expression, ...]]]:
    blades = tuple(combinations(range(entity_dimension), basis.form_degree))
    if not entity_dofs:
        return blades, {}
    if basis.family == "tensor-trimmed":
        if len(free_axes) != entity_dimension:
            raise ValueError(
                "Tensor moment chart does not retain every free entity axis."
            )
        result = {}
        complementary_blades = tuple(
            combinations(range(entity_dimension), entity_dimension - basis.form_degree)
        )
        for dof in entity_dofs:
            _, index, blade = basis.dof_labels[dof]
            if not set(blade).issubset(free_axes):
                raise ValueError("Tensor moment blade is not tangent to its entity.")
            local_blade = tuple(free_axes.index(axis) for axis in blade)
            exponent = tuple(
                int(index[axis]) - int(axis not in blade) for axis in free_axes
            )
            if any(value < 0 for value in exponent):
                raise ValueError("Tensor moment mode has a negative test exponent.")
            if len(local_blade) != basis.form_degree:
                raise ValueError("Tensor moment blade has the wrong form degree.")
            complement = tuple(
                axis for axis in range(entity_dimension) if axis not in local_blade
            )
            component = complementary_blades.index(complement)
            row = [{} for _ in complementary_blades]
            row[component] = {exponent: Fraction(_wedge_sign(local_blade, complement))}
            result[dof] = tuple(row)
        return blades, result
    if basis.family == "full" and basis.order == 0:
        if entity_dimension != basis.dimension:
            raise ValueError("Constant full forms carry only top-cell moments.")
        result = {}
        complementary_blades = tuple(
            combinations(range(entity_dimension), entity_dimension - basis.form_degree)
        )
        source_blades = tuple(combinations(range(entity_dimension), basis.form_degree))
        for dof in entity_dofs:
            _, _, encoded_blade = basis.dof_labels[dof]
            source_blade = tuple(axis - 1 for axis in encoded_blade)
            if source_blade not in source_blades:
                raise ValueError("Constant full-form DOF has an unknown component blade.")
            complement = tuple(
                axis for axis in range(entity_dimension) if axis not in source_blade
            )
            test_component = complementary_blades.index(complement)
            row = [{} for _ in complementary_blades]
            row[test_component] = {
                (0,) * entity_dimension: Fraction(_wedge_sign(source_blade, complement))
            }
            result[dof] = tuple(row)
        return blades, result
    return blades, _simplex_test_forms(basis, entity, entity_dimension, entity_dofs)


def _polynomial_expression(expression: _Expression, /) -> _Expression:
    if not isinstance(expression, algebra.RationalPolynomial):
        return expression
    denominator = expression.denominator
    exponent = next(iter(denominator), ())
    if len(denominator) == 1 and not any(exponent):
        divisor = denominator[exponent]
        if not divisor:
            raise ZeroDivisionError(
                "A rational form moment has a zero constant denominator."
            )
        return {
            key: coefficient / divisor
            for key, coefficient in expression.numerator.items()
            if coefficient
        }
    return expression


def _standard_simplex(dimension: int, /) -> tuple[tuple[Fraction, ...], ...]:
    return (
        (Fraction(0),) * dimension,
        *tuple(
            tuple(Fraction(int(axis == coordinate)) for axis in range(dimension))
            for coordinate in range(dimension)
        ),
    )


def _cube_simplices(dimension: int, /) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    result = []
    for order in permutations(range(dimension)):
        point = [0] * dimension
        path = [tuple(Fraction(value) for value in point)]
        for axis in order:
            point[axis] = 1
            path.append(tuple(Fraction(value) for value in point))
        result.append(tuple(path))
    return tuple(result)


def _entity_simplices(
    entity_dimension: int, family: str, /
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    if entity_dimension == 0:
        return ()
    if family != "tensor-trimmed":
        return (_standard_simplex(entity_dimension),)
    return _cube_simplices(entity_dimension)


def _simplex_chart_arguments(
    vertices: tuple[tuple[Fraction, ...], ...], variables: int, /
) -> tuple[algebra.Polynomial, ...]:
    dimension = len(vertices) - 1
    if (
        dimension != variables
        or not vertices
        or any(len(row) != dimension for row in vertices)
    ):
        raise ValueError("Exact simplex chart has inconsistent dimensions.")
    origin = np.asarray([float(value) for value in vertices[0]], dtype=np.float64)
    jacobian = np.asarray(
        [
            [
                float(vertices[column + 1][axis] - vertices[0][axis])
                for column in range(dimension)
            ]
            for axis in range(dimension)
        ],
        dtype=np.float64,
    )
    return _affine_arguments(origin, jacobian, variables)


def _subdivide_simplex(
    vertices: tuple[tuple[Fraction, ...], ...], /
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    dimension = len(vertices) - 1
    if dimension == 1:
        first, second = vertices
        middle = tuple((a + b) / 2 for a, b in zip(first, second, strict=True))
        return ((first, middle), (middle, second))
    if dimension == 2:
        a, b, c = vertices
        ab = tuple((x + y) / 2 for x, y in zip(a, b, strict=True))
        ac = tuple((x + y) / 2 for x, y in zip(a, c, strict=True))
        bc = tuple((x + y) / 2 for x, y in zip(b, c, strict=True))
        return ((a, ab, ac), (ab, b, bc), (ac, bc, c), (ab, ac, bc))
    if dimension == 3:
        a, b, c, d = vertices
        ab = tuple((x + y) / 2 for x, y in zip(a, b, strict=True))
        ac = tuple((x + y) / 2 for x, y in zip(a, c, strict=True))
        ad = tuple((x + y) / 2 for x, y in zip(a, d, strict=True))
        bc = tuple((x + y) / 2 for x, y in zip(b, c, strict=True))
        bd = tuple((x + y) / 2 for x, y in zip(b, d, strict=True))
        cd = tuple((x + y) / 2 for x, y in zip(c, d, strict=True))
        return (
            (a, ab, ac, ad),
            (b, ab, bc, bd),
            (c, ac, bc, cd),
            (d, ad, bd, cd),
            (ab, ac, ad, cd),
            (ab, ac, bc, cd),
            (ab, bc, bd, cd),
            (ab, ad, bd, cd),
        )
    if dimension >= 4:
        result = []
        for order in permutations(range(dimension + 1)):
            child = []
            for count in range(1, dimension + 2):
                selected = order[:count]
                child.append(
                    tuple(
                        sum(
                            (vertices[index][axis] for index in selected),
                            Fraction(0),
                        )
                        / count
                        for axis in range(dimension)
                    )
                )
            result.append(tuple(child))
        return tuple(result)
    raise ValueError("Simplex subdivision requires positive dimension.")


def _simplex_has_binary64_interior(
    vertices: tuple[tuple[Fraction, ...], ...],
    rule_points: np.ndarray | None = None,
    /,
) -> bool:
    rounded = np.asarray([[float(value) for value in point] for point in vertices])
    if not np.all(np.isfinite(rounded)) or np.unique(rounded, axis=0).shape[0] != len(
        vertices
    ):
        return False
    center = tuple(
        sum((point[axis] for point in vertices), Fraction(0)) / len(vertices)
        for axis in range(len(vertices[0]))
    )
    rounded_center = np.asarray([float(value) for value in center])
    if not any(np.array_equal(rounded_center, point) for point in rounded):
        return True
    if rule_points is None:
        return False
    dimension = len(vertices) - 1
    origin = np.asarray([float(value) for value in vertices[0]], dtype=np.float64)
    frame = np.asarray(
        [
            [
                float(vertices[column + 1][axis] - vertices[0][axis])
                for column in range(dimension)
            ]
            for axis in range(dimension)
        ],
        dtype=np.float64,
    )
    samples = origin[None, :] + np.asarray(rule_points) @ frame.T
    return bool(
        np.all(np.isfinite(samples))
        and any(
            not any(np.array_equal(sample, vertex) for vertex in rounded)
            for sample in samples
        )
    )


def _exact_sign(value: _ExactScalar, /) -> int:
    if isinstance(value, Fraction):
        return (value > 0) - (value < 0)
    return value.sign()


def _strict_expression_sign_on_simplex(
    expression: _Expression,
    vertices: tuple[tuple[Fraction, ...], ...],
    /,
) -> int | None:
    dimension = len(vertices) - 1
    local = _compose(expression, _simplex_chart_arguments(vertices, dimension))
    if isinstance(local, algebra.RationalPolynomial):
        numerator, denominator = local.numerator, local.denominator
    else:
        numerator, denominator = local, None
    numerator_controls = algebra.bernstein_coefficients(numerator, "simplex", dimension)
    numerator_signs = tuple(_exact_sign(value) for value in numerator_controls)
    if not numerator_signs or not (
        all(value > 0 for value in numerator_signs)
        or all(value < 0 for value in numerator_signs)
    ):
        return None
    numerator_sign = numerator_signs[0]
    if denominator is None:
        return numerator_sign
    denominator_controls = algebra.bernstein_coefficients(
        denominator, "simplex", dimension
    )
    denominator_signs = tuple(_exact_sign(value) for value in denominator_controls)
    if not denominator_signs or not (
        all(value > 0 for value in denominator_signs)
        or all(value < 0 for value in denominator_signs)
    ):
        return None
    return numerator_sign * denominator_signs[0]


def _cell_simplex_cover(
    dimension: int, family: str, /
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    if family == "tensor-trimmed":
        return _cube_simplices(dimension)
    return (_standard_simplex(dimension),)


def _orientation_sign(
    arguments: tuple[_Expression, ...], family: str, /, *, required: bool
) -> int:
    if not required:
        return 1
    dimension = len(arguments)
    jacobian = tuple(
        tuple(algebra.derivative(value, axis) for axis in range(dimension))
        for value in arguments
    )
    determinant = _determinant(jacobian, dimension)
    sign: int | None = None
    stack = list(_cell_simplex_cover(dimension, family))
    while stack:
        simplex = stack.pop()
        local_sign = _strict_expression_sign_on_simplex(determinant, simplex)
        if local_sign is not None:
            if sign is None:
                sign = local_sign
            elif sign != local_sign:
                raise ValueError(
                    "Twisted reference chart changes orientation or is singular."
                )
            continue
        if not _simplex_has_binary64_interior(simplex):
            raise ValueError(
                "Twisted reference chart orientation is unresolved at binary64 cell resolution."
            )
        stack.extend(_subdivide_simplex(simplex))
    if sign is None:
        raise ValueError(
            "Twisted reference chart has no certified full-cell orientation."
        )
    return sign


def _prepare_reference_chart_orientation(
    reference_map: NestedSimplexMap | NestedCellReferenceMap | None,
    dimension: int,
    family: str,
    cache: _OrientationCache | None = None,
    /,
) -> _PreparedReferenceChartOrientation:
    arguments = _reference_arguments(reference_map, dimension)
    identity = reference_chart_arguments_identity((arguments,))
    if identity is None:
        raise ValueError("Exact reference arguments must have a prepared identity.")
    key = (family, identity)
    if cache is not None:
        prepared = cache.get(key)
        if prepared is not None:
            return prepared
    prepared = _PreparedReferenceChartOrientation(
        family,
        identity,
        _orientation_sign(arguments, family, required=True),
    )
    if cache is not None:
        cache[key] = prepared
    return prepared


def _as_tetrahedron(vertices: tuple[tuple[Fraction, ...], ...], /) -> _Tetrahedron:
    if len(vertices) != 4 or any(len(point) != 3 for point in vertices):
        raise ValueError("Curved volume form moments require exact tetrahedron pieces.")
    return (
        (vertices[0][0], vertices[0][1], vertices[0][2]),
        (vertices[1][0], vertices[1][1], vertices[1][2]),
        (vertices[2][0], vertices[2][1], vertices[2][2]),
        (vertices[3][0], vertices[3][1], vertices[3][2]),
    )


def _integral_center_and_error(exact: _ExactScalar, /) -> tuple[float, float]:
    lower = coefficient_endpoint(exact, upper=False, saturate=True)
    upper = coefficient_endpoint(exact, upper=True, saturate=True)
    center = (lower + upper) * 0.5
    error = max(center - lower, upper - center)
    if error:
        error = float(np.nextafter(error, np.inf))
    if not np.isfinite(center) or not np.isfinite(error):
        raise ValueError("Exact reference form moment has no finite binary64 enclosure.")
    return center, error


def _simplex_determinant(vertices: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    dimension = len(vertices) - 1
    if dimension < 1 or any(len(point) != dimension for point in vertices):
        raise ValueError("Reference simplex corners have inconsistent dimensions.")
    matrix = tuple(
        tuple(
            vertices[column + 1][axis] - vertices[0][axis] for column in range(dimension)
        )
        for axis in range(dimension)
    )
    result = Fraction(0)
    for permutation in permutations(range(dimension)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(dimension)
            for right in range(left + 1, dimension)
        )
        term = Fraction(-1 if inversions % 2 else 1)
        for row, column in enumerate(permutation):
            term *= matrix[row][column]
        result += term
    return result


def _exact_polynomial_simplex_integral(
    expression: _Expression,
    vertices: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[float, float] | None:
    dimension = len(vertices) - 1
    local = _polynomial_expression(
        _compose(expression, _simplex_chart_arguments(vertices, dimension))
    )
    if isinstance(local, algebra.RationalPolynomial):
        return None
    determinant = abs(_simplex_determinant(vertices))
    total: _ExactScalar = Fraction(0)
    for exponent, coefficient in local.items():
        if len(exponent) != dimension:
            raise ValueError("Polynomial FE moment exponent has the wrong dimension.")
        moment = Fraction(
            prod(factorial(power) for power in exponent),
            factorial(dimension + sum(exponent)),
        )
        total += coefficient * moment
    return _integral_center_and_error(total * determinant)


def _multiindices(dimension: int, total: int, /) -> tuple[tuple[int, ...], ...]:
    if dimension == 1:
        return ((total,),)
    return tuple(
        (first, *rest)
        for first in range(total + 1)
        for rest in _multiindices(dimension - 1, total - first)
    )


def _submultiindices(index: tuple[int, ...], /) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(values) for values in product(*(range(power + 1) for power in index))
    )


def _polynomial_derivative_bank(
    polynomial: _Polynomial, dimension: int, order: int, /
) -> dict[tuple[int, ...], _Polynomial]:
    zero = (0,) * dimension
    derivatives: dict[tuple[int, ...], _Polynomial] = {zero: polynomial}
    for total in range(1, order + 1):
        for multiindex in _multiindices(dimension, total):
            axis = next(axis for axis, power in enumerate(multiindex) if power)
            parent = tuple(
                power - int(index == axis) for index, power in enumerate(multiindex)
            )
            derivatives[multiindex] = algebra.derivative(derivatives[parent], axis)
    return derivatives


def _polynomial_power_table(
    polynomials: Sequence[_Polynomial],
    point: tuple[Fraction, ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            point[axis] ** power
            for power in range(
                max(
                    (
                        exponent[axis]
                        for polynomial in polynomials
                        for exponent in polynomial
                    ),
                    default=0,
                )
                + 1
            )
        )
        for axis in range(len(point))
    )


def _evaluate_polynomial_with_powers(
    polynomial: _Polynomial,
    powers: tuple[tuple[Fraction, ...], ...],
    /,
) -> _ExactScalar:
    total: _ExactScalar = Fraction(0)
    for exponent, coefficient in polynomial.items():
        term = coefficient
        for axis, power in enumerate(exponent):
            term *= powers[axis][power]
        total += term
    return total


def _exact_rational_derivative_values(
    expression: algebra.RationalPolynomial,
    point: tuple[Fraction, ...],
    order: int,
    /,
    *,
    numerator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
    denominator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
) -> dict[tuple[int, ...], _ExactScalar]:
    dimension = len(point)
    if numerator_derivatives is None:
        numerator_derivatives = _polynomial_derivative_bank(
            expression.numerator, dimension, order
        )
    if denominator_derivatives is None:
        denominator_derivatives = _polynomial_derivative_bank(
            expression.denominator, dimension, order
        )
    powers = _polynomial_power_table(
        (*numerator_derivatives.values(), *denominator_derivatives.values()), point
    )
    zero = (0,) * dimension
    denominator_value = _evaluate_polynomial_with_powers(
        denominator_derivatives[zero], powers
    )
    if not denominator_value:
        raise ValueError(
            "Rational derivative denominator vanishes at its expansion point."
        )
    values: dict[tuple[int, ...], _ExactScalar] = {}
    for total in range(order + 1):
        for multiindex in _multiindices(dimension, total):
            value = _evaluate_polynomial_with_powers(
                numerator_derivatives[multiindex], powers
            )
            for beta in _submultiindices(multiindex):
                if beta == zero:
                    continue
                parent = tuple(
                    power - derivative
                    for power, derivative in zip(multiindex, beta, strict=True)
                )
                coefficient = prod(
                    comb(power, derivative)
                    for power, derivative in zip(multiindex, beta, strict=True)
                )
                value -= (
                    coefficient
                    * _evaluate_polynomial_with_powers(
                        denominator_derivatives[beta], powers
                    )
                    * values[parent]
                )
            values[multiindex] = value / denominator_value
    return values


def _interval_sum(
    first: tuple[float, float], second: tuple[float, float], /
) -> tuple[float, float]:
    lower, upper = first[0] + second[0], first[1] + second[1]
    if np.isnan(lower) or np.isnan(upper):
        return -np.inf, np.inf
    return (
        float(np.nextafter(lower, -np.inf)),
        float(np.nextafter(upper, np.inf)),
    )


def _interval_product(
    first: tuple[float, float], second: tuple[float, float], /
) -> tuple[float, float]:
    products = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    if any(np.isnan(value) for value in products):
        return -np.inf, np.inf
    return (
        float(np.nextafter(min(products), -np.inf)),
        float(np.nextafter(max(products), np.inf)),
    )


def _interval_difference(
    first: tuple[float, float], second: tuple[float, float], /
) -> tuple[float, float]:
    lower, upper = first[0] - second[1], first[1] - second[0]
    if np.isnan(lower) or np.isnan(upper):
        return -np.inf, np.inf
    return (
        float(np.nextafter(lower, -np.inf)),
        float(np.nextafter(upper, np.inf)),
    )


def _interval_quotient(
    numerator: tuple[float, float],
    denominator: tuple[float, float],
    /,
) -> tuple[float, float] | None:
    if denominator[0] <= 0.0 <= denominator[1]:
        return None
    reciprocal = (
        float(np.nextafter(1.0 / denominator[1], -np.inf)),
        float(np.nextafter(1.0 / denominator[0], np.inf)),
    )
    return _interval_product(numerator, reciprocal)


def _rational_derivative_interval_bounds(
    expression: algebra.RationalPolynomial,
    vertices: tuple[tuple[Fraction, ...], ...],
    order: int,
    /,
    *,
    domain: Literal["simplex", "box"] | None = None,
    numerator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
    denominator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
) -> dict[tuple[int, ...], tuple[float, float]] | None:
    dimension = len(vertices[0])
    if numerator_derivatives is None:
        numerator_derivatives = _polynomial_derivative_bank(
            expression.numerator, dimension, order
        )
    if denominator_derivatives is None:
        denominator_derivatives = _polynomial_derivative_bank(
            expression.denominator, dimension, order
        )

    def polynomial_bounds(polynomial: _Polynomial, /) -> tuple[float, float]:
        return (
            _polynomial_bounds_on_simplex_box(polynomial, vertices)
            if domain is None
            else algebra.polynomial_bounds(polynomial, domain, dimension)
        )

    numerator_bounds = {
        index: polynomial_bounds(polynomial)
        for index, polynomial in numerator_derivatives.items()
    }
    denominator_bounds = {
        index: polynomial_bounds(polynomial)
        for index, polynomial in denominator_derivatives.items()
    }
    zero = (0,) * dimension
    base_denominator = denominator_bounds[zero]
    if base_denominator[0] <= 0.0 <= base_denominator[1]:
        return None
    derivatives: dict[tuple[int, ...], tuple[float, float]] = {}
    for total in range(order + 1):
        for multiindex in _multiindices(dimension, total):
            value = numerator_bounds[multiindex]
            for beta in _submultiindices(multiindex):
                if beta == zero:
                    continue
                parent = tuple(
                    power - derivative
                    for power, derivative in zip(multiindex, beta, strict=True)
                )
                coefficient = prod(
                    comb(power, derivative)
                    for power, derivative in zip(multiindex, beta, strict=True)
                )
                term = _interval_product(denominator_bounds[beta], derivatives[parent])
                term = _interval_product(term, (float(coefficient), float(coefficient)))
                value = _interval_difference(value, term)
            quotient = _interval_quotient(value, base_denominator)
            if quotient is None:
                return None
            derivatives[multiindex] = quotient
    return derivatives


def _standard_simplex_polynomial_integral(
    polynomial: _Polynomial, dimension: int, /
) -> _ExactScalar:
    total: _ExactScalar = Fraction(0)
    for exponent, coefficient in polynomial.items():
        moment = Fraction(
            prod(factorial(power) for power in exponent),
            factorial(dimension + sum(exponent)),
        )
        total += coefficient * moment
    return total


def _taylor_polynomial_integral(
    expression: _Expression,
    vertices: tuple[tuple[Fraction, ...], ...],
    center: tuple[Fraction, ...],
    order: int,
    /,
    *,
    numerator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
    denominator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
    polynomial_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
) -> tuple[float, float]:
    dimension = len(center)
    arguments = _simplex_chart_arguments(vertices, dimension)
    shifted = tuple(
        algebra.subtract(argument, algebra.constant(value, dimension))
        for argument, value in zip(arguments, center, strict=True)
    )
    shifted_powers = tuple(
        tuple(algebra.power(argument, exponent) for exponent in range(order + 1))
        for argument in shifted
    )
    rational_derivatives: dict[tuple[int, ...], _ExactScalar] | None = None
    if isinstance(expression, algebra.RationalPolynomial):
        rational_derivatives = _exact_rational_derivative_values(
            expression,
            center,
            order,
            numerator_derivatives=numerator_derivatives,
            denominator_derivatives=denominator_derivatives,
        )
    elif polynomial_derivatives is None:
        polynomial_derivatives = _polynomial_derivative_bank(expression, dimension, order)
    total: _ExactScalar = Fraction(0)
    for degree in range(order + 1):
        for multiindex in _multiindices(dimension, degree):
            if rational_derivatives is not None:
                derivative = rational_derivatives[multiindex]
            else:
                if polynomial_derivatives is None:
                    raise RuntimeError("Polynomial Taylor derivatives were not prepared.")
                derivative = _evaluate_expression(
                    polynomial_derivatives[multiindex], center
                )
            coefficient = derivative / prod(factorial(value) for value in multiindex)
            monomial: _Expression = algebra.constant(1, dimension)
            for axis, exponent in enumerate(multiindex):
                if exponent:
                    monomial = algebra.multiply(monomial, shifted_powers[axis][exponent])
            if isinstance(monomial, algebra.RationalPolynomial):
                raise RuntimeError("Affine Taylor moments require polynomial monomials.")
            total += coefficient * _standard_simplex_polynomial_integral(
                monomial, dimension
            )
    return _integral_center_and_error(total * abs(_simplex_determinant(vertices)))


def _polynomial_bounds_on_simplex_box(
    polynomial: _Polynomial,
    vertices: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[float, float]:
    dimension = len(vertices[0])
    lower = tuple(min(vertex[axis] for vertex in vertices) for axis in range(dimension))
    upper = tuple(max(vertex[axis] for vertex in vertices) for axis in range(dimension))
    if any(value < 0 for value in lower):
        raise ValueError("FE reference simplex boxes require nonnegative coordinates.")
    if not polynomial:
        return 0.0, 0.0
    variable_intervals = tuple(
        (
            coefficient_endpoint(value, upper=False, saturate=True),
            coefficient_endpoint(value, upper=True, saturate=True),
        )
        for value in lower
    )
    upper_intervals = tuple(
        coefficient_endpoint(value, upper=True, saturate=True) for value in upper
    )
    variable_intervals = tuple(
        (lower_bound, upper_bound)
        for lower_bound, upper_bound in zip(
            (interval[0] for interval in variable_intervals),
            upper_intervals,
            strict=True,
        )
    )
    result = (0.0, 0.0)
    for exponent, coefficient in polynomial.items():
        coefficient_interval = (
            coefficient_endpoint(coefficient, upper=False, saturate=True),
            coefficient_endpoint(coefficient, upper=True, saturate=True),
        )
        monomial = (1.0, 1.0)
        for axis, power in enumerate(exponent):
            for _ in range(power):
                monomial = _interval_product(monomial, variable_intervals[axis])
        result = _interval_sum(result, _interval_product(coefficient_interval, monomial))
    return result


def _taylor_remainder_bound(
    expression: _Expression,
    vertices: tuple[tuple[Fraction, ...], ...],
    center: tuple[Fraction, ...],
    order: int,
    /,
    *,
    numerator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
    denominator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
    polynomial_derivatives: dict[tuple[int, ...], _Polynomial] | None = None,
    prepared_rational_derivatives: (
        dict[tuple[int, ...], tuple[float, float]] | None
    ) = None,
) -> float:
    dimension = len(center)
    distances = tuple(
        max(abs(vertex[axis] - center[axis]) for vertex in vertices)
        for axis in range(dimension)
    )
    upper_distances = tuple(
        coefficient_endpoint(value, upper=True, saturate=True) for value in distances
    )
    bound = 0.0
    remainder_order = order + 1
    rational_derivatives: dict[tuple[int, ...], tuple[float, float]] | None = None
    if isinstance(expression, algebra.RationalPolynomial):
        rational_derivatives = prepared_rational_derivatives
        if rational_derivatives is None:
            rational_derivatives = _rational_derivative_interval_bounds(
                expression,
                vertices,
                remainder_order,
                numerator_derivatives=numerator_derivatives,
                denominator_derivatives=denominator_derivatives,
            )
        if rational_derivatives is None:
            return np.inf
    elif polynomial_derivatives is None:
        polynomial_derivatives = _polynomial_derivative_bank(
            expression, dimension, remainder_order
        )
    for multiindex in _multiindices(dimension, remainder_order):
        if rational_derivatives is not None:
            lower, upper = rational_derivatives[multiindex]
        else:
            if polynomial_derivatives is None:
                raise RuntimeError("Polynomial Taylor bounds were not prepared.")
            lower, upper = _polynomial_bounds_on_simplex_box(
                polynomial_derivatives[multiindex], vertices
            )
        derivative_bound = max(abs(lower), abs(upper))
        distance_factor = 1.0
        for axis, exponent in enumerate(multiindex):
            for _ in range(exponent):
                distance_factor = float(
                    np.nextafter(distance_factor * upper_distances[axis], np.inf)
                )
        denominator = prod(factorial(value) for value in multiindex)
        term = float(np.nextafter(derivative_bound * distance_factor, np.inf))
        term = float(np.nextafter(term / denominator, np.inf))
        if not np.isfinite(term):
            return np.inf
        bound = float(np.nextafter(bound + term, np.inf))
    determinant = abs(_simplex_determinant(vertices))
    measure = coefficient_endpoint(
        determinant / factorial(dimension), upper=True, saturate=True
    )
    result = bound * measure
    return float(np.nextafter(result, np.inf)) if result else 0.0


def _adaptive_signed_simplex_integral(
    expression: _Expression,
    vertices: tuple[tuple[Fraction, ...], ...],
    policy: _IntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> tuple[float, float]:
    order = 6
    dimension = len(vertices[0])
    numerator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None
    denominator_derivatives: dict[tuple[int, ...], _Polynomial] | None = None
    polynomial_derivatives: dict[tuple[int, ...], _Polynomial] | None = None
    if isinstance(expression, algebra.RationalPolynomial):
        numerator_derivatives = _polynomial_derivative_bank(
            expression.numerator, dimension, order + 1
        )
        denominator_derivatives = _polynomial_derivative_bank(
            expression.denominator, dimension, order + 1
        )
    else:
        polynomial_derivatives = _polynomial_derivative_bank(
            expression, dimension, order + 1
        )

    def record(
        simplex: tuple[tuple[Fraction, ...], ...],
    ) -> tuple[tuple[tuple[Fraction, ...], ...], float, float, float, float]:
        dimension = len(simplex) - 1
        center = tuple(
            sum((vertex[axis] for vertex in simplex), Fraction(0)) / len(simplex)
            for axis in range(dimension)
        )
        value, polynomial_error = _taylor_polynomial_integral(
            expression,
            simplex,
            center,
            order,
            numerator_derivatives=numerator_derivatives,
            denominator_derivatives=denominator_derivatives,
            polynomial_derivatives=polynomial_derivatives,
        )
        remainder = _taylor_remainder_bound(
            expression,
            simplex,
            center,
            order,
            numerator_derivatives=numerator_derivatives,
            denominator_derivatives=denominator_derivatives,
            polynomial_derivatives=polynomial_derivatives,
        )
        if not np.isfinite(remainder):
            return simplex, 0.0, -np.inf, np.inf, np.inf
        error = float(np.nextafter(polynomial_error + remainder, np.inf))
        lower = float(np.nextafter(value - error, -np.inf))
        upper = float(np.nextafter(value + error, np.inf))
        return simplex, value, lower, upper, max(value - lower, upper - value)

    leaves = [record(vertices)]
    blocked = (
        {vertices} if not _simplex_has_binary64_interior(vertices, rule_points) else set()
    )
    while True:
        estimate = fsum(value[1] for value in leaves)
        lower = float(np.nextafter(fsum(value[2] for value in leaves), -np.inf))
        upper = float(np.nextafter(fsum(value[3] for value in leaves), np.inf))
        error = max(estimate - lower, upper - estimate)
        scale = max(
            abs(estimate),
            fsum(max(abs(value[2]), abs(value[3])) for value in leaves),
        )
        tolerance = float(
            np.nextafter(
                policy.absolute_tolerance + policy.relative_tolerance * scale,
                np.inf,
            )
        )
        if np.isfinite(error) and error <= tolerance:
            return estimate, float(np.nextafter(error, np.inf)) if error else 0.0
        eligible = [
            index for index, value in enumerate(leaves) if value[0] not in blocked
        ]
        if not eligible:
            raise ValueError(
                "Rational FE moment integration is unresolved at binary64 simplex resolution."
            )
        index = max(eligible, key=lambda row: leaves[row][4])
        simplex = leaves.pop(index)[0]
        for child in _subdivide_simplex(simplex):
            leaves.append(record(child))
            if not _simplex_has_binary64_interior(child, rule_points):
                blocked.add(child)


def _integrate_entity_expression(
    expression: _Expression,
    entity_dimension: int,
    family: str,
    integration_policy: CurvedOverlapIntegrationPolicy | None,
    rule_points: np.ndarray | None,
    rule_weights: np.ndarray | None,
    piece_policy: CurvedOverlapIntegrationPolicy | None,
    /,
) -> tuple[float, float, bool]:
    expression = _polynomial_expression(expression)
    if entity_dimension == 0:
        exact = _evaluate_expression(expression, ())
        value, error = _integral_center_and_error(exact)
        return value, error, True
    pieces = _entity_simplices(entity_dimension, family)
    if not pieces:
        raise RuntimeError("Positive-dimensional FE entity has no exact simplex cover.")
    if not expression:
        return 0.0, 0.0, True
    rational = isinstance(expression, algebra.RationalPolynomial)
    if rational and integration_policy is None:
        raise ValueError(
            "Nonconstant rational reference-map moments require an explicit CurvedOverlapIntegrationPolicy."
        )
    if rational:
        from ...geometry._common_root_volume_overlap import CurvedOverlapIntegrationPolicy

        if not isinstance(integration_policy, CurvedOverlapIntegrationPolicy):
            raise TypeError("Rational FE moments require CurvedOverlapIntegrationPolicy.")
        if piece_policy is None:
            piece_policy = CurvedOverlapIntegrationPolicy(
                relative_tolerance=(integration_policy.relative_tolerance / len(pieces)),
                absolute_tolerance=integration_policy.absolute_tolerance / len(pieces),
            )
        if not isinstance(piece_policy, CurvedOverlapIntegrationPolicy):
            raise TypeError("Rational FE piece integrations require the curved policy.")
    exact_values: list[Fraction] = []
    exact_errors: list[Fraction] = []
    all_exact = True
    for piece in pieces:
        polynomial_integral = _exact_polynomial_simplex_integral(expression, piece)
        if polynomial_integral is not None:
            value, error = polynomial_integral
            exact = True
        elif entity_dimension == 3:
            from ...geometry._common_root_volume_overlap import _integrate_expression

            if rule_points is None or rule_weights is None or piece_policy is None:
                raise ValueError(
                    "Rational volume moments require a cubature rule and policy."
                )
            value, error, exact = _integrate_expression(
                _expression_for_algebra(expression),
                _as_tetrahedron(piece),
                piece_policy,
                rule_points,
                rule_weights,
            )
        else:
            if rule_points is None or rule_weights is None or piece_policy is None:
                raise ValueError(
                    "Rational trace moments require a cubature rule and policy."
                )
            value, error = _adaptive_signed_simplex_integral(
                expression,
                piece,
                piece_policy,
                rule_points,
                rule_weights,
            )
            exact = False
        if not np.isfinite(value) or not np.isfinite(error) or error < 0.0:
            raise ValueError("Reference form moment integration produced invalid bounds.")
        exact_values.append(Fraction(float(value)))
        exact_errors.append(Fraction(float(error)))
        all_exact = all_exact and exact
    exact_center = sum(exact_values, Fraction(0))
    value = float(exact_center)
    rounding = abs(Fraction(value) - exact_center)
    total_error = sum(exact_errors, Fraction(0)) + rounding
    error = coefficient_endpoint(total_error, upper=True, saturate=True)
    if error:
        error = float(np.nextafter(error, np.inf))
    if not np.isfinite(value) or not np.isfinite(error):
        raise ValueError("Reference form moment has no finite binary64 error enclosure.")
    return value, error, all_exact


def _sum_exact_expressions(
    values: Sequence[_Expression],
    dimension: int,
    /,
) -> _Expression:
    groups: list[tuple[_Polynomial, list[_Polynomial]]] = []
    unit = algebra.constant(1, dimension)
    for value in values:
        if isinstance(value, algebra.RationalPolynomial):
            numerator, denominator = value.numerator, value.denominator
        else:
            numerator, denominator = value, unit
        for group_denominator, numerators in groups:
            if group_denominator == denominator:
                numerators.append(numerator)
                break
        else:
            groups.append((denominator, [numerator]))
    if not groups:
        return algebra.constant(0, dimension)
    rational_groups = tuple(
        (denominator, algebra.sum_polynomials(numerators, dimension))
        for denominator, numerators in groups
    )
    common_denominator = unit
    for denominator, _ in rational_groups:
        common_denominator = algebra.multiply(common_denominator, denominator)
    aligned_numerators = []
    for group_index, (denominator, numerator) in enumerate(rational_groups):
        complement = unit
        for other_index, (other_denominator, _) in enumerate(rational_groups):
            if other_index != group_index:
                complement = algebra.multiply(complement, other_denominator)
        aligned_numerators.append(algebra.multiply(numerator, complement))
    common_numerator = algebra.sum_polynomials(aligned_numerators, dimension)
    if common_denominator == unit:
        return common_numerator
    return algebra.RationalPolynomial(common_numerator, common_denominator)


def _local_reference_moments(
    source: FiniteElementSpec,
    target: FiniteElementSpec,
    reference_map: NestedSimplexMap | NestedCellReferenceMap | None,
    integration_policy: CurvedOverlapIntegrationPolicy | None,
    prepared_orientation: _PreparedReferenceChartOrientation | None,
    /,
) -> MomentIntegralMatrix:
    source_basis, target_basis = source.form_basis, target.form_basis
    if source_basis is None or target_basis is None:
        raise ValueError("Exact form transfer requires canonical polynomial form bases.")
    if (
        source_basis.dimension != target_basis.dimension
        or source_basis.form_degree != target_basis.form_degree
        or source_basis.twist != target_basis.twist
        or source_basis.local_dof_count != source.local_dof_count
        or target_basis.local_dof_count != target.local_dof_count
    ):
        raise ValueError(
            "Source and target form bases do not describe one compatible action."
        )
    dimension, degree = source_basis.dimension, source_basis.form_degree
    arguments = _reference_arguments(reference_map, dimension)
    chart_orientation = (
        _orientation_sign(arguments, target_basis.family, required=True)
        if prepared_orientation is None
        else prepared_orientation.require(arguments, target_basis.family)
    )
    orientation = chart_orientation if target_basis.twist == "twisted" else 1
    source_components = _basis_components(source_basis)
    source_blades = tuple(combinations(range(dimension), degree))
    values = np.zeros((target_basis.local_dof_count, source_basis.local_dof_count))
    errors = np.zeros_like(values)
    exact = np.ones(values.shape, dtype=np.bool_)
    enclosed = np.zeros(values.shape, dtype=np.bool_)
    rules_by_dimension: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    piece_policies: dict[int, CurvedOverlapIntegrationPolicy] = {}
    if integration_policy is not None:
        from ...geometry._common_root_volume_overlap import (
            CurvedOverlapIntegrationPolicy,
        )

        if not isinstance(integration_policy, CurvedOverlapIntegrationPolicy):
            raise TypeError("Rational FE moments require CurvedOverlapIntegrationPolicy.")
        for entity_dimension in range(1, dimension + 1):
            pieces = _entity_simplices(entity_dimension, target_basis.family)
            piece_policies[entity_dimension] = (
                integration_policy
                if len(pieces) == 1
                else CurvedOverlapIntegrationPolicy(
                    relative_tolerance=(
                        integration_policy.relative_tolerance / len(pieces)
                    ),
                    absolute_tolerance=(
                        integration_policy.absolute_tolerance / len(pieces)
                    ),
                )
            )
    seen_target_dofs: set[int] = set()
    for entity_dimension, entities in enumerate(target_basis.entity_vertices):
        for entity_index, entity in enumerate(entities):
            entity_dofs = tuple(target.entity_dofs[entity_dimension][entity_index])
            if any(dof in seen_target_dofs for dof in entity_dofs):
                raise ValueError(
                    "A target FE moment row is assigned to multiple entities."
                )
            seen_target_dofs.update(entity_dofs)
            if not entity_dofs:
                continue
            free, origin, chart = _form_entity_chart(
                target_basis.dimension, target_basis.family, entity
            )
            entity_arguments = _affine_arguments(origin, chart, entity_dimension)
            mapped_arguments = tuple(
                _compose(argument, entity_arguments) for argument in arguments
            )
            jacobian = tuple(
                tuple(
                    algebra.derivative(mapped_arguments[row], axis)
                    for axis in range(entity_dimension)
                )
                for row in range(dimension)
            )
            target_blades, test_forms = _target_test_forms(
                target_basis, entity, entity_dimension, entity_dofs, free
            )
            pulled_components = []
            for components in source_components:
                composed = tuple(
                    _compose(value, mapped_arguments) for value in components
                )
                pulled = []
                for target_blade in target_blades:
                    terms = []
                    for source_blade, value in zip(source_blades, composed, strict=True):
                        minor = _determinant(
                            tuple(
                                tuple(jacobian[row][column] for column in target_blade)
                                for row in source_blade
                            ),
                            entity_dimension,
                        )
                        if value and minor:
                            terms.append(algebra.multiply(value, minor))
                    pulled.append(_sum_exact_expressions(terms, entity_dimension))
                if orientation != 1:
                    pulled = [algebra.scale(value, orientation) for value in pulled]
                pulled_components.append(tuple(pulled))
            rule_points = rule_weights = None
            if entity_dimension > 0:
                rule_pair = rules_by_dimension.get(entity_dimension)
                if rule_pair is None:
                    from ..._polynomial._cubature import (
                        cubature_rule_data,
                        simplex_rule_data,
                        tensor_product_rule_data,
                    )

                    rule_data = (
                        tensor_product_rule_data(1, 5, family="gauss")
                        if entity_dimension == 1
                        else cubature_rule_data("triangle", 8)
                        if entity_dimension == 2
                        else cubature_rule_data("tetrahedron", 8)
                        if entity_dimension == 3
                        else simplex_rule_data(entity_dimension, 0)
                    )
                    rule_pair = (
                        np.asarray(rule_data.points, dtype=np.float64),
                        np.asarray(rule_data.weights, dtype=np.float64),
                    )
                    rules_by_dimension[entity_dimension] = rule_pair
                rule_points, rule_weights = rule_pair
            for target_dof in entity_dofs:
                test = test_forms.get(target_dof)
                if test is None:
                    raise ValueError("A target moment row has no canonical test form.")
                for source_dof, pulled in enumerate(pulled_components):
                    terms = []
                    for target_blade, form_component in zip(
                        target_blades, pulled, strict=True
                    ):
                        for test_blade, test_component in zip(
                            tuple(
                                combinations(
                                    range(entity_dimension),
                                    entity_dimension - degree,
                                )
                            ),
                            test,
                            strict=True,
                        ):
                            if (
                                set(target_blade).isdisjoint(test_blade)
                                and len(target_blade) + len(test_blade)
                                == entity_dimension
                            ):
                                sign = _wedge_sign(target_blade, test_blade)
                                product = algebra.multiply(form_component, test_component)
                                if product:
                                    terms.append(algebra.scale(product, sign))
                    integrand = _sum_exact_expressions(terms, entity_dimension)
                    moment_value, moment_error, moment_exact = (
                        _integrate_entity_expression(
                            integrand,
                            entity_dimension,
                            target_basis.family,
                            integration_policy,
                            rule_points,
                            rule_weights,
                            piece_policies.get(entity_dimension),
                        )
                    )
                    values[target_dof, source_dof] = moment_value
                    errors[target_dof, source_dof] = moment_error
                    exact[target_dof, source_dof] = moment_exact
                    enclosed[target_dof, source_dof] = not moment_exact
    if seen_target_dofs != set(range(target_basis.local_dof_count)):
        raise ValueError("Target form moments do not cover every local entity DOF.")
    return MomentIntegralMatrix(values, errors, exact, enclosed)


def _transform_moments(
    moments: MomentIntegralMatrix,
    source_transform: _CellOrientationMatrix,
    target_transform: _CellOrientationMatrix,
    /,
) -> MomentIntegralMatrix:
    source = _fraction_matrix(source_transform)
    target = _fraction_matrix(target_transform)
    if (
        len(source) != moments.values.shape[1]
        or any(len(row) != len(source) for row in source)
        or len(target) != moments.values.shape[0]
        or any(len(row) != len(target) for row in target)
    ):
        raise ValueError("Cell orientation transforms do not match local moment extents.")
    identity = [
        [Fraction(int(row == column)) for column in range(len(target))]
        for row in range(len(target))
    ]
    inverse_target = solve_exact(target, identity)
    source_width = len(source)
    target_width = len(target)
    raw_values = [[Fraction(float(value)) for value in row] for row in moments.values]
    raw_errors = [[Fraction(float(value)) for value in row] for row in moments.errors]
    intermediate = [
        [
            sum(
                (
                    raw_values[row][inner] * source[inner][column]
                    for inner in range(source_width)
                ),
                Fraction(0),
            )
            for column in range(source_width)
        ]
        for row in range(target_width)
    ]
    intermediate_errors = [
        [
            sum(
                (
                    raw_errors[row][inner] * abs(source[inner][column])
                    for inner in range(source_width)
                ),
                Fraction(0),
            )
            for column in range(source_width)
        ]
        for row in range(target_width)
    ]
    intermediate_exact = [
        [
            sum(
                int(moments.exact[row, inner])
                for inner in range(source_width)
                if source[inner][column]
            )
            for column in range(source_width)
        ]
        for row in range(target_width)
    ]
    intermediate_enclosed = [
        [
            sum(
                int(moments.enclosed[row, inner])
                for inner in range(source_width)
                if source[inner][column]
            )
            for column in range(source_width)
        ]
        for row in range(target_width)
    ]
    values = np.empty((target_width, source_width), dtype=np.float64)
    errors = np.empty_like(values)
    exact = np.zeros(values.shape, dtype=np.bool_)
    enclosed = np.zeros(values.shape, dtype=np.bool_)
    for row in range(target_width):
        for column in range(source_width):
            center = sum(
                (
                    inverse_target[row][inner] * intermediate[inner][column]
                    for inner in range(target_width)
                ),
                Fraction(0),
            )
            error = sum(
                (
                    abs(inverse_target[row][inner]) * intermediate_errors[inner][column]
                    for inner in range(target_width)
                ),
                Fraction(0),
            )
            values[row, column] = float(center)
            rounding = abs(Fraction(float(values[row, column])) - center)
            errors[row, column] = coefficient_endpoint(
                error + rounding, upper=True, saturate=True
            )
            exact[row, column] = all(
                intermediate_enclosed[inner][column] == 0
                for inner in range(target_width)
                if inverse_target[row][inner]
            )
            enclosed[row, column] = any(
                intermediate_enclosed[inner][column] != 0
                for inner in range(target_width)
                if inverse_target[row][inner]
            )
            exact_count = sum(
                intermediate_exact[inner][column]
                for inner in range(target_width)
                if inverse_target[row][inner]
            )
            enclosed_count = sum(
                intermediate_enclosed[inner][column]
                for inner in range(target_width)
                if inverse_target[row][inner]
            )
            if exact_count + enclosed_count == 0:
                exact[row, column] = True
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors)):
        raise ValueError(
            "Transformed exact FE moments have no finite binary64 enclosure."
        )
    return MomentIntegralMatrix(values, errors, exact, enclosed)


def exact_reference_form_moments(
    source: FiniteElementSpec,
    target: FiniteElementSpec,
    reference_map: NestedSimplexMap | NestedCellReferenceMap | None,
    source_transform: _CellOrientationMatrix,
    target_transform: _CellOrientationMatrix,
    integration_policy: CurvedOverlapIntegrationPolicy | None,
    /,
    *,
    prepared_orientation: _PreparedReferenceChartOrientation | None = None,
) -> MomentIntegralMatrix:
    """Prepare one cell's source-to-target FE action with entrywise bounds."""
    return _transform_moments(
        _local_reference_moments(
            source,
            target,
            reference_map,
            integration_policy,
            prepared_orientation,
        ),
        source_transform,
        target_transform,
    )


__all__ = ["MomentIntegralMatrix", "exact_reference_form_moments"]
