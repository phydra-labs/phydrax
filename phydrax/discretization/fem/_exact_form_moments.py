# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Source-faithful exterior moment integrals on declared reference entities.

Coarsening pulls back the complementary test form. It never inverts a chart or
assumes that circulation/flux functionals form an orthonormal coefficient basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import assert_never

import numpy as np
from numpy.typing import NDArray

from phydrax.linalg._exact import solve_exact

from ..._strict import StrictModule
from ...geometry._rational import rational_vector
from ...typing import checked
from .. import _coordinate_enclosure as algebra
from .._coordinate_budget import CoordinateEnclosureBudget
from .._mapped_geometry_measure import integrate_mapped_polynomial
from .._nested_reference_types import (
    NestedReferencePair,
    PolynomialReferencePair,
    rational_pair_arrays,
)
from .._reference_cell import reference_cell_topology
from ._form_basis_types import AbstractReferenceFormBasis
from ._form_elements import form_entity_chart, form_reference_vertices
from ._form_expressions import component_expressions, functional_density_expressions


type _Expressions = tuple[tuple[algebra.Expression, ...], ...]
type _Pair = NestedReferencePair | PolynomialReferencePair

type _ArgumentKey = tuple[tuple[tuple[tuple[int, ...], Fraction], ...], ...]
type _AffineArgumentKey = tuple[
    tuple[int, ...],
    tuple[int, ...],
    bytes,
    bytes,
    tuple[tuple[Fraction, ...], ...] | None,
    tuple[Fraction, ...] | None,
]


def _argument_key(arguments: tuple[algebra.Polynomial, ...], /) -> _ArgumentKey:
    return tuple(
        tuple(
            (index, algebra.require_rational_reference_scalar(coefficient))
            for index, coefficient in sorted(value.items())
        )
        for value in arguments
    )


@checked
def identity_reference_pair(pair: _Pair, /) -> bool:
    match pair:
        case NestedReferencePair():
            dimension = pair.matrix.shape[0]
            return (
                pair.exact_matrix is None
                and pair.exact_offset is None
                and pair.matrix.shape == (dimension, dimension)
                and pair.offset.shape == (dimension,)
                and np.array_equal(pair.matrix, np.eye(dimension, dtype=np.float64))
                and np.array_equal(pair.offset, np.zeros(dimension, dtype=np.float64))
            )
        case PolynomialReferencePair():
            return False
        case _:
            assert_never(pair)


@dataclass(frozen=True)
class MomentIntegralMatrix:
    value: NDArray[np.float64]
    error: NDArray[np.float64]


@checked
def reference_arguments(
    pair: _Pair, *, budget: CoordinateEnclosureBudget
) -> tuple[algebra.Polynomial, ...]:
    match pair:
        case PolynomialReferencePair():
            return pair.arguments
        case NestedReferencePair():
            matrix, offset = rational_pair_arrays(pair)
            return algebra.affine_arguments(offset, matrix, budget=budget)
        case _:
            assert_never(pair)


def _integration_kind(kind: str) -> str:
    match kind:
        case "point" | "simplex:0" | "tensor:0":
            return "point"
        case "simplex:1" | "tensor:1":
            return "interval"
        case "simplex:2":
            return "triangle"
        case "simplex:3":
            return "tetrahedron"
        case "tensor:2":
            return "quadrilateral"
        case "tensor:3":
            return "hexahedron"
        case (
            "interval"
            | "triangle"
            | "tetrahedron"
            | "quadrilateral"
            | "hexahedron"
            | "prism"
            | "pyramid"
        ):
            return kind
        case _:
            raise ValueError(
                f"integration_kind.kind must select a supported declared route; got No algebraic compatible moment integral for reference topology {kind!r}."
            )


def _domain(kind: str) -> str:
    native = _integration_kind(kind)
    return (
        "simplex"
        if native in ("interval", "triangle", "tetrahedron")
        else "prism"
        if native == "prism"
        else "box"
    )


def _pullback_matrix(
    jacobian: _Expressions,
    source_dimension: int,
    target_dimension: int,
    degree: int,
    variable_dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Expressions:
    source_blades = tuple(combinations(range(source_dimension), degree))
    target_blades = tuple(combinations(range(target_dimension), degree))
    return tuple(
        tuple(
            algebra.expression_determinant(
                tuple(
                    tuple(jacobian[row][column] for column in target) for row in source
                ),
                variable_dimension=variable_dimension,
                budget=budget,
            )
            for source in source_blades
        )
        for target in target_blades
    )


def _matrix_action(
    matrix: _Expressions,
    values: tuple[algebra.Expression, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[algebra.Expression, ...]:
    return tuple(
        algebra.expression_sum(
            tuple(
                algebra.expression_multiply(coefficient, value, budget=budget)
                for coefficient, value in zip(row, values, strict=True)
                if coefficient and value
            ),
            budget=budget,
        )
        for row in matrix
    )


def _constant_matrix(
    matrix: NDArray[np.float64], variables: int, *, budget: CoordinateEnclosureBudget
) -> _Expressions:
    return tuple(
        tuple(
            algebra.constant(Fraction(float(value)), variables, budget=budget)
            for value in row
        )
        for row in matrix
    )


def _wedge_sign(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    return (-1) ** sum(a > b for a in first for b in second)


def _complementary_test(
    density: tuple[algebra.Expression, ...],
    dimension: int,
    degree: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[algebra.Expression, ...]:
    blades = tuple(combinations(range(dimension), degree))
    dual = tuple(combinations(range(dimension), dimension - degree))
    result: list[algebra.Expression] = [{} for _ in dual]
    for blade, value in zip(blades, density, strict=True):
        other = tuple(axis for axis in range(dimension) if axis not in blade)
        result[dual.index(other)] = algebra.expression_scale(
            value, _wedge_sign(blade, other), budget=budget
        )
    return tuple(result)


def _density_from_test(
    test: tuple[algebra.Expression, ...],
    dimension: int,
    degree: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[algebra.Expression, ...]:
    blades = tuple(combinations(range(dimension), degree))
    dual = tuple(combinations(range(dimension), dimension - degree))
    return tuple(
        algebra.expression_scale(
            test[dual.index(other)], _wedge_sign(blade, other), budget=budget
        )
        for blade in blades
        for other in (tuple(axis for axis in range(dimension) if axis not in blade),)
    )


def _intrinsic_densities(
    densities: _Expressions,
    chart: NDArray[np.float64],
    degree: int,
    variables: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Expressions:
    ambient, intrinsic = chart.shape
    embedding = _pullback_matrix(
        _constant_matrix(chart, variables, budget=budget),
        ambient,
        intrinsic,
        degree,
        variables,
        budget=budget,
    )
    rows = tuple(combinations(range(intrinsic), degree))
    columns = tuple(combinations(range(ambient), degree))
    if not rows:
        raise ValueError(
            "intrinsic_densities.degree must have dimension at least the form degree; got An entity cannot support a higher-degree form functional."
        )
    pivot: tuple[int, ...] | None = None
    exact: list[list[Fraction]] = []
    for selected in combinations(range(len(columns)), len(rows)):
        values = [
            [
                algebra.require_rational_reference_scalar(
                    algebra.expression_evaluate(
                        embedding[row][column], (Fraction(0),) * variables, budget=budget
                    )
                )
                for row in range(len(rows))
            ]
            for column in selected
        ]
        determinant = algebra.expression_determinant(
            tuple(
                tuple(algebra.constant(value, 0, budget=budget) for value in row)
                for row in values
            ),
            variable_dimension=0,
            budget=budget,
        )
        if algebra.expression_evaluate(determinant, (), budget=budget):
            pivot, exact = selected, values
            break
    if pivot is None:
        raise ValueError(
            "intrinsic_densities.chart must have a nonsingular chart with resolved orientation; got A declared form entity has a singular blade chart."
        )
    result: list[tuple[algebra.Expression, ...]] = []
    for density in densities:
        indices = sorted(
            {
                index
                for column in pivot
                for index in algebra.expression_parts(
                    density[column], variables, budget=budget
                )[0]
            }
        )
        if any(
            isinstance(density[column], algebra.RationalPolynomial) for column in pivot
        ):
            # Body charts are identities, so their rational densities need no solve.
            if intrinsic != ambient or not np.array_equal(
                chart, np.eye(ambient, dtype=np.float64)
            ):
                raise ValueError(
                    "intrinsic_densities.densities must provide their intrinsic chart action; got Rational trace densities require their intrinsic chart action."
                )
            result.append(density)
            continue
        numerators = tuple(
            algebra.expression_parts(value, variables, budget=budget)[0]
            for value in density
        )
        right = [
            [
                algebra.require_rational_reference_scalar(
                    numerators[column].get(index, Fraction(0))
                )
                for index in indices
            ]
            for column in pivot
        ]
        solved = solve_exact(exact, right)
        polynomials: tuple[algebra.Polynomial, ...] = tuple(
            {index: value for index, value in zip(indices, row, strict=True) if value}
            for row in solved
        )
        result.append(polynomials)
    return tuple(result)


def _entity_map(
    coarse: AbstractReferenceFormBasis,
    coarse_entity: tuple[int, ...],
    fine: AbstractReferenceFormBasis,
    fine_entity: tuple[int, ...],
    arguments: tuple[algebra.Polynomial, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[algebra.Polynomial, ...], tuple[algebra.Polynomial, ...], int] | None:
    _, origin, chart = form_entity_chart(coarse.dimension, coarse.family, coarse_entity)
    _, fine_origin, fine_chart = form_entity_chart(
        fine.dimension, fine.family, fine_entity
    )
    dimension = chart.shape[1]
    fine_arguments = algebra.affine_arguments(fine_origin, fine_chart, budget=budget)
    image = tuple(
        algebra.compose(value, fine_arguments, budget=budget) for value in arguments
    )
    if dimension == 0:
        if any(
            algebra.add(
                value,
                algebra.constant(-Fraction(float(offset)), 0, budget=budget),
                budget=budget,
            )
            for value, offset in zip(image, origin, strict=True)
        ):
            return None
        return fine_arguments, (), 1
    pivot: tuple[int, ...] | None = None
    exact: list[list[Fraction]] = []
    for rows in combinations(range(coarse.dimension), dimension):
        values = [
            [Fraction(float(chart[row, column])) for column in range(dimension)]
            for row in rows
        ]
        determinant = algebra.expression_determinant(
            tuple(
                tuple(algebra.constant(value, 0, budget=budget) for value in row)
                for row in values
            ),
            variable_dimension=0,
            budget=budget,
        )
        if algebra.expression_evaluate(determinant, (), budget=budget):
            pivot, exact = rows, values
            break
    if pivot is None:
        raise ValueError(
            "entity_map.coarse must have a nonsingular chart with resolved orientation; got A declared coarse entity has a singular reference chart."
        )
    differences = tuple(
        algebra.add(
            image[row],
            algebra.constant(-Fraction(float(origin[row])), dimension, budget=budget),
            budget=budget,
        )
        for row in pivot
    )
    indices = sorted({index for value in differences for index in value})
    solved = solve_exact(
        exact,
        [
            [
                algebra.require_rational_reference_scalar(value.get(index, Fraction(0)))
                for index in indices
            ]
            for value in differences
        ],
    )
    local: tuple[algebra.Polynomial, ...] = tuple(
        {index: value for index, value in zip(indices, row, strict=True) if value}
        for row in solved
    )
    reconstructed = tuple(
        algebra.add(
            algebra.constant(
                Fraction(float(origin[coordinate])), dimension, budget=budget
            ),
            algebra.sum_polynomials(
                tuple(
                    algebra.scale(
                        value, Fraction(float(chart[coordinate, column])), budget=budget
                    )
                    for column, value in enumerate(local)
                ),
                budget=budget,
            ),
            budget=budget,
        )
        for coordinate in range(coarse.dimension)
    )
    if any(
        algebra.add(actual, algebra.scale(expected, -1, budget=budget), budget=budget)
        for actual, expected in zip(image, reconstructed, strict=True)
    ):
        return None
    jacobian = tuple(
        tuple(algebra.derivative(value, axis, budget=budget) for axis in range(dimension))
        for value in local
    )
    determinant = algebra.expression_determinant(
        jacobian, variable_dimension=dimension, budget=budget
    )
    controls = algebra.expression_bernstein_coefficients(
        determinant, _domain(fine.entity_kind(fine_entity)), dimension, budget=budget
    )
    if min(controls) > 0:
        sign = 1
    elif max(controls) < 0:
        sign = -1
    else:
        raise ValueError(
            "entity_map.fine must have a nonsingular chart with resolved orientation; got A fine entity chart has unresolved orientation or a singular Jacobian."
        )
    return fine_arguments, local, sign


class _CommonEntityCoordinates(StrictModule):
    local: tuple[algebra.Expression, ...]


def _solve_common_entity_chart(
    target_dimension: int,
    dimension: int,
    origin: NDArray[np.float64],
    chart: NDArray[np.float64],
    image: tuple[algebra.Expression, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _CommonEntityCoordinates:
    pivot: tuple[int, ...] | None = None
    matrix: list[list[Fraction]] = []
    for rows in combinations(range(target_dimension), dimension):
        candidate = [
            [Fraction(float(chart[row, column])) for column in range(dimension)]
            for row in rows
        ]
        determinant = algebra.expression_determinant(
            tuple(
                tuple(algebra.constant(value, 0, budget=budget) for value in row)
                for row in candidate
            ),
            variable_dimension=0,
            budget=budget,
        )
        if algebra.expression_evaluate(determinant, (), budget=budget):
            pivot, matrix = rows, candidate
            break
    if pivot is None:
        raise ValueError(
            "solve_common_entity_chart.chart must have a nonsingular chart with resolved orientation; got A common form target entity has a singular declared chart."
        )
    differences = tuple(
        algebra.expression_add(
            image[row],
            algebra.constant(-Fraction(float(origin[row])), dimension, budget=budget),
            budget=budget,
        )
        for row in pivot
    )
    parts = tuple(
        algebra.expression_parts(value, dimension, budget=budget) for value in differences
    )
    denominator = algebra.constant(1, dimension, budget=budget)
    for _, value in parts:
        if algebra.divide_polynomial(denominator, value, budget=budget) is not None:
            continue
        denominator = (
            value
            if algebra.divide_polynomial(value, denominator, budget=budget) is not None
            else algebra.multiply(denominator, value, budget=budget)
        )
    numerators = []
    for numerator, divisor in parts:
        quotient = algebra.divide_polynomial(denominator, divisor, budget=budget)
        if quotient is None:
            raise ValueError(
                "solve_common_entity_chart.configuration must pass the declared scientific certificate; got Common form entity coefficient denominators failed algebraic alignment."
            )
        numerators.append(algebra.multiply(numerator, quotient, budget=budget))
    indices = sorted({index for value in numerators for index in value})
    solved = solve_exact(
        matrix,
        [
            [
                algebra.require_rational_reference_scalar(value.get(index, Fraction(0)))
                for index in indices
            ]
            for value in numerators
        ],
    )
    local = tuple(
        algebra.rational_expression(
            {index: value for index, value in zip(indices, row, strict=True) if value},
            denominator,
            budget=budget,
        )
        for row in solved
    )
    return _CommonEntityCoordinates(local)


def _common_entity_map(
    target: AbstractReferenceFormBasis,
    target_entity: tuple[int, ...],
    integration_kind: str,
    integration_entity: tuple[int, ...],
    arguments: tuple[algebra.Expression, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[algebra.Polynomial, ...], tuple[algebra.Expression, ...], int] | None:
    topology = reference_cell_topology(integration_kind)
    if integration_kind not in ("interval", "triangle", "tetrahedron"):
        raise ValueError(
            "common_entity_map.configuration must provide an simplex integration topology; got Common form pieces require an simplex integration topology."
        )
    dimension = len(integration_entity) - 1
    if (
        dimension < 0
        or dimension > topology.dimension
        or len(set(integration_entity)) != len(integration_entity)
        or frozenset(integration_entity)
        not in tuple(frozenset(entity) for entity in topology.entities[dimension])
    ):
        raise ValueError(
            "common_entity_map.configuration must be a reference entity of the integration topology; got The common form integration entity is not a native reference entity."
        )
    vertices = np.asarray(topology.vertices, dtype=np.float64)[list(integration_entity)]
    inclusion = algebra.affine_arguments(
        vertices[0], (vertices[1:] - vertices[0]).T, budget=budget
    )
    _, origin, chart = form_entity_chart(target.dimension, target.family, target_entity)
    if chart.shape[1] != dimension or len(arguments) != target.dimension:
        return None
    image = tuple(
        algebra.expression_compose(value, inclusion, budget=budget) for value in arguments
    )
    if dimension == 0:
        if any(
            algebra.expression_add(
                value,
                algebra.constant(-Fraction(float(offset)), 0, budget=budget),
                budget=budget,
            )
            for value, offset in zip(image, origin, strict=True)
        ):
            return None
        return inclusion, (), 1
    solved_chart = _solve_common_entity_chart(
        target.dimension, dimension, origin, chart, image, budget=budget
    )
    local = solved_chart.local
    reconstructed = tuple(
        algebra.expression_add(
            algebra.constant(
                Fraction(float(origin[coordinate])), dimension, budget=budget
            ),
            algebra.expression_sum(
                tuple(
                    algebra.expression_scale(
                        value, Fraction(float(chart[coordinate, column])), budget=budget
                    )
                    for column, value in enumerate(local)
                ),
                budget=budget,
            ),
            budget=budget,
        )
        for coordinate in range(target.dimension)
    )
    if any(
        algebra.expression_add(
            actual, algebra.expression_scale(expected, -1, budget=budget), budget=budget
        )
        for actual, expected in zip(image, reconstructed, strict=True)
    ):
        return None
    jacobian = tuple(
        tuple(
            algebra.expression_derivative(value, axis, budget=budget)
            for axis in range(dimension)
        )
        for value in local
    )
    determinant = algebra.expression_determinant(
        jacobian, variable_dimension=dimension, budget=budget
    )
    controls = algebra.expression_bernstein_coefficients(
        determinant, "simplex", dimension, budget=budget
    )
    if min(controls) > 0:
        sign = 1
    elif max(controls) < 0:
        sign = -1
    else:
        raise ValueError(
            "common_entity_map.configuration must have a nonsingular chart with resolved orientation; got A common form entity has unresolved orientation or a singular rational chart."
        )
    return inclusion, local, sign


def _full_chart_orientation(
    arguments: tuple[algebra.Expression, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> int:
    if len(arguments) != dimension:
        raise ValueError(
            "full_chart_orientation.configuration must provide full square reference charts; got Twisted common moments require full square reference charts."
        )
    jacobian = tuple(
        tuple(
            algebra.expression_derivative(value, axis, budget=budget)
            for axis in range(dimension)
        )
        for value in arguments
    )
    determinant = algebra.expression_determinant(
        jacobian, variable_dimension=dimension, budget=budget
    )
    controls = algebra.expression_bernstein_coefficients(
        determinant, "simplex", dimension, budget=budget
    )
    if min(controls) > 0:
        return 1
    if max(controls) < 0:
        return -1
    raise ValueError(
        "full_chart_orientation.configuration must have a nonsingular chart with resolved orientation; got A twisted common chart has unresolved full-cell orientation or a singular Jacobian."
    )


def _integration_factors(
    components: _Expressions,
    densities: _Expressions,
    kind: str,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_Expressions, _Expressions, algebra.Expression | None, str]:
    """Use an collapsed chart before expanding moment contractions."""
    if _integration_kind(kind) != "pyramid":
        return components, densities, None, kind
    arguments = algebra.chart_arguments("pyramid", 3, budget=budget)
    composition = algebra.ExpressionComposition(arguments, budget=budget)
    # These are coefficient functions, not blade pullbacks: their contraction
    # already represents the physical reference-entity moment density.
    components = tuple(
        tuple(composition(value, budget=budget) for value in row) for row in components
    )
    densities = tuple(
        tuple(composition(value, budget=budget) for value in row) for row in densities
    )
    jacobian = tuple(
        tuple(algebra.derivative(value, axis, budget=budget) for axis in range(3))
        for value in arguments
    )
    measure = algebra.expression_determinant(
        jacobian, variable_dimension=3, budget=budget
    )
    return components, densities, measure, "hexahedron"


@dataclass
class ExactFormMoments:
    ledger: CoordinateEnclosureBudget
    components_cache: dict[str, _Expressions] = field(default_factory=dict)
    densities_cache: dict[tuple[str, tuple[int, ...]], _Expressions] = field(
        default_factory=dict
    )
    refine_cache: dict[tuple[str, str, _ArgumentKey], MomentIntegralMatrix] = field(
        default_factory=dict
    )
    coarse_cache: dict[
        tuple[str, str, tuple[int, ...], tuple[int, ...], _ArgumentKey],
        tuple[MomentIntegralMatrix, tuple[tuple[Fraction, ...], ...]] | None,
    ] = field(default_factory=dict)
    arguments_cache: dict[
        _AffineArgumentKey, tuple[tuple[algebra.Polynomial, ...], _ArgumentKey]
    ] = field(default_factory=dict)

    def arguments(
        self, pair: _Pair
    ) -> tuple[tuple[algebra.Polynomial, ...], _ArgumentKey]:
        """Prepare one exact chart once, while binding every mutable array view."""
        if isinstance(pair, PolynomialReferencePair):
            return pair.arguments, _argument_key(pair.arguments)
        matrix, offset = np.asarray(pair.matrix), np.asarray(pair.offset)
        # Admission precedes byte-key creation. The original pair validator
        # still admits every new byte/exact-bank combination before caching it.
        self.ledger.reserve(
            matrix.size + offset.size, 256 + 16 * (matrix.size + offset.size)
        )
        key = (
            matrix.shape,
            offset.shape,
            matrix.astype(np.float64, copy=False).tobytes(),
            offset.astype(np.float64, copy=False).tobytes(),
            pair.exact_matrix,
            pair.exact_offset,
        )
        retained = self.arguments_cache.get(key)
        if retained is None:
            arguments = reference_arguments(pair, budget=self.ledger)
            retained = arguments, _argument_key(arguments)
            self.ledger.retain_basis((key, retained))
            self.arguments_cache[key] = retained
        return retained

    def _retain(self, values: _Expressions) -> None:
        # Cache the expressions that remain live, including their row containers.
        # Splitting polynomial values would manufacture retained unit denominators.
        self.ledger.retain_basis(values)

    def components(self, basis: AbstractReferenceFormBasis) -> _Expressions:
        key = basis.basis_id
        if key not in self.components_cache:
            with self.ledger.temporary_scope():
                values = component_expressions(basis, budget=self.ledger)
                self._retain(values)
                self.components_cache[key] = values
        return self.components_cache[key]

    def densities(
        self, basis: AbstractReferenceFormBasis, entity: tuple[int, ...]
    ) -> _Expressions:
        key = (basis.basis_id, entity)
        if key not in self.densities_cache:
            with self.ledger.temporary_scope():
                values = functional_density_expressions(basis, entity, budget=self.ledger)
                self._retain(values)
                self.densities_cache[key] = values
        return self.densities_cache[key]

    def integral(self, value: algebra.Expression, kind: str) -> tuple[float, float]:
        native = _integration_kind(kind)
        if native == "pyramid":
            arguments = algebra.chart_arguments(native, 3, budget=self.ledger)
            jacobian = tuple(
                tuple(
                    algebra.derivative(value, axis, budget=self.ledger)
                    for axis in range(3)
                )
                for value in arguments
            )
            value = algebra.expression_multiply(
                algebra.expression_compose(value, arguments, budget=self.ledger),
                algebra.expression_determinant(
                    jacobian, variable_dimension=3, budget=self.ledger
                ),
                budget=self.ledger,
            )
        with self.ledger.polynomial_preparation_scope():
            return integrate_mapped_polynomial(value, native, budget=self.ledger)

    def point_value(
        self,
        basis: AbstractReferenceFormBasis,
        value: algebra.Expression,
        point: tuple[Fraction, ...],
    ) -> Fraction:
        if (
            _integration_kind(basis.entity_kind(basis.entity_vertices[-1][0]))
            == "pyramid"
            and point[2] == 1
        ):
            if point != (Fraction(1, 2), Fraction(1, 2), Fraction(1)):
                raise ValueError(
                    "ExactFormMoments.point_value.configuration must lie within the declared reference support; got A scalar pyramid vertex is outside its algebraic reference apex."
                )
            collapsed = algebra.expression_compose(
                value,
                algebra.chart_arguments("pyramid", 3, budget=self.ledger),
                budget=self.ledger,
            )
            u, v = algebra.axes(2, budget=self.ledger)
            trace = algebra.expression_compose(
                collapsed,
                (u, v, algebra.constant(1, 2, budget=self.ledger)),
                budget=self.ledger,
            )
            result = algebra.expression_evaluate(
                trace, (Fraction(1, 2), Fraction(1, 2)), budget=self.ledger
            )
            if algebra.expression_add(
                trace,
                algebra.constant(-result, 2, budget=self.ledger),
                budget=self.ledger,
            ):
                raise ValueError(
                    "ExactFormMoments.point_value.configuration must have an approach-independent apex value; got The scalar pyramid apex depends on the reference approach."
                )
            return algebra.require_rational_reference_scalar(result)
        return algebra.require_rational_reference_scalar(
            algebra.expression_evaluate(value, point, budget=self.ledger)
        )

    def refine(
        self,
        source: AbstractReferenceFormBasis,
        target: AbstractReferenceFormBasis,
        pair: _Pair,
    ) -> MomentIntegralMatrix:
        if (
            source.basis_id == target.basis_id
            and source.local_dof_count == target.local_dof_count
            and identity_reference_pair(pair)
        ):
            return MomentIntegralMatrix(
                np.eye(source.local_dof_count, dtype=np.float64),
                np.zeros(
                    (target.local_dof_count, source.local_dof_count),
                    dtype=np.float64,
                ),
            )
        arguments, argument_key = self.arguments(pair)
        key = (source.basis_id, target.basis_id, argument_key)
        cached = self.refine_cache.get(key)
        if cached is not None:
            return cached
        jacobian = tuple(
            tuple(
                algebra.derivative(value, axis, budget=self.ledger)
                for axis in range(source.dimension)
            )
            for value in arguments
        )
        action = _pullback_matrix(
            jacobian,
            source.dimension,
            target.dimension,
            source.form_degree,
            source.dimension,
            budget=self.ledger,
        )
        source_components = self.components(source)
        composition = algebra.ExpressionComposition(arguments, budget=self.ledger)
        pulled = tuple(
            _matrix_action(
                action,
                tuple(composition(value, budget=self.ledger) for value in row),
                budget=self.ledger,
            )
            for row in source_components
        )
        values = np.zeros(
            (target.local_dof_count, source.local_dof_count), dtype=np.float64
        )
        errors = np.zeros_like(values)
        for entities in target.entity_vertices:
            for entity in entities:
                with self.ledger.temporary_scope():
                    self._refine_entity(
                        source,
                        target,
                        arguments,
                        pulled,
                        source_components,
                        entity,
                        values,
                        errors,
                    )
        result = MomentIntegralMatrix(values, errors)
        self.refine_cache[key] = result
        return result

    def _refine_entity(
        self,
        source: AbstractReferenceFormBasis,
        target: AbstractReferenceFormBasis,
        arguments: tuple[algebra.Polynomial, ...],
        pulled: _Expressions,
        source_components: _Expressions,
        entity: tuple[int, ...],
        values: NDArray[np.float64],
        errors: NDArray[np.float64],
    ) -> None:
        rows = [row for row, label in enumerate(target.dof_labels) if label[0] == entity]
        if not rows:
            return
        _, origin, chart = form_entity_chart(target.dimension, target.family, entity)
        entity_arguments = algebra.affine_arguments(origin, chart, budget=self.ledger)
        densities = self.densities(target, entity)
        if chart.shape[1] == 0:
            point = tuple(
                algebra.require_rational_reference_scalar(
                    algebra.evaluate(value, rational_vector(origin), budget=self.ledger)
                )
                for value in arguments
            )
            for target_row in rows:
                weight = algebra.expression_evaluate(
                    densities[target_row][0], (), budget=self.ledger
                )
                for source_row, components in enumerate(source_components):
                    value = self.point_value(source, components[0], point) * weight
                    values[target_row, source_row], errors[target_row, source_row] = (
                        self.integral(
                            algebra.constant(value, 0, budget=self.ledger), "point"
                        )
                    )
            return
        composition = algebra.ExpressionComposition(entity_arguments, budget=self.ledger)
        restricted = tuple(
            tuple(composition(value, budget=self.ledger) for value in row)
            for row in pulled
        )
        restricted, densities, measure, kind = _integration_factors(
            restricted, densities, target.entity_kind(entity), budget=self.ledger
        )
        for target_row in rows:
            for source_row, components in enumerate(restricted):
                with self.ledger.temporary_scope():
                    integrand = algebra.expression_sum(
                        tuple(
                            algebra.expression_multiply(weight, value, budget=self.ledger)
                            for weight, value in zip(
                                densities[target_row], components, strict=True
                            )
                            if weight and value
                        ),
                        budget=self.ledger,
                    )
                    if measure is not None:
                        integrand = algebra.expression_multiply(
                            integrand, measure, budget=self.ledger
                        )
                    values[target_row, source_row], errors[target_row, source_row] = (
                        self.integral(integrand, kind)
                    )

    def coarse_entity(
        self,
        source: AbstractReferenceFormBasis,
        target: AbstractReferenceFormBasis,
        coarse_entity: tuple[int, ...],
        fine_entity: tuple[int, ...],
        pair: _Pair,
    ) -> tuple[MomentIntegralMatrix, tuple[tuple[Fraction, ...], ...]] | None:
        arguments, argument_key = self.arguments(pair)
        cache_key = (
            source.basis_id,
            target.basis_id,
            coarse_entity,
            fine_entity,
            argument_key,
        )
        if cache_key in self.coarse_cache:
            return self.coarse_cache[cache_key]
        image = _entity_map(
            target, coarse_entity, source, fine_entity, arguments, budget=self.ledger
        )
        if image is None:
            self.coarse_cache[cache_key] = None
            return None
        fine_arguments, local, sign = image
        _, _, coarse_chart = form_entity_chart(
            target.dimension, target.family, coarse_entity
        )
        _, _, fine_chart = form_entity_chart(source.dimension, source.family, fine_entity)
        dimension = coarse_chart.shape[1]
        densities = _intrinsic_densities(
            self.densities(target, coarse_entity),
            coarse_chart,
            target.form_degree,
            dimension,
            budget=self.ledger,
        )
        map_jacobian = tuple(
            tuple(
                algebra.derivative(value, axis, budget=self.ledger)
                for axis in range(dimension)
            )
            for value in local
        )
        test_action = _pullback_matrix(
            map_jacobian,
            dimension,
            dimension,
            dimension - target.form_degree,
            dimension,
            budget=self.ledger,
        )
        fine_action = _pullback_matrix(
            _constant_matrix(fine_chart, dimension, budget=self.ledger),
            source.dimension,
            dimension,
            source.form_degree,
            dimension,
            budget=self.ledger,
        )
        source_composition = algebra.ExpressionComposition(
            fine_arguments, budget=self.ledger
        )
        source_values = tuple(
            _matrix_action(
                fine_action,
                tuple(source_composition(value, budget=self.ledger) for value in row),
                budget=self.ledger,
            )
            for row in self.components(source)
        )
        values = np.zeros(
            (target.local_dof_count, source.local_dof_count), dtype=np.float64
        )
        errors = np.zeros_like(values)
        rows = [
            row
            for row, label in enumerate(target.dof_labels)
            if label[0] == coarse_entity
        ]
        fine_densities = []
        test_composition = algebra.ExpressionComposition(local, budget=self.ledger)
        for target_row in rows:
            test = _complementary_test(
                densities[target_row], dimension, target.form_degree, budget=self.ledger
            )
            test = _matrix_action(
                test_action,
                tuple(test_composition(value, budget=self.ledger) for value in test),
                budget=self.ledger,
            )
            fine_densities.append(
                _density_from_test(
                    test, dimension, source.form_degree, budget=self.ledger
                )
            )
        source_values, integration_densities, measure, kind = _integration_factors(
            source_values,
            tuple(fine_densities),
            source.entity_kind(fine_entity),
            budget=self.ledger,
        )
        for target_row, fine_density in zip(rows, integration_densities, strict=True):
            for source_row, components in enumerate(source_values):
                with self.ledger.temporary_scope():
                    integrand = algebra.expression_scale(
                        algebra.expression_sum(
                            tuple(
                                algebra.expression_multiply(
                                    weight, value, budget=self.ledger
                                )
                                for weight, value in zip(
                                    fine_density, components, strict=True
                                )
                                if weight and value
                            ),
                            budget=self.ledger,
                        ),
                        sign,
                        budget=self.ledger,
                    )
                    if measure is not None:
                        integrand = algebra.expression_multiply(
                            integrand, measure, budget=self.ledger
                        )
                    values[target_row, source_row], errors[target_row, source_row] = (
                        self.integral(integrand, kind)
                    )
        vertices = form_reference_vertices(source.dimension, source.family)[
            list(fine_entity)
        ]
        key = tuple(
            sorted(
                tuple(
                    algebra.require_rational_reference_scalar(
                        algebra.evaluate(
                            value,
                            rational_vector(vertex),
                            budget=self.ledger,
                        )
                    )
                    for value in arguments
                )
                for vertex in vertices
            )
        )
        result = MomentIntegralMatrix(values, errors), key
        self.coarse_cache[cache_key] = result
        return result

    def common_entity(
        self,
        source: AbstractReferenceFormBasis,
        target: AbstractReferenceFormBasis,
        target_entity: tuple[int, ...],
        integration_kind: str,
        integration_entity: tuple[int, ...],
        source_arguments: tuple[algebra.Expression, ...],
        target_arguments: tuple[algebra.Expression, ...],
    ) -> tuple[MomentIntegralMatrix, tuple[tuple[Fraction, ...], ...]] | None:
        """Integrate represented source forms and target tests on one common piece.

        The two maps are independent charts from a native simplex.
        In particular, a projective target chart is not a nested-reference
        chart and is never replaced by its affine corner interpolant.
        """
        topology = reference_cell_topology(integration_kind)
        if (
            source.form_degree != target.form_degree
            or source.twist != target.twist
            or len(source_arguments) != source.dimension
        ):
            raise ValueError(
                "ExactFormMoments.common_entity.configuration must retain the scientific degree and twist; got Common form pieces must retain the scientific degree and twist."
            )
        image = _common_entity_map(
            target,
            target_entity,
            integration_kind,
            integration_entity,
            target_arguments,
            budget=self.ledger,
        )
        if image is None:
            return None
        inclusion, local, sign = image
        dimension = len(integration_entity) - 1
        if dimension < source.form_degree:
            return None
        if source.twist == "twisted":
            if (
                source.dimension != topology.dimension
                or target.dimension != topology.dimension
            ):
                raise ValueError(
                    "ExactFormMoments.common_entity.configuration must provide the source full-cell orientation bundle; got Twisted common moments require the source full-cell orientation bundle."
                )
            sign *= _full_chart_orientation(
                source_arguments, topology.dimension, budget=self.ledger
            ) * _full_chart_orientation(
                target_arguments, topology.dimension, budget=self.ledger
            )
        _, _, target_chart = form_entity_chart(
            target.dimension, target.family, target_entity
        )
        densities = _intrinsic_densities(
            self.densities(target, target_entity),
            target_chart,
            target.form_degree,
            dimension,
            budget=self.ledger,
        )
        source_chart = tuple(
            algebra.expression_compose(value, inclusion, budget=self.ledger)
            for value in source_arguments
        )
        source_jacobian = tuple(
            tuple(
                algebra.expression_derivative(value, axis, budget=self.ledger)
                for axis in range(dimension)
            )
            for value in source_chart
        )
        source_action = _pullback_matrix(
            source_jacobian,
            source.dimension,
            dimension,
            source.form_degree,
            dimension,
            budget=self.ledger,
        )
        source_composition = algebra.ExpressionComposition(
            source_chart, budget=self.ledger
        )
        source_values = tuple(
            _matrix_action(
                source_action,
                tuple(source_composition(value, budget=self.ledger) for value in row),
                budget=self.ledger,
            )
            for row in self.components(source)
        )
        target_jacobian = tuple(
            tuple(
                algebra.expression_derivative(value, axis, budget=self.ledger)
                for axis in range(dimension)
            )
            for value in local
        )
        test_action = _pullback_matrix(
            target_jacobian,
            dimension,
            dimension,
            dimension - target.form_degree,
            dimension,
            budget=self.ledger,
        )
        values = np.zeros(
            (target.local_dof_count, source.local_dof_count), dtype=np.float64
        )
        errors = np.zeros_like(values)
        kind = (
            "point"
            if dimension == 0
            else ("interval", "triangle", "tetrahedron")[dimension - 1]
        )
        test_composition = algebra.ExpressionComposition(local, budget=self.ledger)
        for target_row, label in enumerate(target.dof_labels):
            if label[0] != target_entity:
                continue
            test = _complementary_test(
                densities[target_row], dimension, target.form_degree, budget=self.ledger
            )
            test = _matrix_action(
                test_action,
                tuple(test_composition(value, budget=self.ledger) for value in test),
                budget=self.ledger,
            )
            density = _density_from_test(
                test, dimension, source.form_degree, budget=self.ledger
            )
            for source_row, components in enumerate(source_values):
                with self.ledger.temporary_scope():
                    integrand = algebra.expression_scale(
                        algebra.expression_sum(
                            tuple(
                                algebra.expression_multiply(
                                    weight, value, budget=self.ledger
                                )
                                for weight, value in zip(density, components, strict=True)
                                if weight and value
                            ),
                            budget=self.ledger,
                        ),
                        sign,
                        budget=self.ledger,
                    )
                    values[target_row, source_row], errors[target_row, source_row] = (
                        self.integral(integrand, kind)
                    )
        vertices = tuple(topology.vertices[index] for index in integration_entity)
        key = tuple(
            sorted(
                tuple(
                    algebra.require_rational_reference_scalar(
                        algebra.expression_evaluate(
                            value,
                            rational_vector(vertex),
                            budget=self.ledger,
                        )
                    )
                    for value in target_arguments
                )
                for vertex in vertices
            )
        )
        return MomentIntegralMatrix(values, errors), key
