# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Source-faithful exterior moment integrals on declared reference entities.

Coarsening pulls back the complementary test form. It never inverts a chart or
assumes that circulation/flux functionals form an orthonormal coefficient basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from .. import _coordinate_enclosure as algebra
from .._cell_geometry_transfer import _integrate_mapped_polynomial
from .._nested_reference import _NestedReferencePair, _PolynomialReferencePair
from .._reference_cell import reference_cell_topology
from ._form_elements import _form_entity_chart, _form_reference_vertices, FormBasis


type _Expressions = tuple[tuple[algebra.Expression, ...], ...]
type _Pair = _NestedReferencePair | _PolynomialReferencePair

type _ArgumentKey = tuple[tuple[tuple[tuple[int, ...], Fraction], ...], ...]


def _argument_key(arguments: tuple[algebra.Polynomial, ...], /) -> _ArgumentKey:
    return tuple(tuple(sorted(value.items())) for value in arguments)


def _identity_reference_pair(pair: _Pair, /) -> bool:
    if not isinstance(pair, _NestedReferencePair):
        return False
    dimension = pair.matrix.shape[0]
    return (
        pair.matrix.shape == (dimension, dimension)
        and pair.offset.shape == (dimension,)
        and np.array_equal(pair.matrix, np.eye(dimension, dtype=np.float64))
        and np.array_equal(pair.offset, np.zeros(dimension, dtype=np.float64))
    )


@dataclass(frozen=True)
class MomentIntegralMatrix:
    value: NDArray[np.float64]
    error: NDArray[np.float64]


def reference_arguments(pair: _Pair) -> tuple[algebra.Polynomial, ...]:
    if isinstance(pair, _PolynomialReferencePair):
        return pair.arguments
    return algebra.affine_arguments(pair.offset, pair.matrix)


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
                f"No exact compatible moment integral for reference topology {kind!r}."
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
            )
            for source in source_blades
        )
        for target in target_blades
    )


def _matrix_action(
    matrix: _Expressions, values: tuple[algebra.Expression, ...]
) -> tuple[algebra.Expression, ...]:
    return tuple(
        algebra.expression_sum(
            tuple(
                algebra.expression_multiply(coefficient, value)
                for coefficient, value in zip(row, values, strict=True)
                if coefficient and value
            )
        )
        for row in matrix
    )


def _constant_matrix(matrix: NDArray[np.float64], variables: int) -> _Expressions:
    return tuple(
        tuple(algebra.constant(Fraction(float(value)), variables) for value in row)
        for row in matrix
    )


def _wedge_sign(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    return (-1) ** sum(a > b for a in first for b in second)


def _complementary_test(
    density: tuple[algebra.Expression, ...], dimension: int, degree: int
) -> tuple[algebra.Expression, ...]:
    blades = tuple(combinations(range(dimension), degree))
    dual = tuple(combinations(range(dimension), dimension - degree))
    result: list[algebra.Expression] = [{} for _ in dual]
    for blade, value in zip(blades, density, strict=True):
        other = tuple(axis for axis in range(dimension) if axis not in blade)
        result[dual.index(other)] = algebra.expression_scale(
            value, _wedge_sign(blade, other)
        )
    return tuple(result)


def _density_from_test(
    test: tuple[algebra.Expression, ...], dimension: int, degree: int
) -> tuple[algebra.Expression, ...]:
    blades = tuple(combinations(range(dimension), degree))
    dual = tuple(combinations(range(dimension), dimension - degree))
    return tuple(
        algebra.expression_scale(test[dual.index(other)], _wedge_sign(blade, other))
        for blade in blades
        for other in (tuple(axis for axis in range(dimension) if axis not in blade),)
    )


def _intrinsic_densities(
    densities: _Expressions,
    chart: NDArray[np.float64],
    degree: int,
    variables: int,
) -> _Expressions:
    ambient, intrinsic = chart.shape
    embedding = _pullback_matrix(
        _constant_matrix(chart, variables), ambient, intrinsic, degree, variables
    )
    rows = tuple(combinations(range(intrinsic), degree))
    columns = tuple(combinations(range(ambient), degree))
    if not rows:
        raise ValueError("An entity cannot support a higher-degree form functional.")
    pivot: tuple[int, ...] | None = None
    exact: list[list[Fraction]] = []
    for selected in combinations(range(len(columns)), len(rows)):
        values = [
            [
                algebra.expression_evaluate(
                    embedding[row][column], (Fraction(0),) * variables
                )
                for row in range(len(rows))
            ]
            for column in selected
        ]
        determinant = algebra.expression_determinant(
            tuple(tuple(algebra.constant(value, 0) for value in row) for row in values),
            variable_dimension=0,
        )
        if algebra.expression_evaluate(determinant, ()):
            pivot, exact = selected, values
            break
    if pivot is None:
        raise ValueError("A declared form entity has a singular blade chart.")
    result = []
    for density in densities:
        indices = sorted(
            {
                index
                for column in pivot
                for index in algebra.expression_parts(density[column], variables)[0]
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
                    "Rational trace densities require their owning intrinsic chart action."
                )
            result.append(density)
            continue
        numerators = tuple(
            algebra.expression_parts(value, variables)[0] for value in density
        )
        right = [
            [numerators[column].get(index, Fraction(0)) for index in indices]
            for column in pivot
        ]
        solved = algebra._solve_exact(exact, right)
        result.append(
            tuple(
                {index: value for index, value in zip(indices, row, strict=True) if value}
                for row in solved
            )
        )
    return tuple(result)


def _entity_map(
    coarse: FormBasis,
    coarse_entity: tuple[int, ...],
    fine: FormBasis,
    fine_entity: tuple[int, ...],
    arguments: tuple[algebra.Polynomial, ...],
) -> tuple[tuple[algebra.Polynomial, ...], tuple[algebra.Polynomial, ...], int] | None:
    _, origin, chart = _form_entity_chart(coarse.dimension, coarse.family, coarse_entity)
    _, fine_origin, fine_chart = _form_entity_chart(
        fine.dimension, fine.family, fine_entity
    )
    dimension = chart.shape[1]
    fine_arguments = algebra.affine_arguments(fine_origin, fine_chart)
    image = tuple(algebra.compose(value, fine_arguments) for value in arguments)
    if dimension == 0:
        if any(
            algebra.add(value, algebra.constant(-Fraction(float(offset)), 0))
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
            tuple(tuple(algebra.constant(value, 0) for value in row) for row in values),
            variable_dimension=0,
        )
        if algebra.expression_evaluate(determinant, ()):
            pivot, exact = rows, values
            break
    if pivot is None:
        raise ValueError("A declared coarse entity has a singular reference chart.")
    differences = tuple(
        algebra.add(
            image[row], algebra.constant(-Fraction(float(origin[row])), dimension)
        )
        for row in pivot
    )
    indices = sorted({index for value in differences for index in value})
    solved = algebra._solve_exact(
        exact,
        [[value.get(index, Fraction(0)) for index in indices] for value in differences],
    )
    local: tuple[algebra.Polynomial, ...] = tuple(
        {index: value for index, value in zip(indices, row, strict=True) if value}
        for row in solved
    )
    reconstructed = tuple(
        algebra.add(
            algebra.constant(Fraction(float(origin[coordinate])), dimension),
            algebra.sum_polynomials(
                tuple(
                    algebra.scale(value, Fraction(float(chart[coordinate, column])))
                    for column, value in enumerate(local)
                )
            ),
        )
        for coordinate in range(coarse.dimension)
    )
    if any(
        algebra.add(actual, algebra.scale(expected, -1))
        for actual, expected in zip(image, reconstructed, strict=True)
    ):
        return None
    jacobian = tuple(
        tuple(algebra.derivative(value, axis) for axis in range(dimension))
        for value in local
    )
    determinant = algebra.expression_determinant(jacobian, variable_dimension=dimension)
    controls = algebra.expression_bernstein_coefficients(
        determinant, _domain(fine.entity_kind(fine_entity)), dimension
    )
    if min(controls) > 0:
        sign = 1
    elif max(controls) < 0:
        sign = -1
    else:
        raise ValueError(
            "A fine entity chart has unresolved orientation or a singular Jacobian."
        )
    return fine_arguments, local, sign


def _common_entity_map(
    target: FormBasis,
    target_entity: tuple[int, ...],
    integration_kind: str,
    integration_entity: tuple[int, ...],
    arguments: tuple[algebra.Expression, ...],
) -> tuple[tuple[algebra.Polynomial, ...], tuple[algebra.Expression, ...], int] | None:
    topology = reference_cell_topology(integration_kind)
    if integration_kind not in ("interval", "triangle", "tetrahedron"):
        raise ValueError(
            "Common form pieces require an owning simplex integration topology."
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
            "The common form integration entity is not a native reference entity."
        )
    vertices = np.asarray(topology.vertices, dtype=np.float64)[list(integration_entity)]
    inclusion = algebra.affine_arguments(vertices[0], (vertices[1:] - vertices[0]).T)
    _, origin, chart = _form_entity_chart(target.dimension, target.family, target_entity)
    if chart.shape[1] != dimension or len(arguments) != target.dimension:
        return None
    image = tuple(algebra.expression_compose(value, inclusion) for value in arguments)
    if dimension == 0:
        if any(
            algebra.expression_add(value, algebra.constant(-Fraction(float(offset)), 0))
            for value, offset in zip(image, origin, strict=True)
        ):
            return None
        return inclusion, (), 1
    pivot: tuple[int, ...] | None = None
    matrix: list[list[Fraction]] = []
    for rows in combinations(range(target.dimension), dimension):
        candidate = [
            [Fraction(float(chart[row, column])) for column in range(dimension)]
            for row in rows
        ]
        determinant = algebra.expression_determinant(
            tuple(
                tuple(algebra.constant(value, 0) for value in row) for row in candidate
            ),
            variable_dimension=0,
        )
        if algebra.expression_evaluate(determinant, ()):
            pivot, matrix = rows, candidate
            break
    if pivot is None:
        raise ValueError("A common form target entity has a singular declared chart.")
    differences = tuple(
        algebra.expression_add(
            image[row], algebra.constant(-Fraction(float(origin[row])), dimension)
        )
        for row in pivot
    )
    parts = tuple(algebra.expression_parts(value, dimension) for value in differences)
    denominator = algebra.constant(1, dimension)
    for _, value in parts:
        if algebra.divide_polynomial(denominator, value) is not None:
            continue
        denominator = (
            value
            if algebra.divide_polynomial(value, denominator) is not None
            else algebra.multiply(denominator, value)
        )
    numerators = []
    for numerator, divisor in parts:
        quotient = algebra.divide_polynomial(denominator, divisor)
        if quotient is None:
            raise ValueError(
                "Common form entity coefficient denominators failed exact alignment."
            )
        numerators.append(algebra.multiply(numerator, quotient))
    indices = sorted({index for value in numerators for index in value})
    solved = algebra._solve_exact(
        matrix,
        [[value.get(index, Fraction(0)) for index in indices] for value in numerators],
    )
    local = tuple(
        algebra.rational_expression(
            {index: value for index, value in zip(indices, row, strict=True) if value},
            denominator,
        )
        for row in solved
    )
    reconstructed = tuple(
        algebra.expression_add(
            algebra.constant(Fraction(float(origin[coordinate])), dimension),
            algebra.expression_sum(
                tuple(
                    algebra.expression_scale(
                        value, Fraction(float(chart[coordinate, column]))
                    )
                    for column, value in enumerate(local)
                )
            ),
        )
        for coordinate in range(target.dimension)
    )
    if any(
        algebra.expression_add(actual, algebra.expression_scale(expected, -1))
        for actual, expected in zip(image, reconstructed, strict=True)
    ):
        return None
    jacobian = tuple(
        tuple(algebra.expression_derivative(value, axis) for axis in range(dimension))
        for value in local
    )
    determinant = algebra.expression_determinant(jacobian, variable_dimension=dimension)
    controls = algebra.expression_bernstein_coefficients(
        determinant, "simplex", dimension
    )
    if min(controls) > 0:
        sign = 1
    elif max(controls) < 0:
        sign = -1
    else:
        raise ValueError(
            "A common form entity has unresolved orientation or a singular rational chart."
        )
    return inclusion, local, sign


def _full_chart_orientation(
    arguments: tuple[algebra.Expression, ...],
    dimension: int,
) -> int:
    if len(arguments) != dimension:
        raise ValueError(
            "Twisted common moments require full owning square reference charts."
        )
    jacobian = tuple(
        tuple(algebra.expression_derivative(value, axis) for axis in range(dimension))
        for value in arguments
    )
    determinant = algebra.expression_determinant(jacobian, variable_dimension=dimension)
    controls = algebra.expression_bernstein_coefficients(
        determinant, "simplex", dimension
    )
    if min(controls) > 0:
        return 1
    if max(controls) < 0:
        return -1
    raise ValueError(
        "A twisted common chart has unresolved full-cell orientation or a singular Jacobian."
    )


def _integration_factors(
    components: _Expressions,
    densities: _Expressions,
    kind: str,
) -> tuple[_Expressions, _Expressions, algebra.Expression | None, str]:
    """Use an owning collapsed chart before expanding moment contractions."""
    if _integration_kind(kind) != "pyramid":
        return components, densities, None, kind
    arguments = algebra.chart_arguments("pyramid", 3)
    composition = algebra.ExpressionComposition(arguments)
    # These are coefficient functions, not blade pullbacks: their contraction
    # already represents the physical reference-entity moment density.
    components = tuple(tuple(composition(value) for value in row) for row in components)
    densities = tuple(tuple(composition(value) for value in row) for row in densities)
    jacobian = tuple(
        tuple(algebra.derivative(value, axis) for axis in range(3)) for value in arguments
    )
    measure = algebra.expression_determinant(jacobian, variable_dimension=3)
    return components, densities, measure, "hexahedron"


@dataclass
class ExactFormMoments:
    ledger: algebra.CoordinateEnclosureBudget
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

    def _retain(self, values: _Expressions, variables: int) -> None:
        polynomials = tuple(
            polynomial
            for row in values
            for value in row
            for polynomial in algebra.expression_parts(value, variables)
        )
        self.ledger.retain_basis(polynomials)

    def components(self, basis: FormBasis) -> _Expressions:
        key = basis.basis_id
        if key not in self.components_cache:
            values = basis.component_expressions()
            self._retain(values, basis.dimension)
            self.components_cache[key] = values
        return self.components_cache[key]

    def densities(self, basis: FormBasis, entity: tuple[int, ...]) -> _Expressions:
        key = (basis.basis_id, entity)
        if key not in self.densities_cache:
            values = basis.functional_density_expressions(entity)
            dimension = _form_entity_chart(basis.dimension, basis.family, entity)[
                2
            ].shape[1]
            self._retain(values, dimension)
            self.densities_cache[key] = values
        return self.densities_cache[key]

    def integral(self, value: algebra.Expression, kind: str) -> tuple[float, float]:
        native = _integration_kind(kind)
        if native == "pyramid":
            arguments = algebra.chart_arguments(native, 3)
            jacobian = tuple(
                tuple(algebra.derivative(value, axis) for axis in range(3))
                for value in arguments
            )
            value = algebra.expression_multiply(
                algebra.expression_compose(value, arguments),
                algebra.expression_determinant(jacobian, variable_dimension=3),
            )
        return _integrate_mapped_polynomial(value, native)

    def point_value(
        self,
        basis: FormBasis,
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
                    "A scalar pyramid vertex is outside its exact reference apex."
                )
            collapsed = algebra.expression_compose(
                value, algebra.chart_arguments("pyramid", 3)
            )
            u, v = algebra.axes(2)
            trace = algebra.expression_compose(collapsed, (u, v, algebra.constant(1, 2)))
            result = algebra.expression_evaluate(trace, (Fraction(1, 2), Fraction(1, 2)))
            if algebra.expression_add(trace, algebra.constant(-result, 2)):
                raise ValueError(
                    "The scalar pyramid apex depends on the reference approach."
                )
            return result
        return algebra.expression_evaluate(value, point)

    def refine(
        self, source: FormBasis, target: FormBasis, pair: _Pair
    ) -> MomentIntegralMatrix:
        if (
            source.basis_id == target.basis_id
            and source.local_dof_count == target.local_dof_count
            and _identity_reference_pair(pair)
        ):
            return MomentIntegralMatrix(
                np.eye(source.local_dof_count, dtype=np.float64),
                np.zeros(
                    (target.local_dof_count, source.local_dof_count),
                    dtype=np.float64,
                ),
            )
        arguments = reference_arguments(pair)
        key = (source.basis_id, target.basis_id, _argument_key(arguments))
        cached = self.refine_cache.get(key)
        if cached is not None:
            return cached
        jacobian = tuple(
            tuple(algebra.derivative(value, axis) for axis in range(source.dimension))
            for value in arguments
        )
        action = _pullback_matrix(
            jacobian,
            source.dimension,
            target.dimension,
            source.form_degree,
            source.dimension,
        )
        source_components = self.components(source)
        composition = algebra.ExpressionComposition(arguments)
        pulled = tuple(
            _matrix_action(action, tuple(composition(value) for value in row))
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
        source: FormBasis,
        target: FormBasis,
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
        _, origin, chart = _form_entity_chart(target.dimension, target.family, entity)
        entity_arguments = algebra.affine_arguments(origin, chart)
        densities = self.densities(target, entity)
        if chart.shape[1] == 0:
            point = tuple(
                algebra.evaluate(value, tuple(Fraction(float(x)) for x in origin))
                for value in arguments
            )
            for target_row in rows:
                weight = algebra.expression_evaluate(densities[target_row][0], ())
                for source_row, components in enumerate(source_components):
                    value = self.point_value(source, components[0], point) * weight
                    values[target_row, source_row], errors[target_row, source_row] = (
                        self.integral(algebra.constant(value, 0), "point")
                    )
            return
        composition = algebra.ExpressionComposition(entity_arguments)
        restricted = tuple(tuple(composition(value) for value in row) for row in pulled)
        restricted, densities, measure, kind = _integration_factors(
            restricted, densities, target.entity_kind(entity)
        )
        for target_row in rows:
            for source_row, components in enumerate(restricted):
                with self.ledger.temporary_scope():
                    integrand = algebra.expression_sum(
                        tuple(
                            algebra.expression_multiply(weight, value)
                            for weight, value in zip(
                                densities[target_row], components, strict=True
                            )
                            if weight and value
                        )
                    )
                    if measure is not None:
                        integrand = algebra.expression_multiply(integrand, measure)
                    values[target_row, source_row], errors[target_row, source_row] = (
                        self.integral(integrand, kind)
                    )

    def coarse_entity(
        self,
        source: FormBasis,
        target: FormBasis,
        coarse_entity: tuple[int, ...],
        fine_entity: tuple[int, ...],
        pair: _Pair,
    ) -> tuple[MomentIntegralMatrix, tuple[tuple[Fraction, ...], ...]] | None:
        arguments = reference_arguments(pair)
        cache_key = (
            source.basis_id,
            target.basis_id,
            coarse_entity,
            fine_entity,
            _argument_key(arguments),
        )
        if cache_key in self.coarse_cache:
            return self.coarse_cache[cache_key]
        image = _entity_map(target, coarse_entity, source, fine_entity, arguments)
        if image is None:
            self.coarse_cache[cache_key] = None
            return None
        fine_arguments, local, sign = image
        _, _, coarse_chart = _form_entity_chart(
            target.dimension, target.family, coarse_entity
        )
        _, _, fine_chart = _form_entity_chart(
            source.dimension, source.family, fine_entity
        )
        dimension = coarse_chart.shape[1]
        densities = _intrinsic_densities(
            self.densities(target, coarse_entity),
            coarse_chart,
            target.form_degree,
            dimension,
        )
        map_jacobian = tuple(
            tuple(algebra.derivative(value, axis) for axis in range(dimension))
            for value in local
        )
        test_action = _pullback_matrix(
            map_jacobian, dimension, dimension, dimension - target.form_degree, dimension
        )
        fine_action = _pullback_matrix(
            _constant_matrix(fine_chart, dimension),
            source.dimension,
            dimension,
            source.form_degree,
            dimension,
        )
        source_composition = algebra.ExpressionComposition(fine_arguments)
        source_values = tuple(
            _matrix_action(fine_action, tuple(source_composition(value) for value in row))
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
        test_composition = algebra.ExpressionComposition(local)
        for target_row in rows:
            test = _complementary_test(
                densities[target_row], dimension, target.form_degree
            )
            test = _matrix_action(
                test_action, tuple(test_composition(value) for value in test)
            )
            fine_densities.append(_density_from_test(test, dimension, source.form_degree))
        source_values, integration_densities, measure, kind = _integration_factors(
            source_values, tuple(fine_densities), source.entity_kind(fine_entity)
        )
        for target_row, fine_density in zip(rows, integration_densities, strict=True):
            for source_row, components in enumerate(source_values):
                with self.ledger.temporary_scope():
                    integrand = algebra.expression_scale(
                        algebra.expression_sum(
                            tuple(
                                algebra.expression_multiply(weight, value)
                                for weight, value in zip(
                                    fine_density, components, strict=True
                                )
                                if weight and value
                            )
                        ),
                        sign,
                    )
                    if measure is not None:
                        integrand = algebra.expression_multiply(integrand, measure)
                    values[target_row, source_row], errors[target_row, source_row] = (
                        self.integral(integrand, kind)
                    )
        vertices = _form_reference_vertices(source.dimension, source.family)[
            list(fine_entity)
        ]
        key = tuple(
            sorted(
                tuple(
                    algebra.evaluate(value, tuple(Fraction(float(x)) for x in vertex))
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
        source: FormBasis,
        target: FormBasis,
        target_entity: tuple[int, ...],
        integration_kind: str,
        integration_entity: tuple[int, ...],
        source_arguments: tuple[algebra.Expression, ...],
        target_arguments: tuple[algebra.Expression, ...],
    ) -> tuple[MomentIntegralMatrix, tuple[tuple[Fraction, ...], ...]] | None:
        """Integrate actual source forms and target tests on one common piece.

        The two maps are independent owning charts from a native simplex.
        In particular, a projective target chart is not a nested-reference
        witness and is never replaced by its affine corner interpolant.
        """
        topology = reference_cell_topology(integration_kind)
        if (
            source.form_degree != target.form_degree
            or source.twist != target.twist
            or len(source_arguments) != source.dimension
        ):
            raise ValueError(
                "Common form pieces must retain the scientific degree and twist."
            )
        image = _common_entity_map(
            target, target_entity, integration_kind, integration_entity, target_arguments
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
                    "Twisted common moments require the original full-cell orientation bundle."
                )
            sign *= _full_chart_orientation(
                source_arguments, topology.dimension
            ) * _full_chart_orientation(target_arguments, topology.dimension)
        _, _, target_chart = _form_entity_chart(
            target.dimension, target.family, target_entity
        )
        densities = _intrinsic_densities(
            self.densities(target, target_entity),
            target_chart,
            target.form_degree,
            dimension,
        )
        source_chart = tuple(
            algebra.expression_compose(value, inclusion) for value in source_arguments
        )
        source_jacobian = tuple(
            tuple(algebra.expression_derivative(value, axis) for axis in range(dimension))
            for value in source_chart
        )
        source_action = _pullback_matrix(
            source_jacobian, source.dimension, dimension, source.form_degree, dimension
        )
        source_composition = algebra.ExpressionComposition(source_chart)
        source_values = tuple(
            _matrix_action(
                source_action, tuple(source_composition(value) for value in row)
            )
            for row in self.components(source)
        )
        target_jacobian = tuple(
            tuple(algebra.expression_derivative(value, axis) for axis in range(dimension))
            for value in local
        )
        test_action = _pullback_matrix(
            target_jacobian,
            dimension,
            dimension,
            dimension - target.form_degree,
            dimension,
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
        test_composition = algebra.ExpressionComposition(local)
        for target_row, label in enumerate(target.dof_labels):
            if label[0] != target_entity:
                continue
            test = _complementary_test(
                densities[target_row], dimension, target.form_degree
            )
            test = _matrix_action(
                test_action, tuple(test_composition(value) for value in test)
            )
            density = _density_from_test(test, dimension, source.form_degree)
            for source_row, components in enumerate(source_values):
                with self.ledger.temporary_scope():
                    integrand = algebra.expression_scale(
                        algebra.expression_sum(
                            tuple(
                                algebra.expression_multiply(weight, value)
                                for weight, value in zip(density, components, strict=True)
                                if weight and value
                            )
                        ),
                        sign,
                    )
                    values[target_row, source_row], errors[target_row, source_row] = (
                        self.integral(integrand, kind)
                    )
        vertices = tuple(topology.vertices[index] for index in integration_entity)
        key = tuple(
            sorted(
                tuple(
                    algebra.expression_evaluate(
                        value, tuple(Fraction(float(x)) for x in vertex)
                    )
                    for value in target_arguments
                )
                for vertex in vertices
            )
        )
        return MomentIntegralMatrix(values, errors), key
