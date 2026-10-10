#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scalar transport through actual full-sphere projective material correspondence.

Physical reference maps are rational, not affine maps fitted to corner images.
Every mixed form uses the true projective quotient/Jacobian on an exact old
reference overlap triangle. Positive-denominator series carry an explicit
remainder and reuse the canonical signed sqrt-Gram integration owner.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Literal, overload, TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...linalg import ArraySpace
from ...linalg._hermitian_spectral import _fraction_sqrt_interval
from ...sparse import RowRelation, SparseLinearMap
from .._cell_geometry import CellGeometrySpec
from .._cell_geometry_transfer import (
    _embedded_measure_submaps,
    _embedded_squared_density,
    _mapped_geometry_cells,
    _sqrt_polynomial_integral_piece,
    CellGeometryTransitionError,
)
from .._coordinate_enclosure import (
    add,
    affine_arguments,
    bernstein_coefficients,
    compose,
    constant,
    Expression,
    multiply,
    outward,
    Polynomial,
    rational_expression,
    RationalPolynomial,
    scale,
    source_basis,
    sum_polynomials,
)
from ._generic import FiniteElementDiscretization
from ._surface_chart_transfer import (
    _binding,
    _bound_surface_endpoint,
    _ChartEndpoint,
    _endpoint_cell_measures,
    _field_cells,
    _finish_material_dg_projection,
    _integral,
    _IntegrationBudgets,
    _physical_rows,
    _restricted,
    PreparedSurfaceChartFiniteVolumeContents,
)
from ._topology_transfer import (
    _mapped_dg_cells,
    _owned_rows,
    _tabulated,
    FiniteElementFieldTransfer,
    FiniteElementTopologyTransfer,
    FiniteElementTransferEvidence,
)


if TYPE_CHECKING:
    from ...geometry._sphere_material_atlas import SphereProjectiveReferenceMap
    from .._sphere_chart_deformation import (
        PreparedSphereChartDeformation,
        PreparedSphereChartPiece,
    )

from .. import _coordinate_enclosure as algebra
from ._mapped_form_transfer import _PreparationWork
from ._surface_chart_compatible import (
    _MaterialCompatiblePiece,
    _prepare_material_chart_compatible_transfer,
    PreparedSurfaceChartCompatibleTransfer,
)


type _Point = tuple[Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]
type _CellIndices = tuple[dict[int, int], dict[int, int]]
type _FloatArray = NDArray[np.float64]


class _ProjectiveBudgets(TypedDict):
    absolute_tolerance: float | Fraction
    relative_tolerance: float | Fraction
    maximum_work: int
    maximum_subcells: int
    maximum_binomial_terms: int


def _validate_sphere(
    source: _ChartEndpoint,
    target: _ChartEndpoint,
    prepared: PreparedSphereChartDeformation,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
) -> _CellIndices:
    from .._sphere_chart_deformation import PreparedSphereChartDeformation

    if not isinstance(prepared, PreparedSphereChartDeformation):
        raise TypeError("An actual prepared sphere chart deformation is required.")
    _bound_surface_endpoint(source, source_geometry, "source", prepared)
    _bound_surface_endpoint(target, target_geometry, "target", prepared)
    prepared.require_bound(source.mesh, source_geometry, target.mesh, target_geometry)
    indices: list[dict[int, int]] = []
    for discretization, atlas in (
        (source, prepared.source_atlas),
        (target, prepared.target_atlas),
    ):
        ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in discretization.mesh.blocks
            ]
        )
        if np.unique(ids).size != ids.size or set(ids.tolist()) != set(
            np.asarray(atlas.cell_global_ids, dtype=np.int64).tolist()
        ):
            raise ValueError(
                "Sphere correspondence does not bind the current scientific cells."
            )
        indices.append({int(identity): row for row, identity in enumerate(ids)})
    return indices[0], indices[1]


def _orientation(first: _Point, second: _Point, third: _Point) -> Fraction:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _contains_exact(vertices: _Triangle, point: _Point) -> bool:
    direction = _orientation(*vertices)
    if direction == 0:
        raise ValueError("An exact sphere reference overlap is degenerate.")
    return all(
        _orientation(vertices[i], vertices[(i + 1) % 3], point) * direction >= 0
        for i in range(3)
    )


def _map_exact(projective_map: SphereProjectiveReferenceMap, point: _Point) -> _Point:
    homogeneous = tuple(
        row[0] + row[1] * point[0] + row[2] * point[1]
        for row in projective_map.exact_coefficients
    )
    denominator = sum(homogeneous, Fraction(0))
    if denominator <= 0 or any(value < 0 for value in homogeneous):
        raise ValueError(
            "An actual sphere chart functional leaves its positive reference cone."
        )
    return homogeneous[1] / denominator, homogeneous[2] / denominator


def _prepare_h1(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    indices: _CellIndices,
    name: str,
) -> FiniteElementFieldTransfer:
    si, ti = source._field_index(name), target._field_index(name)
    source_cells, target_cells = _field_cells(source, si), _field_cells(target, ti)
    if any(
        element.conformity != "H1" or source_basis(element) is None
        for element, _ in source_cells + target_cells
    ):
        raise ValueError(
            "Sphere H1 transport requires actual canonical scalar nodal sources."
        )
    source_indices, target_indices = indices
    inverse_maps: dict[tuple[int, int], SphereProjectiveReferenceMap] = {}
    candidates, columns, values = [], [], []
    for piece in prepared.pieces:
        source_row = source_indices[int(piece.source_cell_global_id)]
        target_row = target_indices[int(piece.target_cell_global_id)]
        source_element, source_route = source_cells[source_row]
        target_element, target_route = target_cells[target_row]
        pair = int(piece.source_cell_global_id), int(piece.target_cell_global_id)
        inverse = inverse_maps.get(pair)
        if inverse is None:
            inverse = prepared.target_atlas.projective_reference_map(
                pair[1], prepared.source_atlas, pair[0]
            )
            inverse_maps[pair] = inverse
        nodes = np.asarray(target_element.reference_nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(
                "Sphere H1 functionals require two-dimensional reference nodes."
            )
        for node in range(nodes.shape[0]):
            point = Fraction(float(nodes[node, 0])), Fraction(float(nodes[node, 1]))
            if not _contains_exact(piece.exact_target_reference_vertices, point):
                continue
            reference = _map_exact(inverse, point)
            if not _contains_exact(piece.exact_source_reference_vertices, reference):
                raise ValueError(
                    "A projective target functional has no exact old overlap witness."
                )
            basis, _ = _tabulated(
                source_element,
                np.asarray([[float(value) for value in reference]], dtype=np.float64),
            )
            if not np.all(np.isfinite(basis)):
                raise ValueError("Sphere source nodal functionals are not finite.")
            candidates.append(int(target_route[node]))
            columns.append(source_route)
            values.append(basis[0])
    width = max(element.local_dof_count for element, _ in source_cells)
    padded_columns = np.zeros((len(candidates), width), dtype=np.int64)
    padded_values = np.zeros(padded_columns.shape, dtype=np.float64)
    for row, (route, value) in enumerate(zip(columns, values, strict=True)):
        padded_columns[row, : route.size], padded_values[row, : route.size] = route, value
    ss, ts = source.dof_maps[si].global_dof_count, target.dof_maps[ti].global_dof_count
    rows, coefficients, continuity = _owned_rows(
        np.asarray(candidates, dtype=np.int64)[:, None],
        padded_columns,
        padded_values[:, None],
        ts,
        ss,
    )
    amplification = max(1.0, width * float(np.max(np.sum(np.abs(coefficients), axis=1))))
    evidence = FiniteElementTransferEvidence(
        {
            "continuity": continuity,
            "constants": float(np.max(np.abs(coefficients.sum(axis=1) - 1))),
        },
        64 * np.finfo(np.float64).eps * amplification,
    )
    if not evidence.passed:
        raise ValueError("Sphere rational H1 nodal/trace certificate failed.")
    primal = SparseLinearMap(
        RowRelation(rows.astype(np.int32), source_size=ss),
        coefficients,
        operator_id=canonical_fingerprint(
            {
                "kind": "sphere-rational-h1",
                "field": name,
                "source": source.prepared_id,
                "target": target.prepared_id,
                "deformation": prepared.deformation_id,
                "routes": array_tree_fingerprint(rows),
                "values": array_tree_fingerprint(coefficients),
            }
        ),
    )
    transfer = FiniteElementTopologyTransfer(
        primal,
        prepared.source_topology_id,
        prepared.target_topology_id,
        preserves_constants=True,
        positivity_preserving=bool(np.all(coefficients >= 0)),
        semantics="interpolation",
        action_condition=amplification,
    )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[si],
        target.field_spaces[ti],
        _binding(prepared),
        evidence,
    )


def _degree(polynomial: Polynomial) -> int:
    return max((sum(index) for index in polynomial), default=0)


def _power(polynomial: Polynomial, exponent: int) -> Polynomial:
    result = constant(1, 2)
    for _ in range(exponent):
        result = multiply(result, polynomial)
    return result


def _homogeneous(
    polynomial: Polynomial,
    numerators: Sequence[Polynomial],
    denominator: Polynomial,
    degree: int,
) -> Polynomial:
    """Exact numerator after substituting N/D, with one positive D power."""
    powers = [
        tuple(_power(value, exponent) for exponent in range(degree + 1))
        for value in (*numerators, denominator)
    ]
    return sum_polynomials(
        tuple(
            scale(
                multiply(
                    multiply(powers[0][index[0]], powers[1][index[1]]),
                    powers[2][degree - sum(index)],
                ),
                coefficient,
            )
            for index, coefficient in polynomial.items()
        )
    )


def _projective_rational_density(
    gram: Expression,
    numerators: Sequence[Polynomial],
    denominator: Polynomial,
    jacobian_numerator: Fraction,
) -> tuple[Expression, Polynomial, int]:
    """Keep exact positive even denominator/Jacobian factors outside sqrt(Gram)."""
    if not isinstance(gram, RationalPolynomial):
        raise ValueError(
            "Rational sphere density substitution requires its owning exact quotient."
        )
    controls = bernstein_coefficients(denominator, "simplex", 2)
    if min(controls) <= 0:
        raise ValueError(
            "Rational sphere reference action has no positive denominator proof."
        )
    numerator_degree, denominator_degree = (
        _degree(gram.numerator),
        _degree(gram.denominator),
    )
    numerator = _homogeneous(gram.numerator, numerators, denominator, numerator_degree)
    divisor = _homogeneous(gram.denominator, numerators, denominator, denominator_degree)
    degree_difference = denominator_degree - numerator_degree
    parity = degree_difference % 2
    if parity:
        numerator = multiply(numerator, denominator)
    outside_power = (degree_difference - parity) // 2 - 3
    weight = scale(_power(denominator, max(outside_power, 0)), abs(jacobian_numerator))
    return rational_expression(numerator, divisor), weight, max(-outside_power, 0)


def _projective_arguments(
    piece: PreparedSphereChartPiece, source_arguments: Sequence[Polynomial]
) -> tuple[tuple[Polynomial, ...], Polynomial]:
    homogeneous = tuple(
        sum_polynomials(
            (
                constant(row[0], 2),
                scale(source_arguments[0], row[1]),
                scale(source_arguments[1], row[2]),
            )
        )
        for row in piece.projective_map.exact_coefficients
    )
    return homogeneous[1:], sum_polynomials(homogeneous)


@overload
def _projective_integral(
    gram: Expression,
    numerator: Polynomial,
    denominator: Polynomial,
    exponent: int,
    budgets: _ProjectiveBudgets,
    *,
    fraction_result: Literal[True],
) -> tuple[Fraction, Fraction]: ...


@overload
def _projective_integral(
    gram: Expression,
    numerator: Polynomial,
    denominator: Polynomial,
    exponent: int,
    budgets: _ProjectiveBudgets,
    *,
    fraction_result: Literal[False] = False,
) -> tuple[float, float]: ...


def _projective_integral(
    gram: Expression,
    numerator: Polynomial,
    denominator: Polynomial,
    exponent: int,
    budgets: _ProjectiveBudgets,
    *,
    fraction_result: bool = False,
) -> tuple[Fraction | float, Fraction | float]:
    """Enclose sqrt(Gram)*numerator/D**exponent using the canonical root owner.

    The reciprocal series is exact polynomial data with a uniform remainder.
    Fractions may be retained across a polynomial moment functional, so a DG
    matrix is published once, rather than rounding every intermediate moment.
    Density and denominator subdivisions are genuine source-reference maps.
    Resource exhaustion never switches to a sampled or affine approximation.
    """
    if not numerator:
        return (Fraction(0), Fraction(0)) if fraction_result else (0.0, 0.0)
    if exponent == 0:
        denominator = constant(1, 2)
    absolute, relative = (
        Fraction(budgets["absolute_tolerance"]),
        Fraction(budgets["relative_tolerance"]),
    )
    if absolute < 0 or relative < 0 or absolute == relative == 0:
        raise ValueError("Projective integration requires positive error budgets.")
    maximum_terms, maximum_subcells = (
        int(budgets["maximum_binomial_terms"]),
        int(budgets["maximum_subcells"]),
    )
    if maximum_terms <= 0 or maximum_subcells <= 0:
        raise ValueError("Projective integration requires positive work limits.")
    if isinstance(gram, RationalPolynomial):
        from .._cell_geometry_transfer import _certified_sqrt_prepare_expression_integral
        from .._coordinate_enclosure import (
            _COORDINATE_BUDGET,
            CoordinateEnclosureBudget,
            CoordinateEnclosureResourceError,
        )

        if _COORDINATE_BUDGET.get() is None:
            import sys

            ledger = CoordinateEnclosureBudget(int(budgets["maximum_work"]), sys.maxsize)
            try:
                with ledger.activate():
                    if fraction_result:
                        return _projective_integral(
                            gram,
                            numerator,
                            denominator,
                            exponent,
                            budgets,
                            fraction_result=True,
                        )
                    return _projective_integral(
                        gram, numerator, denominator, exponent, budgets
                    )
            except CoordinateEnclosureResourceError as exhausted:
                raise CellGeometryTransitionError(
                    "resource_limit",
                    "Rational projective source coefficient budget exhausted.",
                    measured=exhausted.requested,
                    limit=exhausted.limit,
                ) from exhausted
        divisor_lower = min(bernstein_coefficients(gram.denominator, "simplex", 2))
        reference_lower = min(bernstein_coefficients(denominator, "simplex", 2))
        if min(divisor_lower, reference_lower) <= 0:
            raise ValueError(
                "Rational projective density has no positive denominator proof."
            )
        density_upper = (
            _fraction_sqrt_interval(
                max(bernstein_coefficients(gram.numerator, "simplex", 2)) / divisor_lower
            )[1]
            / reference_lower**exponent
        )
        magnitude = max(
            abs(value) for value in bernstein_coefficients(numerator, "simplex", 2)
        )
        goal = absolute + relative * density_upper * magnitude / 2
        work = [0, int(budgets["maximum_work"])]
        for refinement in range(maximum_terms):
            try:
                prepared = _certified_sqrt_prepare_expression_integral(
                    gram,
                    "triangle",
                    goal / (max(magnitude, Fraction(1)) * 2**refinement),
                    work,
                    maximum_subcells,
                    maximum_terms,
                    denominator=denominator,
                    denominator_exponent=exponent,
                )
                value, error = prepared.integral(numerator)
            except CoordinateEnclosureResourceError as exhausted:
                raise CellGeometryTransitionError(
                    "resource_limit",
                    "Rational projective source coefficient budget exhausted.",
                    measured=exhausted.requested,
                    limit=exhausted.limit,
                ) from exhausted
            allowed = absolute + relative * max(
                abs(Fraction(value)) - Fraction(error), Fraction(0)
            )
            if Fraction(error) <= allowed:
                return (
                    (Fraction(value), Fraction(error))
                    if fraction_result
                    else (value, error)
                )
        raise CellGeometryTransitionError(
            "resource_limit",
            "Rational projective publication exceeds its exact requested error budget.",
            measured=error,
            limit=float(allowed),
        )
    lower = min(bernstein_coefficients(denominator, "simplex", 2))
    if lower <= 0:
        raise ValueError(
            "Projective density integration requires a positive denominator enclosure."
        )
    gram_upper = max(bernstein_coefficients(gram, "simplex", 2))
    magnitude = max(
        abs(value) for value in bernstein_coefficients(numerator, "simplex", 2)
    )
    scale_bound = (
        _fraction_sqrt_interval(gram_upper)[1] * magnitude / (2 * lower**exponent)
    )
    goal = absolute + relative * scale_bound
    pieces, pending, visited = [], [(gram, numerator, denominator, Fraction(1))], 0
    work = [0, int(budgets["maximum_work"])]

    def product(first: Polynomial, second: Polynomial) -> Polynomial:
        work[0] += len(first) * len(second)
        if work[0] > work[1]:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Projective polynomial work budget exhausted.",
                measured=work[0],
                limit=work[1],
            )
        return multiply(first, second)

    def split(
        g: Polynomial,
        weight: Polynomial,
        d: Polynomial,
        fraction: Fraction,
    ) -> Iterator[tuple[Polynomial, Polynomial, Polynomial, Fraction]]:
        for matrix, offset in _embedded_measure_submaps("triangle"):
            arguments = affine_arguments(offset, matrix)
            yield (
                compose(g, arguments),
                compose(weight, arguments),
                compose(d, arguments),
                fraction / 4,
            )

    while pending:
        g, weight, d, fraction = pending.pop()
        visited += 1
        if visited > maximum_subcells:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Projective denominator subdivision budget exhausted.",
                measured=visited,
                limit=maximum_subcells,
            )
        controls = bernstein_coefficients(d, "simplex", 2)
        lo, hi = min(controls), max(controls)
        if lo <= 0:
            raise ValueError(
                "Projective subdivision lost its positive denominator proof."
            )
        rho = (hi - lo) / (hi + lo)
        if rho > Fraction(1, 8):
            pending.extend(split(g, weight, d, fraction))
        else:
            pieces.append((g, weight, d, fraction, lo, hi, rho))
    refinement = 0
    while refinement < maximum_terms:
        local_goal = goal / (2**refinement)
        central, error = Fraction(0), Fraction(0)
        refined_pieces, complete = [], True
        for g, weight, d, fraction, lo, hi, rho in pieces:
            center = (lo + hi) / 2
            normalized = add(scale(d, 1 / center), constant(-1, 2))
            series: Polynomial = {}
            power, coefficient = constant(1, 2), Fraction(1)
            density_upper = _fraction_sqrt_interval(
                max(bernstein_coefficients(g, "simplex", 2))
            )[1]
            weight_upper = max(
                abs(value) for value in bernstein_coefficients(weight, "simplex", 2)
            )
            remainder_error = None
            for order in range(maximum_terms):
                series = add(series, scale(power, coefficient))
                next_coefficient = -coefficient * Fraction(exponent + order, order + 1)
                ratio = rho * Fraction(exponent + order + 1, order + 2)
                if ratio < 1:
                    tail = abs(next_coefficient) * rho ** (order + 1) / (1 - ratio)
                    bound = density_upper * weight_upper * tail / (2 * center**exponent)
                    if bound <= local_goal / 8:
                        remainder_error = bound
                        break
                power = product(power, normalized)
                coefficient = next_coefficient
            if remainder_error is None:
                raise CellGeometryTransitionError(
                    "resource_limit",
                    "Projective reciprocal-series budget exhausted.",
                    measured=maximum_terms,
                    limit=maximum_terms,
                )
            polynomial_weight = product(weight, scale(series, center ** (-exponent)))
            integral = _sqrt_polynomial_integral_piece(
                g,
                "triangle",
                local_goal / 8,
                0.0,
                maximum_terms,
                work,
                weight=polynomial_weight,
            )
            if integral is None:
                complete = False
                for child_g, child_weight, child_d, child_fraction in split(
                    g, weight, d, fraction
                ):
                    visited += 1
                    if visited > maximum_subcells:
                        raise CellGeometryTransitionError(
                            "resource_limit",
                            "Projective density subdivision budget exhausted.",
                            measured=visited,
                            limit=maximum_subcells,
                        )
                    controls = bernstein_coefficients(child_d, "simplex", 2)
                    child_lo, child_hi = min(controls), max(controls)
                    refined_pieces.append(
                        (
                            child_g,
                            child_weight,
                            child_d,
                            child_fraction,
                            child_lo,
                            child_hi,
                            (child_hi - child_lo) / (child_hi + child_lo),
                        )
                    )
            else:
                value, owner_error, _ = integral
                central += fraction * value
                error += fraction * (owner_error + remainder_error)
                refined_pieces.append((g, weight, d, fraction, lo, hi, rho))
        if not complete:
            pieces = refined_pieces
            continue
        if fraction_result:
            published, publication_error = central, Fraction(0)
        else:
            published = float(central)
            publication_error = abs(Fraction(published) - central)
        error += publication_error
        lower_magnitude = max(abs(central) - error, Fraction(0))
        tolerance = absolute + relative * lower_magnitude
        bound = (
            error if fraction_result else (0.0 if error == 0 else outward(error, np.inf))
        )
        if Fraction(bound) <= tolerance:
            return published, bound
        refinement += 1
    raise CellGeometryTransitionError(
        "resource_limit",
        "Projective signed-integral error budget exhausted.",
        measured=maximum_terms,
        limit=maximum_terms,
    )


def _projective_form(
    first: Sequence[Polynomial],
    second: Sequence[Polynomial],
    gram: Expression,
    denominator: Polynomial,
    exponent: int,
    budgets: _IntegrationBudgets,
) -> tuple[_FloatArray, _FloatArray]:
    """Integrate the actual polynomial functionals through shared rational moments."""
    weights = tuple(tuple(multiply(a, b) for b in second) for a in first)
    powers = sorted({index for row in weights for weight in row for index in weight})
    amplification = max(
        Fraction(1),
        max(
            sum((abs(value) for value in weight.values()), Fraction(0))
            for row in weights
            for weight in row
        ),
    )
    # A constant physical density is one shared algebraic factor, not independent
    # rounded roots in every moment. Preserve that correlation and refine the
    # owner's valid rational root interval by exact squaring when needed.
    constant_value = (
        gram.get((0, 0), Fraction(0))
        if not isinstance(gram, RationalPolynomial) and _degree(gram) == 0
        else None
    )
    constant_density = constant_value is not None
    root_lower, root_upper = (
        _fraction_sqrt_interval(constant_value)
        if constant_value is not None
        else (Fraction(1), Fraction(1))
    )
    moment_gram = constant(1, 2) if constant_density else gram
    maximum_refinements = int(budgets["maximum_binomial_terms"])
    for refinement in range(maximum_refinements):
        moment_budgets: _ProjectiveBudgets = {
            **budgets,
            "absolute_tolerance": Fraction(budgets["absolute_tolerance"])
            / (2 * amplification * root_upper * 2**refinement),
            "relative_tolerance": Fraction(budgets["relative_tolerance"])
            / (2 * amplification * 2**refinement),
        }
        moments = {
            index: _projective_integral(
                moment_gram,
                {index: Fraction(1)},
                denominator,
                exponent,
                moment_budgets,
                fraction_result=True,
            )
            for index in powers
        }
        values = np.empty((len(first), len(second)), dtype=np.float64)
        errors = np.empty_like(values)
        passed = True
        root_center, root_error = (
            (root_lower + root_upper) / 2,
            (root_upper - root_lower) / 2,
        )
        for i, row in enumerate(weights):
            for j, weight in enumerate(row):
                central = sum(
                    (
                        coefficient * moments[index][0]
                        for index, coefficient in weight.items()
                    ),
                    Fraction(0),
                )
                error = sum(
                    (
                        abs(coefficient) * moments[index][1]
                        for index, coefficient in weight.items()
                    ),
                    Fraction(0),
                )
                error = root_upper * error + root_error * abs(central)
                central *= root_center
                published = float(central)
                error += abs(Fraction(published) - central)
                bound = 0.0 if error == 0 else outward(error, np.inf)
                lower = max(abs(central) - error, Fraction(0))
                tolerance = (
                    Fraction(budgets["absolute_tolerance"])
                    + Fraction(budgets["relative_tolerance"]) * lower
                )
                passed &= Fraction(bound) <= tolerance
                values[i, j], errors[i, j] = published, bound
        if passed:
            return values, errors
        if constant_value is not None and root_lower != root_upper:
            midpoint = (root_lower + root_upper) / 2
            if midpoint**2 <= constant_value:
                root_lower = midpoint
            else:
                root_upper = midpoint
    raise CellGeometryTransitionError(
        "resource_limit",
        "Projective form publication exceeds its error budget.",
        measured=maximum_refinements,
        limit=maximum_refinements,
    )


def _prepare_dg(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    indices: _CellIndices,
    name: str,
    semantics: Literal["intensive", "conservative-density"],
    budgets: _IntegrationBudgets,
) -> FiniteElementFieldTransfer:
    si, ti = source._field_index(name), target._field_index(name)
    sc, tc = _mapped_dg_cells(source, si), _mapped_dg_cells(target, ti)
    geometries = (
        _mapped_geometry_cells(source.mesh, prepared.source_geometry),
        _mapped_geometry_cells(target.mesh, prepared.target_geometry),
    )
    grams = tuple(
        tuple(_embedded_squared_density(*cell) for cell in cells) for cells in geometries
    )
    ss, ts = source.dof_maps[si].global_dof_count, target.dof_maps[ti].global_dof_count
    form_count = (
        ss
        + ts
        + sum(len(basis) ** 2 for _, _, basis in tc)
        + len(prepared.pieces)
        * max(len(basis) for _, _, basis in sc)
        * max(len(basis) for _, _, basis in tc)
    )
    local_budgets: _IntegrationBudgets = {
        **budgets,
        "absolute_tolerance": budgets["absolute_tolerance"] / form_count,
        "relative_tolerance": budgets["relative_tolerance"] / form_count,
    }
    sm, tm, se, te = np.zeros(ss), np.zeros(ts), np.zeros(ss), np.zeros(ts)
    masses: list[_FloatArray] = []
    mixed: list[dict[int, _FloatArray]] = [dict() for _ in tc]
    errors = 0.0
    for cells, density, measures, uncertainties, is_target in (
        (sc, grams[0], sm, se, False),
        (tc, grams[1], tm, te, True),
    ):
        for cell, (element, route, basis) in enumerate(cells):
            if element.cell_kind != "triangle":
                raise ValueError(
                    "Sphere DG requires actual canonical scalar triangle fields."
                )
            for dof, term in zip(route, basis, strict=True):
                measures[dof], uncertainties[dof] = _integral(
                    density[cell], term, local_budgets
                )
            if is_target:
                mass = np.empty((len(basis), len(basis)), dtype=np.float64)
                for i, first in enumerate(basis):
                    for j, second in enumerate(basis):
                        mass[i, j], error = _integral(
                            density[cell], multiply(first, second), local_budgets
                        )
                        errors += error
                masses.append(mass)
    source_indices, target_indices = indices
    for piece in prepared.pieces:
        old = source_indices[int(piece.source_cell_global_id)]
        new = target_indices[int(piece.target_cell_global_id)]
        vertices = np.asarray(piece.exact_source_reference_vertices, dtype=object)
        old_gram, old_arguments = _restricted(grams[0][old], vertices)
        numerators, denominator = _projective_arguments(piece, old_arguments)
        target_degree = max(_degree(term) for term in tc[new][2])
        new_basis = tuple(
            _homogeneous(term, numerators, denominator, target_degree)
            for term in tc[new][2]
        )
        old_basis = tuple(compose(term, old_arguments) for term in sc[old][2])
        if semantics == "conservative-density":
            gram, denominator_power = old_gram, target_degree
        elif isinstance(grams[1][new], RationalPolynomial):
            gram, measure_weight, measure_denominator_power = (
                _projective_rational_density(
                    grams[1][new],
                    numerators,
                    denominator,
                    abs(_orientation(*piece.exact_source_reference_vertices))
                    * piece.projective_map.orientation_ratio,
                )
            )
            new_basis = tuple(multiply(term, measure_weight) for term in new_basis)
            denominator_power = target_degree + measure_denominator_power
        else:
            target_gram = grams[1][new]
            if isinstance(target_gram, RationalPolynomial):
                raise ValueError(
                    "Polynomial sphere Gram preparation received an unresolved rational source."
                )
            degree = 2 * ((_degree(target_gram) + 1) // 2)
            source_determinant = abs(_orientation(*piece.exact_source_reference_vertices))
            jacobian_numerator = (
                source_determinant * piece.projective_map.orientation_ratio
            )
            gram = scale(
                _homogeneous(target_gram, numerators, denominator, degree),
                jacobian_numerator**2,
            )
            denominator_power = target_degree + degree // 2 + 3
        form, form_errors = _projective_form(
            new_basis, old_basis, gram, denominator, denominator_power, local_budgets
        )
        errors += float(np.sum(form_errors))
        previous = mixed[new].get(old)
        mixed[new][old] = form if previous is None else previous + form
    return _finish_material_dg_projection(
        source,
        target,
        prepared,
        field_name=name,
        source_cells=sc,
        target_cells=tc,
        source_measures=sm,
        target_measures=tm,
        source_errors=se,
        target_errors=te,
        masses=masses,
        mixed=mixed,
        integration_error=errors,
        semantics=semantics,
        budgets=budgets,
    )


def prepare_sphere_chart_field_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    semantics: Literal["intensive", "conservative-density"] = "intensive",
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
) -> FiniteElementFieldTransfer:
    """Actual scalar material pullback/projection and its algebraic transpose.

    H1 intensive transport applies old nodal functionals through the exact inverse
    projective map. DG integrates actual old density for conserved inventory, or
    actual target density including the projective Jacobian for intensive fields.
    No affine corner approximation, source-nodal density or Piola claim is used.
    """
    if not isinstance(source, FiniteElementDiscretization) or not isinstance(
        target, FiniteElementDiscretization
    ):
        raise TypeError(
            "Sphere scalar field transport requires actual prepared FE endpoints."
        )
    if semantics not in ("intensive", "conservative-density"):
        raise ValueError("Unknown sphere chart field semantics.")
    indices = _validate_sphere(source, target, prepared, source_geometry, target_geometry)
    si, ti = source._field_index(field_name), target._field_index(field_name)
    source_space = source.field_spaces[si].vector_space
    target_space = target.field_spaces[ti].vector_space
    if not isinstance(source_space, ArraySpace) or not isinstance(
        target_space, ArraySpace
    ):
        raise TypeError("Sphere chart transfer requires array coefficient spaces.")
    if source_space.shape[1:] != target_space.shape[1:]:
        raise ValueError("Sphere scalar field payload layouts differ.")
    conformities = {
        element.conformity for element in source.elements[si] + target.elements[ti]
    }
    if conformities == {"H1"} and semantics == "intensive":
        return _prepare_h1(source, target, prepared, indices, field_name)
    if conformities != {"L2"}:
        raise ValueError(
            "Sphere all-field transport is unclosed for compatible/Piola or non-DG density fields."
        )
    budgets = _IntegrationBudgets(
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        maximum_work=maximum_work,
        maximum_subcells=maximum_subcells,
        maximum_binomial_terms=maximum_binomial_terms,
    )
    return _prepare_dg(source, target, prepared, indices, field_name, semantics, budgets)


def prepare_sphere_chart_finite_volume_contents(
    source: _ChartEndpoint,
    target: _ChartEndpoint,
    prepared: PreparedSphereChartDeformation,
    /,
    *,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
) -> PreparedSurfaceChartFiniteVolumeContents:
    """True old physical contents of exact projective overlaps for the FV owner."""
    source_indices, target_indices = _validate_sphere(
        source, target, prepared, source_geometry, target_geometry
    )
    geometries = _mapped_geometry_cells(source.mesh, source_geometry)
    grams = tuple(_embedded_squared_density(*cell) for cell in geometries)
    local_budgets = _IntegrationBudgets(
        absolute_tolerance=absolute_tolerance / len(prepared.pieces),
        relative_tolerance=relative_tolerance / len(prepared.pieces),
        maximum_work=maximum_work,
        maximum_subcells=maximum_subcells,
        maximum_binomial_terms=maximum_binomial_terms,
    )
    rows, targets, values, errors = [], [], [], []
    for piece in prepared.pieces:
        old = source_indices[int(piece.source_cell_global_id)]
        new = target_indices[int(piece.target_cell_global_id)]
        gram, _ = _restricted(
            grams[old], np.asarray(piece.exact_source_reference_vertices, dtype=object)
        )
        value, error = _integral(gram, constant(1, 2), local_budgets)
        rows.append(old)
        targets.append(new)
        values.append(value)
        errors.append(error)
    source_order, target_order = (
        _physical_rows(source, prepared.source_mesh),
        _physical_rows(target, prepared.target_mesh),
    )
    volumes, volume_errors = _endpoint_cell_measures(
        target, prepared, target_order, "target"
    )
    old_volumes, old_errors = _endpoint_cell_measures(
        source, prepared, source_order, "source"
    )
    sums, uncertainties = np.zeros_like(old_volumes), np.zeros_like(old_volumes)
    np.add.at(sums, rows, values)
    np.add.at(uncertainties, rows, errors)
    roundoff = np.finfo(np.float64).eps * (len(values) + 2)
    bounds = np.nextafter(
        (
            np.abs(sums - old_volumes)
            + uncertainties
            + old_errors
            + roundoff * (sums + old_volumes)
        )
        / (1 - roundoff),
        np.inf,
    )
    return PreparedSurfaceChartFiniteVolumeContents(
        np.asarray(rows, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        volumes,
        volume_errors,
        old_volumes,
        old_errors,
        bounds,
        prepared,
    )


def _sphere_piece_charts(
    piece: PreparedSphereChartPiece,
) -> tuple[tuple[Expression, ...], tuple[Expression, ...]]:
    vertices = piece.exact_source_reference_vertices
    variables = algebra.axes(2)
    source: tuple[Expression, ...] = tuple(
        algebra.add(
            algebra.constant(vertices[0][axis], 2),
            algebra.sum_polynomials(
                tuple(
                    algebra.scale(
                        variable, vertices[column + 1][axis] - vertices[0][axis]
                    )
                    for column, variable in enumerate(variables)
                )
            ),
        )
        for axis in range(2)
    )
    projective = piece.projective_map.reference_expressions()
    target = tuple(algebra.expression_compose(value, source) for value in projective)
    return source, target


def _sphere_projective_condition(
    piece: PreparedSphereChartPiece,
    source: tuple[Expression, ...],
    work: _PreparationWork,
) -> None:
    expressions = piece.projective_map.reference_expressions()
    bounds = tuple(
        tuple(
            algebra.expression_bounds(
                algebra.expression_compose(
                    algebra.expression_derivative(value, axis), source
                ),
                "simplex",
                2,
            )
            for axis in range(2)
        )
        for value in expressions
    )
    norm_squared = sum(
        (
            max(abs(Fraction(float(low))), abs(Fraction(float(high)))) ** 2
            for row in bounds
            for low, high in row
        ),
        Fraction(0),
    )
    minimum = piece.projective_bounds.jacobian_lower
    if minimum <= 0:
        raise ValueError(
            "Sphere material transport lacks a positive whole-piece Jacobian."
        )
    condition = algebra.outward(norm_squared / minimum, np.inf)
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "Sphere material reference transport exceeds its whole-piece condition bound."
        )
    work.condition = max(work.condition, condition)


def prepare_sphere_chart_compatible_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    coordinate_budget: algebra.CoordinateEnclosureBudget | None = None,
) -> PreparedSurfaceChartCompatibleTransfer:
    """Transport actual sphere material forms through their genuine projective pieces."""
    indices = _validate_sphere(source, target, prepared, source_geometry, target_geometry)

    def prepare_pieces(work: _PreparationWork) -> tuple[_MaterialCompatiblePiece, ...]:
        pieces: list[_MaterialCompatiblePiece] = []
        for piece in prepared.pieces:
            old = indices[0][piece.source_cell_global_id]
            new = indices[1][piece.target_cell_global_id]
            with work.enclosure.temporary_scope():
                arguments, images = _sphere_piece_charts(piece)
                _sphere_projective_condition(piece, arguments, work)
            pieces.append((old, new, arguments, images))
        return tuple(pieces)

    return _prepare_material_chart_compatible_transfer(
        source,
        target,
        prepared,
        field_name=field_name,
        prepare_pieces=prepare_pieces,
        maximum_work=maximum_work,
        maximum_storage_bytes=maximum_storage_bytes,
        coordinate_budget=coordinate_budget,
    )


__all__ = [
    "prepare_sphere_chart_field_transfer",
    "prepare_sphere_chart_finite_volume_contents",
    "prepare_sphere_chart_compatible_transfer",
]
