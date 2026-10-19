#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"Transport scalar fields through projective sphere material correspondence.\n\nPhysical reference maps are rational, not affine maps fitted to corner images.\nEvery mixed form uses the true projective quotient/Jacobian on an old\nreference overlap triangle. Positive-denominator series carry an explicit\nremainder and reuse the ordered signed sqrt-Gram integration owner."

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction
from typing import assert_never, Literal, overload, TypedDict

import numpy as np
from numpy.typing import NDArray

import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.discretization._sphere_chart_records as imported_sphere_chart_deformation
from phydrax.discretization._sphere_chart_records import (
    PreparedSphereChartDeformation,
    PreparedSphereChartPiece,
)
from phydrax.geometry._rational import fraction_endpoint

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._resource_ticket import HostBatchTicket
from ..._strict import StrictModule
from ...geometry._sphere_material_atlas import SphereProjectiveReferenceMap
from ...linalg import ArraySpace
from ...linalg._hermitian_spectral import fraction_sqrt_interval
from ...sparse import RowRelation, SparseLinearMap
from ...typing import checked, parse
from .. import _coordinate_enclosure as algebra
from .._cell_geometry import CellGeometrySpec
from .._cell_geometry_transition_types import CellGeometryTransitionError
from .._chart_geometry import contains, orientation, projective_point
from .._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from .._coordinate_enclosure import (
    add,
    affine_arguments,
    bernstein_coefficients,
    compose,
    constant,
    Expression,
    multiply,
    Polynomial,
    rational_expression,
    RationalPolynomial,
    scale,
    source_basis,
    sum_polynomials,
)
from .._mapped_geometry_measure import (
    embedded_measure_submaps,
    embedded_squared_density,
    mapped_geometry_cells,
    sqrt_polynomial_integral_piece,
)
from ._generic import FiniteElementDiscretization
from ._mapped_form_transfer import PreparationWork
from ._surface_chart_compatible import (
    MaterialCompatiblePiece,
    prepare_material_chart_compatible_transfer,
    PreparedSurfaceChartCompatibleTransfer,
)
from ._surface_chart_transfer import (
    bound_surface_endpoint,
    ChartEndpoint,
    ChartFieldSemantics,
    deformation_binding,
    endpoint_cell_measures,
    field_cells,
    finish_material_dg_projection,
    integral,
    IntegrationBudgets,
    physical_rows,
    PreparedSurfaceChartFiniteVolumeContents,
    restricted,
)
from ._topology_transfer import (
    field_transfer_query_upper,
    FiniteElementFieldTransfer,
    FiniteElementTopologyTransfer,
    FiniteElementTransferEvidence,
    mapped_dg_cells,
    owned_rows,
    tabulated,
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
    source: ChartEndpoint,
    target: ChartEndpoint,
    prepared: imported_sphere_chart_deformation.PreparedSphereChartDeformation,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CellIndices:

    bound_surface_endpoint(source, source_geometry, "source", prepared, budget=budget)
    bound_surface_endpoint(target, target_geometry, "target", prepared, budget=budget)
    prepared.require_bound(
        source.mesh, source_geometry, target.mesh, target_geometry, budget=budget
    )
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
                "sphere_chart.validate_sphere.pieces must satisfy the material reference contract; got Sphere correspondence does not bind the current scientific cells."
            )
        indices.append({int(identity): row for row, identity in enumerate(ids)})
    return indices[0], indices[1]


def _prepare_h1(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    indices: _CellIndices,
    name: str,
    *,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    si, ti = source.field_index(name), target.field_index(name)
    source_cells, target_cells = field_cells(source, si), field_cells(target, ti)
    if any(
        element.conformity != "H1" or source_basis(element, budget=budget) is None
        for element, _ in source_cells + target_cells
    ):
        raise ValueError(
            "sphere_chart.prepare_h1.pieces must satisfy the material reference contract; got Sphere H1 transport requires ordered scalar nodal sources."
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
                "sphere_chart.prepare_h1.pieces must satisfy the material reference contract; got Sphere H1 functionals require two-dimensional reference nodes."
            )
        for node in range(nodes.shape[0]):
            point = Fraction(float(nodes[node, 0])), Fraction(float(nodes[node, 1]))
            if not contains(
                piece.exact_target_reference_vertices,
                point,
                degenerate_message="sphere_chart.reference_triangle must have nonzero area; got degenerate triangle.",
            ):
                continue
            reference = projective_point(
                inverse.exact_coefficients,
                point,
                invalid_message="sphere_chart.reference_functional must remain inside the positive reference cone; got nonpositive denominator or negative barycentric coordinate.",
            )
            if not contains(
                piece.exact_source_reference_vertices,
                reference,
                degenerate_message="sphere_chart.reference_triangle must have nonzero area; got degenerate triangle.",
            ):
                raise ValueError(
                    "sphere_chart.prepare_h1.binding must satisfy the declared geometry and cell identities; got A projective target functional has no old overlap binding."
                )
            basis, _ = tabulated(
                source_element,
                np.asarray([[float(value) for value in reference]], dtype=np.float64),
            )
            if not np.all(np.isfinite(basis)):
                raise ValueError(
                    "sphere_chart.prepare_h1.pieces must satisfy the material reference contract; got Sphere source nodal functionals are not finite."
                )
            candidates.append(int(target_route[node]))
            columns.append(source_route)
            values.append(basis[0])
    width = max(element.local_dof_count for element, _ in source_cells)
    padded_columns = np.zeros((len(candidates), width), dtype=np.int64)
    padded_values = np.zeros(padded_columns.shape, dtype=np.float64)
    for row, (route, value) in enumerate(zip(columns, values, strict=True)):
        padded_columns[row, : route.size], padded_values[row, : route.size] = route, value
    ss, ts = source.dof_maps[si].global_dof_count, target.dof_maps[ti].global_dof_count
    rows, coefficients, continuity = owned_rows(
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
        raise ValueError(
            "sphere_chart.prepare_h1.pieces must satisfy the material reference contract; got Sphere rational H1 nodal/trace certificate failed."
        )
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
        deformation_binding(prepared),
        evidence,
    )


def _degree(polynomial: Polynomial) -> int:
    return max((sum(index) for index in polynomial), default=0)


def _power(
    polynomial: Polynomial, exponent: int, *, budget: CoordinateEnclosureBudget
) -> Polynomial:
    result = constant(1, 2, budget=budget)
    for _ in range(exponent):
        result = multiply(result, polynomial, budget=budget)
    return result


def _homogeneous(
    polynomial: Polynomial,
    numerators: Sequence[Polynomial],
    denominator: Polynomial,
    degree: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Polynomial:
    "numerator after substituting N/D, with one positive D power."
    powers = [
        tuple(_power(value, exponent, budget=budget) for exponent in range(degree + 1))
        for value in (*numerators, denominator)
    ]
    return sum_polynomials(
        tuple(
            scale(
                multiply(
                    multiply(powers[0][index[0]], powers[1][index[1]], budget=budget),
                    powers[2][degree - sum(index)],
                    budget=budget,
                ),
                coefficient,
                budget=budget,
            )
            for index, coefficient in polynomial.items()
        ),
        budget=budget,
    )


@checked
def projective_rational_density(
    gram: Expression,
    numerators: Sequence[Polynomial],
    denominator: Polynomial,
    jacobian_numerator: Fraction,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, Polynomial, int]:
    "Keep positive even denominator/Jacobian factors outside sqrt(Gram)."
    match gram:
        case RationalPolynomial():
            pass
        case _:
            raise ValueError(
                "sphere_chart.projective_rational_density.measures must satisfy the positive physical measure contract; got Rational sphere density substitution requires its quotient."
            )
    controls = bernstein_coefficients(denominator, "simplex", 2, budget=budget)
    if min(controls) <= 0:
        raise ValueError(
            "sphere_chart.projective_rational_density.pieces must satisfy the material reference contract; got Rational sphere reference action has no positive denominator proof."
        )
    numerator_degree, denominator_degree = (
        _degree(gram.numerator),
        _degree(gram.denominator),
    )
    numerator = _homogeneous(
        gram.numerator, numerators, denominator, numerator_degree, budget=budget
    )
    divisor = _homogeneous(
        gram.denominator, numerators, denominator, denominator_degree, budget=budget
    )
    degree_difference = denominator_degree - numerator_degree
    parity = degree_difference % 2
    if parity:
        numerator = multiply(numerator, denominator, budget=budget)
    outside_power = (degree_difference - parity) // 2 - 3
    weight = scale(
        _power(denominator, max(outside_power, 0), budget=budget),
        abs(jacobian_numerator),
        budget=budget,
    )
    return (
        rational_expression(numerator, divisor, budget=budget),
        weight,
        max(-outside_power, 0),
    )


def _projective_arguments(
    piece: PreparedSphereChartPiece,
    source_arguments: Sequence[Polynomial],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Polynomial, ...], Polynomial]:
    homogeneous = tuple(
        sum_polynomials(
            (
                constant(row[0], 2, budget=budget),
                scale(source_arguments[0], row[1], budget=budget),
                scale(source_arguments[1], row[2], budget=budget),
            ),
            budget=budget,
        )
        for row in piece.projective_map.exact_coefficients
    )
    return homogeneous[1:], sum_polynomials(homogeneous, budget=budget)


def _rational_projective_integral(
    gram: RationalPolynomial,
    numerator: Polynomial,
    denominator: Polynomial,
    exponent: int,
    absolute: Fraction,
    relative: Fraction,
    maximum_terms: int,
    maximum_subcells: int,
    budgets: _ProjectiveBudgets,
    fraction_result: bool,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction | float, Fraction | float]:
    "Enclose a projective moment through the rational density integral owner."
    divisor_lower = algebra.require_rational_reference_scalar(
        min(bernstein_coefficients(gram.denominator, "simplex", 2, budget=budget))
    )
    reference_lower = algebra.require_rational_reference_scalar(
        min(bernstein_coefficients(denominator, "simplex", 2, budget=budget))
    )
    if min(divisor_lower, reference_lower) <= 0:
        raise ValueError(
            "sphere_chart.rational_projective_integral.measures must satisfy the positive physical measure contract; got Rational projective density has no positive denominator proof."
        )
    density_upper = (
        fraction_sqrt_interval(
            algebra.require_rational_reference_scalar(
                max(bernstein_coefficients(gram.numerator, "simplex", 2, budget=budget))
            )
            / divisor_lower
        )[1]
        / reference_lower**exponent
    )
    magnitude = max(
        abs(algebra.require_rational_reference_scalar(value))
        for value in bernstein_coefficients(numerator, "simplex", 2, budget=budget)
    )
    goal = absolute + relative * density_upper * magnitude / 2
    work = [0, int(budgets["maximum_work"])]
    for refinement in range(maximum_terms):
        try:
            with budget.measurement():
                prepared = (
                    mapped_measure_owner.certified_sqrt_prepare_expression_integral(
                        gram,
                        "triangle",
                        goal / (max(magnitude, Fraction(1)) * 2**refinement),
                        work,
                        maximum_subcells,
                        maximum_terms,
                        denominator=denominator,
                        denominator_exponent=exponent,
                        budget=budget,
                    )
                )
                value, error = prepared.integral(numerator, budget=budget)
        except CoordinateEnclosureResourceError as exhausted:
            raise CellGeometryTransitionError(
                "resource_limit",
                "sphere_chart.rational_projective_integral.budget must satisfy the admitted resource limits; got Rational projective source coefficient budget exhausted.",
                measured=float(exhausted.requested),
                limit=float(exhausted.limit),
            ) from exhausted
        allowed = absolute + relative * max(
            abs(Fraction(value)) - Fraction(error), Fraction(0)
        )
        if Fraction(error) <= allowed:
            return (
                (Fraction(value), Fraction(error)) if fraction_result else (value, error)
            )
    raise CellGeometryTransitionError(
        "resource_limit",
        "sphere_chart.rational_projective_integral.budget must satisfy the admitted resource limits; got Rational projective publication exceeds its requested error budget.",
        measured=error,
        limit=float(allowed),
    )


class _ProjectiveReciprocalSeries(StrictModule):
    polynomial_weight: Polynomial
    remainder_error: Fraction


def _projective_reciprocal_series(
    g: Polynomial,
    weight: Polynomial,
    d: Polynomial,
    lo: Fraction,
    hi: Fraction,
    rho: Fraction,
    exponent: int,
    maximum_terms: int,
    local_goal: Fraction,
    product: Callable[[Polynomial, Polynomial], Polynomial],
    budget: CoordinateEnclosureBudget,
) -> _ProjectiveReciprocalSeries:
    "Bound the reciprocal-series remainder before integrating its polynomial."
    center = (lo + hi) / 2
    normalized = add(
        scale(d, 1 / center, budget=budget),
        constant(-1, 2, budget=budget),
        budget=budget,
    )
    series: Polynomial = {}
    power, coefficient = constant(1, 2, budget=budget), Fraction(1)
    density_upper = fraction_sqrt_interval(
        algebra.require_rational_reference_scalar(
            max(bernstein_coefficients(g, "simplex", 2, budget=budget))
        )
    )[1]
    weight_upper = max(
        abs(algebra.require_rational_reference_scalar(value))
        for value in bernstein_coefficients(weight, "simplex", 2, budget=budget)
    )
    remainder_error = None
    for order in range(maximum_terms):
        series = add(series, scale(power, coefficient, budget=budget), budget=budget)
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
            "sphere_chart.projective_reciprocal_series.budget must satisfy the admitted resource limits; got Projective reciprocal-series budget exhausted.",
            measured=maximum_terms,
            limit=maximum_terms,
        )
    polynomial_weight = product(
        weight, scale(series, center ** (-exponent), budget=budget)
    )
    return _ProjectiveReciprocalSeries(
        polynomial_weight=polynomial_weight, remainder_error=remainder_error
    )


def _polynomial_projective_integral(
    gram: Polynomial,
    numerator: Polynomial,
    denominator: Polynomial,
    exponent: int,
    absolute: Fraction,
    relative: Fraction,
    maximum_terms: int,
    maximum_subcells: int,
    budgets: _ProjectiveBudgets,
    fraction_result: bool,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction | float, Fraction | float]:
    "Refine projective polynomial moments until their error contract holds."
    lower = algebra.require_rational_reference_scalar(
        min(bernstein_coefficients(denominator, "simplex", 2, budget=budget))
    )
    if lower <= 0:
        raise ValueError(
            "sphere_chart.polynomial_projective_integral.measures must satisfy the positive physical measure contract; got Projective density integration requires a positive denominator enclosure."
        )
    gram_upper = algebra.require_rational_reference_scalar(
        max(bernstein_coefficients(gram, "simplex", 2, budget=budget))
    )
    magnitude = max(
        abs(algebra.require_rational_reference_scalar(value))
        for value in bernstein_coefficients(numerator, "simplex", 2, budget=budget)
    )
    scale_bound = (
        fraction_sqrt_interval(gram_upper)[1] * magnitude / (2 * lower**exponent)
    )
    goal = absolute + relative * scale_bound
    pieces, pending, visited = [], [(gram, numerator, denominator, Fraction(1))], 0
    work = [0, int(budgets["maximum_work"])]

    def product(first: Polynomial, second: Polynomial) -> Polynomial:
        work[0] += len(first) * len(second)
        if work[0] > work[1]:
            raise CellGeometryTransitionError(
                "resource_limit",
                "sphere_chart.product.budget must satisfy the admitted resource limits; got Projective polynomial work budget exhausted.",
                measured=work[0],
                limit=work[1],
            )
        return multiply(first, second, budget=budget)

    def split(
        g: Polynomial,
        weight: Polynomial,
        d: Polynomial,
        fraction: Fraction,
    ) -> Iterator[tuple[Polynomial, Polynomial, Polynomial, Fraction]]:
        for matrix, offset in embedded_measure_submaps("triangle"):
            arguments = affine_arguments(offset, matrix, budget=budget)
            yield (
                compose(g, arguments, budget=budget),
                compose(weight, arguments, budget=budget),
                compose(d, arguments, budget=budget),
                fraction / 4,
            )

    while pending:
        g, weight, d, fraction = pending.pop()
        visited += 1
        if visited > maximum_subcells:
            raise CellGeometryTransitionError(
                "resource_limit",
                "sphere_chart.polynomial_projective_integral.budget must satisfy the admitted resource limits; got Projective denominator subdivision budget exhausted.",
                measured=visited,
                limit=maximum_subcells,
            )
        controls = bernstein_coefficients(d, "simplex", 2, budget=budget)
        lo = algebra.require_rational_reference_scalar(min(controls))
        hi = algebra.require_rational_reference_scalar(max(controls))
        if lo <= 0:
            raise ValueError(
                "sphere_chart.polynomial_projective_integral.pieces must satisfy the material reference contract; got Projective subdivision lost its positive denominator proof."
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
            _phase = _projective_reciprocal_series(
                g=g,
                weight=weight,
                d=d,
                lo=lo,
                hi=hi,
                rho=rho,
                exponent=exponent,
                maximum_terms=maximum_terms,
                local_goal=local_goal,
                product=product,
                budget=budget,
            )
            polynomial_weight = _phase.polynomial_weight
            remainder_error = _phase.remainder_error
            integral = sqrt_polynomial_integral_piece(
                g,
                "triangle",
                local_goal / 8,
                0.0,
                maximum_terms,
                work,
                weight=polynomial_weight,
                budget=budget,
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
                            "sphere_chart.polynomial_projective_integral.budget must satisfy the admitted resource limits; got Projective density subdivision budget exhausted.",
                            measured=visited,
                            limit=maximum_subcells,
                        )
                    controls = bernstein_coefficients(
                        child_d, "simplex", 2, budget=budget
                    )
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
            error
            if fraction_result
            else (
                0.0 if error == 0 else fraction_endpoint(error, upper=True, saturate=True)
            )
        )
        if Fraction(bound) <= tolerance:
            return published, bound
        refinement += 1
    raise CellGeometryTransitionError(
        "resource_limit",
        "sphere_chart.polynomial_projective_integral.budget must satisfy the admitted resource limits; got Projective signed-integral error budget exhausted.",
        measured=maximum_terms,
        limit=maximum_terms,
    )


@overload
def _projective_integral(
    gram: Expression,
    numerator: Polynomial,
    denominator: Polynomial,
    exponent: int,
    budgets: _ProjectiveBudgets,
    *,
    fraction_result: Literal[True],
    budget: CoordinateEnclosureBudget,
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
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]: ...
def _projective_integral(
    gram: Expression,
    numerator: Polynomial,
    denominator: Polynomial,
    exponent: int,
    budgets: _ProjectiveBudgets,
    *,
    fraction_result: bool = False,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction | float, Fraction | float]:
    "Enclose sqrt(Gram)*numerator/D**exponent using the ordered root owner.\n\n The reciprocal series is polynomial data with a uniform remainder.\n Fractions may be retained across a polynomial moment functional, so a DG\n matrix is published once, rather than rounding every intermediate moment.\n Density and denominator subdivisions are source-reference maps.\n Resource exhaustion never switches to a sampled or affine approximation."
    try:
        with budget.measurement():
            if not numerator:
                return (Fraction(0), Fraction(0)) if fraction_result else (0.0, 0.0)
            if exponent == 0:
                denominator = constant(1, 2, budget=budget)
            absolute, relative = (
                Fraction(budgets["absolute_tolerance"]),
                Fraction(budgets["relative_tolerance"]),
            )
            if absolute < 0 or relative < 0 or absolute == relative == 0:
                raise ValueError(
                    "sphere_chart.projective_integral.budget must satisfy the admitted resource limits; got Projective integration requires positive error budgets."
                )
            maximum_terms, maximum_subcells = (
                int(budgets["maximum_binomial_terms"]),
                int(budgets["maximum_subcells"]),
            )
            if maximum_terms <= 0 or maximum_subcells <= 0:
                raise ValueError(
                    "sphere_chart.projective_integral.budget must satisfy the admitted resource limits; got Projective integration requires positive work limits."
                )
            match gram:
                case RationalPolynomial():
                    return _rational_projective_integral(
                        gram=gram,
                        numerator=numerator,
                        denominator=denominator,
                        exponent=exponent,
                        absolute=absolute,
                        relative=relative,
                        maximum_terms=maximum_terms,
                        maximum_subcells=maximum_subcells,
                        budgets=budgets,
                        fraction_result=fraction_result,
                        budget=budget,
                    )
                case dict():
                    return _polynomial_projective_integral(
                        gram=gram,
                        numerator=numerator,
                        denominator=denominator,
                        exponent=exponent,
                        absolute=absolute,
                        relative=relative,
                        maximum_terms=maximum_terms,
                        maximum_subcells=maximum_subcells,
                        budgets=budgets,
                        fraction_result=fraction_result,
                        budget=budget,
                    )
                case other:
                    assert_never(other)
    except CoordinateEnclosureResourceError as error:
        if isinstance(gram, RationalPolynomial):
            raise CellGeometryTransitionError(
                "resource_limit",
                "sphere_chart.projective_integral.budget must satisfy the admitted resource limits; got Rational projective source coefficient budget exhausted.",
                measured=float(error.requested),
                limit=float(error.limit),
            ) from error
        raise


@checked
def projective_form(
    first: Sequence[Polynomial],
    second: Sequence[Polynomial],
    gram: Expression,
    denominator: Polynomial,
    exponent: int,
    budgets: IntegrationBudgets,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_FloatArray, _FloatArray]:
    "Integrate the polynomial functionals through shared rational moments."
    weights = tuple(tuple(multiply(a, b, budget=budget) for b in second) for a in first)
    powers = sorted({index for row in weights for weight in row for index in weight})
    amplification = max(
        Fraction(1),
        max(
            sum(
                (
                    abs(algebra.require_rational_reference_scalar(value))
                    for value in weight.values()
                ),
                Fraction(0),
            )
            for row in weights
            for weight in row
        ),
    )
    # A constant physical density is one shared algebraic factor, not independent
    # rounded roots in every moment. Preserve that correlation and refine the
    # owner's valid rational root interval by squaring when needed.
    constant_value = (
        algebra.require_rational_reference_scalar(gram.get((0, 0), Fraction(0)))
        if not isinstance(gram, RationalPolynomial) and _degree(gram) == 0
        else None
    )
    constant_density = constant_value is not None
    root_lower, root_upper = (
        fraction_sqrt_interval(constant_value)
        if constant_value is not None
        else (Fraction(1), Fraction(1))
    )
    moment_gram = constant(1, 2, budget=budget) if constant_density else gram
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
                budget=budget,
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
                        algebra.require_rational_reference_scalar(coefficient)
                        * moments[index][0]
                        for index, coefficient in weight.items()
                    ),
                    Fraction(0),
                )
                error = sum(
                    (
                        abs(algebra.require_rational_reference_scalar(coefficient))
                        * moments[index][1]
                        for index, coefficient in weight.items()
                    ),
                    Fraction(0),
                )
                error = root_upper * error + root_error * abs(central)
                central *= root_center
                published = float(central)
                error += abs(Fraction(published) - central)
                bound = (
                    0.0
                    if error == 0
                    else fraction_endpoint(error, upper=True, saturate=True)
                )
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
        "sphere_chart.projective_form.budget must satisfy the admitted resource limits; got Projective form publication exceeds its error budget.",
        measured=maximum_refinements,
        limit=maximum_refinements,
    )


def _prepare_dg(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    indices: _CellIndices,
    name: str,
    semantics: ChartFieldSemantics,
    budgets: IntegrationBudgets,
    *,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    si, ti = source.field_index(name), target.field_index(name)
    sc, tc = (
        mapped_dg_cells(source, si, budget=budget),
        mapped_dg_cells(target, ti, budget=budget),
    )
    geometries = (
        mapped_geometry_cells(source.mesh, prepared.source_geometry, budget=budget),
        mapped_geometry_cells(target.mesh, prepared.target_geometry, budget=budget),
    )
    grams = tuple(
        tuple(embedded_squared_density(*cell, budget=budget) for cell in cells)
        for cells in geometries
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
    local_budgets: IntegrationBudgets = {
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
                    "sphere_chart.prepare_dg.pieces must satisfy the material reference contract; got Sphere DG requires ordered scalar triangle fields."
                )
            for dof, term in zip(route, basis, strict=True):
                measures[dof], uncertainties[dof] = integral(
                    density[cell], term, local_budgets, budget=budget
                )
            if is_target:
                mass = np.empty((len(basis), len(basis)), dtype=np.float64)
                for i, first in enumerate(basis):
                    for j, second in enumerate(basis):
                        mass[i, j], error = integral(
                            density[cell],
                            multiply(first, second, budget=budget),
                            local_budgets,
                            budget=budget,
                        )
                        errors += error
                masses.append(mass)
    source_indices, target_indices = indices
    for piece in prepared.pieces:
        old = source_indices[int(piece.source_cell_global_id)]
        new = target_indices[int(piece.target_cell_global_id)]
        vertices = np.asarray(piece.exact_source_reference_vertices, dtype=object)
        old_gram, old_arguments = restricted(grams[0][old], vertices, budget=budget)
        numerators, denominator = _projective_arguments(
            piece, old_arguments, budget=budget
        )
        target_degree = max(_degree(term) for term in tc[new][2])
        new_basis = tuple(
            _homogeneous(term, numerators, denominator, target_degree, budget=budget)
            for term in tc[new][2]
        )
        old_basis = tuple(
            compose(term, old_arguments, budget=budget) for term in sc[old][2]
        )
        match semantics:
            case "conservative-density":
                gram, denominator_power = old_gram, target_degree
            case "intensive":
                match grams[1][new]:
                    case RationalPolynomial() as target_gram:
                        gram, measure_weight, measure_denominator_power = (
                            projective_rational_density(
                                target_gram,
                                numerators,
                                denominator,
                                abs(orientation(*piece.exact_source_reference_vertices))
                                * piece.projective_map.orientation_ratio,
                                budget=budget,
                            )
                        )
                        new_basis = tuple(
                            multiply(term, measure_weight, budget=budget)
                            for term in new_basis
                        )
                        denominator_power = target_degree + measure_denominator_power
                    case dict() as target_gram:
                        degree = 2 * ((_degree(target_gram) + 1) // 2)
                        source_determinant = abs(
                            orientation(*piece.exact_source_reference_vertices)
                        )
                        jacobian_numerator = (
                            source_determinant * piece.projective_map.orientation_ratio
                        )
                        gram = scale(
                            _homogeneous(
                                target_gram,
                                numerators,
                                denominator,
                                degree,
                                budget=budget,
                            ),
                            jacobian_numerator**2,
                            budget=budget,
                        )
                        denominator_power = target_degree + degree // 2 + 3
                    case invalid_gram:
                        assert_never(invalid_gram)
            case _:
                assert_never(semantics)
        form, form_errors = projective_form(
            new_basis,
            old_basis,
            gram,
            denominator,
            denominator_power,
            local_budgets,
            budget=budget,
        )
        errors += float(np.sum(form_errors))
        previous = mixed[new].get(old)
        mixed[new][old] = form if previous is None else previous + form
    return finish_material_dg_projection(
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


def _prepare_sphere_chart_field_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    semantics: ChartFieldSemantics = "intensive",
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget,
) -> FiniteElementFieldTransfer:
    "scalar material pullback/projection and its algebraic transpose.\n\n H1 intensive transport applies old nodal functionals through the inverse\n projective map. DG integrates old density for conserved inventory, or\n target density including the projective Jacobian for intensive fields.\n No affine corner approximation, source-nodal density or Piola claim is used."
    semantics = parse(semantics, ChartFieldSemantics, "chart.semantics")
    indices = _validate_sphere(
        source, target, prepared, source_geometry, target_geometry, budget=budget
    )
    si, ti = source.field_index(field_name), target.field_index(field_name)
    source_space = source.field_spaces[si].vector_space
    target_space = target.field_spaces[ti].vector_space
    match (source_space, target_space):
        case (ArraySpace() as source_space, ArraySpace() as target_space):
            pass
        case _:
            raise TypeError(
                "sphere_chart.prepare_sphere_chart_field_transfer.pieces must satisfy the material reference contract; got Sphere chart transfer requires array coefficient spaces."
            )
    if source_space.shape[1:] != target_space.shape[1:]:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_field_transfer.pieces must satisfy the material reference contract; got Sphere scalar field payload layouts differ."
        )
    conformities = {
        element.conformity for element in source.elements[si] + target.elements[ti]
    }
    match (tuple(sorted(conformities)), semantics):
        case (("H1",), "intensive"):
            return _prepare_h1(
                source, target, prepared, indices, field_name, budget=budget
            )
        case (("L2",), _):
            pass
        case _:
            raise ValueError(
                "sphere_chart.prepare_sphere_chart_field_transfer.measures must satisfy the positive physical measure contract; got Sphere all-field transport is unclosed for compatible/Piola or non-DG density fields."
            )
    budgets = IntegrationBudgets(
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        maximum_work=maximum_work,
        maximum_subcells=maximum_subcells,
        maximum_binomial_terms=maximum_binomial_terms,
    )
    return _prepare_dg(
        source, target, prepared, indices, field_name, semantics, budgets, budget=budget
    )


@checked
def sphere_chart_finite_volume_contents(
    source: ChartEndpoint,
    target: ChartEndpoint,
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
    budget: CoordinateEnclosureBudget,
) -> PreparedSurfaceChartFiniteVolumeContents:
    "True old physical contents of projective overlaps for the FV owner."
    source_indices, target_indices = _validate_sphere(
        source, target, prepared, source_geometry, target_geometry, budget=budget
    )
    geometries = mapped_geometry_cells(source.mesh, source_geometry, budget=budget)
    grams = tuple(embedded_squared_density(*cell, budget=budget) for cell in geometries)
    local_budgets = IntegrationBudgets(
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
        gram, _ = restricted(
            grams[old],
            np.asarray(piece.exact_source_reference_vertices, dtype=object),
            budget=budget,
        )
        value, error = integral(
            gram, constant(1, 2, budget=budget), local_budgets, budget=budget
        )
        rows.append(old)
        targets.append(new)
        values.append(value)
        errors.append(error)
    source_order, target_order = (
        physical_rows(source, prepared.source_mesh),
        physical_rows(target, prepared.target_mesh),
    )
    volumes, volume_errors = endpoint_cell_measures(
        target, prepared, target_order, "target"
    )
    old_volumes, old_errors = endpoint_cell_measures(
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
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Expression, ...], tuple[Expression, ...]]:
    vertices = piece.exact_source_reference_vertices
    variables = algebra.axes(2, budget=budget)
    source: tuple[Expression, ...] = tuple(
        algebra.add(
            algebra.constant(vertices[0][axis], 2, budget=budget),
            algebra.sum_polynomials(
                tuple(
                    algebra.scale(
                        variable,
                        vertices[column + 1][axis] - vertices[0][axis],
                        budget=budget,
                    )
                    for column, variable in enumerate(variables)
                ),
                budget=budget,
            ),
            budget=budget,
        )
        for axis in range(2)
    )
    projective = piece.projective_map.reference_expressions()
    target = tuple(
        algebra.expression_compose(value, source, budget=budget) for value in projective
    )
    return source, target


def _sphere_projective_condition(
    piece: PreparedSphereChartPiece,
    source: tuple[Expression, ...],
    work: PreparationWork,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    expressions = piece.projective_map.reference_expressions()
    bounds = tuple(
        tuple(
            algebra.expression_bounds(
                algebra.expression_compose(
                    algebra.expression_derivative(value, axis, budget=work.enclosure),
                    source,
                    budget=work.enclosure,
                ),
                "simplex",
                2,
                budget=work.enclosure,
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
            "sphere_chart.sphere_projective_condition.pieces must satisfy the material reference contract; got Sphere material transport lacks a positive whole-piece Jacobian."
        )
    condition = fraction_endpoint(norm_squared / minimum, upper=True, saturate=True)
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "sphere_chart.sphere_projective_condition.reference_map must satisfy the regular reference-map contract; got Sphere material reference transport exceeds its whole-piece condition bound."
        )
    work.condition = max(work.condition, condition)


def _prepare_sphere_chart_compatible_transfer(
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
    budget: CoordinateEnclosureBudget,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Transport sphere material forms through their projective pieces."
    indices = _validate_sphere(
        source, target, prepared, source_geometry, target_geometry, budget=budget
    )

    def prepare_pieces(work: PreparationWork) -> tuple[MaterialCompatiblePiece, ...]:
        pieces: list[MaterialCompatiblePiece] = []
        for piece in prepared.pieces:
            old = indices[0][piece.source_cell_global_id]
            new = indices[1][piece.target_cell_global_id]
            with work.enclosure.temporary_scope():
                arguments, images = _sphere_piece_charts(piece, budget=work.enclosure)
                _sphere_projective_condition(
                    piece, arguments, work, budget=work.enclosure
                )
            pieces.append((old, new, arguments, images))
        return tuple(pieces)

    return prepare_material_chart_compatible_transfer(
        source,
        target,
        prepared,
        field_name=field_name,
        prepare_pieces=prepare_pieces,
        maximum_work=maximum_work,
        maximum_storage_bytes=maximum_storage_bytes,
        budget=budget,
    )


__all__ = [
    "prepare_sphere_chart_field_transfer",
    "prepare_sphere_chart_finite_volume_contents",
    "prepare_sphere_chart_compatible_transfer",
]


@checked
def prepare_sphere_chart_field_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    semantics: ChartFieldSemantics = "intensive",
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget | None = None,
) -> FiniteElementFieldTransfer:
    "Prepare the request with one explicitly bounded host -work ticket."
    if budget is not None:
        return _prepare_sphere_chart_field_transfer(
            source,
            target,
            prepared,
            field_name=field_name,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            semantics=semantics,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_work=maximum_work,
            maximum_subcells=maximum_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
            budget=budget,
        )
    with HostBatchTicket(
        maximum_work,
        queries_upper=field_transfer_query_upper(
            maximum_work, source_geometry, target_geometry
        ),
        bytes_upper=sys.maxsize,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=maximum_work, maximum_memory_bytes=sys.maxsize
        )
        with budget.activate():
            return _prepare_sphere_chart_field_transfer(
                source,
                target,
                prepared,
                field_name=field_name,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                semantics=semantics,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                maximum_work=maximum_work,
                maximum_subcells=maximum_subcells,
                maximum_binomial_terms=maximum_binomial_terms,
                budget=budget,
            )


@checked
def prepare_sphere_chart_finite_volume_contents(
    source: ChartEndpoint,
    target: ChartEndpoint,
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
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedSurfaceChartFiniteVolumeContents:
    "Prepare the request with one explicitly bounded host -work ticket."
    if budget is not None:
        return sphere_chart_finite_volume_contents(
            source,
            target,
            prepared,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_work=maximum_work,
            maximum_subcells=maximum_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
            budget=budget,
        )
    with HostBatchTicket(
        maximum_work,
        queries_upper=field_transfer_query_upper(
            maximum_work, source_geometry, target_geometry
        ),
        bytes_upper=sys.maxsize,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=maximum_work, maximum_memory_bytes=sys.maxsize
        )
        with budget.activate():
            return sphere_chart_finite_volume_contents(
                source,
                target,
                prepared,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                maximum_work=maximum_work,
                maximum_subcells=maximum_subcells,
                maximum_binomial_terms=maximum_binomial_terms,
                budget=budget,
            )


@checked
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
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Prepare the request with one explicitly bounded host -work ticket."
    if budget is not None:
        return _prepare_sphere_chart_compatible_transfer(
            source,
            target,
            prepared,
            field_name=field_name,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            maximum_work=maximum_work,
            maximum_storage_bytes=maximum_storage_bytes,
            budget=budget,
        )
    with HostBatchTicket(
        maximum_work,
        queries_upper=field_transfer_query_upper(
            maximum_work, source_geometry, target_geometry
        ),
        bytes_upper=maximum_storage_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=maximum_storage_bytes,
        )
        with budget.activate():
            return _prepare_sphere_chart_compatible_transfer(
                source,
                target,
                prepared,
                field_name=field_name,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                maximum_work=maximum_work,
                maximum_storage_bytes=maximum_storage_bytes,
                budget=budget,
            )
