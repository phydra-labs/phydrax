#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"Transport scalar fields through projective sphere material correspondence.\n\nPhysical reference maps are rational, not affine maps fitted to corner images.\nEvery mixed form uses the true projective quotient/Jacobian on an old\nreference overlap triangle. Positive-denominator series carry an explicit\nremainder and reuse the ordered signed sqrt-Gram integration owner."

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction
from typing import assert_never, Literal, NamedTuple, overload, TypedDict

import numpy as np
from numpy.typing import NDArray

import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.discretization._sphere_chart_records as imported_sphere_chart_deformation
from phydrax.discretization._sphere_chart_records import (
    PreparedSphereChartDeformation,
    PreparedSphereChartPiece,
    SphereChartPieceBank,
    SphereChartValidation,
)
from phydrax.geometry._rational import fraction_endpoint

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._resource_ticket import HostBatchTicket
from ..._strict import StrictModule
from ...geometry._sphere_material_atlas import (
    SphereProjectiveActionBank,
    SphereProjectiveReferenceMap,
)
from ...linalg import ArraySpace
from ...linalg._hermitian_spectral import fraction_sqrt_interval, reserve_fraction_work
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
    MaterialInvariantCache,
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
    validation: SphereChartValidation | None,
) -> _CellIndices:
    "Bind both endpoints; reuse this request's correspondence proof when given."
    bound_surface_endpoint(source, source_geometry, "source", prepared, budget=budget)
    bound_surface_endpoint(target, target_geometry, "target", prepared, budget=budget)
    if validation is None:
        prepared.require_bound(
            source.mesh, source_geometry, target.mesh, target_geometry, budget=budget
        )
    else:
        validation.require(
            prepared,
            source.mesh,
            source_geometry,
            target.mesh,
            target_geometry,
            budget=budget,
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
    validation: SphereChartValidation,
) -> FiniteElementFieldTransfer:
    si, ti = source.field_index(name), target.field_index(name)
    source_cells, target_cells = field_cells(source, si), field_cells(target, ti)
    key = (
        source.dof_maps[si].dof_map_id,
        target.dof_maps[ti].dof_map_id,
        tuple(algebra.element_identity(element, budget=budget) for element in source.elements[si]),
        tuple(algebra.element_identity(element, budget=budget) for element in target.elements[ti]),
        canonical_fingerprint(array_tree_fingerprint(source.dof_maps[si].cell_dofs)),
        canonical_fingerprint(array_tree_fingerprint(target.dof_maps[ti].cell_dofs)),
        source.dof_maps[si].global_dof_count,
        target.dof_maps[ti].global_dof_count,
    )
    retained = validation.h1_cache.get(key)
    if retained is not None:
        return _finish_h1(source, target, prepared, name, retained)
    if any(
        element.conformity != "H1" or source_basis(element, budget=budget) is None
        for element, _ in source_cells + target_cells
    ):
        raise ValueError(
            "sphere_chart.prepare_h1.pieces must satisfy the material reference contract; got Sphere H1 transport requires ordered scalar nodal sources."
        )
    source_indices, target_indices = indices
    # A target functional at a piece vertex has that vertex's retained source
    # point as its exact preimage: require_bound re-proved every piece target
    # vertex as the projective image of its source vertex. Only functionals
    # strictly inside a piece edge or interior need the inverse projective map.
    bindings: list[tuple[PreparedSphereChartPiece, int, int, _Point | None]] = []
    inverse_pairs: set[tuple[int, int]] = set()
    for piece in prepared.pieces._iter_kernel_pieces(budget=budget):
        target_row = target_indices[int(piece.target_cell_global_id)]
        target_element, _ = target_cells[target_row]
        nodes = np.asarray(target_element.reference_nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(
                "sphere_chart.prepare_h1.pieces must satisfy the material reference contract; got Sphere H1 functionals require two-dimensional reference nodes."
            )
        target_vertices = piece.exact_target_reference_vertices
        for node in range(nodes.shape[0]):
            point = Fraction(float(nodes[node, 0])), Fraction(float(nodes[node, 1]))
            if point in target_vertices:
                vertex = target_vertices.index(point)
                bindings.append(
                    (
                        piece,
                        target_row,
                        node,
                        piece.exact_source_reference_vertices[vertex],
                    )
                )
            elif contains(
                target_vertices,
                point,
                degenerate_message="sphere_chart.reference_triangle must have nonzero area; got degenerate triangle.",
            ):
                bindings.append((piece, target_row, node, None))
                inverse_pairs.add(
                    (int(piece.source_cell_global_id), int(piece.target_cell_global_id))
                )
    inverse_maps: dict[tuple[int, int], SphereProjectiveReferenceMap] = {}
    if inverse_pairs:
        with budget.live_storage() as storage:
            # Both atlases are authenticated once; each needed pair's exact
            # inverse map is charged to this request and shared by its pieces.
            actions = SphereProjectiveActionBank.authenticate(
                prepared.target_atlas,
                prepared.source_atlas,
                budget=budget,
                storage=storage,
            )
            for pair in sorted(inverse_pairs):
                inverse_maps[pair] = prepared.target_atlas.projective_reference_map(
                    pair[1], prepared.source_atlas, pair[0], budget=budget, actions=actions
                )
    candidates, columns, values = [], [], []
    for piece, target_row, node, reference in bindings:
        source_element, source_route = source_cells[
            source_indices[int(piece.source_cell_global_id)]
        ]
        _, target_route = target_cells[target_row]
        if reference is None:
            target_element, _ = target_cells[target_row]
            nodes = np.asarray(target_element.reference_nodes, dtype=np.float64)
            point = Fraction(float(nodes[node, 0])), Fraction(float(nodes[node, 1]))
            reference = projective_point(
                inverse_maps[
                    (int(piece.source_cell_global_id), int(piece.target_cell_global_id))
                ].exact_coefficients,
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
    retained = rows.astype(np.int64), coefficients, continuity
    budget.retain_basis((key, retained))
    validation.h1_cache[key] = retained
    return _finish_h1(source, target, prepared, name, retained)


def _finish_h1(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSphereChartDeformation,
    name: str,
    retained: tuple[NDArray[np.int64], _FloatArray, float],
) -> FiniteElementFieldTransfer:
    """Publish each field binding from one request-owned interpolation action."""
    si, ti = source.field_index(name), target.field_index(name)
    source_cells = field_cells(source, si)
    width = max(element.local_dof_count for element, _ in source_cells)
    ss, ts = source.dof_maps[si].global_dof_count, target.dof_maps[ti].global_dof_count
    rows, coefficients, continuity = retained
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


def _constant_density(density: Expression) -> Fraction | None:
    "Return G when sqrt(G) is one constant on an affine cell, else None."
    if isinstance(density, dict) and _degree(density) == 0:
        return algebra.require_rational_reference_scalar(
            density.get((0, 0), Fraction(0))
        )
    return None


def _reference_triangle_integral(
    polynomial: Polynomial, *, budget: CoordinateEnclosureBudget
) -> Fraction:
    "Exact unit-triangle moments: integral of x**a y**b is a! b! / (a + b + 2)!."
    coefficients = tuple(
        algebra.require_rational_reference_scalar(value) for value in polynomial.values()
    )
    bits = max(
        (
            max(value.numerator.bit_length(), value.denominator.bit_length())
            for value in coefficients
        ),
        default=1,
    )
    # One product and one sum per monomial.
    reserve_fraction_work(budget, 2 * len(coefficients), len(coefficients) + 2, bits + 64)
    return sum(
        (
            coefficient
            * Fraction(
                math.factorial(index[0]) * math.factorial(index[1]),
                math.factorial(index[0] + index[1] + 2),
            )
            for index, coefficient in zip(polynomial, coefficients, strict=True)
        ),
        Fraction(0),
    )


def _density_integral(
    density: Expression,
    polynomial: Polynomial,
    roots: dict[int, tuple[Fraction, Fraction]],
    cell: int,
    budgets: IntegrationBudgets,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    "Integrate on one physical cell; an affine cell shares one root enclosure."
    squared = _constant_density(density)
    if squared is None:
        return integral(density, polynomial, budgets, budget=budget)
    root = roots.get(cell)
    if root is None:
        root = fraction_sqrt_interval(squared, budget=budget)
    value, error, roots[cell] = _affine_scaled_integral(
        squared,
        root,
        _reference_triangle_integral(polynomial, budget=budget),
        budgets,
        budget=budget,
    )
    return value, error


def _vertex_values(
    basis: Sequence[Polynomial], points: _Triangle
) -> tuple[tuple[tuple[Fraction, ...], ...], int]:
    "Exact values of degree-one basis polynomials at three points, with op count."
    rows: list[tuple[Fraction, ...]] = []
    operations = 0
    for polynomial in basis:
        row = []
        for point in points:
            value = None
            for index, coefficient in polynomial.items():
                term = algebra.require_rational_reference_scalar(coefficient)
                if not term:
                    continue
                for axis, power in enumerate(index):
                    if power:
                        coordinate = point[axis]
                        if term == 1:
                            term = coordinate
                        elif term == -1:
                            term = -coordinate
                        elif coordinate != 1:
                            term *= coordinate
                            operations += 1
                if value is None:
                    value = term
                else:
                    value += term
                    operations += 1
            row.append(Fraction(0) if value is None else value)
        rows.append(tuple(row))
    return tuple(rows), operations


_DYADIC_BITS = 96
# Relative rounding error of ``_dyadic``: half a unit in the 96-bit mantissa.
_DYADIC_ROUNDOFF = Fraction(1, 2**_DYADIC_BITS)


def _dyadic(value: Fraction, /) -> Fraction:
    "Round to the nearest 96-bit-mantissa dyadic; relative error at most 2**-96."
    if value == 0:
        return value
    numerator, denominator = value.as_integer_ratio()
    exponent = abs(numerator).bit_length() - denominator.bit_length()
    scale = _DYADIC_BITS - exponent
    if scale >= 0:
        return Fraction(round(Fraction(numerator << scale, denominator)), 1 << scale)
    return Fraction(round(Fraction(numerator, denominator << -scale)) << -scale)


_BINARY64_ACTION_ROUNDOFF = Fraction(16, 2**53 - 16)
_BINARY64_ACTION_UNDERFLOW = Fraction(256, 2**1022)
_BINARY64_CONTRACTION_LIMIT = Fraction(float(np.finfo(np.float64).max)) / 4
type _AffineValues = tuple[tuple[Fraction, ...], ...]
type _AffinePointKey = tuple[tuple[int, int], ...]
type _AffineBasisKey = tuple[tuple[tuple[int, int], ...], ...]


class _AffineBasisOperand(NamedTuple):
    coefficients: _AffineValues
    values: dict[_AffinePointKey, tuple[tuple[Fraction, ...], tuple[Fraction, ...]]]


type _AffineOperandCache = dict[_AffineBasisKey, _AffineBasisOperand]


def _affine_operand(
    coefficients: _AffineValues,
    cache: _AffineOperandCache,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AffineBasisOperand:
    """Canonicalize actual affine coefficients, never an element/map label.

    Keys contain integer pairs only. The private numerical graph owns copies
    of every Fraction, even for an identity action at a bank vertex.
    """
    budget.reserve(3 * len(coefficients), 128 + 256 * len(coefficients))
    key = tuple(tuple(value.as_integer_ratio() for value in row) for row in coefficients)
    previous = cache.get(key)
    if previous is not None:
        return previous
    bits = max(
        max(abs(numerator).bit_length(), denominator.bit_length())
        for row in key for numerator, denominator in row
    )
    reserve_fraction_work(budget, 9 * len(coefficients), 3 * len(coefficients), bits)
    owned = tuple(tuple(Fraction(*value) for value in row) for row in key)
    result = _AffineBasisOperand(owned, {})
    budget.retain_basis((key, owned))
    budget.retain_operand_metadata(256)
    cache[key] = result
    return result


def _affine_basis_operand(
    basis: Sequence[Polynomial],
    cache: _AffineOperandCache,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AffineBasisOperand:
    budget.reserve(sum(len(term) for term in basis), 128 + 128 * len(basis))
    rows = []
    for term in basis:
        if any(
            len(index) != 2 or any(power < 0 for power in index) or sum(index) > 1
            for index, value in term.items() if value
        ):
            raise ValueError(
                "sphere_chart.affine_basis_operand.basis must be affine in two variables; got unsupported exponent support."
            )
        rows.append(
            tuple(
                algebra.require_rational_reference_scalar(term.get(index, Fraction(0)))
                for index in ((0, 0), (1, 0), (0, 1))
            )
        )
    return _affine_operand(tuple(rows), cache, budget=budget)


def _affine_vertex_values(
    operand: _AffineBasisOperand,
    points: _Triangle,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_AffineValues, _AffineValues]:
    """Reuse exact evaluations and their one-time dyadic rounding by point.

    Repeated corners of the actual triangulated overlaps share only pure
    operands. Area, density, error goals and acceptance are not cached.
    """
    count = len(operand.coefficients)
    budget.reserve(0, 256 + 256 * count)
    vertices = []
    for point in points:
        budget.reserve(2, 256)
        key = tuple(value.as_integer_ratio() for value in point)
        previous = operand.values.get(key)
        if previous is None:
            # Count the identity-eliminated affine arithmetic before performing
            # it; copying/rounding each resulting scalar is also admitted here.
            operations = 0
            for row in operand.coefficients:
                operations += max(0, sum(bool(value) for value in row) - 1)
                operations += sum(
                    bool(coefficient) and coefficient not in (1, -1) and coordinate != 1
                    for coefficient, coordinate in zip(row[1:], point, strict=True)
                )
            bits = max(
                max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                for value in (*point, *(value for row in operand.coefficients for value in row))
            )
            # Three normalization visits for each owned Fraction, separately
            # from the dyadic-rounding visits of the original exact kernel.
            reserve_fraction_work(budget, operations + 6 * count, 4 * count, 8 * bits + 128)
            exact = []
            for row in operand.coefficients:
                total = row[0] if row[0] else None
                for coefficient, coordinate in zip(row[1:], point, strict=True):
                    if not coefficient:
                        continue
                    term = (
                        coordinate if coefficient == 1
                        else -coordinate if coefficient == -1
                        else coefficient if coordinate == 1
                        else coefficient * coordinate
                    )
                    total = term if total is None else total + term
                value = Fraction(0) if total is None else total
                exact.append(Fraction(*value.as_integer_ratio()))
            values = tuple(exact)
            previous = values, tuple(_dyadic(value) for value in values)
            budget.retain_basis((key, previous))
            budget.retain_operand_metadata(256)
            operand.values[key] = previous
        vertices.append(previous)
    return (
        tuple(tuple(vertex[0][row] for vertex in vertices) for row in range(count)),
        tuple(tuple(vertex[1][row] for vertex in vertices) for row in range(count)),
    )



type _ReciprocalMomentKey = tuple[tuple[Fraction, ...], int]
type _ReciprocalMoments = tuple[tuple[tuple[Fraction, ...], ...], Fraction]
type _ReciprocalMomentCache = dict[_ReciprocalMomentKey, _ReciprocalMoments]
type _ReciprocalSeries = tuple[int, tuple[int, ...], tuple[Fraction, ...]]
type _ReciprocalSeriesCache = dict[int, _ReciprocalSeries]
type _ReciprocalJetCache = dict[tuple[int, int, int], tuple[Fraction, ...]]


def _finite_reciprocal_series(
    terms: int,
    cache: _ReciprocalSeriesCache,
    *,
    budget: CoordinateEnclosureBudget,
) -> _ReciprocalSeries:
    """Factor the exact finite polynomial's integer coefficients by its order.

    These coefficients, not a convergence decision, are shared across actual
    denominator nodes. The three boundary coefficients use successive integer
    products instead of independently constructing three larger factorials.
    """
    previous = cache.get(terms)
    if previous is not None:
        return previous
    order = terms + 1
    bits = order * max(1, order.bit_length()) + 16
    reserve_fraction_work(budget, order + 6, order + 8, bits)
    last = math.factorial(order)
    coefficients = tuple(
        -(last // degree) if degree % 2 == 0 else last // degree
        for degree in range(1, order + 1)
    )
    product = 1
    boundary = []
    for degree in range(2, 5):
        product *= terms + degree
        boundary.append(Fraction(-1 if terms % 2 else 1, product))
    result = last, coefficients, tuple(boundary)
    budget.retain_basis((terms, result))
    budget.retain_operand_metadata(128)
    cache[terms] = result
    return result


def _unit_reciprocal_moments(
    delta: tuple[Fraction, ...],
    terms: int,
    cache: _ReciprocalMomentCache,
    *,
    budget: CoordinateEnclosureBudget,
    series_cache: _ReciprocalSeriesCache | None = None,
    jet_cache: _ReciprocalJetCache | None = None,
) -> _ReciprocalMoments:
    """Earn pure unit-area moments, independent of chart, area and scale.

    The caller orders the denominator ratios canonically. Only actual immutable
    ratios and the exact truncation order identify an operand; neither an error
    verdict nor a piece/container identity is reusable here. The finite series
    is the fourth confluent divided difference of

        F(x) = 4 sum(n=0..K) (-1)**n n! x**(n+4)/(n+4)!.

    Repeated nodes use exact derivatives, not perturbed nodes. To evaluate all
    derivatives, only one finite log polynomial per distinct nonzero ratio is
    needed. Write S_p = sum (-1)**n n! x**(n+p)/(n+p)!; S_1 is that polynomial,
    and (p-1) S_p = (1+x) S_(p-1) - x**(p-1)/(p-1)!
    - (-1)**K (K+1)! x**(K+p)/(K+p)!. This is a finite polynomial identity:
    no logarithm approximation or new error term replaces the dyadic series.
    """
    key = delta, terms
    previous = cache.get(key)
    if previous is not None:
        return previous
    distinct = tuple(dict.fromkeys(delta))
    nonzero = tuple(value for value in distinct if value)
    order = terms + 1
    # Admit the finite node/difference index before building it. Its capacity
    # is bounded by three length-five confluent tables, regardless of order.
    budget.reserve(0, 8192)
    nodes = {
        value: tuple(sorted((*delta, value, value))) for value in distinct
    }
    differences = {
        points[start:start + degree + 1]
        for points in nodes.values()
        for degree in range(1, 5)
        for start in range(5 - degree)
        if points[start] != points[start + degree]
    }
    # Difference arithmetic remains paid on every new actual node tuple.
    # Only the order-only polynomial and individual exact derivative jets
    # may be shared; truncation/error admission stays with each mixed form.
    bits = 8 * (
        max(
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            for value in delta
        )
        * (terms + 5)
        + order * max(1, order.bit_length()) + 32
    )
    reserve_fraction_work(
        budget, 3 * len(differences) + 8 * len(distinct) + 33, 128, bits
    )
    finite = (
        _finite_reciprocal_series(
            terms, {} if series_cache is None else series_cache, budget=budget
        )
        if nonzero
        else None
    )
    jets: dict[Fraction, tuple[Fraction, ...]] = {}
    for value in distinct:
        jet_key = (*value.as_integer_ratio(), terms)
        previous_jet = None if jet_cache is None else jet_cache.get(jet_key)
        if previous_jet is not None:
            jets[value] = previous_jet
            continue
        if value == 0:
            # F has no terms below degree four, and its x**4 coefficient is 1/6.
            jets[value] = (Fraction(0),) * 4 + (Fraction(1, 6),)
            continue
        assert finite is not None
        last, coefficients, boundary = finite
        reserve_fraction_work(budget, 3 * order + 33, 48 + order, bits)
        scaled, denominator = value.as_integer_ratio()
        shift = denominator.bit_length() - 1
        numerator, power = 0, 1
        for degree in range(order, 0, -1):
            numerator = (
                numerator * scaled
                + (coefficients[degree - 1] << (shift * (order - degree)))
            )
            power *= scaled
        logarithm = Fraction(numerator * scaled, last << (shift * order))
        high_power = Fraction(power, 1 << (shift * order))
        one = 1 + value
        signed_power = -high_power if terms % 2 else high_power
        primitives = [(1 + signed_power) / one, logarithm]
        low_power = Fraction(1)
        for degree, correction in zip(range(2, 5), boundary, strict=True):
            high_power *= value
            low_power *= value
            primitives.append(
                (
                    one * primitives[-1]
                    - low_power / math.factorial(degree - 1)
                    - correction * high_power
                ) / (degree - 1)
            )
        jet = tuple(
            Fraction(4, math.factorial(degree)) * primitives[4 - degree]
            for degree in range(5)
        )
        jets[value] = jet
        # Bound optional jet retention independently of the number of pieces.
        # Saturation changes only reuse: all misses still earn the same exact
        # finite polynomial and confluent derivatives on the caller's budget.
        if jet_cache is not None and len(jet_cache) < 64:
            budget.retain_basis((jet_key, jet))
            budget.retain_operand_metadata(128)
            jet_cache[jet_key] = jet
    # Several diagonals share lower divided differences. Earn each actual
    # immutable node tuple once, including confluent derivative operands.
    divided: dict[tuple[Fraction, ...], Fraction] = {}
    for points in sorted(differences, key=len):
        left, right = points[:-1], points[1:]
        left_value = (
            jets[left[0]][len(left) - 1]
            if left[0] == left[-1]
            else divided[left]
        )
        right_value = (
            jets[right[0]][len(right) - 1]
            if right[0] == right[-1]
            else divided[right]
        )
        divided[points] = (right_value - left_value) / (points[-1] - points[0])
    diagonals: dict[Fraction, Fraction] = {}
    moment_bound = Fraction(0)
    for value, points in nodes.items():
        moment = jets[value][4] if points[0] == points[-1] else divided[points]
        diagonals[value] = _dyadic(moment)
        moment_bound = max(moment_bound, abs(moment))
    denominators = tuple(1 + value for value in delta)
    diagonal = tuple(diagonals[value] for value in delta)
    excess = tuple(
        value * (Fraction(1, 3) - value * moment)
        for value, moment in zip(denominators, diagonal, strict=True)
    )
    weighted = [[Fraction(0)] * 3 for _ in range(3)]
    for k in range(3):
        weighted[k][k] = denominators[k] * diagonal[k]
        for l in range(k + 1, 3):
            m = 3 - k - l
            numerator = excess[k] + excess[l] - excess[m]
            # Cancel the row denominator before doing the two exact divisions.
            weighted[k][l] = numerator / (2 * denominators[l])
            weighted[l][k] = numerator / (2 * denominators[k])
    result = tuple(tuple(row) for row in weighted), moment_bound
    # The immutable numerical operands outlive a piece's temporary scope.
    # Retention authenticates/accounts the complete key/value graph, not a
    # mutable piece or a producer's acceptance verdict.
    budget.retain_basis((key, result))
    budget.retain_operand_metadata(256)
    cache[key] = result
    return result


def _factorized_float_form(
    tau: tuple[tuple[Fraction, ...], ...],
    weighted: tuple[tuple[Fraction, ...], ...],
    phi: tuple[tuple[Fraction, ...], ...],
    root: tuple[Fraction, Fraction],
    common: Fraction,
    absolute: Fraction,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_FloatArray, Fraction] | None:
    """Publish the same factorized form with a certified binary64 contraction.

    Each inner product has length three. Input rounding, both products and
    the final root multiplication are covered by gamma_16; this also covers
    fused inner products. The separate absolute term covers gradual or flushed
    subnormal underflow in inputs, products and sums, and its amplification.
    If that enclosure cannot meet the requested goal, exact contractions
    below remain available; the mathematical mixed form is unchanged.
    """
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for value in (
            *(value for matrix in (tau, weighted, phi) for row in matrix for value in row),
            *root, common, absolute, _BINARY64_ACTION_UNDERFLOW,
        )
    )
    reserve_fraction_work(budget, 40, 24, 16 * bits + 128)
    tau_max = max((abs(value) for row in tau for value in row), default=Fraction(0))
    phi_sum = max(sum((abs(value) for value in row), Fraction(0)) for row in phi)
    weight_max = max(abs(value) for row in weighted for value in row)
    lower, upper = root
    reference_bound = 3 * tau_max * weight_max * phi_sum
    underflow = (
        _BINARY64_ACTION_UNDERFLOW
        * max(Fraction(1), tau_max)
        * max(Fraction(1), weight_max)
        * max(Fraction(1), phi_sum)
        * max(Fraction(1), upper)
    )
    bound = (
        common
        + (upper - lower) / 2 * reference_bound
        + _BINARY64_ACTION_ROUNDOFF * upper * reference_bound
        + underflow
    )
    # Bound every intermediate, not only the final physical form, before
    # entering the ordinary binary64 dense contraction.
    if bound > absolute or max(
        tau_max, phi_sum, weight_max, upper,
        3 * weight_max * phi_sum, reference_bound, upper * reference_bound,
    ) > _BINARY64_CONTRACTION_LIMIT:
        return None
    form = (
        np.asarray(tau, dtype=np.float64)
        @ np.asarray(weighted, dtype=np.float64)
        @ np.asarray(phi, dtype=np.float64).T
    )
    form *= float((lower + upper) / 2)
    return form, len(tau) * len(phi) * bound


def _affine_conservative_form(
    piece: PreparedSphereChartPiece,
    old_basis: Sequence[Polynomial],
    new_basis: Sequence[Polynomial],
    root: tuple[Fraction, Fraction],
    absolute: Fraction,
    maximum_terms: int,
    *,
    budget: CoordinateEnclosureBudget,
    moment_cache: _ReciprocalMomentCache | None = None,
    series_cache: _ReciprocalSeriesCache | None = None,
    jet_cache: _ReciprocalJetCache | None = None,
    operand_cache: _AffineOperandCache | None = None,
    basis_operands: tuple[_AffineBasisOperand, _AffineBasisOperand] | None = None,
) -> tuple[_FloatArray, Fraction]:
    """Mixed conservative-density form on one affine-source projective piece.

    With piece barycentrics mu, the projective target basis is
    psi_i(M x) = sum_k tau_ik D_k mu_k / D, D = sum_k D_k mu_k > 0, and the old
    basis is phi_j = sum_l Phi_jl mu_l (degree-one bases are fixed by their
    vertex values). Hence E = sqrt(G) tau diag(D) W Phi^T with the homogeneous
    moments W_kl = int mu_k mu_l / D. The exact identities
    sum_k D_k W_kl = |T|/3 leave three transcendental moments w_k = W_kk,
    summed from 1/D = c^-1 sum_n (-delta.mu)^n with Dirichlet moments
    int mu^a = 2|T| a!/(|a|+2)!: w_k = (2|T|/c) sum_n (-1)^n 2 n!/(n+4)!
    [t^n] prod_m (1 - delta_m t)^-1 (1 - delta_k t)^-2. The off-diagonal
    moments are 2 D_k D_l W_kl = u_k + u_l - u_m, u_k = D_k (|T|/3 - D_k w_k).

    Inputs are rounded once to 96-bit dyadics (relative error u = 2**-96); the
    series is exact integer arithmetic on the dyadic ratios
    delta_k = D_k/c - 1 (c a dyadic center, so D_k = c (1 + delta_k) exactly
    defines the perturbed denominator); the moment identities are exact rational
    arithmetic. Basis actions use enclosed binary64 contractions when their
    rigorous error fits, otherwise exact rational contractions. The published
    per-entry bound covers input
    rounding (8u |T| max|tau| max|Phi| sqrt(G)), the series remainder
    (|T|/(6c)) rho^(K+1)/(1-rho) and the dyadic rounding of each unit-area
    diagonal moment (scaled exactly by |T|/c),
    amplified through the identities and both basis actions, the sqrt(G)
    enclosure and binary64 publication.
    ``absolute`` is the per-entry absolute error goal; the remainder receives
    half of it.
    """
    shape = (len(new_basis), len(old_basis))
    area = abs(piece.source_reference_area)
    if area == 0:
        return np.zeros(shape, dtype=np.float64), Fraction(0)
    source = piece.exact_source_reference_vertices
    target = piece.exact_target_reference_vertices
    weights = piece.projective_map.exact_denominator
    operands = {} if operand_cache is None else operand_cache
    denominator_operand = _affine_operand((weights,), operands, budget=budget)
    exact_rows, rounded_rows = _affine_vertex_values(
        denominator_operand, source, budget=budget
    )
    exact, rounded = exact_rows[0], rounded_rows[0]
    if basis_operands is None:
        basis_operands = (
            _affine_basis_operand(old_basis, operands, budget=budget),
            _affine_basis_operand(new_basis, operands, budget=budget),
        )
    _, phi = _affine_vertex_values(basis_operands[0], source, budget=budget)
    _, tau = _affine_vertex_values(basis_operands[1], target, budget=budget)
    if min(exact) <= 0:
        raise ValueError(
            "sphere_chart.affine_conservative_form.pieces must satisfy the material reference contract; got a nonpositive projective denominator at a piece vertex."
        )
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for value in (
            *rounded, area, *root, absolute,
            *(value for matrix in (tau, phi) for row in matrix for value in row),
        )
    )
    reserve_fraction_work(
        budget, 67 + 2 * len(tau) + 4 * len(phi),
        64 + 3 * sum(shape), 32 * bits + 256,
    )
    center = _dyadic((min(rounded) + max(rounded)) / 2)
    # D_k := c (1 + delta_k) with dyadic delta_k defines the perturbed problem.
    delta = tuple(_dyadic(value / center - 1) for value in rounded)
    rho = max(abs(value) for value in delta)
    if rho > Fraction(1, 2):
        raise CellGeometryTransitionError(
            "resource_limit",
            "sphere_chart.affine_conservative_form.budget must satisfy the admitted resource limits; got Projective denominator spread exceeds the moment series domain.",
            measured=float(rho),
            limit=0.5,
        )
    # The common positive center cancels exactly from diag(D) W and from the
    # null-direction error bound. Keep only dimensionless denominator ratios.
    denominators = tuple(1 + value for value in delta)
    size = _dyadic(area)
    lower_root, upper_root = root
    lower, upper = min(denominators), max(denominators)
    tau_bound = max((abs(value) for row in tau for value in row), default=Fraction(0))
    phi_bound = max((abs(value) for row in phi for value in row), default=Fraction(0))
    # Both the true and reconstructed moments satisfy W D = |T|/3.
    # Therefore their error annihilates D on either side. Center the two
    # basis actions in that null direction before bounding their amplification;
    # constant target tests have identically zero moment-series error.
    rows = max(
        sum(
            (abs(row[k] - row[0]) * denominators[k] for k in (1, 2)),
            Fraction(0),
        )
        for row in tau
    )
    ratios = tuple(denominators[k] / denominators[0] for k in (1, 2))
    columns = max(
        sum(
            (abs(row[k] - row[0] * ratios[k - 1]) for k in (1, 2)),
            Fraction(0),
        )
        for row in phi
    )
    # The identities amplify diagonal moment errors by at most
    # 3 max(D)^2 / (2 min(D)^2); the centered actions by rows * columns.
    coupling = upper_root * max(Fraction(1), 3 * upper**2 / (2 * lower**2)) * rows * columns
    perturbation = 8 * _DYADIC_ROUNDOFF * upper_root * area * tau_bound * phi_bound
    tail = coupling * size / 6 * rho / (1 - rho)
    goal = absolute / 2
    terms = 0
    while tail > goal and terms < maximum_terms:
        terms += 1
        recurrence_bits = sum(
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            for value in (tail, rho)
        )
        reserve_fraction_work(budget, 1, 1, recurrence_bits + 16)
        tail *= rho
    if rho >= 1 or tail > goal:
        raise CellGeometryTransitionError(
            "resource_limit",
            "sphere_chart.affine_conservative_form.budget must satisfy the admitted resource limits; got Projective moment series exceeds its term budget.",
            measured=maximum_terms,
            limit=maximum_terms,
        )
    # Reindex all corners together: a permutation changes neither the moment
    # integral nor either basis action. Area and the positive center are not
    # operands of the unit-area reciprocal functional.
    order = tuple(sorted(range(3), key=delta.__getitem__))
    canonical_delta = tuple(delta[k] for k in order)
    unit_weighted, moment_bound = _unit_reciprocal_moments(
        canonical_delta,
        terms,
        {} if moment_cache is None else moment_cache,
        budget=budget,
        series_cache=series_cache,
        jet_cache=jet_cache,
    )
    tau = tuple(tuple(row[k] for k in order) for row in tau)
    phi = tuple(tuple(row[k] for k in order) for row in phi)
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for value in (
            size, coupling, tail, moment_bound, perturbation,
            *(value for row in unit_weighted for value in row),
        )
    )
    reserve_fraction_work(budget, 14, 32, 8 * bits + 128)
    weighted = tuple(tuple(size * value for value in row) for row in unit_weighted)
    # Unit diagonal rounding is followed only by exact rational rescaling.
    # The null-direction coupling still bounds the same reconstructed W.
    remainder = tail + coupling * 2 * _DYADIC_ROUNDOFF * size * moment_bound
    fast = _factorized_float_form(
        tau, weighted, phi, root, perturbation + remainder, absolute, budget=budget
    )
    if fast is not None:
        return fast
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for value in (
            *(value for matrix in (tau, weighted, phi) for row in matrix for value in row),
            *root, perturbation, remainder, absolute,
        )
    )
    contractions = (
        6 * sum(bool(value) for row in phi for value in row)
        + 2 * len(phi) * sum(bool(value) for row in tau for value in row)
    )
    reserve_fraction_work(
        budget, contractions + 6 * shape[0] * shape[1],
        32 + 3 * len(phi) + 2 * shape[0] * shape[1], 16 * bits + 128,
    )
    right = []
    for k in range(3):
        row = []
        for values in phi:
            total = Fraction(0)
            for l in range(3):
                if values[l]:
                    total += weighted[k][l] * values[l]
            row.append(total)
        right.append(row)
    reference = []
    for values in tau:
        row = []
        for j in range(len(phi)):
            total = Fraction(0)
            for k in range(3):
                if values[k]:
                    total += values[k] * right[k][j]
            row.append(total)
        reference.append(row)
    middle, half_width = (lower_root + upper_root) / 2, (upper_root - lower_root) / 2
    form = np.empty(shape, dtype=np.float64)
    error = Fraction(0)
    common = perturbation + remainder
    for i, row in enumerate(reference):
        for j, value in enumerate(row):
            central = value * middle
            form[i, j] = float(central)
            bound = (
                common
                + half_width * abs(value)
                + abs(Fraction(float(form[i, j])) - central)
            )
            if bound > absolute:
                raise CellGeometryTransitionError(
                    "resource_limit",
                    "sphere_chart.affine_conservative_form.budget must satisfy the admitted resource limits; got Projective mixed form exceeds its error budget.",
                    measured=float(bound),
                    limit=float(absolute),
                )
            error += bound
    return form, error

type _PolynomialBasisKey = tuple[
    tuple[tuple[tuple[int, ...], int, int], ...], ...
]


def _reference_basis_key(
    basis: Sequence[Polynomial], *, budget: CoordinateEnclosureBudget
) -> _PolynomialBasisKey:
    """Snapshot every current exponent/coefficient, not an element's identity.

    Unit-triangle reference forms have no geometry operand. They can be
    shared across cells only when this complete actual numerical basis agrees.
    No caller-owned Fraction is retained as a dictionary key.
    """
    terms = sum(len(term) for term in basis)
    budget.reserve(terms, 128 + 256 * terms + 64 * len(basis))
    return tuple(
        tuple(
            (tuple(index), *algebra.require_rational_reference_scalar(value).as_integer_ratio())
            for index, value in sorted(term.items())
            if value
        )
        for term in basis
    )


def _dg_reference_forms(
    basis: Sequence[Polynomial], *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]]:
    """Prepare geometry-independent triangle moments once per actual basis."""
    measures = tuple(_reference_triangle_integral(term, budget=budget) for term in basis)
    mass = [[Fraction(0)] * len(basis) for _ in basis]
    for i, first in enumerate(basis):
        for j in range(i, len(basis)):
            value = _reference_triangle_integral(
                multiply(first, basis[j], budget=budget), budget=budget
            )
            mass[i][j] = mass[j][i] = value
    return measures, tuple(tuple(row) for row in mass)



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
    base_count = ss + ts + sum(len(basis) ** 2 for _, _, basis in tc)
    mixed_count = (
        len(prepared.pieces)
        * max(len(basis) for _, _, basis in sc)
        * max(len(basis) for _, _, basis in tc)
    )
    form_count = base_count + mixed_count
    affine_mixed = semantics == "conservative-density" and all(
        _constant_density(density) is not None
        and all(_degree(term) <= 1 for term in basis)
        for cells, densities in ((sc, grams[0]), (tc, grams[1]))
        for (_, _, basis), density in zip(cells, densities, strict=True)
    )
    local_budgets: IntegrationBudgets = {
        **budgets,
        "absolute_tolerance": float(np.nextafter(
            budgets["absolute_tolerance"]
            / (4 * base_count if affine_mixed else form_count), 0.0
        )),
        "relative_tolerance": budgets["relative_tolerance"] / form_count,
    }
    # Numerical forms outlive each piece's moment scratch. Admit their full
    # simultaneous storage before allocating them, including one summation
    # temporary and the bounded mixed-row dictionaries.
    largest_block = max(len(basis) for _, _, basis in sc) * max(
        len(basis) for _, _, basis in tc
    )
    budget.reserve(
        0,
        8 * (form_count + ss + ts + largest_block)
        + 512 * (len(tc) + len(prepared.pieces) + 4),
    )
    sm, tm, se, te = np.zeros(ss), np.zeros(ts), np.zeros(ss), np.zeros(ts)
    masses: list[_FloatArray] = []
    mixed: list[dict[int, _FloatArray]] = [dict() for _ in tc]
    errors = 0.0
    roots: tuple[dict[int, tuple[Fraction, Fraction]], ...] = ({}, {})
    reference_forms: dict[
        _PolynomialBasisKey,
        tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]],
    ] = {}
    for cells, density, measures, uncertainties, is_target in (
        (sc, grams[0], sm, se, False),
        (tc, grams[1], tm, te, True),
    ):
        cell_roots = roots[int(is_target)]
        for cell, (element, route, basis) in enumerate(cells):
            if element.cell_kind != "triangle":
                raise ValueError(
                    "sphere_chart.prepare_dg.pieces must satisfy the material reference contract; got Sphere DG requires ordered scalar triangle fields."
                )
            squared = _constant_density(density[cell])
            if squared is not None:
                key = _reference_basis_key(basis, budget=budget)
                reference = reference_forms.get(key)
                if reference is None:
                    reference = reference_forms[key] = _dg_reference_forms(
                        basis, budget=budget
                    )
                root = cell_roots.get(cell)
                if root is None:
                    root = fraction_sqrt_interval(squared, budget=budget)
                published: dict[Fraction, tuple[float, float]] = {}
                moments = dict.fromkeys(reference[0])
                if is_target:
                    for row in reference[1]:
                        moments.update(dict.fromkeys(row))
                for moment in moments:
                    value, error, root = _affine_scaled_integral(
                        squared, root, moment, local_budgets, budget=budget
                    )
                    published[moment] = value, error

                for dof, moment in zip(route, reference[0], strict=True):
                    measures[dof], uncertainties[dof] = published[moment]
                if is_target:
                    mass = np.empty((len(basis), len(basis)), dtype=np.float64)
                    for i, row in enumerate(reference[1]):
                        for j, moment in enumerate(row):
                            mass[i, j], error = published[moment]
                            errors += error
                    masses.append(mass)
                cell_roots[cell] = root
            else:
                for dof, term in zip(route, basis, strict=True):
                    measures[dof], uncertainties[dof] = _density_integral(
                        density[cell], term, cell_roots, cell, local_budgets, budget=budget
                    )
                if is_target:
                    mass = np.empty((len(basis), len(basis)), dtype=np.float64)
                    for i, first in enumerate(basis):
                        for j, second in enumerate(basis):
                            mass[i, j], error = _density_integral(
                                density[cell],
                                multiply(first, second, budget=budget),
                                cell_roots,
                                cell,
                                local_budgets,
                                budget=budget,
                            )
                            errors += error
                    masses.append(mass)
    source_indices, target_indices = indices
    absolute = Fraction(local_budgets["absolute_tolerance"])
    # Allocate the mixed-form error by physical overlap content, not by a
    # uniform count of entries. Each nonnegative share is exact, sums to one,
    # and covers every entry of every piece. A small overlap must not force all
    # the large overlaps to the same unnecessarily small absolute error goal.
    shares: tuple[Fraction, ...] = ()
    if affine_mixed:
        budget.reserve(0, 256 + 128 * len(prepared.pieces))
        operands = tuple(
            (source_area, roots[0][source_indices[source_id]][1])
            for source_id, source_area in prepared.pieces._iter_kernel_source_areas(
                budget=budget
            )
        )
        bits = max(
            (
                max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                for pair in operands for value in pair
            ),
            default=1,
        )
        reserve_fraction_work(budget, len(operands), 2 * len(operands) + 1, 2 * bits + 16)
        contents = tuple(abs(area) * root for area, root in operands)
        # A product of all input denominators bounds every partial exact sum.
        # Only two accumulator objects coexist; do not multiply that large
        # bit bound by the number of contents retained alongside them.
        sum_bits = (
            sum(value.denominator.bit_length() for value in contents)
            + max((abs(value.numerator).bit_length() for value in contents), default=1)
            + len(contents).bit_length() + 16
        )
        reserve_fraction_work(budget, len(contents), 2, sum_bits)
        total = sum(contents, Fraction(0))
        if total <= 0:
            raise ValueError(
                "sphere_chart.prepare_dg.pieces must have positive total physical overlap content; got no positive source coverage."
            )
        share_bits = max(
            (
                max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                for value in contents
            ),
            default=1,
        ) + max(abs(total.numerator).bit_length(), total.denominator.bit_length()) + 16
        reserve_fraction_work(budget, len(contents), len(contents), share_bits)
        shares = tuple(content / total for content in contents)
    budget.retain_operand_metadata(1024 + 256 * (len(sc) + len(tc)))
    moment_cache: _ReciprocalMomentCache = {}
    series_cache: _ReciprocalSeriesCache = {}
    jet_cache: _ReciprocalJetCache = {}
    operand_cache: _AffineOperandCache = {}
    source_operands: dict[int, _AffineBasisOperand] = {}
    target_operands: dict[int, _AffineBasisOperand] = {}
    for piece_index, piece in enumerate(prepared.pieces._iter_kernel_pieces(budget=budget)):
        old = source_indices[int(piece.source_cell_global_id)]
        new = target_indices[int(piece.target_cell_global_id)]
        squared = _constant_density(grams[0][old])
        if (
            semantics == "conservative-density"
            and squared is not None
            and all(_degree(term) <= 1 for term in (*sc[old][2], *tc[new][2]))
        ):
            root = roots[0].get(old)
            if root is None:
                root = roots[0][old] = fraction_sqrt_interval(squared, budget=budget)
            old_operand = source_operands.get(old)
            if old_operand is None:
                old_operand = source_operands[old] = _affine_basis_operand(
                    sc[old][2], operand_cache, budget=budget
                )
            new_operand = target_operands.get(new)
            if new_operand is None:
                new_operand = target_operands[new] = _affine_basis_operand(
                    tc[new][2], operand_cache, budget=budget
                )
            with budget.temporary_scope():
                form, form_error = _affine_conservative_form(
                    piece,
                    sc[old][2],
                    tc[new][2],
                    root,
                    (
                        3 * Fraction(budgets["absolute_tolerance"]) * shares[piece_index]
                        / (4 * len(sc[old][2]) * len(tc[new][2]))
                        if affine_mixed
                        else absolute
                    ),
                    int(local_budgets["maximum_binomial_terms"]),
                    budget=budget,
                    moment_cache=moment_cache,
                    series_cache=series_cache,
                    jet_cache=jet_cache,
                    operand_cache=operand_cache,
                    basis_operands=(old_operand, new_operand),
                )
            errors += (
                0.0
                if form_error == 0
                else fraction_endpoint(form_error, upper=True, saturate=True)
            )
            previous = mixed[new].get(old)
            mixed[new][old] = form if previous is None else previous + form
            continue
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
    validation: SphereChartValidation | None = None,
) -> FiniteElementFieldTransfer:
    "scalar material pullback/projection and its algebraic transpose.\n\n H1 intensive transport applies old nodal functionals through the inverse\n projective map. DG integrates old density for conserved inventory, or\n target density including the projective Jacobian for intensive fields.\n No affine corner approximation, source-nodal density or Piola claim is used."
    semantics = parse(semantics, ChartFieldSemantics, "chart.semantics")
    if validation is None:
        validation = prepared.require_bound(
            source.mesh, source_geometry, target.mesh, target_geometry, budget=budget
        )
    indices = _validate_sphere(
        source,
        target,
        prepared,
        source_geometry,
        target_geometry,
        budget=budget,
        validation=validation,
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
                source, target, prepared, indices, field_name, budget=budget,
                validation=validation,
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


def _affine_scaled_integral(
    squared: Fraction,
    root: tuple[Fraction, Fraction],
    reference: Fraction,
    budgets: IntegrationBudgets,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float, tuple[Fraction, Fraction]]:
    """Enclose sqrt(G)*reference under the owner's per-value error contract.

    On an affine cell sqrt(G) is one constant, so a physical integral is that
    root times the exact reference integral. ``root`` encloses sqrt(G) by
    squaring; it is bisected (each step proved by squaring) only if the
    published value misses its tolerance.
    """
    if squared <= 0:
        raise ValueError(
            "sphere_chart.affine_scaled_integral.measures must satisfy the positive physical measure contract; got a nonpositive affine density."
        )
    lower, upper = root
    for _ in range(int(budgets["maximum_binomial_terms"])):
        bits = max(
            value.bit_length()
            for fraction in (reference, lower, upper)
            for value in (fraction.numerator, fraction.denominator)
        )
        # Product, half-width product and the publication residual.
        reserve_fraction_work(budget, 4, 6, 2 * bits + 64)
        central = reference * (lower + upper) / 2
        published = float(central)
        error = abs(reference) * (upper - lower) / 2 + abs(
            Fraction(published) - central
        )
        bound = 0.0 if error == 0 else fraction_endpoint(error, upper=True, saturate=True)
        allowed = Fraction(budgets["absolute_tolerance"]) + Fraction(
            budgets["relative_tolerance"]
        ) * max(abs(central) - error, Fraction(0))
        if Fraction(bound) <= allowed:
            return published, bound, (lower, upper)
        reserve_fraction_work(budget, 3, 3, 4 * bits + 64)
        midpoint = (lower + upper) / 2
        if midpoint * midpoint <= squared:
            lower = midpoint
        else:
            upper = midpoint
    raise CellGeometryTransitionError(
        "resource_limit",
        "sphere_chart.affine_scaled_integral.budget must satisfy the admitted resource limits; got Affine sphere integral exceeds its error budget.",
        measured=float(bound),
        limit=float(allowed),
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
    validation: SphereChartValidation | None = None,
) -> PreparedSurfaceChartFiniteVolumeContents:
    "True old physical contents of projective overlaps for the FV owner."
    source_indices, target_indices = _validate_sphere(
        source,
        target,
        prepared,
        source_geometry,
        target_geometry,
        budget=budget,
        validation=validation,
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
    roots: dict[int, tuple[Fraction, Fraction]] = {}
    rows, targets, values, errors = [], [], [], []
    for piece in prepared.pieces._iter_kernel_pieces(budget=budget):
        old = source_indices[int(piece.source_cell_global_id)]
        new = target_indices[int(piece.target_cell_global_id)]
        density = grams[old]
        if isinstance(density, dict) and _degree(density) == 0:
            # An affine cell has one constant physical density sqrt(G): the
            # piece content is that root times the exact reference area.
            squared = algebra.require_rational_reference_scalar(
                density.get((0, 0), Fraction(0))
            )
            root = roots.get(old)
            if root is None:
                root = fraction_sqrt_interval(squared, budget=budget)
            value, error, roots[old] = _affine_scaled_integral(
                squared,
                root,
                abs(piece.source_reference_area),
                local_budgets,
                budget=budget,
            )
        else:
            gram, _ = restricted(
                density,
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
    """Compose the two true projective quotients through one argument owner.

    The common denominator and affine argument powers are formed once, not
    independently for each target coordinate. Generic compatible integration
    still receives the exact rational charts, including at higher order.
    """
    vertices = piece.exact_source_reference_vertices
    reserve_fraction_work(
        budget, 4, 4,
        _projective_bits(tuple(value for point in vertices for value in point)),
    )
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
    composition = algebra.ExpressionComposition(source, budget=budget)
    target = tuple(
        composition(value, budget=budget) for value in projective
    )
    return source, target


def _generic_projective_condition(
    piece: PreparedSphereChartPiece, *, budget: CoordinateEnclosureBudget
) -> Fraction:
    """Bound the condition through the generic rational enclosure.

    Only a denominator without a constant term can cancel a monomial factor of
    the quotient-rule form and so change its control net; such pieces keep the
    generic enclosure instead of the direct control formula.
    """
    with budget.temporary_scope():
        source, _ = _sphere_piece_charts(piece, budget=budget)
        bounds = tuple(
            tuple(
                algebra.expression_bounds(
                    algebra.expression_compose(
                        algebra.expression_derivative(value, axis, budget=budget),
                        source,
                        budget=budget,
                    ),
                    "simplex",
                    2,
                    budget=budget,
                )
                for axis in range(2)
            )
            for value in piece.projective_map.reference_expressions()
        )
    return sum(
        (
            max(abs(Fraction(float(low))), abs(Fraction(float(high)))) ** 2
            for row in bounds
            for low, high in row
        ),
        Fraction(0),
    ) / piece.projective_bounds.jacobian_lower


def _projective_bits(values: Sequence[Fraction]) -> int:
    return 4 * max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for value in values
    ) + 16


def _projective_controls(
    piece: PreparedSphereChartPiece, *, budget: CoordinateEnclosureBudget
) -> Fraction:
    """Return the exact control-net condition of one projective piece.

    With homogeneous rows h_k and denominator D = sum h_k, the derivative of
    t_k = h_k/D along source axis j is L/D^2 with L = D*(A_kj - t_k*d_j). Its
    degree-two Bernstein control ratios are (A_kj - t_k(a)*d_j)/D(a) at the
    vertices and their cross means ((A_kj - t_k(a)*d_j)/D(b) +
    (A_kj - t_k(b)*d_j)/D(a))/2 at edge nodes; for a constant D the
    degree-one net is the vertex subset with the same maximum. These are the
    generic enclosure's controls before its outward binary64 rounding.
    """
    rows = piece.projective_map.exact_coefficients
    slope = piece.projective_map.exact_denominator
    sources = piece.exact_source_reference_vertices
    targets = piece.exact_target_reference_vertices
    reserve_fraction_work(
        budget,
        12 + 4 * 34 + 1,
        3 + 4 * 7,
        _projective_bits(
            (
                *slope,
                *(row[column] for row in rows[1:] for column in (1, 2)),
                *(value for point in (*sources, *targets) for value in point),
                piece.projective_bounds.jacobian_lower,
            )
        ),
    )
    denominators = tuple(
        slope[0] + slope[1] * point[0] + slope[2] * point[1] for point in sources
    )
    total = Fraction(0)
    for coordinate in range(2):
        for axis in range(2):
            entry = rows[coordinate + 1][axis + 1]
            rate = slope[axis + 1]
            residuals = tuple(entry - point[coordinate] * rate for point in targets)
            ratios = (
                *(
                    residual / denominator
                    for residual, denominator in zip(
                        residuals, denominators, strict=True
                    )
                ),
                *(
                    (
                        residuals[first] / denominators[second]
                        + residuals[second] / denominators[first]
                    )
                    / 2
                    for first, second in ((0, 1), (1, 2), (0, 2))
                ),
            )
            bound = max(abs(ratio) for ratio in ratios)
            total += bound * bound
    return total / piece.projective_bounds.jacobian_lower


def _sphere_projective_condition(
    pieces: SphereChartPieceBank, *, budget: CoordinateEnclosureBudget
) -> float:
    """Return the request's largest whole-piece reference-transport condition.

    Each piece's value is the sum over target coordinates k and source axes j
    of its squared Bernstein derivative bound, divided by the piece Jacobian
    lower bound, as in the generic rational enclosure but without rounding the
    controls (never larger). Target coordinates lie in [0, 1], so
    max(|A_kj|, |A_kj - d_j|)/D_lower dominates every control: pieces whose
    cheap bound cannot exceed the running exact maximum skip their controls.
    Positive denominators also certify straight native segment images.
    """
    reserve_fraction_work(budget, 2 * len(pieces), 0, 1)
    maps: dict[str, Fraction] = {}
    candidates: list[tuple[Fraction, PreparedSphereChartPiece]] = []
    largest = Fraction(0)
    for piece in pieces._iter_kernel_pieces(budget=budget):
        if piece.projective_bounds.jacobian_lower <= 0:
            raise ValueError(
                "sphere_chart.sphere_projective_condition.pieces must satisfy the material reference contract; got Sphere material transport lacks a positive whole-piece Jacobian."
            )
        if piece.projective_bounds.denominator_lower <= 0:
            raise ValueError(
                "sphere_chart.sphere_projective_condition.pieces must satisfy the material reference contract; got Sphere material transport lacks a positive whole-piece denominator."
            )
        mapping = piece.projective_map
        if mapping.exact_denominator[0] == 0:
            largest = max(largest, _generic_projective_condition(piece, budget=budget))
            continue
        summed = maps.get(mapping.map_id)
        if summed is None:
            entries = tuple(
                (mapping.exact_coefficients[coordinate + 1][axis + 1], axis)
                for coordinate in range(2)
                for axis in range(2)
            )
            # Per entry one difference, two squares and a comparison; three sums.
            reserve_fraction_work(
                budget,
                4 * 4 + 3,
                5,
                _projective_bits(
                    (*mapping.exact_denominator[1:], *(entry for entry, _ in entries))
                ),
            )
            summed = Fraction(0)
            for entry, axis in entries:
                shifted = entry - mapping.exact_denominator[axis + 1]
                summed += max(entry * entry, shifted * shifted)
            maps[mapping.map_id] = summed
        bounds = piece.projective_bounds
        reserve_fraction_work(
            budget,
            4,
            1,
            _projective_bits((summed, bounds.denominator_lower, bounds.jacobian_lower)),
        )
        candidates.append(
            (
                summed
                / (bounds.denominator_lower * bounds.denominator_lower)
                / bounds.jacobian_lower,
                piece,
            )
        )
    if candidates:
        _, first = max(candidates, key=lambda candidate: candidate[0])
        largest = max(largest, _projective_controls(first, budget=budget))
        for cheap, piece in candidates:
            if cheap > largest:
                largest = max(largest, _projective_controls(piece, budget=budget))
    condition = fraction_endpoint(largest, upper=True, saturate=True)
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "sphere_chart.sphere_projective_condition.reference_map must satisfy the regular reference-map contract; got Sphere material reference transport exceeds its whole-piece condition bound."
        )
    return condition


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
    validation: SphereChartValidation | None = None,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Transport sphere material forms through their projective pieces.\n\n A request proof shares the condition bound and the exact piece geometry\n across its compatible fields."
    indices = _validate_sphere(
        source,
        target,
        prepared,
        source_geometry,
        target_geometry,
        budget=budget,
        validation=validation,
    )

    def condition() -> float:
        return _sphere_projective_condition(prepared.pieces, budget=budget)

    def prepare_pieces(
        work: PreparationWork, charts: bool
    ) -> tuple[MaterialCompatiblePiece, ...]:
        work.condition = max(
            work.condition,
            condition()
            if validation is None
            else validation.derive("sphere-projective-condition", condition),
        )
        pieces: list[MaterialCompatiblePiece] = []
        for piece in prepared.pieces.iter_pieces(budget=work.enclosure):
            pair = None
            if charts:
                with work.enclosure.temporary_scope():
                    pair = _sphere_piece_charts(piece, budget=work.enclosure)
            pieces.append(
                MaterialCompatiblePiece(
                    indices[0][piece.source_cell_global_id],
                    indices[1][piece.target_cell_global_id],
                    piece.exact_source_reference_vertices,
                    piece.exact_target_reference_vertices,
                    # Signed areas re-proved against the vertices by require_bound.
                    piece.source_reference_area,
                    piece.target_reference_area,
                    pair,
                )
            )
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
        invariant_cache=None
        if validation is None
        else validation.derive(
            "sphere-compatible-invariant-geometry", MaterialInvariantCache
        ),
    )


__all__ = [
    "prepare_sphere_chart_field_transfer",
    "prepare_sphere_chart_finite_volume_contents",
    "prepare_sphere_chart_compatible_transfer",
]


def _refuse_foreign_validation(validation: SphereChartValidation | None, /) -> None:
    if validation is not None:
        raise ValueError(
            "sphere_chart.validation must belong to the caller's request ledger; got a proof for a request that would open its own ticket"
        )


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
    validation: SphereChartValidation | None = None,
) -> FiniteElementFieldTransfer:
    "Prepare the request with one explicitly bounded host -work ticket.\n\n ``validation`` reuses this request ledger's correspondence proof."
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
            validation=validation,
        )
    _refuse_foreign_validation(validation)
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
    validation: SphereChartValidation | None = None,
) -> PreparedSurfaceChartFiniteVolumeContents:
    "Prepare the request with one explicitly bounded host -work ticket.\n\n ``validation`` reuses this request ledger's correspondence proof."
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
            validation=validation,
        )
    _refuse_foreign_validation(validation)
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
    validation: SphereChartValidation | None = None,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Prepare the request with one explicitly bounded host -work ticket.\n\n ``validation`` reuses this request ledger's correspondence proof."
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
            validation=validation,
        )
    _refuse_foreign_validation(validation)
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
