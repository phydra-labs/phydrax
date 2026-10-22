#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"Prepare commuting forms between independent material-reference charts."

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from phydrax.discretization._sphere_chart_records import PreparedSphereChartDeformation
from phydrax.discretization._surface_chart_records import PreparedSurfaceChartDeformation
from phydrax.geometry._rational import fraction_endpoint

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._resource_ticket import HostBatchTicket
from ..._strict import StrictModule
from ...geometry._rational import rational_vector
from ...linalg import (
    _small_batched as imported_linalg__small_batched,
    ArraySpace,
    OperatorProperties,
)
from ...linalg._hermitian_spectral import reserve_fraction_work
from ...sparse import RowRelation, SparseLinearMap
from ...typing import checked
from .. import _coordinate_enclosure as algebra
from .._cell_geometry import CellGeometrySpec
from .._chart_geometry import orientation
from .._coordinate_budget import CoordinateEnclosureBudget
from .._reference_cell import reference_cell_topology
from ._exact_form_moments import (
    _intrinsic_densities,
    ExactFormMoments,
    MomentIntegralMatrix,
)
from ._form_basis_types import AbstractReferenceFormBasis
from ._form_elements import form_entity_chart, FormBasis
from ._generic import FiniteElementDiscretization
from ._mapped_form_transfer import (
    correct_commuting_moments,
    derivative_cell,
    embed_columns,
    form_cells,
    FormCell,
    PreparationWork,
    transform_integrals,
)
from ._surface_chart_transfer import deformation_binding, piece_maps, validate
from ._topology_transfer import (
    CLAIM_ULPS,
    coalesced_map,
    field_transfer_query_upper,
    FiniteElementFieldTransfer,
    FiniteElementTopologyTransfer,
    FiniteElementTransferEvidence,
    owned_rows,
)


type _Point = tuple[Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]
type _Linear = tuple[Fraction, Fraction, Fraction]
type _Charts = tuple[tuple[algebra.Expression, ...], tuple[algebra.Expression, ...]]
type _FormsKey = tuple[str, tuple[int, ...], bytes]
type _Weights = tuple[tuple[Fraction, ...], ...]


class MaterialCompatiblePiece(NamedTuple):
    """One straight material piece and, only on request, its generic charts.

    ``source_vertices`` and ``target_vertices`` are the exact images of the
    native triangle vertices and ``source_area``/``target_area`` their exact
    signed areas. The producer certifies that the target chart maps every
    native segment onto the straight segment between its vertex images with
    constant orientation: an affine chart, or a projective chart whose
    denominator is positive on the whole piece.
    """

    source_row: int
    target_row: int
    source_vertices: _Triangle
    target_vertices: _Triangle
    source_area: Fraction
    target_area: Fraction
    charts: _Charts | None


class PreparedSurfaceChartCompatibleTransfer(NamedTuple):
    field_transfer: FiniteElementFieldTransfer
    source_differential: SparseLinearMap
    target_differential: SparseLinearMap
    differential_companion: SparseLinearMap


def _material_entity_coverage(
    entity: tuple[int, ...],
    keys: Sequence[tuple[tuple[Fraction, ...], ...]],
) -> None:
    vertices = tuple(
        rational_vector(point) for point in reference_cell_topology("triangle").vertices
    )
    dimension = len(entity) - 1
    if dimension == 0:
        if set(keys) != {(vertices[entity[0]],)}:
            raise ValueError(
                "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material correspondence does not support its target vertex functional."
            )
    elif dimension == 1:
        first, last = vertices[entity[0]], vertices[entity[1]]
        axis = next(axis for axis in range(2) if first[axis] != last[axis])
        intervals = sorted(
            tuple(
                sorted(
                    (point[axis] - first[axis]) / (last[axis] - first[axis])
                    for point in key
                )
            )
            for key in keys
        )
        previous = Fraction(0)
        for lower, upper in intervals:
            if lower != previous or upper <= lower:
                raise ValueError(
                    "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces leave a gap or overlapping target edge functional."
                )
            previous = upper
        if previous != 1:
            raise ValueError(
                "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces do not cover the entire target edge functional."
            )
    elif dimension == 2:
        area = sum(
            (
                abs(
                    orientation(
                        (key[0][0], key[0][1]),
                        (key[1][0], key[1][1]),
                        (key[2][0], key[2][1]),
                    )
                )
                / 2
                for key in keys
            ),
            Fraction(0),
        )
        if area != Fraction(1, 2):
            raise ValueError(
                "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces do not cover the entire target cell functional."
            )
    else:
        raise ValueError(
            "surface_chart.material_entity_coverage.binding must satisfy the declared geometry and cell identities; got Material forms require triangular reference entities."
        )


_LINEAR_INDICES = ((0, 0), (1, 0), (0, 1))
_ORIENTATION_REFUSAL = "common_entity_map.configuration must have a nonsingular chart with resolved orientation; got A common form entity has unresolved orientation or a singular rational chart."
_TWIST_REFUSAL = "full_chart_orientation.configuration must have a nonsingular chart with resolved orientation; got A twisted common chart has unresolved full-cell orientation or a singular Jacobian."


@dataclass(eq=False)
class MaterialInvariantCache:
    """Exact straight-piece integrals reused by one request's compatible fields.

    Entries are keyed by the exact data that determine them (vertex images,
    route-form weights, test entities), so reuse never changes a value; each
    entry is charged to the owning request ledger when first computed.
    """

    incidences: dict[tuple[_Point, _Linear], bool] = field(default_factory=dict)
    segments: dict[tuple[_Point, _Point], dict[int, Fraction]] = field(
        default_factory=dict
    )
    faces: dict[tuple[_Triangle, Fraction], dict[int, Fraction]] = field(
        default_factory=dict
    )
    forms: dict[_FormsKey, _Weights] = field(default_factory=dict)
    positions: dict[_Weights, tuple[int, ...]] = field(default_factory=dict)
    integrals: dict[tuple[_Weights, _Point, _Point], tuple[Fraction, ...]] = field(
        default_factory=dict
    )
    face_integrals: dict[
        tuple[_Weights, _Triangle, Fraction], tuple[Fraction, ...]
    ] = field(default_factory=dict)
    images: dict[Hashable, _InvariantImageGeometry | None] = field(
        default_factory=dict
    )
    edge_coverage: set[
        tuple[tuple[Fraction, Fraction], tuple[tuple[Fraction, Fraction], ...]]
    ] = field(default_factory=set)
    moments: dict[Hashable, tuple[MomentIntegralMatrix, float]] = field(
        default_factory=dict
    )
    twists: dict[tuple[Fraction, Fraction], int] = field(default_factory=dict)
    covered: set[frozenset[frozenset[_Point]]] = field(default_factory=set)


class _InvariantEntity(NamedTuple):
    """A target DoF entity whose dimension equals the form degree.

    ``rows`` pair each target DoF row with its constant intrinsic density.
    Edges carry the exact line ``a*x + b*y == r`` of their declared chart, its
    pivot coordinate, pivot sign and the ordered pivot span of the entity; the
    cell carries its chart-determinant sign.
    """

    entity: tuple[int, ...]
    rows: tuple[tuple[int, Fraction], ...]
    line: _Linear | None
    pivot: int
    orientation: int
    span: tuple[Fraction, Fraction] | None


@dataclass(eq=False)
class _InvariantPlan:
    targets: dict[str, tuple[_InvariantEntity, ...]]
    sources: dict[str, tuple[tuple[_Linear, ...], ...]]
    cache: MaterialInvariantCache


class _InvariantContribution(NamedTuple):
    rows: tuple[tuple[int, Fraction], ...]
    sign: int
    routes: NDArray[np.int64]
    values: tuple[Fraction, ...]


class _InvariantImageGeometry(NamedTuple):
    """An admitted oriented image, independent of routes, densities and twist."""

    key: frozenset[_Point]
    sign: int
    interval: tuple[Fraction, Fraction] | None
    start: _Point
    end: _Point


def _fraction_bits(values: Iterable[Fraction]) -> int:
    """Bound combined numerator/denominator growth through exact operations.

    Different rational operands need not share a denominator. Their bit sizes
    add under products and common-denominator sums, rather than taking only
    the largest operand's size.
    """
    return max(
        sum(
            max(abs(value.numerator).bit_length(), value.denominator.bit_length(), 1)
            for value in values
        ),
        1,
    )


def _reserve_exact(
    work: PreparationWork,
    operations: int,
    count: int,
    operands: Iterable[Fraction],
) -> None:
    """Admit exact term visits and the combined operands' temporary storage."""
    reserve_fraction_work(
        work.enclosure, operations, count, 2 * _fraction_bits(operands) + 8
    )


def _dot_operations(weights: tuple[Fraction, ...]) -> int:
    """Count the products, sums and leading negation that ``_dot`` performs."""
    operations, first = 0, True
    for weight in weights:
        if not weight:
            continue
        operations += (weight not in (1, -1)) + (not first or weight == -1)
        first = False
    return operations


def _dot(weights: tuple[Fraction, ...], values: dict[int, Fraction]) -> Fraction:
    total: Fraction | None = None
    for index, weight in enumerate(weights):
        if not weight:
            continue
        value = values[index]
        if total is None:
            total = value if weight == 1 else -value if weight == -1 else weight * value
        elif weight == 1:
            total += value
        elif weight == -1:
            total -= value
        else:
            total += weight * value
    return Fraction(0) if total is None else total


def _linear_components(
    exact: ExactFormMoments, basis: AbstractReferenceFormBasis
) -> tuple[tuple[_Linear, ...], ...] | None:
    """Return exact affine reference components, or None for any other kind."""
    rows: list[tuple[_Linear, ...]] = []
    for row in exact.components(basis):
        linear: list[_Linear] = []
        for value in row:
            if not isinstance(value, dict) or any(
                index not in _LINEAR_INDICES for index in value
            ):
                return None
            coefficients: list[Fraction] = []
            for index in _LINEAR_INDICES:
                item = value.get(index, Fraction(0))
                if not isinstance(item, (Fraction, int)):
                    return None
                coefficients.append(Fraction(item))
            linear.append((coefficients[0], coefficients[1], coefficients[2]))
        rows.append(tuple(linear))
    return tuple(rows)


def _invariant_target(
    target: AbstractReferenceFormBasis,
    exact: ExactFormMoments,
    work: PreparationWork,
) -> tuple[_InvariantEntity, ...] | None:
    """Admit tests whose moments are parametrization-invariant line/area forms.

    Every target DoF must sit on an entity of dimension equal to the form
    degree with a constant intrinsic density, so its test is a constant
    zero-form and the moment is the oriented integral of the source form.
    """
    if target.dimension != 2 or target.form_degree not in (1, 2):
        return None
    entities: list[_InvariantEntity] = []
    for dimension, group in enumerate(target.entity_vertices):
        for entity in group:
            rows = tuple(
                row
                for row, label in enumerate(target.dof_labels)
                if label[0] == entity
            )
            if not rows:
                continue
            if dimension != target.form_degree:
                return None
            _, origin, chart = form_entity_chart(target.dimension, target.family, entity)
            if chart.shape != (2, dimension):
                return None
            densities = _intrinsic_densities(
                exact.densities(target, entity),
                chart,
                target.form_degree,
                dimension,
                budget=work.enclosure,
            )
            constants: list[tuple[int, Fraction]] = []
            for row in rows:
                density = densities[row]
                if (
                    len(density) != 1
                    or not isinstance(density[0], dict)
                    or any(any(index) for index in density[0])
                ):
                    return None
                value = density[0].get((0,) * dimension, 0)
                if not isinstance(value, (Fraction, int)):
                    return None
                constants.append((row, Fraction(value)))
            point = (Fraction(float(origin[0])), Fraction(float(origin[1])))
            frame = tuple(
                tuple(Fraction(float(chart[axis, column])) for column in range(dimension))
                for axis in range(2)
            )
            _reserve_exact(
                work, 11, 5, (*point, *(value for row in frame for value in row))
            )
            span: tuple[Fraction, Fraction] | None = None
            if dimension == 1:
                pivot = 0 if frame[0][0] else 1
                if not frame[pivot][0]:
                    return None
                # Normalize the leading coefficient of a*x + b*y == r to one.
                first, second = frame[1][0], -frame[0][0]
                lead = first if first else second
                line = (
                    first / lead,
                    second / lead,
                    (first * point[0] + second * point[1]) / lead,
                )
                orientation = 1 if frame[pivot][0] > 0 else -1
                end = point[pivot] + frame[pivot][0]
                span = (point[pivot], end) if orientation > 0 else (end, point[pivot])
            else:
                determinant = frame[0][0] * frame[1][1] - frame[0][1] * frame[1][0]
                if not determinant:
                    return None
                line, pivot, orientation = None, 0, 1 if determinant > 0 else -1
            entities.append(
                _InvariantEntity(
                    entity, tuple(constants), line, pivot, orientation, span
                )
            )
    return tuple(entities)


def _invariant_plan(
    sources: tuple[FormCell, ...],
    derivative_sources: Sequence[FormCell],
    targets: tuple[FormCell, ...],
    derivative_cache: dict[int, tuple[FormBasis, NDArray[np.float64]]],
    work: PreparationWork,
    exact: ExactFormMoments,
    budget: CoordinateEnclosureBudget,
    cache: MaterialInvariantCache,
) -> _InvariantPlan | None:
    """Decide once per request whether every moment is an invariant integral."""
    tables: dict[str, tuple[tuple[_Linear, ...], ...]] = {}
    twists = {cell.basis.twist for cell in sources + targets}
    for degree, cells in ((1, sources), (2, tuple(derivative_sources))):
        for cell in cells:
            basis = cell.basis
            if basis.basis_id in tables:
                continue
            if basis.dimension != 2 or basis.form_degree != degree:
                return None
            table = _linear_components(exact, basis)
            if table is None:
                return None
            tables[basis.basis_id] = table
    entities: dict[str, tuple[_InvariantEntity, ...]] = {}
    for cell in targets:
        if cell.basis.basis_id in entities:
            continue
        following = derivative_cell(
            FormCell(
                cell.basis,
                cell.routes,
                np.eye(cell.basis.local_dof_count, dtype=np.float64),
            ),
            derivative_cache,
            budget=budget,
        ).basis
        for degree, basis in ((1, cell.basis), (2, following)):
            if basis.basis_id in entities:
                continue
            tests = _invariant_target(basis, exact, work)
            if tests is None or basis.form_degree != degree:
                return None
            twists.add(basis.twist)
            entities[basis.basis_id] = tests
    if len(twists) != 1:
        return None
    return _InvariantPlan(entities, tables, cache)


def _route_weights(
    cell: FormCell,
    table: tuple[tuple[_Linear, ...], ...],
    work: PreparationWork,
    cache: MaterialInvariantCache,
) -> _Weights:
    """Fold the binary64 cell transform into exact route-form weights.

    One-form routes weigh (dx, dy, d(x^2), d(y^2), d(xy), x dy - y dx) over a
    segment; two-form routes weigh (A, A*sum x, A*sum y) over a triangle with
    signed area A. Both rules are exact for affine components.
    """
    key = (cell.basis.basis_id, cell.transform.shape, cell.transform.tobytes())
    weights = cache.forms.get(key)
    if weights is not None:
        return weights
    transform = tuple(
        tuple(Fraction(float(value)) for value in row) for row in cell.transform
    )
    count, routes = cell.transform.shape
    components = len(table[0])
    _reserve_exact(
        work,
        2 * count * routes * components * 3 + 6 * routes,
        routes * (components * 3 + 6),
        (
            *(value for row in transform for value in row),
            *(value for row in table for linear in row for value in linear),
        ),
    )
    folded = tuple(
        tuple(
            tuple(
                sum(
                    (
                        transform[row][route] * table[row][component][monomial]
                        for row in range(count)
                        if transform[row][route] and table[row][component][monomial]
                    ),
                    Fraction(0),
                )
                for monomial in range(3)
            )
            for component in range(components)
        )
        for route in range(routes)
    )
    if components == 2:
        weights = tuple(
            (
                first[0],
                second[0],
                first[1] / 2,
                second[2] / 2,
                (first[2] + second[1]) / 2,
                (second[1] - first[2]) / 2,
            )
            for first, second in folded
        )
    else:
        weights = tuple(
            (linear[0], linear[1] / 3, linear[2] / 3) for (linear,) in folded
        )
    cache.forms[key] = weights
    cache.positions[weights] = tuple(
        index
        for index in range(len(weights[0]))
        if any(route[index] for route in weights)
    )
    return weights


def _on_line(
    point: _Point, line: _Linear, work: PreparationWork, cache: MaterialInvariantCache
) -> bool:
    key = (point, line)
    incident = cache.incidences.get(key)
    if incident is None:
        _reserve_exact(work, _dot_operations(line[:2]) + 1, 1, (*point, *line))
        incident = _dot(line[:2], {0: point[0], 1: point[1]}) == line[2]
        cache.incidences[key] = incident
    return incident


def _segment_geometry(
    first: _Point,
    second: _Point,
    needed: tuple[int, ...],
    work: PreparationWork,
    cache: MaterialInvariantCache,
) -> dict[int, Fraction]:
    entries = cache.segments.setdefault((first, second), {})
    missing = tuple(index for index in needed if index not in entries)
    if missing:
        _reserve_exact(
            work,
            sum(1 if index < 2 else 3 for index in missing),
            len(missing),
            (*first, *second),
        )
    for index in missing:
        if index < 2:
            entries[index] = second[index] - first[index]
        elif index < 4:
            axis = index - 2
            entries[index] = (second[axis] - first[axis]) * (second[axis] + first[axis])
        elif index == 4:
            entries[index] = second[0] * second[1] - first[0] * first[1]
        else:
            entries[index] = first[0] * second[1] - first[1] * second[0]
    return entries


def _face_geometry(
    triangle: _Triangle,
    area: Fraction,
    needed: tuple[int, ...],
    work: PreparationWork,
    cache: MaterialInvariantCache,
) -> dict[int, Fraction]:
    entries = cache.faces.setdefault((triangle, area), {0: area})
    missing = tuple(index for index in needed if index not in entries)
    if missing:
        _reserve_exact(
            work,
            3 * len(missing),
            len(missing),
            (area, *(value for point in triangle for value in point)),
        )
    for index in missing:
        axis = index - 1
        entries[index] = area * (triangle[0][axis] + triangle[1][axis] + triangle[2][axis])
    return entries


def _route_integrals(
    weights: _Weights,
    geometry: dict[int, Fraction],
    work: PreparationWork,
) -> tuple[Fraction, ...]:
    _reserve_exact(
        work,
        sum(_dot_operations(route) for route in weights),
        len(weights),
        (*geometry.values(), *(value for route in weights for value in route)),
    )
    return tuple(_dot(route, geometry) for route in weights)


def _invariant_image_geometry(
    piece: MaterialCompatiblePiece,
    native: tuple[int, ...],
    entity: _InvariantEntity,
    work: PreparationWork,
    cache: MaterialInvariantCache,
) -> _InvariantImageGeometry | None:
    """Prove image incidence and orientation once for the exact geometry.

    The test rows and bundle twist do not enter this proof. In particular,
    exterior-derivative and companion moments share the same area admission,
    and curl/flux share line admission even when their route tables differ.
    """
    geometry_key = (
        piece.source_vertices,
        piece.target_vertices,
        piece.target_area,
        native,
        entity.line,
        entity.pivot,
        entity.orientation,
    )
    if geometry_key in cache.images:
        return cache.images[geometry_key]
    images = piece.target_vertices
    if entity.line is not None:
        first, second = images[native[0]], images[native[1]]
        if not (
            _on_line(first, entity.line, work, cache)
            and _on_line(second, entity.line, work, cache)
        ):
            cache.images[geometry_key] = None
            return None
        start = piece.source_vertices[native[0]]
        end = piece.source_vertices[native[1]]
        _reserve_exact(work, 3, 0, (*first, *second, *start, *end))
        if first[entity.pivot] == second[entity.pivot]:
            raise ValueError(_ORIENTATION_REFUSAL)
        sign = entity.orientation
        interval = (first[entity.pivot], second[entity.pivot])
        if second[entity.pivot] < first[entity.pivot]:
            sign = -sign
            interval = (second[entity.pivot], first[entity.pivot])
        if end < start:
            start, end, sign = end, start, -sign
        image = _InvariantImageGeometry(
            frozenset((first, second)), sign, interval, start, end
        )
    else:
        _reserve_exact(work, 1, 0, (piece.target_area,))
        if not piece.target_area:
            raise ValueError(_ORIENTATION_REFUSAL)
        sign = entity.orientation if piece.target_area > 0 else -entity.orientation
        image = _InvariantImageGeometry(
            frozenset(images),
            sign,
            None,
            piece.source_vertices[0],
            piece.source_vertices[0],
        )
    cache.images[geometry_key] = image
    return image


def _invariant_image(
    cell: FormCell,
    weights: _Weights,
    piece: MaterialCompatiblePiece,
    twist: int,
    native: tuple[int, ...],
    entity: _InvariantEntity,
    work: PreparationWork,
    cache: MaterialInvariantCache,
) -> (
    tuple[
        frozenset[_Point], _InvariantContribution, tuple[Fraction, Fraction] | None
    ]
    | None
):
    """Integrate one native entity exactly from its straight vertex images.

    The key is the set of target vertex images: equal exactly when the sorted
    image tuples keyed by the generic common-entity integral are equal. Edges
    also return their ordered pivot interval for the coverage check.
    """
    geometry = _invariant_image_geometry(piece, native, entity, work, cache)
    if geometry is None:
        return None
    needed = cache.positions[weights]
    if geometry.interval is not None:
        segment = (weights, geometry.start, geometry.end)
        values = cache.integrals.get(segment)
        if values is None:
            values = _route_integrals(
                weights,
                _segment_geometry(geometry.start, geometry.end, needed, work, cache),
                work,
            )
            cache.integrals[segment] = values
    else:
        face = (weights, piece.source_vertices, piece.source_area)
        values = cache.face_integrals.get(face)
        if values is None:
            values = _route_integrals(
                weights,
                _face_geometry(
                    piece.source_vertices, piece.source_area, needed, work, cache
                ),
                work,
            )
            cache.face_integrals[face] = values
    return (
        geometry.key,
        _InvariantContribution(entity.rows, geometry.sign * twist, cell.routes, values),
        geometry.interval,
    )


def _accumulate(
    totals: dict[tuple[int, int], Fraction],
    contribution: _InvariantContribution,
    work: PreparationWork,
) -> None:
    """Add one contribution exactly, charging each product, sum and negation."""
    routes = contribution.routes.tolist()
    factors = tuple(
        (row, density if contribution.sign > 0 else -density)
        for row, density in contribution.rows
    )
    operations = sum(
        contribution.sign < 0 and density not in (1, -1)
        for _, density in contribution.rows
    )
    for row, factor in factors:
        for route, value in zip(routes, contribution.values, strict=True):
            if value:
                operations += (factor not in (1, -1)) + (
                    (row, route) in totals or factor == -1
                )
    _reserve_exact(
        work,
        operations,
        len(factors) * len(routes),
        (
            *contribution.values,
            *(factor for _, factor in factors),
            *(
                totals[row, route]
                for row, _ in factors
                for route in routes
                if (row, route) in totals
            ),
        ),
    )
    for row, factor in factors:
        for route, value in zip(routes, contribution.values, strict=True):
            if not value:
                continue
            entry = (row, route)
            previous = totals.get(entry)
            if factor == 1:
                totals[entry] = value if previous is None else previous + value
            elif factor == -1:
                totals[entry] = -value if previous is None else previous - value
            else:
                term = factor * value
                totals[entry] = term if previous is None else previous + term


def _contribution_entries(
    contribution: _InvariantContribution, work: PreparationWork
) -> dict[tuple[int, int], Fraction]:
    totals: dict[tuple[int, int], Fraction] = {}
    _accumulate(totals, contribution, work)
    return totals


def _invariant_defect(
    previous: _InvariantContribution,
    current: _InvariantContribution,
    work: PreparationWork,
) -> float:
    """Compare two contributions to one target sub-entity exactly."""
    first = _contribution_entries(previous, work)
    second = _contribution_entries(current, work)
    keys = set(first) | set(second)
    _reserve_exact(
        work,
        2 * len(keys) + len(first) + len(second),
        len(keys),
        (*first.values(), *second.values()),
    )
    difference = max(
        (abs(first.get(key, Fraction(0)) - second.get(key, Fraction(0))) for key in keys),
        default=Fraction(0),
    )
    scale_ = max(
        float(max((abs(value) for value in first.values()), default=Fraction(0))),
        float(max((abs(value) for value in second.values()), default=Fraction(0))),
        1.0,
    )
    return fraction_endpoint(difference, upper=True, saturate=True) / scale_


def _invariant_edge_coverage(
    span: tuple[Fraction, Fraction],
    intervals: list[tuple[Fraction, Fraction]],
    work: PreparationWork,
) -> None:
    """Require the pivot intervals of distinct images to tile the target edge.

    The edge chart is affine with a nonzero pivot slope, so tiling in pivot
    coordinates is tiling of the generic chart parameter on [0, 1].
    """
    count = len(intervals)
    # A run scan plus binary insertion compares at most (n - 1) * (bits + 1).
    _reserve_exact(
        work,
        max(count - 1, 0) * ((count - 1).bit_length() + 1) + count + 1,
        0,
        (value for interval in intervals for value in interval),
    )
    previous = span[0]
    for lower, upper in sorted(intervals, key=lambda interval: interval[0]):
        if lower != previous:
            raise ValueError(
                "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces leave a gap or overlapping target edge functional."
            )
        previous = upper
    if previous != span[1]:
        raise ValueError(
            "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces do not cover the entire target edge functional."
        )


def _invariant_common_moments(
    sources: tuple[FormCell, ...],
    target: AbstractReferenceFormBasis,
    pieces: tuple[MaterialCompatiblePiece, ...],
    columns: NDArray[np.int64],
    work: PreparationWork,
    plan: _InvariantPlan,
) -> tuple[MomentIntegralMatrix, float]:
    """Accumulate the generic common-entity moments from exact vertex data.

    A constant zero-form test on an entity of the form's dimension makes each
    moment the oriented integral of the source form over the straight image
    of the native entity, independent of its parametrization. Sums stay exact
    and are rounded once per published entry, so no summation term enters the
    published error. The result is memoized by every exact datum it depends
    on, so fields of one request with identical data share it.
    """
    cache = plan.cache
    entities = plan.targets[target.basis_id]
    weights = tuple(
        _route_weights(
            sources[piece.source_row],
            plan.sources[sources[piece.source_row].basis.basis_id],
            work,
            cache,
        )
        for piece in pieces
    )
    twists = [1] * len(pieces)
    if target.twist == "twisted":
        # The twisted bundle multiplies by both full-cell orientation signs.
        for index, piece in enumerate(pieces):
            areas = (piece.source_area, piece.target_area)
            twist = cache.twists.get(areas)
            if twist is None:
                _reserve_exact(work, 2, 0, areas)
                if not piece.source_area or not piece.target_area:
                    raise ValueError(_TWIST_REFUSAL)
                twist = 1 if (piece.source_area > 0) == (piece.target_area > 0) else -1
                cache.twists[areas] = twist
            twists[index] = twist
    key = (
        entities,
        target.local_dof_count,
        columns.tobytes(),
        tuple(
            (
                piece.source_vertices,
                piece.target_vertices,
                piece.source_area,
                piece.target_area,
                sources[piece.source_row].routes.tobytes(),
                route_weights,
                twist,
            )
            for piece, route_weights, twist in zip(pieces, weights, twists, strict=True)
        ),
    )
    memo = cache.moments.get(key)
    if memo is not None:
        return memo
    work.charge(0, 2 * target.local_dof_count * columns.size)
    totals: dict[tuple[int, int], Fraction] = {}
    continuity = 0.0
    edges = reference_cell_topology("triangle").entities[1]
    for entity in entities:
        seen: dict[frozenset[_Point], _InvariantContribution] = {}
        intervals: list[tuple[Fraction, Fraction]] = []
        areas: list[Fraction] = []
        for piece, route_weights, twist in zip(pieces, weights, twists, strict=True):
            cell = sources[piece.source_row]
            for native in edges if entity.line is not None else ((0, 1, 2),):
                image = _invariant_image(
                    cell, route_weights, piece, twist, native, entity, work, cache
                )
                if image is None:
                    continue
                image_key, contribution, interval = image
                previous = seen.get(image_key)
                if previous is not None:
                    continuity = max(
                        continuity, _invariant_defect(previous, contribution, work)
                    )
                    continue
                seen[image_key] = contribution
                if interval is not None:
                    intervals.append(interval)
                else:
                    areas.append(piece.target_area)
                _accumulate(totals, contribution, work)
        if entity.span is not None:
            coverage = (entity.span, tuple(intervals))
            if coverage not in cache.edge_coverage:
                _invariant_edge_coverage(entity.span, intervals, work)
                cache.edge_coverage.add(coverage)
            continue
        images = frozenset(seen)
        if images in cache.covered:
            continue
        _reserve_exact(work, 2 * len(areas) + 1, 1, areas)
        if sum((abs(area) for area in areas), Fraction(0)) != Fraction(1, 2):
            raise ValueError(
                "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces do not cover the entire target cell functional."
            )
        cache.covered.add(images)
    values = np.zeros((target.local_dof_count, columns.size), dtype=np.float64)
    errors = np.zeros_like(values)
    if totals:
        # One binary64 conversion, magnitude and spacing per published entry.
        work.charge(3 * len(totals), 3 * len(totals))
        rows = np.fromiter((entry[0] for entry in totals), dtype=np.int64)
        positions = np.searchsorted(
            columns, np.fromiter((entry[1] for entry in totals), dtype=np.int64)
        )
        published = np.fromiter(
            (float(value) for value in totals.values()), dtype=np.float64
        )
        values[rows, positions] = published
        # Fraction-to-binary64 conversion rounds correctly: within half an ulp.
        errors[rows, positions] = np.spacing(np.abs(published))
    values.flags.writeable = False
    errors.flags.writeable = False
    result = MomentIntegralMatrix(values, errors), continuity
    cache.moments[key] = result
    return result


def _generic_common_moments(
    sources: tuple[FormCell, ...],
    target: AbstractReferenceFormBasis,
    pieces: tuple[MaterialCompatiblePiece, ...],
    columns: NDArray[np.int64],
    work: PreparationWork,
    exact: ExactFormMoments,
) -> tuple[MomentIntegralMatrix, float]:
    work.charge(0, 2 * target.local_dof_count * columns.size)
    values = np.zeros((target.local_dof_count, columns.size), dtype=np.float64)
    errors = np.zeros_like(values)
    continuity = 0.0
    topology = reference_cell_topology("triangle")
    for dimension, entities in enumerate(target.entity_vertices):
        for entity in entities:
            if not any(label[0] == entity for label in target.dof_labels):
                continue
            seen: dict[tuple[tuple[Fraction, ...], ...], MomentIntegralMatrix] = {}
            for piece in pieces:
                source = sources[piece.source_row]
                if piece.charts is None:
                    raise ValueError(
                        "surface_chart.material_common_moments.pieces must satisfy the material reference contract; got Material pieces omit their requested generic charts."
                    )
                source_arguments, target_arguments = piece.charts
                for native_entity in topology.entities[dimension]:
                    work.charge(
                        target.local_dof_count
                        * source.basis.local_dof_count
                        * source.routes.size,
                        2
                        * target.local_dof_count
                        * (source.basis.local_dof_count + source.routes.size),
                    )
                    with (
                        work.enclosure.temporary_scope(),
                        work.enclosure.polynomial_preparation_scope(),
                    ):
                        result = exact.common_entity(
                            source.basis,
                            target,
                            entity,
                            "triangle",
                            native_entity,
                            source_arguments,
                            target_arguments,
                        )
                        if result is None:
                            continue
                        raw, key = result
                        transformed = transform_integrals(raw, source.transform)
                        contribution = MomentIntegralMatrix(
                            embed_columns(
                                transformed.value, source.routes, columns, work
                            ),
                            embed_columns(
                                transformed.error, source.routes, columns, work
                            ),
                        )
                    previous = seen.get(key)
                    if previous is not None:
                        scale_ = max(
                            float(np.max(np.abs(previous.value), initial=0)),
                            float(np.max(np.abs(contribution.value), initial=0)),
                            1.0,
                        )
                        continuity = max(
                            continuity,
                            float(
                                np.max(
                                    np.abs(previous.value - contribution.value)
                                    + previous.error
                                    + contribution.error,
                                    initial=0,
                                )
                            )
                            / scale_,
                        )
                    else:
                        work.enclosure.retain_basis((key,))
                        values += contribution.value
                        errors += contribution.error
                        seen[key] = contribution
            _material_entity_coverage(entity, tuple(seen))
    gamma = np.finfo(np.float64).eps * max(len(pieces), 1)
    return MomentIntegralMatrix(
        values, np.nextafter(errors + gamma * np.abs(values) / (1 - gamma), np.inf)
    ), continuity


def _material_common_moments(
    sources: tuple[FormCell, ...],
    target: AbstractReferenceFormBasis,
    pieces: tuple[MaterialCompatiblePiece, ...],
    columns: NDArray[np.int64],
    work: PreparationWork,
    exact: ExactFormMoments,
    plan: _InvariantPlan | None,
) -> tuple[MomentIntegralMatrix, float]:
    if plan is None:
        return _generic_common_moments(sources, target, pieces, columns, work, exact)
    return _invariant_common_moments(sources, target, pieces, columns, work, plan)


def _material_sparse_blocks(
    blocks: Sequence[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]],
    target_size: int,
    source_size: int,
    name: str,
) -> SparseLinearMap:
    rows, columns, values = [], [], []
    for targets, sources, coefficients in blocks:
        rows.append(np.broadcast_to(targets[:, None], coefficients.shape).reshape((-1,)))
        columns.append(np.broadcast_to(sources[None], coefficients.shape).reshape((-1,)))
        values.append(coefficients.reshape((-1,)))
    return coalesced_map(
        np.concatenate(rows),
        np.concatenate(columns),
        np.concatenate(values),
        target_size=target_size,
        source_size=source_size,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint(
            {
                "kind": name,
                "rows": array_tree_fingerprint(tuple(rows)),
                "columns": array_tree_fingerprint(tuple(columns)),
                "values": array_tree_fingerprint(tuple(values)),
            }
        ),
    )


class _MaterialSourceDifferentials(StrictModule):
    old_offset: int


def _prepare_material_source_differentials(
    sources: tuple[FormCell, ...],
    derivative_cache: dict[int, tuple[FormBasis, NDArray[np.float64]]],
    work: PreparationWork,
    budget: CoordinateEnclosureBudget,
    source_differentials: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
    ],
    companion_owners: dict[int, tuple[int, int]],
    derivative_sources: list[FormCell],
    companion_sources: list[FormCell],
    old_offset: int,
) -> _MaterialSourceDifferentials:
    "Prepare source exterior derivatives and companion coefficient routes."
    for cell in sources:
        work.observe_basis(cell.basis)
        following = derivative_cell(cell, derivative_cache, budget=budget)
        work.observe_basis(following.basis)
        routes = np.arange(
            old_offset, old_offset + following.basis.local_dof_count, dtype=np.int64
        )
        work.charge(
            following.transform.size,
            routes.size + following.basis.local_dof_count**2 + following.transform.size,
        )
        source_differentials.append((routes, cell.routes, following.transform))
        companion_owners.update(
            (int(identifier), (len(source_differentials) - 1, index))
            for index, identifier in enumerate(routes)
        )
        derivative_sources.append(following)
        companion_sources.append(
            FormCell(
                following.basis,
                routes,
                np.eye(following.basis.local_dof_count, dtype=np.float64),
            )
        )
        old_offset += routes.size
    return _MaterialSourceDifferentials(old_offset=old_offset)


class _MaterialTargetMoments(StrictModule):
    new_offset: int
    continuity: float


def _signed_permutation_solve(
    transform: NDArray[np.float64],
    moments: NDArray[np.float64],
    work: PreparationWork,
    permutations: dict[
        tuple[tuple[int, ...], bytes],
        tuple[NDArray[np.int64], NDArray[np.bool_]] | None,
    ],
) -> NDArray[np.float64] | None:
    """Solve an orientation transform exactly when it is a signed permutation.

    Its inverse is its transpose, so each coefficient is one signed moment and
    the solve residual is exactly zero; other transforms return None. Each
    distinct transform is classified once per request.
    """
    key = (transform.shape, transform.tobytes())
    if key not in permutations:
        work.charge(transform.size)
        nonzero = transform != 0
        permutations[key] = None
        if (
            transform.ndim == 2
            and transform.shape[0] == transform.shape[1]
            and np.all(np.abs(transform[nonzero]) == 1)
            and np.all(np.count_nonzero(nonzero, axis=0) == 1)
            and np.all(np.count_nonzero(nonzero, axis=1) == 1)
        ):
            rows = np.argmax(nonzero, axis=0)
            permutations[key] = (
                rows,
                transform[rows, np.arange(transform.shape[1])] < 0,
            )
    permutation = permutations[key]
    if permutation is None:
        return None
    rows, negative = permutation
    # Rows only reorder; rows with a negative sign perform one negation each.
    work.charge(int(np.count_nonzero(negative)) * moments.shape[1], moments.size)
    local = moments[rows]
    local[negative] *= -1.0
    return local


def _prepare_material_target_moments(
    sources: tuple[FormCell, ...],
    targets: tuple[FormCell, ...],
    charts: list[list[MaterialCompatiblePiece]],
    derivative_cache: dict[int, tuple[FormBasis, NDArray[np.float64]]],
    work: PreparationWork,
    exact: ExactFormMoments,
    plan: _InvariantPlan | None,
    budget: CoordinateEnclosureBudget,
    derivative_sources: list[FormCell],
    companion_sources: list[FormCell],
    candidate_rows: list[int],
    candidate_columns: list[NDArray[np.int64]],
    candidate_values: list[NDArray[np.float64]],
    target_differentials: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
    ],
    companion_blocks: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
    ],
    published_checks: list[
        tuple[
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            MomentIntegralMatrix,
            NDArray[np.int64],
            MomentIntegralMatrix,
        ]
    ],
    new_offset: int,
    continuity: float,
) -> _MaterialTargetMoments:
    "Integrate target moments and assemble their differential companion."
    permutations: dict[
        tuple[tuple[int, ...], bytes],
        tuple[NDArray[np.int64], NDArray[np.bool_]] | None,
    ] = {}
    for row, cell in enumerate(targets):
        work.observe_basis(cell.basis)
        pieces = tuple(charts[row])
        if not pieces:
            raise ValueError(
                "surface_chart.prepare_material_target_moments.pieces must satisfy the material reference contract; got Material correspondence leaves a target compatible cell unsupported."
            )
        count = sum(sources[piece.source_row].routes.size for piece in pieces)
        work.charge(count, count)
        columns = np.unique(
            np.concatenate([sources[piece.source_row].routes for piece in pieces])
        )
        raw, defect = _material_common_moments(
            sources, cell.basis, pieces, columns, work, exact, plan
        )
        continuity = max(continuity, defect)
        following = derivative_cell(
            FormCell(
                cell.basis,
                cell.routes,
                np.eye(cell.basis.local_dof_count, dtype=np.float64),
            ),
            derivative_cache,
            budget=budget,
        )
        work.observe_basis(following.basis)
        expected, defect = _material_common_moments(
            tuple(derivative_sources),
            following.basis,
            pieces,
            columns,
            work,
            exact,
            plan,
        )
        continuity = max(continuity, defect)
        corrected = correct_commuting_moments(
            cell.basis, raw, expected, following.transform, columns, work
        )
        local = _signed_permutation_solve(
            cell.transform, corrected.value, work, permutations
        )
        if local is None:
            local = work.orientation_solve(cell.transform, corrected.value)
        spectrum = np.asarray(work.factor_matrix(cell.transform).singular_values())
        work.integration_error = max(
            work.integration_error,
            float(np.max(corrected.error, initial=0))
            * np.sqrt(cell.basis.local_dof_count)
            / float(spectrum[-1]),
        )
        candidate_rows.extend(cell.routes.tolist())
        candidate_columns.extend(np.broadcast_to(columns, local.shape))
        candidate_values.extend(local)
        new_routes = np.arange(
            new_offset, new_offset + following.basis.local_dof_count, dtype=np.int64
        )
        work.charge(
            following.transform.size * cell.transform.shape[1],
            following.transform.shape[0] * cell.transform.shape[1] + new_routes.size,
        )
        differential = following.transform @ cell.transform
        target_differentials.append((new_routes, cell.routes, differential))
        companion_columns = np.unique(
            np.concatenate(
                [companion_sources[piece.source_row].routes for piece in pieces]
            )
        )
        companion, defect = _material_common_moments(
            tuple(companion_sources),
            following.basis,
            pieces,
            companion_columns,
            work,
            exact,
            plan,
        )
        continuity = max(continuity, defect)
        companion_blocks.append((new_routes, companion_columns, companion.value))
        published_checks.append(
            (
                cell.routes,
                differential,
                columns,
                expected,
                companion_columns,
                companion,
            )
        )
        new_offset += new_routes.size
    return _MaterialTargetMoments(new_offset=new_offset, continuity=continuity)


def _check_published_material_commuting(
    published_checks: list[
        tuple[
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            MomentIntegralMatrix,
            NDArray[np.int64],
            MomentIntegralMatrix,
        ]
    ],
    routes: NDArray[np.int64],
    coefficients: NDArray[np.float64],
    companion_owners: dict[int, tuple[int, int]],
    source_differentials: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
    ],
    work: PreparationWork,
) -> None:
    "Bound commuting defects using the published field coefficient rows."
    for (
        dofs,
        differential,
        expected_columns,
        expected,
        companion_columns,
        companion,
    ) in published_checks:
        columns = np.unique(
            np.concatenate((expected_columns, routes[dofs].reshape((-1,))))
        )
        work.charge(
            differential.size * columns.size,
            dofs.size * columns.size + 4 * differential.shape[0] * columns.size,
        )
        values = np.zeros((dofs.size, columns.size), dtype=np.float64)
        np.add.at(
            values,
            (np.arange(dofs.size)[:, None], np.searchsorted(columns, routes[dofs])),
            coefficients[dofs],
        )
        actual = differential @ values
        desired = np.zeros((differential.shape[0], columns.size), dtype=np.float64)
        error = np.zeros_like(desired)
        for index, identifier in enumerate(companion_columns):
            owner, local_row = companion_owners[int(identifier)]
            _, source_columns, source_differential = source_differentials[owner]
            differential_row = source_differential[local_row]
            work.charge(
                differential.shape[0] * source_columns.size,
                2 * differential.shape[0] * source_columns.size,
            )
            contribution = companion.value[:, index, None] * differential_row[None]
            uncertainty = companion.error[:, index, None] * np.abs(differential_row[None])
            uncertainty += np.finfo(np.float64).eps * np.abs(contribution)
            selected = np.searchsorted(columns, source_columns)
            np.add.at(
                desired,
                (np.arange(differential.shape[0])[:, None], selected[None]),
                contribution,
            )
            np.add.at(
                error,
                (np.arange(differential.shape[0])[:, None], selected[None]),
                uncertainty,
            )
        gamma = np.finfo(np.float64).eps * max(
            companion_columns.size, differential.shape[1]
        )
        error = np.nextafter(
            error
            + gamma
            * (np.abs(desired) + np.abs(differential) @ np.abs(values))
            / (1 - gamma),
            np.inf,
        )
        independently_integrated = embed_columns(
            expected.value, expected_columns, columns, work
        )
        independent_error = embed_columns(expected.error, expected_columns, columns, work)
        scale_ = max(
            float(np.max(np.abs(desired), initial=0)),
            float(np.max(np.abs(independently_integrated), initial=0)),
            1.0,
        )
        work.commuting_defect = max(
            work.commuting_defect,
            float(
                np.max(
                    np.abs(desired - independently_integrated)
                    + error
                    + independent_error,
                    initial=0,
                )
            )
            / scale_,
        )
        scale_ = max(
            float(np.max(np.abs(actual), initial=0)),
            float(np.max(np.abs(desired), initial=0)),
            1.0,
        )
        work.commuting_defect = max(
            work.commuting_defect,
            float(np.max(np.abs(actual - desired) + error, initial=0)) / scale_,
        )


def _material_owned_rows(
    candidate_rows: list[int],
    candidate_columns: list[NDArray[np.int64]],
    candidate_values: list[NDArray[np.float64]],
    target_size: int,
    source_size: int,
    work: PreparationWork,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Charge padded storage before reconciling shared material coefficient rows."""
    width = max(len(columns) for columns in candidate_columns)
    work.charge(0, 2 * len(candidate_rows) * width)
    padded_columns = np.zeros((len(candidate_rows), width), dtype=np.int64)
    padded_values = np.zeros_like(padded_columns, dtype=np.float64)
    for row, (columns, values) in enumerate(
        zip(candidate_columns, candidate_values, strict=True)
    ):
        padded_columns[row, : columns.size], padded_values[row, : values.size] = (
            columns,
            values,
        )
    routes, coefficients, shared = owned_rows(
        np.asarray(candidate_rows, dtype=np.int64)[:, None],
        padded_columns,
        padded_values[:, None],
        target_size,
        source_size,
    )
    return routes, coefficients, shared, width


@checked
def prepare_material_chart_compatible_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation,
    /,
    *,
    field_name: str,
    prepare_pieces: Callable[
        [PreparationWork, bool], tuple[MaterialCompatiblePiece, ...]
    ],
    maximum_work: int,
    maximum_storage_bytes: int,
    budget: CoordinateEnclosureBudget,
    invariant_cache: MaterialInvariantCache | None = None,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Publish independent-chart moments and their whole-column companion.\n\n producers supply whole partitions, not rounded corner fits.\n Interior-only commuting correction preserves all target boundary moments.\n The returned companion is checked against the published field map.\n Constant-density tests on entities of the form's dimension integrate from\n exact straight vertex images; ``prepare_pieces`` builds generic charts only\n when some test is not of that kind. ``invariant_cache`` shares that exact\n geometry across one request's fields."
    si, ti = source.field_index(field_name), target.field_index(field_name)
    conformities = {
        element.conformity for element in source.elements[si] + target.elements[ti]
    }
    if conformities not in ({"Hcurl"}, {"Hdiv"}):
        raise ValueError(
            "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material compatible transport must retain H(curl) or H(div) conformity."
        )
    identities = {
        (
            element.value_spec.value_spec_id,
            element.degree,
            element.representation,
            element.family,
        )
        for endpoint, index in ((source, si), (target, ti))
        for element in endpoint.elements[index]
    }
    if len(identities) != 1:
        raise ValueError(
            "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material forms cannot change degree, twist, proxy, order, or family."
        )
    sources, targets = form_cells(source, si), form_cells(target, ti)
    if any(
        cell.basis.dimension != 2 or cell.basis.form_degree != 1
        for cell in sources + targets
    ):
        raise ValueError(
            "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material compatible fields require triangular one-form bases."
        )
    ss, ts = source.dof_maps[si].global_dof_count, target.dof_maps[ti].global_dof_count
    old_space, new_space = (
        source.field_spaces[si].vector_space,
        target.field_spaces[ti].vector_space,
    )
    match (old_space, new_space):
        case (ArraySpace() as old_space, ArraySpace() as new_space) if not (
            old_space.shape != (ss,)
            or new_space.shape != (ts,)
            or old_space.dtype != new_space.dtype
        ):
            pass
        case _:
            raise ValueError(
                "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material compatible coefficient shape/precision contracts differ."
            )
    work = PreparationWork(maximum_work, maximum_storage_bytes, budget=budget)
    exact = ExactFormMoments(work.enclosure)
    derivative_cache: dict[int, tuple[FormBasis, NDArray[np.float64]]] = {}
    candidate_rows, candidate_columns, candidate_values = [], [], []
    source_differentials, target_differentials, companion_blocks = [], [], []
    companion_sources: list[FormCell] = []
    derivative_sources: list[FormCell] = []
    companion_owners: dict[int, tuple[int, int]] = {}
    old_offset = 0
    continuity = 0.0
    published_checks = []
    with work.enclosure.activate():
        _phase = _prepare_material_source_differentials(
            sources=sources,
            derivative_cache=derivative_cache,
            work=work,
            budget=budget,
            source_differentials=source_differentials,
            companion_owners=companion_owners,
            derivative_sources=derivative_sources,
            companion_sources=companion_sources,
            old_offset=old_offset,
        )
        old_offset = _phase.old_offset
        plan = _invariant_plan(
            sources,
            derivative_sources,
            targets,
            derivative_cache,
            work,
            exact,
            budget,
            MaterialInvariantCache() if invariant_cache is None else invariant_cache,
        )
        charts: list[list[MaterialCompatiblePiece]] = [[] for _ in targets]
        with work.enclosure.polynomial_preparation_scope():
            for piece in prepare_pieces(work, plan is None):
                if not 0 <= piece.source_row < len(sources) or not (
                    0 <= piece.target_row < len(targets)
                ):
                    raise ValueError(
                        "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material pieces do not bind the source and target cell rows."
                    )
                if plan is None:
                    if piece.charts is None:
                        raise ValueError(
                            "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material pieces omit their requested generic charts."
                        )
                    work.enclosure.retain_basis(piece.charts)
                charts[piece.target_row].append(piece)
        new_offset = 0
        _phase = _prepare_material_target_moments(
            sources=sources,
            targets=targets,
            charts=charts,
            derivative_cache=derivative_cache,
            work=work,
            exact=exact,
            plan=plan,
            budget=budget,
            derivative_sources=derivative_sources,
            companion_sources=companion_sources,
            candidate_rows=candidate_rows,
            candidate_columns=candidate_columns,
            candidate_values=candidate_values,
            target_differentials=target_differentials,
            companion_blocks=companion_blocks,
            published_checks=published_checks,
            new_offset=new_offset,
            continuity=continuity,
        )
        new_offset = _phase.new_offset
        continuity = _phase.continuity
        routes, coefficients, shared, width = _material_owned_rows(
            candidate_rows, candidate_columns, candidate_values, ts, ss, work
        )
        _check_published_material_commuting(
            published_checks,
            routes,
            coefficients,
            companion_owners,
            source_differentials,
            work,
        )
    condition = max(work.condition, 1.0) * width
    evidence = FiniteElementTransferEvidence(
        {
            "commuting": work.commuting_defect,
            "patch_trace": continuity,
            "continuity": shared,
            "moment_solve": work.solve_defect,
            "basis_duality": work.basis_solve_defect,
            "integration": work.integration_error,
        },
        CLAIM_ULPS * np.finfo(np.float64).eps * condition,
        bounds={
            "moments": work.integration_error,
            "geometry-displacement": prepared.maximum_displacement_bound,
        },
        estimates={
            "moment_condition": work.condition,
            "minimum_local_rank": work.minimum_rank,
            "projection_moment_change": work.projection_change,
            "preparation_work": float(work.preparation_work),
            "preparation_storage_bytes": float(work.preparation_storage_upper),
        },
    )
    if not evidence.passed:
        raise ValueError(
            f"surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material compatible moments failed their certificate:{evidence.defects}.."
        )
    primal = SparseLinearMap(
        RowRelation(routes.astype(np.int32), source_size=ss),
        coefficients,
        operator_id=canonical_fingerprint(
            {
                "kind": "material-chart-compatible-moments",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "field": field_name,
                "deformation": prepared.deformation_id,
                "routes": array_tree_fingerprint(routes),
                "values": array_tree_fingerprint(coefficients),
            }
        ),
    )
    transfer = FiniteElementTopologyTransfer(
        primal,
        prepared.source_topology_id,
        prepared.target_topology_id,
        action_condition=condition,
        semantics="covariant-piola"
        if conformities == {"Hcurl"}
        else "contravariant-piola",
    )
    field = FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[si],
        target.field_spaces[ti],
        deformation_binding(prepared),
        evidence,
    )
    return PreparedSurfaceChartCompatibleTransfer(
        field,
        _material_sparse_blocks(
            source_differentials, old_offset, ss, "material-chart-source-differential"
        ),
        _material_sparse_blocks(
            target_differentials, new_offset, ts, "material-chart-target-differential"
        ),
        _material_sparse_blocks(
            companion_blocks, new_offset, old_offset, "material-chart-two-form-companion"
        ),
    )


def _admit_fraction_action(
    work: PreparationWork,
    values: tuple[Fraction, ...],
    operations: int,
    coefficients: int,
) -> None:
    bits = (
        4
        * sum(
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for value in values
        )
        + 8
    )
    reserve_fraction_work(
        work.enclosure, operations + 2 * len(values), coefficients, bits
    )


def _material_affine_arguments(
    vertices: tuple[_Point, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[algebra.Expression, ...]:
    variables = algebra.axes(2, budget=budget)
    return tuple(
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


def _observe_material_affine_condition(
    source: tuple[_Point, ...],
    target: tuple[_Point, ...],
    work: PreparationWork,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:

    _admit_fraction_action(
        work, tuple(value for point in (*source, *target) for value in point), 8, 8
    )
    source_frame = tuple(
        tuple(source[column + 1][axis] - source[0][axis] for axis in range(2))
        for column in range(2)
    )
    target_frame = tuple(
        tuple(target[column + 1][axis] - target[0][axis] for axis in range(2))
        for column in range(2)
    )
    prepared = imported_linalg__small_batched.prepare_exact_small_linear_actions(
        source_frame, target_frame, budget=work.enclosure
    )
    if prepared.actions is None:
        raise ValueError(
            "surface_chart.observe_material_affine_condition.reference_map must satisfy the regular reference-map contract; got Material references lack an regular affine action."
        )
    entries = tuple(value for row in prepared.actions for value in row)
    _admit_fraction_action(work, entries, 12, 12)
    determinant = abs(orientation(*target) / orientation(*source))
    upper = fraction_endpoint(
        sum((value * value for value in entries), Fraction(0)) / determinant,
        upper=True,
        saturate=True,
    )
    if not np.isfinite(upper) or upper > 1e12:
        raise ValueError(
            "surface_chart.observe_material_affine_condition.reference_map must satisfy the regular reference-map contract; got Material reference action exceeds its source whole-piece condition bound."
        )
    work.condition = max(work.condition, upper)


def _exact_triangle(points: NDArray[np.object_]) -> _Triangle:
    """Read an exact reference triangle without rounding its coordinates."""
    values = tuple(tuple(point) for point in points)
    if len(values) != 3 or any(
        len(point) != 2 or not all(isinstance(value, Fraction) for value in point)
        for point in values
    ):
        raise TypeError(
            "surface_chart.prepare_pieces.vertices must contain three pairs of Fraction coordinates; got another representation"
        )
    first, second, third = (
        (Fraction(point[0]), Fraction(point[1])) for point in values
    )
    return first, second, third


class _SurfaceMaterialPiecePreparation(StrictModule):
    """Prepare paired material actions with visible endpoint arrays."""

    prepared: PreparedSurfaceChartDeformation
    source_order: NDArray[np.int64]
    target_order: NDArray[np.int64]

    def __call__(
        self, work: PreparationWork, charts: bool
    ) -> tuple[MaterialCompatiblePiece, ...]:
        pieces: list[MaterialCompatiblePiece] = []
        source_coverage = [Fraction(0) for _ in self.source_order]
        target_coverage = [Fraction(0) for _ in self.target_order]
        for occurrence in self.prepared.occurrences:
            old_rows = self.source_order[
                np.asarray(occurrence.source_rows, dtype=np.int64)
            ]
            new_rows = self.target_order[
                np.asarray(occurrence.target_rows, dtype=np.int64)
            ]
            for _, old_row, new_row, old_reference, new_reference in piece_maps(
                occurrence
            ):
                old = int(old_rows[old_row])
                new = int(new_rows[new_row])
                old_triangle = _exact_triangle(old_reference)
                new_triangle = _exact_triangle(new_reference)
                _admit_fraction_action(
                    work,
                    tuple((value for point in old_triangle for value in point)),
                    7,
                    7,
                )
                source_orientation = orientation(*old_triangle)
                if source_orientation == 0:
                    raise ValueError(
                        "surface_chart.prepare_pieces.reference_map must satisfy the regular reference-map contract; got A material piece has a singular source reference action."
                    )
                if source_orientation < 0:
                    old_triangle = (old_triangle[0], old_triangle[2], old_triangle[1])
                    new_triangle = (new_triangle[0], new_triangle[2], new_triangle[1])
                    source_orientation = -source_orientation
                _admit_fraction_action(
                    work,
                    tuple((value for point in new_triangle for value in point)),
                    7,
                    7,
                )
                image_orientation = orientation(*new_triangle)
                if image_orientation == 0:
                    raise ValueError(
                        "surface_chart.prepare_pieces.reference_map must satisfy the regular reference-map contract; got A material piece has a singular target reference action."
                    )
                source_area, target_area = source_orientation / 2, image_orientation / 2
                source_coverage[old] += source_area
                target_coverage[new] += abs(target_area)
                _observe_material_affine_condition(
                    old_triangle, new_triangle, work, budget=work.enclosure
                )
                pair = None
                if charts:
                    pair = (
                        _material_affine_arguments(old_triangle, budget=work.enclosure),
                        _material_affine_arguments(new_triangle, budget=work.enclosure),
                    )
                    work.enclosure.retain_basis(pair)
                pieces.append(
                    MaterialCompatiblePiece(
                        old,
                        new,
                        old_triangle,
                        new_triangle,
                        source_area,
                        target_area,
                        pair,
                    )
                )
        if any((area != Fraction(1, 2) for area in (*source_coverage, *target_coverage))):
            raise ValueError(
                "surface_chart.prepare_pieces.pieces must satisfy the material reference contract; got material pieces do not partition the full source and target physical references."
            )
        return tuple(pieces)


def _prepare_surface_chart_compatible_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    budget: CoordinateEnclosureBudget,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Integrate same-order one-forms on ordered material pieces.\n\n Physical element reference triangles and their paired affine actions\n are retained by the deformation owner; no UV reconstruction is performed."
    source_order, target_order = validate(
        source, target, prepared, source_geometry, target_geometry, budget=budget
    )

    prepare_pieces = _SurfaceMaterialPiecePreparation(
        prepared=prepared, source_order=source_order, target_order=target_order
    )

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
    "PreparedSurfaceChartCompatibleTransfer",
    "prepare_surface_chart_compatible_transfer",
]


@checked
def prepare_surface_chart_compatible_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation,
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
        return _prepare_surface_chart_compatible_transfer(
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
            return _prepare_surface_chart_compatible_transfer(
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
