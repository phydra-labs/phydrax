#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import math
import sys
from enum import StrEnum
from fractions import Fraction
from typing import assert_never

import numpy as np

from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._embedding_state import boxes, candidate_pairs

from .._strict import StrictModule
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_algebraic_geometry import ExactAlgebraicCellGeometrySource
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._exact_rational_geometry import ExactRationalCellGeometrySource
from ..typing import checked
from ._exact_polyhedral_geometry import (
    exact_vertices,
    ExactOverlapRefusal,
    ExactTetrahedronOverlap,
    star_tetrahedra,
    StarTetrahedronPreparation,
    tetrahedra_disjoint,
    Tetrahedron,
    tetrahedron_overlap,
    TetrahedronPlanePreparation,
)
from ._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementEvidence,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    Counts,
    Decomposition,
    Entries,
    PreparedCommonRefinement,
)


def _rne_error(value: Fraction, /) -> float:
    error = abs(value - Fraction(float(value)))
    return 0.0 if not error else float(np.nextafter(float(error), math.inf))


def _cell_integrals(
    stars: tuple[tuple[Tetrahedron, ...], ...],
    /,
    *,
    preparation: StarTetrahedronPreparation,
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]]:
    measures, moments = [], []
    for star in stars:
        volume, moment = Fraction(0), [Fraction(0)] * 3
        for tet in star:
            measure = preparation.determinant(tet) / 6
            volume += measure
            for axis in range(3):
                moment[axis] += (
                    measure * sum((point[axis] for point in tet), Fraction(0)) / 4
                )
        measures.append(volume)
        moments.append(tuple(moment))
    return tuple(measures), tuple(moments)


class _PowerDecomposition(StrictModule):
    carrier: Decomposition
    stars: tuple[tuple[Tetrahedron, ...], ...]
    measures: tuple[Fraction, ...]


class _PowerGeometries(StrictModule):
    source: CellGeometrySpec
    target: CellGeometrySpec


def _decomposition(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PowerDecomposition:
    points = exact_vertices(mesh, geometry, budget=budget)
    preparation = StarTetrahedronPreparation(budget)
    stars = star_tetrahedra(mesh, points, budget=budget, preparation=preparation)
    measures, moments = _cell_integrals(stars, preparation=preparation)
    cell_rows = tuple(
        row[valid]
        for block in mesh.blocks
        for row, valid in zip(
            np.asarray(block.vertices), np.asarray(block.vertex_valid), strict=True
        )
    )
    lower, upper = [], []
    for row in cell_rows:
        low, high = boxes(points, row[None, :])
        lower.append(low[0])
        upper.append(high[0])
    offsets = np.concatenate(([0], np.cumsum(tuple(len(star) for star in stars)))).astype(
        np.int64
    )
    identifiers = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    part = Decomposition(
        len(stars),
        identifiers,
        np.asarray(tuple(tet for star in stars for tet in star), dtype=np.float64),
        None,
        offsets,
        np.asarray(measures, dtype=np.float64),
        np.asarray(moments, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        0,
        0,
    )
    return _PowerDecomposition(part, stars, measures)


def _fraction_bytes(stars: tuple[tuple[Tetrahedron, ...], ...], /) -> int:
    return sum(
        sys.getsizeof(star)
        + sum(
            sys.getsizeof(tet)
            + sum(
                sys.getsizeof(point)
                + sum(
                    sys.getsizeof(value)
                    + sys.getsizeof(value.numerator)
                    + sys.getsizeof(value.denominator)
                    for value in point
                )
                for point in tet
            )
            for tet in star
        )
        for star in stars
    )


class _PowerRefusal(StrEnum):
    NONE = ""
    DECOMPOSITION = (
        "Exact power decomposition or candidates exceed their resource bounds."
    )
    WORK = "Exact power overlap coefficient work exhausted maximum_exact_work."
    RATIONAL_STORAGE = (
        "Exact power overlap rational working storage exhausted maximum_memory_bytes."
    )
    PAIRS = "Exact power overlap retained pairs exhausted maximum_accepted_pairs."
    PUBLICATION = "Exact power overlap publication exhausted maximum_memory_bytes."
    DOUBLE_COVERAGE = "Exact source common refinement proves double coverage."
    COVERAGE_GAP = "Exact source common refinement proves incomplete coverage."


class _PowerPair(StrictModule):
    volume: Fraction
    moment: tuple[Fraction, ...]
    second_moment: tuple[tuple[Fraction, ...], ...] | None
    pieces: tuple[Tetrahedron, ...]
    work: int
    piece_pairs: int
    working_bytes: int
    refusal: _PowerRefusal


def _power_pair(
    first: tuple[Tetrahedron, ...],
    second: tuple[Tetrahedron, ...],
    policy: CommonRefinementPolicy,
    work: int,
    base_bytes: int,
    retained_bytes: int,
    working_bytes: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparation: TetrahedronPlanePreparation,
) -> _PowerPair:
    volume, moment = Fraction(0), [Fraction(0)] * 3
    second_moment = (
        [[Fraction(0)] * 3 for _ in range(3)] if policy.second_moments else None
    )
    pieces: list[Tetrahedron] = []
    consumed, piece_pairs = 0, 0
    refusal = _PowerRefusal.NONE
    for first_tet in first:
        for second_tet in second:
            piece_pairs += 1
            if tetrahedra_disjoint(first_tet, second_tet):
                continue
            overlap = tetrahedron_overlap(
                first_tet,
                second_tet,
                second_moments=policy.second_moments,
                retain_simplices=policy.overlap_simplices,
                maximum_work=policy.maximum_exact_work - work - consumed,
                budget=budget,
                preparation=preparation,
            )
            match overlap:
                case ExactOverlapRefusal():
                    refusal = _PowerRefusal.WORK
                    break
                case ExactTetrahedronOverlap():
                    pass
                case invalid:
                    assert_never(invalid)
            consumed += overlap.operation_count
            if work + consumed > policy.maximum_exact_work:
                refusal = _PowerRefusal.WORK
                break
            volume += overlap.volume
            for axis in range(3):
                moment[axis] += overlap.first_moment[axis]
            if second_moment is not None and overlap.second_moment is not None:
                for i in range(3):
                    for j in range(3):
                        second_moment[i][j] += overlap.second_moment[i][j]
            if overlap.simplices is not None:
                pieces.extend(overlap.simplices)
            scalars = (
                (volume, *moment, *(value for row in second_moment for value in row))
                if second_moment is not None
                else (volume, *moment)
            )
            transient_bytes = sum(
                sys.getsizeof(value)
                + sys.getsizeof(value.numerator)
                + sys.getsizeof(value.denominator)
                for value in scalars
            ) + _fraction_bytes((tuple(pieces),))
            working_bytes = max(
                working_bytes, base_bytes + retained_bytes + transient_bytes
            )
            if working_bytes > policy.maximum_memory_bytes:
                refusal = _PowerRefusal.RATIONAL_STORAGE
                break
        if refusal != _PowerRefusal.NONE:
            break
    return _PowerPair(
        volume,
        tuple(moment),
        None if second_moment is None else tuple(tuple(row) for row in second_moment),
        tuple(pieces),
        consumed,
        piece_pairs,
        working_bytes,
        refusal,
    )


def _power_geometries(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    source_geometry: CellGeometrySpec | None,
    target_geometry: CellGeometrySpec | None,
    /,
) -> _PowerGeometries:
    source_geometry = (
        CellGeometrySpec.affine(source_mesh)
        if source_geometry is None
        else source_geometry
    )
    target_geometry = (
        CellGeometrySpec.affine(target_mesh)
        if target_geometry is None
        else target_geometry
    )
    power = False
    for geometry in (source_geometry, target_geometry):
        match geometry.exact_source:
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource()
            ):
                power = True
            case ExactAlgebraicCellGeometrySource():
                raise ValueError(
                    "common_refinement.geometry.exact_source must be a power source or None; got an algebraic nodal source"
                )
            case (
                ExactPlcCellGeometrySource()
                | ExactPlcCellGeometryConvexSource()
                | ExactRationalCellGeometrySource()
            ):
                raise ValueError(
                    "common_refinement.geometry.exact_source must be a power source or None; got PLC source"
                )
            case None:
                pass
            case invalid:
                assert_never(invalid)
    if not power:
        raise ValueError(
            "common_refinement.geometry.exact_source must contain at least one power source; got none"
        )
    return _PowerGeometries(source_geometry, target_geometry)


@checked
def prepare_exact_power_refinement(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    source_geometry: CellGeometrySpec | None,
    target_geometry: CellGeometrySpec | None,
    policy: CommonRefinementPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PreparedCommonRefinement:
    """Prepare rational power-cell overlaps and their coverage evidence."""
    geometries = _power_geometries(
        source_mesh, target_mesh, source_geometry, target_geometry
    )
    source_geometry, target_geometry = geometries.source, geometries.target
    source = _decomposition(source_mesh, source_geometry, budget=budget)
    target = _decomposition(target_mesh, target_geometry, budget=budget)
    first, first_stars, first_measures = source.carrier, source.stars, source.measures
    second, second_stars, second_measures = target.carrier, target.stars, target.measures
    identities = (
        source_mesh.mesh_id,
        target_mesh.mesh_id,
        source_mesh.topology_id,
        target_mesh.topology_id,
    )
    geometry_ids = (cell_geometry_id(source_geometry), cell_geometry_id(target_geometry))
    base_bytes = (
        _fraction_bytes(first_stars)
        + _fraction_bytes(second_stars)
        + sum(
            value.nbytes
            for part in (first, second)
            for value in (
                part.pieces,
                part.piece_offsets,
                part.measures,
                part.first_moments,
                part.bbox_min,
                part.bbox_max,
            )
        )
    )
    first_cells, second_cells, exceeded = candidate_pairs(
        first.bbox_min,
        first.bbox_max,
        policy.maximum_candidate_pairs,
        other=(second.bbox_min, second.bbox_max),
    )
    candidates = tuple(
        sorted(zip(second_cells.tolist(), first_cells.tolist(), strict=True))
    )
    covered_first, covered_second = (
        [Fraction(0)] * first.cell_count,
        [Fraction(0)] * second.cell_count,
    )
    records, error_bounds, simplex_rows, simplex_offsets = [], [], [], [0]
    work, piece_pairs = 0, 0
    working_bytes, retained_bytes = base_bytes, 0
    status, reason = CommonRefinementStatus.SUCCESS, _PowerRefusal.NONE
    if exceeded or base_bytes > policy.maximum_memory_bytes:
        status, reason = (
            CommonRefinementStatus.RESOURCE_LIMIT,
            _PowerRefusal.DECOMPOSITION,
        )
    preparation = TetrahedronPlanePreparation(budget)
    for target, source in candidates if status == CommonRefinementStatus.SUCCESS else ():
        pair = _power_pair(
            first_stars[source],
            second_stars[target],
            policy,
            work,
            base_bytes,
            retained_bytes,
            working_bytes,
            budget=budget,
            preparation=preparation,
        )
        work += pair.work
        piece_pairs += pair.piece_pairs
        working_bytes = pair.working_bytes
        if pair.refusal != _PowerRefusal.NONE:
            status, reason = CommonRefinementStatus.RESOURCE_LIMIT, pair.refusal
            break
        volume, moment, second_moment, pieces = (
            pair.volume,
            pair.moment,
            pair.second_moment,
            pair.pieces,
        )
        if volume == 0:
            continue
        if len(records) >= policy.maximum_accepted_pairs:
            status, reason = (
                CommonRefinementStatus.RESOURCE_LIMIT,
                _PowerRefusal.PAIRS,
            )
            break
        records.append(
            (
                target,
                source,
                float(volume),
                tuple(float(value) for value in moment),
                second_moment,
            )
        )
        error_bounds.append(_rne_error(volume))
        covered_first[source] += volume
        covered_second[target] += volume
        simplex_rows.extend(pieces)
        simplex_offsets.append(len(simplex_rows))
        retained_bytes += (
            8 * (7 + (9 if policy.second_moments else 0))
            + _fraction_bytes((tuple(pieces),))
            + len(pieces) * 4 * 3 * 8
        )
        if base_bytes + retained_bytes > policy.maximum_memory_bytes:
            status, reason = (
                CommonRefinementStatus.RESOURCE_LIMIT,
                _PowerRefusal.PUBLICATION,
            )
            break
    first_defects = tuple(
        covered - measure
        for covered, measure in zip(covered_first, first_measures, strict=True)
    )
    second_defects = tuple(
        covered - measure
        for covered, measure in zip(covered_second, second_measures, strict=True)
    )
    if status == CommonRefinementStatus.SUCCESS:
        if any(value > 0 for value in (*first_defects, *second_defects)):
            status, reason = (
                CommonRefinementStatus.DOUBLE_COVERAGE,
                _PowerRefusal.DOUBLE_COVERAGE,
            )
        elif (
            policy.coverage
            in (CommonRefinementCoverage.COMPLETE, CommonRefinementCoverage.SOURCE)
            and any(first_defects)
        ) or (
            policy.coverage
            in (CommonRefinementCoverage.COMPLETE, CommonRefinementCoverage.TARGET)
            and any(second_defects)
        ):
            status, reason = (
                CommonRefinementStatus.COVERAGE_GAP,
                _PowerRefusal.COVERAGE_GAP,
            )
    entries = Entries(
        np.asarray(tuple(row[0] for row in records), dtype=np.int32),
        np.asarray(tuple(row[1] for row in records), dtype=np.int32),
        np.asarray(tuple(row[2] for row in records), dtype=np.float64),
        np.asarray(tuple(row[3] for row in records), dtype=np.float64).reshape((-1, 3)),
        np.asarray(tuple(row[4] for row in records), dtype=np.float64).reshape((-1, 3, 3))
        if policy.second_moments
        else None,
        np.asarray(simplex_offsets, dtype=np.int32) if policy.overlap_simplices else None,
        np.asarray(simplex_rows, dtype=np.float64).reshape((-1, 4, 3))
        if policy.overlap_simplices
        else None,
    )
    source_errors, target_errors = (
        np.asarray(
            tuple(_rne_error(value) for value in first_measures), dtype=np.float64
        ),
        np.asarray(
            tuple(_rne_error(value) for value in second_measures), dtype=np.float64
        ),
    )
    if records:
        np.add.at(source_errors, entries.source_cells, error_bounds)
        np.add.at(target_errors, entries.target_cells, error_bounds)
    evidence = CommonRefinementEvidence(
        status=status,
        reason=reason.value,
        source_coverage_defects=np.asarray(first_defects, dtype=np.float64),
        target_coverage_defects=np.asarray(second_defects, dtype=np.float64),
        source_measures=first.measures,
        target_measures=second.measures,
        source_tolerances=source_errors,
        target_tolerances=target_errors,
        counts=Counts(len(candidates), len(records), piece_pairs, 0, 0, 0),
        retained_bytes=retained_bytes,
        working_bytes=working_bytes,
    )
    return PreparedCommonRefinement(
        source=first,
        target=second,
        entries=entries,
        identities=identities,
        geometry_ids=geometry_ids,
        volume_error_bounds=np.asarray(error_bounds, dtype=np.float64),
        policy=policy,
        evidence=evidence,
    )
