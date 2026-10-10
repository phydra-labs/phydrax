#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import math
import sys
from fractions import Fraction
from typing import assert_never

import numpy as np

from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ._exact_polyhedral_geometry import (
    determinant3,
    exact_vertices,
    star_tetrahedra,
    Tetrahedron,
    tetrahedron_overlap,
)
from ._mesh_certificates import _boxes, _candidate_pairs
from ._supermesh import (
    _Counts,
    _Decomposition,
    _Entries,
    CommonRefinementCoverage,
    CommonRefinementEvidence,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    PreparedCommonRefinement,
)


def _rne_error(value: Fraction, /) -> float:
    error = abs(value - Fraction(float(value)))
    return 0.0 if not error else float(np.nextafter(float(error), math.inf))


def _cell_integrals(
    stars: tuple[tuple[Tetrahedron, ...], ...], /
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]]:
    measures, moments = [], []
    for star in stars:
        volume, moment = Fraction(0), [Fraction(0)] * 3
        for tet in star:
            measure = (
                determinant3(
                    *(
                        tuple(
                            value - base
                            for value, base in zip(point, tet[0], strict=True)
                        )
                        for point in tet[1:]
                    )
                )
                / 6
            )
            volume += measure
            for axis in range(3):
                moment[axis] += (
                    measure * sum((point[axis] for point in tet), Fraction(0)) / 4
                )
        measures.append(volume)
        moments.append(tuple(moment))
    return tuple(measures), tuple(moments)


def _decomposition(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[_Decomposition, tuple[tuple[Tetrahedron, ...], ...], tuple[Fraction, ...]]:
    points = exact_vertices(mesh, geometry)
    stars = star_tetrahedra(mesh, points)
    measures, moments = _cell_integrals(stars)
    cell_rows = tuple(
        row[valid]
        for block in mesh.blocks
        for row, valid in zip(
            np.asarray(block.vertices), np.asarray(block.vertex_valid), strict=True
        )
    )
    lower, upper = [], []
    for row in cell_rows:
        low, high = _boxes(points, row[None, :])
        lower.append(low[0])
        upper.append(high[0])
    offsets = np.concatenate(([0], np.cumsum(tuple(len(star) for star in stars)))).astype(
        np.int64
    )
    identifiers = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    part = _Decomposition(
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
    return part, stars, measures


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


def prepare_exact_power_refinement(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    source_geometry: CellGeometrySpec | None,
    target_geometry: CellGeometrySpec | None,
    policy: CommonRefinementPolicy,
    /,
) -> PreparedCommonRefinement:
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
            case ExactPlcCellGeometrySource() | ExactPlcCellGeometryConvexSource():
                raise ValueError(
                    "Source-aware polyhedral refinement refuses exact PLC "
                    "sources; they are not power polyhedra."
                )
            case None:
                pass
            case invalid:
                assert_never(invalid)
    if not power:
        raise ValueError(
            "Source-aware polyhedral refinement requires an exact power construction on at least one side."
        )
    first, first_stars, first_measures = _decomposition(source_mesh, source_geometry)
    second, second_stars, second_measures = _decomposition(target_mesh, target_geometry)
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
    first_cells, second_cells, exceeded = _candidate_pairs(
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
    status, reason = CommonRefinementStatus.SUCCESS, ""
    if exceeded or base_bytes > policy.maximum_memory_bytes:
        status, reason = (
            CommonRefinementStatus.RESOURCE_LIMIT,
            "Exact power decomposition or candidates exceed their resource bounds.",
        )
    for target, source in candidates if status == CommonRefinementStatus.SUCCESS else ():
        volume, moment = Fraction(0), [Fraction(0)] * 3
        second_moment = (
            [[Fraction(0)] * 3 for _ in range(3)] if policy.second_moments else None
        )
        pieces = []
        for first_tet in first_stars[source]:
            for second_tet in second_stars[target]:
                piece_pairs += 1
                if any(
                    max(
                        min(point[axis] for point in first_tet),
                        min(point[axis] for point in second_tet),
                    )
                    >= min(
                        max(point[axis] for point in first_tet),
                        max(point[axis] for point in second_tet),
                    )
                    for axis in range(3)
                ):
                    continue
                from ..discretization._coordinate_enclosure import (
                    CoordinateEnclosureResourceError,
                )

                try:
                    overlap = tetrahedron_overlap(
                        first_tet,
                        second_tet,
                        second_moments=policy.second_moments,
                        retain_simplices=policy.overlap_simplices,
                        maximum_work=policy.maximum_exact_work - work,
                    )
                except CoordinateEnclosureResourceError:
                    status, reason = (
                        CommonRefinementStatus.RESOURCE_LIMIT,
                        "Exact power overlap coefficient work exhausted maximum_exact_work.",
                    )
                    break
                work += overlap.operation_count
                if work > policy.maximum_exact_work:
                    status, reason = (
                        CommonRefinementStatus.RESOURCE_LIMIT,
                        "Exact power overlap coefficient work exhausted maximum_exact_work.",
                    )
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
                exact_scalars = (
                    (volume, *moment, *(value for row in second_moment for value in row))
                    if second_moment is not None
                    else (volume, *moment)
                )
                transient_bytes = sum(
                    sys.getsizeof(value)
                    + sys.getsizeof(value.numerator)
                    + sys.getsizeof(value.denominator)
                    for value in exact_scalars
                ) + _fraction_bytes((tuple(pieces),))
                working_bytes = max(
                    working_bytes, base_bytes + retained_bytes + transient_bytes
                )
                if working_bytes > policy.maximum_memory_bytes:
                    status, reason = (
                        CommonRefinementStatus.RESOURCE_LIMIT,
                        "Exact power overlap rational working storage exhausted maximum_memory_bytes.",
                    )
                    break
            if status != CommonRefinementStatus.SUCCESS:
                break
        if status != CommonRefinementStatus.SUCCESS:
            break
        if volume == 0:
            continue
        if len(records) >= policy.maximum_accepted_pairs:
            status, reason = (
                CommonRefinementStatus.RESOURCE_LIMIT,
                "Exact power overlap retained pairs exhausted maximum_accepted_pairs.",
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
                "Exact power overlap publication exhausted maximum_memory_bytes.",
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
                "Exact source common refinement proves double coverage.",
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
                "Exact source common refinement proves incomplete coverage.",
            )
    entries = _Entries(
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
        reason=reason,
        source_coverage_defects=np.asarray(first_defects, dtype=np.float64),
        target_coverage_defects=np.asarray(second_defects, dtype=np.float64),
        source_measures=first.measures,
        target_measures=second.measures,
        source_tolerances=source_errors,
        target_tolerances=target_errors,
        counts=_Counts(len(candidates), len(records), piece_pairs, 0, 0, 0),
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
