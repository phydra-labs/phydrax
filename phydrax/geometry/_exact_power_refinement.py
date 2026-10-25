#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import math
from enum import StrEnum
from fractions import Fraction
from typing import assert_never

import numpy as np

from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from phydrax.geometry._embedding_state import boxes

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._geometry_predicates import exact_bits
from .._host_storage import declared_integer_storage_bytes
from .._strict import StrictModule
from ..discretization._cell_complex import PolyhedralConnectivity
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
    _OverlapStorageBounds,
    convex_cell_overlap,
    convex_cells_disjoint,
    exact_convex_cells,
    exact_vertices,
    ExactConvexCell,
    ExactOverlapRefusal,
    ExactTetrahedronOverlap,
    star_determinant,
    star_tetrahedra,
    tetrahedra_disjoint,
    Tetrahedron,
    tetrahedron_overlap,
    TetrahedronPlanePreparation,
)
from ._supermesh import (
    _admit_minimum_publication,
    _bvh_storage_bound,
    _RefinementStorage,
    CommonRefinementCoverage,
    CommonRefinementEvidence,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    Counts,
    Decomposition,
    Entries,
    PreparedCommonRefinement,
)


def _power_candidates(
    first: Decomposition, second: Decomposition, limit: int,
    storage: _RefinementStorage, /,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Preserve the exact path's touching-inclusive, source-first BVH schedule."""
    counts = first.cell_count, second.cell_count
    depth = sum(count.bit_length() for count in counts)
    node_block = min(1 << 16, (2 * counts[0] - 1) * (2 * counts[1] - 1))
    leaf_block = max(256, min(1 << 20, counts[0] * counts[1]))
    scratch = (
        sum(_bvh_storage_bound(count, 3) for count in counts)
        + 16384 + 1024 * depth
        + 8 * node_block * (8 * depth + 56)
        + 8 * leaf_block * 52
    )
    with storage.scratch("bvh", scratch):
        storage.set("candidates", 2048)
        empty = np.empty((0,), dtype=np.int64)
        if not all(counts):
            return empty, empty, False
        first_bvh = prepare_bvh(first.bbox_min, first.bbox_max, dtype=np.float64)
        second_bvh = prepare_bvh(second.bbox_min, second.bbox_max, dtype=np.float64)
        firsts, seconds, total = [empty], [empty], 0
        exceeded = False
        for first_rows, second_rows in bvh_overlap_pair_blocks(
            first_bvh, second_bvh, include_touching=True
        ):
            if total + first_rows.size > limit:
                exceeded = True
                break
            total += first_rows.size
            storage.set("candidates", 2048 + 256 * total + 1024 * len(firsts))
            firsts.append(first_rows)
            seconds.append(second_rows)
        return np.concatenate(firsts), np.concatenate(seconds), exceeded



def _rational_storage(count: int, bits: int, /) -> int:
    """Bound Fraction headers, both integers, and owning collection slots."""
    return count * (320 + 2 * declared_integer_storage_bytes(bits))


def _decomposition_bound(
    mesh: CellMesh, points: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> int:
    """Bound the actual packed face loops, stars, and their rational integrals.

    A face occurrence contributes m-2 star tetrahedra. A vertex mean sums at
    most n input denominators; differences and the determinant use eighteen
    such operands. The moment adds four coordinates to that determinant.
    Accumulation multiplies denominators once per star, not per policy cap.
    Plane construction uses the existing plane_key's 51 operand bound.
    """
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("exact power refinement requires polyhedral connectivity")
    bits = exact_bits(points, budget=budget)
    vertex_offsets = np.asarray(connectivity.cell_vertex_offsets)
    offsets = np.asarray(connectivity.cell_face_offsets)
    face_offsets = np.asarray(connectivity.face_vertex_offsets)
    faces = np.asarray(connectivity.cell_face_values)
    total = 4096 + 16 * points.size
    for cell in range(connectivity.cell_count):
        vertices = int(vertex_offsets[cell + 1] - vertex_offsets[cell])
        occurrences = int(offsets[cell + 1] - offsets[cell])
        corners, stars = 0, 0
        for occurrence in range(int(offsets[cell]), int(offsets[cell + 1])):
            face = int(faces[occurrence])
            count = int(face_offsets[face + 1] - face_offsets[face])
            corners += count
            stars += max(0, count - 2)
        center = vertices * (bits + 1) + vertices.bit_length() + 2
        determinant = 18 * (2 * center + 1) + 18
        integral = stars * (determinant + 4 * center + 16) + 1
        plane = 51 * (bits + 1)
        # Lists/tuples, face-loop maps, orientation copies, cell_rows, float
        # exports and their simultaneous NumPy conversion inputs.
        total += (
            4096 + 1024 * (vertices + corners + occurrences + stars)
            + _rational_storage(3, center)
            + _rational_storage(stars, determinant)
            + _rational_storage(8, integral)
            + _rational_storage(8 * occurrences + 7 * vertices, plane)
            + _rational_storage(vertices, 4 * plane + 3 * bits + 4)
        )
    return total


def _overlap_bound(
    first: Tetrahedron | ExactConvexCell,
    second: Tetrahedron | ExactConvexCell,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _OverlapStorageBounds:
    """Profile only operand widths; live helper banks admit their own stages.

    The old bound multiplied all possible edge/plane candidates into a vertex
    mean and every integral, then reserved that result twice (pair and scratch).
    Only discovered, retained intersection vertices enter those expressions.
    Their banks grow before insertion; side rows and face primitives have
    separate, explicit lifetimes in the exact overlap kernel.
    """
    if isinstance(first, ExactConvexCell):
        assert isinstance(second, ExactConvexCell)
        bits = exact_bits(
            np.asarray((*first.vertices, *second.vertices), dtype=object), budget=budget
        )
        plane_bits = exact_bits(
            np.asarray((*first.planes, *second.planes), dtype=object), budget=budget
        )
    else:
        assert not isinstance(second, ExactConvexCell)
        bits = exact_bits(np.asarray((*first, *second), dtype=object), budget=budget)
        plane_bits = 51 * (bits + 1)
    return _OverlapStorageBounds(bits, plane_bits)


def _rne_error(value: Fraction, /) -> float:
    error = abs(value - Fraction(float(value)))
    return 0.0 if not error else float(np.nextafter(float(error), math.inf))


def _cell_integrals(
    stars: tuple[tuple[Tetrahedron, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]]:
    measures, moments = [], []
    for star in stars:
        volume, moment = Fraction(0), [Fraction(0)] * 3
        for tet in star:
            measure = star_determinant(tet, budget=budget) / 6
            # The volume sum and three scaled four-corner moment sums.
            budget.reserve(2 + 3 * 6)
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
    convex: tuple[ExactConvexCell | None, ...]


class _PowerGeometries(StrictModule):
    source: CellGeometrySpec
    target: CellGeometrySpec


def _decomposition(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    storage: CoordinateLiveStorage,
) -> _PowerDecomposition:
    # Source caches have their canonical retained owners; all other arithmetic
    # scratch ends here, while the returned decomposition keeps its live credit.
    with budget.temporary_scope():
        return _decomposition_impl(mesh, geometry, budget=budget, storage=storage)


def _decomposition_impl(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    storage: CoordinateLiveStorage,
) -> _PowerDecomposition:
    # The canonical source preparation owns its rational bank. This admission
    # precedes the object-array view/copy made by exact_vertices.
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("exact power refinement requires polyhedral connectivity")
    topology_items = sum(
        value.size for value in (
            connectivity.cell_vertex_offsets, connectivity.cell_vertex_values,
            connectivity.cell_face_offsets, connectivity.cell_face_values,
            connectivity.cell_face_sign_values, connectivity.face_vertex_offsets,
            connectivity.face_vertex_values,
        )
    )
    topology_items += sum(
        block.vertices.size + block.vertex_valid.size + block.global_ids.size
        for block in mesh.blocks
    )
    storage.set_bound(
        storage.bound + 4096 + 32 * mesh.coordinates.size + 32 * topology_items
    )
    points = exact_vertices(mesh, geometry, budget=budget)
    storage.set_bound(storage.bound + _decomposition_bound(mesh, points, budget=budget))
    stars = star_tetrahedra(mesh, points, budget=budget)
    measures, moments = _cell_integrals(stars, budget=budget)
    # Positive stars about each vertex mean make supporting faces a hull proof.
    convex = exact_convex_cells(mesh, points, budget=budget)
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
    return _PowerDecomposition(part, stars, measures, convex)


class _PowerRefusal(StrEnum):
    NONE = ""
    DECOMPOSITION = (
        "Exact power decomposition or candidates exceed their resource bounds."
    )
    WORK = "Exact power overlap coefficient work exhausted maximum_exact_work."
    RATIONAL_STORAGE = (
        "Exact power overlap rational working storage exhausted maximum_memory_bytes."
    )
    PUBLICATION = "Exact power overlap publication exhausted maximum_memory_bytes."
    PAIRS = "Exact power overlap retained pairs exhausted maximum_accepted_pairs."
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
    first_convex: ExactConvexCell | None,
    second_convex: ExactConvexCell | None,
    policy: CommonRefinementPolicy,
    work: int,
    working_bytes: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparation: TetrahedronPlanePreparation,
    storage: CoordinateLiveStorage,
    workspace: CoordinateLiveStorage,
) -> _PowerPair:
    """Integrate one candidate cell pair.

    Two verified convex cells intersect in one convex polytope, integrated
    directly; otherwise every star tetrahedron pair is integrated.
    """
    storage.set_bound(4096)
    volume, moment = Fraction(0), [Fraction(0)] * 3
    second_moment = (
        [[Fraction(0)] * 3 for _ in range(3)] if policy.second_moments else None
    )
    pieces: list[Tetrahedron] = []
    consumed, piece_pairs = 0, 0
    refusal = _PowerRefusal.NONE
    convex = first_convex is not None and second_convex is not None
    pairs = (
        ((first_convex, second_convex),)
        if convex
        else ((first_tet, second_tet) for first_tet in first for second_tet in second)
    )
    for first_piece, second_piece in pairs:
        piece_pairs += 1
        if isinstance(first_piece, ExactConvexCell) and isinstance(second_piece, ExactConvexCell):
            disjoint = convex_cells_disjoint(first_piece, second_piece)
        elif isinstance(first_piece, tuple) and isinstance(second_piece, tuple):
            disjoint = tetrahedra_disjoint(first_piece, second_piece)
        else:
            raise RuntimeError("Power overlap pieces must use the same exact cell representation.")
        if disjoint:
            continue
        prior_storage = storage.bound
        try:
            with budget.temporary_scope():
                budget.reserve(0, 4096 + 128 * (
                    len(first_piece.vertices) + len(second_piece.vertices)
                    + len(first_piece.planes) + len(second_piece.planes)
                    if isinstance(first_piece, ExactConvexCell)
                    and isinstance(second_piece, ExactConvexCell) else 16
                ))
                bounds = _overlap_bound(first_piece, second_piece, budget=budget)
        except CoordinateEnclosureResourceError as error:
            if error.resource == "coefficient_work" or error.limit != policy.maximum_memory_bytes:
                raise
            storage.set_bound(prior_storage)
            workspace.set_bound(0)
            refusal = _PowerRefusal.RATIONAL_STORAGE
            break
        # Each helper preadmits its actual live stages on this same ledger.
        # The pair owner directly retains its returned Fraction/recipe bank.
        if isinstance(first_piece, ExactConvexCell) and isinstance(second_piece, ExactConvexCell):
            overlap = convex_cell_overlap(
                first_piece,
                second_piece,
                second_moments=policy.second_moments,
                retain_simplices=policy.overlap_simplices,
                maximum_work=policy.maximum_exact_work - work - consumed,
                budget=budget,
                _storage_bounds=bounds,
                _output_storage=storage,
                _workspace=workspace,
            )
        elif isinstance(first_piece, tuple) and isinstance(second_piece, tuple):
            overlap = tetrahedron_overlap(
                first_piece,
                second_piece,
                second_moments=policy.second_moments,
                retain_simplices=policy.overlap_simplices,
                maximum_work=policy.maximum_exact_work - work - consumed,
                budget=budget,
                preparation=preparation,
                _storage_bounds=bounds,
                _output_storage=storage,
                _workspace=workspace,
            )
        else:
            raise RuntimeError("Power overlap pieces must use the same exact cell representation.")
        consumed += overlap.operation_count
        working_bytes = max(working_bytes, budget.peak_bytes_upper)
        match overlap:
            case ExactOverlapRefusal():
                refusal = (
                    _PowerRefusal.WORK if overlap.resource == "coefficient_work"
                    else _PowerRefusal.RATIONAL_STORAGE
                )
                break
            case ExactTetrahedronOverlap():
                pass
            case invalid:
                assert_never(invalid)
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
        del overlap
        workspace.set_bound(0)
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
    *,
    budget: CoordinateEnclosureBudget,
) -> _PowerGeometries:
    source_geometry = (
        CellGeometrySpec.affine(source_mesh, budget=budget)
        if source_geometry is None
        else source_geometry
    )
    target_geometry = (
        CellGeometrySpec.affine(target_mesh, budget=budget)
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
    """Admit exact power storage on the existing request, retaining real outputs.

    Helper scratch and rational pair accumulators are separate live owners;
    only the published scientific graph survives preparation. The canonical
    host profiler counts ordinary NumPy payloads and excludes native-backed
    payloads already owned by the native request.
    """
    with (
        budget.bound_stage(budget.maximum_work_units, policy.maximum_memory_bytes),
        budget.temporary_scope(),
        budget.live_storage() as storage,
        budget.live_storage() as pair_storage,
        budget.live_storage() as workspace,
        budget.live_storage() as candidate_storage,
    ):
        prepared = _prepare_exact_power_refinement(
            source_mesh, target_mesh, source_geometry, target_geometry, policy,
            budget=budget, storage=storage, pair_storage=pair_storage,
            workspace=workspace,
            candidate_storage=candidate_storage,
        )
        # The callee's rational recipes and candidate stream have no remaining
        # consumers. Only final scientific arrays now cross the handover.
        pair_storage.set_bound(0)
        workspace.set_bound(0)
        candidate_storage.set_bound(0)
        budget.retain_owner(prepared, temporary_storage_credit=storage)
        return prepared


@checked
def _prepare_exact_power_refinement(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    source_geometry: CellGeometrySpec | None,
    target_geometry: CellGeometrySpec | None,
    policy: CommonRefinementPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    storage: CoordinateLiveStorage,
    pair_storage: CoordinateLiveStorage,
    workspace: CoordinateLiveStorage,
    candidate_storage: CoordinateLiveStorage,
) -> PreparedCommonRefinement:
    """Prepare rational power-cell overlaps and their coverage evidence."""
    geometries = _power_geometries(
        source_mesh, target_mesh, source_geometry, target_geometry, budget=budget
    )
    source_geometry, target_geometry = geometries.source, geometries.target
    _admit_minimum_publication(source_mesh, target_mesh, policy, budget)
    candidates_storage = _RefinementStorage(budget, candidate_storage)
    source = _decomposition(source_mesh, source_geometry, budget=budget, storage=storage)
    target = _decomposition(target_mesh, target_geometry, budget=budget, storage=storage)
    first, first_stars, first_measures = source.carrier, source.stars, source.measures
    second, second_stars, second_measures = target.carrier, target.stars, target.measures
    source_convex, target_convex = source.convex, target.convex
    identities = (
        source_mesh.mesh_id,
        target_mesh.mesh_id,
        source_mesh.topology_id,
        target_mesh.topology_id,
    )
    geometry_ids = (cell_geometry_id(source_geometry), cell_geometry_id(target_geometry))
    base_bytes = storage.bound
    try:
        first_cells, second_cells, exceeded = _power_candidates(
            first, second, policy.maximum_candidate_pairs, candidates_storage
        )
        storage.set_bound(storage.bound + 512 + 256 * len(first_cells))
        candidates = tuple(
            sorted(zip(second_cells.tolist(), first_cells.tolist(), strict=True))
        )
    except CoordinateEnclosureResourceError as error:
        if error.resource == "coefficient_work" or error.limit != policy.maximum_memory_bytes:
            raise
        # Both exact decompositions exist; their admitted banks also cover
        # truthful zero-attempt coverage evidence and the empty entry schema.
        candidates, exceeded = (), True
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
            source_convex[source],
            target_convex[target],
            policy,
            work,
            working_bytes,
            budget=budget,
            preparation=preparation,
            storage=pair_storage,
            workspace=workspace,
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
        # Records, growing exact coverage sums, their defects, conversion
        # tuples and final float exports coexist until publication.
        addition = 4 * pair_storage.bound + 4096 + len(pieces) * 4 * 3 * 8
        try:
            storage.set_bound(storage.bound + addition)
        except CoordinateEnclosureResourceError as error:
            if error.resource == "coefficient_work" or error.limit != policy.maximum_memory_bytes:
                raise
            status, reason = CommonRefinementStatus.RESOURCE_LIMIT, _PowerRefusal.PUBLICATION
            del pair, volume, moment, second_moment, pieces
            pair_storage.set_bound(0)
            workspace.set_bound(0)
            break
        retained_bytes += addition
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
        del pair, volume, moment, second_moment, pieces
        pair_storage.set_bound(0)
        workspace.set_bound(0)
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
