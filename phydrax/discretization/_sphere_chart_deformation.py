#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"Prepare projective correspondence between physical sphere material cells.\n\nNative arbitrary-integer halfplane clipping consumes source rational rows\nafter positive denominator clearing. source supporting-plane pairs\nreconstruct every retained vertex exactly; boundary/feasibility checks and\nwhole rational partitions of BOTH physical reference meshes decide\nacceptance. Projective target reference maps retain their quotients."

from __future__ import annotations

import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._rational import fraction_endpoint

from .. import _geometry_predicates as imported_geometry_predicates, _meshcore
from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeBatchTicket, NativeExecutionScope
from .._strict import StrictModule
from .._validation import positive_integer
from ..geometry import (
    _domain_numerics as domain_numerics,
    _meshing_domain as imported_geometry__meshing_domain,
    _sphere_material_atlas as imported_geometry__sphere_material_atlas,
)
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..geometry._interval_enclosure import interval_divide
from ..geometry._sphere_material_atlas import (
    SphereMaterialCellAtlas,
    SphereProjectiveReferenceMap,
    SphereProjectiveTriangleBounds,
)
from ..linalg import _hermitian_spectral as imported_linalg__hermitian_spectral
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ..typing import checked, ConvertibleToArray
from . import (
    _cell_geometry_transfer as imported_cell_geometry_transfer,
    _chart_execution as chart_execution,
    _chart_geometry as chart_geometry,
    _coordinate_enclosure as imported_coordinate_enclosure,
    _sphere_chart_records as sphere_chart_records,
)
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_transition_types import CellGeometryTransitionPolicy
from ._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ._cell_mesh import CellMesh
from ._chart_certification import certify_chart_endpoint
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import Expression, ExpressionComposition
from ._exact_algebraic_geometry import rational_coefficient


type _Point = tuple[Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]
type _Plane = tuple[Fraction, Fraction, Fraction]
type _FloatArray = NDArray[np.float64]
type _IntArray = NDArray[np.int64]


def _tokens(vertices: tuple[_Point, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple((value.numerator, value.denominator) for value in point)
        for point in vertices
    )


def _direction_boxes(atlas: SphereMaterialCellAtlas) -> tuple[_FloatArray, _FloatArray]:
    directions = np.asarray(atlas.directions, dtype=np.float64)
    low, high = interval_divide(
        (np.min(directions, axis=1), np.max(directions, axis=1)),
        (
            np.asarray(atlas.hemisphere_lower)[:, None],
            np.asarray(atlas.direction_norm_upper)[:, None],
        ),
    )
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


@checked
def cone_candidates(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    source_rows: _IntArray,
    target_rows: _IntArray,
    memory: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_IntArray, _IntArray]:
    """Discard only certified halfspace-separated or zero-area cone pairs."""

    budget.reserve(0, source_rows.size)
    keep = np.ones(source_rows.shape, dtype=np.bool_)
    banks = (np.asarray(source.directions), np.asarray(target.directions))
    orientations = tuple(
        np.asarray(
            [1 if value > 0 else -1 for value in atlas.exact_determinants],
            dtype=np.int8,
        )
        for atlas in (source, target)
    )
    chunk = min(4096, memory // 2048)
    if chunk < 1:
        raise ValueError(
            "sphere_chart.cone_candidates.budget must satisfy the admitted resource limits; got Sphere cone candidate filtering exceeds its source resident memory limit."
        )
    for start in range(0, source_rows.size, chunk):
        stop = min(start + chunk, source_rows.size)
        with budget.temporary_scope():
            budget.reserve(0, (stop - start) * 2048)
            rows = (source_rows[start:stop], target_rows[start:stop])
            rays = (banks[0][rows[0]], banks[1][rows[1]])
            for owner, other in ((0, 1), (1, 0)):
                predicate = imported_geometry_predicates.orient3d(
                    np.zeros((3,), dtype=np.float64),
                    rays[owner][:, (0, 1, 2), None, :],
                    rays[owner][:, (1, 2, 0), None, :],
                    rays[other][:, None, :, :],
                    mode=imported_geometry_predicates.PredicateMode.EXACT,
                    budget=budget,
                )
                signs = (
                    np.asarray(predicate.signs)
                    * orientations[owner][rows[owner], None, None]
                )
                # Resolve uncertain filtered signs before forming projective
                # coefficient maps. A closed exterior halfspace admits only
                # zero-area contact, which cannot contribute reference coverage.
                separated = np.any(
                    np.all(np.asarray(predicate.certain) & (signs <= 0), axis=-1), axis=-1
                )
                keep[start:stop] &= ~separated
    budget.reserve(0, source_rows.size * np.dtype(np.int64).itemsize)
    kept = np.flatnonzero(keep)
    budget.reserve(0, 2 * kept.size * np.dtype(np.int64).itemsize)
    return source_rows[kept], target_rows[kept]


def _candidates(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    limit: int,
    memory: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_IntArray, _IntArray]:
    source_low, source_high = _direction_boxes(source)
    target_low, target_high = _direction_boxes(target)
    source_bvh = prepare_bvh(source_low, source_high, dtype=jnp.float64)
    target_bvh = prepare_bvh(target_low, target_high, dtype=jnp.float64)
    targets: list[_IntArray] = []
    sources: list[_IntArray] = []
    count = 0
    for target_rows, source_rows in bvh_overlap_pair_blocks(target_bvh, source_bvh):
        count += target_rows.size
        if count > limit or count * 32 > memory:
            raise ValueError(
                "sphere_chart.candidates.budget must satisfy the admitted resource limits; got Sphere native candidate pairs exceed their work or resident memory limit."
            )
        targets.append(np.asarray(target_rows, dtype=np.int64))
        sources.append(np.asarray(source_rows, dtype=np.int64))
    if not sources:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    source_rows, target_rows = np.concatenate(sources), np.concatenate(targets)
    order = np.lexsort((source_rows, target_rows))
    return cone_candidates(
        source, target, source_rows[order], target_rows[order], memory, budget=budget
    )


def _planes(mapping: SphereProjectiveReferenceMap) -> tuple[_Plane, ...]:
    return (
        (Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
    ) + tuple((-row[1], -row[2], row[0]) for row in mapping.exact_coefficients)


def _support(label: int, planes: tuple[_Plane, ...]) -> _Plane:
    if label < 0 or label >= len(planes):
        raise ValueError(
            "sphere_chart.support.pieces must satisfy the material reference contract; got Native polygon names an unknown source rational halfplane."
        )
    return planes[label]


def _native_polygon(
    mapping: SphereProjectiveReferenceMap,
    memory: int,
    maximum_work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[_Point, ...], tuple[int, ...], int]:
    planes = _planes(mapping)

    supports, labels, work = _meshcore.clip_reference_triangle_exact(
        planes,
        maximum_scratch_bytes=memory,
        maximum_work_units=maximum_work,
    )
    count = supports.shape[0]
    if count < 3:
        return (), (), work
    edge_labels = tuple(int(value) for value in labels)
    vertices: list[_Point] = []
    source_corners = {
        (0, 1): (Fraction(0), Fraction(0)),
        (0, 2): (Fraction(0), Fraction(1)),
        (1, 2): (Fraction(1), Fraction(0)),
    }
    for slot in range(count):
        first_label, second_label = map(int, supports[slot])
        first = _support(first_label, planes)
        second = _support(second_label, planes)
        # Native support identities, not coordinate coincidence, name the
        # already-authored corners of the source reference triangle.
        point = source_corners.get(
            (min(first_label, second_label), max(first_label, second_label))
        )
        if point is None:
            with budget.temporary_scope():
                construction = prepare_exact_small_linear_actions(
                    ((first[0], first[1]), (second[0], second[1])),
                    ((first[2],), (second[2],)),
                    budget=budget,
                )
            work += construction.operation_count
            if construction.actions is None:
                raise ValueError(
                    "sphere_chart.native_polygon.reference_map must satisfy the regular reference-map contract; got Native boundary cannot reconstruct a nonsingular source rational vertex."
                )
            point = construction.actions[0][0], construction.actions[1][0]
        if any(
            normal[0] * point[0] + normal[1] * point[1] > normal[2] for normal in planes
        ) or any(value < 0 or value > 1 for value in point):
            raise ValueError(
                "sphere_chart.native_polygon.pieces must satisfy the material reference contract; got Native topology fails source rational halfplane feasibility."
            )
        vertices.append(point)
    polygon = tuple(vertices)
    if len(set(polygon)) != len(polygon) or chart_geometry.area(polygon) <= 0:
        raise ValueError(
            "sphere_chart.native_polygon.pieces must satisfy the material reference contract; got Native polygon fails positive distinct-vertex construction."
        )
    for slot in range(count):
        first, second, third = (
            polygon[slot - 1],
            polygon[slot],
            polygon[(slot + 1) % count],
        )
        if (second[0] - first[0]) * (third[1] - second[1]) - (second[1] - first[1]) * (
            third[0] - second[0]
        ) <= 0:
            raise ValueError(
                "sphere_chart.native_polygon.pieces must satisfy the material reference contract; got Native polygon is not the strictly convex source rational intersection."
            )
        support = _support(edge_labels[slot], planes)
        if any(
            support[0] * point[0] + support[1] * point[1] != support[2]
            for point in (second, third)
        ):
            raise ValueError(
                "sphere_chart.native_polygon.pieces must satisfy the material reference contract; got Native edge does not retain its source rational boundary support."
            )
    return polygon, edge_labels, work


class _SphereEndpointCertificates(StrictModule):
    validities: list[CellValidityCertificate]
    embeddings: list[GlobalEmbeddingCertificate]
    measures: list[_FloatArray]
    errors: list[_FloatArray]


def _certify_sphere_endpoints(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None,
    prepared_target_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None,
    validity_policy: CellValidityPolicy | None,
    certificate_limits: MeshCertificateLimits | None,
    candidates_limit: int,
    work_limit: int,
    memory_limit: int,
    measure_absolute_tolerance: float,
    measure_relative_tolerance: float,
    maximum_measure_work: int,
    maximum_measure_subcells: int,
    maximum_binomial_terms: int,
    budget: CoordinateEnclosureBudget,
) -> _SphereEndpointCertificates:
    "Certify sphere endpoints and enclose their physical cell measures."
    validity_policy = (
        CellValidityPolicy(
            maximum_piece_count=min(work_limit, memory_limit // 1024),
            maximum_bernstein_nodes=min(work_limit, memory_limit // 1024),
        )
        if validity_policy is None
        else validity_policy
    )
    limits = (
        MeshCertificateLimits(
            maximum_candidate_pairs=candidates_limit,
            maximum_source_samples=work_limit,
            maximum_distance_evaluations=work_limit,
            maximum_subdivision_pieces=min(work_limit, memory_limit // 1024),
            maximum_bernstein_nodes=min(work_limit, memory_limit // 1024),
        )
        if certificate_limits is None
        else certificate_limits
    )
    validities: list[CellValidityCertificate] = []
    embeddings: list[GlobalEmbeddingCertificate] = []
    measures: list[_FloatArray] = []
    errors: list[_FloatArray] = []
    with budget.activate():
        for atlas, prepared in (
            (source_atlas, prepared_source_certificates),
            (target_atlas, prepared_target_certificates),
        ):
            _endpoint = certify_chart_endpoint(
                atlas.source_mesh,
                atlas.source_geometry,
                validity_policy,
                limits,
                prepared,
                budget=budget,
                stale_message="sphere_chart.certify_sphere_endpoints.pieces must satisfy the material reference contract; got Prepared sphere certificates change the source proof controls.",
            )
            validity, embedding = _endpoint.validity, _endpoint.embedding
            if not validity.all_certified or embedding.status != "certified":
                raise ValueError(
                    "sphere_chart.certify_sphere_endpoints.validity must satisfy the positive cell validity contract; got Sphere deformation requires independent source/target validity and global embedding."
                )
            values, bounds, _ = mapped_measure_owner.certified_cell_measures(
                atlas.source_mesh,
                atlas.source_geometry,
                absolute_tolerance=measure_absolute_tolerance,
                relative_tolerance=measure_relative_tolerance,
                maximum_work=maximum_measure_work,
                maximum_subcells=maximum_measure_subcells,
                maximum_binomial_terms=maximum_binomial_terms,
                budget=budget,
            )
            if (
                np.any(~np.isfinite(values))
                or np.any(~np.isfinite(bounds))
                or np.any(bounds < 0)
                or np.any(values <= bounds)
            ):
                raise ValueError(
                    "sphere_chart.certify_sphere_endpoints.measures must satisfy the positive physical measure contract; got Sphere physical cell measures require quantitative strictly positive enclosures."
                )
            validities.append(validity)
            embeddings.append(embedding)
            measures.append(values)
            errors.append(bounds)
    return _SphereEndpointCertificates(
        validities=validities, embeddings=embeddings, measures=measures, errors=errors
    )


class _SphereMaterialOverlaps(StrictModule):
    pieces: list[sphere_chart_records.PreparedSphereChartPiece]
    source_coverage: list[Fraction]
    target_coverage: list[Fraction]
    maximum_bound: float
    retained: int
    work: int


class _SphereOverlapPublication(StrictModule):
    retained: int


def _publish_sphere_overlap_piece(
    vertices: _Triangle,
    target_vertices: _Triangle,
    mapping: SphereProjectiveReferenceMap,
    bounds: SphereProjectiveTriangleBounds,
    source_area: Fraction,
    target_area: Fraction,
    retained: int,
    memory_limit: int,
    pieces_limit: int,
    pieces: list[sphere_chart_records.PreparedSphereChartPiece],
    clip_id: str,
    slot: int,
    source_id: int,
    target_id: int,
    physical_source: int,
    physical_target: int,
    source_row: int,
    target_row: int,
    group: tuple[str, tuple[str, ...]],
    labels: tuple[int, ...],
    displacement: float,
    budget: CoordinateEnclosureBudget,
) -> _SphereOverlapPublication:
    "Retain one projective overlap artifact under work and storage limits."
    if source_area <= 0 or target_area <= 0:
        raise ValueError(
            "sphere_chart.publish_sphere_overlap_piece.pieces must satisfy the material reference contract; got Sphere overlap triangle must have positive source and target reference area."
        )
    fraction_bytes = sum(
        (value.numerator.bit_length() + value.denominator.bit_length() + 7) // 8 + 128
        for point in (*vertices, *target_vertices)
        for value in point
    )
    retained += 2048 + fraction_bytes
    if retained > memory_limit or len(pieces) >= pieces_limit:
        raise ValueError(
            "sphere_chart.publish_sphere_overlap_piece.budget must satisfy the admitted resource limits; got Sphere retained overlap artifacts exceed memory or piece limits."
        )
    piece_id = canonical_fingerprint(
        {
            "kind": "sphere-exact-reference-piece",
            "clip": clip_id,
            "source": _tokens(vertices),
            "target": _tokens(target_vertices),
            "indices": (0, slot, slot + 1),
        }
    )
    budget.retain_basis(
        (
            vertices,
            target_vertices,
            mapping.exact_coefficients,
            mapping.exact_denominator,
            mapping.orientation_ratio,
            bounds.denominator_lower,
            bounds.denominator_upper,
            bounds.jacobian_lower,
            bounds.jacobian_upper,
            source_area,
            target_area,
        )
    )
    budget.reserve(2)
    pieces.append(
        sphere_chart_records.PreparedSphereChartPiece(
            source_cell_global_id=source_id,
            target_cell_global_id=target_id,
            source_row=physical_source,
            target_row=physical_target,
            source_atlas_row=source_row,
            target_atlas_row=target_row,
            geometry_entity_id=group[0],
            occurrence_path=group[1],
            projective_map=mapping,
            exact_source_reference_vertices=vertices,
            exact_target_reference_vertices=target_vertices,
            source_reference_vertices=jnp.asarray(
                sphere_chart_records.reference_representatives(vertices),
                dtype=jnp.float64,
            ),
            target_reference_vertices=jnp.asarray(
                sphere_chart_records.reference_representatives(target_vertices),
                dtype=jnp.float64,
            ),
            projective_bounds=bounds,
            source_reference_area=source_area,
            target_reference_area=target_area,
            native_boundary_labels=labels,
            native_polygon_vertex_indices=(0, slot, slot + 1),
            native_clip_id=clip_id,
            displacement_bound=displacement,
            piece_id=piece_id,
        )
    )
    return _SphereOverlapPublication(retained=retained)


def _sphere_material_overlaps(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    source_candidates: _IntArray,
    target_candidates: _IntArray,
    source_bounds: _FloatArray,
    target_bounds: _FloatArray,
    maximum_displacement: float,
    pieces_limit: int,
    work_limit: int,
    memory_limit: int,
    retained: int,
    work: int,
    budget: CoordinateEnclosureBudget,
) -> _SphereMaterialOverlaps:
    "Clip sphere material candidates and certify both reference partitions."
    pieces: list[sphere_chart_records.PreparedSphereChartPiece] = []
    source_coverage = [Fraction(0) for _ in range(source_atlas.num_charts)]
    target_coverage = [Fraction(0) for _ in range(target_atlas.num_charts)]
    maximum_bound = 0.0
    coordinate_maps: (
        tuple[
            tuple[tuple[Expression, ...], ...],
            tuple[tuple[Expression, ...], ...],
        ]
        | None
    ) = None
    for source_index, target_index in zip(
        source_candidates, target_candidates, strict=True
    ):
        with budget.temporary_scope():
            source_row, target_row = int(source_index), int(target_index)
            group = (
                source_atlas.geometry_entity_ids[source_row],
                source_atlas.occurrence_paths[source_row],
            )
            if group != (
                target_atlas.geometry_entity_ids[target_row],
                target_atlas.occurrence_paths[target_row],
            ):
                continue
            source_id = int(np.asarray(source_atlas.cell_global_ids)[source_row])
            target_id = int(np.asarray(target_atlas.cell_global_ids)[target_row])
            mapping = source_atlas.projective_reference_map(
                source_id, target_atlas, target_id, budget=budget
            )
            if work_limit - work - 64 <= 0:
                raise ValueError(
                    "sphere_chart.sphere_material_overlaps.budget must satisfy the admitted resource limits; got Sphere correspondence exhausts work before native clipping."
                )
            polygon, labels, clipping_work = _native_polygon(
                mapping,
                memory_limit - retained,
                work_limit - work - 64,
                budget=budget,
            )
            work += clipping_work + 64
            if work > work_limit:
                raise ValueError(
                    "sphere_chart.sphere_material_overlaps.budget must satisfy the admitted resource limits; got Sphere clipping/construction exceeds its declared work budget."
                )
            if not polygon:
                continue
            clip_id = canonical_fingerprint(
                {
                    "kind": "native-sphere-exact-boundary",
                    "map": mapping.map_id,
                    "vertices": _tokens(polygon),
                    "support_labels": labels,
                }
            )
            physical_source = int(np.asarray(source_atlas.physical_rows)[source_row])
            physical_target = int(np.asarray(target_atlas.physical_rows)[target_row])
            # Both coordinate carriers approximate the same radial
            # point selected by the projective map. Their certified
            # source-fidelity radii therefore give a whole triangle-inequality
            # displacement bound without expanding a removable rational
            # difference. Fall back to the coefficient proof only when that
            # independently earned bound is too coarse for this policy.
            budget.reserve(1, 512)
            displacement = fraction_endpoint(
                Fraction(float(source_bounds[physical_source]))
                + Fraction(float(target_bounds[physical_target])),
                upper=True,
                saturate=True,
            )
            displacement_map: tuple[Expression, ...] | None = None
            if displacement > maximum_displacement:
                if coordinate_maps is None:
                    coordinate_maps = (
                        _sphere_full_coordinate_expressions(source_atlas, budget=budget),
                        _sphere_full_coordinate_expressions(target_atlas, budget=budget),
                    )
                displacement_map = _sphere_displacement_map(
                    coordinate_maps[0][physical_source],
                    coordinate_maps[1][physical_target],
                    mapping,
                    budget=budget,
                )
            for slot in range(1, len(polygon) - 1):
                vertices: _Triangle = polygon[0], polygon[slot], polygon[slot + 1]
                bounds, target_vertices, source_area, target_area = (
                    _sphere_piece_geometry(mapping, vertices, budget=budget)
                )
                if displacement_map is not None:
                    displacement = sphere_full_map_displacement(
                        displacement_map, vertices, budget=budget
                    )
                maximum_bound = max(maximum_bound, displacement)
                if displacement > maximum_displacement:
                    raise ValueError(
                        "sphere_chart.sphere_material_overlaps.displacement must satisfy the whole-piece displacement bound; got full-coordinate sphere deformation exceeds its whole-piece displacement bound."
                    )
                _phase = _publish_sphere_overlap_piece(
                    vertices=vertices,
                    target_vertices=target_vertices,
                    mapping=mapping,
                    bounds=bounds,
                    source_area=source_area,
                    target_area=target_area,
                    retained=retained,
                    memory_limit=memory_limit,
                    pieces_limit=pieces_limit,
                    pieces=pieces,
                    clip_id=clip_id,
                    slot=slot,
                    source_id=source_id,
                    target_id=target_id,
                    physical_source=physical_source,
                    physical_target=physical_target,
                    source_row=source_row,
                    target_row=target_row,
                    group=group,
                    labels=labels,
                    displacement=displacement,
                    budget=budget,
                )
                retained = _phase.retained
                source_coverage[source_row] += source_area
                target_coverage[target_row] += target_area
    return _SphereMaterialOverlaps(
        pieces=pieces,
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        maximum_bound=maximum_bound,
        retained=retained,
        work=work,
    )


class _SphereOccurrences(StrictModule):
    occurrences: list[sphere_chart_records.PreparedSphereChartOccurrence]


def _group_sphere_occurrences(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    source_groups: set[tuple[str, tuple[str, ...]]],
    pieces: list[sphere_chart_records.PreparedSphereChartPiece],
) -> _SphereOccurrences:
    "Group overlap pieces without merging scientific source occurrences."
    occurrences: list[sphere_chart_records.PreparedSphereChartOccurrence] = []
    for entity, path in sorted(source_groups):
        indices = tuple(
            index
            for index, piece in enumerate(pieces)
            if (piece.geometry_entity_id, piece.occurrence_path) == (entity, path)
        )
        source_ids = tuple(
            int(identifier)
            for identifier, group in zip(
                np.asarray(source_atlas.cell_global_ids),
                zip(
                    source_atlas.geometry_entity_ids,
                    source_atlas.occurrence_paths,
                    strict=True,
                ),
                strict=True,
            )
            if group == (entity, path)
        )
        target_ids = tuple(
            int(identifier)
            for identifier, group in zip(
                np.asarray(target_atlas.cell_global_ids),
                zip(
                    target_atlas.geometry_entity_ids,
                    target_atlas.occurrence_paths,
                    strict=True,
                ),
                strict=True,
            )
            if group == (entity, path)
        )
        occurrences.append(
            sphere_chart_records.PreparedSphereChartOccurrence(
                geometry_entity_id=entity,
                occurrence_path=path,
                piece_indices=indices,
                source_cell_global_ids=source_ids,
                target_cell_global_ids=target_ids,
                coverage_status="certified",
            )
        )
    return _SphereOccurrences(occurrences=occurrences)


# This bounded host coordinator retains admission, overlap, certification, and publication order; numerical phases already have owners.
def _prepare_sphere_chart_deformation(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    /,
    *,
    maximum_fidelity: float,
    maximum_displacement: float,
    maximum_candidate_pairs: int = 1_000_000,
    maximum_pieces: int = 1_000_000,
    maximum_work_units: int = 10_000_000,
    maximum_memory_bytes: int = 256 * 1024**2,
    measure_absolute_tolerance: float = 1e-10,
    measure_relative_tolerance: float = 1e-10,
    maximum_measure_work: int = 100_000_000,
    maximum_measure_subcells: int = 10_000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    prepared_target_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
) -> sphere_chart_records.PreparedSphereChartDeformation:
    "Certify closed material cells and retained projective overlap pieces."

    candidates_limit = positive_integer(
        maximum_candidate_pairs, "maximum_candidate_pairs"
    )
    pieces_limit = positive_integer(maximum_pieces, "maximum_pieces")
    work_limit = positive_integer(maximum_work_units, "maximum_work_units")
    memory_limit = positive_integer(maximum_memory_bytes, "maximum_memory_bytes")
    if any(
        not math.isfinite(value) or value < 0
        for value in (maximum_fidelity, maximum_displacement)
    ):
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.fidelity must satisfy the declared fidelity bound; got Sphere deformation fidelity/displacement limits must be finite and nonnegative."
        )
    for atlas in (source_atlas, target_atlas):
        atlas.require_bound(
            atlas.domain,
            atlas.source_mesh,
            atlas.source_geometry,
            atlas.coordinate_contract,
        )
        atlas.source_mesh.require_dense("Sphere material deformation")
    if (
        source_atlas.domain_id,
        source_atlas.source_id,
        source_atlas.source_revision,
        source_atlas.coordinate_contract.spatial_id,
    ) != (
        target_atlas.domain_id,
        target_atlas.source_id,
        target_atlas.source_revision,
        target_atlas.coordinate_contract.spatial_id,
    ):
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.binding must satisfy the declared geometry and cell identities; got Sphere source and target atlases must retain one revision, domain and spatial contract."
        )
    source_groups = set(
        zip(source_atlas.geometry_entity_ids, source_atlas.occurrence_paths, strict=True)
    )
    target_groups = set(
        zip(target_atlas.geometry_entity_ids, target_atlas.occurrence_paths, strict=True)
    )
    if source_groups != target_groups:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.pieces must satisfy the material reference contract; got Sphere correspondence cannot omit or merge source occurrences."
        )
    count = source_atlas.num_charts + target_atlas.num_charts
    budget.reserve(count * 64, count * 1024)
    retained = count * 1024
    work = count * 64
    if retained >= memory_limit or work > work_limit:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.pieces must satisfy the material reference contract; got Sphere correspondence exceeds its bounded preparation footprint."
        )
    source_bounds = sphere_chart_records.physical_order(
        source_atlas, source_atlas.source_fidelity_bounds
    )
    target_bounds = sphere_chart_records.physical_order(
        target_atlas, target_atlas.source_fidelity_bounds
    )
    if max(float(np.max(source_bounds)), float(np.max(target_bounds))) > maximum_fidelity:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.fidelity must satisfy the declared fidelity bound; got source or target polynomial coordinate fidelity exceeds the sphere bound."
        )
    source_candidates, target_candidates = _candidates(
        source_atlas,
        target_atlas,
        candidates_limit,
        memory_limit - retained,
        budget=budget,
    )
    retained += source_candidates.nbytes + target_candidates.nbytes
    if work + source_candidates.size * 64 > work_limit:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.budget must satisfy the admitted resource limits; got Sphere coefficient candidates exceed their preparation work budget."
        )
    _phase = _sphere_material_overlaps(
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        source_candidates=source_candidates,
        target_candidates=target_candidates,
        source_bounds=source_bounds,
        target_bounds=target_bounds,
        maximum_displacement=maximum_displacement,
        pieces_limit=pieces_limit,
        work_limit=work_limit,
        memory_limit=memory_limit,
        retained=retained,
        work=work,
        budget=budget,
    )
    pieces = _phase.pieces
    source_coverage = _phase.source_coverage
    target_coverage = _phase.target_coverage
    maximum_bound = _phase.maximum_bound
    retained = _phase.retained
    work = _phase.work
    if any(value != Fraction(1, 2) for value in (*source_coverage, *target_coverage)):
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.pieces must satisfy the material reference contract; got Native sphere correspondence has a source/target gap or double cover; whole rational reference coverage required."
        )
    _phase = _certify_sphere_endpoints(
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        prepared_source_certificates=prepared_source_certificates,
        prepared_target_certificates=prepared_target_certificates,
        validity_policy=validity_policy,
        certificate_limits=certificate_limits,
        candidates_limit=candidates_limit,
        work_limit=work_limit,
        memory_limit=memory_limit,
        measure_absolute_tolerance=measure_absolute_tolerance,
        measure_relative_tolerance=measure_relative_tolerance,
        maximum_measure_work=maximum_measure_work,
        maximum_measure_subcells=maximum_measure_subcells,
        maximum_binomial_terms=maximum_binomial_terms,
        budget=budget,
    )
    validities = _phase.validities
    embeddings = _phase.embeddings
    measures = _phase.measures
    errors = _phase.errors
    _phase = _group_sphere_occurrences(
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        source_groups=source_groups,
        pieces=pieces,
    )
    occurrences = _phase.occurrences
    identity = canonical_fingerprint(
        {
            "kind": "prepared-sphere-projective-deformation",
            "source": source_atlas.atlas_id,
            "target": target_atlas.atlas_id,
            "pieces": tuple(piece.piece_id for piece in pieces),
            "source_areas": _tokens(
                tuple((area, Fraction(0)) for area in source_coverage)
            ),
            "target_areas": _tokens(
                tuple((area, Fraction(0)) for area in target_coverage)
            ),
            "measure_values": array_tree_fingerprint(tuple(measures)),
            "measure_errors": array_tree_fingerprint(tuple(errors)),
            "embedding": tuple(value.certificate_id for value in embeddings),
        }
    )
    return sphere_chart_records.PreparedSphereChartDeformation(
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        source_mesh=source_atlas.source_mesh,
        target_mesh=target_atlas.source_mesh,
        source_geometry=source_atlas.source_geometry,
        target_geometry=target_atlas.source_geometry,
        source_geometry_id=source_atlas.source_geometry_id,
        target_geometry_id=target_atlas.source_geometry_id,
        source_topology_id=source_atlas.source_topology_id,
        target_topology_id=target_atlas.source_topology_id,
        domain_id=source_atlas.domain_id,
        pieces=tuple(pieces),
        occurrences=tuple(occurrences),
        source_reference_coverage=tuple(source_coverage),
        target_reference_coverage=tuple(target_coverage),
        source_fidelity_bounds=jnp.asarray(source_bounds, dtype=jnp.float64),
        target_fidelity_bounds=jnp.asarray(target_bounds, dtype=jnp.float64),
        source_cell_measures=jnp.asarray(measures[0], dtype=jnp.float64),
        target_cell_measures=jnp.asarray(measures[1], dtype=jnp.float64),
        source_measure_errors=jnp.asarray(errors[0], dtype=jnp.float64),
        target_measure_errors=jnp.asarray(errors[1], dtype=jnp.float64),
        source_validity=validities[0],
        target_validity=validities[1],
        source_embedding=embeddings[0],
        target_embedding=embeddings[1],
        maximum_displacement_bound=maximum_bound,
        candidate_pair_count=int(source_candidates.size),
        work_units=work,
        retained_bytes_upper=retained,
        deformation_id=identity,
    )


@checked
def prepare_sphere_chart_deformation(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    /,
    *,
    maximum_fidelity: float,
    maximum_displacement: float,
    maximum_candidate_pairs: int = 1_000_000,
    maximum_pieces: int = 1_000_000,
    maximum_work_units: int = 10_000_000,
    maximum_memory_bytes: int = 256 * 1024**2,
    measure_absolute_tolerance: float = 1e-10,
    measure_relative_tolerance: float = 1e-10,
    maximum_measure_work: int = 100_000_000,
    maximum_measure_subcells: int = 10_000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    prepared_target_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    scope: NativeExecutionScope | None = None,
) -> sphere_chart_records.PreparedSphereChartDeformation:
    """Prepare bounded chart correspondence with an explicit request ledger."""
    work = positive_integer(maximum_work_units, "maximum_work_units")
    memory = positive_integer(maximum_memory_bytes, "maximum_memory_bytes")
    with chart_execution.chart_native_execution(
        budget,
        scope,
        maximum_work=work,
        maximum_memory=memory,
        maximum_queries=work,
    ) as (budget, _native_ticket):
        return _prepare_sphere_chart_deformation(
            source_atlas,
            target_atlas,
            maximum_fidelity=maximum_fidelity,
            maximum_displacement=maximum_displacement,
            maximum_candidate_pairs=maximum_candidate_pairs,
            maximum_pieces=maximum_pieces,
            maximum_work_units=maximum_work_units,
            maximum_memory_bytes=maximum_memory_bytes,
            measure_absolute_tolerance=measure_absolute_tolerance,
            measure_relative_tolerance=measure_relative_tolerance,
            maximum_measure_work=maximum_measure_work,
            maximum_measure_subcells=maximum_measure_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
            certificate_limits=certificate_limits,
            validity_policy=validity_policy,
            prepared_source_certificates=prepared_source_certificates,
            prepared_target_certificates=prepared_target_certificates,
            budget=budget,
        )


__all__ = [
    "prepare_sphere_chart_deformation",
    "reconstruct_sphere_material_cell_geometry",
]


class _SphereRadialNodeEnclosure(StrictModule):
    error: float


def _sphere_radial_node_enclosure(
    source_atlas: SphereMaterialCellAtlas,
    domain: imported_geometry__meshing_domain.MeshingDomain,
    patch: int,
    slot: int,
    physical: _FloatArray,
    local: int,
    nodes: _FloatArray,
    rays: _FloatArray,
    directions: _FloatArray,
    row: int,
    budget: CoordinateEnclosureBudget,
) -> _SphereRadialNodeEnclosure:
    "Bound radial node realization and barycentric ray rounding errors."
    low, high = imported_geometry__sphere_material_atlas.source_image_enclosure(
        domain.patches[patch].surface, rays, budget=budget
    )
    point_error = max(
        sum(
            (
                max(
                    abs(
                        Fraction(float(physical[local, node, axis]))
                        - Fraction(float(low[node, axis]))
                    ),
                    abs(
                        Fraction(float(physical[local, node, axis]))
                        - Fraction(float(high[node, axis]))
                    ),
                )
                for axis in range(3)
            ),
            Fraction(0),
        )
        for node in range(nodes.shape[0])
    )
    # barycentric ray arithmetic separately encloses host dot-product
    # rounding; no fitted radial polynomial is used as the source.
    ray_error = Fraction(0)
    for node in range(nodes.shape[0]):
        weights = (
            1 - Fraction(float(nodes[node, 0])) - Fraction(float(nodes[node, 1])),
            Fraction(float(nodes[node, 0])),
            Fraction(float(nodes[node, 1])),
        )
        exact = tuple(
            sum(
                (
                    weights[corner] * Fraction(float(directions[row, corner, axis]))
                    for corner in range(3)
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        ray_error = max(
            ray_error,
            sum(
                (
                    abs(exact[axis] - Fraction(float(rays[node, axis])))
                    for axis in range(3)
                ),
                Fraction(0),
            ),
        )
    floor = float(np.min(domain_numerics.interval_vector_norm((rays, rays))[0]))
    if Fraction(floor) <= ray_error:
        raise ValueError(
            "sphere_chart.sphere_radial_node_enclosure.pieces must satisfy the material reference contract; got Sphere coordinate reconstruction has an unresolved radial normalization."
        )
    radius = sum(
        (
            abs(Fraction(float(term)))
            for term in np.asarray(source_atlas.radius_terms)[slot]
        ),
        Fraction(0),
    )
    amplification = (
        radius
        * Fraction(domain_numerics.matrix_norm_upper(np.asarray(source_atlas.axes)[slot]))
        * Fraction(
            domain_numerics.matrix_norm_upper(np.asarray(source_atlas.rotations)[slot])
        )
    )
    error = fraction_endpoint(
        point_error + 6 * amplification * ray_error / Fraction(floor),
        upper=True,
        saturate=True,
    )
    return _SphereRadialNodeEnclosure(error=error)


class _SphereReconstructedNodes(StrictModule):
    prescribed_routes: list[_IntArray] | None
    values: list[_FloatArray]
    errors: list[float]
    evaluations: int


def _sphere_reconstructed_node_values(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    blocks: tuple[imported_cell_geometry_transfer.SimplexGeometryBlock, ...],
    cells: _IntArray,
    count: int,
    patches: np.ndarray,
    directions: _FloatArray,
    domain: imported_geometry__meshing_domain.MeshingDomain,
    budget: CoordinateEnclosureBudget,
) -> _SphereReconstructedNodes:
    "Realize target coordinate nodes while preserving prescribed corners."
    corner_reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    prescribed_routes: list[_IntArray] | None = []
    for block in blocks:
        nodes = np.asarray(block.element.reference_nodes, dtype=np.float64)
        matches = np.all(nodes[:, None, :] == corner_reference[None, :, :], axis=-1)
        if not np.all(np.sum(matches, axis=1) == 1):
            prescribed_routes = None
            break
        prescribed_routes.append(np.asarray(np.argmax(matches, axis=1), dtype=np.int64))
    values: list[_FloatArray] = []
    errors: list[float] = []
    first, evaluations = 0, 3 * count
    if prescribed_routes is not None:
        for block, route in zip(blocks, prescribed_routes, strict=True):
            rows = block.cell_ids.size
            budget.reserve(rows * route.size * 3, rows * route.size * 3 * 32)
            values.append(
                np.asarray(target_mesh.coordinates)[cells[first : first + rows][:, route]]
            )
            evaluations += rows * route.size
            first += rows
    else:
        for block in blocks:
            nodes = np.asarray(block.element.reference_nodes, dtype=np.float64)
            barycentric = np.column_stack((1 - np.sum(nodes, axis=1), nodes))
            rows = block.cell_ids.size
            budget.reserve(rows * nodes.shape[0] * 128, rows * nodes.shape[0] * 3 * 32)
            physical = np.empty((rows, nodes.shape[0], 3), dtype=np.float64)
            for local in range(rows):
                row = first + local
                patch = int(patches[row])
                slot = int(
                    np.asarray(source_atlas.source_slots)[
                        np.flatnonzero(np.asarray(source_atlas.patches) == patch)[0]
                    ]
                )
                rays = barycentric @ directions[row]
                normalized = rays / np.linalg.norm(rays, axis=-1)[:, None]
                unit = normalized @ np.asarray(source_atlas.axes)[slot].T
                points = np.broadcast_to(
                    np.asarray(source_atlas.centers)[slot], unit.shape
                ).copy()
                for radius in np.asarray(source_atlas.radius_terms)[slot]:
                    points += radius * unit
                physical[local] = (
                    points @ np.asarray(source_atlas.rotations)[slot].T
                    + np.asarray(source_atlas.translations)[slot]
                )
                # Prescribed vertices are scientific edit coordinates. Retaining
                # them exactly avoids a second rounding epoch at shared corners.
                for corner, reference in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))):
                    selected = np.flatnonzero(np.all(nodes == reference, axis=1))
                    physical[local, selected] = np.asarray(target_mesh.coordinates)[
                        cells[row, corner]
                    ]
                _phase = _sphere_radial_node_enclosure(
                    source_atlas=source_atlas,
                    domain=domain,
                    patch=patch,
                    slot=slot,
                    physical=physical,
                    local=local,
                    nodes=nodes,
                    rays=rays,
                    directions=directions,
                    row=row,
                    budget=budget,
                )
                error = _phase.error
                errors.append(error)
                evaluations += nodes.shape[0]
            values.append(physical)
            first += rows
    return _SphereReconstructedNodes(
        prescribed_routes=prescribed_routes,
        values=values,
        errors=errors,
        evaluations=evaluations,
    )


class _SphereReconstructionInputs(StrictModule):
    blocks: tuple[imported_cell_geometry_transfer.SimplexGeometryBlock, ...]
    cells: _IntArray
    count: int
    patches: np.ndarray
    dimensions: np.ndarray
    indices: np.ndarray
    parameters: _FloatArray
    domain: imported_geometry__meshing_domain.MeshingDomain
    before: int
    planned_evaluations: int
    directions: _FloatArray


def _prepare_sphere_reconstruction_nodes(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    cell_patches: ConvertibleToArray,
    corner_source_dimensions: ConvertibleToArray,
    corner_source_indices: ConvertibleToArray,
    corner_source_parameters: ConvertibleToArray,
    corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    corner_source_entity_ids: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: transition_types_owner.CellGeometryTransitionPolicy,
    certificate_limits: MeshCertificateLimits,
    validity_policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
) -> _SphereReconstructionInputs:
    "Validate sphere reconstruction strata before admitting native queries."
    source_atlas.require_bound(
        source_atlas.domain,
        source_atlas.source_mesh,
        source_atlas.source_geometry,
        source_atlas.coordinate_contract,
    )
    if policy.reconstruction != "bounded_chart_deformation":
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.pieces must satisfy the material reference contract; got Sphere reconstruction requires the explicit bounded chart-deformation policy."
        )
    if not np.isfinite(maximum_fidelity) or maximum_fidelity < 0:
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.fidelity must satisfy the declared fidelity bound; got Sphere reconstruction requires finite nonnegative authored fidelity."
        )
    degree = imported_cell_geometry_transfer.nested_geometry_degree(
        source_atlas.source_mesh, source_atlas.source_geometry, budget=budget
    )
    blocks = imported_cell_geometry_transfer.simplex_blocks(
        target_mesh, target_layout, "sphere target", budget=budget
    )
    if any(
        block.element.cell_kind != "triangle" or block.degree != degree
        for block in blocks
    ):
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.pieces must satisfy the material reference contract; got Sphere reconstruction must retain the whole source coordinate degree."
        )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in target_mesh.blocks]
    )
    count = cells.shape[0]
    patches = np.asarray(cell_patches)
    dimensions, indices = (
        np.asarray(corner_source_dimensions),
        np.asarray(corner_source_indices),
    )
    parameters = np.asarray(corner_source_parameters, dtype=np.float64)
    if (
        patches.shape != (count,)
        or not np.issubdtype(patches.dtype, np.integer)
        or np.any(patches < 0)
        or np.any(patches >= len(source_atlas.domain.patches))
        or dimensions.shape != (count, 3)
        or not np.issubdtype(dimensions.dtype, np.integer)
        or indices.shape != (count, 3)
        or not np.issubdtype(indices.dtype, np.integer)
        or parameters.shape != (count, 3, 2)
        or len(corner_source_occurrence_paths) != count
        or len(corner_source_entity_ids) != count
        or any(
            len(row) != 3
            for row in (*corner_source_occurrence_paths, *corner_source_entity_ids)
        )
    ):
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction strata must follow real target ordered corners."
        )
    domain = source_atlas.domain
    lookup = tuple(
        {
            (path, index): row
            for row, (path, index) in enumerate(zip(paths, ids, strict=True))
        }
        for paths, ids in zip(
            domain.source_occurrences, domain.source_indices, strict=True
        )
    )
    before = budget.work_units
    budget.reserve(count * 64, count * 2048)
    planned_evaluations = 3 * count + sum(
        block.cell_ids.size * block.element.reference_nodes.shape[0] for block in blocks
    )
    if planned_evaluations > policy.maximum_evaluations:
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.budget must satisfy the admitted resource limits; got Sphere coordinate reconstruction exceeds source evaluation allowance."
        )
    native_ticket.consume(geometry_queries=planned_evaluations)
    directions = np.empty((count, 3, 3), dtype=np.float64)
    for row in range(count):
        patch = int(patches[row])
        for corner in range(3):
            dimension = int(dimensions[row, corner])
            if not 0 <= dimension <= 2:
                raise ValueError(
                    "sphere_chart.prepare_sphere_reconstruction_nodes.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction requires source strata."
                )
            index = lookup[dimension].get(
                (corner_source_occurrence_paths[row][corner], int(indices[row, corner]))
            )
            if (
                index is None
                or domain.entity_id(dimension, index)
                != corner_source_entity_ids[row][corner]
            ):
                raise ValueError(
                    "sphere_chart.prepare_sphere_reconstruction_nodes.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction changes source entity or occurrence identity."
                )
            directions[row, corner], _ = (
                imported_geometry__sphere_material_atlas.corner_direction(
                    domain, patch, dimension, index, parameters[row, corner]
                )
            )
    return _SphereReconstructionInputs(
        blocks=blocks,
        cells=cells,
        count=count,
        patches=patches,
        dimensions=dimensions,
        indices=indices,
        parameters=parameters,
        domain=domain,
        before=before,
        planned_evaluations=planned_evaluations,
        directions=directions,
    )


def _reconstruct_sphere_material_cell_geometry(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    /,
    *,
    cell_patches: ConvertibleToArray,
    corner_source_dimensions: ConvertibleToArray,
    corner_source_indices: ConvertibleToArray,
    corner_source_parameters: ConvertibleToArray,
    corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    corner_source_entity_ids: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: transition_types_owner.CellGeometryTransitionPolicy,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
    certificate_limits: MeshCertificateLimits,
    validity_policy: CellValidityPolicy,
) -> sphere_chart_records.SphereGeometryReconstruction:
    "Reconstruct the source radial source at real target coordinate nodes.\n\n source expressions and coordinate banks remain bound in source_atlas.\n The successor's whole coordinate degree is retained, while its new full\n polynomial map receives independent fidelity, validity and embedding proofs."

    _phase = _prepare_sphere_reconstruction_nodes(
        source_atlas=source_atlas,
        target_mesh=target_mesh,
        target_layout=target_layout,
        cell_patches=cell_patches,
        corner_source_dimensions=corner_source_dimensions,
        corner_source_indices=corner_source_indices,
        corner_source_parameters=corner_source_parameters,
        corner_source_occurrence_paths=corner_source_occurrence_paths,
        corner_source_entity_ids=corner_source_entity_ids,
        maximum_fidelity=maximum_fidelity,
        policy=policy,
        certificate_limits=certificate_limits,
        validity_policy=validity_policy,
        budget=budget,
        native_ticket=native_ticket,
    )
    blocks = _phase.blocks
    cells = _phase.cells
    count = _phase.count
    patches = _phase.patches
    dimensions = _phase.dimensions
    indices = _phase.indices
    parameters = _phase.parameters
    domain = _phase.domain
    before = _phase.before
    planned_evaluations = _phase.planned_evaluations
    directions = _phase.directions
    _phase = _sphere_reconstructed_node_values(
        source_atlas=source_atlas,
        target_mesh=target_mesh,
        blocks=blocks,
        cells=cells,
        count=count,
        patches=patches,
        directions=directions,
        domain=domain,
        budget=budget,
    )
    prescribed_routes = _phase.prescribed_routes
    values = _phase.values
    errors = _phase.errors
    evaluations = _phase.evaluations
    if evaluations > policy.maximum_evaluations:
        raise ValueError(
            "sphere_chart.reconstruct_sphere_material_cell_geometry.budget must satisfy the admitted resource limits; got Sphere coordinate reconstruction exceeds source evaluation allowance."
        )
    if evaluations != planned_evaluations:
        raise ValueError(
            "sphere_chart.reconstruct_sphere_material_cell_geometry.pieces must satisfy the material reference contract; got Sphere reconstruction evaluation accounting disagrees with the admitted node batch."
        )
    placed = imported_cell_geometry_transfer.place(
        blocks, values, target_layout.coordinates.shape[0]
    )
    extent = max(1.0, float(np.max(np.abs(placed.coordinates), initial=0.0)))
    if placed.continuity > policy.continuity_tolerance * extent:
        raise ValueError(
            "sphere_chart.reconstruct_sphere_material_cell_geometry.pieces must satisfy the material reference contract; got source sphere source realizations disagree at shared coordinate nodes."
        )
    elements, routes, _ = target_layout.resolve(target_mesh, budget=budget)
    names = tuple(block.name for block in target_mesh.blocks)
    geometry = CellGeometrySpec(
        dict(zip(names, elements, strict=True)),
        dict(zip(names, routes, strict=True)),
        placed.coordinates,
        budget=budget,
    )
    if target_mesh.periodic_topology is not None:
        geometry = geometry.with_periodic_source(target_mesh, budget=budget)
    with budget.activate(), budget.temporary_scope():
        atlas = SphereMaterialCellAtlas(
            domain,
            target_mesh,
            geometry,
            source_atlas.coordinate_contract,
            cell_patches=patches,
            corner_source_dimensions=dimensions,
            corner_source_indices=indices,
            corner_source_parameters=parameters,
            corner_source_occurrence_paths=corner_source_occurrence_paths,
            corner_source_entity_ids=corner_source_entity_ids,
            maximum_fidelity=maximum_fidelity,
            maximum_cells=count,
            maximum_work_units=budget.maximum_work_units,
            maximum_coordinate_memory_bytes=budget.maximum_memory_bytes,
            budget=budget,
        )
    if prescribed_routes is not None:
        # Corner-only layouts already retain the prescribed native edit points.
        # The fresh atlas above earned their full source-source enclosure;
        # there is no interior barycentric-ray rounding term at these nodes.
        errors.extend(
            float(value)
            for value in sphere_chart_records.physical_order(
                atlas, atlas.source_corner_enclosure_bounds
            )
        )
    with budget.activate(), budget.temporary_scope():
        validity = certify_cell_geometry_validity(
            geometry, mesh=target_mesh, policy=validity_policy, budget=budget
        )
        embedding = certify_global_embedding(
            target_mesh, geometry, validity, limits=certificate_limits, budget=budget
        )
    if not validity.all_certified or embedding.status != "certified":
        raise ValueError(
            "sphere_chart.reconstruct_sphere_material_cell_geometry.validity must satisfy the positive cell validity contract; got Sphere reconstruction fails full-map validity or global embedding."
        )
    bounds = sphere_chart_records.physical_order(atlas, atlas.source_fidelity_bounds)
    identity = canonical_fingerprint(
        {
            "kind": "sphere-radial-coordinate-reconstruction",
            "source": source_atlas.atlas_id,
            "target": atlas.atlas_id,
            "policy": policy.policy_id,
            "node_errors": array_tree_fingerprint(np.asarray(errors, dtype=np.float64)),
            "validity": validity.certificate_id,
            "embedding": embedding.certificate_id,
        }
    )
    return sphere_chart_records.SphereGeometryReconstruction(
        source_geometry_id=source_atlas.source_geometry_id,
        target_geometry_id=cell_geometry_id(geometry),
        source_topology_id=source_atlas.source_topology_id,
        target_topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        geometry=geometry,
        vertex_coordinates=jnp.asarray(
            imported_cell_geometry_transfer.vertex_rows(
                target_mesh, blocks, placed.coordinates
            )
        ),
        target_atlas=atlas,
        target_validity=validity,
        target_embedding=embedding,
        cell_global_ids=jnp.asarray(np.concatenate([block.cell_ids for block in blocks])),
        cell_patches=jnp.asarray(patches),
        fidelity_bounds=jnp.asarray(bounds),
        node_enclosure_bounds=jnp.asarray(np.asarray(errors, dtype=np.float64)),
        node_owner_cells=jnp.asarray(placed.owner_cells),
        node_owner_locals=jnp.asarray(placed.owner_locals),
        continuity_residual=placed.continuity,
        evaluation_count=evaluations,
        preparation_work_units=budget.work_units - before,
        reconstruction_id=identity,
    )


@checked
def reconstruct_sphere_material_cell_geometry(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    /,
    *,
    cell_patches: ConvertibleToArray,
    corner_source_dimensions: ConvertibleToArray,
    corner_source_indices: ConvertibleToArray,
    corner_source_parameters: ConvertibleToArray,
    corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    corner_source_entity_ids: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: CellGeometryTransitionPolicy,
    budget: CoordinateEnclosureBudget,
    certificate_limits: MeshCertificateLimits,
    validity_policy: CellValidityPolicy,
    scope: NativeExecutionScope | None = None,
) -> sphere_chart_records.SphereGeometryReconstruction:
    "Reconstruct sphere coordinates within the admitted /native batch."
    with chart_execution.chart_native_execution(
        budget,
        scope,
        maximum_work=budget.maximum_work_units,
        maximum_memory=budget.maximum_memory_bytes,
        maximum_queries=policy.maximum_evaluations,
    ) as (budget, native_ticket):
        return _reconstruct_sphere_material_cell_geometry(
            source_atlas,
            target_mesh,
            target_layout,
            cell_patches=cell_patches,
            corner_source_dimensions=corner_source_dimensions,
            corner_source_indices=corner_source_indices,
            corner_source_parameters=corner_source_parameters,
            corner_source_occurrence_paths=corner_source_occurrence_paths,
            corner_source_entity_ids=corner_source_entity_ids,
            maximum_fidelity=maximum_fidelity,
            policy=policy,
            budget=budget,
            native_ticket=native_ticket,
            certificate_limits=certificate_limits,
            validity_policy=validity_policy,
        )


def _sphere_piece_geometry(
    mapping: SphereProjectiveReferenceMap,
    vertices: _Triangle,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[SphereProjectiveTriangleBounds, _Triangle, Fraction, Fraction]:
    "Admit the real whole projective piece using its operand heights."

    budget.admit_work_bound(138 + 15)
    numerator_bits, denominator_bits = (
        imported_linalg__hermitian_spectral.fraction_matrix_profile(
            (mapping.exact_coefficients, vertices), budget
        )
    )
    # Homogeneous three-term sums, two reference quotients, degree-three
    # Jacobian quotients and the two three-edge areas have bounded scalar
    # operation counts (61 + 51 + 26). Each rational denominator is a
    # product of at most the 15 source denominators before cancellation.
    imported_linalg__hermitian_spectral.reserve_fraction_work(
        budget, 138, 256, 32 * (numerator_bits + 15 * denominator_bits + 1)
    )
    bounds = mapping.certify_triangle(vertices)
    target: _Triangle = (
        chart_geometry.projective_point(
            mapping.exact_coefficients,
            vertices[0],
            invalid_message="sphere_chart.project.pieces must remain inside the positive reference cone; got nonpositive denominator or negative barycentric coordinate.",
        ),
        chart_geometry.projective_point(
            mapping.exact_coefficients,
            vertices[1],
            invalid_message="sphere_chart.project.pieces must remain inside the positive reference cone; got nonpositive denominator or negative barycentric coordinate.",
        ),
        chart_geometry.projective_point(
            mapping.exact_coefficients,
            vertices[2],
            invalid_message="sphere_chart.project.pieces must remain inside the positive reference cone; got nonpositive denominator or negative barycentric coordinate.",
        ),
    )
    return bounds, target, chart_geometry.area(vertices), chart_geometry.area(target)


def _sphere_full_coordinate_expressions(
    atlas: SphereMaterialCellAtlas,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Expression, ...], ...]:
    "Retain the immutable coefficient maps in scientific physical-cell order."

    with budget.activate():
        cells = mapped_measure_owner.mapped_geometry_cells(
            atlas.source_mesh, atlas.source_geometry, budget=budget
        )
        expressions: list[tuple[Expression, ...]] = []
        for element, bank in cells:
            with budget.temporary_scope():
                values = mapped_measure_owner.mapped_coordinate_expressions(
                    element, bank, budget=budget
                )
            expressions.append(values)
    return tuple(expressions)


def _sphere_displacement_map(
    source: tuple[Expression, ...],
    target: tuple[Expression, ...],
    mapping: SphereProjectiveReferenceMap,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:
    "Subtract the whole coordinate maps once in the source chart."

    with budget.activate():
        composition = ExpressionComposition(
            mapping.reference_expressions(), budget=budget
        )
        differences: list[Expression] = []
        for old, new in zip(source, target, strict=True):
            if (
                composition.composition is not None
                or isinstance(old, imported_coordinate_enclosure.RationalPolynomial)
                or isinstance(new, imported_coordinate_enclosure.RationalPolynomial)
            ):
                difference = imported_coordinate_enclosure.expression_add(
                    composition(new, budget=budget),
                    imported_coordinate_enclosure.expression_scale(
                        old, -1, budget=budget
                    ),
                    budget=budget,
                )
            else:
                numerator, denominator = composition._polynomial(new, budget=budget)
                numerator = imported_coordinate_enclosure.add(
                    numerator,
                    imported_coordinate_enclosure.scale(
                        imported_coordinate_enclosure.multiply(
                            old, denominator, budget=budget
                        ),
                        -1,
                        budget=budget,
                    ),
                    budget=budget,
                )
                # The source homogeneous denominator is already certified by
                # the projective owner. An enclosure needs its ratio, not
                # an expensive removable-factor search on every coordinate.
                difference = (
                    imported_coordinate_enclosure.RationalPolynomial(
                        numerator, denominator
                    )
                    if numerator
                    else {}
                )
            differences.append(difference)
        return tuple(differences)


@checked
def sphere_full_map_displacement(
    displacement_map: tuple[Expression, ...],
    vertices: _Triangle,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    "Enclose the whole displacement on an entire material piece."

    with budget.activate(), budget.temporary_scope():
        budget.reserve(len(displacement_map))
        if all(
            not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
            and not value
            for value in displacement_map
        ):
            # A ordered empty coefficient support is the zero polynomial on
            # every affine subpiece, independently of mesh or chart identity.
            return 0.0
        variables = imported_coordinate_enclosure.axes(2, budget=budget)
        reference = tuple(
            imported_coordinate_enclosure.add(
                imported_coordinate_enclosure.constant(
                    vertices[0][axis], 2, budget=budget
                ),
                imported_coordinate_enclosure.add(
                    imported_coordinate_enclosure.scale(
                        variables[0], vertices[1][axis] - vertices[0][axis], budget=budget
                    ),
                    imported_coordinate_enclosure.scale(
                        variables[1], vertices[2][axis] - vertices[0][axis], budget=budget
                    ),
                    budget=budget,
                ),
                budget=budget,
            )
            for axis in range(2)
        )
        # affine substitution commutes with subtraction. The source-chart
        # difference retains both full coordinate maps and is shared by all
        # triangles of this projective overlap, including cancellation.
        at_source = ExpressionComposition(reference, budget=budget)
        bound = Fraction(0)
        for difference in displacement_map:
            if (
                not isinstance(
                    difference, imported_coordinate_enclosure.RationalPolynomial
                )
                and not difference
            ):
                continue
            if isinstance(difference, imported_coordinate_enclosure.RationalPolynomial):
                polynomial_action = at_source.composition
                if polynomial_action is None:
                    raise RuntimeError(
                        "sphere_chart.sphere_full_map_displacement.pieces must satisfy the material reference contract; got A material piece lost its affine polynomial action."
                    )
                image = imported_coordinate_enclosure.RationalPolynomial(
                    polynomial_action(difference.numerator, budget=budget),
                    polynomial_action(difference.denominator, budget=budget),
                )
            else:
                image = at_source(difference, budget=budget)
            controls = imported_coordinate_enclosure.expression_bernstein_coefficients(
                image, "simplex", 2, budget=budget
            )
            bound += max(abs(value) for value in controls)
        # coefficient cancellation, not an identity-case bypass. The
        # ordered realization enclosure preserves a zero.
        return imported_geometry__sphere_material_atlas.float_enclosure(
            rational_coefficient(
                bound, role="sphere_chart.sphere_full_map_displacement.bound"
            )
        )[1]
