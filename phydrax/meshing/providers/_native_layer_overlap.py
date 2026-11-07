#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source/target common refinement for staged layer boundaries.

The input source triangles remain the authority: output triangles are merely
carriers and may cross a triangulation diagonal of the original source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ...discretization._cell_geometry_validity import cell_geometry_id
from ...geometry import _coplanar_overlap
from ...geometry._domain_boundary import MeshingDomainBoundarySource
from ...geometry._meshing_domain import MeshingDomain
from ...geometry._planar_primitives import box_overlap_pairs, polygon_boxes, signed_area2
from ...geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryFaceImageWitness,
)
from ...geometry._source_fidelity import validate_source_fidelity_witness
from .._layer_core_topology import validate_staged_layer_domain


if TYPE_CHECKING:
    from ...discretization import CellGeometrySpec, CellMesh
    from ...discretization._exact_algebraic_geometry import ExactScalar
    from ...geometry._source_fidelity import ValidatedSourceFidelityWitness
    from .._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
    from .._layer_core_topology import LayerCoreConstruction
    from .._lineage import MeshLineage
    from .._result import CellMeshingResult
    from ._native_layer import _LayerOutputFace
    from ._native_sources import NativeLayerCoreSource

type _ExactSegment = tuple[tuple[Fraction, ...], ...]
type _SegmentInput = Sequence[Sequence[ExactScalar | int | np.integer]]
type _Interval = tuple[Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class CollinearSegmentPiece:
    """One source curve/target edge intersection in both exact unit charts."""

    source_row: int
    target_row: int
    source_interval: tuple[Fraction, Fraction]
    target_interval: tuple[Fraction, Fraction]


def _exact_segment(segment: _SegmentInput) -> _ExactSegment:
    if len(segment) != 2 or any(len(point) != 3 for point in segment):
        raise ValueError("Overlap segments require two exact 3D points.")

    def exact_value(value: ExactScalar | int | np.integer) -> Fraction:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                "Overlap coordinates must be exact rational values, not floats."
            )
        if isinstance(value, Fraction):
            return value
        if isinstance(value, (int, np.integer)):
            return Fraction(int(value))
        raise TypeError("Overlap coordinates must be exact rational values, not floats.")

    result = tuple(tuple(exact_value(value) for value in point) for point in segment)
    if result[0] == result[1]:
        raise ValueError("Source and target overlap segments must be nondegenerate.")
    return result


def _line_key(
    segment: _ExactSegment,
) -> tuple[int, tuple[Fraction, ...], tuple[Fraction, ...]]:
    direction = tuple(segment[1][axis] - segment[0][axis] for axis in range(3))
    axis = next(index for index, value in enumerate(direction) if value)
    direction = tuple(value / direction[axis] for value in direction)
    anchor = tuple(
        segment[0][index] - segment[0][axis] * direction[index] for index in range(3)
    )
    return axis, direction, anchor


def _require_interval_partition(intervals: Sequence[_Interval], complete: bool) -> None:
    end = Fraction(0)
    for left, right in sorted(tuple(sorted(interval)) for interval in intervals):
        if left < 0 or right > 1 or left >= right or left < end:
            raise ValueError(
                "Source curve or target edge overlap is invalid or overlapping."
            )
        if complete and left != end:
            raise ValueError("Source curve or target edge overlap has a gap.")
        end = right
    if complete and end != 1:
        raise ValueError("Source curve or target edge overlap is incomplete.")


def exact_collinear_segment_partitions(
    source_segments: Sequence[_SegmentInput],
    target_segments: Sequence[_SegmentInput],
    /,
    *,
    require_source: bool = False,
    require_target: bool = False,
) -> tuple[CollinearSegmentPiece, ...]:
    """Intersect exact curves and edges, preserving both row identities.

    A target edge can cover several original curves and conversely. Unprotected
    triangulation diagonals need not be covered by target edges; protected
    source curves must be passed in a bank with ``require_source=True``.
    Partial banks still reject positive-length overlaps.
    """
    sources = tuple(_exact_segment(segment) for segment in source_segments)
    targets = tuple(_exact_segment(segment) for segment in target_segments)
    source_groups, target_groups = {}, {}
    for row, segment in enumerate(sources):
        source_groups.setdefault(_line_key(segment), []).append(row)
    for row, segment in enumerate(targets):
        target_groups.setdefault(_line_key(segment), []).append(row)
    by_source = [[] for _ in sources]
    by_target = [[] for _ in targets]
    pieces = []
    for line in source_groups.keys() & target_groups.keys():
        axis = line[0]
        source_rows, target_rows = source_groups[line], target_groups[line]
        source_boxes = polygon_boxes(
            tuple(
                tuple((point[axis], Fraction(0)) for point in sources[row])
                for row in source_rows
            )
        )
        target_boxes = polygon_boxes(
            tuple(
                tuple((point[axis], Fraction(0)) for point in targets[row])
                for row in target_rows
            )
        )
        if source_boxes is None or target_boxes is None:
            raise ValueError(
                "Exact overlap coordinates exceed the bounded spatial index."
            )
        for source_index, target_index in box_overlap_pairs(
            len(source_rows),
            len(target_rows),
            source_boxes,
            target_boxes,
            self_pairs=False,
        ):
            source_row, target_row = (
                source_rows[int(source_index)],
                target_rows[int(target_index)],
            )
            source = sources[source_row]
            target = targets[target_row]
            source_axis = (source[0][axis], source[1][axis])
            target_axis = (target[0][axis], target[1][axis])
            left = max(min(source_axis), min(target_axis))
            right = min(max(source_axis), max(target_axis))
            if left >= right:
                continue
            # Both intervals are ordered in target-edge traversal order.
            first, last = (
                (left, right) if target_axis[1] > target_axis[0] else (right, left)
            )
            source_interval = (
                (first - source_axis[0]) / (source_axis[1] - source_axis[0]),
                (last - source_axis[0]) / (source_axis[1] - source_axis[0]),
            )
            target_interval = (
                (first - target_axis[0]) / (target_axis[1] - target_axis[0]),
                (last - target_axis[0]) / (target_axis[1] - target_axis[0]),
            )
            pieces.append(
                CollinearSegmentPiece(
                    source_row, target_row, source_interval, target_interval
                )
            )
            by_source[source_row].append(source_interval)
            by_target[target_row].append(target_interval)
    for intervals in by_source:
        _require_interval_partition(intervals, require_source)
    for intervals in by_target:
        _require_interval_partition(intervals, require_target)
    return tuple(
        sorted(
            pieces,
            key=lambda piece: (piece.target_row, piece.target_interval, piece.source_row),
        )
    )


def _evaluate_boundary_face_images(
    mesh: CellMesh, geometry: CellGeometrySpec, faces: tuple[_LayerOutputFace, ...], /
) -> tuple[
    tuple[tuple[tuple[ExactScalar, ...], ...], ...],
    dict[int, tuple[ExactScalar, ...]],
]:
    """Evaluate actual face corners and cell centers with one exact map preparation."""
    from ...discretization._reference_cell import reference_cell_topology

    requests: dict[int, list[tuple[Fraction, ...]]] = {}
    kinds: dict[int, str] = {}
    spans: list[tuple[int, int, int]] = []
    for _, cell, _, kind, order, references in faces:
        row = requests.setdefault(int(cell), [])
        first = len(row)
        row.extend(references[int(vertex)] for vertex in order)
        spans.append((int(cell), first, len(row)))
        previous = kinds.setdefault(int(cell), kind)
        if previous != kind:
            raise ValueError("A target cell changes its boundary-face reference kind.")
    identifiers = tuple(requests)
    for cell in identifiers:
        vertices = reference_cell_topology(kinds[cell]).vertices
        center = tuple(
            sum(
                (Fraction.from_float(float(vertex[axis])) for vertex in vertices),
                Fraction(0),
            )
            / Fraction(len(vertices))
            for axis in range(3)
        )
        requests[cell].append(center)
    evaluated = geometry.evaluate_exact(
        mesh,
        np.asarray(identifiers, dtype=np.int64),
        tuple(tuple(requests[cell]) for cell in identifiers),
    )
    by_cell = dict(zip(identifiers, evaluated, strict=True))
    images = tuple(by_cell[cell][first:last] for cell, first, last in spans)
    centers = {cell: by_cell[cell][-1] for cell in identifiers}
    return images, centers


def _staged_layer_overlap_witness(
    source: NativeLayerCoreSource,
    construction: LayerCoreConstruction,
    query: MeshingDomainBoundarySource,
    /,
) -> SourceBoundaryFaceImageWitness:
    """Compose retained source charts with actual conforming target facets."""
    from ._native_layer import _layer_output_boundary_faces

    domain = source.source_domain
    if (
        not isinstance(domain, MeshingDomain)
        or query.domain.domain_id != domain.domain_id
    ):
        raise ValueError("Staged overlap requires the original source domain query.")
    if not isinstance(source.fidelity_source, MeshingDomainBoundarySource) or (
        query.source_id != source.fidelity_source.source_id
        or query.source_revision != source.fidelity_source.source_revision
        or not set(query.patches) <= set(source.fidelity_source.patches)
    ):
        raise ValueError("Overlap requires a scope of the retained original exterior.")
    atlas = source.source_boundary_charts
    staged = validate_staged_layer_domain(source) if atlas is None else None
    mesh = construction.mesh
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    carrier = construction.overlap_exact_geometry
    if carrier is not None:
        from ...discretization._exact_plc_geometry import ExactPlcCellGeometrySource
        from .._layer_core_subdivision import require_overlap_source_provenance

        require_overlap_source_provenance(source, construction)

        if not isinstance(carrier, ExactPlcCellGeometrySource):
            raise TypeError("Overlap geometry requires an exact PLC source carrier.")
        carrier = ExactPlcCellGeometrySource.from_stored_fields(carrier)
        if (
            carrier.source_id != source.source_id
            or carrier.source_revision != source.source_revision
            or not np.array_equal(np.asarray(carrier.vertex_global_ids), vertex_ids)
            or not np.array_equal(
                np.asarray(carrier.rounded_vertices).view(np.uint64),
                points.view(np.uint64),
            )
        ):
            raise ValueError(
                "Exact overlap carrier changes source or target vertex identity."
            )
    # The retained carrier proves subdivision provenance; it is not the
    # coordinate map of a subsequently curved target cell.
    source_metadata = []
    if staged is not None:
        source_points = tuple(
            tuple(Fraction.from_float(float(x)) for x in point)
            for point in np.asarray(staged.vertices, dtype=np.float64)
        )
        source_facets = np.asarray(staged.facets, dtype=np.int64)
        source_triangles = tuple(
            tuple(source_points[int(vertex)] for vertex in source_facets[patch])
            for patch in query.patches
        )
        source_metadata = [
            (
                patch,
                (
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(0)),
                    (Fraction(0), Fraction(1)),
                ),
            )
            for patch in query.patches
        ]
    else:
        if atlas is None:
            raise ValueError("Staged CAD overlap lost its source atlas.")
        from .._layer_core_staged_domain import exact_planar_atlas_triangles

        source_triangles, proved_metadata = exact_planar_atlas_triangles(atlas, query)
        source_metadata = list(proved_metadata)
    target_triangles = []
    target_metadata = []
    faces = []
    for matches in _layer_output_boundary_faces(mesh).values():
        if len(matches) != 1:
            raise ValueError("Staged output boundary facet has ambiguous incidence.")
        faces.append(matches[0])
    images, centers = _evaluate_boundary_face_images(
        mesh, construction.geometry, tuple(faces)
    )
    for (face_id, cell_id, local_face, _, order, _), physical in zip(
        faces, images, strict=True
    ):
        if any(not isinstance(value, Fraction) for point in physical for value in point):
            raise ValueError(
                "Staged source image requires rational target face coordinates."
            )
        rational_physical = tuple(
            tuple(value for value in point if isinstance(value, Fraction))
            for point in physical
        )
        if len(order) == 3:
            triangles = ((0, 1, 2),)
        elif len(order) == 4:
            triangles = ((0, 1, 2), (0, 2, 3))
        else:
            raise ValueError(
                "Staged output boundary must have triangular or quadrilateral facets."
            )
        for triangle in triangles:
            target_triangles.append(tuple(rational_physical[index] for index in triangle))
            target_metadata.append((face_id, cell_id, local_face))
    pieces = _coplanar_overlap.exact_coplanar_triangle_partitions(
        source_triangles,
        target_triangles,
        require_target=set(query.patches) == set(source.fidelity_source.patches),
    )
    if not pieces:
        raise ValueError("Staged source overlap produced no boundary pieces.")
    (
        cell_ids,
        face_ids,
        local_faces,
        image_rows,
        chart_ids,
        occurrences,
        region_sides,
        orientations,
    ) = ([] for _ in range(8))
    source_sides = (
        None if staged is None else np.asarray(staged.facet_regions, dtype=np.int64)
    )
    sorted_cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    regions = np.asarray(construction.cell_regions, dtype=np.int64)
    if regions.shape != sorted_cells.shape:
        raise ValueError(
            "Staged boundary image requires actual composite region coverage."
        )
    region_by_cell = dict(zip(sorted_cells.tolist(), regions.tolist(), strict=True))
    from .._layer_core_association import composite_region_identity

    for piece in pieces:
        face_id, cell_id, local_face = target_metadata[piece.target_row]
        original = source_triangles[piece.source_row]
        image_rows.append(
            tuple(
                tuple(
                    original[0][axis]
                    + uv[0] * (original[1][axis] - original[0][axis])
                    + uv[1] * (original[2][axis] - original[0][axis])
                    for axis in range(3)
                )
                for uv in piece.source_chart
            )
        )
        patch, root_uv = source_metadata[piece.source_row]
        sides: list[str | None] = [None, None]
        if source_sides is not None:
            side_rows = tuple(int(value) for value in source_sides[patch])
            sides = [
                None if value < 0 else domain.entity_id(3, value) for value in side_rows
            ]
        else:
            for region_row, region in enumerate(domain.regions):
                for region_patch, sign in region.boundary:
                    if region_patch == patch:
                        slot = 0 if sign > 0 else 1
                        identity = domain.entity_id(3, region_row)
                        if sides[slot] not in (None, identity):
                            raise ValueError(
                                "CAD source occurrence has conflicting material sides."
                            )
                        sides[slot] = identity
        if sum(value is not None for value in sides) != 1:
            raise ValueError(
                "A source exterior occurrence must retain exactly one material side."
            )
        first = tuple(original[1][i] - original[0][i] for i in range(3))
        second = tuple(original[2][i] - original[0][i] for i in range(3))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        center = centers[int(cell_id)]
        sign = (1 if signed_area2(root_uv) > 0 else -1) * domain.patches[
            patch
        ].orientation
        side = sign * sum(
            normal[axis] * (center[axis] - original[0][axis]) for axis in range(3)
        )
        material = composite_region_identity(source, int(region_by_cell[int(cell_id)]))
        if not side or material != sides[1 if side > 0 else 0]:
            raise ValueError(
                "A conforming source image changed its original oriented material side."
            )
        region_sides.append((sides[0], sides[1]))
        cell_ids.append(cell_id)
        face_ids.append(face_id)
        local_faces.append(local_face)
        chart_ids.append(domain.entity_id(2, patch))
        occurrences.append(tuple(domain.source_occurrences[2][patch]))
        source_uv = tuple(
            (
                root_uv[0][0]
                + point[0] * (root_uv[1][0] - root_uv[0][0])
                + point[1] * (root_uv[2][0] - root_uv[0][0]),
                root_uv[0][1]
                + point[0] * (root_uv[1][1] - root_uv[0][1])
                + point[1] * (root_uv[2][1] - root_uv[0][1]),
            )
            for point in piece.source_chart
        )
        orientations.append(
            (1 if signed_area2(source_uv) > 0 else -1) * domain.patches[patch].orientation
        )
    witness = SourceBoundaryFaceImageWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(construction.geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_scope_id,
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(face_ids, dtype=np.int64),
        local_entity_indices=np.asarray(local_faces, dtype=np.int32),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(chart_ids),
        occurrence_paths=tuple(occurrences),
        source_region_sides=tuple(region_sides),
        image_triangle_rational_tokens=rational_chart_tokens(image_rows),
        orientations=np.asarray(orientations, dtype=np.int8),
    )
    return witness


def issue_layer_boundary_image_witness(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    query: MeshingDomainBoundarySource,
    root: LayerCoreRootGeometry | None,
    accepted_source: CellMeshingResult,
    lineage: MeshLineage,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> tuple[
    MeshingDomainBoundarySource,
    SourceBoundaryFaceImageWitness,
    ValidatedSourceFidelityWitness,
]:
    """Reissue original exterior or cap image authority on an actual target mesh."""
    from .._association import GeometrySourceEntityRole
    from .._layer_core_boundary_replay import replay_composite_regions
    from .._layer_source_association_replay import replay_layer_source_associations
    from ._native_layer import _layer_output_boundary_faces

    domain = source.source_domain
    if (
        not isinstance(domain, MeshingDomain)
        or not isinstance(query, MeshingDomainBoundarySource)
        or query.domain.domain_id != domain.domain_id
    ):
        raise ValueError("Layer image renewal requires its original domain and query.")
    query.require_current()
    layer = replay_layer_source_associations(
        source, mesh, geometry, root, computed_root=computed_root
    )
    _, region_table, _ = replay_composite_regions(source, accepted_source, mesh, lineage)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    if (
        region_table.source_id != source.source_id
        or region_table.source_revision != source.source_revision
        or region_table.target_entity_set_id != mesh.entity_set(3).entity_set_id
        or set(np.asarray(region_table.target_global_ids, dtype=np.int64).tolist())
        != set(cell_ids.tolist())
        or any(
            role is not GeometrySourceEntityRole.REGION
            for role in region_table.source_roles
        )
    ):
        raise ValueError(
            "Image renewal needs one original-source region row per target cell."
        )
    region_by_cell = dict(
        zip(
            np.asarray(region_table.target_global_ids, dtype=np.int64).tolist(),
            region_table.source_entity_ids,
            strict=True,
        )
    )
    facet_rows = np.asarray(source.complex.polygon_facets, dtype=np.int64)
    cap_patches = set()
    for polygon in np.asarray(source.cap_polygon_ids, dtype=np.int64):
        facet = int(facet_rows[int(polygon)])
        identifier = source.core_facet_source_ids[facet]
        occurrence = source.core_facet_source_occurrence_paths[facet]
        if identifier is None or occurrence is None:
            if source.cap_refinement == "conforming":
                raise ValueError(
                    "A conforming original cap facet lacks its source occurrence."
                )
            continue
        cap_patches.add(
            domain.resolve_entity_id(2, identifier, occurrence_path=occurrence)
        )
    is_cap = bool(cap_patches) and set(query.patches) == cap_patches
    if not is_cap and (
        not isinstance(source.fidelity_source, MeshingDomainBoundarySource)
        or query.source_scope_id != source.fidelity_source.source_scope_id
    ):
        raise ValueError(
            "Image renewal query is neither the original exterior nor its cap."
        )
    original = []
    source_metadata = []
    atlas = source.source_boundary_charts
    if atlas is None:
        staged = validate_staged_layer_domain(source)
        points = np.asarray(staged.vertices, dtype=np.float64)
        facets = np.asarray(staged.facets, dtype=np.int64)
        for patch in query.patches:
            original.append(
                tuple(
                    tuple(
                        Fraction.from_float(float(value)) for value in points[int(vertex)]
                    )
                    for vertex in facets[patch]
                )
            )
            source_metadata.append(
                (
                    patch,
                    (
                        (Fraction(0), Fraction(0)),
                        (Fraction(1), Fraction(0)),
                        (Fraction(0), Fraction(1)),
                    ),
                )
            )
    else:
        from .._layer_core_staged_domain import exact_planar_atlas_triangles

        triangles, proved_metadata = exact_planar_atlas_triangles(atlas, query)
        original.extend(triangles)
        source_metadata.extend(proved_metadata)
    targets, metadata = [], []
    selected_faces = []
    core = set(layer.core_cell_ids)
    for incident in _layer_output_boundary_faces(mesh, include_internal=is_cap).values():
        if is_cap:
            core_rows = [row for row in incident if int(row[1]) in core]
            layer_rows = [row for row in incident if int(row[1]) not in core]
            if not core_rows or not layer_rows:
                continue
            if len(core_rows) != 1 or len(layer_rows) != 1:
                raise ValueError(
                    "Renewed cap requires exactly one core and one layer cell."
                )
            chosen = core_rows[0]
        else:
            if len(incident) != 1:
                raise ValueError("Renewed exterior face has ambiguous cell incidence.")
            chosen = incident[0]
        selected_faces.append((chosen, tuple(incident)))
    all_faces = tuple(face for _, incident in selected_faces for face in incident)
    images, centers = _evaluate_boundary_face_images(mesh, geometry, all_faces)
    image_by_incidence = {
        (int(face[1]), int(face[2])): image
        for face, image in zip(all_faces, images, strict=True)
    }
    for chosen, incident in selected_faces:
        face_id, cell_id, local_face, _, order, _ = chosen
        physical = image_by_incidence[(int(cell_id), int(local_face))]
        if any(not isinstance(value, Fraction) for point in physical for value in point):
            raise ValueError(
                "Exact image renewal currently requires rational face coordinates."
            )
        if len(order) == 3:
            triangles = ((0, 1, 2),)
        elif len(order) == 4:
            triangles = ((0, 1, 2), (0, 2, 3))
        else:
            raise ValueError(
                "Layer image renewal requires triangular or quadrilateral faces."
            )
        for indices in triangles:
            targets.append(tuple(physical[index] for index in indices))
            metadata.append(
                (
                    int(face_id),
                    int(cell_id),
                    int(local_face),
                    tuple(int(row[1]) for row in incident),
                )
            )
    pieces = _coplanar_overlap.exact_coplanar_triangle_partitions(
        tuple(original), tuple(targets)
    )
    if not pieces:
        raise ValueError("Layer image renewal has no exact source overlap.")
    target_cells, target_faces, local_faces, chart_ids = [], [], [], []
    occurrences, side_rows, orientations, image_rows = [], [], [], []
    for piece in pieces:
        patch, uv_root = source_metadata[piece.source_row]
        face_id, cell_id, local_face, incident_cells = metadata[piece.target_row]
        triangle = original[piece.source_row]
        uv = tuple(
            (
                uv_root[0][0]
                + point[0] * (uv_root[1][0] - uv_root[0][0])
                + point[1] * (uv_root[2][0] - uv_root[0][0]),
                uv_root[0][1]
                + point[0] * (uv_root[1][1] - uv_root[0][1])
                + point[1] * (uv_root[2][1] - uv_root[0][1]),
            )
            for point in piece.source_chart
        )
        source_orientation = (1 if signed_area2(uv) > 0 else -1) * domain.patches[
            patch
        ].orientation
        region_sides = [None, None]
        for region_row, region in enumerate(domain.regions):
            for region_patch, sign in region.boundary:
                if region_patch == patch:
                    slot = 0 if sign > 0 else 1
                    identity = domain.entity_id(3, region_row)
                    if region_sides[slot] not in (None, identity):
                        raise ValueError(
                            "Original source face has conflicting material sides."
                        )
                    region_sides[slot] = identity
        first = tuple(triangle[1][i] - triangle[0][i] for i in range(3))
        second = tuple(triangle[2][i] - triangle[0][i] for i in range(3))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        normal_sign = (1 if signed_area2(uv_root) > 0 else -1) * domain.patches[
            patch
        ].orientation
        for incident_cell in incident_cells:
            side = normal_sign * sum(
                normal[axis] * (centers[incident_cell][axis] - triangle[0][axis])
                for axis in range(3)
            )
            if (
                not side
                or region_by_cell[incident_cell] != region_sides[1 if side > 0 else 0]
            ):
                raise ValueError(
                    "Renewed image changed an original oriented material side."
                )
        image_rows.append(
            tuple(
                tuple(
                    triangle[0][axis]
                    + point[0] * (triangle[1][axis] - triangle[0][axis])
                    + point[1] * (triangle[2][axis] - triangle[0][axis])
                    for axis in range(3)
                )
                for point in piece.source_chart
            )
        )
        target_cells.append(cell_id)
        target_faces.append(face_id)
        local_faces.append(local_face)
        chart_ids.append(domain.entity_id(2, patch))
        occurrences.append(tuple(domain.source_occurrences[2][patch]))
        side_rows.append((region_sides[0], region_sides[1]))
        orientations.append(source_orientation)
    witness = SourceBoundaryFaceImageWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_scope_id,
        cell_global_ids=np.asarray(target_cells, dtype=np.int64),
        entity_global_ids=np.asarray(target_faces, dtype=np.int64),
        local_entity_indices=np.asarray(local_faces, dtype=np.int64),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(chart_ids),
        occurrence_paths=tuple(occurrences),
        source_region_sides=tuple(side_rows),
        orientations=np.asarray(orientations, dtype=np.int8),
        image_triangle_rational_tokens=rational_chart_tokens(image_rows),
    )
    token = validate_source_fidelity_witness(mesh, geometry, query, witness)
    return query, witness, token
