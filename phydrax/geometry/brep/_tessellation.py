#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Derived CAD tessellation through the canonical native surface scheduler.

Exact model topology and carrier identity remain authoritative. This adapter
only lowers the requested tessellation fidelity and capacity to the owning
surface construction API; it has no independent triangulation or refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ._model import BRepGeometry
from ._patches import AbstractSurfacePatch


if TYPE_CHECKING:
    from ._constructors import BRepTessellationPolicy


@dataclass(frozen=True, slots=True)
class BRepTessellation:
    vertices: np.ndarray
    triangles: np.ndarray
    face_ids: np.ndarray
    parameters: np.ndarray
    deviation_bounds: np.ndarray
    normal_bounds: np.ndarray
    vertex_source_dimensions: np.ndarray
    vertex_source_indices: np.ndarray
    vertex_parameters: np.ndarray
    chart_restriction_vertices: np.ndarray
    chart_restriction_edges: np.ndarray
    chart_restriction_endpoint_parameters: np.ndarray
    chart_restriction_parameters: np.ndarray
    triangle_occurrence_ids: np.ndarray
    vertex_occurrence_ids: np.ndarray


def _pose_fidelity(
    points: np.ndarray,
    world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    cells: np.ndarray,
    deviation: np.ndarray,
    normal: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the actual affine pose and its floating vertex evaluation error."""
    from .._interval_enclosure import interval_add, interval_multiply

    gram = [
        [
            sum(
                (
                    Fraction(float(rotation[k, i])) * Fraction(float(rotation[k, j]))
                    for k in range(3)
                ),
                Fraction(0),
            )
            for j in range(3)
        ]
        for i in range(3)
    ]
    upper = max(
        gram[i][i] + sum((abs(gram[i][j]) for j in range(3) if j != i), Fraction(0))
        for i in range(3)
    )
    lower = min(
        gram[i][i] - sum((abs(gram[i][j]) for j in range(3) if j != i), Fraction(0))
        for i in range(3)
    )
    if lower <= 0:
        raise ValueError("A placed CAD artifact has no certified nonsingular pose.")

    def square_root_upper(value: Fraction, /) -> float:
        operand = float(value)
        if Fraction(operand) < value:
            operand = float(np.nextafter(operand, np.inf))
        result = float(np.sqrt(operand))
        return (
            float(np.nextafter(result, np.inf))
            if Fraction(result) ** 2 < value
            else result
        )

    stretch, condition = square_root_upper(upper), square_root_upper(upper / lower)
    lo = np.broadcast_to(translation, world.shape).copy()
    hi = lo.copy()
    for axis in range(3):
        product = interval_multiply(
            (points[:, axis : axis + 1], points[:, axis : axis + 1]),
            (rotation[:, axis], rotation[:, axis]),
        )
        lo, hi = interval_add((lo, hi), product)
    error = np.nextafter(np.maximum(np.abs(lo - world), np.abs(hi - world)), np.inf)
    squared = np.zeros((world.shape[0],), dtype=np.float64)
    for axis in range(3):
        squared = np.nextafter(
            squared + np.nextafter(error[:, axis] ** 2, np.inf), np.inf
        )
    vertex_error = np.nextafter(np.sqrt(squared), np.inf)
    positional = np.nextafter(
        np.nextafter(deviation * stretch, np.inf) + np.max(vertex_error[cells], axis=1),
        np.inf,
    )
    angular = np.nextafter(normal * condition, np.inf)
    return positional, angular


def _place_occurrences(
    template: BRepTessellation, geometry: BRepGeometry, maximum_triangles: int, /
) -> BRepTessellation:
    """Expand definition artifacts by authored incidences; never weld placements."""
    if not geometry.occurrences:
        return template
    topology = geometry.topology()
    selections = [
        (index, topology.solid_faces[occurrence.solid])
        for index, occurrence in enumerate(geometry.occurrences)
    ]
    unshelled = tuple(
        face for face, owners in enumerate(topology.face_solids) if not owners
    )
    if unshelled:
        selections.append((-1, unshelled))
    face_counts = np.bincount(template.face_ids, minlength=topology.num_faces)
    count = sum(int(np.sum(face_counts[list(faces)])) for _, faces in selections)
    if count > maximum_triangles:
        raise ValueError("Placed CAD tessellation exceeds maximum_triangles.")
    blocks = []
    offset = 0
    for occurrence_index, faces in selections:
        rows = np.flatnonzero(np.isin(template.face_ids, faces))
        if not rows.size:
            continue
        used, inverse = np.unique(template.triangles[rows], return_inverse=True)
        triangles = inverse.reshape((-1, 3)).astype(np.int32) + offset
        parameters = template.parameters[rows]
        local_cells = inverse.reshape((-1, 3))
        deviation, normal = template.deviation_bounds[rows], template.normal_bounds[rows]
        points = template.vertices[used]
        restriction_rows = np.flatnonzero(
            np.isin(template.chart_restriction_vertices, used)
        )
        restricted_vertices = template.chart_restriction_vertices[restriction_rows]
        restricted_edges = template.chart_restriction_edges[restriction_rows]
        if restricted_edges.size and not np.all(np.isin(restricted_edges, used)):
            raise ValueError(
                "A placed tessellation lost an exact chart-restriction endpoint."
            )
        placed_restriction_vertices = (
            np.searchsorted(used, restricted_vertices).astype(np.int64) + offset
        )
        placed_restriction_edges = (
            np.searchsorted(used, restricted_edges).astype(np.int64) + offset
        )
        if occurrence_index >= 0:
            occurrence = geometry.occurrences[occurrence_index]
            local = points
            points = occurrence.place(local)
            deviation, normal = _pose_fidelity(
                local,
                points,
                np.asarray(occurrence.rotation, dtype=np.float64),
                np.asarray(occurrence.translation, dtype=np.float64),
                local_cells,
                deviation,
                normal,
            )
            sides = dict(
                zip(
                    topology.solid_faces[occurrence.solid],
                    topology.solid_face_orientations[occurrence.solid],
                    strict=True,
                )
            )
            reverse = np.asarray(
                [sides[face] < 0 for face in template.face_ids[rows]],
                dtype=np.bool_,
            )
            triangles[reverse] = triangles[reverse][:, (0, 2, 1)]
            parameters[reverse] = parameters[reverse][:, (0, 2, 1)]
        blocks.append(
            BRepTessellation(
                points,
                triangles,
                template.face_ids[rows],
                parameters,
                deviation,
                normal,
                template.vertex_source_dimensions[used],
                template.vertex_source_indices[used],
                template.vertex_parameters[used],
                placed_restriction_vertices,
                placed_restriction_edges,
                template.chart_restriction_endpoint_parameters[restriction_rows],
                template.chart_restriction_parameters[restriction_rows],
                np.full((rows.size,), occurrence_index, dtype=np.int32),
                np.full((used.size,), occurrence_index, dtype=np.int32),
            )
        )
        offset += used.size
    return BRepTessellation(
        np.concatenate([block.vertices for block in blocks]),
        np.concatenate([block.triangles for block in blocks]),
        np.concatenate([block.face_ids for block in blocks]),
        np.concatenate([block.parameters for block in blocks]),
        np.concatenate([block.deviation_bounds for block in blocks]),
        np.concatenate([block.normal_bounds for block in blocks]),
        np.concatenate([block.vertex_source_dimensions for block in blocks]),
        np.concatenate([block.vertex_source_indices for block in blocks]),
        np.concatenate([block.vertex_parameters for block in blocks]),
        np.concatenate([block.chart_restriction_vertices for block in blocks]),
        np.concatenate([block.chart_restriction_edges for block in blocks], axis=0),
        np.concatenate(
            [block.chart_restriction_endpoint_parameters for block in blocks], axis=0
        ),
        np.concatenate([block.chart_restriction_parameters for block in blocks], axis=0),
        np.concatenate([block.triangle_occurrence_ids for block in blocks]),
        np.concatenate([block.vertex_occurrence_ids for block in blocks]),
    )


def tessellate_brep(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    orientation: np.ndarray,
    policy: BRepTessellationPolicy,
    /,
    *,
    source_id: str,
    source_revision: str,
) -> BRepTessellation:
    """Generate shared-edge, trim-conforming triangles with hard fidelity bounds."""
    if not patches or not policy.realize:
        return BRepTessellation(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 3, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 2, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    from ...discretization._coordinate_enclosure import coordinate_enclosure_budget
    from ...meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        MeshingLimits,
        SurfaceMeshingSpec,
    )
    from ...meshing._controls import FeatureKind, ProtectedFeature
    from ...meshing._domain import compile_surface_domain
    from ...meshing._scope import MeshingEntityKind, MeshingScope
    from ...meshing._sizing import SizeControlStrength, UniformSizeControl
    from ...meshing._surface_generation import generate_surface
    from ...meshing.providers._native_options import NativeSurfaceSchedule
    from .._meshing_domain import MeshingDomain, MeshingDomainBoundarySource

    domain = MeshingDomain.from_brep_geometry(
        geometry,
        patches,
        orientation,
        source_id=source_id,
        source_revision=source_revision,
        authority_id=geometry.geometry_id,
    )
    scope = MeshingScope(
        source_id,
        source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.source_indices[2], dtype=np.int64),
    )
    maximum = policy.maximum_triangles
    limits = MeshingLimits(
        maximum_vertices=max(1, 3 * maximum),
        maximum_edges=max(1, 3 * maximum),
        maximum_faces=maximum,
        maximum_cells=maximum,
        maximum_connectivity_entries=max(1, 12 * maximum),
        maximum_data_bytes=max(1, 256 * maximum),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        # CAD policy requests fidelity, not a statistical mesh density.
        size_controls=(
            UniformSizeControl(
                scope,
                domain.scale,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        protected_features=(
            ProtectedFeature(
                # The owning scheduler now charges the carrier and actual source
                # trim homotopy against this one original physical allowance.
                scope,
                FeatureKind.SURFACE,
                maximum_deviation=policy.linear_deflection,
            ),
        ),
        limits=limits,
    )
    compiled = compile_surface_domain(domain, specification)
    owner = coordinate_enclosure_budget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )
    with owner.activate():
        construction = generate_surface(
            compiled,
            NativeSurfaceSchedule(quality_angle_degrees=0.0),
            limits,
            0.0,
            maximum_normal_angle=policy.angular_deflection,
        )
        source = MeshingDomainBoundarySource(
            domain,
            tuple(compiled.patches.tolist()),
            chart_triangulations=construction.chart_triangulations,
        )
        retained = sum(
            array.nbytes
            for array in (
                construction.vertices,
                construction.vertex_source_dimensions,
                construction.vertex_source_indices,
                construction.vertex_parameters,
                construction.chart_restriction_vertices,
                construction.chart_restriction_edges,
                construction.chart_restriction_endpoint_parameters,
                construction.chart_restriction_parameters,
                construction.triangles,
                construction.triangle_patches,
                construction.triangle_charts,
                construction.triangle_parameters,
                construction.triangle_deviations,
                construction.triangle_deviation_bounds,
                construction.triangle_normal_bounds,
                construction.curve_edges,
                construction.curve_edge_curves,
                construction.curve_edge_deviations,
                construction.curve_edge_deviation_bounds,
                construction.unresolved,
                *(
                    array
                    for (
                        _,
                        charts,
                        points,
                        cells,
                        boundary,
                        provenance,
                        restriction_required,
                        restriction_vertices,
                        restriction_edges,
                        restriction_parameters,
                    ) in construction.chart_triangulations
                    for array in (
                        charts,
                        points,
                        cells,
                        boundary,
                        provenance,
                        restriction_required,
                        restriction_vertices,
                        restriction_edges,
                        restriction_parameters,
                    )
                ),
                *(array for _, array in construction.curve_parameters),
            )
        )
        remaining_work = limits.maximum_work_units - construction.work_units
        remaining_memory = max(0, limits.maximum_scratch_bytes - retained)
        cover_queries = construction.geometry_queries

        def reserve_queries(count: int) -> None:
            nonlocal cover_queries
            if cover_queries + count > limits.maximum_geometry_queries:
                from ...meshing._contracts import MeshingFailure, MeshingFailureCategory

                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Native CAD source cover exhausted the original geometry-query allowance.",
                    requested=(
                        ("maximum_geometry_queries", limits.maximum_geometry_queries),
                    ),
                    achieved=(("geometry_queries", cover_queries + count),),
                )
            cover_queries += count

        with owner.bound_stage(remaining_work, remaining_memory):
            cover = source.boundary_chart_cover(
                remaining_work,
                budget=owner,
                reserve_queries=reserve_queries,
            )
    covering_bound = float(np.max(cover.deviation_bounds, initial=0.0))
    if (
        np.any(construction.unresolved)
        or not cover.complete
        or cover.semantics != "certified"
        or not np.isfinite(covering_bound)
        or covering_bound > policy.linear_deflection
        or np.any(construction.triangle_deviation_bounds > policy.linear_deflection)
        or np.any(construction.curve_edge_deviation_bounds > policy.linear_deflection)
        or np.any(construction.triangle_normal_bounds > policy.angular_deflection)
    ):
        raise ValueError(
            "Native CAD tessellation exhausted its source fidelity or topology "
            f"budget (unresolved={int(np.sum(construction.unresolved))}, "
            f"cover_complete={cover.complete}, cover_semantics={cover.semantics!r}, "
            f"cover_bound={covering_bound:.17g}, "
            f"triangle_bound={float(np.max(construction.triangle_deviation_bounds, initial=0.0)):.17g}, "
            f"curve_bound={float(np.max(construction.curve_edge_deviation_bounds, initial=0.0)):.17g}, "
            f"normal_bound={float(np.max(construction.triangle_normal_bounds, initial=0.0)):.17g}, "
            f"chart_restrictions={construction.chart_restriction_vertices.size})."
        )
    source_indices = construction.vertex_source_indices.copy()
    for dimension in range(3):
        selected = construction.vertex_source_dimensions == dimension
        source_indices[selected] = np.asarray(
            domain.source_indices[dimension], dtype=np.int64
        )[source_indices[selected]]
    template = BRepTessellation(
        construction.vertices,
        construction.triangles.astype(np.int32),
        construction.triangle_patches.astype(np.int32),
        construction.triangle_parameters,
        construction.triangle_deviation_bounds,
        construction.triangle_normal_bounds,
        construction.vertex_source_dimensions,
        source_indices,
        construction.vertex_parameters,
        construction.chart_restriction_vertices,
        construction.chart_restriction_edges,
        construction.chart_restriction_endpoint_parameters,
        construction.chart_restriction_parameters,
        np.full((construction.triangles.shape[0],), -1, dtype=np.int32),
        np.full((construction.vertices.shape[0],), -1, dtype=np.int32),
    )
    placed = _place_occurrences(template, geometry, policy.maximum_triangles)
    if np.any(placed.deviation_bounds > policy.linear_deflection) or np.any(
        placed.normal_bounds > policy.angular_deflection
    ):
        raise ValueError(
            "Placed CAD tessellation misses the requested continuous pose fidelity."
        )
    return placed


__all__ = ["BRepTessellation", "tessellate_brep"]
