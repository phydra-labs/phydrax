#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepare source-stratum incidence and native B-Rep domain views."""

from __future__ import annotations

from typing import assert_never, final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.geometry._domain_numerics as domain_numerics_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.geometry.brep._constructors as constructors_owner
import phydrax.geometry.brep._correspondence as correspondence_owner
import phydrax.geometry.brep._model as model_owner
import phydrax.geometry.brep._projection_contracts as projection_contracts_owner

from .._strict import StrictModule
from .._validation import finite_real_scalar, nonnegative_integer
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked
from ._atlas import AbstractTrimCurve, BoundaryAtlas
from ._interval_enclosure import interval_add
from .brep._coordinate_request import coordinate_source_request
from .brep._intersection import NativePeriodEndpoint, RootEndpoint
from .brep._intersection_curve import IntersectionPCurve
from .brep._model import BRepGeometry, BRepModel
from .brep._patches import AbstractCurve, AbstractSurfacePatch
from .brep._placed import PlacedCurve, PlacedSurface, source_transform_bounds
from .brep._request_preparations import TrimRequestPreparations
from .brep._root_bindings import BRepVertexRoot


@final
class SourceStrata(StrictModule):
    """Unique occurrence/definition pairs in declaration order."""

    indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    paths: tuple[tuple[tuple[str, ...], ...], ...] = eqx.field(static=True)


@final
class SourceRegions(StrictModule):
    """Unique region occurrence/definition pairs in declaration order."""

    indices: tuple[int, ...] = eqx.field(static=True)
    paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)


def prepare_source_strata(
    counts: tuple[int, int, int],
    source_indices: tuple[tuple[int, ...], ...] | None,
    source_occurrences: tuple[tuple[tuple[str, ...], ...], ...] | None,
) -> SourceStrata:
    """Normalize and validate identity pairs for corners, curves, and surfaces."""
    indices = (
        tuple(tuple(range(count)) for count in counts)
        if source_indices is None
        else tuple(
            tuple(nonnegative_integer(index, "source index") for index in row)
            for row in source_indices
        )
    )
    paths = (
        tuple(((),) * count for count in counts)
        if source_occurrences is None
        else source_occurrences
    )
    if (
        not isinstance(paths, tuple)
        or len(indices) != 3
        or len(paths) != 3
        or any(
            len(row) != count
            or len(path_row) != count
            or len(set(zip(path_row, row, strict=True))) != count
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(name, str) or not name for name in path)
                for path in path_row
            )
            for row, path_row, count in zip(indices, paths, counts, strict=True)
        )
    ):
        raise ValueError(
            "MeshingDomain.source_strata must contain unique occurrence/definition pairs aligned with declared counts; got invalid source indices or occurrence paths"
        )
    return SourceStrata(indices, paths)


def prepare_source_regions(
    region_count: int,
    region_source_indices: tuple[int, ...] | None,
    region_source_occurrences: tuple[tuple[str, ...], ...] | None,
) -> SourceRegions:
    """Normalize and validate region identity pairs without inferring sharing."""
    region_indices = (
        tuple(range(region_count))
        if region_source_indices is None
        else tuple(
            nonnegative_integer(index, "region source index")
            for index in region_source_indices
        )
    )
    region_paths = (
        ((),) * region_count
        if region_source_occurrences is None
        else region_source_occurrences
    )
    if (
        not isinstance(region_paths, tuple)
        or len(region_indices) != region_count
        or len(region_paths) != region_count
        or len(set(zip(region_paths, region_indices, strict=True))) != region_count
        or any(
            not isinstance(path, tuple)
            or any(not isinstance(name, str) or not name for name in path)
            for path in region_paths
        )
    ):
        raise ValueError(
            "MeshingDomain.source_regions must contain unique occurrence/definition pairs aligned with region counts; got invalid source indices or occurrence paths"
        )
    return SourceRegions(region_indices, region_paths)


@final
class _BRepStrata(StrictModule):
    """Native edge ownership and vertex indices in source declaration order."""

    ranges: np.ndarray
    first_senses: dict[int, int]
    source_edges: tuple[int, ...] = eqx.field(static=True)
    local_edges: dict[int, int]
    source_vertices: tuple[int, ...] = eqx.field(static=True)
    local_vertices: dict[int, int]
    original_endpoints: tuple[tuple[int, int], ...] = eqx.field(static=True)
    endpoints: tuple[tuple[int, int], ...] = eqx.field(static=True)
    owner_ranges: np.ndarray


@final
class _BRepPatches(StrictModule):
    """Native source patches with their declared oriented coedge loops."""

    values: tuple[domain_owner.MeshingSurfacePatch, ...]


def has_brep_gaps(geometry: BRepGeometry, /) -> bool:
    """Whether source bounds declare a positive physical discrepancy."""
    return (
        geometry.edge_vertex_gap_bounds is not None
        and any(any(row) for row in geometry.edge_vertex_gap_bounds)
    ) or (
        geometry.edge_face_gap_bounds is not None and any(geometry.edge_face_gap_bounds)
    )


def _gap_coedge_range(
    geometry: BRepGeometry,
    coedge: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[float, float]:
    """Reuse the source owner's full endpoint enclosures for a coedge proof."""
    ends = geometry.edge_parameter_enclosure(
        geometry.coedge_edges[coedge], preparations=preparations, budget=budget
    )
    for endpoint, root in enumerate(geometry.coedge_endpoint_roots[coedge]):
        if root is not None:
            lower, upper = root.parameter_enclosure(
                preparations=preparations, budget=budget
            )
            ends[endpoint, 0] = max(ends[endpoint, 0], lower)
            ends[endpoint, 1] = min(ends[endpoint, 1], upper)
    if np.any(ends[:, 0] > ends[:, 1]):
        raise ValueError(
            "MeshingDomain.coedge_roots must have common source endpoint enclosures; got disjoint ranges"
        )
    return float(ends[0, 0]), float(ends[1, 1])


def _gap_face_box(
    geometry: BRepGeometry,
    face: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    """Enclose the authored face trims without using sampled chart positions."""
    boxes = []
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            first, last = _gap_coedge_range(geometry, coedge, preparations, budget)
            pcurve = geometry.pcurves[coedge]
            boxes.append(
                pcurve.bounding_box(
                    first,
                    last,
                    endpoint_roots=geometry.coedge_endpoint_roots[coedge],
                    preparations=preparations,
                    budget=budget,
                )
                if isinstance(pcurve, IntersectionPCurve)
                else pcurve.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
            )
    enclosures = np.asarray(boxes, dtype=np.float64)
    return np.stack((np.min(enclosures[:, 0], axis=0), np.max(enclosures[:, 1], axis=0)))


def _certify_zero_coedge_gaps(
    geometry: BRepGeometry,
    surfaces: tuple[AbstractSurfacePatch, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Require the correspondence owner to prove unsupported zero rows exactly."""
    bounds = geometry.edge_face_gap_bounds
    if bounds is None:
        return
    preparations = TrimRequestPreparations()
    face_boxes: dict[int, np.ndarray] = {}
    ranges = np.asarray(geometry.edge_ranges)
    points = np.asarray(geometry.vertex_points)
    for coedge, bound in enumerate(bounds):
        if bound != 0.0:
            continue
        edge, face = geometry.coedge_edges[coedge], geometry.coedge_faces[coedge]
        curve_index = geometry.edge_curves[edge]
        curve = None if curve_index == -1 else geometry.curves[curve_index]
        with budget.temporary_scope():
            budget.reserve(128, 8192)
            squared = (
                None
                if curve is None
                else model_owner._edge_face_gap_squared(
                    curve,
                    geometry.pcurves[coedge],
                    surfaces[face],
                    float(ranges[edge, 0]),
                    float(ranges[edge, 1]),
                    budget=budget,
                )
            )
        if squared is not None:
            continue
        if face not in face_boxes:
            face_boxes[face] = _gap_face_box(geometry, face, preparations, budget)
        first, last = _gap_coedge_range(geometry, coedge, preparations, budget)
        witness = correspondence_owner.certify_curve_surface(
            curve,
            geometry.pcurves[coedge],
            surfaces[face],
            face_boxes[face],
            first,
            last,
            point=points[geometry.edge_vertices[edge][0]],
            tolerance=0.0,
            endpoint_roots=geometry.coedge_endpoint_roots[coedge],
            preparations=preparations,
            budget=budget,
        )
        if not witness.complete or witness.deviation_bound != 0.0:
            raise ValueError(
                "MeshingDomain.edge_face_gap_bounds must independently prove zero carrier agreement; got an unsupported or nonzero row"
            )


def _validate_brep_gap_rows(geometry: BRepGeometry, /) -> None:
    """Validate source gap-table shapes even after source tree surgery."""
    vertex_bounds, face_bounds = (
        geometry.edge_vertex_gap_bounds,
        geometry.edge_face_gap_bounds,
    )
    if vertex_bounds is not None and (
        len(vertex_bounds) != len(geometry.edge_curves)
        or any(
            len(row) != 2 or any(not np.isfinite(x) or x < 0.0 for x in row)
            for row in vertex_bounds
        )
    ):
        raise ValueError(
            "BRepGeometry.edge_vertex_gap_bounds must align with finite nonnegative endpoint bounds; got invalid rows"
        )
    if face_bounds is not None and (
        len(face_bounds) != len(geometry.coedge_edges)
        or any(not np.isfinite(bound) or bound < 0.0 for bound in face_bounds)
    ):
        raise ValueError(
            "BRepGeometry.edge_face_gap_bounds must align with finite nonnegative coedge bounds; got invalid rows"
        )


def _certify_vertex_gap_bounds(
    geometry: BRepGeometry,
    ranges: np.ndarray,
    points: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Reprove endpoint rows against unchanged source curves and vertices."""
    bounds = geometry.edge_vertex_gap_bounds
    if bounds is None:
        return
    for edge, curve_index in enumerate(geometry.edge_curves):
        if curve_index == -1:
            continue
        model_owner._certify_endpoint_gaps(
            geometry.curves[curve_index],
            points,
            geometry.edge_vertices[edge],
            (float(ranges[edge, 0]), float(ranges[edge, 1])),
            bounds[edge],
            geometry.vertex_roots,
            budget=budget,
            endpoint_roots=geometry.edge_endpoint_roots[edge],
        )


def certify_brep_gap_bounds(
    geometry: BRepGeometry,
    surfaces: tuple[AbstractSurfacePatch, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> None:
    """Reprove declared source gaps before lowering or source certification."""
    vertex_bounds, face_bounds = (
        geometry.edge_vertex_gap_bounds,
        geometry.edge_face_gap_bounds,
    )
    if vertex_bounds is None and face_bounds is None:
        return
    _validate_brep_gap_rows(geometry)
    if any(
        occurrence.rotation != ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        or occurrence.translation != (0.0, 0.0, 0.0)
        for occurrence in geometry.occurrences
    ) or any(isinstance(surface, PlacedSurface) for surface in surfaces):
        raise ValueError(
            "MeshingDomain.gap_bounds must have a placed-source proof; got unsupported placed gap tables"
        )
    if len(surfaces) != len(geometry.face_loops):
        raise ValueError(
            "MeshingDomain.gap_surfaces must align with source faces; got inconsistent face counts"
        )
    with coordinate_source_request(
        (geometry, surfaces), budget=budget, rows=len(geometry.coedge_edges)
    ) as admitted:
        ranges = np.asarray(geometry.edge_ranges)
        points = np.asarray(geometry.vertex_points)
        _certify_vertex_gap_bounds(geometry, ranges, points, admitted)
        if face_bounds is not None:
            for coedge, bound in enumerate(face_bounds):
                edge = geometry.coedge_edges[coedge]
                if bound > 0.0 and any(
                    root is not None and not isinstance(root, NativePeriodEndpoint)
                    for root in (
                        *geometry.edge_endpoint_roots[edge],
                        *geometry.coedge_endpoint_roots[coedge],
                    )
                ):
                    raise ValueError(
                        "MeshingDomain.edge_face_gap_bounds must use authored scalar endpoints; got unsupported rooted gaps"
                    )
            model_owner._certify_edge_face_gap_bounds(
                geometry.curves,
                geometry.edge_curves,
                ranges,
                geometry.pcurves,
                geometry.coedge_edges,
                geometry.coedge_faces,
                surfaces,
                budget=admitted,
                edge_endpoint_roots=geometry.edge_endpoint_roots,
                coedge_endpoint_roots=geometry.coedge_endpoint_roots,
                declared_bounds=face_bounds,
            )
            _certify_zero_coedge_gaps(geometry, surfaces, admitted)


def _prepare_brep_strata(geometry: BRepGeometry) -> _BRepStrata:
    """Prepare nondegenerate source edges and their first-use orientations."""
    ranges = np.asarray(geometry.edge_ranges)
    first_senses: dict[int, int] = {}
    for loops in geometry.face_loops:
        for loop in loops:
            for coedge in loop:
                first_senses.setdefault(
                    geometry.coedge_edges[coedge], geometry.coedge_senses[coedge]
                )
    source_edges = tuple(
        edge for edge in sorted(first_senses) if geometry.edge_curves[edge] >= 0
    )
    local_edges = {edge: local for local, edge in enumerate(source_edges)}
    source_vertices = tuple(
        sorted(
            {vertex for edge in first_senses for vertex in geometry.edge_vertices[edge]}
        )
    )
    local_vertices = {vertex: local for local, vertex in enumerate(source_vertices)}
    original_endpoints = tuple(
        geometry.edge_vertices[edge]
        if first_senses[edge] > 0
        else geometry.edge_vertices[edge][::-1]
        for edge in source_edges
    )
    endpoints = tuple(
        (local_vertices[first], local_vertices[last])
        for first, last in original_endpoints
    )
    owner_ranges = np.asarray(
        [
            ranges[edge] if first_senses[edge] > 0 else ranges[edge][::-1]
            for edge in source_edges
        ]
    )
    return _BRepStrata(
        ranges,
        first_senses,
        source_edges,
        local_edges,
        source_vertices,
        local_vertices,
        original_endpoints,
        endpoints,
        owner_ranges,
    )


def _prepare_brep_patches(
    geometry: BRepGeometry,
    surfaces: tuple[AbstractSurfacePatch, ...],
    signs: np.ndarray,
    strata: _BRepStrata,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> _BRepPatches:
    """Lower oriented source coedges while retaining endpoint and spatial roots."""
    ranges, local_edges, local_vertices = (
        strata.ranges,
        strata.local_edges,
        strata.local_vertices,
    )
    patches = []
    for face, loops in enumerate(geometry.face_loops):
        uses = []
        for loop in loops:
            members = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                first, last = ranges[edge]
                if geometry.coedge_senses[coedge] < 0:
                    first, last = last, first
                pcurve = geometry.pcurves[coedge]
                if edge in local_edges:
                    roots = geometry.coedge_endpoint_roots[coedge]
                    vertices = geometry.edge_vertices[edge]
                    if geometry.coedge_senses[coedge] < 0:
                        roots, vertices = roots[::-1], vertices[::-1]
                    members.append(
                        domain_owner.PatchCurveUse(
                            local_edges[edge],
                            pcurve,
                            first,
                            last,
                            first_root=roots[0],
                            last_root=roots[1],
                            start_vertex_root=geometry.vertex_roots[vertices[0]],
                            end_vertex_root=geometry.vertex_roots[vertices[1]],
                            trim_curve=constructors_owner.brep_trim_curve(
                                geometry, coedge, surfaces[face], budget=budget
                            ),
                        )
                    )
                else:
                    ends = np.asarray(pcurve.evaluate(jnp.asarray((first, last))))
                    vertex = geometry.edge_vertices[edge][0]
                    members.append(
                        domain_owner.PatchPoleUse(
                            local_vertices[vertex],
                            tuple(ends[0]),
                            tuple(ends[1]),
                            trim_curve=constructors_owner.brep_trim_curve(
                                geometry, coedge, surfaces[face], budget=budget
                            ),
                        )
                    )
            uses.append(tuple(members))
        patches.append(
            domain_owner.MeshingSurfacePatch(
                surfaces[face],
                tuple(uses),
                reversed=signs[face] < 0,
            )
        )
    return _BRepPatches(tuple(patches))


def prepare_brep_domain(
    domain_type: type[domain_owner.MeshingDomain],
    geometry: BRepGeometry,
    surfaces: tuple[AbstractSurfacePatch, ...],
    orientation: np.ndarray,
    *,
    source_id: str,
    source_revision: str,
    authority_id: str,
    budget: CoordinateEnclosureBudget | None,
    authority: BRepGeometry | BRepModel,
) -> domain_owner.MeshingDomain:
    """Prepare native strata, patch incidence, regions, and physical curve evidence."""
    signs = np.asarray(orientation, dtype=np.float64)
    if len(surfaces) != len(geometry.face_loops) or signs.shape != (len(surfaces),):
        raise ValueError(
            "MeshingDomain.brep_surfaces must align orientation and carrier arrays with native face loops; got inconsistent face counts or orientation shape"
        )
    if not np.all(np.isin(signs, (-1, 1))):
        raise ValueError(
            "MeshingDomain.brep_orientation must be -1 or +1; got invalid face signs"
        )
    certify_brep_gap_bounds(geometry, surfaces, budget=budget)
    strata = _prepare_brep_strata(geometry)
    prepared_patches = _prepare_brep_patches(
        geometry, surfaces, signs, strata, budget=budget
    )
    patches = prepared_patches.values
    source_vertices, source_edges = strata.source_vertices, strata.source_edges
    endpoints, owner_ranges = strata.endpoints, strata.owner_ranges
    first_senses, original_endpoints = strata.first_senses, strata.original_endpoints
    regions = []
    for solid, shells in enumerate(geometry.solid_shells):
        boundary = []
        for shell in shells:
            for face, sign in zip(
                geometry.shell_faces[shell],
                geometry.shell_orientations[shell],
                strict=True,
            ):
                boundary.append((face, sign))
        regions.append(
            domain_owner.MeshingDomainRegion(f"solid:{solid}", tuple(boundary))
        )
    domain = domain_type(
        tuple(patches),
        tuple(domain_owner.MeshingDomainCurve(*ends) for ends in endpoints),
        len(source_vertices),
        source_id=source_id,
        source_revision=source_revision,
        regions=tuple(regions),
        source_indices=(
            source_vertices,
            source_edges,
            tuple(range(len(patches))),
        ),
        source_kinds=("vertex", "edge", "face"),
        authority_id=authority_id or geometry.geometry_id,
        _native_source=DefinitionDomainSource(
            authority,
            geometry,
            source_edges,
            source_vertices,
            owner_ranges,
            first_senses,
            original_endpoints,
            source_id,
        ),
    )
    return domain


@final
class _PlacedIncidence(StrictModule):
    """Occurrence-qualified source closure with one placement for every shared stratum."""

    geometry: BRepGeometry
    outgoing: dict[model_owner.BRepEntityId, list[model_owner.BRepQualifiedIncidence]]
    placements: dict[model_owner.BRepEntityId, model_owner.BRepOccurrence]
    solid_ids: list[model_owner.BRepEntityId]
    face_ids: list[model_owner.BRepEntityId]
    vertex_ids: list[model_owner.BRepEntityId]
    nondegenerate: list[model_owner.BRepEntityId]
    faces: dict[model_owner.BRepEntityId, int]
    edges: dict[model_owner.BRepEntityId, int]
    vertices: dict[model_owner.BRepEntityId, int]
    transforms: dict[tuple[str, ...], tuple[Array, Array]]

    def transform(self, entity: model_owner.BRepEntityId) -> tuple[Array, Array]:
        occurrence = self.placements.get(entity)
        return (
            (jnp.eye(3, dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64))
            if occurrence is None
            else self.transforms[occurrence.path]
        )

    def child(
        self,
        container: model_owner.BRepEntityId,
        kind: Literal["vertex", "edge", "face", "solid"],
        *,
        definition_index: int | None = None,
        use_index: int | None = None,
    ) -> model_owner.BRepEntityId:
        matches = [
            record.member
            for record in self.outgoing.get(container, ())
            if record.member.kind == kind
            and (definition_index is None or record.member.index == definition_index)
            and (use_index is None or record.use_index == use_index)
        ]
        if len(set(matches)) != 1:
            raise ValueError(
                f"MeshingDomain.qualified_incidence must identify one source member per use; got {len(set(matches))} matching members"
            )
        return matches[0]


@final
class _PlacedPatches(StrictModule):
    """Placed source patches and the first-use orientation of their native edges."""

    values: list[domain_owner.MeshingSurfacePatch]
    first_senses: dict[model_owner.BRepEntityId, int]
    ranges: np.ndarray


@final
class _PlacedCurves(StrictModule):
    """Placed physical carriers with endpoint realization bounds."""

    values: list[domain_owner.MeshingDomainCurve]
    carriers: list[PlacedCurve]
    ranges: list[np.ndarray]
    vertex_errors: list[np.ndarray]


def _prepare_placed_incidence(model: BRepModel) -> _PlacedIncidence:
    """Validate authored shared placements and index their qualified source closure."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "MeshingDomain.placed_model.geometry must contain native B-Rep carriers and incidence; got None"
        )
    records = geometry.qualified_entity_incidence(model.source_revision)
    outgoing = {}
    for record in records:
        if record.container != record.member:
            outgoing.setdefault(record.container, []).append(record)
    occurrences = {occurrence.path: occurrence for occurrence in geometry.occurrences}
    placements = {}
    solid_ids = []
    for occurrence in geometry.occurrences:
        solid = model_owner.BRepEntityId(
            model.source_revision, "solid", occurrence.solid, occurrence.path
        )
        solid_ids.append(solid)
        pending, seen = [solid], set()
        while pending:
            entity = pending.pop()
            if entity in seen:
                continue
            seen.add(entity)
            old = placements.setdefault(entity, occurrence)
            if (
                old.rotation != occurrence.rotation
                or old.translation != occurrence.translation
            ):
                raise ValueError(
                    "MeshingDomain.shared_placements must agree on rotation and translation; got inconsistent occurrence poses"
                )
            pending.extend(record.member for record in outgoing.get(entity, ()))
    face_ids = sorted(
        {record.member for record in records if record.member.kind == "face"}
    )
    edge_ids = sorted(
        {record.member for record in records if record.member.kind == "edge"}
    )
    vertex_ids = sorted(
        {record.member for record in records if record.member.kind == "vertex"}
    )
    nondegenerate = [edge for edge in edge_ids if geometry.edge_curves[edge.index] >= 0]
    faces = {entity: index for index, entity in enumerate(face_ids)}
    edges = {entity: index for index, entity in enumerate(nondegenerate)}
    vertices = {entity: index for index, entity in enumerate(vertex_ids)}
    transforms = {
        path: (
            jnp.asarray(occurrence.rotation, dtype=jnp.float64),
            jnp.asarray(occurrence.translation, dtype=jnp.float64),
        )
        for path, occurrence in occurrences.items()
    }
    return _PlacedIncidence(
        geometry,
        outgoing,
        placements,
        solid_ids,
        face_ids,
        vertex_ids,
        nondegenerate,
        faces,
        edges,
        vertices,
        transforms,
    )


def _prepare_placed_patches(
    model: BRepModel,
    graph: _PlacedIncidence,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> _PlacedPatches:
    """Lower coedge roots and oriented patches under their declared occurrence poses."""
    geometry, face_ids, edges, vertices = (
        graph.geometry,
        graph.face_ids,
        graph.edges,
        graph.vertices,
    )
    patch_values, first_senses = [], {}
    ranges = np.asarray(geometry.edge_ranges)
    for face in face_ids:
        loops = []
        for loop in geometry.face_loops[face.index]:
            uses = []
            for coedge in loop:
                edge = graph.child(face, "edge", use_index=coedge)
                first, last = ranges[edge.index]
                sense = geometry.coedge_senses[coedge]
                if sense < 0:
                    first, last = last, first
                first_senses.setdefault(edge, sense)
                pcurve = geometry.pcurves[coedge]
                if edge in edges:
                    roots = geometry.coedge_endpoint_roots[coedge]
                    endpoints = geometry.edge_vertices[edge.index]
                    if sense < 0:
                        roots, endpoints = roots[::-1], endpoints[::-1]
                    uses.append(
                        domain_owner.PatchCurveUse(
                            edges[edge],
                            pcurve,
                            first,
                            last,
                            first_root=roots[0],
                            last_root=roots[1],
                            start_vertex_root=geometry.vertex_roots[endpoints[0]],
                            end_vertex_root=geometry.vertex_roots[endpoints[1]],
                            trim_curve=constructors_owner.brep_trim_curve(
                                geometry, coedge, model.patches[face.index], budget=budget
                            ),
                        )
                    )
                else:
                    vertex = graph.child(
                        edge,
                        "vertex",
                        definition_index=geometry.edge_vertices[edge.index][0],
                    )
                    points = np.asarray(pcurve.evaluate(jnp.asarray((first, last))))
                    uses.append(
                        domain_owner.PatchPoleUse(
                            vertices[vertex],
                            tuple(points[0]),
                            tuple(points[1]),
                            trim_curve=constructors_owner.brep_trim_curve(
                                geometry, coedge, model.patches[face.index], budget=budget
                            ),
                        )
                    )
            loops.append(tuple(uses))
        rotation, translation = graph.transform(face)
        patch_values.append(
            domain_owner.MeshingSurfacePatch(
                PlacedSurface(model.patches[face.index], rotation, translation),
                tuple(loops),
                reversed=float(model.orientation[face.index]) < 0,
            )
        )
    return _PlacedPatches(patch_values, first_senses, ranges)


def _prepare_placed_curves(
    graph: _PlacedIncidence, patches: _PlacedPatches
) -> _PlacedCurves:
    """Transport native curve carriers and bound their vertex realization error."""
    geometry, nondegenerate, vertices = (
        graph.geometry,
        graph.nondegenerate,
        graph.vertices,
    )
    first_senses = patches.first_senses
    ranges = patches.ranges
    curve_values, curve_carriers, owner_ranges, vertex_errors = [], [], [], []
    for edge in nondegenerate:
        endpoints = geometry.edge_vertices[edge.index]
        if first_senses[edge] < 0:
            endpoints = endpoints[::-1]
        start, end = [
            graph.child(edge, "vertex", definition_index=endpoint)
            for endpoint in endpoints
        ]
        curve_values.append(
            domain_owner.MeshingDomainCurve(vertices[start], vertices[end])
        )
        rotation, translation = graph.transform(edge)
        curve_carriers.append(
            PlacedCurve(
                geometry.curves[geometry.edge_curves[edge.index]], rotation, translation
            )
        )
        owner_ranges.append(
            ranges[edge.index] if first_senses[edge] > 0 else ranges[edge.index][::-1]
        )
        original_points = np.asarray(geometry.vertex_points)[
            np.asarray(endpoints, dtype=np.int64)
        ]
        placement_roundoff = (
            64
            * np.finfo(np.float64).eps
            * np.linalg.norm(
                np.abs(np.asarray(rotation)) @ np.abs(original_points).T
                + np.abs(np.asarray(translation))[:, None],
                axis=0,
            )
        )
        vertex_errors.append(
            np.nextafter(
                domain_numerics_owner.matrix_norm_upper(np.asarray(rotation))
                * np.asarray(geometry.vertex_evaluation_bounds)[
                    np.asarray(endpoints, dtype=np.int64)
                ]
                + placement_roundoff,
                np.inf,
            )
        )
    return _PlacedCurves(curve_values, curve_carriers, owner_ranges, vertex_errors)


def prepare_placed_brep_domain(
    model: BRepModel,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> domain_owner.MeshingDomain:
    """Build a placed meshing view from qualified incidence without spatial welding."""
    certify_brep_gap_bounds(model.require_geometry(), model.patches, budget=budget)
    graph = _prepare_placed_incidence(model)
    prepared_patches = _prepare_placed_patches(model, graph, budget=budget)
    prepared_curves = _prepare_placed_curves(graph, prepared_patches)
    vertex_ids, nondegenerate, face_ids = (
        graph.vertex_ids,
        graph.nondegenerate,
        graph.face_ids,
    )
    faces, outgoing, solid_ids = graph.faces, graph.outgoing, graph.solid_ids
    patch_values = prepared_patches.values
    curve_values = prepared_curves.values
    regions = tuple(
        domain_owner.MeshingDomainRegion(
            projection_contracts_owner.brep_entity_id(
                model.source_revision,
                3,
                solid.index,
                occurrence_path=solid.occurrence_path,
            ),
            tuple(
                (faces[record.member], record.orientation)
                for record in outgoing.get(solid, ())
                if record.member.kind == "face"
            ),
        )
        for solid in solid_ids
    )
    strata = (vertex_ids, nondegenerate, face_ids)
    domain = domain_owner.MeshingDomain(
        tuple(patch_values),
        tuple(curve_values),
        len(vertex_ids),
        source_id=model.source_id,
        source_revision=model.source_revision,
        regions=regions,
        source_indices=tuple(tuple(entity.index for entity in row) for row in strata),
        source_occurrences=tuple(
            tuple(entity.occurrence_path for entity in row) for row in strata
        ),
        region_source_indices=tuple(solid.index for solid in solid_ids),
        region_source_occurrences=tuple(solid.occurrence_path for solid in solid_ids),
        source_kinds=("vertex", "edge", "face"),
        authority_id=model.model_id,
        _native_source=PlacedDomainSource(
            model, graph, prepared_curves, prepared_patches
        ),
    )
    return domain


@final
class CurveParameterRange(StrictModule):
    """A nonempty finite parameter traversal after curve-dimension admission."""

    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)


@final
class CurveRootBindings(StrictModule):
    """Vertex-root and trim capabilities admitted after scalar-root realization."""

    start: BRepVertexRoot | None
    end: BRepVertexRoot | None
    trim: AbstractTrimCurve | None


@final
class PoleSide(StrictModule):
    """A declared corner and trim capability for a validated straight chart side."""

    corner: int = eqx.field(static=True)
    start: tuple[float, float] = eqx.field(static=True)
    end: tuple[float, float] = eqx.field(static=True)
    trim: AbstractTrimCurve | None


@checked
def prepare_curve_parameter_range(
    pcurve: AbstractCurve | AbstractTrimCurve,
    first: float,
    last: float,
) -> CurveParameterRange:
    """Admit the native parameter carrier before validating its finite traversal."""
    match pcurve:
        case AbstractCurve():
            if pcurve.ambient_dimension != 2:
                raise TypeError(
                    "PatchCurveUse.pcurve must have ambient dimension 2; "
                    f"got {pcurve.ambient_dimension}"
                )
        case AbstractTrimCurve():
            pass
        case _:
            assert_never(pcurve)
    first_ = finite_real_scalar(float(first), "PatchCurveUse.first")
    last_ = finite_real_scalar(float(last), "PatchCurveUse.last")
    if first_ == last_:
        raise ValueError(
            f"PatchCurveUse.parameter_range must be nonempty; got ({first_}, {last_})"
        )
    return CurveParameterRange(first_, last_)


@checked
def validate_root_parameter(parameter: float, endpoint: RootEndpoint) -> None:
    """Require the realized scalar parameter to lie in its source-root enclosure."""
    lower, upper = endpoint.parameter_enclosure()
    if not lower <= parameter <= upper:
        raise ValueError(
            "PatchCurveUse.endpoint_parameter must lie in its source-root enclosure; "
            f"got {parameter} outside [{lower}, {upper}]"
        )


@checked
def prepare_curve_root_bindings(
    start: BRepVertexRoot | None,
    end: BRepVertexRoot | None,
    trim: AbstractTrimCurve | None,
) -> CurveRootBindings:
    """Admit ordered vertex-root and trim capability fields for a curve use."""
    return CurveRootBindings(start, end, trim)


@checked
def prepare_pole_side(
    corner: int,
    start: tuple[float, float],
    end: tuple[float, float],
    trim: AbstractTrimCurve | None,
) -> PoleSide:
    """Admit the trim capability before normalizing the declared corner index."""
    return PoleSide(nonnegative_integer(corner, "corner"), start, end, trim)


@final
class NativeDomainPhysicalState(StrictModule):
    """Native physical carriers and vertex positions after domain validation."""

    atlas: BoundaryAtlas
    points: np.ndarray


@final
class DefinitionDomainSource(StrictModule):
    """Native definition inputs retaining the caller's BRep source object."""

    authority: BRepGeometry | BRepModel
    geometry: BRepGeometry
    source_edges: tuple[int, ...] = eqx.field(static=True)
    source_vertices: tuple[int, ...] = eqx.field(static=True)
    owner_ranges: np.ndarray
    first_senses: dict[int, int]
    original_endpoints: tuple[tuple[int, int], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def validation_gap_bounds(
        self,
    ) -> tuple[dict[tuple[int, int, int], float], np.ndarray]:
        """Map source coedges and first-sense-oriented endpoint bounds."""
        geometry = self.geometry
        face_bounds = geometry.edge_face_gap_bounds
        use_bounds = (
            {}
            if face_bounds is None
            else {
                (face, loop, position): face_bounds[coedge]
                for face, loops in enumerate(geometry.face_loops)
                for loop, coedges in enumerate(loops)
                for position, coedge in enumerate(coedges)
            }
        )
        vertex_bounds = geometry.edge_vertex_gap_bounds
        endpoint_bounds = (
            np.zeros((len(self.source_edges), 2), dtype=np.float64)
            if vertex_bounds is None
            else np.asarray(
                [
                    vertex_bounds[edge]
                    if self.first_senses[edge] > 0
                    else vertex_bounds[edge][::-1]
                    for edge in self.source_edges
                ],
                dtype=np.float64,
            ).reshape((-1, 2))
        )
        return use_bounds, endpoint_bounds

    def physical_state(
        self, domain: domain_owner.MeshingDomain
    ) -> NativeDomainPhysicalState:
        """Build physical curve evidence after the domain's parametric admission."""
        geometry, source_edges, source_vertices = (
            self.geometry,
            self.source_edges,
            self.source_vertices,
        )
        owner_ranges, first_senses = self.owner_ranges, self.first_senses
        original_endpoints, source_id = self.original_endpoints, self.source_id
        atlas = BoundaryAtlas(
            domain_owner.PhysicalCurveMap(
                tuple(
                    geometry.curves[geometry.edge_curves[edge]] for edge in source_edges
                ),
                owner_ranges,
                tuple(
                    geometry.edge_endpoint_roots[edge]
                    if first_senses[edge] > 0
                    else geometry.edge_endpoint_roots[edge][::-1]
                    for edge in source_edges
                ),
                np.asarray(geometry.vertex_evaluation_bounds)[
                    np.asarray(original_endpoints, dtype=np.int64).reshape((-1, 2))
                ],
            ),
            source_entity_ids=jnp.arange(len(source_edges), dtype=jnp.int32),
            source_id=source_id,
            physical_tags=tuple(f"edge:{edge}" for edge in source_edges),
        )
        points = np.array(
            np.asarray(geometry.vertex_points)[np.asarray(source_vertices)],
            dtype=np.float64,
            copy=True,
        )
        points.setflags(write=False)
        return NativeDomainPhysicalState(atlas, points)


@final
class PlacedDomainSource(StrictModule):
    """Qualified placed inputs retaining the caller's model and prepared incidence."""

    authority: BRepModel
    graph: _PlacedIncidence
    curves: _PlacedCurves
    patches: _PlacedPatches

    def validation_corners(self) -> tuple[np.ndarray, np.ndarray]:
        """Transport source vertices and their evaluation enclosures into world space."""
        geometry, graph = self.graph.geometry, self.graph
        points, errors = [], []
        for vertex in graph.vertex_ids:
            rotation, translation = graph.transform(vertex)
            matrix, shift = np.asarray(rotation), np.asarray(translation)
            source = np.asarray(geometry.vertex_points)[vertex.index]
            error = float(geometry.vertex_evaluation_bounds[vertex.index])
            point = matrix @ source + shift
            low, high = source_transform_bounds(
                matrix,
                np.nextafter(source - error, -np.inf),
                np.nextafter(source + error, np.inf),
            )
            low, high = interval_add((low, high), (shift, shift))
            if not np.all(np.isfinite((low, high, point))):
                raise ValueError(
                    "MeshingDomain.placed_corner_enclosure must be finite; "
                    f"got a nonfinite enclosure for vertex {vertex.index}"
                )
            # Directed multiply/add bounds own the pose's operation count;
            # 8 eps covers the 3D norm's six roundings and radius subtraction.
            radius = np.linalg.norm(np.maximum(np.abs(low - point), np.abs(high - point)))
            errors.append(
                np.nextafter(radius * (1 + 8 * np.finfo(np.float64).eps), np.inf)
            )
            points.append(point)
        values = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        values.setflags(write=False)
        return values, np.asarray(errors, dtype=np.float64)

    def physical_state(
        self, domain: domain_owner.MeshingDomain
    ) -> NativeDomainPhysicalState:
        """Build placed vertex and curve evidence in the existing native charge order."""
        model, graph = self.authority, self.graph
        geometry, nondegenerate = (
            graph.geometry,
            graph.nondegenerate,
        )
        curve_values, curve_carriers, owner_ranges, vertex_errors = (
            self.curves.values,
            self.curves.carriers,
            self.curves.ranges,
            self.curves.vertex_errors,
        )
        first_senses = self.patches.first_senses
        atlas = BoundaryAtlas(
            domain_owner.PhysicalCurveMap(
                tuple(curve_carriers),
                np.asarray(owner_ranges),
                tuple(
                    geometry.edge_endpoint_roots[edge.index]
                    if first_senses[edge] > 0
                    else geometry.edge_endpoint_roots[edge.index][::-1]
                    for edge in nondegenerate
                ),
                np.asarray(vertex_errors, dtype=np.float64).reshape((-1, 2)),
            ),
            source_entity_ids=jnp.arange(len(curve_values), dtype=jnp.int32),
            source_id=model.source_id,
            physical_tags=tuple(
                domain.entity_id(1, index) for index in range(len(curve_values))
            ),
        )
        return NativeDomainPhysicalState(atlas, domain.corner_points)


type NativeDomainSource = DefinitionDomainSource | PlacedDomainSource
