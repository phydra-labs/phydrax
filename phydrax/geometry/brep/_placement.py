#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact occurrence-to-world CAD cutover with explicit source incidence lineage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..._fingerprint import canonical_fingerprint
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADOccurrence,
    CADRevision,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._constructors import assemble_brep_model, BRepTessellationPolicy
from ._intersection import RootEndpoint
from ._model import BRepEntityId, BRepGeometry, BRepModel, BRepOccurrence, BRepPCurve
from ._patches import AbstractSurfacePatch
from ._placed import _exact_determinant, PlacedCurve, PlacedSurface
from ._projection_contracts import brep_entity_id
from ._root_bindings import BRepPlacedVertex, BRepVertexRoot


@dataclass(frozen=True, slots=True)
class BRepPlacementResult:
    model: BRepModel
    association_graph: AssociationGraph
    source_entities: tuple[BRepEntityId, ...]
    target_entities: tuple[BRepEntityId, ...]
    certificate_id: str


def _entity_text(entity: BRepEntityId, /) -> str:
    dimensions = {"vertex": 0, "edge": 1, "face": 2, "solid": 3}
    return brep_entity_id(
        entity.source_revision,
        dimensions[entity.kind],
        entity.index,
        occurrence_path=entity.occurrence_path,
    )


def _entity_occurrence(entity: BRepEntityId, /) -> str:
    return canonical_fingerprint(
        {"kind": "cad-qualified-entity-occurrence", "entity": _entity_text(entity)}
    )


def qualified_placement_revision(model: BRepModel, /) -> CADRevision:
    """Inventory every authored physical entity, with its exact instance namespace."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError("A placement revision requires exact native geometry.")
    entities = sorted(
        {
            record.member
            for record in geometry.qualified_entity_incidence(model.source_revision)
        }
    )
    namespaces = sorted(
        {entity.occurrence_path for entity in entities if entity.occurrence_path}
        | {container.path for container in geometry.assembly_containers}
        | {occurrence.path for occurrence in geometry.occurrences}
    )
    by_namespace = {
        namespace: canonical_fingerprint(
            {
                "kind": "cad-instance-namespace",
                "source_revision": model.source_revision,
                "path": namespace,
            }
        )
        for namespace in namespaces
    }
    parents = {
        child: container.path
        for container in geometry.assembly_containers
        for child in (*container.member_paths, *container.child_paths)
    }
    namespace_paths: dict[tuple[str, ...], tuple[str, ...]] = {}

    def namespace_path(namespace: tuple[str, ...]) -> tuple[str, ...]:
        if namespace not in namespace_paths:
            parent = parents.get(namespace)
            namespace_paths[namespace] = (
                (by_namespace[namespace],)
                if parent is None
                else (*namespace_path(parent), by_namespace[namespace])
            )
        return namespace_paths[namespace]

    occurrences: list[CADOccurrence] = []
    for namespace in namespaces:
        identifier = by_namespace[namespace]
        parent = parents.get(namespace)
        occurrences.append(
            CADOccurrence(
                model.source_revision,
                identifier,
                identifier,
                "assembly",
                namespace_path(namespace),
                None if parent is None else by_namespace[parent],
            )
        )
    for entity in entities:
        identifier = _entity_occurrence(entity)
        parent = by_namespace.get(entity.occurrence_path)
        path = (
            (identifier,)
            if parent is None
            else (*namespace_path(entity.occurrence_path), identifier)
        )
        occurrences.append(
            CADOccurrence(
                model.source_revision,
                identifier,
                _entity_text(entity),
                entity.kind,
                path,
                parent,
            )
        )
    return CADRevision(
        model.source_revision, model.source_id, tuple(occurrences), geometry.geometry_id
    )


def materialize_brep_occurrences(
    model: BRepModel,
    /,
    *,
    tessellation: BRepTessellationPolicy,
) -> BRepPlacementResult:
    """Publish one exact world model, retaining independent authored instances.

    Topology sharing follows qualified incidence alone. Curves, patches and
    vertices remain authored pose operation trees around the original exact
    sources, including complete implicit root/branch graphs. UV parameters and
    source scalar endpoint constraints are unchanged by spatial placement.
    """
    if not isinstance(model, BRepModel) or model.geometry is None:
        raise ValueError("World occurrence materialization requires an exact BRepModel.")
    if not isinstance(tessellation, BRepTessellationPolicy):
        raise TypeError("tessellation must be the actual BRepTessellationPolicy.")
    original = model.geometry
    records = original.qualified_entity_incidence(model.source_revision)
    incidence = {
        (record.container, record.member.kind, record.member.index): record.member
        for record in records
    }
    mapped: dict[BRepEntityId, int] = {}
    points: list[np.ndarray] = []
    roots: list[BRepVertexRoot | None] = []
    curves: list[PlacedCurve] = []
    edge_curves: list[int] = []
    edge_ranges: list[tuple[float, float]] = []
    edge_vertices: list[tuple[int, int]] = []
    edge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = []
    pcurves: list[BRepPCurve] = []
    coedge_edges: list[int] = []
    coedge_senses: list[int] = []
    coedge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = []
    patches: list[AbstractSurfacePatch] = []
    bounds: list[np.ndarray] = []
    orientation: list[float] = []
    tags: list[str] = []
    face_loops: list[tuple[tuple[int, ...], ...]] = []
    shell_faces: list[tuple[int, ...]] = []
    shell_orientations: list[tuple[int, ...]] = []
    solid_shells: list[tuple[int, ...]] = []
    occurrences: list[BRepOccurrence] = []
    sources: list[BRepEntityId] = []
    targets: list[tuple[str, int, tuple[str, ...]]] = []
    curve_definitions: dict[tuple[tuple[str, ...], int], int] = {}
    ranges_host = np.asarray(original.edge_ranges)
    points_host = np.asarray(original.vertex_points)
    bounds_host = np.asarray(model.parameter_bounds)
    signs_host = np.asarray(model.orientation)

    def remember(entity: BRepEntityId, kind: str, index: int, /) -> None:
        mapped[entity] = index
        sources.append(entity)
        targets.append((kind, index, entity.occurrence_path))

    def vertex(entity: BRepEntityId, occurrence: BRepOccurrence, /) -> int:
        if entity in mapped:
            return mapped[entity]
        expression = BRepPlacedVertex(
            points_host[entity.index, :],
            occurrence.rotation,
            occurrence.translation,
            _entity_text(entity),
            source_root=original.vertex_roots[entity.index],
        )
        root = BRepVertexRoot(expression)
        point, _, certified = root.evaluate()
        if not certified:
            raise ValueError(
                "A posed source vertex lost its original root qualification."
            )
        index = len(points)
        points.append(point)
        roots.append(root)
        remember(entity, "vertex", index)
        return index

    def edge(entity: BRepEntityId, occurrence: BRepOccurrence, /) -> int:
        if entity in mapped:
            return mapped[entity]
        source = entity.index
        endpoints = tuple(
            vertex(
                incidence[(entity, "vertex", original.edge_vertices[source][endpoint])],
                occurrence,
            )
            for endpoint in (0, 1)
        )
        curve_index = original.edge_curves[source]
        if curve_index >= 0:
            key = (occurrence.path, curve_index)
            if key not in curve_definitions:
                curve_definitions[key] = len(curves)
                curves.append(
                    PlacedCurve(
                        original.curves[curve_index],
                        occurrence.rotation,
                        occurrence.translation,
                    )
                )
            curve_index = curve_definitions[key]
        index = len(edge_curves)
        edge_curves.append(curve_index)
        edge_ranges.append((float(ranges_host[source, 0]), float(ranges_host[source, 1])))
        edge_vertices.append((endpoints[0], endpoints[1]))
        edge_roots.append(original.edge_endpoint_roots[source])
        remember(entity, "edge", index)
        return index

    def face(entity: BRepEntityId, occurrence: BRepOccurrence, /) -> int:
        if entity in mapped:
            return mapped[entity]
        source = entity.index
        loops: list[tuple[int, ...]] = []
        for source_loop in original.face_loops[source]:
            loop: list[int] = []
            for source_coedge in source_loop:
                source_edge = original.coedge_edges[source_coedge]
                world_edge = edge(incidence[(entity, "edge", source_edge)], occurrence)
                coedge = len(pcurves)
                pcurves.append(original.pcurves[source_coedge])
                coedge_edges.append(world_edge)
                coedge_senses.append(original.coedge_senses[source_coedge])
                coedge_roots.append(original.coedge_endpoint_roots[source_coedge])
                loop.append(coedge)
            loops.append(tuple(loop))
        index = len(patches)
        patches.append(
            PlacedSurface(
                model.patches[source], occurrence.rotation, occurrence.translation
            )
        )
        bounds.append(bounds_host[source, :, :])
        coorientation = (
            -1.0 if _exact_determinant(np.asarray(occurrence.rotation)) < 0 else 1.0
        )
        orientation.append(coorientation * float(signs_host[source]))
        tags.append(model.physical_tags[source])
        face_loops.append(tuple(loops))
        remember(entity, "face", index)
        return index

    for occurrence in original.occurrences:
        source_solid = BRepEntityId(
            model.source_revision, "solid", occurrence.solid, occurrence.path
        )
        shells: list[int] = []
        for source_shell in original.solid_shells[occurrence.solid]:
            faces = tuple(
                face(incidence[(source_solid, "face", source_face)], occurrence)
                for source_face in original.shell_faces[source_shell]
            )
            shell = len(shell_faces)
            shell_faces.append(faces)
            shell_orientations.append(original.shell_orientations[source_shell])
            shells.append(shell)
        solid = len(solid_shells)
        solid_shells.append(tuple(shells))
        occurrences.append(BRepOccurrence(occurrence.path, solid))
        remember(source_solid, "solid", solid)
    # Free source strata are not assembly instances and remain in definition
    # coordinates; a literal identity pose retains their operation tree.
    identity = BRepOccurrence(("definition",), 0)
    for record in records:
        entity = record.member
        if entity.occurrence_path or entity in mapped:
            continue
        if entity.kind == "face":
            face(entity, identity)
        elif entity.kind == "edge":
            edge(entity, identity)
        elif entity.kind == "vertex":
            vertex(entity, identity)
    geometry = BRepGeometry(
        vertex_points=np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        curves=tuple(curves),
        edge_curves=tuple(edge_curves),
        edge_ranges=np.asarray(edge_ranges, dtype=np.float64).reshape((-1, 2)),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(pcurves),
        coedge_edges=tuple(coedge_edges),
        coedge_senses=tuple(coedge_senses),
        face_loops=tuple(face_loops),
        shell_faces=tuple(shell_faces),
        shell_orientations=tuple(shell_orientations),
        solid_shells=tuple(solid_shells),
        occurrences=tuple(occurrences),
        assembly_containers=original.assembly_containers,
        vertex_roots=tuple(roots),
        edge_endpoint_roots=tuple(edge_roots),
        coedge_endpoint_roots=tuple(coedge_roots),
    )
    certificate = canonical_fingerprint(
        {
            "kind": "exact-cad-world-occurrence-materialization",
            "source": model.model_id,
            "world_geometry": geometry.geometry_id,
        }
    )
    world = assemble_brep_model(
        geometry,
        tuple(patches),
        np.asarray(bounds, dtype=np.float64).reshape((-1, 2, 2)),
        np.asarray(orientation, dtype=np.float64),
        tuple(tags),
        coordinate_contract=model.coordinate_contract,
        source_id=f"native-world-occurrences:{certificate}",
        source_format="native",
        source_digest=certificate,
        import_policy_id=model.import_policy_id,
        tessellation=tessellation,
        curve_surface_tolerance=model.report.curve_surface_tolerance,
    )
    target_entities = tuple(
        BRepEntityId(world.source_revision, kind, index, path)
        for kind, index, path in targets
    )
    source_revision = qualified_placement_revision(model)
    target_revision = qualified_placement_revision(world)
    target_inventory = {
        occurrence.entity_id: occurrence.occurrence_id
        for occurrence in target_revision.occurrences
    }
    pairs = tuple(
        OccurrenceCorrespondence(
            _entity_occurrence(source),
            target_inventory[_entity_text(target)],
            certificate,
        )
        for source, target in zip(sources, target_entities, strict=True)
    )
    # Namespace occurrences express authored assembly identity separately from
    # entity topology, and map by their explicit original path, not position.
    source_namespaces = {
        occurrence.entity_id: occurrence
        for occurrence in source_revision.occurrences
        if occurrence.kind == "assembly"
    }
    target_namespaces = {
        occurrence.entity_id: occurrence
        for occurrence in target_revision.occurrences
        if occurrence.kind == "assembly"
    }
    for namespace, source in source_namespaces.items():
        paths = sorted(
            {entity.occurrence_path for entity in sources if entity.occurrence_path}
            | {container.path for container in original.assembly_containers}
            | {occurrence.path for occurrence in original.occurrences}
        )
        path = next(
            path
            for path in paths
            if canonical_fingerprint(
                {
                    "kind": "cad-instance-namespace",
                    "source_revision": model.source_revision,
                    "path": path,
                }
            )
            == namespace
        )
        target_namespace = canonical_fingerprint(
            {
                "kind": "cad-instance-namespace",
                "source_revision": world.source_revision,
                "path": path,
            }
        )
        pairs += (
            OccurrenceCorrespondence(
                source.occurrence_id,
                target_namespaces[target_namespace].occurrence_id,
                certificate,
            ),
        )
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(
            sorted(
                pairs,
                key=lambda pair: (pair.source_occurrence_id, pair.target_occurrence_id),
            )
        ),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-exact-cad-occurrence-materialization"
        ),
    )
    graph = AssociationGraph(source_revision, target_revision, transaction)
    return BRepPlacementResult(world, graph, tuple(sources), target_entities, certificate)
