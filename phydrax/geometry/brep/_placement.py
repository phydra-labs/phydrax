#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact occurrence materialization and source-entity lineage."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...geometry._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ...typing import checked
from ._assembly import BRepAssemblyGraph
from ._constructors import BRepTessellationPolicy, _Builder, assemble_brep_model
from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint, RootEndpoint
from ._model import BRepEntityId, BRepModel
from ._patches import AbstractCurve
from ._placed import PlacedCurve, PlacedSurface
from ._root_bindings import BRepPlacedVertex, BRepVertexRoot
from ._partition import cad_revision_from_brep_model


type _EntityIndexMap = dict[str, dict[int, int]]


def entity_text(entity: BRepEntityId, /) -> str:
    from ._projection_contracts import brep_entity_id

    dimensions = {"vertex": 0, "edge": 1, "face": 2, "solid": 3}
    return brep_entity_id(
        entity.source_id,
        entity.source_revision,
        dimensions[entity.kind],
        entity.index,
        occurrence_path=entity.occurrence_path,
    )


def entity_occurrence(entity: BRepEntityId, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "cad-qualified-entity-occurrence",
            "entity": entity_text(entity),
        }
    )


@dataclass(frozen=True, slots=True)
class BRepPlacementResult:
    """Flattened physical instances with exhaustive source and occurrence lineage."""

    model: BRepModel
    source_entities: tuple[BRepEntityId, ...]
    target_entities: tuple[BRepEntityId, ...]
    association_graph: AssociationGraph
    certificate_id: str
    deleted_source_occurrences: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, BRepModel):
            raise TypeError("BRepPlacementResult.model must be a BRepModel.")
        if len(self.source_entities) != len(self.target_entities):
            raise ValueError("Placement source and target entity maps must align.")
        if any(
            not isinstance(value, BRepEntityId)
            for value in (*self.source_entities, *self.target_entities)
        ):
            raise TypeError("Placement entity maps must contain BRepEntityId values.")
        if not isinstance(self.association_graph, AssociationGraph):
            raise TypeError(
                "BRepPlacementResult.association_graph must be an AssociationGraph."
            )
        if not isinstance(self.certificate_id, str) or not self.certificate_id:
            raise ValueError("BRepPlacementResult.certificate_id must be nonempty.")
        object.__setattr__(
            self, "deleted_source_occurrences", tuple(self.deleted_source_occurrences)
        )


@checked
def materialize_brep_occurrences(
    model: BRepModel,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
) -> BRepPlacementResult:
    """Expand exact assembly leaves as separate affine-placed native B-Reps.

    No coordinate-based welding is performed: every leaf owns independent
    entity rows, even when two authored instances occupy coincident positions.
    Source assembly occurrence paths remain on the source side of the returned
    association graph.
    """
    geometry = model.geometry
    if not model.topology.num_solids:
        raise ValueError(
            "Occurrence materialization requires at least one source solid occurrence."
        )
    from ._carrier_pieces import carrier_payload
    from ...linalg._exact import determinant_exact

    source_occurrences = tuple(
        sorted(geometry.iter_occurrences(), key=lambda item: item.path)
    )
    if not source_occurrences:
        raise ValueError(
            "Occurrence materialization requires at least one source solid occurrence."
        )
    builder = _Builder()
    target_shell_faces: list[tuple[int, ...]] = []
    target_shell_orientations: list[tuple[int, ...]] = []
    target_solid_shells: list[tuple[int, ...]] = []
    occurrence_maps: list[tuple[tuple[str, ...], tuple[str, ...], _EntityIndexMap]] = []
    source_face_set = set(range(model.topology.num_faces))
    source_edge_set = set(range(model.topology.num_edges))
    source_vertex_set = set(range(model.topology.num_vertices))
    claimed_faces: set[int] = set()
    claimed_edges: set[int] = set()
    claimed_vertices: set[int] = set()
    source_tags = model.physical_tags
    source_bounds = np.asarray(model.parameter_bounds, dtype=np.float64)
    source_orientation = np.asarray(model.orientation, dtype=np.float64)

    for occurrence_index, occurrence in enumerate(source_occurrences):
        placement = occurrence.placement
        matrix = np.asarray(
            [[float(value) for value in row] for row in placement.matrix],
            dtype=np.float64,
        )
        if any(
            Fraction.from_float(float(matrix[row, column]))
            != placement.matrix[row][column]
            for row in range(4)
            for column in range(4)
        ):
            raise ValueError(
                "Exact occurrence materialization requires a binary64-realizable affine map."
            )
        linear, translation = matrix[:3, :3], matrix[:3, 3]
        orientation_factor = (
            1
            if determinant_exact(
                tuple(
                    tuple(Fraction.from_float(float(value)) for value in row)
                    for row in linear
                )
            )
            > 0
            else -1
        )
        solid = occurrence.solid_index
        faces = tuple(model.topology.solid_faces[solid])
        if not faces:
            raise ValueError(
                "Occurrence materialization cannot lower a solid without source faces."
            )
        claimed_faces.update(faces)
        coedge_ids = tuple(
            coedge
            for face in faces
            for loop in geometry.face_loops[face]
            for coedge in loop
        )
        edges = tuple(sorted({geometry.coedge_edges[coedge] for coedge in coedge_ids}))
        vertices = tuple(
            sorted({vertex for edge in edges for vertex in geometry.edge_vertices[edge]})
        )
        claimed_edges.update(edges)
        claimed_vertices.update(vertices)
        vertex_map: dict[int, int] = {}
        edge_map: dict[int, int] = {}
        face_map: dict[int, int] = {}
        curve_map: dict[int, PlacedCurve] = {}
        patch_map: dict[int, PlacedSurface] = {}
        patch_payload_map: dict[str, PlacedSurface] = {}
        curve_payload_map: dict[str, AbstractCurve] = {}

        for face in faces:
            placed = PlacedSurface(
                model.patches[face],
                linear,
                translation,
                affine=placement,
            )
            patch_map[face] = placed
            patch_payload_map[
                canonical_fingerprint(carrier_payload(model.patches[face]))
            ] = placed

        for vertex in vertices:
            exact_point = geometry.vertex_exact_points[vertex]
            placed_exact_point: tuple[Fraction, Fraction, Fraction] | None = None
            if exact_point is not None:
                placed_exact = placement.apply(exact_point)
                first_exact, second_exact, third_exact = placed_exact
                if not (
                    isinstance(first_exact, Fraction)
                    and isinstance(second_exact, Fraction)
                    and isinstance(third_exact, Fraction)
                ):
                    raise ValueError(
                        "A rational source vertex must remain rational under exact placement."
                    )
                placed_exact_point = (first_exact, second_exact, third_exact)
                point = np.asarray(
                    (float(first_exact), float(second_exact), float(third_exact)),
                    dtype=np.float64,
                )
            else:
                point = (
                    np.asarray(geometry.vertex_points[vertex], dtype=np.float64)
                    @ linear.T
                    + translation
                )
            source_root = geometry.vertex_roots[vertex]
            placed_root = None
            if source_root is not None:
                placed_record = BRepPlacedVertex(
                    geometry.vertex_points[vertex],
                    linear,
                    translation,
                    entity_text(
                        BRepEntityId(
                            model.source_id,
                            model.source_revision,
                            "vertex",
                            vertex,
                            occurrence.path,
                        )
                    ),
                    source_root=source_root,
                    affine=placement,
                )
                placed_root = BRepVertexRoot(placed_record)
            vertex_map[vertex] = builder.vertex(
                point,
                exact_point=placed_exact_point,
                root=placed_root,
            )

        def mapped_period_root(
            root: RootEndpoint | None,
            /,
            *,
            edge_curve: AbstractCurve | None,
            pcurve: AbstractCurve | None = None,
            curve_payload_map: dict[str, AbstractCurve] = curve_payload_map,
            patch_payload_map: dict[str, PlacedSurface] = patch_payload_map,
        ) -> RootEndpoint | None:
            if isinstance(root, ConicParameterEndpoint):
                source_payload = canonical_fingerprint(carrier_payload(root.carrier))
                if pcurve is not None and source_payload == canonical_fingerprint(
                    carrier_payload(pcurve)
                ):
                    target_curve = pcurve
                elif edge_curve is not None and source_payload in curve_payload_map:
                    target_curve = curve_payload_map[source_payload]
                else:
                    raise ValueError(
                        "A conic endpoint root has no materialized source carrier."
                    )
                return ConicParameterEndpoint(
                    root.root,
                    target_curve,
                    root.parameter_exact,
                    placement.apply(root.point_exact),
                )
            if not isinstance(root, NativePeriodEndpoint):
                return root
            source_payload = canonical_fingerprint(carrier_payload(root.curve))
            if pcurve is not None and source_payload == canonical_fingerprint(
                carrier_payload(pcurve)
            ):
                target_curve = pcurve
            elif edge_curve is not None and source_payload in curve_payload_map:
                target_curve = curve_payload_map[source_payload]
            else:
                raise ValueError(
                    "A native-period root has no materialized source carrier."
                )
            target_patch = None
            if root.patch is not None:
                patch_key = canonical_fingerprint(carrier_payload(root.patch))
                target_patch = patch_payload_map.get(patch_key)
                if target_patch is None:
                    raise ValueError(
                        "A native-period root has no materialized source surface."
                    )
            return NativePeriodEndpoint(
                target_curve,
                target_patch,
                root.axis,
                rational=root.rational,
                turns=root.turns,
            )

        for edge in edges:
            source_curve_index = geometry.edge_curves[edge]
            first, last = (
                float(value) for value in np.asarray(geometry.edge_ranges[edge])
            )
            if source_curve_index < 0:
                curve = None
            else:
                curve = curve_map.get(source_curve_index)
                if curve is None:
                    source_curve = geometry.curves[source_curve_index]
                    curve = PlacedCurve(
                        source_curve,
                        linear,
                        translation,
                        affine=placement,
                    )
                    curve_map[source_curve_index] = curve
                    curve_payload_map[
                        canonical_fingerprint(carrier_payload(source_curve))
                    ] = curve
            roots = geometry.edge_endpoint_roots[edge]
            edge_pcurve = (
                next(
                    (
                        geometry.pcurves[coedge]
                        for coedge in coedge_ids
                        if geometry.coedge_edges[coedge] == edge
                    ),
                    None,
                )
                if curve is None
                else None
            )
            mapped_roots = (
                mapped_period_root(
                    roots[0], edge_curve=curve, pcurve=edge_pcurve
                ),
                mapped_period_root(
                    roots[1], edge_curve=curve, pcurve=edge_pcurve
                ),
            )
            endpoints = tuple(vertex_map[index] for index in geometry.edge_vertices[edge])
            if curve is None:
                new_edge = builder.degenerate(
                    endpoints[0], first, last, endpoint_roots=mapped_roots
                )
            else:
                new_edge = builder.edge(
                    curve,
                    first,
                    last,
                    endpoints[0],
                    endpoints[1],
                    endpoint_roots=mapped_roots,
                )
            edge_map[edge] = new_edge

        for face in faces:
            loops: list[list[int]] = []
            for loop in geometry.face_loops[face]:
                new_loop = []
                for coedge in loop:
                    source_edge = geometry.coedge_edges[coedge]
                    pcurve = geometry.pcurves[coedge]
                    coedge_roots = geometry.coedge_endpoint_roots[coedge]
                    edge_curve_index = geometry.edge_curves[source_edge]
                    coedge_edge_curve = (
                        None
                        if edge_curve_index < 0
                        else curve_map[edge_curve_index]
                    )
                    endpoint_roots = (
                        mapped_period_root(
                            coedge_roots[0],
                            edge_curve=coedge_edge_curve,
                            pcurve=pcurve,
                        ),
                        mapped_period_root(
                            coedge_roots[1],
                            edge_curve=coedge_edge_curve,
                            pcurve=pcurve,
                        ),
                    )
                    new_coedge = builder.coedge(
                        edge_map[source_edge],
                        geometry.coedge_senses[coedge],
                        pcurve,
                        endpoint_roots=endpoint_roots,
                    )
                    new_loop.append(new_coedge)
                loops.append(new_loop)
            new_face = builder.face(
                patch_map[face],
                source_bounds[face],
                int(source_orientation[face]) * orientation_factor,
                source_tags[face],
                loops,
            )
            face_map[face] = new_face

        solid_shells = []
        for shell in geometry.solid_shells[solid]:
            target_shell = len(target_shell_faces)
            shell_faces = geometry.shell_faces[shell]
            target_shell_faces.append(tuple(face_map[face] for face in shell_faces))
            target_shell_orientations.append(geometry.shell_orientations[shell])
            solid_shells.append(target_shell)
        target_solid_shells.append(tuple(solid_shells))
        target_path = (f"solid-{occurrence_index}",)
        occurrence_maps.append(
            (
                occurrence.path,
                target_path,
                {
                    "solid": {solid: occurrence_index},
                    "face": face_map,
                    "edge": edge_map,
                    "vertex": vertex_map,
                },
            )
        )

    if (
        claimed_faces != source_face_set
        or claimed_edges != source_edge_set
        or claimed_vertices != source_vertex_set
    ):
        raise ValueError(
            "Occurrence materialization requires every source entity to belong to a solid leaf."
        )
    certificate_id = canonical_fingerprint(
        {
            "kind": "exact-brep-occurrence-materialization",
            "source_revision": model.source_revision,
            "occurrences": tuple(
                (
                    item.path,
                    item.solid_index,
                    item.placement.identity_payload(),
                )
                for item in source_occurrences
            ),
        }
    )
    result_geometry, patches, bounds, orientations, tags, trims = builder.geometry(
        solid=True,
        shell_faces=tuple(target_shell_faces),
        shell_orientations=tuple(target_shell_orientations),
        solid_shells=tuple(target_solid_shells),
        assembly_graph=BRepAssemblyGraph.single_solids(len(source_occurrences)),
    )
    result_model = assemble_brep_model(
        result_geometry,
        patches,
        bounds,
        orientations,
        tags,
        coordinate_contract=model.coordinate_contract,
        source_id=f"native-placement:{certificate_id}",
        tessellation=tessellation,
        trim_domains=trims,
    )
    source_revision = cad_revision_from_brep_model(model)
    target_revision = cad_revision_from_brep_model(result_model)
    source_entities = tuple(
        sorted(
            {
                record.member
                for record in geometry.qualified_entity_incidence(
                    model.source_id, model.source_revision
                )
            }
        )
    )
    target_by_source: dict[BRepEntityId, BRepEntityId] = {}
    for source_path, target_path, maps in occurrence_maps:
        for kind, local_map in maps.items():
            for source_index, target_index in local_map.items():
                source_entity = BRepEntityId(
                    model.source_id,
                    model.source_revision,
                    kind,
                    source_index,
                    source_path,
                )
                target_entity = BRepEntityId(
                    result_model.source_id,
                    result_model.source_revision,
                    kind,
                    target_index,
                    target_path,
                )
                if source_entity in target_by_source:
                    raise ValueError(
                        "A source entity occurrence was materialized more than once."
                    )
                target_by_source[source_entity] = target_entity
    if set(target_by_source) != set(source_entities):
        raise ValueError(
            "Occurrence materialization did not map every source entity occurrence."
        )
    target_entities = tuple(target_by_source[entity] for entity in source_entities)
    pairs = {
        (
            entity_occurrence(source),
            entity_occurrence(target_by_source[source]),
        )
        for source in source_entities
    }
    source_namespace_paths = {
        container.path for container in geometry.iter_assembly_containers()
    } | {item.path for item in geometry.iter_occurrences()}
    for source_path, target_path, _maps in occurrence_maps:
        target_namespace_id = canonical_fingerprint(
            {
                "kind": "cad-instance-namespace",
                "source_id": result_model.source_id,
                "source_revision": result_model.source_revision,
                "path": target_path,
            }
        )
        for length in range(1, len(source_path) + 1):
            namespace = source_path[:length]
            if namespace in source_namespace_paths:
                source_namespace_id = canonical_fingerprint(
                    {
                        "kind": "cad-instance-namespace",
                        "source_id": model.source_id,
                        "source_revision": model.source_revision,
                        "path": namespace,
                    }
                )
                pairs.add((source_namespace_id, target_namespace_id))
    correspondences = tuple(
        OccurrenceCorrespondence(source_id, target_id, certificate_id)
        for source_id, target_id in sorted(pairs)
    )
    transaction = OccurrenceCorrespondenceTransaction(
        certificate_id,
        source_revision.revision_id,
        target_revision.revision_id,
        correspondences,
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True,
            True,
            certificate_id,
            "native-exact-affine-occurrence-materialization",
        ),
    )
    graph = AssociationGraph(source_revision, target_revision, transaction)
    mapped_source = {edge.source_occurrence_id for edge in correspondences}
    deleted = tuple(
        sorted(
            item.occurrence_id
            for item in source_revision.occurrences
            if item.occurrence_id not in mapped_source
        )
    )
    return BRepPlacementResult(
        result_model,
        source_entities,
        target_entities,
        graph,
        certificate_id,
        deleted,
    )


__all__ = [
    "BRepPlacementResult",
    "entity_occurrence",
    "entity_text",
    "materialize_brep_occurrences",
]
