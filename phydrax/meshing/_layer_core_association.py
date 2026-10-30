#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source layer/core provenance tables and their topology-preserving transfer."""

from __future__ import annotations

from typing import final

import numpy as np

import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._meshing_domain as meshing_domain_owner
import phydrax.geometry._source_identity as source_identity_owner
import phydrax.meshing._association as association_owner
import phydrax.meshing._boundary_layer as boundary_layer_owner
import phydrax.meshing._canonical as canonical_owner
import phydrax.meshing._layer_core_topology as layer_topology
import phydrax.meshing._polyhedral_generation as polyhedral_generation_owner
import phydrax.meshing._volume_generation as volume_generation_owner
import phydrax.meshing.providers._native_sources as sources_owner

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..typing import checked


def _row_entity_vertex_keys(
    mesh: CellMesh, dimension: int, /
) -> tuple[tuple[int, ...], ...]:
    """Return each entity's canonical key in mesh vertex-row coordinates."""
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    row_by_id = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    return tuple(
        tuple(sorted(row_by_id[identifier] for identifier in key))
        for key in canonical_owner._entity_vertex_keys(mesh, dimension)
    )


@final
class LayerSourceTables(StrictModule, NonTrainableState):
    domain: domain_coverage_owner.PiecewiseLinearDomain
    edge_vertices: np.ndarray
    edge_indices: np.ndarray
    triangle_vertices: np.ndarray
    triangle_facets: np.ndarray
    facet_regions: np.ndarray
    facet_indices: np.ndarray
    region_indices: np.ndarray


@checked
def layer_source_tables(
    layers: boundary_layer_owner.BoundaryLayerMesh,
    layer_regions: np.ndarray,
    region_ids: tuple[str, ...],
    /,
) -> LayerSourceTables:
    """Describe the exact piecewise-linear boundary and interfaces of accepted layers."""
    mesh = layers.mesh
    edges = np.asarray(_row_entity_vertex_keys(mesh, 1), dtype=np.int64)
    face_rows = _row_entity_vertex_keys(mesh, 2)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    cell_ids = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    if layer_regions.shape != cell_ids.shape or not np.issubdtype(
        layer_regions.dtype, np.integer
    ):
        raise ValueError("Layer region IDs must identify every accepted layer cell.")
    materials = np.unique(np.asarray(layer_regions, dtype=np.int64))
    if np.any(materials < 0) or np.any(materials >= len(region_ids)):
        raise ValueError("Layer regions must belong to the explicit composite namespace.")
    local_regions = np.searchsorted(materials, layer_regions).astype(np.int64)
    face_incidents = layer_topology.faces(mesh, local_regions)
    cap_keys: dict[tuple[int, ...], tuple[int, ...]] = {}
    if layers.cap is not None:
        cap_vertices = np.asarray(layers.cap_vertices, dtype=np.int64)
        for block in layers.cap.blocks:
            for row in np.asarray(block.vertices, dtype=np.int64):
                oriented = tuple(int(cap_vertices[vertex]) for vertex in row)
                key = tuple(sorted(oriented))
                if key in cap_keys:
                    raise ValueError("The accepted layer cap contains duplicate facets.")
                cap_keys[key] = oriented
    facet_indices = np.array(face_ids, dtype=np.int64, copy=True)
    if not set(cap_keys) <= set(face_rows):
        raise ValueError(
            "Every accepted cap facet must be present on the layer boundary."
        )
    triangles: list[tuple[int, int, int]] = []
    triangle_facets: list[int] = []
    incidence = np.empty((len(face_rows), 2), dtype=np.int64)
    boundary_triangles: list[tuple[int, int, int]] = []
    boundary_incidence: list[tuple[int, int]] = []
    for row, key in enumerate(face_rows):
        incidents = face_incidents[key]
        first = incidents[0]
        other = incidents[1].region if len(incidents) == 2 else -1
        pair = (first.region, other)
        incidence[row] = pair
        oriented = cap_keys[key] if key in cap_keys else first.vertices
        pieces = layer_topology.triangles(oriented)
        triangles.extend(pieces)
        triangle_facets.extend((row,) * len(pieces))
        if first.region != other:
            boundary_triangles.extend(pieces)
            boundary_incidence.extend((pair,) * len(pieces))
    domain = domain_coverage_owner.PiecewiseLinearDomain(
        np.asarray(mesh.coordinates, dtype=np.float64),
        np.asarray(boundary_triangles, dtype=np.int64),
        np.asarray(boundary_incidence, dtype=np.int64),
        tuple(region_ids[int(index)] for index in materials),
        source_id=layers.result_id,
    )
    return LayerSourceTables(
        domain,
        edges,
        np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64),
        np.asarray(triangles, dtype=np.int64),
        np.asarray(triangle_facets, dtype=np.int64),
        incidence,
        facet_indices,
        materials,
    )


@checked
def layer_source_associations(
    layers: boundary_layer_owner.BoundaryLayerMesh,
    target: CellMesh,
    layer_regions: np.ndarray,
    region_ids: tuple[str, ...],
    /,
) -> tuple[association_owner.GeometryAssociation, ...]:
    """Transfer exact accepted layer vertex/edge/facet identities by incidence."""
    tables = layer_source_tables(layers, layer_regions, region_ids)
    source_vertex_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    records: list[association_owner.GeometryAssociation] = []
    for dimension in (0, 1, 2):
        source_ids = np.asarray(
            layers.mesh.entity_set(dimension).entity_ids, dtype=np.int64
        )
        if dimension == 0:
            target_ids = source_vertex_ids
            source_indices = source_vertex_ids
            role = association_owner.GeometrySourceEntityRole.VERTEX
            sides: tuple[tuple[str | None, str | None], ...] | None = None
        else:
            target_keys = {
                key: int(identifier)
                for identifier, key in zip(
                    np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64),
                    canonical_owner._entity_vertex_keys(target, dimension),
                    strict=True,
                )
            }
            source_keys = canonical_owner._entity_vertex_keys(layers.mesh, dimension)
            target_ids = np.asarray(
                [target_keys[key] for key in source_keys], dtype=np.int64
            )
            if dimension == 1:
                source_indices = source_ids
                role = association_owner.GeometrySourceEntityRole.EDGE
                sides = None
            else:
                source_indices = tables.facet_indices
                role = association_owner.GeometrySourceEntityRole.FACET
                materials = tables.region_indices
                sides = tuple(
                    (
                        None
                        if negative < 0
                        else region_ids[int(materials[int(negative)])],
                        None
                        if positive < 0
                        else region_ids[int(materials[int(positive)])],
                    )
                    for negative, positive in tables.facet_regions
                )
        if target_ids.size == 0:
            continue
        records.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                layers.result_id,
                layers.result_id,
                target.entity_set(dimension).entity_set_id,
                target_ids,
                tuple(
                    source_identity_owner.source_entity_id(
                        layers.result_id,
                        layers.result_id,
                        role.value,
                        int(index),
                    )
                    for index in source_indices
                ),
                np.zeros(target_ids.shape, dtype=np.float64),
                exact=True,
                source_roles=(role,) * target_ids.size,
                orientations=np.zeros(target_ids.shape, dtype=np.int8),
                source_occurrence_paths=((),) * target_ids.size,
                source_region_sides=sides,
            )
        )
    return tuple(records)


def _edge_witness_parameter(
    core: volume_generation_owner.VolumeConstruction,
    vertex: int,
    source_segment_row: int,
    edge: np.ndarray,
    /,
) -> float:
    """Resolve a constrained edge endpoint through exact source ancestry."""
    input_vertex = int(np.asarray(core.input_vertices, dtype=np.int64)[vertex])
    if input_vertex == int(edge[0]):
        return 0.0
    if input_vertex == int(edge[1]):
        return 1.0
    strata = np.asarray(core.witness_strata, dtype=np.int8)
    entities = np.asarray(core.witness_entities, dtype=np.int64)
    if (
        input_vertex < 0
        and strata[vertex] == 1
        and entities[vertex] == source_segment_row
    ):
        return float(np.asarray(core.witness_parameters)[vertex, 0])
    raise ValueError(
        "A constrained segment endpoint lacks its exact source-edge witness."
    )


@checked
def layer_wall_associations(
    source: sources_owner.NativeLayerCoreSource,
    target: CellMesh,
    /,
) -> tuple[association_owner.GeometryAssociation, ...]:
    """Carry unchanged source-wall vertex associations through exact row lineage."""
    parent = source.layers.wall_association
    if parent is None:
        return ()
    wall = source.layers.source_wall
    wall_vertices = np.asarray(source.layers.wall_vertices, dtype=np.int64)
    wall_vertex_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    layer_vertex_ids = np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    if wall_vertices.shape != wall_vertex_ids.shape:
        raise ValueError("Accepted layer wall-vertex ancestry has incompatible extents.")
    target_rows = {
        int(identifier): row for row, identifier in enumerate(target_vertex_ids)
    }
    parent_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(parent.target_global_ids, dtype=np.int64)
        )
    }
    target_ids: list[int] = []
    source_rows: list[int] = []
    wall_points = np.asarray(wall.coordinates, dtype=np.float64)
    layer_points = np.asarray(source.layers.mesh.coordinates, dtype=np.float64)
    for wall_row, layer_row in enumerate(wall_vertices):
        if layer_row < 0:
            continue
        source_vertex_id = int(wall_vertex_ids[wall_row])
        parent_row = parent_rows.get(source_vertex_id)
        if parent_row is None:
            continue
        if not np.array_equal(
            wall_points[wall_row].view(np.uint64),
            layer_points[int(layer_row)].view(np.uint64),
        ):
            raise ValueError(
                "Accepted layer wall vertices moved their source coordinates."
            )
        target_id = int(layer_vertex_ids[int(layer_row)])
        if target_id not in target_rows:
            raise ValueError("An accepted wall vertex is absent from the combined mesh.")
        target_ids.append(target_id)
        source_rows.append(parent_row)
    if not target_ids:
        return ()
    order = np.argsort(np.asarray(target_ids, dtype=np.int64), kind="stable")
    ordered_targets = np.asarray(target_ids, dtype=np.int64)[order]
    ordered_sources = np.asarray(source_rows, dtype=np.int64)[order]
    return (
        association_owner.inherit_association_rows(
            parent,
            target.entity_set(0),
            ordered_targets,
            ordered_sources,
            residuals=np.asarray(parent.residuals, dtype=np.float64)[ordered_sources],
        ),
    )


def _core_source_identity(
    source: sources_owner.NativeLayerCoreSource,
    dimension: int,
    identifier: str,
    occurrence_path: tuple[str, ...],
    /,
) -> tuple[str, tuple[str, ...]]:
    """Bind a PLC stratum key to its authored source identity and occurrence."""
    complex_ = source.complex
    path = tuple(occurrence_path)

    def bind(
        identity: str, entity_path: tuple[str, ...], /
    ) -> tuple[str, tuple[str, ...]]:
        domain = source.source_domain
        if isinstance(domain, meshing_domain_owner.MeshingDomain):
            domain.resolve_entity_id(dimension, identity, occurrence_path=entity_path)
        return identity, entity_path

    if dimension == 0:
        vertex_keys = {
            (name, tuple(row_path))
            for name, row_path in zip(
                complex_.vertex_ids, complex_.vertex_occurrence_paths, strict=True
            )
        }
        if (identifier, path) not in vertex_keys:
            raise ValueError(
                "A native core vertex lost its authored PLC source identity."
            )
        identity = source_identity_owner.source_entity_id(
            source.source_id,
            source.source_revision,
            "vertex",
            identifier,
            occurrence_path=path,
        )
        return bind(identity, path)
    if dimension == 1:
        edge_keys = {
            (name, tuple(row_path))
            for name, row_path in zip(
                complex_.plc_edge_ids,
                complex_.plc_edge_occurrence_paths,
                strict=True,
            )
        }
        if (identifier, path) not in edge_keys:
            raise ValueError("A native core edge lost its authored PLC source identity.")
        identity = source_identity_owner.source_entity_id(
            source.source_id,
            source.source_revision,
            "edge",
            identifier,
            occurrence_path=path,
        )
        return bind(identity, path)
    if dimension == 2:
        facet_rows = {
            (name, tuple(row_path)): row
            for row, (name, row_path) in enumerate(
                zip(
                    complex_.facet_ids,
                    complex_.facet_occurrence_paths,
                    strict=True,
                )
            )
        }
        row = facet_rows.get((identifier, path))
        if row is None:
            raise ValueError("A native core facet lost its authored PLC source identity.")
        external_id = source.core_facet_source_ids[row]
        external_path = source.core_facet_source_occurrence_paths[row]
        if external_id is not None:
            if external_path is None:
                raise ValueError(
                    "A source-bound PLC facet is missing its occurrence path."
                )
            return bind(external_id, tuple(external_path))
        if source.source_boundary_scope is not None:
            raise ValueError("A source-bound PLC facet has no stable source identity.")
        identity = source_identity_owner.source_entity_id(
            source.source_id,
            source.source_revision,
            "face",
            identifier,
            occurrence_path=path,
        )
        return bind(identity, path)
    if dimension == 3:
        region_rows = {
            (name, tuple(row_path)): row
            for row, (name, row_path) in enumerate(
                zip(
                    complex_.region_ids,
                    complex_.region_occurrence_paths,
                    strict=True,
                )
            )
        }
        row = region_rows.get((identifier, path))
        if row is None:
            raise ValueError(
                "A native core region lost its authored PLC source identity."
            )
        composite = int(np.asarray(source.core_region_map, dtype=np.int64)[row])
        region_id = int(source.region_entity_ids[composite])
        region_path = tuple(source.region_occurrence_paths[composite])
        identity = source_identity_owner.source_entity_id(
            source.source_id,
            source.source_revision,
            "region",
            region_id,
            occurrence_path=region_path,
        )
        return bind(identity, region_path)
    raise ValueError("A native core source witness has an invalid stratum dimension.")


def _polyhedral_entity_id_maps(
    core: polyhedral_generation_owner.PolyhedralConstruction,
    target: CellMesh,
    core_vertex_map: np.ndarray,
    core_cell_ids: np.ndarray,
    /,
) -> tuple[dict[int, int], ...]:
    """Rebind exact core vertex/edge/face/cell identities to mixed output rows."""
    core_mesh = core.mesh
    core_vertex_ids = np.asarray(core_mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    if (
        core_vertex_map.shape != core_vertex_ids.shape
        or np.any(core_vertex_map < 0)
        or np.any(core_vertex_map >= target_vertex_ids.size)
    ):
        raise ValueError(
            "Power-core vertex ancestry does not bind the mixed output mesh."
        )
    vertex_ids = {
        int(identifier): int(target_vertex_ids[int(row)])
        for identifier, row in zip(core_vertex_ids, core_vertex_map, strict=True)
    }
    maps: list[dict[int, int]] = [vertex_ids]
    for dimension in (1, 2):
        target_keys = canonical_owner._entity_vertex_keys(target, dimension)
        target_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        target_by_key = dict(zip(target_keys, target_ids.tolist(), strict=True))
        source_keys = canonical_owner._entity_vertex_keys(core_mesh, dimension)
        source_ids = np.asarray(
            core_mesh.entity_set(dimension).entity_ids, dtype=np.int64
        )
        mapped: dict[int, int] = {}
        for identifier, key in zip(source_ids.tolist(), source_keys, strict=True):
            target_key = tuple(sorted(vertex_ids[int(vertex)] for vertex in key))
            target_id = target_by_key.get(target_key)
            if target_id is None:
                raise ValueError(
                    "Power-core source association has no conforming mixed-mesh entity."
                )
            mapped[int(identifier)] = int(target_id)
        maps.append(mapped)
    source_cells = np.sort(np.asarray(core_mesh.entity_set(3).entity_ids, dtype=np.int64))
    if core_cell_ids.shape != source_cells.shape:
        raise ValueError("Power-core cell identities do not align with the mixed mesh.")
    maps.append(
        {
            int(source): int(target_id)
            for source, target_id in zip(source_cells, core_cell_ids, strict=True)
        }
    )
    return tuple(maps)


def _polyhedral_region_identity(
    source: sources_owner.NativeLayerCoreSource,
    region: int,
    /,
) -> str:
    complex_ = source.complex
    return _core_source_identity(
        source,
        3,
        complex_.region_ids[region],
        tuple(complex_.region_occurrence_paths[region]),
    )[0]


def _polyhedral_core_source_associations(
    source: sources_owner.NativeLayerCoreSource,
    core: polyhedral_generation_owner.PolyhedralConstruction,
    target: CellMesh,
    core_vertex_map: np.ndarray,
    core_cell_ids: np.ndarray,
    /,
) -> tuple[association_owner.GeometryAssociation, ...]:
    """Transfer exact restricted-power ancestry onto canonical mixed-mesh entities."""
    power_source = source.core_power_source
    if power_source is None:
        raise ValueError(
            "Polyhedral layer/core associations require the authored power source."
        )
    from .providers._native_polyhedral import _polyhedral_associations

    records = _polyhedral_associations(power_source, core)
    identifier_maps = _polyhedral_entity_id_maps(
        core, target, core_vertex_map, core_cell_ids
    )
    facet_rows = {
        (identifier, tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(
                source.complex.facet_ids,
                source.complex.facet_occurrence_paths,
                strict=True,
            )
        )
    }
    region_rows = {
        (identifier, tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(
                source.complex.region_ids,
                source.complex.region_occurrence_paths,
                strict=True,
            )
        )
    }
    rebound: list[association_owner.GeometryAssociation] = []
    role_dimensions = {
        association_owner.GeometrySourceEntityRole.VERTEX: 0,
        association_owner.GeometrySourceEntityRole.EDGE: 1,
        association_owner.GeometrySourceEntityRole.FACET: 2,
        association_owner.GeometrySourceEntityRole.REGION: 3,
    }
    for dimension, record in enumerate(records):
        role_values = record.source_roles
        source_entities: list[str] = []
        source_paths: list[tuple[str, ...]] = []
        region_sides: list[tuple[str | None, str | None]] = []
        for identifier, path, role in zip(
            record.source_entity_ids,
            record.source_occurrence_paths,
            role_values,
            strict=True,
        ):
            if role is None or role not in role_dimensions:
                raise ValueError("Power-core association lost its source-stratum role.")
            entity, occurrence = _core_source_identity(
                source,
                role_dimensions[role],
                identifier,
                tuple(path),
            )
            source_entities.append(entity)
            source_paths.append(occurrence)
            if role is association_owner.GeometrySourceEntityRole.FACET:
                facet_row = facet_rows.get((identifier, tuple(path)))
                if facet_row is None:
                    raise ValueError(
                        "Power-core facet association lost its PLC identity."
                    )
                sides = tuple(
                    None
                    if int(region) < 0
                    else _polyhedral_region_identity(source, int(region))
                    for region in np.asarray(source.complex.facet_regions)[facet_row]
                )
                region_sides.append((sides[0], sides[1]))
            elif role is association_owner.GeometrySourceEntityRole.REGION:
                region_row = region_rows.get((identifier, tuple(path)))
                if region_row is None:
                    raise ValueError(
                        "Power-core region association lost its PLC identity."
                    )
                region_id = _polyhedral_region_identity(source, region_row)
                region_sides.append((region_id, region_id))
            else:
                region_sides.append((None, None))
        target_ids = np.asarray(record.target_global_ids, dtype=np.int64)
        mapped_ids = np.asarray(
            [identifier_maps[dimension][int(value)] for value in target_ids],
            dtype=np.int64,
        )
        rebound.append(
            association_owner.GeometryAssociation(
                record.association_kind,
                source.source_id,
                source.source_revision,
                target.entity_set(dimension).entity_set_id,
                mapped_ids,
                tuple(source_entities),
                np.asarray(record.residuals, dtype=np.float64),
                resolved=np.asarray(record.resolved, dtype=np.bool_),
                ambiguous=np.asarray(record.ambiguous, dtype=np.bool_),
                exact=record.exact,
                source_dimensions=np.asarray(record.source_dimensions, dtype=np.int64),
                source_indices=np.asarray(record.source_indices, dtype=np.int64),
                source_roles=role_values,
                source_occurrence_paths=tuple(source_paths),
                source_region_sides=tuple(region_sides),
                source_parameter_intervals=record.source_parameter_intervals,
                source_parameter_points=record.source_parameter_points,
                parameters=np.asarray(record.parameters, dtype=np.float64),
                orientations=np.asarray(record.orientations, dtype=np.int8),
                parent_dimensions=np.asarray(record.parent_dimensions, dtype=np.int64),
                parent_ids=np.asarray(record.parent_ids, dtype=np.int64),
                parent_association_id=record.parent_association_id,
                provenance=record.provenance,
            )
        )
    return tuple(rebound)


@checked
def core_source_associations(
    source: sources_owner.NativeLayerCoreSource,
    core: (
        volume_generation_owner.VolumeConstruction
        | polyhedral_generation_owner.PolyhedralConstruction
    ),
    target: CellMesh,
    core_vertex_map: np.ndarray,
    combined_regions: np.ndarray,
    region_ids: tuple[str, ...],
    core_cell_ids: np.ndarray,
    /,
) -> tuple[association_owner.GeometryAssociation, ...]:
    """Compose exact PLC ancestry onto the mixed layer/core mesh."""
    if isinstance(core, polyhedral_generation_owner.PolyhedralConstruction):
        return _polyhedral_core_source_associations(
            source, core, target, core_vertex_map, core_cell_ids
        )
    core_cells = np.sort(np.asarray(core.mesh.entity_set(3).entity_ids, dtype=np.int64))
    if core_cell_ids.shape != core_cells.shape:
        raise ValueError("Native core cell IDs do not match the accepted core cells.")
    complex_ = source.complex
    records: list[association_owner.GeometryAssociation] = []
    vertex_sources = tuple(core.vertex_source_ids)
    vertex_dimensions_all = np.asarray(core.vertex_dimension, dtype=np.int64)
    vertex_source_paths_all = tuple(core.vertex_source_occurrence_paths)
    if vertex_dimensions_all.shape != (len(vertex_sources),) or len(
        vertex_source_paths_all
    ) != len(vertex_sources):
        raise ValueError("Native core vertex source ancestry arrays are incompatible.")
    explicit_segment_count = complex_.segments.shape[0]
    derived_edge_keys = {
        (identifier, tuple(path))
        for edge_row, (identifier, path) in enumerate(
            zip(complex_.plc_edge_ids, complex_.plc_edge_occurrence_paths, strict=True)
        )
        if edge_row >= explicit_segment_count
    }
    vertex_rows = np.asarray(
        [
            row
            for row, identifier in enumerate(vertex_sources)
            if identifier is not None
            and not (
                int(vertex_dimensions_all[row]) == 1
                and (str(identifier), tuple(vertex_source_paths_all[row]))
                in derived_edge_keys
            )
        ],
        dtype=np.int64,
    )
    if vertex_rows.size:
        vertex_targets = np.asarray(target.vertex_global_ids, dtype=np.int64)[
            core_vertex_map[vertex_rows]
        ]
        vertex_dimensions = vertex_dimensions_all[vertex_rows]
        if np.any(vertex_dimensions < 0) or np.any(
            vertex_dimensions >= len(association_owner.GeometrySourceEntityRole)
        ):
            raise ValueError("A PLC-associated vertex has no source-stratum role.")
        role_by_dimension = (
            association_owner.GeometrySourceEntityRole.VERTEX,
            association_owner.GeometrySourceEntityRole.EDGE,
            association_owner.GeometrySourceEntityRole.FACET,
            association_owner.GeometrySourceEntityRole.REGION,
        )
        identities = tuple(
            _core_source_identity(
                source,
                int(vertex_dimensions_all[int(row)]),
                str(vertex_sources[int(row)]),
                tuple(vertex_source_paths_all[int(row)]),
            )
            for row in vertex_rows
        )
        vertex_source_ids = tuple(identity for identity, _ in identities)
        vertex_paths = tuple(path for _, path in identities)
        vertex_residuals = np.asarray(core.witness_deviations, dtype=np.float64)[
            vertex_rows
        ]
        records.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                target.entity_set(0).entity_set_id,
                vertex_targets,
                vertex_source_ids,
                vertex_residuals,
                exact=bool(np.all(vertex_residuals == 0.0)),
                source_roles=tuple(
                    role_by_dimension[int(value)] for value in vertex_dimensions
                ),
                source_occurrence_paths=vertex_paths,
                orientations=np.zeros(vertex_targets.shape, dtype=np.int8),
            )
        )
    target_edge_ids = np.asarray(target.entity_set(1).entity_ids, dtype=np.int64)
    target_edge_keys = {
        key: int(identifier)
        for key, identifier in zip(
            canonical_owner._entity_vertex_keys(target, 1), target_edge_ids, strict=True
        )
    }
    segments = np.asarray(core.segments, dtype=np.int64)
    segment_active_rows = np.asarray(core.segment_source_edges, dtype=np.int64)
    segment_source_ids = tuple(core.segment_source_ids)
    segment_source_paths = tuple(core.segment_source_occurrence_paths)
    raw_plc_edges = np.asarray(core.plc_edges, dtype=np.int64)
    raw_to_active_rows = np.asarray(core.plc_edge_source_rows, dtype=np.int64)
    raw_plc_edge_ids = tuple(str(value) for value in core.plc_edge_ids)
    raw_plc_edge_paths = tuple(core.plc_edge_occurrence_paths)
    active_source_vertices = np.asarray(core.source_segment_vertices, dtype=np.int64)
    active_source_complex_rows = np.asarray(
        core.source_segment_complex_rows, dtype=np.int64
    )
    active_source_ids = tuple(core.segment_ids)
    active_source_paths = tuple(core.segment_occurrence_paths)
    canonical_source_edges = np.asarray(
        source.complex.plc_edge_vertices, dtype=np.int64
    ).reshape((-1, 2))
    if (
        segments.ndim != 2
        or segments.shape[1] != 2
        or segment_active_rows.shape != (segments.shape[0],)
        or len(segment_source_ids) != segments.shape[0]
        or len(segment_source_paths) != segments.shape[0]
        or raw_plc_edges.ndim != 2
        or raw_plc_edges.shape[1] != 2
        or raw_to_active_rows.shape != (raw_plc_edges.shape[0],)
        or len(raw_plc_edge_ids) != raw_plc_edges.shape[0]
        or len(raw_plc_edge_paths) != raw_plc_edges.shape[0]
        or active_source_vertices.shape != (active_source_complex_rows.size, 2)
        or len(active_source_ids) != active_source_complex_rows.size
        or len(active_source_paths) != active_source_complex_rows.size
        or np.any(
            (active_source_complex_rows < 0)
            | (active_source_complex_rows >= canonical_source_edges.shape[0])
        )
        or not np.array_equal(
            active_source_vertices,
            canonical_source_edges[active_source_complex_rows],
        )
    ):
        raise ValueError("Native constrained-edge ancestry arrays are incompatible.")
    raw_row_by_active = np.full((active_source_complex_rows.size,), -1, dtype=np.int64)
    for raw_row, active_row in enumerate(raw_to_active_rows.tolist()):
        if not 0 <= active_row < raw_row_by_active.size:
            raise ValueError(
                "A raw PLC carrier row lacks its active source-edge mapping."
            )
        if raw_row_by_active[active_row] >= 0:
            raise ValueError("A source edge has multiple raw PLC carrier rows.")
        raw_row_by_active[active_row] = raw_row
    if np.any(raw_row_by_active < 0):
        raise ValueError("An active source edge lacks its raw PLC carrier row.")
    edge_rows: dict[tuple[str, tuple[str, ...]], dict[int, int]] = {}
    for segment_row in range(segments.shape[0]):
        segment = (
            int(segments[segment_row, 0]),
            int(segments[segment_row, 1]),
        )
        active_row = int(segment_active_rows[segment_row])
        if not 0 <= active_row < active_source_complex_rows.size:
            raise ValueError(
                "A constrained segment names an absent active source-edge row."
            )
        raw_row = int(raw_row_by_active[active_row])
        complex_row = int(active_source_complex_rows[active_row])
        edge = active_source_vertices[active_row]
        source_id = segment_source_ids[segment_row]
        occurrence_path = tuple(segment_source_paths[segment_row])
        if (
            source_id != active_source_ids[active_row]
            or occurrence_path != active_source_paths[active_row]
            or raw_plc_edge_ids[raw_row] != source_id
            or raw_plc_edge_paths[raw_row] != occurrence_path
            or tuple(sorted(int(value) for value in raw_plc_edges[raw_row]))
            != tuple(sorted(int(value) for value in canonical_source_edges[complex_row]))
        ):
            raise ValueError(
                "Raw carrier, active segment and canonical PLC edge identities disagree."
            )
        source_id, occurrence_path = _core_source_identity(
            source, 1, str(source_id), occurrence_path
        )
        first_parameter = _edge_witness_parameter(core, int(segment[0]), active_row, edge)
        second_parameter = _edge_witness_parameter(
            core, int(segment[1]), active_row, edge
        )
        if first_parameter == second_parameter:
            raise ValueError(
                "A constrained source segment has zero source-parameter extent."
            )
        combined_rows = np.asarray(
            (int(core_vertex_map[segment[0]]), int(core_vertex_map[segment[1]])),
            dtype=np.int64,
        )
        vertices = np.asarray(target.vertex_global_ids, dtype=np.int64)[combined_rows]
        target_key = tuple(sorted(int(value) for value in vertices))
        target_id = target_edge_keys[target_key]
        canonical_parameter = (
            first_parameter if vertices[0] < vertices[1] else second_parameter
        )
        other_parameter = (
            second_parameter if vertices[0] < vertices[1] else first_parameter
        )
        orientation = 1 if other_parameter > canonical_parameter else -1
        key = (source_id, tuple(occurrence_path))
        targets = edge_rows.setdefault(key, {})
        previous = targets.setdefault(target_id, orientation)
        if previous != orientation:
            raise ValueError("One constrained mesh edge has conflicting PLC orientation.")
    for (source_id, occurrence_path), targets in edge_rows.items():
        target_ids = np.asarray(sorted(targets), dtype=np.int64)
        records.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                target.entity_set(1).entity_set_id,
                target_ids,
                (source_id,) * target_ids.size,
                np.zeros(target_ids.shape, dtype=np.float64),
                exact=True,
                source_roles=(association_owner.GeometrySourceEntityRole.EDGE,)
                * target_ids.size,
                source_occurrence_paths=(occurrence_path,) * target_ids.size,
                orientations=np.asarray(
                    [targets[int(identifier)] for identifier in target_ids],
                    dtype=np.int8,
                ),
            )
        )
    target_face_ids = np.asarray(target.entity_set(2).entity_ids, dtype=np.int64)
    target_face_keys = {
        key: int(identifier)
        for key, identifier in zip(
            canonical_owner._entity_vertex_keys(target, 2), target_face_ids, strict=True
        )
    }
    boundary_faces = np.asarray(core.boundary_faces, dtype=np.int64)
    face_source_facets = np.asarray(core.face_source_facets, dtype=np.int64)
    face_orientations = np.asarray(core.face_orientations, dtype=np.int8)
    face_source_ids = tuple(str(value) for value in core.face_source_ids)
    face_source_paths = tuple(core.face_source_occurrence_paths)
    if (
        face_source_facets.shape != (boundary_faces.shape[0],)
        or face_orientations.shape != face_source_facets.shape
        or len(face_source_ids) != face_source_facets.size
        or len(face_source_paths) != face_source_facets.size
    ):
        raise ValueError("Native constrained-facet ancestry arrays are incompatible.")
    face_targets: list[int] = []
    face_entities: list[str] = []
    face_sides: list[tuple[str | None, str | None]] = []
    face_signs: list[int] = []
    face_paths: list[tuple[str, ...]] = []
    core_region_map = np.asarray(source.core_region_map, dtype=np.int64)
    composite_region_ids = tuple(
        source_identity_owner.source_entity_id(
            source.source_id,
            source.source_revision,
            "region",
            int(entity_id),
            occurrence_path=path,
        )
        for entity_id, path in zip(
            source.region_entity_ids,
            source.region_occurrence_paths,
            strict=True,
        )
    )
    for face_row, (face, facet, orientation) in enumerate(
        zip(boundary_faces, face_source_facets, face_orientations, strict=True)
    ):
        combined_rows = core_vertex_map[face]
        global_vertices = np.asarray(target.vertex_global_ids, dtype=np.int64)[
            combined_rows
        ]
        face_targets.append(target_face_keys[tuple(sorted(global_vertices.tolist()))])
        inversions = sum(
            first > second
            for index, first in enumerate(global_vertices.tolist())
            for second in global_vertices[index + 1 :].tolist()
        )
        face_signs.append(int(orientation) * (-1 if inversions % 2 else 1))
        face_identity, face_path = _core_source_identity(
            source,
            2,
            face_source_ids[face_row],
            tuple(face_source_paths[face_row]),
        )
        face_entities.append(face_identity)
        positive, negative = (int(value) for value in complex_.facet_regions[int(facet)])
        face_sides.append(
            (
                None
                if negative < 0
                else composite_region_ids[int(core_region_map[negative])],
                None
                if positive < 0
                else composite_region_ids[int(core_region_map[positive])],
            )
        )
        face_paths.append(face_path)
    if face_targets:
        target_ids = np.asarray(face_targets, dtype=np.int64)
        if np.unique(target_ids).size != target_ids.size:
            raise ValueError("Two constrained PLC facets target one combined mesh face.")
        records.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                target.entity_set(2).entity_set_id,
                target_ids,
                tuple(face_entities),
                np.zeros(target_ids.shape, dtype=np.float64),
                exact=True,
                source_roles=(association_owner.GeometrySourceEntityRole.FACET,)
                * target_ids.size,
                source_occurrence_paths=tuple(face_paths),
                orientations=np.asarray(face_signs, dtype=np.int8),
                source_region_sides=tuple(face_sides),
            )
        )
    cells = np.sort(np.asarray(target.entity_set(3).entity_ids, dtype=np.int64))
    if combined_regions.shape != cells.shape:
        raise ValueError("Combined region ancestry must identify every target cell.")
    region_entities = tuple(
        composite_region_ids[int(region)] for region in combined_regions
    )
    records.append(
        association_owner.GeometryAssociation(
            association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            source.source_revision,
            target.entity_set(3).entity_set_id,
            cells,
            region_entities,
            np.zeros(cells.shape, dtype=np.float64),
            exact=True,
            source_roles=(association_owner.GeometrySourceEntityRole.REGION,)
            * cells.size,
            source_occurrence_paths=tuple(
                source.region_occurrence_paths[int(region)] for region in combined_regions
            ),
            orientations=np.zeros(cells.shape, dtype=np.int8),
        )
    )
    layer_vertex_columns = np.asarray(source.layers.vertex_column_ids, dtype=np.int64)
    layer_vertex_ids = np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64)
    if layer_vertex_columns.shape != layer_vertex_ids.shape:
        raise ValueError("Layer vertex column ancestry has incompatible extents.")
    generated_vertices = {
        int(identifier)
        for identifier, column in zip(layer_vertex_ids, layer_vertex_columns, strict=True)
        if column >= 0
    }
    if generated_vertices:
        vertex_regions: dict[int, set[int]] = {}
        target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
        for block in target.blocks:
            block_ids = np.asarray(block.global_ids, dtype=np.int64)
            block_vertices = np.asarray(block.vertices, dtype=np.int64)
            for cell_row in range(block_ids.size):
                cell_id = int(block_ids[cell_row])
                cell_position = int(np.searchsorted(cells, cell_id))
                if cell_position >= cells.size or cells[cell_position] != cell_id:
                    raise ValueError("A layer/core cell is absent from its region bank.")
                region = int(combined_regions[cell_position])
                for local_vertex in range(block_vertices.shape[1]):
                    vertex_row = int(block_vertices[cell_row, local_vertex])
                    vertex_id = int(target_vertex_ids[vertex_row])
                    if vertex_id in generated_vertices:
                        vertex_regions.setdefault(vertex_id, set()).add(region)
        existing_roles: dict[
            int, set[association_owner.GeometrySourceEntityRole | None]
        ] = {}
        vertex_set_id = target.entity_set(0).entity_set_id
        for association in records:
            if association.target_entity_set_id != vertex_set_id:
                continue
            for identifier, role in zip(
                np.asarray(association.target_global_ids, dtype=np.int64),
                association.source_roles,
                strict=True,
            ):
                existing_roles.setdefault(int(identifier), set()).add(role)
        interior_ids: list[int] = []
        interior_regions: list[int] = []
        for vertex_id in sorted(generated_vertices):
            adjacent = vertex_regions.get(vertex_id, set())
            if len(adjacent) != 1:
                continue
            region = next(iter(adjacent))
            roles = existing_roles.get(vertex_id, set())
            if any(
                role
                in (
                    association_owner.GeometrySourceEntityRole.VERTEX,
                    association_owner.GeometrySourceEntityRole.EDGE,
                    association_owner.GeometrySourceEntityRole.FACET,
                )
                for role in roles
            ):
                continue
            if association_owner.GeometrySourceEntityRole.REGION in roles:
                continue
            interior_ids.append(vertex_id)
            interior_regions.append(region)
        if interior_ids:
            identifiers = tuple(
                source_identity_owner.source_entity_id(
                    source.source_id,
                    source.source_revision,
                    "region",
                    int(source.region_entity_ids[region]),
                    occurrence_path=source.region_occurrence_paths[region],
                )
                for region in interior_regions
            )
            paths = tuple(
                source.region_occurrence_paths[region] for region in interior_regions
            )
            if isinstance(source.source_domain, meshing_domain_owner.MeshingDomain):
                for identifier, path in zip(identifiers, paths, strict=True):
                    source.source_domain.resolve_entity_id(
                        3, identifier, occurrence_path=path
                    )
            target_ids = np.asarray(interior_ids, dtype=np.int64)
            records.append(
                association_owner.GeometryAssociation(
                    association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                    source.source_id,
                    source.source_revision,
                    target.entity_set(0).entity_set_id,
                    target_ids,
                    identifiers,
                    np.zeros(target_ids.shape, dtype=np.float64),
                    exact=True,
                    source_roles=(association_owner.GeometrySourceEntityRole.REGION,)
                    * target_ids.size,
                    source_occurrence_paths=paths,
                    orientations=np.zeros(target_ids.shape, dtype=np.int8),
                )
            )
    return tuple(records)


__all__ = [
    "LayerSourceTables",
    "core_source_associations",
    "layer_source_associations",
    "layer_wall_associations",
    "layer_source_tables",
]
