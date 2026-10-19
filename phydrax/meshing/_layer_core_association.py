#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source layer/core provenance tables and their native transfer composition."""

from __future__ import annotations

from contextlib import ExitStack
from typing import final

import numpy as np

import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.geometry._domain_coverage as _dependency_phydrax_geometry__mesh_certificates
import phydrax.meshing._association as _dependency_phydrax_meshing__association
import phydrax.meshing._association_composition as _dependency_phydrax_meshing__association_composition
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._contracts as _owner_phydrax_meshing__contracts
import phydrax.meshing._layer_core_resources as _dependency_phydrax_meshing__layer_core_resources
import phydrax.meshing._layer_core_topology as _owner_phydrax_meshing__layer_core
import phydrax.meshing._result as _owner_phydrax_meshing__result
import phydrax.meshing._volume_generation as _owner_phydrax_meshing__volume_generation
import phydrax.meshing.providers._native_sources as _owner_phydrax_meshing_providers__native_sources

from .._meshcore import NativeExecutionScope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, reference_cell_topology
from ..typing import checked
from ._boundary_layer import BoundaryLayerMesh


@final
class _LayerSourceTables(StrictModule, NonTrainableState):
    domain: _dependency_phydrax_geometry__mesh_certificates.PiecewiseLinearDomain
    edge_vertices: np.ndarray
    edge_indices: np.ndarray
    triangle_vertices: np.ndarray
    triangle_facets: np.ndarray
    facet_regions: np.ndarray
    facet_indices: np.ndarray
    region_indices: np.ndarray


@checked
def layer_source_tables(
    layers: BoundaryLayerMesh,
    layer_regions: np.ndarray,
    region_ids: tuple[str, ...],
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> _LayerSourceTables:

    mesh = layers.mesh
    edges = np.asarray(
        _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(mesh, 1),
        dtype=np.int64,
    )
    face_rows = _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
        mesh, 2
    )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    materials = np.unique(np.asarray(layer_regions, dtype=np.int64))
    local_regions = np.searchsorted(materials, layer_regions).astype(np.int64)
    faces = _owner_phydrax_meshing__layer_core.faces(mesh, local_regions, work=work)
    cap = layers.cap
    cap_keys: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
    if cap is not None:
        cap_vertices = np.asarray(layers.cap_vertices, dtype=np.int64)
        for block in cap.blocks:
            for identifier, row in zip(
                np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
            ):
                oriented = tuple(int(cap_vertices[vertex]) for vertex in row)
                cap_keys[tuple(sorted(oriented))] = (int(identifier), oriented)
    next_identifier = max((value[0] for value in cap_keys.values()), default=-1) + 1
    noncap_count = len(face_rows) - len(cap_keys)
    if next_identifier + noncap_count - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "BoundaryLayerMesh.cap must retain distinct declared facet identities within int64 capacity; got the source cap registry has no int64 room for the remaining layer facets"
        )
    identifiers = np.empty(len(face_rows), dtype=np.int64)
    for row in np.argsort(face_ids, kind="stable"):
        key = face_rows[row]
        if key in cap_keys:
            identifiers[row] = cap_keys[key][0]
        else:
            identifiers[row] = next_identifier
            next_identifier += 1
    if not set(cap_keys) <= set(face_rows):
        raise ValueError(
            "BoundaryLayerMesh.cap must be realized immutable layer facets; got incompatible Source cap identities"
        )
    triangles: list[tuple[int, int, int]] = []
    triangle_facets: list[int] = []
    incidence = np.empty((len(face_rows), 2), dtype=np.int64)
    boundary_triangles: list[tuple[int, int, int]] = []
    boundary_incidence: list[tuple[int, int]] = []
    for row, key in enumerate(face_rows):
        owners = faces[key]
        first = owners[0]
        other = owners[1].region if len(owners) == 2 else -1
        pair = (first.region, other)
        incidence[row] = pair
        oriented = cap_keys[key][1] if key in cap_keys else first.vertices
        pieces = _owner_phydrax_meshing__layer_core.triangles(oriented)
        triangles.extend(pieces)
        triangle_facets.extend((row,) * len(pieces))
        if first.region != other:
            boundary_triangles.extend(pieces)
            boundary_incidence.extend((pair,) * len(pieces))
    work.charge(
        mesh.coordinates.shape[0] + edges.shape[0] + len(triangles) + len(face_rows)
    )
    domain = _dependency_phydrax_geometry__mesh_certificates.PiecewiseLinearDomain(
        np.asarray(mesh.coordinates, dtype=np.float64),
        np.asarray(boundary_triangles, dtype=np.int64),
        np.asarray(boundary_incidence, dtype=np.int64),
        tuple(region_ids[int(index)] for index in materials),
        source_id=layers.result_id,
    )
    return _LayerSourceTables(
        domain,
        edges,
        np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64),
        np.asarray(triangles, dtype=np.int64),
        np.asarray(triangle_facets, dtype=np.int64),
        incidence,
        identifiers,
        materials,
    )


@checked
def layer_source_associations(
    layers: BoundaryLayerMesh,
    target: CellMesh,
    layer_regions: np.ndarray,
    region_ids: tuple[str, ...],
    /,
    *,
    cap_association: association_state_owner.GeometryAssociation | None,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> tuple[association_state_owner.GeometryAssociation, ...]:
    tables = layer_source_tables(layers, layer_regions, region_ids, work=work)
    source_vertex_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    records: list[association_state_owner.GeometryAssociation] = []
    for dimension in (0, 1, 2):
        original_ids = np.asarray(
            layers.mesh.entity_set(dimension).entity_ids, dtype=np.int64
        )
        if dimension == 0:
            target_ids = source_vertex_ids
            source_indices = source_vertex_ids
        else:
            target_vertex_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
            target_keys = {
                tuple(sorted(int(target_vertex_ids[row]) for row in vertices)): int(
                    identifier
                )
                for identifier, vertices in zip(
                    np.asarray(target.entity_set(dimension).entity_ids),
                    _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                        target, dimension
                    ),
                    strict=True,
                )
            }
            target_ids = np.asarray(
                [
                    target_keys[
                        tuple(sorted(int(source_vertex_ids[row]) for row in vertices))
                    ]
                    for vertices in _dependency_phydrax_meshing__layer_core_resources.row_entity_vertex_keys(
                        layers.mesh, dimension
                    )
                ],
                dtype=np.int64,
            )
            source_indices = original_ids if dimension == 1 else tables.facet_indices
        signs = (
            np.ones(target_ids.shape, dtype=np.int8)
            if dimension == 1
            else np.zeros(target_ids.shape, dtype=np.int8)
        )
        if dimension == 2 and cap_association is not None:
            original_signs = dict(
                zip(
                    np.asarray(cap_association.source_indices).tolist(),
                    np.asarray(cap_association.orientations).tolist(),
                    strict=True,
                )
            )
            for row, identifier in enumerate(source_indices):
                if int(identifier) in original_signs:
                    signs[row] = original_signs[int(identifier)]
        role = (
            association_state_owner.GeometrySourceEntityRole.VERTEX,
            association_state_owner.GeometrySourceEntityRole.EDGE,
            association_state_owner.GeometrySourceEntityRole.FACET,
        )[dimension]
        records.append(
            association_state_owner.GeometryAssociation(
                association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                layers.result_id,
                layers.result_id,
                target.entity_set(dimension).entity_set_id,
                target_ids,
                tuple(
                    f"{layers.result_id}:{role.value}:{int(index)}"
                    for index in source_indices
                ),
                np.zeros(target_ids.shape, dtype=np.float64),
                exact=True,
                source_dimensions=np.full(target_ids.shape, dimension, dtype=np.int8),
                source_indices=source_indices,
                source_entity_roles=(role,) * target_ids.size,
                orientations=signs,
            )
        )
    return tuple(records)


@checked
def prepare_native_layer_association_transfer(
    source: _owner_phydrax_meshing_providers__native_sources.NativeLayerCoreSource,
    specification: _owner_phydrax_meshing__contracts.VolumeMeshingSpec,
    result: _owner_phydrax_meshing__result.CellMeshingResult,
    /,
) -> (
    _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer
    | _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer
):
    """Prepare source banks without regenerating or proximity pairing a mesh."""
    with ExitStack() as resources:
        if (
            type(source)
            is not _owner_phydrax_meshing_providers__native_sources.NativeLayerCoreSource
            or type(specification)
            is not _owner_phydrax_meshing__contracts.VolumeMeshingSpec
            or type(result) is not _owner_phydrax_meshing__result.CellMeshingResult
        ):
            raise TypeError(
                "prepare_native_layer_association_transfer.source must provide its authored source, hard request, and accepted carrier; got missing or incompatible its authored source, hard request, and accepted carrier"
            )
        if (
            result.certification is None
            or not result.certification.passed
            or result.compliance.specification_id != specification.specification_id
        ):
            raise ValueError(
                "prepare_native_layer_association_transfer.source must provide the source accepted scientific generation request; got missing or incompatible the source accepted scientific generation request"
            )
        binding = result.trace.binding
        if binding is None or (binding.source_id, binding.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            raise ValueError(
                "prepare_native_layer_association_transfer.source must bind the realized source revision; got incompatible Layer transfer"
            )
        domain = result.certification.request.domain
        if source.mapped_domain is not None:
            if domain is None or domain.domain_id != source.mapped_domain.domain_id:
                raise ValueError(
                    "prepare_native_layer_association_transfer.source must retain its realized source roots and material domain; got incompatible Mapped layer transfer"
                )
            return _dependency_phydrax_meshing__association.MappedReferenceAssociationTransfer(
                source.mapped_domain,
                maximum_support_queries=min(
                    specification.limits.maximum_geometry_queries,
                    specification.limits.maximum_work_units,
                ),
            )
        if (
            type(domain)
            is not _dependency_phydrax_geometry__mesh_certificates.PiecewiseLinearDomain
            or domain.source_id != source.source_id
        ):
            raise ValueError(
                "prepare_native_layer_association_transfer.source must provide its full independently retained material domain; got missing or incompatible its full independently retained material domain"
            )
        limits = specification.limits
        estimate = 256 * (
            source.layers.mesh.coordinates.shape[0]
            + source.layers.mesh.entity_set(1).count
            + source.layers.mesh.entity_set(2).count
            + source.complex.vertices.shape[0]
            + source.complex.polygon_vertices.size
            + domain.facets.shape[0]
        )
        if estimate > limits.maximum_data_bytes:
            raise ValueError(
                "prepare_native_layer_association_transfer.source must retain accepted request, source revision, roots, and material support within admission limits; got source layer/core source-bank preparation exceeds declared byte admission"
            )
        scope = resources.enter_context(NativeExecutionScope(limits.native_limits()))
        with scope.batch(
            source.layers.mesh.coordinates.shape[0]
            + sum(
                block.cell_count
                * (
                    2 * len(reference_cell_topology(block.cell_kind).entities[2])
                    + len(reference_cell_topology(block.cell_kind).entities[1])
                    + sum(
                        len(face) - 2
                        for face in reference_cell_topology(block.cell_kind).entities[2]
                    )
                )
                for block in source.layers.mesh.blocks
            )
        ) as ticket:
            work = _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork(
                limits.maximum_work_units, ticket=ticket
            )
            layer = layer_source_tables(
                source.layers, source.layer_regions, source.region_ids, work=work
            )
        core = _owner_phydrax_meshing__volume_generation.prepare_plc_source(
            source.complex,
            source.complex.complex_id,
            source.complex.complex_id,
            result.coordinate_contract,
            limits=limits,
            scope=scope,
            source_work_units=work.work_units,
        )
        preparation = dict(core.preparation_counters)
        native_work = preparation["work_units"]
        native_queries = (
            preparation["native_geometry_primitive_queries"]
            + preparation["source_geometry_queries"]
        )
        queries = min(
            limits.maximum_geometry_queries - native_queries,
            limits.maximum_work_units - native_work,
        )
        if queries <= 0:
            raise ValueError(
                "prepare_native_layer_association_transfer.source must retain accepted request, source revision, roots, and material support within admission limits; got source preparation exhausts composed source support admission"
            )
        layer_plan = _dependency_phydrax_meshing__association.PlcAssociationTransfer(
            layer.domain,
            result.coordinate_contract,
            source.layers.result_id,
            edge_vertices=layer.edge_vertices,
            edge_indices=layer.edge_indices,
            vertex_indices=np.asarray(
                source.layers.mesh.vertex_global_ids, dtype=np.int64
            ),
            triangle_vertices=layer.triangle_vertices,
            triangle_facets=layer.triangle_facets,
            facet_regions=layer.facet_regions,
            region_indices=layer.region_indices,
            maximum_support_queries=queries,
        )
        edges = np.unique(
            np.sort(domain.facets[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1),
            axis=0,
        )
        material_plan = _dependency_phydrax_meshing__association.PlcAssociationTransfer(
            domain,
            result.coordinate_contract,
            source.source_revision,
            edge_vertices=edges,
            triangle_vertices=domain.facets,
            triangle_facets=np.arange(domain.facets.shape[0], dtype=np.int64),
            facet_regions=domain.facet_regions,
            maximum_support_queries=queries,
        )
        facets = (
            np.arange(domain.facets.shape[0], dtype=np.int64),
            np.arange(source.complex.facet_count, dtype=np.int64),
            layer.facet_indices,
        )
        regions = (
            np.arange(len(source.region_ids), dtype=np.int64),
            source.core_region_map,
            layer.region_indices,
        )
        return _dependency_phydrax_meshing__association_composition.ComposedAssociationTransfer(
            (material_plan, core.association_transfer, layer_plan),
            np.concatenate(facets),
            np.cumsum(np.asarray((0, *(value.size for value in facets)), dtype=np.int64)),
            np.concatenate(regions),
            np.cumsum(
                np.asarray((0, *(value.size for value in regions)), dtype=np.int64)
            ),
            (None, None, source.layers.mesh),
            (
                None,
                None,
                _owner_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(
                    source.layers.mesh
                ),
            ),
            material_namespace=(source.source_id, source.source_revision),
            maximum_support_queries=min(
                queries, core.association_transfer.maximum_support_queries
            ),
            maximum_data_bytes=limits.maximum_data_bytes,
            preparation_work_units=native_work,
        )


__all__ = ["prepare_native_layer_association_transfer"]
