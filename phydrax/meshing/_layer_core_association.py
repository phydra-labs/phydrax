#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Original layer/core authority tables and their native transfer composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..geometry._mesh_certificates import PiecewiseLinearDomain
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
    MappedReferenceAssociationTransfer,
    PlcAssociationTransfer,
)
from ._association_composition import ComposedAssociationTransfer
from ._layer_core_resources import _row_entity_vertex_keys, LayerCoreSourceWork


if TYPE_CHECKING:
    from ..discretization import CellMesh
    from ._boundary_layer import BoundaryLayerMesh
    from ._contracts import VolumeMeshingSpec
    from ._result import CellMeshingResult
    from .providers._native_sources import NativeLayerCoreSource


@dataclass(frozen=True, slots=True)
class _LayerSourceTables:
    domain: PiecewiseLinearDomain
    edge_vertices: np.ndarray
    edge_indices: np.ndarray
    triangle_vertices: np.ndarray
    triangle_facets: np.ndarray
    facet_regions: np.ndarray
    facet_indices: np.ndarray
    region_indices: np.ndarray


def _layer_source_tables(
    layers: BoundaryLayerMesh,
    layer_regions: np.ndarray,
    region_ids: tuple[str, ...],
    /,
    *,
    work: LayerCoreSourceWork | None = None,
) -> _LayerSourceTables:
    from ._layer_core import _faces, _triangles

    mesh = layers.mesh
    edges = np.asarray(_row_entity_vertex_keys(mesh, 1), dtype=np.int64)
    face_rows = _row_entity_vertex_keys(mesh, 2)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    materials = np.unique(np.asarray(layer_regions, dtype=np.int64))
    local_regions = np.searchsorted(materials, layer_regions).astype(np.int64)
    faces = _faces(mesh, local_regions, work=work)
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
            "The original cap registry has no int64 room for the remaining layer facets."
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
        raise ValueError("Original cap identities must be actual immutable layer facets.")
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
        pieces = _triangles(oriented)
        triangles.extend(pieces)
        triangle_facets.extend((row,) * len(pieces))
        if first.region != other:
            boundary_triangles.extend(pieces)
            boundary_incidence.extend((pair,) * len(pieces))
    if work is not None:
        work.charge(
            mesh.coordinates.shape[0] + edges.shape[0] + len(triangles) + len(face_rows)
        )
    domain = PiecewiseLinearDomain(
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


def _layer_source_associations(
    layers: BoundaryLayerMesh,
    target: CellMesh,
    layer_regions: np.ndarray,
    region_ids: tuple[str, ...],
    /,
    *,
    cap_association: GeometryAssociation | None,
    work: LayerCoreSourceWork,
) -> tuple[GeometryAssociation, ...]:
    tables = _layer_source_tables(layers, layer_regions, region_ids, work=work)
    source_vertex_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    records: list[GeometryAssociation] = []
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
                    _row_entity_vertex_keys(target, dimension),
                    strict=True,
                )
            }
            target_ids = np.asarray(
                [
                    target_keys[
                        tuple(sorted(int(source_vertex_ids[row]) for row in vertices))
                    ]
                    for vertices in _row_entity_vertex_keys(layers.mesh, dimension)
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
            GeometrySourceEntityRole.VERTEX,
            GeometrySourceEntityRole.EDGE,
            GeometrySourceEntityRole.FACET,
        )[dimension]
        records.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
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


def prepare_native_layer_association_transfer(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    result: CellMeshingResult,
    /,
) -> ComposedAssociationTransfer | MappedReferenceAssociationTransfer:
    """Prepare original source banks without regenerating or proximity pairing a mesh."""
    from ..discretization._cell_geometry import CellGeometrySpec
    from ._contracts import VolumeMeshingSpec
    from ._result import CellMeshingResult
    from ._volume_generation import prepare_plc_source
    from .providers._native_sources import NativeLayerCoreSource

    if (
        type(source) is not NativeLayerCoreSource
        or type(specification) is not VolumeMeshingSpec
        or type(result) is not CellMeshingResult
    ):
        raise TypeError(
            "Layer transfer requires its exact authored source, hard request, and accepted carrier."
        )
    if (
        result.certification is None
        or not result.certification.passed
        or result.compliance.specification_id != specification.specification_id
    ):
        raise ValueError(
            "Layer transfer requires the original accepted scientific generation request."
        )
    binding = result.trace.binding
    if binding is None or (binding.source_id, binding.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise ValueError("Layer transfer must bind the actual original source revision.")
    domain = result.certification.request.domain
    if source.mapped_domain is not None:
        if domain is None or domain.domain_id != source.mapped_domain.domain_id:
            raise ValueError(
                "Mapped layer transfer must retain its actual original source roots and material domain."
            )
        return MappedReferenceAssociationTransfer(
            source.mapped_domain,
            maximum_support_queries=min(
                specification.limits.maximum_geometry_queries,
                specification.limits.maximum_work_units,
            ),
        )
    if type(domain) is not PiecewiseLinearDomain or domain.source_id != source.source_id:
        raise ValueError(
            "Layer transfer requires its complete independently retained material domain."
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
            "Original layer/core source-bank preparation exceeds declared byte admission."
        )
    work = LayerCoreSourceWork(limits.maximum_work_units)
    layer = _layer_source_tables(
        source.layers, source.layer_regions, source.region_ids, work=work
    )
    core = prepare_plc_source(
        source.complex,
        source.complex.complex_id,
        source.complex.complex_id,
        result.coordinate_contract,
        limits=limits,
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
            "Original source preparation exhausts composed source support admission."
        )
    layer_plan = PlcAssociationTransfer(
        layer.domain,
        result.coordinate_contract,
        source.layers.result_id,
        edge_vertices=layer.edge_vertices,
        edge_indices=layer.edge_indices,
        vertex_indices=np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64),
        triangle_vertices=layer.triangle_vertices,
        triangle_facets=layer.triangle_facets,
        facet_regions=layer.facet_regions,
        region_indices=layer.region_indices,
        maximum_support_queries=queries,
    )
    edges = np.unique(
        np.sort(domain.facets[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0
    )
    material_plan = PlcAssociationTransfer(
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
    return ComposedAssociationTransfer(
        (material_plan, core.association_transfer, layer_plan),
        np.concatenate(facets),
        np.cumsum(np.asarray((0, *(value.size for value in facets)), dtype=np.int64)),
        np.concatenate(regions),
        np.cumsum(np.asarray((0, *(value.size for value in regions)), dtype=np.int64)),
        (None, None, source.layers.mesh),
        (None, None, CellGeometrySpec.affine(source.layers.mesh)),
        material_namespace=(source.source_id, source.source_revision),
        maximum_support_queries=min(
            queries, core.association_transfer.maximum_support_queries
        ),
        maximum_data_bytes=limits.maximum_data_bytes,
        preparation_work_units=native_work,
    )


__all__ = ["prepare_native_layer_association_transfer"]
