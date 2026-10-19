#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Layer/core incidence and identity preparation shared by composition phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import numpy as np
from jax.typing import ArrayLike

import phydrax.geometry._domain_coverage as _dependency_phydrax_geometry__mesh_certificates
import phydrax.geometry._mapped_reference_domain as _dependency_phydrax_geometry__mapped_reference_domain
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._boundary_layer as _dependency_phydrax_meshing__boundary_layer
import phydrax.meshing._layer_core_resources as _dependency_phydrax_meshing__layer_core_resources
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace
import phydrax.meshing._volume_generation as _dependency_phydrax_meshing__volume_generation

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, reference_cell_topology
from ..typing import checked
from ._compliance_failure import compliance_failure


def cycle(row: tuple[int, ...], /) -> tuple[int, ...]:
    offset = row.index(min(row))
    return row[offset:] + row[:offset]


def bitwise_equal_points(left: np.ndarray, right: np.ndarray, /) -> bool:
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def triangles(row: tuple[int, ...], /) -> tuple[tuple[int, int, int], ...]:
    if len(row) == 3:
        return ((row[0], row[1], row[2]),)
    if len(row) != 4:
        raise ValueError(
            f"LayerCoreFace.vertices must contain three or four vertices; got {len(row)}"
        )
    a, b, c, d = cycle(row)
    return ((a, b, c), (a, c, d))


def integer_array(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"{'NativeLayerCoreSource.' + name} must be an integer array of shape {shape}; got shape {array.shape} and dtype {array.dtype}"
        )
    return array.astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class LayerCoreFace:
    vertices: tuple[int, ...]
    cell: int
    region: int


@checked
def faces(
    mesh: CellMesh,
    regions: np.ndarray,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> dict[tuple[int, ...], list[LayerCoreFace]]:
    """Reference-oriented faces in global-cell-ID order, with ownership."""
    identifiers = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    table: dict[tuple[int, ...], list[LayerCoreFace]] = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        rows = np.asarray(block.vertices, dtype=np.int64)
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        for row_index in range(block.cell_count):
            identifier = cell_ids[row_index]
            index = int(np.searchsorted(identifiers, identifier))
            for local in topology.entities[2]:
                work.charge(1)
                vertices = tuple(int(rows[row_index, vertex]) for vertex in local)
                key = tuple(sorted(vertices))
                table.setdefault(key, []).append(
                    LayerCoreFace(vertices, index, int(regions[index]))
                )
    for incidents in table.values():
        if len(incidents) > 2:
            raise compliance_failure(
                "CellMesh.blocks must have at most two oppositely oriented incidents per shared face; got layer/core faces have more than two incident cells",
                stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
            )
        if len(incidents) == 2 and cycle(incidents[0].vertices) != cycle(
            tuple(reversed(incidents[1].vertices))
        ):
            raise compliance_failure(
                "CellMesh.blocks must have at most two oppositely oriented incidents per shared face; got layer/core shared faces have inconsistent orientation",
                stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
            )
    return table


@final
class LayerCoreConstruction(StrictModule, NonTrainableState):
    """Combined mesh and full coverage inputs for native result publication."""

    mesh: CellMesh
    domain: (
        _dependency_phydrax_geometry__mesh_certificates.PiecewiseLinearDomain
        | _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
    )
    cell_regions: np.ndarray
    zones: tuple[_dependency_phydrax_meshing__organization.MeshZone, ...]
    patches: tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...]
    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...]
    attributes: tuple[_dependency_phydrax_meshing__organization.MeshAttribute, ...]
    associations: tuple[association_state_owner.GeometryAssociation, ...]
    stages: tuple[_dependency_phydrax_meshing__trace.MeshingStageReport, ...]
    core: _dependency_phydrax_meshing__volume_generation.VolumeConstruction
    construction_id: str
    work_units: int


@checked
def prepare_identity(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if complex_.boundary != "fixed":
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids must provide the immutable PLC boundary policy; got missing or incompatible the immutable PLC boundary policy"
        )
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    mapping = integer_array(
        vertex_layer_ids, (complex_.vertices.shape[0],), "vertex_layer_ids"
    )
    work.charge(mapping.size)
    if np.any(mapping < -1) or np.any(mapping >= layer_points.shape[0]):
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids must bind a fixed triangular PLC, distinct shared vertices, and an oriented cap; got vertex_layer_ids index absent layer vertices"
        )
    shared = mapping >= 0
    if np.unique(mapping[shared]).size != np.count_nonzero(shared):
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids must bind a fixed triangular PLC, distinct shared vertices, and an oriented cap; got distinct PLC vertices cannot identify one layer vertex"
        )
    work.charge(int(np.count_nonzero(shared)))
    if not bitwise_equal_points(complex_.vertices[shared], layer_points[mapping[shared]]):
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids must be bitwise identical; got incompatible Authoritatively shared cap/rim coordinates"
        )
    if layers.cap is None:
        cap_rows = np.empty((0, 3), dtype=np.int64)
    else:
        if any(block.cell_kind != "triangle" for block in layers.cap.blocks):
            raise ValueError(
                "NativeLayerCoreSource.vertex_layer_ids must bind a fixed triangular PLC, distinct shared vertices, and an oriented cap; got a tetrahedral core needs a triangular cap; enable transition pyramids"
            )
        cap_rows = np.asarray(layers.cap_vertices, dtype=np.int64)[
            np.concatenate(
                [
                    np.asarray(block.vertices, dtype=np.int64)
                    for block in layers.cap.blocks
                ]
            )
        ]
    cap_ids = integer_array(cap_polygon_ids, (cap_rows.shape[0],), "cap_polygon_ids")
    polygon_count = complex_.polygon_facets.shape[0]
    if (
        np.any(cap_ids < 0)
        or np.any(cap_ids >= polygon_count)
        or np.unique(cap_ids).size != cap_ids.size
    ):
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids must select each cap polygon exactly once; got incompatible cap_polygon_ids"
        )
    lengths = np.diff(complex_.polygon_offsets)
    if np.any(lengths != 3):
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids must all be triangles; got incompatible Fixed layer-core PLC polygons"
        )
    polygons = complex_.polygon_vertices.reshape(-1, 3)
    for cap_row, polygon in zip(cap_rows, polygons[cap_ids], strict=True):
        work.charge(1)
        mapped = mapping[polygon]
        if np.any(mapped < 0) or cycle(tuple(mapped.tolist())) != cycle(
            tuple(cap_row.tolist())
        ):
            raise ValueError(
                "NativeLayerCoreSource.cap_polygon_ids must match layer vertex IDs and orientation; got a mismatched cap polygon"
            )
    cap_incidence = complex_.facet_regions[complex_.polygon_facets[cap_ids]]
    if np.any(cap_incidence[:, 0] < 0) or np.any(cap_incidence[:, 1] != -1):
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids must have core on its positive side and void on its negative side; got incompatible oriented cap"
        )
    combined_map = mapping.copy()
    combined_map[~shared] = layer_points.shape[0] + np.arange(np.count_nonzero(~shared))
    return (
        combined_map,
        cap_ids,
        polygons,
        np.concatenate((layer_points, complex_.vertices[~shared])),
    )


@checked
def combined_domain(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    layer_regions: np.ndarray,
    mapping: np.ndarray,
    cap_ids: np.ndarray,
    polygons: np.ndarray,
    points: np.ndarray,
    source_id: str,
    region_ids: tuple[str, ...],
    core_region_map: np.ndarray,
    /,
    *,
    work: _dependency_phydrax_meshing__layer_core_resources.LayerCoreSourceWork,
) -> tuple[
    _dependency_phydrax_geometry__mesh_certificates.PiecewiseLinearDomain,
    dict[tuple[int, ...], list[LayerCoreFace]],
]:
    layer_faces = faces(layers.mesh, layer_regions, work=work)
    cap_keys = {
        tuple(sorted(mapping[polygons[index]].tolist())): int(index) for index in cap_ids
    }
    facets: list[tuple[int, int, int]] = []
    incidence: list[tuple[int, int]] = []
    for key, incidents in layer_faces.items():
        if key in cap_keys:
            if len(incidents) != 1:
                raise compliance_failure(
                    "NativeLayerCoreSource.cap_polygon_ids must identify exterior layer facets with matching cap orientation; got a declared cap is not an exterior face of the layers",
                    stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
                )
            polygon = mapping[polygons[cap_keys[key]]]
            if cycle(incidents[0].vertices) != cycle(tuple(polygon.tolist())):
                raise compliance_failure(
                    "NativeLayerCoreSource.cap_polygon_ids must identify exterior layer facets with matching cap orientation; got the declared cap orientation disagrees with the layer cell boundary",
                    stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
                )
            continue
        first = incidents[0]
        second_region = -1 if len(incidents) == 1 else incidents[1].region
        if first.region != second_region:
            for triangle in triangles(first.vertices):
                facets.append(triangle)
                incidence.append((first.region, second_region))
    if not set(cap_keys) <= set(layer_faces):
        raise compliance_failure(
            "NativeLayerCoreSource.cap_polygon_ids must identify exterior layer facets with matching cap orientation; got the core includes a cap facet absent from the layer boundary",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
        )
    for index, (polygon, facet) in enumerate(
        zip(polygons, complex_.polygon_facets, strict=True)
    ):
        positive, negative = (int(value) for value in complex_.facet_regions[facet])
        positive = int(core_region_map[positive]) if positive >= 0 else -1
        negative = int(core_region_map[negative]) if negative >= 0 else -1
        mapped = tuple(mapping[polygon].tolist())
        if index in cap_ids:
            negative = layer_faces[tuple(sorted(mapped))][0].region
        if positive != negative:
            facets.append((mapped[0], mapped[1], mapped[2]))
            # PLC incidence is (positive, negative); domain incidence follows normal.
            incidence.append((negative, positive))
    return _dependency_phydrax_geometry__mesh_certificates.PiecewiseLinearDomain(
        points,
        np.asarray(facets, dtype=np.int64),
        np.asarray(incidence, dtype=np.int64),
        region_ids,
        source_id=source_id,
    ), layer_faces
