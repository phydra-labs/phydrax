#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Layer/core incidence and identity preparation shared by composition phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._nested_reference_types as nested_reference_owner
import phydrax.discretization._periodic_topology as periodic_topology_owner
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._mapped_reference_domain as mapped_domain_owner
import phydrax.meshing._association as association_owner
import phydrax.meshing._boundary_layer as boundary_layer_owner
import phydrax.meshing._organization as organization_owner
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._volume_generation as volume_generation_owner
import phydrax.meshing._polyhedral_generation as polyhedral_generation_owner

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, reference_cell_topology
from ..typing import checked
from ._scope import MeshingEntityKind, MeshingScope


def cycle(row: tuple[int, ...], /) -> tuple[int, ...]:
    """Return the cyclic rotation with the smallest vertex first."""
    offset = row.index(min(row))
    return row[offset:] + row[:offset]


def bitwise_equal_points(left: np.ndarray, right: np.ndarray, /) -> bool:
    """Compare coordinate arrays without treating signed zero as identical."""
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def triangles(row: tuple[int, ...], /) -> tuple[tuple[int, int, int], ...]:
    """Triangulate a triangle or quad using its canonical minimum-vertex diagonal."""
    if len(row) == 3:
        return ((row[0], row[1], row[2]),)
    if len(row) != 4:
        raise ValueError(
            f"LayerCoreFace.vertices must contain three or four vertices; got {len(row)}"
        )
    a, b, c, d = cycle(row)
    return ((a, b, c), (a, c, d))


def integer_array(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> np.ndarray:
    """Require an integer array with a representable int64 identity range."""
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"NativeLayerCoreSource.{name} must be an integer array of shape {shape}; got shape {array.shape} and dtype {array.dtype}"
        )
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if np.any(array > np.iinfo(np.int64).max):
            raise ValueError(
                f"NativeLayerCoreSource.{name} contains identities outside int64 range"
            )
    elif array.dtype.itemsize > np.dtype(np.int64).itemsize and (
        np.any(array < np.iinfo(np.int64).min) or np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(
            f"NativeLayerCoreSource.{name} contains identities outside int64 range"
        )
    return array.astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class LayerCoreFace:
    vertices: tuple[int, ...]
    cell: int
    region: int


@checked
def polyhedral_cell_faces(
    mesh: CellMesh, /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return exact outward cell loops using canonical signed face incidence."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity):
        raise TypeError("mesh must have canonical polyhedral connectivity.")
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    result: list[tuple[tuple[int, ...], ...]] = []
    for cell in range(connectivity.cell_count):
        loops: list[tuple[int, ...]] = []
        for occurrence in range(
            int(cell_offsets[cell]), int(cell_offsets[cell + 1])
        ):
            face = int(cell_faces[occurrence])
            orientation = int(cell_signs[occurrence])
            if orientation not in (-1, 1):
                raise ValueError("Polyhedral face-use orientation must be signed unit data.")
            loop = face_vertices[
                int(face_offsets[face]) : int(face_offsets[face + 1])
            ]
            if orientation < 0:
                loop = loop[::-1]
            loops.append(tuple(int(value) for value in loop))
        result.append(tuple(loops))
    return tuple(result)


@checked
def faces(
    mesh: CellMesh, regions: np.ndarray, /
) -> dict[tuple[int, ...], list[LayerCoreFace]]:
    """Return oriented faces and their owning region in global-cell-ID order."""
    identifiers = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    if regions.shape != identifiers.shape or not np.issubdtype(regions.dtype, np.integer):
        raise ValueError("Layer/core regions must identify every volume cell.")
    table: dict[tuple[int, ...], list[LayerCoreFace]] = {}
    if isinstance(mesh.connectivity, cell_complex_owner.PolyhedralConnectivity):
        connectivity = mesh.connectivity
        connectivity_cell_ids = np.asarray(
            connectivity.cell_global_ids, dtype=np.int64
        )
        cell_faces_by_row = polyhedral_cell_faces(mesh)
        for cell_row, cell_id in enumerate(connectivity_cell_ids.tolist()):
            region_row = int(np.searchsorted(identifiers, cell_id))
            if (
                region_row == identifiers.size
                or identifiers[region_row] != cell_id
            ):
                raise ValueError("A volume cell is absent from its canonical entity set.")
            for vertices in cell_faces_by_row[cell_row]:
                key = tuple(sorted(vertices))
                table.setdefault(key, []).append(
                    LayerCoreFace(vertices, region_row, int(regions[region_row]))
                )
    else:
        for block in mesh.blocks:
            topology = reference_cell_topology(block.cell_kind)
            rows = np.asarray(block.vertices, dtype=np.int64)
            cell_ids = np.asarray(block.global_ids, dtype=np.int64)
            for row_index in range(block.cell_count):
                identifier = int(cell_ids[row_index])
                index = int(np.searchsorted(identifiers, identifier))
                if index == identifiers.size or identifiers[index] != identifier:
                    raise ValueError(
                        "A volume cell is absent from its canonical entity set."
                    )
                for local in topology.entities[2]:
                    vertices = tuple(int(rows[row_index, vertex]) for vertex in local)
                    key = tuple(sorted(vertices))
                    table.setdefault(key, []).append(
                        LayerCoreFace(vertices, index, int(regions[index]))
                    )
    for incidents in table.values():
        if len(incidents) > 2:
            raise ValueError("Layer/core faces must have at most two incident cells.")
        if len(incidents) == 2 and cycle(incidents[0].vertices) != cycle(
            tuple(reversed(incidents[1].vertices))
        ):
            raise ValueError("Layer/core shared faces have inconsistent orientation.")
    return table


@final
class LayerCoreConstruction(StrictModule, NonTrainableState):
    """Complete layer/core construction data awaiting canonical acceptance.

    Mapped compositions retain their precomposition reference-frame owners so
    source-root charts can be rederived without inverting physical maps.
    """

    mesh: CellMesh
    domain: (
        domain_coverage_owner.PiecewiseLinearDomain
        | mapped_domain_owner.MappedReferenceDomain
    )
    cell_regions: np.ndarray
    zones: tuple[organization_owner.MeshZone, ...]
    patches: tuple[organization_owner.MeshPatch, ...]
    labels: tuple[organization_owner.MeshLabel, ...]
    attributes: tuple[organization_owner.MeshAttribute, ...]
    associations: tuple[association_owner.GeometryAssociation, ...]
    geometry: cell_geometry_owner.CellGeometrySpec
    core: volume_generation_owner.VolumeConstruction
    core_vertex_map: np.ndarray
    core_cell_ids: np.ndarray
    region_ids: tuple[str, ...]
    source_id: str
    source_revision: str
    periodic_topology: periodic_topology_owner.PeriodicMeshTopology | None
    periodic_requested: tuple[tuple[str, float], ...]
    periodic_achieved: tuple[tuple[str, float], ...]
    reference_maps: tuple[nested_reference_owner.ReferenceCellMapWitnesses, ...]
    source_mesh: CellMesh | None
    source_geometry: cell_geometry_owner.CellGeometrySpec | None
    reference_mesh: CellMesh | None = None
    reference_geometry: cell_geometry_owner.CellGeometrySpec | None = None
    polyhedral_core: polyhedral_generation_owner.PolyhedralConstruction | None = None


@checked
def prepare_identity(
    layers: boundary_layer_owner.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    vertex_layer_ids: ArrayLike,
    cap_polygon_ids: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bind fixed PLC vertices and oriented cap facets to accepted layer IDs."""
    if complex_.boundary != "fixed":
        raise ValueError(
            "NativeLayerCoreSource requires an immutable fixed PLC boundary."
        )
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    mapping = integer_array(
        vertex_layer_ids, (complex_.vertices.shape[0],), "vertex_layer_ids"
    )
    if np.any(mapping < -1) or np.any(mapping >= layer_points.shape[0]):
        raise ValueError(
            "NativeLayerCoreSource.vertex_layer_ids indexes absent layer vertices."
        )
    shared = mapping >= 0
    if np.unique(mapping[shared]).size != np.count_nonzero(shared):
        raise ValueError(
            "Distinct PLC vertices cannot identify one accepted layer vertex."
        )
    if not bitwise_equal_points(complex_.vertices[shared], layer_points[mapping[shared]]):
        raise ValueError(
            "Shared PLC and accepted layer vertices must be bitwise identical."
        )
    if layers.cap is None:
        cap_rows = np.empty((0, 3), dtype=np.int64)
    else:
        if any(block.cell_kind != "triangle" for block in layers.cap.blocks):
            raise ValueError("A tetrahedral core requires a triangular layer cap.")
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
            "cap_polygon_ids must select distinct polygons in the fixed PLC."
        )
    lengths = np.diff(complex_.polygon_offsets)
    if np.any(lengths != 3):
        raise ValueError("The fixed layer/core PLC must consist of triangles.")
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    for cap_row, polygon in zip(cap_rows, polygons[cap_ids], strict=True):
        mapped = mapping[polygon]
        if np.any(mapped < 0) or cycle(tuple(mapped.tolist())) != cycle(
            tuple(cap_row.tolist())
        ):
            raise ValueError(
                "Each PLC cap polygon must retain its accepted layer vertices and orientation."
            )
    cap_incidence = complex_.facet_regions[complex_.polygon_facets[cap_ids]]
    if np.any(cap_incidence[:, 0] < 0) or np.any(cap_incidence[:, 1] != -1):
        raise ValueError(
            "Cap facets must orient the core to the positive side and void to the negative side."
        )
    combined_map = mapping.copy()
    combined_map[~shared] = layer_points.shape[0] + np.arange(
        np.count_nonzero(~shared), dtype=np.int64
    )
    return (
        combined_map,
        cap_ids,
        polygons,
        np.concatenate(
            (layer_points, np.asarray(complex_.vertices[~shared], dtype=np.float64))
        ),
    )


def mesh_entity_scope(
    mesh: CellMesh, dimension: int, identifiers: ArrayLike, /
) -> MeshingScope:
    """Bind mesh IDs to the exact output entity set represented by a scope."""
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        identifiers,
    )


@checked
def combined_domain(
    layers: boundary_layer_owner.BoundaryLayerMesh,
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
) -> tuple[
    domain_coverage_owner.PiecewiseLinearDomain,
    dict[tuple[int, ...], list[LayerCoreFace]],
]:
    """Describe outer boundary and every region interface of the combined domain."""
    layer_faces = faces(layers.mesh, layer_regions)
    cap_keys = {
        tuple(sorted(mapping[polygons[index]].tolist())): int(index) for index in cap_ids
    }
    cap_indices = frozenset(cap_ids.tolist())
    facets: list[tuple[int, int, int]] = []
    incidence: list[tuple[int, int]] = []
    for key, incidents in layer_faces.items():
        if key in cap_keys:
            if len(incidents) != 1:
                raise ValueError("A declared cap must be an exterior layer facet.")
            polygon = mapping[polygons[cap_keys[key]]]
            if cycle(incidents[0].vertices) != cycle(tuple(polygon.tolist())):
                raise ValueError(
                    "The cap orientation must agree with layer-cell incidence."
                )
            continue
        first = incidents[0]
        other_region = -1 if len(incidents) == 1 else incidents[1].region
        if first.region != other_region:
            for triangle in triangles(first.vertices):
                facets.append(triangle)
                incidence.append((first.region, other_region))
    if not set(cap_keys) <= set(layer_faces):
        raise ValueError("Every declared core cap must be present on the layer boundary.")
    for index, (polygon, facet) in enumerate(
        zip(polygons, complex_.polygon_facets, strict=True)
    ):
        positive, negative = (int(value) for value in complex_.facet_regions[facet])
        positive = int(core_region_map[positive]) if positive >= 0 else -1
        negative = int(core_region_map[negative]) if negative >= 0 else -1
        mapped = tuple(mapping[polygon].tolist())
        if index in cap_indices:
            negative = layer_faces[tuple(sorted(mapped))][0].region
        if positive != negative:
            facets.append(mapped)
            incidence.append((negative, positive))
    domain = domain_coverage_owner.PiecewiseLinearDomain(
        points,
        np.asarray(facets, dtype=np.int64),
        np.asarray(incidence, dtype=np.int64),
        region_ids,
        source_id=source_id,
    )
    return domain, layer_faces


__all__ = [
    "LayerCoreConstruction",
    "LayerCoreFace",
    "bitwise_equal_points",
    "combined_domain",
    "cycle",
    "faces",
    "polyhedral_cell_faces",
    "integer_array",
    "mesh_entity_scope",
    "prepare_identity",
    "triangles",
]
