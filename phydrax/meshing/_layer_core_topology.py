#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Layer/core incidence and identity preparation shared by composition phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final, TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._exact_plc_geometry as exact_plc_geometry_owner
import phydrax.discretization._nested_reference_types as nested_reference_owner
import phydrax.discretization._periodic_topology as periodic_topology_owner
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._mapped_reference_domain as mapped_domain_owner
import phydrax.geometry._meshing_domain as meshing_domain_owner
import phydrax.meshing._association as association_owner
import phydrax.meshing._boundary_layer as boundary_layer_owner
import phydrax.meshing._layer_core_subdivision as subdivision_owner
import phydrax.meshing._organization as organization_owner
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._polyhedral_generation as polyhedral_generation_owner
import phydrax.meshing._volume_generation as volume_generation_owner

from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, reference_cell_topology
from ..typing import checked
from ._scope import MeshingEntityKind, MeshingScope


if TYPE_CHECKING:
    from ..geometry._surface_source_support import SurfaceSourceCharts
    from ._controls import BoundaryLayerControl
    from .providers._native_sources import NativeLayerCoreSource


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
def polyhedral_cell_faces(mesh: CellMesh, /) -> tuple[tuple[tuple[int, ...], ...], ...]:
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
        for occurrence in range(int(cell_offsets[cell]), int(cell_offsets[cell + 1])):
            face = int(cell_faces[occurrence])
            orientation = int(cell_signs[occurrence])
            if orientation not in (-1, 1):
                raise ValueError(
                    "Polyhedral face-use orientation must be signed unit data."
                )
            loop = face_vertices[int(face_offsets[face]) : int(face_offsets[face + 1])]
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
        connectivity_cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        cell_faces_by_row = polyhedral_cell_faces(mesh)
        for cell_row, cell_id in enumerate(connectivity_cell_ids.tolist()):
            region_row = int(np.searchsorted(identifiers, cell_id))
            if region_row == identifiers.size or identifiers[region_row] != cell_id:
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
    overlap_exact_geometry: exact_plc_geometry_owner.ExactPlcCellGeometrySource | None = (
        None
    )
    overlap_source_stencils: subdivision_owner.OverlapSourceStencils | None = None
    layer_core_column_certification: (
        subdivision_owner.RefinedColumnCertification | None
    ) = None
    layer_root_partition: nested_reference_owner.ReferenceCellPartition | None = None
    execution_carrier: object | None = None


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
    region_occurrence_paths: tuple[tuple[str, ...], ...],
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
        region_occurrence_paths=region_occurrence_paths,
    )
    return domain, layer_faces


def declared_charts(
    domain: domain_coverage_owner.PiecewiseLinearDomain,
    revision: str,
    /,
    *,
    corner_keys: tuple[tuple[int, tuple[str, ...]], ...] | None = None,
    facet_keys: tuple[tuple[int, tuple[str, ...]], ...] | None = None,
    region_keys: tuple[tuple[int, tuple[str, ...]], ...] | None = None,
) -> tuple[
    meshing_domain_owner.MeshingDomain, domain_boundary_owner.MeshingDomainBoundarySource
]:
    """Bind a staged layer and declared core boundary to oriented planar charts."""
    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
    from phydrax.geometry._meshing_domain import (
        MeshingDomain,
        MeshingDomainCurve,
        MeshingDomainRegion,
        MeshingSurfacePatch,
        PatchCurveUse,
    )
    from phydrax.geometry.brep._patches import LineCurve
    from phydrax.geometry.reconstruction._core import _ReconstructedTrianglePatch

    points = np.asarray(domain.vertices, dtype=np.float64)
    facets = np.asarray(domain.facets, dtype=np.int64)
    sides = np.asarray(domain.facet_regions, dtype=np.int64)
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    edge_rows: dict[tuple[int, int], int] = {}
    curves: list[MeshingDomainCurve] = []
    patches: list[MeshingSurfacePatch] = []
    for facet in facets:
        uses = []
        for local in range(3):
            first, last = int(facet[local]), int(facet[(local + 1) % 3])
            edge = (min(first, last), max(first, last))
            if edge not in edge_rows:
                edge_rows[edge] = len(curves)
                curves.append(MeshingDomainCurve(*edge))
            forward = (first, last) == edge
            start = uv[local] if forward else uv[(local + 1) % 3]
            end = uv[(local + 1) % 3] if forward else uv[local]
            uses.append(
                PatchCurveUse(
                    edge_rows[edge],
                    LineCurve(start, end - start),
                    0.0,
                    1.0,
                    sense=1 if forward else -1,
                )
            )
        corners = points[facet]
        patches.append(
            MeshingSurfacePatch(
                _ReconstructedTrianglePatch(corners),
                (tuple(uses),),
            )
        )
    regions = tuple(
        MeshingDomainRegion(
            name,
            tuple(
                (row, 1 if int(sides[row, 0]) == region else -1)
                for row in range(sides.shape[0])
                if region == sides[row, 0] or region == sides[row, 1]
            ),
        )
        for region, name in enumerate(domain.region_ids)
    )
    source_indices = None
    source_occurrences = None
    region_source_indices = None
    region_source_occurrences = None
    if corner_keys is not None or facet_keys is not None or region_keys is not None:
        if corner_keys is None or facet_keys is None or region_keys is None:
            raise ValueError(
                "Declared chart identities must cover corners, facets, and regions."
            )
        if len(corner_keys) != len(points) or len(facet_keys) != len(facets):
            raise ValueError(
                "Declared chart identities must align with their exact source rows."
            )
        if len(region_keys) != len(domain.region_ids):
            raise ValueError(
                "Declared region identities must align with the source regions."
            )
        edge_keys = []
        for first, last in edge_rows:
            a, b = corner_keys[first], corner_keys[last]
            if a[1] != b[1]:
                raise ValueError(
                    "A source edge cannot join distinct boundary occurrences."
                )
            endpoints = tuple(sorted((a, b)))
            identity = int(
                canonical_fingerprint(
                    {
                        "kind": "layer-core-source-edge",
                        "source": (domain.source_id, revision),
                        "endpoints": endpoints,
                    }
                )[:16],
                16,
            ) & ((1 << 63) - 1)
            edge_keys.append((identity, a[1]))
        source_indices = tuple(
            tuple(key[0] for key in level)
            for level in (corner_keys, tuple(edge_keys), facet_keys)
        )
        source_occurrences = tuple(
            tuple(key[1] for key in level)
            for level in (corner_keys, tuple(edge_keys), facet_keys)
        )
        region_source_indices = tuple(key[0] for key in region_keys)
        region_source_occurrences = tuple(key[1] for key in region_keys)
    source = MeshingDomain(
        tuple(patches),
        tuple(curves),
        len(points),
        source_id=domain.source_id,
        source_revision=revision,
        regions=regions,
        source_indices=source_indices,
        source_occurrences=source_occurrences,
        region_source_indices=region_source_indices,
        region_source_occurrences=region_source_occurrences,
        corner_points=points,
    )
    exterior = tuple(row for row, pair in enumerate(sides) if np.any(pair < 0))
    return source, MeshingDomainBoundarySource(source, exterior)


def _closed_boundary_identity_keys(
    layers: boundary_layer_owner.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    vertex_map: np.ndarray,
    cap_ids: np.ndarray,
    mapping: np.ndarray,
    polygons: np.ndarray,
    domain: domain_coverage_owner.PiecewiseLinearDomain,
    wall_path: tuple[str, ...],
    outer_path: tuple[str, ...],
    /,
) -> tuple[
    tuple[tuple[int, tuple[str, ...]], ...],
    tuple[tuple[int, tuple[str, ...]], ...],
]:
    """Bind each combined corner and facet to its actual wall or PLC entity."""
    wall = layers.source_wall
    if not isinstance(wall, CellMesh):
        raise TypeError("Closed core source requires its actual wall mesh.")
    layer_global = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    wall_vertices = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    wall_faces = {
        tuple(sorted(int(wall_vertices[int(vertex)]) for vertex in row)): int(identifier)
        for block in wall.blocks
        for row, identifier in zip(
            np.asarray(block.vertices), np.asarray(block.global_ids), strict=True
        )
    }
    if len(wall_faces) != sum(block.cell_count for block in wall.blocks):
        raise ValueError("Wall faces must retain distinct authored cell identities.")
    nonshared = np.flatnonzero(vertex_map < 0)
    outer_vertex_ids = plc_source_owner._numeric_entity_ids(
        "layer-core-outer-vertex",
        tuple(complex_.vertex_ids[index] for index in nonshared),
        tuple(complex_.vertex_occurrence_paths[index] for index in nonshared),
    )
    corner_keys = tuple((int(value), wall_path) for value in layer_global) + tuple(
        (value, outer_path) for value in outer_vertex_ids
    )
    cap_set = frozenset(int(value) for value in cap_ids)
    outer_polygons = [index for index in range(len(polygons)) if index not in cap_set]
    facet_rows = np.asarray(complex_.polygon_facets, dtype=np.int64)
    outer_facet_ids = plc_source_owner._numeric_entity_ids(
        "layer-core-outer-facet",
        tuple(complex_.facet_ids[int(facet_rows[index])] for index in outer_polygons),
        tuple(
            complex_.facet_occurrence_paths[int(facet_rows[index])]
            for index in outer_polygons
        ),
    )
    outer_faces = {
        tuple(sorted(int(value) for value in mapping[polygons[index]])): identity
        for index, identity in zip(outer_polygons, outer_facet_ids, strict=True)
    }
    if len(outer_faces) != len(outer_polygons):
        raise ValueError("Outer facets must retain distinct authored cell identities.")
    layer_sources = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)
    facet_keys = []
    for facet in np.asarray(domain.facets, dtype=np.int64):
        key = tuple(sorted(int(value) for value in facet))
        if key in outer_faces:
            facet_keys.append((outer_faces[key], outer_path))
        elif np.all(facet < len(layer_global)):
            wall_key = tuple(sorted(int(value) for value in layer_sources[facet]))
            if wall_key not in wall_faces:
                raise ValueError(
                    "A layer boundary facet lost its original wall ancestry."
                )
            facet_keys.append((wall_faces[wall_key], wall_path))
        else:
            raise ValueError(
                "A combined boundary facet has no authored source occurrence."
            )
    return corner_keys, tuple(facet_keys)


def _closed_boundary_charts(
    combined: domain_coverage_owner.PiecewiseLinearDomain,
    domain: meshing_domain_owner.MeshingDomain,
    query: domain_boundary_owner.MeshingDomainBoundarySource,
    contract: SpatialCoordinateContract,
    /,
) -> SurfaceSourceCharts:
    """Retain the actual complete exterior as source-bound affine triangle charts."""
    from ..geometry._surface_source_support import SurfaceSourceCharts

    patches = np.asarray(query.patches, dtype=np.int64)
    facets = np.asarray(combined.facets, dtype=np.int64)[patches]
    used = np.unique(facets)
    inverse = np.full(combined.vertices.shape[0], -1, dtype=np.int64)
    inverse[used] = np.arange(used.size, dtype=np.int64)
    scope_ids = tuple(
        {
            domain.source_entity_key_for_scope(dimension, int(identifier)): int(
                identifier
            )
            for identifier in domain.scope_entity_ids(dimension)
        }
        for dimension in (0, 2)
    )
    vertex_ids = np.asarray(
        [
            scope_ids[0][
                (domain.entity_id(0, int(row)), domain.source_occurrences[0][int(row)])
            ]
            for row in used
        ],
        dtype=np.int64,
    )
    root_ids = np.asarray(
        [
            scope_ids[1][
                (domain.entity_id(2, int(row)), domain.source_occurrences[2][int(row)])
            ]
            for row in patches
        ],
        dtype=np.int64,
    )
    mesh = CellMesh.from_triangles(
        np.asarray(combined.vertices, dtype=np.float64)[used],
        inverse[facets],
        vertex_global_ids=vertex_ids,
        cell_global_ids=root_ids,
    )
    row_by_id = {
        int(identifier): int(row)
        for identifier, row in zip(root_ids, patches, strict=True)
    }
    target_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    target_patches = tuple(row_by_id[int(identifier)] for identifier in target_ids)
    association = association_owner.GeometryAssociation(
        association_owner.GeometryAssociationKind.SURFACE,
        domain.source_id,
        domain.source_revision,
        mesh.entity_set(2).entity_set_id,
        target_ids,
        tuple(domain.entity_id(2, row) for row in target_patches),
        np.zeros(target_ids.size, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(target_ids.size, 2, dtype=np.int8),
        source_indices=np.asarray(
            [domain.source_indices[2][row] for row in target_patches], dtype=np.int64
        ),
        source_roles=(association_owner.GeometrySourceEntityRole.FACET,)
        * target_ids.size,
        source_occurrence_paths=tuple(
            domain.source_occurrences[2][row] for row in target_patches
        ),
    )
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    return SurfaceSourceCharts(
        domain,
        mesh,
        cell_geometry_owner.CellGeometrySpec.affine(mesh),
        contract,
        np.broadcast_to(uv, (patches.size, 3, 2)),
        patches,
        (association,),
        query,
        source_root_cell_ids=root_ids,
        source_piece_ids=tuple(
            (int(row), int(domain.source_indices[2][int(row)])) for row in patches
        ),
    )


def _closed_wall_charts(
    layers: boundary_layer_owner.BoundaryLayerMesh,
    combined: domain_coverage_owner.PiecewiseLinearDomain,
    domain: meshing_domain_owner.MeshingDomain,
    query: domain_boundary_owner.MeshingDomainBoundarySource,
    contract: SpatialCoordinateContract,
    /,
) -> SurfaceSourceCharts:
    """Bind the original wall cells, including their orientation, to exterior charts."""
    from ..geometry._surface_source_support import SurfaceSourceCharts

    wall = layers.source_wall
    if not isinstance(wall, CellMesh):
        raise ValueError("A closed core requires its original wall.")
    layer_sources = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)
    wall_vertices = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    wall_points = np.asarray(wall.coordinates, dtype=np.float64)
    wall_point_by_id = {
        int(identifier): wall_points[row] for row, identifier in enumerate(wall_vertices)
    }
    facets = np.asarray(combined.facets, dtype=np.int64)
    patches = {}
    for row in query.patches:
        facet = facets[int(row)]
        if np.any(facet >= len(layer_sources)):
            continue
        identifiers = tuple(int(layer_sources[int(vertex)]) for vertex in facet)
        if any(
            identifier not in wall_point_by_id
            or not np.array_equal(
                layer_points[int(vertex)].view(np.uint64),
                wall_point_by_id[identifier].view(np.uint64),
            )
            for vertex, identifier in zip(facet, identifiers, strict=True)
        ):
            continue
        key = tuple(sorted(identifiers))
        if key in patches:
            raise ValueError("Wall source facets must have unique vertex incidence.")
        patches[key] = (int(row), identifiers)
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    root_patches = []
    parameters = []
    cell_ids = []
    for block in wall.blocks:
        for vertices, identifier in zip(block.vertices, block.global_ids, strict=True):
            ordered = tuple(int(wall_vertices[int(vertex)]) for vertex in vertices)
            match = patches.get(tuple(sorted(ordered)))
            if match is None:
                raise ValueError(
                    "An original wall cell has no exact exterior source chart."
                )
            row, source_vertices = match
            root_patches.append(row)
            parameters.append(uv[[source_vertices.index(vertex) for vertex in ordered]])
            cell_ids.append(int(identifier))
    if len(root_patches) != len(patches):
        raise ValueError("The wall charts do not cover the exact original wall.")
    rows_by_id = dict(zip(cell_ids, root_patches, strict=True))
    target_ids = np.asarray(wall.entity_set(2).entity_ids, dtype=np.int64)
    target_patches = tuple(rows_by_id[int(identifier)] for identifier in target_ids)
    association = association_owner.GeometryAssociation(
        association_owner.GeometryAssociationKind.SURFACE,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        target_ids,
        tuple(domain.entity_id(2, row) for row in target_patches),
        np.zeros(target_ids.size, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(target_ids.size, 2, dtype=np.int8),
        source_indices=np.asarray(
            [domain.source_indices[2][row] for row in target_patches], dtype=np.int64
        ),
        source_roles=(association_owner.GeometrySourceEntityRole.FACET,)
        * target_ids.size,
        source_occurrence_paths=tuple(
            domain.source_occurrences[2][row] for row in target_patches
        ),
    )
    scope_ids = {
        domain.source_entity_key_for_scope(2, int(identifier)): int(identifier)
        for identifier in domain.scope_entity_ids(2)
    }
    return SurfaceSourceCharts(
        domain,
        wall,
        cell_geometry_owner.CellGeometrySpec.affine(wall),
        contract,
        np.asarray(parameters),
        np.asarray(root_patches, dtype=np.int64),
        (association,),
        query,
        source_root_cell_ids=np.asarray(
            [
                scope_ids[(domain.entity_id(2, row), domain.source_occurrences[2][row])]
                for row in root_patches
            ],
            dtype=np.int64,
        ),
        source_piece_ids=tuple(
            (row, int(domain.source_indices[2][row])) for row in root_patches
        ),
    )


def closed_fixed_core_source(
    layers: boundary_layer_owner.BoundaryLayerMesh,
    outer: CellMesh | None,
    coordinate_contract: SpatialCoordinateContract,
    source_id: str,
    revision: str,
    /,
    *,
    region_id: str,
    region_entity_id: int,
    region_occurrence_path: tuple[str, ...],
    wall_occurrence_path: tuple[str, ...],
    outer_occurrence_path: tuple[str, ...] | None,
    layer_control: BoundaryLayerControl,
    layer_policy: boundary_layer_owner.BoundaryLayerPolicy,
) -> NativeLayerCoreSource:
    """Assemble a closed cap, optionally with an independent outer shell."""
    from .providers._native_sources import NativeLayerCoreSource

    if outer is None:
        if outer_occurrence_path is not None or not isinstance(
            wall_occurrence_path, tuple
        ):
            raise ValueError(
                "A cap-only core requires its original wall occurrence and no outer occurrence."
            )
    elif (
        not wall_occurrence_path
        or not outer_occurrence_path
        or wall_occurrence_path == outer_occurrence_path
    ):
        raise ValueError(
            "Wall and outer boundary require distinct explicit occurrence paths."
        )
    cap = layers.cap
    wall = layers.source_wall
    if cap is None or not layers.closed_cap or not isinstance(wall, CellMesh):
        raise ValueError(
            "A closed fixed core requires the actual closed accepted layer cap."
        )
    if outer is None and (
        (source_id, revision) != (str(wall.mesh_id), str(wall.numeric_version))
        or wall_occurrence_path != ()
        or region_occurrence_path != ()
    ):
        raise ValueError(
            "A cap-only core must retain its original wall and region identities."
        )
    for mesh in (wall, cap) if outer is None else (wall, cap, outer):
        if (
            mesh.topological_dimension != 2
            or mesh.ambient_dimension != 3
            or any(block.cell_kind != "triangle" for block in mesh.blocks)
        ):
            raise ValueError(
                "Closed layer/core boundaries require triangular ambient-3 meshes."
            )
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks]
    )
    cap_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in cap.blocks]
    )
    outer_rows = (
        np.empty((0, 3), dtype=np.int64)
        if outer is None
        else np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in outer.blocks]
        )
    )
    outer_ids = (
        np.empty(0, dtype=np.int64)
        if outer is None
        else np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in outer.blocks]
        )
    )
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    cap_layer = np.asarray(layers.cap_vertices, dtype=np.int64)
    cap_count = cap_layer.size
    outer_count = 0 if outer is None else outer.coordinates.shape[0]
    points = (
        layer_points[cap_layer]
        if outer is None
        else np.concatenate(
            (layer_points[cap_layer], np.asarray(outer.coordinates, dtype=np.float64))
        )
    )
    polygons = np.concatenate((cap_rows, outer_rows + cap_count))
    incidence = np.concatenate(
        (
            np.tile(np.asarray((0, -1), dtype=np.int64), (len(cap_rows), 1)),
            np.tile(np.asarray((-1, 0), dtype=np.int64), (len(outer_rows), 1)),
        )
    )
    layer_global = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    outer_global = (
        np.empty(0, dtype=np.int64)
        if outer is None
        else np.asarray(outer.vertex_global_ids, dtype=np.int64)
    )
    complex_ = plc_source_owner.PiecewiseLinearComplex(
        points,
        tuple(polygons),
        np.arange(len(polygons), dtype=np.int64),
        incidence,
        (region_id,),
        vertex_ids=(
            tuple(
                f"{source_id}:cap-vertex:{int(value)}" for value in cap.vertex_global_ids
            )
            if outer is None
            else tuple(f"cap-vertex:{int(value)}" for value in layer_global[cap_layer])
            + tuple(f"outer-vertex:{int(value)}" for value in outer_global)
        ),
        vertex_occurrence_paths=(wall_occurrence_path,) * cap_count
        + (
            ()
            if outer_occurrence_path is None
            else (outer_occurrence_path,) * outer_count
        ),
        facet_ids=(
            tuple(f"{source_id}:cap-facet:{row}" for row in range(len(cap_rows)))
            if outer is None
            else tuple(f"cap-cell:{int(value)}" for value in cap_ids)
            + tuple(f"outer-cell:{int(value)}" for value in outer_ids)
        ),
        facet_occurrence_paths=(wall_occurrence_path,) * len(cap_rows)
        + (
            ()
            if outer_occurrence_path is None
            else (outer_occurrence_path,) * len(outer_rows)
        ),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=(region_occurrence_path,),
        boundary="fixed",
    )
    if outer is None and region_entity_id != complex_.region_entity_ids[0]:
        raise ValueError("A cap-only core must retain its original PLC region identity.")
    vertex_map = np.concatenate((cap_layer, np.full(outer_count, -1, dtype=np.int64)))
    cap_polygons = np.arange(len(cap_rows), dtype=np.int64)
    layer_regions = np.zeros(
        sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
    )
    mapping, cap_polygons, polygons, combined_points = prepare_identity(
        layers, complex_, vertex_map, cap_polygons
    )
    domain, _ = combined_domain(
        layers,
        complex_,
        layer_regions,
        mapping,
        cap_polygons,
        polygons,
        combined_points,
        source_id,
        (region_id,),
        np.asarray((0,), dtype=np.int64),
        (region_occurrence_path,),
    )
    corner_keys, facet_keys = _closed_boundary_identity_keys(
        layers,
        complex_,
        vertex_map,
        cap_polygons,
        mapping,
        polygons,
        domain,
        wall_occurrence_path,
        () if outer_occurrence_path is None else outer_occurrence_path,
    )
    source_domain, query = declared_charts(
        domain,
        revision,
        corner_keys=corner_keys,
        facet_keys=tuple(facet_keys),
        region_keys=((region_entity_id, region_occurrence_path),),
    )
    boundary_charts = _closed_boundary_charts(
        domain, source_domain, query, coordinate_contract
    )
    wall_charts = (
        _closed_wall_charts(layers, domain, source_domain, query, coordinate_contract)
        if outer is None
        else None
    )
    domain_facets = np.asarray(domain.facets, dtype=np.int64)
    rows_by_vertices = {
        tuple(
            sorted(
                int(domain_facets[row, column])
                for column in range(domain_facets.shape[1])
            )
        ): row
        for row in range(domain_facets.shape[0])
    }
    source_ids: list[str | None] = [None] * len(cap_rows)
    source_paths: list[tuple[str, ...] | None] = [None] * len(cap_rows)
    for polygon in polygons[len(cap_rows) :]:
        key = tuple(sorted(int(value) for value in mapping[polygon]))
        row = rows_by_vertices.get(key)
        if row is None or cycle(tuple(int(value) for value in mapping[polygon])) != cycle(
            tuple(
                int(domain_facets[row, column])
                for column in range(domain_facets.shape[1])
            )
        ):
            raise ValueError("An outer core facet changed its exact source orientation.")
        source_ids.append(source_domain.entity_id(2, row))
        source_paths.append(tuple(source_domain.source_occurrences[2][row]))
    source = NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        revision,
        vertex_layer_ids=vertex_map,
        cap_polygon_ids=cap_polygons,
        layer_regions=layer_regions,
        region_entity_ids=np.asarray((region_entity_id,), dtype=np.int64),
        region_occurrence_paths=(region_occurrence_path,),
        region_ids=(region_id,),
        core_region_map=np.asarray((0,), dtype=np.int64),
        source_domain=source_domain,
        fidelity_source=query,
        source_wall_charts=wall_charts,
        source_boundary_charts=boundary_charts,
        maximum_deviation=0.0,
        layer_control=layer_control,
        layer_policy=layer_policy,
        core_facet_source_ids=tuple(source_ids),
        core_facet_source_occurrence_paths=tuple(source_paths),
    )
    validate_closed_boundary_source(source)
    return source


def validate_staged_layer_domain(
    source: object, /
) -> domain_coverage_owner.PiecewiseLinearDomain:
    """Prove a triangular declared source matches the complete layer/core boundary."""
    from ..geometry._domain_boundary import MeshingDomainBoundarySource
    from ..geometry._mesh_certificates import _same_tree_content
    from ..geometry._meshing_domain import (
        MeshingDomain,
        MeshingDomainCurve,
        MeshingSurfacePatch,
        PatchCurveUse,
    )
    from ..geometry.brep._patches import LineCurve
    from ..geometry.reconstruction._core import _ReconstructedTrianglePatch
    from .providers._native_sources import NativeLayerCoreSource

    if type(source) is not NativeLayerCoreSource:
        raise TypeError("Expected a native layer/core source.")
    domain = source.source_domain
    query = source.fidelity_source
    if (
        type(domain) is not MeshingDomain
        or type(query) is not MeshingDomainBoundarySource
    ):
        raise TypeError(
            "Staged layer source requires its exact domain and boundary query."
        )
    if domain.domain_id != query.domain.domain_id:
        raise ValueError("Staged boundary query belongs to a different source domain.")
    domain.require_current(source.source_id, source.source_revision)
    query.require_current()
    layers, complex_ = source.layers, source.complex
    mapping, cap, polygons, points = prepare_identity(
        layers, complex_, source.vertex_layer_ids, source.cap_polygon_ids
    )
    combined, _ = combined_domain(
        layers,
        complex_,
        source.layer_regions,
        mapping,
        cap,
        polygons,
        points,
        source.source_id,
        source.region_ids,
        source.core_region_map,
        source.region_occurrence_paths,
    )
    vertices = np.asarray(combined.vertices, dtype=np.float64)
    facets = np.asarray(combined.facets, dtype=np.int64)
    sides = np.asarray(combined.facet_regions, dtype=np.int64)
    if (
        not bitwise_equal_points(
            vertices, np.asarray(domain.corner_points, dtype=np.float64)
        )
        or len(domain.patches) != len(facets)
        or tuple(region.name for region in domain.regions) != source.region_ids
        or tuple(tuple(path) for path in domain.region_source_occurrences)
        != source.region_occurrence_paths
    ):
        raise ValueError("Declared source does not cover the combined layer/core domain.")
    edge_rows: dict[tuple[int, int], int] = {}
    curves = []
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    for row, facet in enumerate(facets.tolist()):
        uses = []
        for local in range(3):
            first, last = int(facet[local]), int(facet[(local + 1) % 3])
            edge = (min(first, last), max(first, last))
            if edge not in edge_rows:
                edge_rows[edge] = len(curves)
                curves.append(MeshingDomainCurve(*edge))
            forward = (first, last) == edge
            start = uv[local] if forward else uv[(local + 1) % 3]
            end = uv[(local + 1) % 3] if forward else uv[local]
            uses.append(
                PatchCurveUse(
                    edge_rows[edge],
                    LineCurve(start, end - start),
                    0.0,
                    1.0,
                    sense=1 if forward else -1,
                )
            )
        corners = vertices[facet]
        expected = MeshingSurfacePatch(
            _ReconstructedTrianglePatch(corners),
            (tuple(uses),),
        )
        if not _same_tree_content(domain.patches[row], expected):
            raise ValueError(
                "Declared source patch differs from its oriented exact facet."
            )
    if not _same_tree_content(domain.curves, tuple(curves)):
        raise ValueError("Declared source curves do not cover the combined facets.")
    for region, declared in enumerate(domain.regions):
        boundary = tuple(
            (row, 1 if int(pair[0]) == region else -1)
            for row, pair in enumerate(sides.tolist())
            if region in pair
        )
        if declared.boundary != boundary:
            raise ValueError(
                "Declared source material sides differ from the combined domain."
            )
    exterior = tuple(row for row, pair in enumerate(sides) if np.any(pair < 0))
    if query.patches != exterior:
        raise ValueError("Source boundary query does not cover the exact exterior.")
    by_vertices = {
        tuple(sorted(int(vertex) for vertex in facet)): row
        for row, facet in enumerate(facets.tolist())
    }
    if len(by_vertices) != len(facets):
        raise ValueError("Combined domain repeats a source facet.")
    cap_indices = frozenset(int(value) for value in cap)
    hidden_cap = {
        index
        for index in cap_indices
        if tuple(sorted(int(value) for value in mapping[polygons[index]]))
        not in by_vertices
    }
    if hidden_cap:
        if hidden_cap != cap_indices or len(source.region_ids) != 1:
            raise ValueError(
                "An unexposed cap must join one complete same-material core."
            )
        facet_rows = np.asarray(complex_.polygon_facets, dtype=np.int64)
        cap_paths = {
            complex_.facet_occurrence_paths[int(facet_rows[index])]
            for index in cap_indices
        }
        outer_paths = {
            complex_.facet_occurrence_paths[int(facet_rows[index])]
            for index in range(len(polygons))
            if index not in cap_indices
        }
        if len(cap_paths) != 1:
            raise ValueError("A closed cap requires one wall occurrence.")
        wall_path = next(iter(cap_paths))
        if len(cap_indices) == len(polygons):
            if wall_path != () or outer_paths:
                raise ValueError(
                    "A cap-only core must retain its original wall occurrence."
                )
            outer_path = ()
        else:
            if len(outer_paths) != 1:
                raise ValueError(
                    "Closed cap and outer boundary require one occurrence each."
                )
            outer_path = next(iter(outer_paths))
            if not wall_path or not outer_path or wall_path == outer_path:
                raise ValueError(
                    "Closed wall and outer boundary occurrences must be distinct."
                )
        vertex_map = np.asarray(source.vertex_layer_ids, dtype=np.int64)
        if any(
            path != (wall_path if int(owner) >= 0 else outer_path)
            for path, owner in zip(
                complex_.vertex_occurrence_paths, vertex_map, strict=True
            )
        ):
            raise ValueError("Closed core vertex occurrences differ from actual owners.")
        corner_keys, facet_keys = _closed_boundary_identity_keys(
            layers,
            complex_,
            vertex_map,
            cap,
            mapping,
            polygons,
            combined,
            wall_path,
            outer_path,
        )
        expected_domain, expected_query = declared_charts(
            combined,
            source.source_revision,
            corner_keys=corner_keys,
            facet_keys=facet_keys,
            region_keys=(
                (int(source.region_entity_ids[0]), source.region_occurrence_paths[0]),
            ),
        )
        if not _same_tree_content(domain, expected_domain):
            raise ValueError(
                "Closed source chart identities differ from their actual boundary owners."
            )
        charts = source.source_boundary_charts
        if charts is None:
            raise ValueError("Closed source requires its complete exterior chart owner.")
        restored_contract = SpatialCoordinateContract.from_dict(
            charts.coordinate_contract.to_dict()
        )
        if not _same_tree_content(restored_contract, charts.coordinate_contract):
            raise ValueError("Closed source coordinate contract differs from its owner.")
        expected_charts = _closed_boundary_charts(
            combined, expected_domain, expected_query, restored_contract
        )
        if not _same_tree_content(charts, expected_charts):
            raise ValueError(
                "Closed source charts differ from their actual boundary owners."
            )
    for index, (polygon, facet) in enumerate(
        zip(polygons.tolist(), complex_.polygon_facets, strict=True)
    ):
        if index in hidden_cap:
            if (
                source.core_facet_source_ids[int(facet)] is not None
                or source.core_facet_source_occurrence_paths[int(facet)] is not None
            ):
                raise ValueError("An internal cap cannot claim a source boundary facet.")
            continue
        actual = tuple(int(vertex) for vertex in mapping[polygon])
        row = by_vertices.get(tuple(sorted(actual)))
        if row is None or cycle(actual) != cycle(tuple(int(v) for v in facets[row])):
            raise ValueError("Core facet does not retain its oriented source facet.")
        if source.core_facet_source_ids[int(facet)] != domain.entity_id(
            2, row
        ) or source.core_facet_source_occurrence_paths[int(facet)] != tuple(
            domain.source_occurrences[2][row]
        ):
            raise ValueError("Core facet source identity or occurrence changed.")
    return combined


def validate_closed_boundary_source(source: object, /) -> None:
    """Reprove a boundary-only atlas of one actual closed, same-material core."""
    from ..geometry._mesh_certificates import _same_tree_content
    from .providers._native_sources import NativeLayerCoreSource

    if type(source) is not NativeLayerCoreSource:
        raise TypeError("Expected a native layer/core source.")
    combined = validate_staged_layer_domain(source)
    cap = np.asarray(source.cap_polygon_ids, dtype=np.int64)
    facet_rows = np.asarray(source.complex.polygon_facets, dtype=np.int64)
    vertex_map = np.asarray(source.vertex_layer_ids, dtype=np.int64)
    cap_only = cap.size == len(source.complex.polygon_facets) and np.all(vertex_map >= 0)
    if (
        not source.layers.closed_cap
        or cap.size == 0
        or len(source.region_ids) != 1
        or source.source_boundary_charts is None
        or (source.source_wall_charts is None) == cap_only
        or source.source_boundary_scope is not None
        or source.source_boundary_schedule is not None
        or source.conforming_execution
        or source.layer_control is None
        or source.layer_policy is None
        or source.layers.wall_association is not None
        or source.source_wall_projection is not None
        or source.source_wall_association_policy is not None
        or source.mapped_domain is not None
        or any(
            source.core_facet_source_ids[int(facet_rows[index])] is not None
            for index in cap
        )
        or any(
            source.core_facet_source_occurrence_paths[int(facet_rows[index])] is not None
            for index in cap
        )
    ):
        raise ValueError(
            "Boundary-only charts require their actual closed same-material source."
        )
    cap_mesh = source.layers.cap
    if cap_mesh is None:
        raise ValueError("Closed source lost its accepted cap mesh.")
    cap_global_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in cap_mesh.blocks]
    )
    if cap_global_ids.shape != cap.shape:
        raise ValueError(
            "Closed source cap facet identities differ from the accepted cap."
        )
    if cap_only:
        wall = source.layers.source_wall
        if (
            not isinstance(wall, CellMesh)
            or (source.source_id, source.source_revision)
            != (str(wall.mesh_id), str(wall.numeric_version))
            or source.region_occurrence_paths != ((),)
            or source.region_entity_ids[0] != source.complex.region_entity_ids[0]
            or any(path != () for path in source.complex.facet_occurrence_paths)
            or any(path != () for path in source.complex.vertex_occurrence_paths)
            or any(
                source.complex.facet_ids[int(facet_rows[int(polygon)])]
                != f"{source.source_id}:cap-facet:{row}"
                for row, polygon in enumerate(cap)
            )
            or source.complex.vertex_ids
            != tuple(
                f"{source.source_id}:cap-vertex:{int(value)}"
                for value in cap_mesh.vertex_global_ids
            )
        ):
            raise ValueError(
                "Cap-only source identities differ from their original wall and cap."
            )
        wall_charts = source.source_wall_charts
        source_domain = source.source_domain
        query = source.fidelity_source
        if (
            wall_charts is None
            or not isinstance(source_domain, meshing_domain_owner.MeshingDomain)
            or not isinstance(query, domain_boundary_owner.MeshingDomainBoundarySource)
        ):
            raise ValueError("Cap-only source lost its original wall chart authority.")
        wall_charts.require_root(wall, wall_charts.geometry)
        expected = _closed_wall_charts(
            source.layers,
            combined,
            source_domain,
            query,
            source.source_boundary_charts.coordinate_contract,
        )
        if not _same_tree_content(wall_charts, expected):
            raise ValueError(
                "Cap-only wall charts differ from the exact original wall authority."
            )
    else:
        if any(
            source.complex.facet_ids[int(facet_rows[int(polygon)])]
            != f"cap-cell:{int(identifier)}"
            for polygon, identifier in zip(cap, cap_global_ids, strict=True)
        ):
            raise ValueError(
                "Closed source cap facet identities differ from the accepted cap."
            )
        layer_ids = np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64)
        if any(
            source.complex.vertex_ids[index] != f"cap-vertex:{int(layer_ids[int(owner)])}"
            for index, owner in enumerate(vertex_map)
            if owner >= 0
        ):
            raise ValueError(
                "Closed source cap vertices differ from their accepted layer identities."
            )


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
    "validate_staged_layer_domain",
    "triangles",
]
