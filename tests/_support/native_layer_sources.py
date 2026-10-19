"""Layer/core source fixtures with explicit cap incidence and material identity."""

from dataclasses import dataclass

import numpy as np

import phydrax as phx
from tests._support.source_authority import _cube_domain, _cube_query


M = phx.meshing


@dataclass(frozen=True)
class _Case:
    layers: M.BoundaryLayerMesh
    core: M.PiecewiseLinearComplex
    vertex_layer_ids: np.ndarray
    cap_polygon_ids: np.ndarray
    layer_regions: np.ndarray
    region_ids: tuple[str, ...]
    core_region_map: np.ndarray


def _case(*, material_interface: bool = False) -> _Case:
    wall = phx.discretization.CellMesh(
        np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "wall", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
            ),
        ),
    )
    entities = wall.entity_set(2)
    scope = M.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        entities.entity_set_id,
        entities.entity_ids,
    )
    control = M.BoundaryLayerControl(
        scope,
        M.LayerSchedule.geometric(2, 0.1, growth_rate=1.0),
        route=M.BoundaryLayerRoute.ADVANCING,
    )
    layers = M.prepare_boundary_layers(wall, control)
    if layers.cap is None:
        raise RuntimeError(
            "BoundaryLayerMesh.cap must be present for the square wall; got None."
        )
    cap = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in layers.cap.blocks]
    )
    cap_points = np.asarray(layers.cap.coordinates, dtype=np.float64)
    upper = cap_points.copy()
    upper[:, 2] = 1.0
    outer = np.asarray(
        (
            (4, 5, 6),
            (4, 6, 7),
            (0, 5, 4),
            (0, 1, 5),
            (1, 6, 5),
            (1, 2, 6),
            (2, 7, 6),
            (2, 3, 7),
            (3, 4, 7),
            (3, 0, 4),
        ),
        dtype=np.int64,
    )
    polygons = np.concatenate((cap, outer))
    core_region = 0
    region_ids = ("wall-material", "core-material") if material_interface else ("fluid",)
    complex_ = M.PiecewiseLinearComplex(
        np.concatenate((cap_points, upper)),
        tuple(polygons),
        np.concatenate(
            (
                np.zeros(cap.shape[0], dtype=np.int64),
                np.ones(outer.shape[0], dtype=np.int64),
            )
        ),
        np.asarray(((core_region, -1), (-1, core_region)), dtype=np.int64),
        (region_ids[-1],),
        boundary="fixed",
    )
    return _Case(
        layers,
        complex_,
        np.concatenate(
            (
                np.asarray(layers.cap_vertices, dtype=np.int64),
                np.full(4, -1, dtype=np.int64),
            )
        ),
        np.arange(cap.shape[0], dtype=np.int64),
        np.zeros(sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64),
        region_ids,
        np.asarray([1 if material_interface else 0], dtype=np.int64),
    )


def _result(case: _Case) -> M.CellMeshingResult:
    source = M.NativeLayerCoreSource(
        case.layers,
        case.core,
        "layered-square",
        "reference",
        vertex_layer_ids=case.vertex_layer_ids,
        cap_polygon_ids=case.cap_polygon_ids,
        layer_regions=case.layer_regions,
        region_ids=case.region_ids,
        core_region_map=case.core_region_map,
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "layered-core-facets",
        np.arange(case.core.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("prism", "tetrahedron"), allow_mixed=True)
        ),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(scope, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions("layer_core"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )


def _physical_layer_source(
    *,
    source_domain: phx.geometry.MeshingDomain | None = None,
) -> tuple[M.NativeLayerCoreSource, M.VolumeMeshingSpec]:
    """Return a source-face-bound wall and core for the six-plane cube."""
    case = _case()
    domain = _cube_domain()
    query = _cube_query(domain)
    boundary = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.arange(6, dtype=np.int64),
    )
    wall_scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray([0], dtype=np.int64),
    )
    volume = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        3,
        domain.entity_set_id(3),
        np.asarray([0], dtype=np.int64),
    )
    wall = case.layers.source_wall
    wall_ids = np.asarray(wall.entity_set(2).entity_ids, dtype=np.int64)
    association = M.GeometryAssociation(
        M.GeometryAssociationKind.SURFACE,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        wall_ids,
        (f"{domain.source_revision}:surface:0",) * wall_ids.size,
        np.zeros(wall_ids.size, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(wall_ids.size, 2, dtype=np.int8),
        source_indices=np.zeros(wall_ids.size, dtype=np.int64),
        orientations=-np.ones(wall_ids.size, dtype=np.int8),
    )
    control = M.BoundaryLayerControl(
        wall_scope,
        case.layers.control.schedule,
        route=M.BoundaryLayerRoute.ADVANCING,
        volume_scope=volume,
    )
    layers = M.prepare_boundary_layers(
        wall, control, wall_association=association, source_domain=domain
    )
    core = M.PiecewiseLinearComplex(
        case.core.vertices,
        tuple(case.core.polygon_vertices.reshape(-1, 3)),
        np.repeat(np.arange(6, dtype=np.int64), 2),
        np.asarray(((0, -1), *[(-1, 0)] * 5), dtype=np.int64),
        case.core.region_ids,
        boundary="fixed",
    )
    source = M.NativeLayerCoreSource(
        layers,
        core,
        domain.source_id,
        domain.source_revision,
        vertex_layer_ids=case.vertex_layer_ids,
        cap_polygon_ids=case.cap_polygon_ids,
        layer_regions=case.layer_regions,
        region_ids=case.region_ids,
        core_region_map=case.core_region_map,
        source_boundary_scope=boundary,
        source_domain=domain if source_domain is None else source_domain,
        fidelity_source=query,
        core_facet_source_ids=np.asarray((-1, 1, 2, 3, 4, 5), dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("prism", "tetrahedron"), allow_mixed=True)
        ),
        boundary,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(boundary, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
        layer_controls=(control,),
    )
    return source, specification
