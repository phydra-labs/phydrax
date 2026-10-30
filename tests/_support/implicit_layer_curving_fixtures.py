#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitSurfacePolicy,
    discover_adaptive_implicit_surface,
)
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._curving import HighOrderCurvingPolicy
from phydrax.meshing._layer_curving import (
    curve_layer_core_construction,
    LayerCurvingConstructionResult,
    LayerCurvingStatus,
)
from phydrax.meshing.providers._native_implicit import (
    _implicit_surface_chart_witness,
    _NativeImplicitBoundaryQuery,
)
from phydrax.meshing.providers._native_sources import NativeImplicitSource

from .layer_curving_fixtures import _layer_mesh, _mixed_construction


def make_implicit_layer_curving_archive_case() -> LayerCurvingConstructionResult:
    """Build accepted high-order layer evidence from an adaptive smooth implicit source."""
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.04).compile()
    source_id = "archive-adaptive-implicit-layer-source"
    source_revision = "revision-1"
    discovery = discover_adaptive_implicit_surface(
        geometry,
        domain=np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))),
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.005,
            root_tolerance=1.0e-8,
            allow_nonsmooth_field=True,
        ),
        source_id=source_id,
    )
    if discovery.mesh is None:
        raise ValueError(
            "The implicit layer archive fixture requires a resolved adaptive surface."
        )
    native_source = NativeImplicitSource(
        geometry,
        source_id,
        source_revision,
        discovery=discovery,
    )
    contract = SpatialCoordinateContract.si()
    adaptive_query = discovery.boundary_source
    fidelity_source = _NativeImplicitBoundaryQuery(
        adaptive_query,
        native_source.source_id,
        native_source.source_revision,
        contract.spatial_id,
    )
    wall = CellMesh.from_triangles(
        discovery.mesh.vertices,
        discovery.mesh.faces.astype(np.int32),
        numeric_version=native_source.source_revision,
    )
    wall_geometry = CellGeometrySpec.affine(wall)
    root_witness = _implicit_surface_chart_witness(
        native_source.source_id,
        native_source.source_revision,
        discovery,
        wall,
        wall_geometry,
        adaptive_query,
    )
    surface_scope_id = int(native_source.scope_entity_ids(2)[0])
    surface_entity_id, occurrence = native_source.source_entity_key_for_scope(
        2, surface_scope_id
    )
    target_vertices = wall.entity_set(0)
    wall_vertex_ids = np.asarray(target_vertices.entity_ids, dtype=np.int64)
    wall_vertex_association = GeometryAssociation(
        GeometryAssociationKind.IMPLICIT,
        native_source.source_id,
        native_source.source_revision,
        target_vertices.entity_set_id,
        wall_vertex_ids,
        (surface_entity_id,) * wall_vertex_ids.size,
        np.full(
            (wall_vertex_ids.size,),
            discovery.policy.root_tolerance,
            dtype=np.float64,
        ),
        source_dimensions=np.full((wall_vertex_ids.size,), 2, dtype=np.int8),
        source_indices=np.full((wall_vertex_ids.size,), surface_scope_id, dtype=np.int64),
        exact=False,
        source_roles=(GeometrySourceEntityRole.FACET,) * wall_vertex_ids.size,
        source_occurrence_paths=(occurrence,) * wall_vertex_ids.size,
    )
    layers = _layer_mesh(wall, wall_vertex_association, thickness=0.0005)
    _, construction = _mixed_construction(
        layers,
        native_source.source_id,
        native_source.source_revision,
        wall,
        contract.spatial_id,
        fidelity_source=fidelity_source,
    )
    result = curve_layer_core_construction(
        construction,
        layers,
        None,
        policy=HighOrderCurvingPolicy(degree=2, relaxation_rounds=0),
        fidelity_tolerance=0.0005,
        maximum_normal_deviation=np.pi / 4,
        fidelity_source=fidelity_source,
        implicit_root_witness=root_witness,
        frame_id=contract.spatial_id,
    )
    if (
        result.status is not LayerCurvingStatus.CURVED
        or result.candidate_geometry is None
        or result.source_evidence is None
        or not result.source_evidence.implicit_chart_evidence
        or result.source_fidelity is None
        or result.source_fidelity.status != "certified"
    ):
        raise ValueError(
            "The adaptive implicit layer archive fixture requires accepted, certified root-chart evidence: "
            f"{result.failure or result.status.value}"
        )
    return result


__all__ = ["make_implicit_layer_curving_archive_case"]
