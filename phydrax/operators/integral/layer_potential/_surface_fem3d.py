#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import (
    CellMesh,
    discontinuous_element,
    EntitySet,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from ....geometry import CompiledGeometry, MeshRegion
from ._surface3d import SurfacePanelization3D


class _SurfaceFEMBinding3D(StrictModule, NonTrainableState):
    """Exact fixed-geometry binding of one mesh region to scalar surface DP0."""

    region: MeshRegion
    geometry: CompiledGeometry
    mesh: CellMesh
    discretization: FiniteElementDiscretization
    panelization: SurfacePanelization3D
    surface_entities: EntitySet
    face_component_ids: Array
    face_areas: Array
    component_count: int = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        region: MeshRegion,
        /,
        *,
        quadrature_order: int,
        numeric_version: str,
    ):
        if not isinstance(region, MeshRegion):
            raise TypeError("[geometry] 3D Galerkin preparation requires a MeshRegion.")
        version = str(numeric_version)
        if not version:
            raise ValueError("[numeric-version] numeric_version must be non-empty.")

        triangle_mesh = region.triangle_mesh
        topology = triangle_mesh.topology
        if not topology.watertight:
            raise ValueError(
                "[geometry] Surface Galerkin preparation requires a watertight mesh."
            )
        vertices = np.asarray(triangle_mesh.vertices, dtype=float)
        faces = np.asarray(triangle_mesh.faces, dtype=np.int32)
        component_ids = np.asarray(topology.face_component_ids, dtype=np.int32)
        component_count = int(topology.num_face_components)
        triangles = vertices[faces]
        doubled_area = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        scale = max(float(np.max(np.ptp(vertices, axis=0))), 1.0)
        area_tolerance = 64.0 * np.finfo(float).eps * scale * scale
        if np.any(~np.isfinite(vertices)) or np.any(~np.isfinite(doubled_area)):
            raise ValueError("[geometry] Surface Galerkin geometry must be finite.")
        if np.any(doubled_area <= area_tolerance):
            raise ValueError("[geometry] Surface Galerkin faces must be nondegenerate.")

        component_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        volume_tolerance = 64.0 * np.finfo(float).eps * scale**3
        for component in range(component_count):
            component_faces = faces[component_ids == component]
            component_triangles = vertices[component_faces]
            signed_volume = (
                np.sum(
                    np.sum(
                        component_triangles[:, 0]
                        * np.cross(
                            component_triangles[:, 1],
                            component_triangles[:, 2],
                        ),
                        axis=1,
                    )
                )
                / 6.0
            )
            if not np.isfinite(signed_volume) or signed_volume <= volume_tolerance:
                raise ValueError(
                    "[geometry] Every conductor component must have positive "
                    "outward signed volume."
                )
            component_vertices = vertices[np.unique(component_faces)]
            component_bounds.append(
                (np.min(component_vertices, axis=0), np.max(component_vertices, axis=0))
            )
        for left in range(component_count):
            for right in range(left + 1, component_count):
                left_min, left_max = component_bounds[left]
                right_min, right_max = component_bounds[right]
                gap = np.maximum(
                    np.maximum(left_min - right_max, right_min - left_max),
                    0.0,
                )
                if float(np.linalg.norm(gap)) <= 64.0 * np.finfo(float).eps * scale:
                    raise ValueError(
                        "[geometry] Initial capacitance geometry requires strictly "
                        "separated component bounding boxes."
                    )

        mesh = CellMesh.from_triangles(
            triangle_mesh.vertices,
            triangle_mesh.faces,
            vertex_global_ids=np.arange(vertices.shape[0], dtype=np.int64),
            cell_global_ids=np.arange(faces.shape[0], dtype=np.int64),
        )
        field = FiniteElementFieldSpec(
            "surface_density",
            discontinuous_element("triangle", 0),
        )
        discretization = FiniteElementPlan(mesh, field).prepare(numeric_version=version)
        dof_map = discretization.dof_maps[0]
        if (
            len(mesh.blocks) != 1
            or mesh.blocks[0].cell_kind != "triangle"
            or dof_map.association != "cell"
            or dof_map.global_dof_count != faces.shape[0]
            or np.asarray(dof_map.cell_dofs[0]).shape != (faces.shape[0], 1)
            or not np.array_equal(
                np.asarray(dof_map.cell_dofs[0])[:, 0],
                np.arange(faces.shape[0], dtype=np.int32),
            )
        ):
            raise ValueError(
                "[geometry] Prepared surface DP0 routes do not match face order."
            )

        geometry = region.compile()
        panelization = SurfacePanelization3D(
            geometry.boundary_atlas,
            quadrature_order=int(quadrature_order),
            geometry=geometry,
        )
        nodes_per_panel = panelization.nodes_per_panel
        expected_charts = np.repeat(
            np.arange(faces.shape[0], dtype=np.int32), nodes_per_panel
        )
        if (
            panelization.panel_count != faces.shape[0]
            or not np.array_equal(np.asarray(panelization.chart_indices), expected_charts)
            or not np.array_equal(np.asarray(panelization.panel_ids), expected_charts)
        ):
            raise ValueError(
                "[geometry] Surface panel order does not match face/DP0 order."
            )

        surface_entities = mesh.topology.entity_sets[mesh.topological_dimension]
        face_areas = jnp.asarray(doubled_area * 0.5)
        self.region = region
        self.geometry = geometry
        self.mesh = mesh
        self.discretization = discretization
        self.panelization = panelization
        self.surface_entities = surface_entities
        self.face_component_ids = jnp.asarray(component_ids, dtype=jnp.int32)
        self.face_areas = face_areas
        self.component_count = component_count
        self.numeric_version = version
        self.binding_id = canonical_fingerprint(
            {
                "kind": "surface-fem-binding-3d-v1",
                "region": region.feature_id,
                "faces": array_tree_fingerprint(triangle_mesh.faces),
                "mesh": mesh.mesh_id,
                "fem": discretization.prepared_id,
                "panelization": panelization.panelization_id,
                "components": array_tree_fingerprint(self.face_component_ids),
                "numeric_version": version,
            }
        )

    @property
    def face_count(self) -> int:
        return int(self.face_areas.shape[0])


__all__: list[str] = []
