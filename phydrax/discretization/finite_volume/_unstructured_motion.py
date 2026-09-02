#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import (
    polygonal_connectivity,
    PolygonalConnectivity,
    tetrahedral_connectivity,
    TetrahedralConnectivity,
)
from ._geometry_protocol import (
    ALEGeometryConsistencyPolicy,
    FiniteVolumeGeometryStatus,
    FiniteVolumeStageFaceBlock,
    FiniteVolumeStageFaceLayout,
    FiniteVolumeStageGeometryEvidence,
    FiniteVolumeStageMetrics,
)
from ._unstructured import (
    _TETRAHEDRAL_FACE_QUADRATURE_BARYCENTRIC,
    evaluate_unstructured_fv_geometry,
    UnstructuredFiniteVolumePlan,
)


VertexMotion = Callable[[Array, Array, Any], ArrayLike]


def _boundary_patches(plan: UnstructuredFiniteVolumePlan, /):
    connectivity = (
        polygonal_connectivity(
            plan.triangles, plan.quadrilaterals, plan.vertices.shape[0]
        )
        if plan.cell_dimension == 2
        else tetrahedral_connectivity(plan.tetrahedra, plan.vertices.shape[0])
    )
    if isinstance(connectivity, PolygonalConnectivity):
        faces = np.asarray(connectivity.edges, dtype=np.int32)
    elif isinstance(connectivity, TetrahedralConnectivity):
        faces = np.asarray(connectivity.faces, dtype=np.int32)
    else:
        raise TypeError("Unsupported unstructured connectivity.")
    return {
        name: faces[np.asarray(indices, dtype=np.int32)]
        for name, indices in zip(plan.patch_names, plan.patch_faces, strict=True)
    }


class UnstructuredFiniteVolumeGeometryState(StrictModule):
    """Fixed-route coordinates bound to one dynamic geometry version."""

    vertices: Array
    time: Array
    geometry_version: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)


class UnstructuredMotionReport(StrictModule):
    maximum_gcl_residual: Array
    minimum_cell_volume: Array
    maximum_cell_volume_change: Array
    maximum_vertex_speed: Array
    passed: Array
    status: Array
    proposed_reduction_factor: Array


class UnstructuredMotionMetrics(StrictModule):
    old_geometry: UnstructuredFiniteVolumeGeometryState
    new_geometry: UnstructuredFiniteVolumeGeometryState
    vertex_velocity: Array
    face_grid_normal_velocity: Array
    swept_face_volumes: Array
    cell_volume_change: Array
    gcl_residual: Array
    report: UnstructuredMotionReport


class _InstantaneousALEGeometry(StrictModule, NonTrainableState):
    vertices: Array
    vertex_velocity: Array
    cell_volumes: Array
    cell_centers: Array
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    face_closure: Array
    face_closure_reference: Array
    quadrature_points: Array
    quadrature_weights: Array
    quadrature_grid_normal_velocity: Array
    face_mesh_volume_rate: Array
    cell_mesh_volume_rate: Array
    coordinate_valid: Array


class UnstructuredALEStepGeometry(StrictModule, NonTrainableState):
    """All certified fixed-connectivity geometry for one SSPRK(3,3) attempt."""

    start_geometry: UnstructuredFiniteVolumeGeometryState
    end_geometry: UnstructuredFiniteVolumeGeometryState
    stage_1: FiniteVolumeStageMetrics
    stage_2: FiniteVolumeStageMetrics
    stage_3: FiniteVolumeStageMetrics
    accepted_geometry: FiniteVolumeStageMetrics
    stage_1_vertex_velocity: Array
    stage_2_vertex_velocity: Array
    stage_3_vertex_velocity: Array
    stage_1_face_mesh_volume_rate: Array
    stage_2_face_mesh_volume_rate: Array
    stage_3_face_mesh_volume_rate: Array
    g1: Array
    g2: Array
    g3: Array
    passed: Array
    status: Array
    proposed_reduction_factor: Array
    topology_epoch_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    motion_plan_id: str = eqx.field(static=True)


class FixedConnectivityMotionPlan(StrictModule, NonTrainableState):
    """JAX-traceable fixed-connectivity ALE geometry for SSPRK(3,3)."""

    base_plan: UnstructuredFiniteVolumePlan
    motion: VertexMotion = eqx.field(static=True)
    consistency_policy: ALEGeometryConsistencyPolicy
    connectivity: PolygonalConnectivity | TetrahedralConnectivity
    owner_cells: Array
    neighbour_cells: Array
    owner_signs: Array
    face_vertices: Array
    face_quadrature_vertex_weights: Array
    face_layout: FiniteVolumeStageFaceLayout
    mapping_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base_plan: UnstructuredFiniteVolumePlan,
        motion: VertexMotion,
        /,
        *,
        mapping_id: str,
        consistency_policy: ALEGeometryConsistencyPolicy | None = None,
    ):
        if not isinstance(base_plan, UnstructuredFiniteVolumePlan):
            raise TypeError("base_plan must be UnstructuredFiniteVolumePlan.")
        if not callable(motion):
            raise TypeError("motion must be callable.")
        if (
            not isinstance(mapping_id, str)
            or not mapping_id
            or mapping_id != mapping_id.strip()
        ):
            raise ValueError("mapping_id must be a non-empty canonical stripped string.")
        policy = (
            ALEGeometryConsistencyPolicy()
            if consistency_policy is None
            else consistency_policy
        )
        if not isinstance(policy, ALEGeometryConsistencyPolicy):
            raise TypeError("consistency_policy must be ALEGeometryConsistencyPolicy.")

        prepared = base_plan.prepare()
        connectivity = prepared.connectivity
        if isinstance(connectivity, PolygonalConnectivity):
            face_vertices = jnp.asarray(connectivity.edges, dtype=jnp.int32)
            offset = 0.5 / np.sqrt(3.0)
            quadrature_vertex_weights = jnp.asarray(
                (
                    (0.5 + offset, 0.5 - offset),
                    (0.5 - offset, 0.5 + offset),
                ),
                dtype=base_plan.vertices.dtype,
            )
        elif isinstance(connectivity, TetrahedralConnectivity):
            face_vertices = jnp.asarray(connectivity.faces, dtype=jnp.int32)
            quadrature_vertex_weights = jnp.asarray(
                _TETRAHEDRAL_FACE_QUADRATURE_BARYCENTRIC,
                dtype=base_plan.vertices.dtype,
            )
        else:
            raise TypeError("Unsupported unstructured connectivity.")

        base_block = prepared.face_blocks[0]
        quadrature_shape = tuple(prepared.face_quadrature_weights.shape)
        face_layout_id = canonical_fingerprint(
            {
                "kind": "fixed-connectivity-ale-face-layout",
                "topology": base_plan.topology_id,
                "face_ids": array_tree_fingerprint(base_block.face_ids),
                "owner_cells": array_tree_fingerprint(prepared.owner_cells),
                "neighbour_cells": array_tree_fingerprint(prepared.neighbour_cells),
                "boundary_policy_ids": array_tree_fingerprint(
                    base_block.boundary_patch_ids
                ),
                "boundary_policy_count": len(prepared.boundary_patch_names),
                "active_mask": array_tree_fingerprint(base_block.active_mask),
                "spatial_shape": tuple(base_block.face_centers.shape),
                "quadrature_shape": quadrature_shape,
                "quadrature_vertex_weights": array_tree_fingerprint(
                    quadrature_vertex_weights
                ),
            }
        )
        face_layout = FiniteVolumeStageFaceLayout(
            face_ids=base_block.face_ids,
            owner_cells=prepared.owner_cells,
            neighbour_cells=prepared.neighbour_cells,
            boundary_policy_ids=base_block.boundary_patch_ids,
            boundary_policy_count=len(prepared.boundary_patch_names),
            active_mask=base_block.active_mask,
            spatial_shape=tuple(base_block.face_centers.shape),
            quadrature_shape=quadrature_shape,
            block_id=face_layout_id,
        )
        geometry_layout_id = canonical_fingerprint(
            {
                "kind": "fixed-connectivity-ale-geometry-layout",
                "topology": base_plan.topology_id,
                "base_geometry": base_plan.geometry_id,
                "mapping": mapping_id,
                "cell_shape": tuple(prepared.cell_volumes.shape),
                "center_shape": tuple(prepared.cell_centers.shape),
                "face_layout": face_layout_id,
                "metric_rule": "coordinate-polytopes-with-owner-oriented-faces",
                "velocity_rule": "jax-jvp-of-vertex-motion",
                "face_rate_rule": "quadrature-integral-of-grid-normal-velocity",
            }
        )

        self.base_plan = base_plan
        self.motion = motion
        self.consistency_policy = policy
        self.connectivity = connectivity
        self.owner_cells = jnp.asarray(prepared.owner_cells, dtype=jnp.int32)
        self.neighbour_cells = jnp.asarray(prepared.neighbour_cells, dtype=jnp.int32)
        self.owner_signs = jnp.asarray(
            prepared.owner_signs, dtype=base_plan.vertices.dtype
        )
        self.face_vertices = face_vertices
        self.face_quadrature_vertex_weights = quadrature_vertex_weights
        self.face_layout = face_layout
        self.mapping_id = mapping_id
        self.geometry_layout_id = geometry_layout_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-connectivity-unstructured-ale-motion",
                "topology": base_plan.topology_id,
                "base_geometry": base_plan.geometry_id,
                "mapping": mapping_id,
                "geometry_layout": geometry_layout_id,
                "evidence_policy": policy.policy_id,
                "ssprk33_abscissae": (0.0, 1.0, 0.5),
                "ssprk33_gcl_rule": {
                    "v1": "vn+dt*g1",
                    "v2": "3/4*vn+1/4*(v1+dt*g2)",
                    "vnew": "1/3*vn+2/3*(v2+dt*g3)",
                },
                "stage_consistency_expected_order": 2,
                "final_consistency_expected_order": 4,
            }
        )

    def evaluate_plan(
        self, time: ArrayLike, args: Any = None, /
    ) -> UnstructuredFiniteVolumePlan:
        """Evaluate one immutable geometry epoch; intentionally host-side."""

        time_ = jnp.asarray(time).reshape(())
        vertices = np.asarray(
            self.motion(time_, self.base_plan.vertices, args), dtype=float
        )
        if vertices.shape != self.base_plan.vertices.shape:
            raise ValueError("Fixed-connectivity motion must preserve vertex shape.")
        plan = UnstructuredFiniteVolumePlan(
            vertices,
            triangles=self.base_plan.triangles,
            quadrilaterals=self.base_plan.quadrilaterals,
            tetrahedra=self.base_plan.tetrahedra,
            vertex_global_ids=self.base_plan.vertex_global_ids,
            cell_global_ids=self.base_plan.cell_global_ids,
            boundary_patches=_boundary_patches(self.base_plan),
            field_name=self.base_plan.field_name,
            component_names=self.base_plan.component_names,
        )
        if plan.topology_id != self.base_plan.topology_id:
            raise ValueError("Fixed-connectivity motion changed topology identity.")
        return plan

    def geometry_state(
        self,
        time: ArrayLike,
        args: Any = None,
        /,
        *,
        geometry_version: ArrayLike = 0,
    ) -> UnstructuredFiniteVolumeGeometryState:
        plan = self.evaluate_plan(time, args)
        version = self._version(geometry_version, "geometry_version", reserve=0)
        return UnstructuredFiniteVolumeGeometryState(
            vertices=plan.vertices,
            time=jnp.asarray(time).reshape(()),
            geometry_version=version,
            topology_id=plan.topology_id,
            geometry_layout_id=self.geometry_layout_id,
        )

    @staticmethod
    def _version(value: ArrayLike, name: str, /, *, reserve: int) -> Array:
        scalar = jnp.asarray(value)
        if scalar.shape != () or scalar.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a scalar integer.")
        scalar = eqx.error_if(
            scalar,
            (scalar < 0) | (scalar > np.iinfo(np.int32).max - reserve),
            f"{name} must leave room for every nonnegative SSPRK geometry version.",
        )
        return scalar.astype(jnp.int32)

    def _coordinate_geometry_is_valid(
        self,
        vertices: Array,
        vertex_velocity: Array,
        /,
    ) -> Array:
        """Return whether a trial coordinate geometry is safe to evaluate."""

        points = jnp.asarray(vertices)
        valid = jnp.all(jnp.isfinite(points)) & jnp.all(jnp.isfinite(vertex_velocity))
        if isinstance(self.connectivity, PolygonalConnectivity):
            triangle_points = points[self.base_plan.triangles]
            triangle_cross = (triangle_points[:, 1, 0] - triangle_points[:, 0, 0]) * (
                triangle_points[:, 2, 1] - triangle_points[:, 0, 1]
            ) - (triangle_points[:, 1, 1] - triangle_points[:, 0, 1]) * (
                triangle_points[:, 2, 0] - triangle_points[:, 0, 0]
            )
            triangle_volumes = 0.5 * triangle_cross
            triangle_centers = jnp.mean(triangle_points, axis=1)

            root = 1.0 / np.sqrt(3.0)
            reference = jnp.asarray(
                ((-root, -root), (root, -root), (root, root), (-root, root)),
                dtype=points.dtype,
            )
            xi = reference[:, 0]
            eta = reference[:, 1]
            shape = 0.25 * jnp.stack(
                (
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                    (1.0 - xi) * (1.0 + eta),
                ),
                axis=-1,
            )
            gradient = 0.25 * jnp.stack(
                (
                    jnp.stack((-(1.0 - eta), -(1.0 - xi)), axis=-1),
                    jnp.stack((1.0 - eta, -(1.0 + xi)), axis=-1),
                    jnp.stack((1.0 + eta, 1.0 + xi), axis=-1),
                    jnp.stack((-(1.0 + eta), 1.0 - xi), axis=-1),
                ),
                axis=1,
            )
            quadrilateral_points = points[self.base_plan.quadrilaterals]
            mapped = ein.contract("qv,cvd->cqd", shape, quadrilateral_points)
            jacobian = ein.contract("qva,cvd->cqad", gradient, quadrilateral_points)
            determinant = (
                jacobian[..., 0, 0] * jacobian[..., 1, 1]
                - jacobian[..., 0, 1] * jacobian[..., 1, 0]
            )
            quadrilateral_volumes = jnp.sum(determinant, axis=1)
            safe_quadrilateral_volumes = jnp.where(
                jnp.isfinite(quadrilateral_volumes) & (quadrilateral_volumes > 0.0),
                quadrilateral_volumes,
                1.0,
            )
            quadrilateral_centers = (
                jnp.sum(mapped * determinant[..., None], axis=1)
                / safe_quadrilateral_volumes[:, None]
            )
            cell_volumes = jnp.concatenate((triangle_volumes, quadrilateral_volumes))
            cell_centers = jnp.concatenate(
                (triangle_centers, quadrilateral_centers),
                axis=0,
            )
            valid = (
                valid
                & jnp.all(jnp.isfinite(determinant) & (determinant > 0.0))
                & jnp.all(jnp.isfinite(cell_volumes) & (cell_volumes > 0.0))
                & jnp.all(jnp.isfinite(cell_centers))
            )
            face_points = points[self.connectivity.edges]
            face_centers = 0.5 * (face_points[:, 0] + face_points[:, 1])
            tangent = face_points[:, 1] - face_points[:, 0]
            canonical_area = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        else:
            cell_points = points[self.base_plan.tetrahedra]
            determinant = jnp.linalg.det(
                jnp.stack(
                    (
                        cell_points[:, 1] - cell_points[:, 0],
                        cell_points[:, 2] - cell_points[:, 0],
                        cell_points[:, 3] - cell_points[:, 0],
                    ),
                    axis=-1,
                )
            )
            cell_volumes = determinant / 6.0
            cell_centers = jnp.mean(cell_points, axis=1)
            valid = (
                valid
                & jnp.all(jnp.isfinite(cell_volumes) & (cell_volumes > 0.0))
                & jnp.all(jnp.isfinite(cell_centers))
            )
            face_points = points[self.connectivity.faces]
            face_centers = jnp.mean(face_points, axis=1)
            canonical_area = 0.5 * jnp.cross(
                face_points[:, 1] - face_points[:, 0],
                face_points[:, 2] - face_points[:, 0],
            )

        area_vectors = self.owner_signs[:, None] * canonical_area
        face_measures = jnp.linalg.norm(area_vectors, axis=-1)
        owner_centers = cell_centers[self.owner_cells]
        outward = jnp.sum((face_centers - owner_centers) * area_vectors, axis=-1)
        return (
            valid
            & jnp.all(jnp.isfinite(face_centers))
            & jnp.all(jnp.isfinite(area_vectors))
            & jnp.all(jnp.isfinite(face_measures) & (face_measures > 0.0))
            & jnp.all(jnp.isfinite(outward) & (outward > 0.0))
        )

    def _instantaneous_geometry(
        self,
        time: Array,
        args: Any,
        /,
    ) -> _InstantaneousALEGeometry:
        def evaluate_vertices(value):
            return jnp.asarray(self.motion(value, self.base_plan.vertices, args))

        vertices, vertex_velocity = jax.jvp(
            evaluate_vertices,
            (time,),
            (jnp.ones_like(time),),
        )
        if vertices.shape != self.base_plan.vertices.shape:
            raise ValueError("Fixed-connectivity motion must preserve vertex shape.")
        if vertex_velocity.shape != self.base_plan.vertices.shape:
            raise ValueError(
                "The time derivative of fixed-connectivity motion must preserve "
                "vertex shape."
            )
        coordinate_valid = self._coordinate_geometry_is_valid(
            vertices,
            vertex_velocity,
        )

        def evaluate_geometry(points):
            return evaluate_unstructured_fv_geometry(
                points,
                self.base_plan.triangles,
                self.base_plan.quadrilaterals,
                self.base_plan.tetrahedra,
                self.connectivity,
                self.owner_cells,
                self.owner_signs,
            )

        fallback_vertices = jnp.asarray(self.base_plan.vertices, dtype=vertices.dtype)
        geometry_values = jax.lax.cond(
            coordinate_valid,
            evaluate_geometry,
            lambda _: evaluate_geometry(fallback_vertices),
            vertices,
        )
        (
            cell_volumes,
            cell_centers,
            face_centers,
            area_vectors,
            face_measures,
            face_closure,
            quadrature_points,
            quadrature_weights,
        ) = geometry_values
        safe_vertices = jnp.where(jnp.isfinite(vertices), vertices, fallback_vertices)
        safe_vertex_velocity = jnp.where(
            jnp.isfinite(vertex_velocity),
            vertex_velocity,
            jnp.zeros_like(vertex_velocity),
        )
        face_vertex_velocity = safe_vertex_velocity[self.face_vertices]
        quadrature_grid_velocity = ein.contract(
            "qv,fvd->fqd",
            self.face_quadrature_vertex_weights,
            face_vertex_velocity,
        )
        unit_normals = area_vectors / face_measures[:, None]
        quadrature_grid_normal_velocity = ein.contract(
            "fqd,fd->fq",
            quadrature_grid_velocity,
            unit_normals,
        )
        face_mesh_volume_rate = ein.contract(
            "fq,fq->f",
            quadrature_weights,
            quadrature_grid_normal_velocity,
        )
        cell_mesh_volume_rate = jnp.zeros_like(cell_volumes)
        cell_mesh_volume_rate = cell_mesh_volume_rate.at[self.owner_cells].add(
            face_mesh_volume_rate
        )
        neighbour_active = self.neighbour_cells >= 0
        safe_neighbours = jnp.where(neighbour_active, self.neighbour_cells, 0)
        cell_mesh_volume_rate = cell_mesh_volume_rate.at[safe_neighbours].add(
            jnp.where(neighbour_active, -face_mesh_volume_rate, 0.0)
        )
        face_closure_reference = jnp.zeros_like(cell_volumes)
        face_closure_reference = face_closure_reference.at[self.owner_cells].add(
            face_measures
        )
        face_closure_reference = face_closure_reference.at[safe_neighbours].add(
            jnp.where(neighbour_active, face_measures, 0.0)
        )
        return _InstantaneousALEGeometry(
            vertices=safe_vertices,
            vertex_velocity=safe_vertex_velocity,
            cell_volumes=cell_volumes,
            cell_centers=cell_centers,
            face_centers=face_centers,
            area_vectors=area_vectors,
            face_measures=face_measures,
            face_closure=face_closure,
            face_closure_reference=face_closure_reference,
            quadrature_points=quadrature_points,
            quadrature_weights=quadrature_weights,
            quadrature_grid_normal_velocity=quadrature_grid_normal_velocity,
            face_mesh_volume_rate=face_mesh_volume_rate,
            cell_mesh_volume_rate=cell_mesh_volume_rate,
            coordinate_valid=coordinate_valid,
        )

    def _evidence(
        self,
        geometry: _InstantaneousALEGeometry,
        effective_volumes: Array,
        coordinate_volumes: Array,
        gcl_target_volumes: Array,
        gcl_right_hand_side: Array,
        validity: Array,
        /,
        *,
        expected_order: int,
        evidence_version: Array,
    ) -> FiniteVolumeStageGeometryEvidence:
        dtype = jnp.result_type(
            effective_volumes.dtype,
            coordinate_volumes.dtype,
            gcl_target_volumes.dtype,
            gcl_right_hand_side.dtype,
        )
        finite_limit = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).max, dtype=dtype)) / 8.0

        def bounded(values):
            array = jnp.asarray(values, dtype=dtype)
            return jnp.clip(
                jnp.nan_to_num(
                    array,
                    nan=0.0,
                    posinf=finite_limit,
                    neginf=-finite_limit,
                ),
                -finite_limit,
                finite_limit,
            )

        effective = bounded(effective_volumes)
        coordinate = bounded(coordinate_volumes)
        gcl_target = bounded(gcl_target_volumes)
        gcl_right_hand_side_ = bounded(gcl_right_hand_side)
        closure = bounded(jnp.linalg.norm(geometry.face_closure, axis=-1))
        closure_reference = bounded(geometry.face_closure_reference)
        coordinate_reference = jnp.maximum(jnp.abs(effective), jnp.abs(coordinate))
        coordinate_defect = jnp.abs(effective - coordinate)
        all_finite = (
            jnp.isfinite(effective_volumes)
            & jnp.isfinite(coordinate_volumes)
            & jnp.isfinite(gcl_target_volumes)
            & jnp.isfinite(gcl_right_hand_side)
            & jnp.isfinite(closure)
            & jnp.isfinite(closure_reference)
        )
        valid = jnp.broadcast_to(jnp.asarray(validity, dtype=bool), effective.shape)
        valid = valid & all_finite
        absolute = jnp.asarray(
            self.consistency_policy.absolute_tolerance,
            dtype=dtype,
        )
        relative = jnp.asarray(
            self.consistency_policy.relative_tolerance,
            dtype=dtype,
        )
        coordinate_tolerance = absolute + relative * coordinate_reference
        invalid_defect = 4.0 * coordinate_tolerance
        coordinate_defect = jnp.where(
            valid,
            coordinate_defect,
            jnp.maximum(coordinate_defect, invalid_defect),
        )
        return self.consistency_policy.evidence(
            coordinate_effective_volume_defect=coordinate_defect,
            coordinate_effective_volume_reference=coordinate_reference,
            face_closure_defect=closure,
            face_closure_reference=closure_reference,
            gcl_identity_defect=jnp.abs(gcl_target - gcl_right_hand_side_),
            gcl_identity_reference=jnp.maximum(
                jnp.abs(gcl_target),
                jnp.abs(gcl_right_hand_side_),
            ),
            expected_order=expected_order,
            evidence_version=evidence_version,
        )

    def _stage_metrics(
        self,
        geometry: _InstantaneousALEGeometry,
        effective_volumes: Array,
        evidence: FiniteVolumeStageGeometryEvidence,
        /,
        *,
        time: Array,
        topology_epoch_id: str,
        geometry_version: Array,
    ) -> FiniteVolumeStageMetrics:
        face_block = FiniteVolumeStageFaceBlock(
            layout=self.face_layout,
            face_centers=geometry.face_centers,
            area_vectors=geometry.area_vectors,
            face_measures=geometry.face_measures,
            quadrature_points=geometry.quadrature_points,
            quadrature_weights=geometry.quadrature_weights,
            quadrature_grid_normal_velocity=(geometry.quadrature_grid_normal_velocity),
        )
        return FiniteVolumeStageMetrics(
            topology_epoch_id=topology_epoch_id,
            geometry_family_id=self.plan_id,
            geometry_layout_id=self.geometry_layout_id,
            geometry_version=geometry_version,
            time=time,
            effective_cell_volumes=effective_volumes,
            coordinate_effective_cell_volumes=geometry.cell_volumes,
            mesh_volume_rate=geometry.cell_mesh_volume_rate,
            cell_centers=geometry.cell_centers,
            active_cell_mask=jnp.ones_like(geometry.cell_volumes, dtype=bool),
            face_blocks=(face_block,),
            evidence=evidence,
        )

    def prepare_ssprk33_step(
        self,
        start_time: ArrayLike,
        dt: ArrayLike,
        topology_epoch_id: str,
        start_geometry_version: ArrayLike,
        start_evidence_version: ArrayLike,
        args: Any = None,
        /,
        *,
        prior_effective_cell_volumes: ArrayLike,
    ) -> UnstructuredALEStepGeometry:
        """Prepare and certify all c=(0, 1, 1/2) ALE stage geometry."""

        if (
            not isinstance(topology_epoch_id, str)
            or not topology_epoch_id
            or topology_epoch_id != topology_epoch_id.strip()
        ):
            raise ValueError(
                "topology_epoch_id must be a non-empty canonical stripped string."
            )
        time_value = jnp.asarray(start_time)
        step = jnp.asarray(dt)
        if time_value.shape != () or step.shape != ():
            raise ValueError("start_time and dt must be scalars.")
        dtype = jnp.result_type(
            self.base_plan.vertices.dtype,
            time_value.dtype,
            step.dtype,
        )
        if dtype.kind != "f":
            raise ValueError("ALE start_time and dt must have real floating dtype.")
        prior_volumes = jnp.asarray(prior_effective_cell_volumes)
        expected_cell_shape = (int(self.base_plan.cell_global_ids.size),)
        if prior_volumes.shape != expected_cell_shape:
            raise ValueError(
                "prior_effective_cell_volumes must have exact shape "
                f"{expected_cell_shape}."
            )
        if prior_volumes.dtype.kind not in "fiu":
            raise ValueError("prior_effective_cell_volumes must have real numeric dtype.")
        time_value = jnp.asarray(time_value, dtype=dtype)
        step = jnp.asarray(step, dtype=dtype)
        prior_volumes = jnp.asarray(prior_volumes, dtype=dtype)
        prior_volumes = eqx.error_if(
            prior_volumes,
            jnp.any(~jnp.isfinite(prior_volumes) | (prior_volumes <= 0.0)),
            "Active prior_effective_cell_volumes must be positive and finite.",
        )
        time_value = eqx.error_if(
            time_value,
            ~jnp.isfinite(time_value),
            "ALE start_time must be finite.",
        )
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "ALE dt must be positive and finite.",
        )
        geometry_version = self._version(
            start_geometry_version,
            "start_geometry_version",
            reserve=3,
        )
        evidence_version = self._version(
            start_evidence_version,
            "start_evidence_version",
            reserve=3,
        )
        end_time = time_value + step
        midpoint_time = time_value + 0.5 * step

        geometry_1 = self._instantaneous_geometry(time_value, args)
        geometry_2 = self._instantaneous_geometry(end_time, args)
        geometry_3 = self._instantaneous_geometry(midpoint_time, args)
        vn = prior_volumes
        v1_raw = vn + step * geometry_1.cell_mesh_volume_rate
        v2_right_hand_side = 0.75 * vn + 0.25 * (
            v1_raw + step * geometry_2.cell_mesh_volume_rate
        )
        v2_raw = v2_right_hand_side
        vnew_right_hand_side = (1.0 / 3.0) * vn + (2.0 / 3.0) * (
            v2_raw + step * geometry_3.cell_mesh_volume_rate
        )
        vnew_raw = vnew_right_hand_side
        v1_valid = jnp.isfinite(v1_raw) & (v1_raw > 0.0)
        v2_valid = jnp.isfinite(v2_raw) & (v2_raw > 0.0)
        vnew_valid = jnp.isfinite(vnew_raw) & (vnew_raw > 0.0)
        v1 = jnp.where(v1_valid, v1_raw, vn)
        v2 = jnp.where(v2_valid, v2_raw, vn)
        vnew = jnp.where(vnew_valid, vnew_raw, vn)

        evidence_1 = self._evidence(
            geometry_1,
            vn,
            geometry_1.cell_volumes,
            v1_raw,
            vn + step * geometry_1.cell_mesh_volume_rate,
            geometry_1.coordinate_valid,
            expected_order=2,
            evidence_version=evidence_version,
        )
        evidence_2 = self._evidence(
            geometry_2,
            v1_raw,
            geometry_2.cell_volumes,
            v2_raw,
            v2_right_hand_side,
            geometry_2.coordinate_valid & jnp.all(v1_valid),
            expected_order=2,
            evidence_version=evidence_version + 1,
        )
        evidence_3 = self._evidence(
            geometry_3,
            v2_raw,
            geometry_3.cell_volumes,
            vnew_raw,
            vnew_right_hand_side,
            geometry_3.coordinate_valid & jnp.all(v2_valid),
            expected_order=2,
            evidence_version=evidence_version + 2,
        )
        accepted_evidence = self._evidence(
            geometry_2,
            vnew_raw,
            geometry_2.cell_volumes,
            vnew_raw,
            vnew_right_hand_side,
            geometry_2.coordinate_valid & jnp.all(vnew_valid),
            expected_order=4,
            evidence_version=evidence_version + 3,
        )

        stage_1 = self._stage_metrics(
            geometry_1,
            vn,
            evidence_1,
            time=time_value,
            topology_epoch_id=topology_epoch_id,
            geometry_version=geometry_version,
        )
        stage_2 = self._stage_metrics(
            geometry_2,
            v1,
            evidence_2,
            time=end_time,
            topology_epoch_id=topology_epoch_id,
            geometry_version=geometry_version + 1,
        )
        stage_3 = self._stage_metrics(
            geometry_3,
            v2,
            evidence_3,
            time=midpoint_time,
            topology_epoch_id=topology_epoch_id,
            geometry_version=geometry_version + 2,
        )
        accepted_geometry = self._stage_metrics(
            geometry_2,
            vnew,
            accepted_evidence,
            time=end_time,
            topology_epoch_id=topology_epoch_id,
            geometry_version=geometry_version + 3,
        )
        all_evidence = (
            evidence_1,
            evidence_2,
            evidence_3,
            accepted_evidence,
        )
        passed = jnp.all(jnp.stack(tuple(item.passed for item in all_evidence)))
        proposed_reduction_factor = jnp.min(
            jnp.stack(tuple(item.proposed_reduction_factor for item in all_evidence))
        )
        status = jnp.where(
            passed,
            int(FiniteVolumeGeometryStatus.SUCCESS),
            int(FiniteVolumeGeometryStatus.FAILED),
        )
        return UnstructuredALEStepGeometry(
            start_geometry=UnstructuredFiniteVolumeGeometryState(
                vertices=geometry_1.vertices,
                time=time_value,
                geometry_version=geometry_version,
                topology_id=self.base_plan.topology_id,
                geometry_layout_id=self.geometry_layout_id,
            ),
            end_geometry=UnstructuredFiniteVolumeGeometryState(
                vertices=geometry_2.vertices,
                time=end_time,
                geometry_version=geometry_version + 3,
                topology_id=self.base_plan.topology_id,
                geometry_layout_id=self.geometry_layout_id,
            ),
            stage_1=stage_1,
            stage_2=stage_2,
            stage_3=stage_3,
            accepted_geometry=accepted_geometry,
            stage_1_vertex_velocity=geometry_1.vertex_velocity,
            stage_2_vertex_velocity=geometry_2.vertex_velocity,
            stage_3_vertex_velocity=geometry_3.vertex_velocity,
            stage_1_face_mesh_volume_rate=geometry_1.face_mesh_volume_rate,
            stage_2_face_mesh_volume_rate=geometry_2.face_mesh_volume_rate,
            stage_3_face_mesh_volume_rate=geometry_3.face_mesh_volume_rate,
            g1=geometry_1.cell_mesh_volume_rate,
            g2=geometry_2.cell_mesh_volume_rate,
            g3=geometry_3.cell_mesh_volume_rate,
            passed=passed,
            status=status,
            proposed_reduction_factor=proposed_reduction_factor,
            topology_epoch_id=topology_epoch_id,
            geometry_layout_id=self.geometry_layout_id,
            motion_plan_id=self.plan_id,
        )

    def advance(
        self,
        old_time: ArrayLike,
        new_time: ArrayLike,
        args: Any = None,
        /,
    ) -> UnstructuredMotionMetrics:
        """Return the legacy interval diagnostic from the SSPRK geometry path."""

        old_time_ = jnp.asarray(old_time)
        new_time_ = jnp.asarray(new_time)
        if old_time_.shape != () or new_time_.shape != ():
            raise ValueError("Motion interval endpoints must be scalars.")
        step_size = new_time_ - old_time_
        start_dtype = jnp.result_type(
            self.base_plan.vertices.dtype,
            old_time_.dtype,
            new_time_.dtype,
        )
        coordinate_start = self._instantaneous_geometry(
            jnp.asarray(old_time_, dtype=start_dtype),
            args,
        )
        step_geometry = self.prepare_ssprk33_step(
            old_time_,
            step_size,
            self.base_plan.topology_id,
            0,
            0,
            args,
            prior_effective_cell_volumes=coordinate_start.cell_volumes,
        )
        swept_face_volumes = step_size * (
            (1.0 / 6.0) * step_geometry.stage_1_face_mesh_volume_rate
            + (1.0 / 6.0) * step_geometry.stage_2_face_mesh_volume_rate
            + (2.0 / 3.0) * step_geometry.stage_3_face_mesh_volume_rate
        )
        swept_cell_volumes = step_size * (
            (1.0 / 6.0) * step_geometry.g1
            + (1.0 / 6.0) * step_geometry.g2
            + (2.0 / 3.0) * step_geometry.g3
        )
        cell_volume_change = (
            step_geometry.accepted_geometry.coordinate_effective_cell_volumes
            - step_geometry.stage_1.coordinate_effective_cell_volumes
        )
        gcl_residual = cell_volume_change - swept_cell_volumes
        maximum_vertex_speed = jnp.max(
            jnp.stack(
                (
                    jnp.max(
                        jnp.linalg.norm(step_geometry.stage_1_vertex_velocity, axis=-1)
                    ),
                    jnp.max(
                        jnp.linalg.norm(step_geometry.stage_2_vertex_velocity, axis=-1)
                    ),
                    jnp.max(
                        jnp.linalg.norm(step_geometry.stage_3_vertex_velocity, axis=-1)
                    ),
                )
            )
        )
        return UnstructuredMotionMetrics(
            old_geometry=step_geometry.start_geometry,
            new_geometry=step_geometry.end_geometry,
            vertex_velocity=step_geometry.stage_1_vertex_velocity,
            face_grid_normal_velocity=step_geometry.stage_1.face_blocks[
                0
            ].grid_normal_velocity,
            swept_face_volumes=swept_face_volumes,
            cell_volume_change=cell_volume_change,
            gcl_residual=gcl_residual,
            report=UnstructuredMotionReport(
                maximum_gcl_residual=jnp.max(jnp.abs(gcl_residual)),
                minimum_cell_volume=jnp.min(
                    step_geometry.accepted_geometry.coordinate_effective_cell_volumes
                ),
                maximum_cell_volume_change=jnp.max(jnp.abs(cell_volume_change)),
                maximum_vertex_speed=maximum_vertex_speed,
                passed=step_geometry.passed,
                status=step_geometry.status,
                proposed_reduction_factor=(step_geometry.proposed_reduction_factor),
            ),
        )


__all__ = [
    "ALEGeometryConsistencyPolicy",
    "FixedConnectivityMotionPlan",
    "UnstructuredALEStepGeometry",
    "UnstructuredFiniteVolumeGeometryState",
    "UnstructuredMotionMetrics",
    "UnstructuredMotionReport",
]
