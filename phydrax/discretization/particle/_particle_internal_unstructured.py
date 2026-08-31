#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..finite_volume import UnstructuredFiniteVolumePlan
from ._particle_internal_mesh import (
    AbstractParticleInternalMeshPlan,
    AbstractPreparedParticleInternalMesh,
)
from ._rigid_body import quaternion_rotation_matrix


class ParticleInternalMeshMetrics(StrictModule):
    cell_measures: Array
    cell_centers: Array
    face_measures: Array
    face_centers: Array
    face_normals: Array
    center_distances: Array
    owner_cells: Array
    neighbour_cells: Array
    boundary_faces: Array
    active_cells: Array
    active_faces: Array
    surface_measure: Array
    successful: Array
    mesh_id: str = eqx.field(static=True)


class ParticleBoundaryTrace(StrictModule):
    owner_cells: Array
    positions: Array
    normals: Array
    measures: Array
    active: Array
    mesh_id: str = eqx.field(static=True)


class UnstructuredParticleInternalMeshPlan(AbstractParticleInternalMeshPlan):
    finite_volume: UnstructuredFiniteVolumePlan
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        /,
        *,
        triangles: ArrayLike | None = None,
        quadrilaterals: ArrayLike | None = None,
        tetrahedra: ArrayLike | None = None,
        boundary_patches=None,
        mesh_id: str | None = None,
    ):
        finite_volume = UnstructuredFiniteVolumePlan(
            vertices,
            triangles=triangles,
            quadrilaterals=quadrilaterals,
            tetrahedra=tetrahedra,
            boundary_patches=boundary_patches,
            field_name="particle_internal",
            component_names=("content",),
        )
        generated = canonical_fingerprint(
            {
                "kind": "unstructured-particle-internal-mesh",
                "finite_volume": finite_volume.plan_id,
            }
        )
        self.finite_volume = finite_volume
        self.mesh_id = generated if mesh_id is None else str(mesh_id)
        if not self.mesh_id:
            raise ValueError("mesh_id must be nonempty.")

    @property
    def cell_capacity(self) -> int:
        return int(self.finite_volume.mesh.connectivity.cell_count)

    def prepare(self):
        return PreparedUnstructuredParticleInternalMesh(self)


class PreparedUnstructuredParticleInternalMesh(AbstractPreparedParticleInternalMesh):
    plan: UnstructuredParticleInternalMeshPlan
    discretization: object
    dimension: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    face_capacity: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: UnstructuredParticleInternalMeshPlan, /):
        if not isinstance(plan, UnstructuredParticleInternalMeshPlan):
            raise TypeError("plan must be UnstructuredParticleInternalMeshPlan.")
        discretization = plan.finite_volume.prepare()
        self.plan = plan
        self.discretization = discretization
        self.dimension = int(discretization.cell_dimension)
        self.cell_capacity = int(discretization.cell_count)
        self.face_capacity = int(discretization.owner_cells.shape[0])
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-particle-internal-mesh",
                "plan": plan.mesh_id,
                "finite_volume": discretization.prepared_id,
            }
        )

    def metrics(
        self,
        outer_scale: ArrayLike,
        /,
        *,
        active_cells: ArrayLike | None = None,
    ) -> ParticleInternalMeshMetrics:
        scale = jnp.asarray(outer_scale)
        if scale.ndim != 1:
            raise ValueError("outer_scale must have particle-batch shape.")
        count = scale.shape[0]
        if active_cells is None:
            cells_active = jnp.ones((count, self.cell_capacity), dtype=bool)
        else:
            cells_active = jnp.asarray(active_cells, dtype=bool)
            if cells_active.shape != (count, self.cell_capacity):
                raise ValueError("active_cells must have particle-cell shape.")
        dimension = self.dimension
        cell_scale = scale[:, None] ** dimension
        face_scale = scale[:, None] ** (dimension - 1)
        cell_measures = cell_scale * self.discretization.cell_volumes[None, :]
        cell_centers = scale[:, None, None] * self.discretization.cell_centers[None, :, :]
        face_measures = face_scale * self.discretization.face_measures[None, :]
        face_centers = scale[:, None, None] * self.discretization.face_centers[None, :, :]
        reference_normals = self.discretization.area_vectors / jnp.maximum(
            self.discretization.face_measures[:, None], 1.0e-30
        )
        face_normals = jnp.broadcast_to(
            reference_normals[None, :, :], (count, self.face_capacity, dimension)
        )
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        boundary = neighbour < 0
        safe_neighbour = jnp.maximum(neighbour, 0)
        interior_distance = jnp.linalg.norm(
            cell_centers[:, safe_neighbour, :] - cell_centers[:, owner, :], axis=-1
        )
        boundary_distance = jnp.linalg.norm(
            face_centers - cell_centers[:, owner, :], axis=-1
        )
        center_distances = jnp.where(
            boundary[None, :], boundary_distance, interior_distance
        )
        active_faces = cells_active[:, owner] & jnp.where(
            boundary[None, :], True, cells_active[:, safe_neighbour]
        )
        surface_measure = jnp.sum(
            jnp.where(boundary[None, :] & active_faces, face_measures, 0.0), axis=1
        )
        successful = (
            jnp.all(jnp.isfinite(scale) & (scale > 0.0))
            & jnp.all(
                ~cells_active | (jnp.isfinite(cell_measures) & (cell_measures > 0.0))
            )
            & jnp.all(
                ~active_faces | (jnp.isfinite(face_measures) & (face_measures > 0.0))
            )
            & jnp.all(~active_faces | (center_distances > 0.0))
        )
        return ParticleInternalMeshMetrics(
            cell_measures,
            cell_centers,
            face_measures,
            face_centers,
            face_normals,
            center_distances,
            owner,
            neighbour,
            boundary,
            cells_active,
            active_faces,
            surface_measure,
            successful,
            self.prepared_id,
        )

    def boundary_trace(
        self,
        metrics: ParticleInternalMeshMetrics,
        position: ArrayLike,
        orientation: ArrayLike,
        /,
    ) -> ParticleBoundaryTrace:
        center = jnp.asarray(position)
        quaternion = jnp.asarray(orientation)
        count = metrics.cell_measures.shape[0]
        if center.shape != (count, 3) or quaternion.shape != (count, 4):
            raise ValueError(
                "Boundary trace poses must have particle three-dimensional shape."
            )
        if self.dimension != 3:
            raise ValueError(
                "World boundary traces currently require a 3-D internal mesh."
            )
        rotation = quaternion_rotation_matrix(quaternion)
        world_position = center[:, None, :] + contract(
            "pij,pfj->pfi", rotation, metrics.face_centers
        )
        world_normal = contract("pij,pfj->pfi", rotation, metrics.face_normals)
        boundary_active = metrics.boundary_faces[None, :] & metrics.active_faces
        return ParticleBoundaryTrace(
            metrics.owner_cells,
            world_position,
            world_normal,
            metrics.face_measures,
            boundary_active,
            self.prepared_id,
        )


__all__ = [
    "ParticleBoundaryTrace",
    "ParticleInternalMeshMetrics",
    "PreparedUnstructuredParticleInternalMesh",
    "UnstructuredParticleInternalMeshPlan",
]
