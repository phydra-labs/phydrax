#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization


class AbstractParticleInternalMeshPlan(StrictModule, NonTrainableState):
    mesh_id: AbstractAttribute[str]
    cell_capacity: AbstractAttribute[int]

    @abc.abstractmethod
    def prepare(self):
        raise NotImplementedError


class AbstractPreparedParticleInternalMesh(StrictModule, NonTrainableState):
    prepared_id: AbstractAttribute[str]
    cell_capacity: AbstractAttribute[int]

    @abc.abstractmethod
    def metrics(self, outer_scale: ArrayLike, /):
        raise NotImplementedError


class ParticleInternalGeometry(StrEnum):
    SLAB = "slab"
    CYLINDER = "cylinder"
    SPHERE = "sphere"


class ParticleShellMetrics(StrictModule):
    face_coordinates: Array
    cell_coordinates: Array
    cell_measures: Array
    face_measures: Array
    center_distances: Array
    surface_measure: Array
    successful: Array
    mesh_id: str = eqx.field(static=True)


class RadialShellMeshPlan(AbstractParticleInternalMeshPlan):
    geometry: ParticleInternalGeometry = eqx.field(static=True)
    reference_faces: Array
    transverse_measure: float = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: ParticleInternalGeometry,
        cell_count: int,
        /,
        *,
        reference_faces: ArrayLike | None = None,
        transverse_measure: float = 1.0,
        mesh_id: str | None = None,
    ):
        if not isinstance(geometry, ParticleInternalGeometry):
            raise TypeError("geometry must be a ParticleInternalGeometry.")
        count = int(cell_count)
        transverse = float(transverse_measure)
        if count <= 0 or not np.isfinite(transverse) or transverse <= 0.0:
            raise ValueError("Cell count and transverse measure must be positive.")
        faces = (
            np.linspace(0.0, 1.0, count + 1)
            if reference_faces is None
            else np.asarray(reference_faces, dtype=float)
        )
        if (
            faces.shape != (count + 1,)
            or np.any(~np.isfinite(faces))
            or faces[0] != 0.0
            or faces[-1] != 1.0
            or np.any(np.diff(faces) <= 0.0)
        ):
            raise ValueError("reference_faces must increase strictly from zero to one.")
        generated = canonical_fingerprint(
            {
                "kind": "radial-shell-mesh-plan",
                "geometry": geometry.value,
                "faces": array_tree_fingerprint(faces),
                "transverse_measure": transverse,
            }
        )
        self.geometry = geometry
        self.reference_faces = jnp.asarray(faces)
        self.transverse_measure = transverse
        self.cell_count = count
        self.mesh_id = generated if mesh_id is None else str(mesh_id)
        if not self.mesh_id:
            raise ValueError("mesh_id must be nonempty.")

    @property
    def cell_capacity(self) -> int:
        return self.cell_count

    def prepare(self) -> PreparedRadialShellMesh:
        return PreparedRadialShellMesh(self)


class PreparedRadialShellMesh(AbstractPreparedParticleInternalMesh):
    plan: RadialShellMeshPlan
    reference_faces: Array
    reference_cells: Array
    reference_widths: Array
    geometry_exponent: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: RadialShellMeshPlan, /):
        if not isinstance(plan, RadialShellMeshPlan):
            raise TypeError("plan must be a RadialShellMeshPlan.")
        exponent = {
            ParticleInternalGeometry.SLAB: 0,
            ParticleInternalGeometry.CYLINDER: 1,
            ParticleInternalGeometry.SPHERE: 2,
        }[plan.geometry]
        self.plan = plan
        self.reference_faces = plan.reference_faces
        self.reference_cells = 0.5 * (
            plan.reference_faces[1:] + plan.reference_faces[:-1]
        )
        self.reference_widths = jnp.diff(plan.reference_faces)
        self.geometry_exponent = exponent
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-radial-shell-mesh", "plan": plan.mesh_id}
        )

    @property
    def cell_capacity(self) -> int:
        return self.plan.cell_capacity

    def metrics(self, outer_scale: ArrayLike, /) -> ParticleShellMetrics:
        scale = jnp.asarray(outer_scale)
        if scale.ndim != 1:
            raise ValueError("outer_scale must be a rank-1 particle-batch array.")
        face = scale[:, None] * self.reference_faces[None, :]
        cell = scale[:, None] * self.reference_cells[None, :]
        exponent = self.geometry_exponent
        if exponent == 0:
            face_measure = jnp.full_like(face, self.plan.transverse_measure)
            face_measure = face_measure.at[:, 0].set(0.0)
            cell_measure = self.plan.transverse_measure * (face[:, 1:] - face[:, :-1])
        elif exponent == 1:
            face_measure = 2.0 * jnp.pi * self.plan.transverse_measure * face
            cell_measure = (
                jnp.pi
                * self.plan.transverse_measure
                * (face[:, 1:] ** 2 - face[:, :-1] ** 2)
            )
        else:
            face_measure = 4.0 * jnp.pi * face**2
            cell_measure = (4.0 / 3.0) * jnp.pi * (face[:, 1:] ** 3 - face[:, :-1] ** 3)
        center_distance = cell[:, 1:] - cell[:, :-1]
        successful = (
            jnp.all(jnp.isfinite(scale) & (scale > 0.0))
            & jnp.all(jnp.isfinite(cell_measure) & (cell_measure > 0.0))
            & jnp.all(jnp.isfinite(face_measure) & (face_measure >= 0.0))
            & jnp.all(center_distance > 0.0)
        )
        return ParticleShellMetrics(
            face,
            cell,
            cell_measure,
            face_measure,
            center_distance,
            face_measure[:, -1],
            successful,
            self.prepared_id,
        )


class ParticleInternalBatchPlan(StrictModule, NonTrainableState):
    owner_indices: Array
    mesh: AbstractParticleInternalMeshPlan
    species_count: int = eqx.field(static=True)
    front_count: int = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner_indices: ArrayLike,
        mesh: AbstractParticleInternalMeshPlan,
        species_count: int,
        /,
        *,
        front_count: int = 0,
        batch_id: str | None = None,
    ):
        owners = np.asarray(owner_indices)
        species = int(species_count)
        fronts = int(front_count)
        if (
            owners.ndim != 1
            or owners.size == 0
            or not np.issubdtype(owners.dtype, np.integer)
            or np.any(owners < 0)
            or len(set(int(value) for value in owners)) != owners.size
        ):
            raise ValueError("owner_indices must be unique nonnegative integers.")
        if not isinstance(mesh, AbstractParticleInternalMeshPlan):
            raise TypeError("mesh must be an AbstractParticleInternalMeshPlan.")
        if species <= 0 or fronts < 0:
            raise ValueError(
                "species_count must be positive and front_count nonnegative."
            )
        generated = canonical_fingerprint(
            {
                "kind": "particle-internal-batch-plan",
                "owners": array_tree_fingerprint(owners),
                "mesh": mesh.mesh_id,
                "species_count": species,
                "front_count": fronts,
            }
        )
        self.owner_indices = jnp.asarray(owners, dtype=jnp.int32)
        self.mesh = mesh
        self.species_count = species
        self.front_count = fronts
        self.batch_id = generated if batch_id is None else str(batch_id)
        if not self.batch_id:
            raise ValueError("batch_id must be nonempty.")

    def prepare(self, particles: ParticleDiscretization, /):
        return PreparedParticleInternalBatch(self, particles)


class PreparedParticleInternalBatch(StrictModule, NonTrainableState):
    plan: ParticleInternalBatchPlan
    particles: ParticleDiscretization
    mesh: AbstractPreparedParticleInternalMesh
    owner_indices: Array
    active: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: ParticleInternalBatchPlan, particles: ParticleDiscretization, /
    ):
        if not isinstance(plan, ParticleInternalBatchPlan):
            raise TypeError("plan must be a ParticleInternalBatchPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if np.any(np.asarray(plan.owner_indices) >= particles.capacity):
            raise ValueError("Particle-internal owner index exceeds particle capacity.")
        self.plan = plan
        self.particles = particles
        self.mesh = plan.mesh.prepare()
        self.owner_indices = plan.owner_indices
        self.active = particles.active_mask[plan.owner_indices]
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-particle-internal-batch",
                "plan": plan.batch_id,
                "particles": particles.prepared_id,
                "mesh": self.mesh.prepared_id,
            }
        )

    @property
    def particle_count(self) -> int:
        return int(self.owner_indices.shape[0])

    @property
    def cell_capacity(self) -> int:
        return self.mesh.cell_capacity

    @property
    def species_count(self) -> int:
        return self.plan.species_count

    @property
    def front_count(self) -> int:
        return self.plan.front_count

    def gather_owner_values(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[0] != self.particles.capacity:
            raise ValueError("Owner values must use outer particle capacity.")
        return array[self.owner_indices]

    def scatter_owner_values(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[0] != self.particle_count:
            raise ValueError("Batch values must use internal particle count.")
        result = jnp.zeros(
            (self.particles.capacity,) + array.shape[1:],
            dtype=array.dtype,
        )
        return result.at[self.owner_indices].add(array)


__all__ = [
    "AbstractParticleInternalMeshPlan",
    "AbstractPreparedParticleInternalMesh",
    "ParticleInternalBatchPlan",
    "ParticleInternalGeometry",
    "ParticleShellMetrics",
    "PreparedParticleInternalBatch",
    "PreparedRadialShellMesh",
    "RadialShellMeshPlan",
]
