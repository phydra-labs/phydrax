#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import ParticlePopulationState
from ..discretization.pic import (
    PICChargeModelPlan,
    PICChargeState,
    PICParticleState,
    RelativisticBorisPlan,
    UnstructuredWhitneyCurrentPlan,
)
from ._maxwell import CompatibleMaxwellState
from ._maxwell_unstructured import PreparedUnstructuredMaxwell


class UnstructuredElectromagneticPICState(StrictModule):
    particles: PICParticleState
    population: ParticlePopulationState
    charge: PICChargeState
    maxwell: CompatibleMaxwellState
    time: Array


class UnstructuredElectromagneticPICResult(StrictModule):
    candidate_state: UnstructuredElectromagneticPICState
    accepted_state: UnstructuredElectromagneticPICState
    continuity_defect: Array
    electric_constraint: Array
    magnetic_constraint: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class UnstructuredElectromagneticPICPlan(StrictModule, NonTrainableState):
    maxwell: PreparedUnstructuredMaxwell
    current: UnstructuredWhitneyCurrentPlan
    charge_model: PICChargeModelPlan
    pusher: RelativisticBorisPlan
    gradients: Array
    face_reconstruction: Array
    cell_faces: Array
    cell_face_signs: Array
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maxwell: PreparedUnstructuredMaxwell,
        current: UnstructuredWhitneyCurrentPlan,
        charge_model: PICChargeModelPlan,
        /,
        *,
        pusher: RelativisticBorisPlan | None = None,
        tolerance: float = 1.0e-8,
    ):
        if not isinstance(maxwell, PreparedUnstructuredMaxwell):
            raise TypeError("maxwell must be PreparedUnstructuredMaxwell.")
        if not isinstance(current, UnstructuredWhitneyCurrentPlan):
            raise TypeError("current must be UnstructuredWhitneyCurrentPlan.")
        if current.locator.dimension != 3:
            raise ValueError("Unstructured electromagnetic PIC requires tetrahedra.")
        if maxwell.plan.cochain.cell_counts[1] != current.edges.shape[0]:
            raise ValueError(
                "Whitney current edge space differs from Maxwell degree one."
            )
        cells = np.asarray(current.locator.cells, dtype=np.int32)
        coordinates = np.asarray(current.locator.mesh.coordinates, dtype=float)
        gradients = []
        face_map: dict[tuple[int, int, int], int] = {}
        cell_faces = []
        cell_signs = []
        reconstruction = []
        local_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
        for cell in cells:
            vertices = coordinates[cell]
            inverse = np.linalg.inv((vertices[1:] - vertices[0]).T)
            grad = np.concatenate(
                (-np.sum(inverse, axis=0, keepdims=True), inverse), axis=0
            )
            gradients.append(grad)
            local_ids, local_sign = [], []
            normal_rows = []
            for face_local in local_faces:
                oriented = tuple(int(cell[index]) for index in face_local)
                canonical = tuple(sorted(oriented))
                if canonical not in face_map:
                    face_map[canonical] = len(face_map)
                local_ids.append(face_map[canonical])
                permutation = [canonical.index(value) for value in oriented]
                inversions = sum(
                    permutation[i] > permutation[j]
                    for i in range(3)
                    for j in range(i + 1, 3)
                )
                local_sign.append(-1 if inversions % 2 else 1)
                a, b, c = coordinates[list(oriented)]
                normal_rows.append(0.5 * np.cross(b - a, c - a))
            cell_faces.append(local_ids)
            cell_signs.append(local_sign)
            reconstruction.append(np.linalg.pinv(np.asarray(normal_rows)))
        if maxwell.plan.cochain.cell_counts[2] != len(face_map):
            raise ValueError("Whitney face ordering differs from Maxwell degree two.")
        self.maxwell = maxwell
        self.current = current
        self.charge_model = charge_model
        self.pusher = RelativisticBorisPlan() if pusher is None else pusher
        self.gradients = jnp.asarray(gradients)
        self.face_reconstruction = jnp.asarray(reconstruction)
        self.cell_faces = jnp.asarray(cell_faces, dtype=jnp.int32)
        self.cell_face_signs = jnp.asarray(cell_signs)
        self.tolerance = float(tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-electromagnetic-pic",
                "maxwell": maxwell.prepared_id,
                "current": current.plan_id,
                "charge_model": charge_model.plan_id,
                "pusher": self.pusher.plan_id,
            }
        )

    def gather_fields(
        self,
        particles: PICParticleState,
        state: CompatibleMaxwellState,
        /,
    ) -> tuple[Array, Array, Array]:
        location = self.current.locator.locate(particles.position)
        cell = jnp.maximum(location.cell_ids, 0)
        electric_cochain = self.maxwell.electric_field(state)
        local_edges = self.current.cell_edges[cell]
        local_signs = self.current.cell_edge_signs[cell]
        local_pairs = tuple(itertools.combinations(range(4), 2))
        electric = jnp.zeros(
            (particles.position.shape[0], 3), dtype=particles.position.dtype
        )
        for local_index, (a, b) in enumerate(local_pairs):
            whitney = (
                location.barycentric[:, a, None] * self.gradients[cell, b]
                - location.barycentric[:, b, None] * self.gradients[cell, a]
            )
            coefficient = (
                electric_cochain[local_edges[:, local_index]]
                * local_signs[:, local_index]
            )
            electric = electric + coefficient[:, None] * whitney
        magnetic_flux = (
            state.primary.magnetic_flux[self.cell_faces[cell]]
            * self.cell_face_signs[cell]
        )
        magnetic = contract("pij,pj->pi", self.face_reconstruction[cell], magnetic_flux)
        return electric, magnetic, location.successful

    def step(
        self,
        state: UnstructuredElectromagneticPICState,
        step_size: ArrayLike,
        /,
    ) -> UnstructuredElectromagneticPICResult:
        dt = jnp.asarray(step_size, dtype=state.time.dtype).reshape(())
        electric, magnetic, located = self.gather_fields(state.particles, state.maxwell)
        specific = (
            self.charge_model.base_specific_charge
            * state.charge.charge_number.astype(state.population.mass.dtype)
        )
        pushed = self.pusher.push(
            state.particles.proper_velocity,
            electric,
            magnetic,
            specific,
            state.population.active,
            dt,
        )
        position = state.particles.position + dt * pushed.velocity
        macrocharge = self.charge_model.macrocharge(state.population, state.charge)
        deposited = self.current.deposit(
            state.particles.position,
            position,
            macrocharge,
            state.population.active,
            dt,
        )
        maxwell = self.maxwell.step(
            state.time,
            state.maxwell,
            dt,
            electric_current=deposited.edge_current,
        )
        constraints = self.maxwell.constraints(maxwell)
        electric_constraint = jnp.max(jnp.abs(constraints[0]), initial=0.0)
        magnetic_constraint = jnp.max(jnp.abs(constraints[1]), initial=0.0)
        finite = (
            jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(pushed.proper_velocity))
            & jnp.isfinite(electric_constraint + magnetic_constraint)
        )
        successful = (
            located.all()
            & pushed.successful
            & deposited.successful
            & finite
            & (electric_constraint <= self.tolerance)
            & (magnetic_constraint <= self.tolerance)
        )
        candidate = UnstructuredElectromagneticPICState(
            PICParticleState(position, pushed.proper_velocity),
            state.population,
            state.charge,
            maxwell,
            state.time + dt,
        )
        accepted = jax.tree.map(
            lambda proposed, old: jnp.where(successful, proposed, old), candidate, state
        )
        return UnstructuredElectromagneticPICResult(
            candidate,
            accepted,
            deposited.maximum_continuity_defect,
            electric_constraint,
            magnetic_constraint,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "UnstructuredElectromagneticPICPlan",
    "UnstructuredElectromagneticPICResult",
    "UnstructuredElectromagneticPICState",
]
