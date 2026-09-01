#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import AbstractFixedStepMethod, FixedStepResult
from ._free_surface_ale import FaceTuple
from ._free_surface_step import (
    FreeSurfaceALEContinuationState,
    OnePhaseFreeSurfaceALEMethod,
    PreparedOnePhaseFreeSurfaceALE,
)


def _quaternion_matrix(quaternion: Array) -> Array:
    q = quaternion / jnp.linalg.norm(quaternion)
    w, x, y, z = q
    return jnp.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


class RigidBodyState(StrictModule):
    position: Array
    quaternion: Array
    linear_velocity: Array
    angular_velocity: Array


class HydroelasticBodyState(StrictModule):
    rigid: RigidBodyState
    modal_coordinates: Array
    modal_velocity: Array


class MappedMarkerTransferEvidence(StrictModule):
    gathered_normal_velocity: Array
    adjoint_defect: Array
    route_minimum_weight: Array
    finite: Array
    valid: Array
    transfer_id: str = eqx.field(static=True)


class BodyCouplingEvidence(StrictModule):
    constraint_residual: Array
    force: Array
    torque: Array
    fluid_work: Array
    body_work: Array
    modal_work: Array
    power_defect: Array
    viscous_dissipation: Array
    finite: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MappedRigidHydroelasticBodyPlan(StrictModule, NonTrainableState):
    """Fully submerged mapped normal-constraint rigid/modal coupling."""

    reference_markers: Array
    reference_normals: Array
    marker_weights: Array
    mass: float = eqx.field(static=True)
    inertia: Array
    moving: bool = eqx.field(static=True)
    viscous_drag: float = eqx.field(static=True)
    modal_basis: Array
    modal_mass: Array
    modal_stiffness: Array
    modal_damping: Array
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_markers: ArrayLike,
        reference_normals: ArrayLike,
        marker_weights: ArrayLike,
        /,
        *,
        mass: float = 1.0,
        inertia: ArrayLike = (1.0, 1.0, 1.0),
        moving: bool = True,
        viscous_drag: float = 0.0,
        modal_basis: ArrayLike | None = None,
        modal_mass: ArrayLike | None = None,
        modal_stiffness: ArrayLike | None = None,
        modal_damping: ArrayLike | None = None,
        tolerance: float = 1.0e-9,
    ):
        markers = jnp.asarray(reference_markers)
        normals = jnp.asarray(reference_normals, dtype=markers.dtype)
        weights = jnp.asarray(marker_weights, dtype=markers.dtype)
        if markers.ndim != 2 or markers.shape[1] != 3:
            raise ValueError("Body markers require shape (markers, 3).")
        if normals.shape != markers.shape or weights.shape != markers.shape[:1]:
            raise ValueError("Body marker normals/weights have invalid shapes.")
        norms = jnp.linalg.norm(normals, axis=-1)
        if (
            bool(jnp.any(~jnp.isfinite(markers)))
            or bool(jnp.any(~jnp.isfinite(normals)))
            or bool(jnp.any(weights <= 0.0))
            or bool(jnp.any(norms <= 0.0))
        ):
            raise ValueError("Body marker geometry must be finite and nondegenerate.")
        normals = normals / norms[:, None]
        mass_ = float(mass)
        inertia_ = jnp.asarray(inertia, dtype=markers.dtype)
        drag = float(viscous_drag)
        tolerance_ = float(tolerance)
        if (
            mass_ <= 0.0
            or inertia_.shape != (3,)
            or bool(jnp.any(inertia_ <= 0.0))
            or drag < 0.0
            or tolerance_ <= 0.0
        ):
            raise ValueError("Body mass/inertia/drag/tolerance are invalid.")
        basis = (
            jnp.zeros((markers.shape[0], 0), dtype=markers.dtype)
            if modal_basis is None
            else jnp.asarray(modal_basis, dtype=markers.dtype)
        )
        if basis.ndim != 2 or basis.shape[0] != markers.shape[0]:
            raise ValueError("Modal basis must have shape (markers, modes).")
        modes = basis.shape[1]
        modal_mass_ = _modal_vector(modal_mass, modes, 1.0, markers.dtype)
        modal_stiffness_ = _modal_vector(modal_stiffness, modes, 0.0, markers.dtype)
        modal_damping_ = _modal_vector(modal_damping, modes, 0.0, markers.dtype)
        self.reference_markers = markers
        self.reference_normals = normals
        self.marker_weights = weights
        self.mass = mass_
        self.inertia = inertia_
        self.moving = bool(moving)
        self.viscous_drag = drag
        self.modal_basis = basis
        self.modal_mass = modal_mass_
        self.modal_stiffness = modal_stiffness_
        self.modal_damping = modal_damping_
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-rigid-hydroelastic-body-plan",
                "markers": array_tree_fingerprint(np.asarray(markers)),
                "normals": array_tree_fingerprint(np.asarray(normals)),
                "weights": array_tree_fingerprint(np.asarray(weights)),
                "mass": mass_,
                "inertia": np.asarray(inertia_).tolist(),
                "moving": bool(moving),
                "viscous_drag": drag,
                "modal_basis": array_tree_fingerprint(np.asarray(basis)),
                "modal_mass": np.asarray(modal_mass_).tolist(),
                "modal_stiffness": np.asarray(modal_stiffness_).tolist(),
                "modal_damping": np.asarray(modal_damping_).tolist(),
            }
        )

    def initial_state(
        self,
        *,
        position: ArrayLike = (0.0, 0.0, 0.0),
        quaternion: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        linear_velocity: ArrayLike = (0.0, 0.0, 0.0),
        angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
    ) -> HydroelasticBodyState:
        rigid = RigidBodyState(
            jnp.asarray(position, dtype=self.reference_markers.dtype),
            jnp.asarray(quaternion, dtype=self.reference_markers.dtype),
            jnp.asarray(linear_velocity, dtype=self.reference_markers.dtype),
            jnp.asarray(angular_velocity, dtype=self.reference_markers.dtype),
        )
        return HydroelasticBodyState(
            rigid,
            jnp.zeros_like(self.modal_mass),
            jnp.zeros_like(self.modal_mass),
        )

    def geometry(self, state: HydroelasticBodyState, /):
        rotation = _quaternion_matrix(state.rigid.quaternion)
        markers = state.rigid.position + self.reference_markers @ rotation.T
        normals = self.reference_normals @ rotation.T
        arms = markers - state.rigid.position
        return markers, normals, arms

    def _interpolation_weights(self, cell_centers: Array, markers: Array, /) -> Array:
        flattened = cell_centers.reshape((-1, 3))
        distance = jnp.sum((markers[:, None, :] - flattened[None, :, :]) ** 2, axis=-1)
        scale = jnp.maximum(jnp.mean(distance, axis=-1, keepdims=True), 1.0e-12)
        weights = jnp.exp(-distance / scale)
        return weights / jnp.sum(weights, axis=-1, keepdims=True)

    def gather_normal_velocity(
        self,
        hydrodynamics: PreparedOnePhaseFreeSurfaceALE,
        geometry,
        velocity: FaceTuple,
        body: HydroelasticBodyState,
        /,
    ) -> tuple[Array, MappedMarkerTransferEvidence]:
        markers, normals, _ = self.geometry(body)
        cell_velocity = geometry.reconstruct_cell_velocity(velocity)
        weights = self._interpolation_weights(geometry.cell_centers, markers)
        gathered = weights @ cell_velocity.reshape((-1, 3))
        normal_velocity = jnp.sum(gathered * normals, axis=-1)

        def gather(candidate):
            cells = geometry.reconstruct_cell_velocity(candidate)
            sampled = weights @ cells.reshape((-1, 3))
            return jnp.sum(sampled * normals, axis=-1)

        probe = tuple(jnp.ones_like(component) for component in velocity)
        multiplier = jnp.ones_like(normal_velocity)
        spread = jax.linear_transpose(gather, velocity)(multiplier)[0]
        adjoint = jnp.real(jnp.vdot(gather(probe), multiplier)) - sum(
            jnp.real(jnp.vdot(component, covector))
            for component, covector in zip(probe, spread, strict=True)
        )
        finite = jnp.all(jnp.isfinite(normal_velocity)) & jnp.isfinite(adjoint)
        evidence = MappedMarkerTransferEvidence(
            gathered_normal_velocity=normal_velocity,
            adjoint_defect=adjoint,
            route_minimum_weight=jnp.min(weights),
            finite=finite,
            valid=finite & (jnp.abs(adjoint) <= self.tolerance),
            transfer_id=canonical_fingerprint(
                {
                    "kind": "mapped-marker-transfer",
                    "plan": self.plan_id,
                    "surface": hydrodynamics.surface.surface_id,
                }
            ),
        )
        return normal_velocity, evidence

    def couple(
        self,
        hydrodynamics: PreparedOnePhaseFreeSurfaceALE,
        geometry,
        momentum: FaceTuple,
        velocity: FaceTuple,
        body: HydroelasticBodyState,
        step_size: Array,
        /,
    ) -> tuple[FaceTuple, FaceTuple, HydroelasticBodyState, BodyCouplingEvidence]:
        markers, normals, arms = self.geometry(body)
        weights = self._interpolation_weights(geometry.cell_centers, markers)

        def gather(candidate):
            cells = geometry.reconstruct_cell_velocity(candidate)
            sampled = weights @ cells.reshape((-1, 3))
            return jnp.sum(sampled * normals, axis=-1)

        def spread(multiplier):
            return jax.linear_transpose(gather, velocity)(multiplier)[0]

        body_map = jnp.concatenate((normals, jnp.cross(arms, normals)), axis=-1)
        if not self.moving:
            body_inverse = jnp.zeros((6, 6), dtype=markers.dtype)
        else:
            body_inverse = jnp.diag(
                jnp.concatenate(
                    (
                        jnp.full((3,), 1.0 / self.mass),
                        1.0 / self.inertia,
                    )
                )
            )
        modal_response = (
            jnp.zeros((markers.shape[0], markers.shape[0]), dtype=markers.dtype)
            if self.modal_basis.shape[1] == 0
            else self.modal_basis @ jnp.diag(1.0 / self.modal_mass) @ self.modal_basis.T
        )
        body_normal_velocity = (
            body_map
            @ jnp.concatenate((body.rigid.linear_velocity, body.rigid.angular_velocity))
            + self.modal_basis @ body.modal_velocity
        )
        slip = gather(velocity) - body_normal_velocity

        def fluid_response(multiplier):
            covector = spread(multiplier)
            inverse = hydrodynamics.surface.inverse_hodge(geometry, covector)
            return gather(inverse.velocity)

        identity = jnp.eye(markers.shape[0], dtype=markers.dtype)
        fluid_matrix = jax.vmap(fluid_response)(identity).T
        rigid_matrix = body_map @ body_inverse @ body_map.T
        response = fluid_matrix + rigid_matrix + modal_response
        multiplier = jnp.linalg.solve(response + self.tolerance * identity, slip)
        fluid_covector = spread(multiplier)
        corrected_momentum = tuple(
            value - correction
            for value, correction in zip(momentum, fluid_covector, strict=True)
        )
        corrected_velocity = hydrodynamics.surface.inverse_hodge(
            geometry, corrected_momentum
        ).velocity
        body_impulse = body_map.T @ multiplier
        twist = (
            jnp.concatenate((body.rigid.linear_velocity, body.rigid.angular_velocity))
            + body_inverse @ body_impulse
        )
        modal_impulse = self.modal_basis.T @ multiplier
        modal_velocity = body.modal_velocity + jnp.where(
            self.modal_mass > 0.0,
            modal_impulse / self.modal_mass,
            0.0,
        )
        modal_acceleration = -(
            self.modal_stiffness * body.modal_coordinates
            + self.modal_damping * modal_velocity
        ) / jnp.where(self.modal_mass > 0.0, self.modal_mass, 1.0)
        modal_velocity = modal_velocity + step_size * modal_acceleration
        modal_coordinates = body.modal_coordinates + step_size * modal_velocity
        new_rigid = RigidBodyState(
            body.rigid.position + step_size * twist[:3],
            body.rigid.quaternion,
            twist[:3],
            twist[3:],
        )
        new_body = HydroelasticBodyState(new_rigid, modal_coordinates, modal_velocity)
        residual = gather(corrected_velocity) - (
            body_map @ twist + self.modal_basis @ modal_velocity
        )
        force = -jnp.sum(multiplier[:, None] * normals, axis=0)
        torque = -jnp.sum(multiplier[:, None] * jnp.cross(arms, normals), axis=0)
        fluid_work = -jnp.sum(multiplier * gather(corrected_velocity))
        body_work = jnp.dot(force, twist[:3]) + jnp.dot(torque, twist[3:])
        modal_work = jnp.dot(modal_impulse, modal_velocity)
        viscous_dissipation = self.viscous_drag * jnp.sum(residual**2)
        power_defect = fluid_work + body_work + modal_work
        finite = (
            jnp.all(jnp.isfinite(multiplier))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(power_defect)
        )
        converged = jnp.linalg.norm(residual) <= self.tolerance * jnp.maximum(
            jnp.linalg.norm(slip), 1.0
        )
        evidence = BodyCouplingEvidence(
            constraint_residual=jnp.linalg.norm(residual),
            force=force,
            torque=torque,
            fluid_work=fluid_work,
            body_work=body_work,
            modal_work=modal_work,
            power_defect=power_defect,
            viscous_dissipation=viscous_dissipation,
            finite=finite,
            converged=converged,
            successful=finite & converged,
            plan_id=self.plan_id,
        )
        return (
            corrected_momentum,
            corrected_velocity,
            new_body,
            evidence,
        )


class RigidHydroelasticContinuationState(StrictModule):
    fluid: FreeSurfaceALEContinuationState
    body: HydroelasticBodyState
    body_work: Array
    body_energy: Array


class RigidHydroelasticALEMethod(AbstractFixedStepMethod):
    """Mapped free-surface fluid step followed by monolithic marker/body KKT."""

    fluid_method: OnePhaseFreeSurfaceALEMethod
    body_plan: MappedRigidHydroelasticBodyPlan
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid_method: OnePhaseFreeSurfaceALEMethod,
        body_plan: MappedRigidHydroelasticBodyPlan,
        /,
    ):
        self.fluid_method = fluid_method
        self.body_plan = body_plan
        self.method_id = canonical_fingerprint(
            {
                "kind": "rigid-hydroelastic-ale-method",
                "fluid": fluid_method.method_id,
                "body": body_plan.plan_id,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: RigidHydroelasticContinuationState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        fluid_result = self.fluid_method.step(
            step_index, time, state.fluid, step_size, args
        )
        hydro = self.fluid_method.hydrodynamics
        fluid = fluid_result.accepted_state
        view = hydro.view(fluid.state, fluid.eta_rate, time + step_size)
        (
            momentum,
            velocity,
            body,
            evidence,
        ) = self.body_plan.couple(
            hydro,
            view.geometry,
            fluid.state.momentum,
            view.velocity,
            state.body,
            step_size,
        )
        del velocity
        coupled_state = type(fluid.state)(
            fluid.state.eta,
            momentum,
            fluid.state.scalar_content,
        )
        ledger = eqx.tree_at(
            lambda value: (
                value.body_work,
                value.body_energy_change,
            ),
            fluid.ledger,
            (
                fluid.ledger.body_work + step_size * evidence.body_work,
                fluid.ledger.body_energy_change
                + step_size * (evidence.body_work + evidence.modal_work),
            ),
        )
        coupled_fluid = eqx.tree_at(
            lambda value: (value.state, value.ledger),
            fluid,
            (coupled_state, ledger),
        )
        candidate = RigidHydroelasticContinuationState(
            coupled_fluid,
            body,
            state.body_work + step_size * evidence.body_work,
            state.body_energy + step_size * (evidence.body_work + evidence.modal_work),
        )
        successful = fluid_result.successful & evidence.successful
        accepted = jax.tree.map(
            lambda proposal, current: jnp.where(successful, proposal, current),
            candidate,
            state,
        )
        return FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=successful,
            residual=jnp.maximum(fluid_result.residual, evidence.constraint_residual),
            iterations=fluid_result.iterations + 1,
            work=fluid_result.work + 1,
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.asarray(0.0, dtype=step_size.dtype),
        )


def _modal_vector(value, size, default, dtype):
    if value is None:
        return jnp.full((size,), default, dtype=dtype)
    array = jnp.asarray(value, dtype=dtype)
    if array.shape != (size,) or bool(jnp.any(array <= 0.0)) and default > 0.0:
        raise ValueError("Modal coefficient shape/value is invalid.")
    return array


def write_rigid_hydroelastic_checkpoint(
    path: str | Path,
    method: RigidHydroelasticALEMethod,
    time: ArrayLike,
    accepted_step: ArrayLike,
    state: RigidHydroelasticContinuationState,
    /,
) -> Path:
    arrays: dict[str, object] = {
        "time": jnp.asarray(time),
        "accepted_step": jnp.asarray(accepted_step),
    }
    specification = pack_array_tree("state", state, arrays)
    return write_array_archive(
        path,
        manifest={
            "kind": "rigid-hydroelastic-free-surface-checkpoint",
            "method_id": method.method_id,
            "state": specification,
        },
        arrays=arrays,
    )


def read_rigid_hydroelastic_checkpoint(
    path: str | Path,
    method: RigidHydroelasticALEMethod,
    template: RigidHydroelasticContinuationState,
    /,
) -> tuple[Array, Array, RigidHydroelasticContinuationState]:
    manifest, arrays = read_array_archive(path)
    if manifest.get("kind") != "rigid-hydroelastic-free-surface-checkpoint":
        raise ValueError("Archive is not a rigid hydroelastic checkpoint.")
    if manifest.get("method_id") != method.method_id:
        raise ValueError("Rigid hydroelastic checkpoint method identity mismatch.")
    state = unpack_array_tree(manifest["state"], arrays, template)
    return (
        jnp.asarray(arrays["time"]),
        jnp.asarray(arrays["accepted_step"]),
        state,
    )


__all__ = [
    "BodyCouplingEvidence",
    "HydroelasticBodyState",
    "MappedMarkerTransferEvidence",
    "MappedRigidHydroelasticBodyPlan",
    "RigidBodyState",
    "RigidHydroelasticALEMethod",
    "RigidHydroelasticContinuationState",
    "read_rigid_hydroelastic_checkpoint",
    "write_rigid_hydroelastic_checkpoint",
]
