#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..finite_volume import FaceVelocity
from ..particle import ParticlePopulationState
from ._transfer import PreparedFLIPParticleTransfer
from ._types import FLIPParticleState


class MultiphaseFLIPState(StrictModule):
    particles: FLIPParticleState
    population: ParticlePopulationState
    phase_id: Array
    pressure: Array


class MultiphaseFLIPTransferResult(StrictModule):
    phase_fraction: Array
    mixture_density: Array
    mixture_viscosity: Array
    face_mass: FaceVelocity
    face_momentum: FaceVelocity
    velocity: FaceVelocity
    phase_face_mass: tuple[Array, ...]
    phase_face_momentum: tuple[Array, ...]
    phase_velocity: tuple[Array, ...]
    face_inverse_density: FaceVelocity
    phase_mass: Array
    phase_volume: Array
    mass_defect: Array
    pairwise_impulse: Array
    pairwise_work: Array
    momentum_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MultiphaseFLIPPlan(StrictModule, NonTrainableState):
    """Finite-phase multivelocity incompressible FLIP reconstruction."""

    transfer: PreparedFLIPParticleTransfer
    densities: Array
    viscosities: Array
    surface_tension: Array
    drag: Array
    phase_count: int = eqx.field(static=True)
    maximum_phases: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedFLIPParticleTransfer,
        densities: ArrayLike,
        viscosities: ArrayLike,
        surface_tension: ArrayLike,
        drag: ArrayLike | None = None,
        maximum_phases: int | None = None,
        /,
    ):
        if not isinstance(transfer, PreparedFLIPParticleTransfer):
            raise TypeError("transfer must be PreparedFLIPParticleTransfer.")
        rho = np.asarray(densities, dtype=float)
        mu = np.asarray(viscosities, dtype=float)
        sigma = np.asarray(surface_tension, dtype=float)
        if rho.ndim != 1 or rho.size < 1 or mu.shape != rho.shape:
            raise ValueError(
                "Multiphase FLIP densities/viscosities must be phase vectors."
            )
        phase_count = int(rho.size)
        maximum = phase_count if maximum_phases is None else int(maximum_phases)
        if phase_count > maximum or sigma.shape != (phase_count, phase_count):
            raise ValueError("Multiphase material arrays exceed maximum_phases.")
        drag_ = np.zeros_like(sigma) if drag is None else np.asarray(drag, dtype=float)
        if drag_.shape != sigma.shape:
            raise ValueError("Drag matrix must match the phase-pair shape.")
        if (
            np.any(rho <= 0.0)
            or np.any(mu < 0.0)
            or np.any(sigma < 0.0)
            or np.any(drag_ < 0.0)
            or not np.all(np.isfinite(rho))
            or not np.all(np.isfinite(mu))
        ):
            raise ValueError("Multiphase material values are invalid.")
        if (
            not np.allclose(sigma, sigma.T)
            or not np.allclose(np.diag(sigma), 0.0)
            or not np.allclose(drag_, drag_.T)
            or not np.allclose(np.diag(drag_), 0.0)
        ):
            raise ValueError("Pair matrices must be symmetric with zero diagonal.")
        self.transfer = transfer
        self.densities = jnp.asarray(rho)
        self.viscosities = jnp.asarray(mu)
        self.surface_tension = jnp.asarray(sigma)
        self.drag = jnp.asarray(drag_)
        self.phase_count = phase_count
        self.maximum_phases = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiphase-flip",
                "transfer": transfer.prepared_id,
                "densities": array_tree_fingerprint(rho),
                "viscosities": array_tree_fingerprint(mu),
                "surface_tension": array_tree_fingerprint(sigma),
                "drag": array_tree_fingerprint(drag_),
                "maximum_phases": maximum,
            }
        )

    def evaluate(
        self, state: MultiphaseFLIPState, /, *, step_size: ArrayLike = 1.0
    ) -> MultiphaseFLIPTransferResult:
        phase = jnp.asarray(state.phase_id, dtype=jnp.int32)
        if phase.shape != state.population.active.shape:
            raise ValueError("phase_id must preserve particle capacity.")
        dt = jnp.asarray(step_size, dtype=state.particles.position.dtype).reshape(())
        phase = eqx.error_if(
            phase,
            jnp.any(
                state.population.active & ((phase < 0) | (phase >= self.phase_count))
            ),
            "Active FLIP particle phase_id is outside the prepared phase count.",
        )
        cell_volumes = []
        phase_masses = []
        face_mass_by_phase = []
        face_momentum_by_phase = []
        successful = jnp.asarray(True)
        for phase_index in range(self.phase_count):
            active = state.population.active & (phase == phase_index)
            routes = self.transfer.build(state.particles.position, active_mask=active)
            result = self.transfer.particle_to_grid(
                routes,
                state.particles.velocity,
                self.densities[phase_index],
                masses=state.population.mass,
            )
            cell_volumes.append(result.particle_volume_content)
            phase_masses.append(jnp.sum(jnp.where(active, state.population.mass, 0.0)))
            face_mass_by_phase.append(result.face_mass)
            face_momentum_by_phase.append(result.face_momentum)
            successful = successful & result.successful
        volume_stack = jnp.stack(tuple(cell_volumes), axis=-1)
        cell_measure = self.transfer.plan.operators.discretization.cell_volumes
        phase_fraction = volume_stack / cell_measure[..., None]
        total_fraction = jnp.sum(phase_fraction, axis=-1)
        normalized = phase_fraction / jnp.maximum(total_fraction[..., None], 1.0e-30)
        mixture_density = jnp.sum(normalized * self.densities, axis=-1)
        mixture_viscosity = jnp.sum(normalized * self.viscosities, axis=-1)
        phase_velocity = []
        corrected_phase_momentum = []
        pairwise_impulse = []
        pairwise_work = []
        for axis in range(self.transfer.dimension):
            masses = jnp.stack(tuple(value[axis] for value in face_mass_by_phase), axis=0)
            momenta = jnp.stack(
                tuple(value[axis] for value in face_momentum_by_phase), axis=0
            )
            velocities = jnp.where(
                masses > jnp.finfo(masses.dtype).eps,
                momenta / jnp.where(masses > 0.0, masses, 1.0),
                0.0,
            )
            axis_impulses = jnp.zeros(
                (self.phase_count, self.phase_count, *masses.shape[1:]),
                dtype=masses.dtype,
            )
            axis_work = jnp.zeros(
                (self.phase_count, self.phase_count), dtype=masses.dtype
            )
            for left in range(self.phase_count):
                for right in range(left + 1, self.phase_count):
                    left_mass = masses[left]
                    right_mass = masses[right]
                    supported = (left_mass > 0.0) & (right_mass > 0.0)
                    coefficient = dt * self.drag[left, right]
                    denominator = 1.0 + coefficient * (
                        1.0 / jnp.where(supported, left_mass, 1.0)
                        + 1.0 / jnp.where(supported, right_mass, 1.0)
                    )
                    impulse = jnp.where(
                        supported,
                        coefficient
                        * (velocities[right] - velocities[left])
                        / denominator,
                        0.0,
                    )
                    energy_before = 0.5 * (
                        jnp.where(left_mass > 0.0, momenta[left] ** 2 / left_mass, 0.0)
                        + jnp.where(
                            right_mass > 0.0, momenta[right] ** 2 / right_mass, 0.0
                        )
                    )
                    momenta = momenta.at[left].add(impulse)
                    momenta = momenta.at[right].add(-impulse)
                    velocities = jnp.where(
                        masses > 0.0,
                        momenta / jnp.where(masses > 0.0, masses, 1.0),
                        0.0,
                    )
                    energy_after = 0.5 * (
                        jnp.where(left_mass > 0.0, momenta[left] ** 2 / left_mass, 0.0)
                        + jnp.where(
                            right_mass > 0.0, momenta[right] ** 2 / right_mass, 0.0
                        )
                    )
                    axis_impulses = axis_impulses.at[left, right].set(impulse)
                    axis_impulses = axis_impulses.at[right, left].set(-impulse)
                    axis_work = axis_work.at[left, right].set(
                        jnp.sum(energy_after - energy_before)
                    )
                    axis_work = axis_work.at[right, left].set(axis_work[left, right])
            phase_velocity.append(velocities)
            corrected_phase_momentum.append(momenta)
            pairwise_impulse.append(axis_impulses)
            pairwise_work.append(axis_work)

        face_mass = []
        face_momentum = []
        velocity = []
        inverse_density = []
        for axis in range(self.transfer.dimension):
            mass = jnp.sum(
                jnp.stack(tuple(value[axis] for value in face_mass_by_phase)), axis=0
            )
            momentum = jnp.sum(corrected_phase_momentum[axis], axis=0)
            supported = mass > jnp.finfo(mass.dtype).eps
            face_mass.append(mass)
            face_momentum.append(momentum)
            velocity.append(
                jnp.where(supported, momentum / jnp.where(supported, mass, 1.0), 0.0)
            )
            phase_volume_face = tuple(
                face_mass_by_phase[index][axis] / self.densities[index]
                for index in range(self.phase_count)
            )
            total_volume_face = sum(phase_volume_face)
            density_face = mass / jnp.maximum(total_volume_face, 1.0e-30)
            inverse_density.append(
                jnp.where(
                    supported,
                    1.0 / jnp.maximum(density_face, 1.0e-30),
                    1.0 / jnp.min(self.densities),
                )
            )
        deposited_mass = jnp.sum(jnp.asarray(phase_masses))
        active_mass = jnp.sum(
            jnp.where(state.population.active, state.population.mass, 0.0)
        )
        mass_defect = deposited_mass - active_mass
        impulse_stack = jnp.stack(tuple(pairwise_impulse), axis=0)
        work_stack = jnp.stack(tuple(pairwise_work), axis=0)
        momentum_defect = jnp.max(
            jnp.abs(jnp.sum(impulse_stack, axis=(1, 2))), initial=0.0
        )
        finite = (
            jnp.all(jnp.isfinite(phase_fraction))
            & jnp.all(jnp.isfinite(mixture_density))
            & jnp.all(jnp.isfinite(mixture_viscosity))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in velocity))
            )
            & jnp.all(jnp.isfinite(impulse_stack))
            & jnp.all(work_stack <= 128.0 * jnp.finfo(work_stack.dtype).eps)
        )
        return MultiphaseFLIPTransferResult(
            phase_fraction,
            mixture_density,
            mixture_viscosity,
            tuple(face_mass),
            tuple(face_momentum),
            tuple(velocity),
            tuple(
                jnp.stack(tuple(value[axis] for value in face_mass_by_phase), axis=0)
                for axis in range(self.transfer.dimension)
            ),
            tuple(corrected_phase_momentum),
            tuple(phase_velocity),
            tuple(inverse_density),
            jnp.asarray(phase_masses),
            jnp.asarray([jnp.sum(value) for value in cell_volumes]),
            mass_defect,
            impulse_stack,
            work_stack,
            momentum_defect,
            finite,
            successful & finite,
            self.plan_id,
        )


__all__ = [
    "MultiphaseFLIPPlan",
    "MultiphaseFLIPState",
    "MultiphaseFLIPTransferResult",
]
