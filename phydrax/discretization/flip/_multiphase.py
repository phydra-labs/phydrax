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
    face_inverse_density: FaceVelocity
    phase_mass: Array
    phase_volume: Array
    mass_defect: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MultiphaseFLIPPlan(StrictModule, NonTrainableState):
    """Two-phase, one-velocity incompressible FLIP material reconstruction."""

    transfer: PreparedFLIPParticleTransfer
    densities: Array
    viscosities: Array
    surface_tension: Array
    phase_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedFLIPParticleTransfer,
        densities: ArrayLike,
        viscosities: ArrayLike,
        surface_tension: ArrayLike,
        /,
    ):
        if not isinstance(transfer, PreparedFLIPParticleTransfer):
            raise TypeError("transfer must be PreparedFLIPParticleTransfer.")
        rho = np.asarray(densities, dtype=float)
        mu = np.asarray(viscosities, dtype=float)
        sigma = np.asarray(surface_tension, dtype=float)
        if rho.shape != (2,) or mu.shape != (2,) or sigma.shape != (2, 2):
            raise ValueError("Initial multiphase FLIP requires exactly two phases.")
        if np.any(rho <= 0.0) or np.any(mu < 0.0) or np.any(sigma < 0.0):
            raise ValueError("Multiphase material values are invalid.")
        if not np.allclose(sigma, sigma.T) or not np.allclose(np.diag(sigma), 0.0):
            raise ValueError(
                "Surface-tension matrix must be symmetric with zero diagonal."
            )
        self.transfer = transfer
        self.densities = jnp.asarray(rho)
        self.viscosities = jnp.asarray(mu)
        self.surface_tension = jnp.asarray(sigma)
        self.phase_count = 2
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiphase-flip",
                "transfer": transfer.prepared_id,
                "densities": array_tree_fingerprint(rho),
                "viscosities": array_tree_fingerprint(mu),
                "surface_tension": array_tree_fingerprint(sigma),
            }
        )

    def evaluate(self, state: MultiphaseFLIPState, /) -> MultiphaseFLIPTransferResult:
        phase = jnp.asarray(state.phase_id, dtype=jnp.int32)
        if phase.shape != state.population.active.shape:
            raise ValueError("phase_id must preserve particle capacity.")
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
        face_mass = []
        face_momentum = []
        velocity = []
        inverse_density = []
        for axis in range(self.transfer.dimension):
            mass = sum(value[axis] for value in face_mass_by_phase)
            momentum = sum(value[axis] for value in face_momentum_by_phase)
            supported = mass > jnp.finfo(mass.dtype).eps
            face_mass.append(mass)
            face_momentum.append(momentum)
            velocity.append(
                jnp.where(supported, momentum / jnp.where(supported, mass, 1.0), 0.0)
            )
            phase_volume_face = []
            for phase_index in range(self.phase_count):
                phase_volume_face.append(
                    face_mass_by_phase[phase_index][axis] / self.densities[phase_index]
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
        finite = (
            jnp.all(jnp.isfinite(phase_fraction))
            & jnp.all(jnp.isfinite(mixture_density))
            & jnp.all(jnp.isfinite(mixture_viscosity))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in velocity))
            )
        )
        return MultiphaseFLIPTransferResult(
            phase_fraction,
            mixture_density,
            mixture_viscosity,
            tuple(face_mass),
            tuple(face_momentum),
            tuple(velocity),
            tuple(inverse_density),
            jnp.asarray(phase_masses),
            jnp.asarray([jnp.sum(value) for value in cell_volumes]),
            mass_defect,
            finite,
            successful & finite,
            self.plan_id,
        )


__all__ = [
    "MultiphaseFLIPPlan",
    "MultiphaseFLIPState",
    "MultiphaseFLIPTransferResult",
]
