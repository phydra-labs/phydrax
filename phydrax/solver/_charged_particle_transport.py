#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle._radiation_geometry import VoxelRadiationGeometryPlan
from ..equations._charged_radiation_interactions import (
    ChargedRadiationMaterialLibrary,
    ChargedRadiationParticleKind,
)


_POSITRON_ANNIHILATION_ENERGY_EV = 2.0 * 510998.95


class ChargedParticleTransportStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    MATERIAL_UNSUPPORTED = 2
    STEP_CAPACITY_EXHAUSTED = 3
    NONFINITE = 4


class ChargedParticleTransportResult(StrictModule, NonTrainableState):
    history_ids: Array
    deposited_energy: Array
    escaped_energy: Array
    bremsstrahlung_energy: Array
    annihilation_photon_energy: Array
    truncated_energy: Array
    terminal_position: Array
    terminal_direction: Array
    terminal_kinetic_energy: Array
    path_length: Array
    step_count: Array
    status: Array
    successful: Array
    maximum_kinetic_ledger_residual: Array
    all_successful: Array
    plan_id: str = eqx.field(static=True)


def _key(key, history, step, stream):
    return jr.fold_in(jr.fold_in(jr.fold_in(key, history), step), stream)


def _scatter_direction(direction, theta, phi):
    z = jnp.asarray((0.0, 0.0, 1.0), dtype=direction.dtype)
    x = jnp.asarray((1.0, 0.0, 0.0), dtype=direction.dtype)
    reference = jnp.where(jnp.abs(direction[2]) < 0.9, z, x)
    first = jnp.cross(reference, direction)
    first = first / jnp.linalg.norm(first)
    second = jnp.cross(direction, first)
    result = jnp.cos(theta) * direction + jnp.sin(theta) * (
        jnp.cos(phi) * first + jnp.sin(phi) * second
    )
    return result / jnp.linalg.norm(result)


class ChargedParticleTransportPlan(StrictModule, NonTrainableState):
    geometry: VoxelRadiationGeometryPlan
    materials: ChargedRadiationMaterialLibrary
    maximum_steps: int = eqx.field(static=True)
    maximum_step_length: float = eqx.field(static=True)
    maximum_fractional_energy_loss: float = eqx.field(static=True)
    cutoff_energy_ev: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: VoxelRadiationGeometryPlan,
        materials: ChargedRadiationMaterialLibrary,
        /,
        *,
        maximum_steps: int,
        maximum_step_length: float,
        maximum_fractional_energy_loss: float = 0.05,
        cutoff_energy_ev: float,
    ):
        if not isinstance(geometry, VoxelRadiationGeometryPlan):
            raise TypeError("geometry must be VoxelRadiationGeometryPlan.")
        if not isinstance(materials, ChargedRadiationMaterialLibrary):
            raise TypeError("materials must be ChargedRadiationMaterialLibrary.")
        steps = int(maximum_steps)
        length = float(maximum_step_length)
        fraction = float(maximum_fractional_energy_loss)
        cutoff = float(cutoff_energy_ev)
        if (
            geometry.material_count != materials.material_count
            or steps < 1
            or not isfinite(length)
            or length <= 0.0
            or not isfinite(fraction)
            or not 0.0 < fraction <= 0.25
            or not isfinite(cutoff)
            or cutoff <= 0.0
            or cutoff < float(materials.energy_ev[0])
            or cutoff > float(materials.energy_ev[-1])
        ):
            raise ValueError(
                "Charged transport materials, steps, or cutoffs are invalid."
            )
        self.geometry = geometry
        self.materials = materials
        self.maximum_steps = steps
        self.maximum_step_length = length
        self.maximum_fractional_energy_loss = fraction
        self.cutoff_energy_ev = cutoff
        self.plan_id = canonical_fingerprint(
            {
                "kind": "charged-particle-condensed-history",
                "geometry": geometry.geometry_id,
                "materials": materials.library_id,
                "maximum_steps": steps,
                "maximum_step_length": length,
                "maximum_fractional_energy_loss": fraction,
                "cutoff_energy_ev": cutoff,
                "secondary_photons": "tallied-not-transported",
                "pathwise_differentiability": False,
            }
        )

    def _one(self, key, history, origin, direction, energy, particle_kind):
        norm = jnp.linalg.norm(direction)
        kind_valid = (particle_kind == int(ChargedRadiationParticleKind.ELECTRON)) | (
            particle_kind == int(ChargedRadiationParticleKind.POSITRON)
        )
        initial_valid = (
            jnp.all(jnp.isfinite(origin))
            & jnp.all(jnp.isfinite(direction))
            & jnp.isfinite(energy)
            & (energy >= self.cutoff_energy_ev)
            & (norm > 0.0)
            & kind_valid
            & self.geometry.locate(origin).inside
        )
        direction = direction / jnp.where(norm > 0.0, norm, 1.0)
        state = (
            origin,
            direction,
            energy,
            initial_valid,
            jnp.asarray(0.0, energy.dtype),
            jnp.asarray(0.0, energy.dtype),
            jnp.asarray(0.0, energy.dtype),
            jnp.asarray(0.0, energy.dtype),
            jnp.asarray(0.0, energy.dtype),
            jnp.asarray(0.0, energy.dtype),
            jnp.asarray(0, jnp.int32),
            jnp.where(
                initial_valid,
                int(ChargedParticleTransportStatus.SUCCESS),
                int(ChargedParticleTransportStatus.INVALID_INPUT),
            ).astype(jnp.int32),
        )

        def advance(step_index, carry):
            (
                position,
                ray,
                kinetic,
                live,
                deposited,
                escaped,
                brems_energy,
                annihilation,
                truncated,
                path_length,
                step_count,
                status,
            ) = carry
            location = self.geometry.locate(position)
            material = self.materials.evaluate(location.material_index, kinetic)
            supported = live & location.inside & material.successful
            boundary = self.geometry.distance_to_voxel_boundary(position, ray)
            range_step = (
                self.maximum_fractional_energy_loss
                * kinetic
                / jnp.where(material.stopping_power > 0.0, material.stopping_power, 1.0)
            )
            unconstrained = jnp.minimum(self.maximum_step_length, range_step)
            step_length = jnp.minimum(unconstrained, boundary)
            boundary_limited = supported & (boundary <= unconstrained)
            continuous_loss = jnp.minimum(material.stopping_power * step_length, kinetic)
            remaining = kinetic - jnp.where(supported, continuous_loss, 0.0)
            brems_probability = 1.0 - jnp.exp(-material.bremsstrahlung_rate * step_length)
            brems_event = supported & (
                jr.uniform(_key(key, history, step_index, 0)) < brems_probability
            )
            brems_fraction = 0.5 * jr.uniform(_key(key, history, step_index, 1))
            emitted = jnp.where(brems_event, brems_fraction * remaining, 0.0)
            remaining = remaining - emitted
            theta = jnp.minimum(
                jnp.sqrt(jnp.maximum(material.scattering_power * step_length, 0.0))
                * jr.normal(_key(key, history, step_index, 2)),
                jnp.pi,
            )
            phi = 2.0 * jnp.pi * jr.uniform(_key(key, history, step_index, 3))
            scattered = _scatter_direction(ray, theta, phi)
            trial = position + step_length * ray
            nudge = jnp.where(boundary_limited, 1.0e-10, 0.0)
            trial = trial + nudge * ray
            next_location = self.geometry.locate(trial)
            leaves = boundary_limited & ~next_location.inside
            below_cutoff = supported & ~leaves & (remaining <= self.cutoff_energy_ev)
            is_positron = particle_kind == int(ChargedRadiationParticleKind.POSITRON)
            deposited = (
                deposited
                + jnp.where(supported, continuous_loss, 0.0)
                + jnp.where(below_cutoff, remaining, 0.0)
            )
            escaped = escaped + jnp.where(leaves, remaining, 0.0)
            annihilation = annihilation + jnp.where(
                below_cutoff & is_positron,
                _POSITRON_ANNIHILATION_ENERGY_EV,
                0.0,
            )
            unsupported = live & ~supported
            truncated = truncated + jnp.where(unsupported, kinetic, 0.0)
            next_live = supported & ~leaves & ~below_cutoff
            status = jnp.where(
                unsupported,
                int(ChargedParticleTransportStatus.MATERIAL_UNSUPPORTED),
                status,
            ).astype(jnp.int32)
            return (
                jnp.where(live, trial, position),
                jnp.where(supported, scattered, ray),
                jnp.where(leaves | below_cutoff, 0.0, remaining),
                next_live,
                deposited,
                escaped,
                brems_energy + emitted,
                annihilation,
                truncated,
                path_length + jnp.where(supported, step_length, 0.0),
                step_count + live.astype(jnp.int32),
                status,
            )

        state = jax.lax.fori_loop(0, self.maximum_steps, advance, state)
        (
            position,
            direction,
            kinetic,
            live,
            deposited,
            escaped,
            brems_energy,
            annihilation,
            truncated,
            path_length,
            step_count,
            status,
        ) = state
        truncated = truncated + jnp.where(live, kinetic, 0.0)
        status = jnp.where(
            live,
            int(ChargedParticleTransportStatus.STEP_CAPACITY_EXHAUSTED),
            status,
        ).astype(jnp.int32)
        ledger = energy - deposited - escaped - brems_energy - truncated
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        position,
                        direction,
                        jnp.asarray(
                            (
                                kinetic,
                                deposited,
                                escaped,
                                brems_energy,
                                truncated,
                                path_length,
                                ledger,
                            )
                        ),
                    )
                )
            )
        )
        tolerance = 256.0 * jnp.finfo(energy.dtype).eps * jnp.maximum(energy, 1.0)
        successful = (
            initial_valid
            & finite
            & (jnp.abs(ledger) <= tolerance)
            & (status == int(ChargedParticleTransportStatus.SUCCESS))
        )
        return (
            deposited,
            escaped,
            brems_energy,
            annihilation,
            truncated,
            position,
            direction,
            kinetic,
            path_length,
            step_count,
            status,
            successful,
            ledger,
        )

    def simulate(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        kinetic_energies_ev: ArrayLike,
        particle_kinds: ArrayLike,
        key: Array,
        /,
        *,
        history_ids: ArrayLike | None = None,
    ) -> ChargedParticleTransportResult:
        origins_ = jnp.asarray(origins, dtype=self.geometry.lower.dtype)
        directions_ = jnp.asarray(directions, dtype=origins_.dtype)
        energy = jnp.asarray(kinetic_energies_ev, dtype=origins_.dtype)
        kinds = jnp.asarray(particle_kinds, dtype=jnp.int32)
        if (
            origins_.ndim != 2
            or origins_.shape[-1] != 3
            or directions_.shape != origins_.shape
        ):
            raise ValueError(
                "Charged particle positions/directions require (history, 3)."
            )
        count = origins_.shape[0]
        if energy.shape != (count,) or kinds.shape != (count,):
            raise ValueError(
                "Charged particle energy/kind arrays must align with histories."
            )
        histories = (
            jnp.arange(count, dtype=jnp.uint32)
            if history_ids is None
            else jnp.asarray(history_ids, dtype=jnp.uint32)
        )
        if histories.shape != (count,) or jnp.asarray(key).shape not in ((), (2,)):
            raise ValueError("Charged particle history IDs or PRNG key are invalid.")
        values = jax.lax.map(
            lambda inputs: self._one(key, *inputs),
            (histories, origins_, directions_, energy, kinds),
        )
        (
            deposited,
            escaped,
            brems,
            annihilation,
            truncated,
            positions,
            terminal_directions,
            terminal_energy,
            path_length,
            step_count,
            status,
            successful,
            ledger,
        ) = values
        return ChargedParticleTransportResult(
            histories,
            deposited,
            escaped,
            brems,
            annihilation,
            truncated,
            positions,
            terminal_directions,
            terminal_energy,
            path_length,
            step_count,
            status,
            successful,
            jnp.max(jnp.abs(ledger)),
            jnp.all(successful),
            self.plan_id,
        )


__all__ = [
    "ChargedParticleTransportPlan",
    "ChargedParticleTransportResult",
    "ChargedParticleTransportStatus",
]
