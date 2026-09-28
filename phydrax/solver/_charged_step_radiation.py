#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transition-radiation and Cherenkov processes over charged step banks."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._matter_radiation_interactions import (
    cherenkov_yield_in_band,
    FoilStackTransitionRadiationPlan,
)
from ._charged_particle_transport import ChargedStepBank
from ._secondary_stack import SecondaryParticleStack, SecondaryStackSpec


class ChargedStepRadiationResult(StrictModule, NonTrainableState):
    """Expected yields and atomically admitted photon-energy packets."""

    cherenkov_photon_count: Array
    cherenkov_energy_ev: Array
    transition_photon_count: Array
    transition_energy_ev: Array
    local_deposit_ev: Array
    refused_energy_ev: Array
    secondary_photons: SecondaryParticleStack
    requested_count: Array
    refused: Array
    ledger_residual_ev: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ChargedStepRadiationPlan(StrictModule, NonTrainableState):
    """Consume an M1a step bank and generate bounded optical/X-ray packets.

    Frank--Tamm yield is integrated analytically over the declared wavelength
    band on every active step.  An optional Garibian/Cherry foil-stack model is
    integrated on declared photon-energy and angle grids for histories that
    traverse the stack.  One packet per radiating step carries the exact total
    expected photon energy; its expected photon multiplicity remains explicit.
    Capacity refusal is atomic per history.
    """

    refractive_index: Array
    wavelength_min_m: float = eqx.field(static=True)
    wavelength_max_m: float = eqx.field(static=True)
    transition_radiation: FoilStackTransitionRadiationPlan | None
    transition_energy_ev: Array
    transition_angle_rad: Array
    photon_stack: SecondaryStackSpec
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        refractive_index: ArrayLike,
        /,
        *,
        wavelength_min_m: float,
        wavelength_max_m: float,
        photon_stack: SecondaryStackSpec,
        transition_radiation: FoilStackTransitionRadiationPlan | None = None,
        transition_energy_ev: ArrayLike | None = None,
        transition_angle_rad: ArrayLike | None = None,
    ) -> None:
        index = np.asarray(refractive_index, dtype=np.float64)
        lower, upper = float(wavelength_min_m), float(wavelength_max_m)
        if (
            index.ndim != 1
            or index.size < 1
            or np.any(~np.isfinite(index))
            or np.any(index <= 0.0)
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or not 0.0 < lower < upper
        ):
            raise ValueError("Cherenkov indices and wavelength band are invalid.")
        if not isinstance(photon_stack, SecondaryStackSpec):
            raise TypeError("photon_stack must be SecondaryStackSpec.")
        if transition_radiation is not None and not isinstance(
            transition_radiation, FoilStackTransitionRadiationPlan
        ):
            raise TypeError(
                "transition_radiation must be FoilStackTransitionRadiationPlan or None."
            )
        if transition_radiation is None:
            energy = np.asarray((1.0, 2.0), dtype=np.float64)
            angle = np.asarray((0.0, 1.0), dtype=np.float64)
            if transition_energy_ev is not None or transition_angle_rad is not None:
                raise ValueError(
                    "Transition grids require an attached foil-stack radiation plan."
                )
        else:
            if transition_energy_ev is None or transition_angle_rad is None:
                raise ValueError(
                    "Transition radiation requires energy and angle integration grids."
                )
            energy = np.asarray(transition_energy_ev, dtype=np.float64)
            angle = np.asarray(transition_angle_rad, dtype=np.float64)
            if (
                energy.ndim != 1
                or energy.size < 2
                or np.any(~np.isfinite(energy))
                or np.any(energy <= 0.0)
                or np.any(np.diff(energy) <= 0.0)
                or angle.ndim != 1
                or angle.size < 2
                or np.any(~np.isfinite(angle))
                or np.any(angle < 0.0)
                or np.any(np.diff(angle) <= 0.0)
            ):
                raise ValueError("Transition-radiation integration grids are invalid.")
        self.refractive_index = jnp.asarray(index)
        self.wavelength_min_m = lower
        self.wavelength_max_m = upper
        self.transition_radiation = transition_radiation
        self.transition_energy_ev = jnp.asarray(energy)
        self.transition_angle_rad = jnp.asarray(angle)
        self.photon_stack = photon_stack
        self.plan_id = canonical_fingerprint(
            {
                "kind": "charged-step-radiation",
                "refractive_index": array_tree_fingerprint(index),
                "wavelength_band_m": (lower, upper),
                "transition_radiation": (
                    None if transition_radiation is None else transition_radiation.plan_id
                ),
                "transition_energy_ev": array_tree_fingerprint(energy),
                "transition_angle_rad": array_tree_fingerprint(angle),
                "photon_stack": photon_stack.spec_id,
            }
        )

    def _transition_yield(self, start_beta: Array) -> tuple[Array, Array]:
        transition = self.transition_radiation
        if transition is None:
            zero = jnp.zeros_like(start_beta)
            return zero, zero
        gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - start_beta**2, 1.0e-30))
        energy = self.transition_energy_ev
        angle = self.transition_angle_rad
        yield_density = jax.vmap(
            lambda gamma_: transition.spectral_angular_yield(
                energy[:, None], gamma_, angle[None, :]
            )
        )(gamma)
        angle_squared = angle**2
        angular_integral = jnp.trapezoid(yield_density, angle_squared, axis=-1)
        count = jnp.trapezoid(angular_integral, energy, axis=-1)
        radiated_energy = jnp.trapezoid(
            angular_integral * energy[None, :], energy, axis=-1
        )
        return count, radiated_energy

    def evaluate(self, step_bank: ChargedStepBank, /) -> ChargedStepRadiationResult:
        if not isinstance(step_bank, ChargedStepBank):
            raise TypeError("step_bank must be ChargedStepBank.")
        material_supported = jnp.all(
            ~step_bank.active
            | (
                (step_bank.material_index >= 0)
                & (step_bank.material_index < self.refractive_index.size)
            ),
            axis=1,
        )
        segment = step_bank.end_positions - step_bank.start_positions
        length = jnp.linalg.norm(segment, axis=-1)
        safe_length = jnp.where(length > 0.0, length, 1.0)
        direction = segment / safe_length[..., None]
        beta = 0.5 * (step_bank.start_beta + step_bank.end_beta)
        material = jnp.clip(step_bank.material_index, 0, self.refractive_index.size - 1)
        cherenkov_count, cherenkov_energy = cherenkov_yield_in_band(
            length,
            beta,
            self.refractive_index[material],
            self.wavelength_min_m,
            self.wavelength_max_m,
        )
        cherenkov_count = jnp.where(step_bank.active, cherenkov_count, 0.0)
        cherenkov_energy = jnp.where(step_bank.active, cherenkov_energy, 0.0)
        first_slot = jnp.argmax(step_bank.active, axis=1)
        first_beta = jnp.take_along_axis(
            step_bank.start_beta, first_slot[:, None], axis=1
        )[:, 0]
        transition_count, transition_energy = self._transition_yield(first_beta)
        if self.transition_radiation is None:
            traverses = jnp.zeros((step_bank.active.shape[0],), dtype=jnp.bool_)
        else:
            lower = self.transition_radiation.interface_positions_m[0]
            upper = self.transition_radiation.interface_positions_m[-1]
            minimum_z = jnp.min(
                jnp.where(
                    step_bank.active,
                    jnp.minimum(
                        step_bank.start_positions[..., 2],
                        step_bank.end_positions[..., 2],
                    ),
                    jnp.inf,
                ),
                axis=1,
            )
            maximum_z = jnp.max(
                jnp.where(
                    step_bank.active,
                    jnp.maximum(
                        step_bank.start_positions[..., 2],
                        step_bank.end_positions[..., 2],
                    ),
                    -jnp.inf,
                ),
                axis=1,
            )
            traverses = (minimum_z <= lower) & (maximum_z >= upper)
        transition_count = jnp.where(traverses, transition_count, 0.0)
        transition_energy = jnp.where(traverses, transition_energy, 0.0)
        packet_energy = cherenkov_energy.at[
            jnp.arange(step_bank.active.shape[0]), first_slot
        ].add(transition_energy)
        packet_active = step_bank.active & (
            packet_energy >= self.photon_stack.minimum_energy
        )
        requested = jnp.sum(packet_active, axis=1, dtype=jnp.int32)
        refused = requested > self.photon_stack.capacity
        sort_key = jnp.where(
            packet_active,
            jnp.arange(step_bank.capacity, dtype=jnp.int32)[None, :],
            step_bank.capacity,
        )
        order = jnp.argsort(sort_key, axis=1, stable=True)[
            :, : self.photon_stack.capacity
        ]

        def gather(values: Array) -> Array:
            return jnp.take_along_axis(
                values,
                order[(...,) + (None,) * (values.ndim - 2)],
                axis=1,
            )

        active = (
            jnp.arange(self.photon_stack.capacity)[None, :] < requested[:, None]
        ) & ~refused[:, None]
        energies = jnp.where(active, gather(packet_energy), 0.0)
        positions = jnp.where(active[..., None], gather(step_bank.start_positions), 0.0)
        directions = jnp.where(active[..., None], gather(direction), 0.0)
        materials = jnp.where(active, gather(step_bank.material_index), -1)
        creations = jnp.where(active, order, -1).astype(jnp.int32)
        kinds = jnp.zeros_like(creations)
        stack_energy = jnp.sum(energies, axis=1)
        overflow = jnp.where(refused, requested - self.photon_stack.capacity, 0)
        stack = SecondaryParticleStack(
            positions,
            directions,
            energies,
            materials,
            creations,
            kinds,
            active,
            jnp.sum(active, axis=1, dtype=jnp.int32),
            stack_energy,
            overflow,
            self.photon_stack.spec_id,
        )
        total_energy = jnp.sum(cherenkov_energy, axis=1) + transition_energy
        local_deposit = jnp.sum(
            jnp.where(step_bank.active & ~packet_active, packet_energy, 0.0), axis=1
        )
        admitted_energy = jnp.sum(jnp.where(packet_active, packet_energy, 0.0), axis=1)
        refused_energy = jnp.where(refused, admitted_energy, 0.0)
        residual = total_energy - stack_energy - local_deposit - refused_energy
        tolerance = 256.0 * jnp.finfo(residual.dtype).eps * jnp.maximum(total_energy, 1.0)
        successful = (
            ~refused
            & material_supported
            & step_bank.complete
            & (jnp.abs(residual) <= tolerance)
        )
        return ChargedStepRadiationResult(
            jnp.sum(cherenkov_count, axis=1),
            jnp.sum(cherenkov_energy, axis=1),
            transition_count,
            transition_energy,
            local_deposit,
            refused_energy,
            stack,
            requested,
            refused,
            residual,
            successful,
            self.plan_id,
        )


__all__ = ["ChargedStepRadiationPlan", "ChargedStepRadiationResult"]
