#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_RADIATION_CONSTANT = 7.565733250033928e-16
_LIGHT_SPEED = 299792458.0


class IMCDDMCState(StrictModule):
    packet_cell: Array
    packet_group: Array
    packet_direction: Array
    packet_energy: Array
    packet_live: Array
    material_energy: Array
    material_temperature: Array
    time: Array
    accepted_step: Array
    plan_id: str = eqx.field(static=True)


class IMCDDMCEvidence(StrictModule):
    initial_total_energy: Array
    final_total_energy: Array
    escaped_energy: Array
    energy_residual: Array
    minimum_material_energy: Array
    minimum_packet_energy: Array
    ddmc_packet_count: Array
    imc_packet_count: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class IMCDDMCStepResult(StrictModule):
    candidate: IMCDDMCState
    accepted: IMCDDMCState
    escaped_energy: Array
    fleck_factor: Array
    evidence: IMCDDMCEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class HybridIMCDDMCPlan(StrictModule, NonTrainableState):
    cell_widths: Array
    absorption: Array
    scattering: Array
    heat_capacity: Array
    ddmc_optical_depth: float = eqx.field(static=True)
    transport_light_speed: float = eqx.field(static=True)
    maximum_packet_energy: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_widths: ArrayLike,
        absorption: ArrayLike,
        scattering: ArrayLike,
        heat_capacity: ArrayLike,
        /,
        *,
        ddmc_optical_depth: float = 3.0,
        transport_light_speed: float = _LIGHT_SPEED,
        maximum_packet_energy: float,
    ):
        widths = np.asarray(cell_widths, dtype=float)
        absorption_ = np.asarray(absorption, dtype=float)
        scattering_ = np.asarray(scattering, dtype=float)
        capacity = np.asarray(heat_capacity, dtype=float)
        threshold = float(ddmc_optical_depth)
        light_speed = float(transport_light_speed)
        maximum = float(maximum_packet_energy)
        if (
            widths.ndim != 1
            or widths.size < 1
            or np.any(~np.isfinite(widths))
            or np.any(widths <= 0.0)
            or absorption_.ndim != 2
            or absorption_.shape[0] != widths.size
            or scattering_.shape != absorption_.shape
            or np.any(~np.isfinite(absorption_))
            or np.any(absorption_ < 0.0)
            or np.any(~np.isfinite(scattering_))
            or np.any(scattering_ < 0.0)
            or capacity.shape != widths.shape
            or np.any(~np.isfinite(capacity))
            or np.any(capacity <= 0.0)
            or not isfinite(threshold)
            or threshold <= 0.0
            or not isfinite(light_speed)
            or not 0.0 < light_speed <= _LIGHT_SPEED
            or not isfinite(maximum)
            or maximum <= 0.0
        ):
            raise ValueError(
                "IMC/DDMC geometry, material, or resource policy is invalid."
            )
        self.cell_widths = jnp.asarray(widths)
        self.absorption = jnp.asarray(absorption_)
        self.scattering = jnp.asarray(scattering_)
        self.heat_capacity = jnp.asarray(capacity)
        self.ddmc_optical_depth = threshold
        self.transport_light_speed = light_speed
        self.maximum_packet_energy = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hybrid-imc-ddmc",
                "cell_widths": array_tree_fingerprint(widths),
                "absorption": array_tree_fingerprint(absorption_),
                "scattering": array_tree_fingerprint(scattering_),
                "heat_capacity": array_tree_fingerprint(capacity),
                "ddmc_optical_depth": threshold,
                "transport_light_speed": light_speed,
                "maximum_packet_energy": maximum,
            }
        )

    @property
    def cell_count(self) -> int:
        return int(self.cell_widths.size)

    @property
    def group_count(self) -> int:
        return int(self.absorption.shape[1])

    def initialize(
        self,
        packet_cell: ArrayLike,
        packet_group: ArrayLike,
        packet_direction: ArrayLike,
        packet_energy: ArrayLike,
        packet_live: ArrayLike,
        material_temperature: ArrayLike,
        /,
    ) -> IMCDDMCState:
        cell = jnp.asarray(packet_cell, dtype=jnp.int32)
        group = jnp.asarray(packet_group, dtype=jnp.int32)
        direction = jnp.asarray(packet_direction, dtype=jnp.int32)
        energy = jnp.asarray(packet_energy, dtype=self.cell_widths.dtype)
        live = jnp.asarray(packet_live, dtype=bool)
        temperature = jnp.asarray(material_temperature, dtype=self.cell_widths.dtype)
        if (
            cell.ndim != 1
            or group.shape != cell.shape
            or direction.shape != cell.shape
            or energy.shape != cell.shape
            or live.shape != cell.shape
            or temperature.shape != (self.cell_count,)
        ):
            raise ValueError("IMC/DDMC packet or material arrays have invalid shape.")
        if bool(
            jnp.any(
                live
                & (
                    (cell < 0)
                    | (cell >= self.cell_count)
                    | (group < 0)
                    | (group >= self.group_count)
                    | ((direction != -1) & (direction != 1))
                    | ~jnp.isfinite(energy)
                    | (energy <= 0.0)
                    | (energy > self.maximum_packet_energy)
                )
            )
        ) or bool(jnp.any(~jnp.isfinite(temperature) | (temperature <= 0.0))):
            raise ValueError("IMC/DDMC initial packet or material state is invalid.")
        material_energy = self.heat_capacity * temperature
        return IMCDDMCState(
            cell,
            group,
            direction,
            jnp.where(live, energy, 0.0),
            live,
            material_energy,
            temperature,
            jnp.asarray(0.0, dtype=temperature.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def advance(
        self, state: IMCDDMCState, step_size: ArrayLike, key: Array, /
    ) -> IMCDDMCStepResult:
        if not isinstance(state, IMCDDMCState) or state.plan_id != self.plan_id:
            raise TypeError("state must belong to this HybridIMCDDMCPlan.")
        step = jnp.asarray(step_size, dtype=self.cell_widths.dtype)
        if step.shape != () or jnp.asarray(key).shape not in ((), (2,)):
            raise ValueError("IMC/DDMC step or PRNG key has invalid shape.")
        temperature = state.material_energy / self.heat_capacity
        fleck = 1.0 / (
            1.0
            + 4.0
            * _RADIATION_CONSTANT
            * self.transport_light_speed
            * step
            * self.absorption
            * temperature[:, None] ** 3
            / self.heat_capacity[:, None]
        )
        initial_packet_energy = jnp.sum(state.packet_energy)
        initial_material_energy = jnp.sum(state.material_energy)

        def one(inputs):
            index, cell, group, direction, energy, live = inputs
            safe_cell = jnp.clip(cell, 0, self.cell_count - 1)
            safe_group = jnp.clip(group, 0, self.group_count - 1)
            absorption = self.absorption[safe_cell, safe_group]
            scattering = self.scattering[safe_cell, safe_group]
            fleck_value = fleck[safe_cell, safe_group]
            optical_depth = (absorption + scattering) * self.cell_widths[safe_cell]
            ddmc = live & (optical_depth >= self.ddmc_optical_depth)
            absorption_probability = 1.0 - jnp.exp(
                -self.transport_light_speed * fleck_value * absorption * step
            )
            absorbed = live & (
                jr.uniform(jr.fold_in(key, 5 * index)) < absorption_probability
            )
            diffusion = self.transport_light_speed / jnp.maximum(
                3.0 * (absorption + scattering), jnp.finfo(energy.dtype).tiny
            )
            leakage_probability = 1.0 - jnp.exp(
                -2.0 * diffusion * step / self.cell_widths[safe_cell] ** 2
            )
            leaks = (
                ddmc
                & ~absorbed
                & (jr.uniform(jr.fold_in(key, 5 * index + 1)) < leakage_probability)
            )
            random_direction = jnp.where(
                jr.uniform(jr.fold_in(key, 5 * index + 2)) < 0.5, -1, 1
            ).astype(jnp.int32)
            ddmc_direction = jnp.where(leaks, random_direction, direction)
            imc_crosses = (
                (~ddmc)
                & ~absorbed
                & (self.transport_light_speed * step >= self.cell_widths[safe_cell])
            )
            moves = leaks | imc_crosses
            next_direction = jnp.where(ddmc, ddmc_direction, direction)
            next_cell = cell + jnp.where(moves, next_direction, 0)
            escapes = (
                live
                & ~absorbed
                & moves
                & ((next_cell < 0) | (next_cell >= self.cell_count))
            )
            next_live = live & ~absorbed & ~escapes
            deposited = jnp.where(absorbed, energy, 0.0)
            escaped = jnp.where(escapes, energy, 0.0)
            return (
                jnp.clip(next_cell, 0, self.cell_count - 1),
                next_direction,
                jnp.where(next_live, energy, 0.0),
                next_live,
                deposited,
                escaped,
                ddmc,
            )

        count = state.packet_cell.size
        values = jax.lax.map(
            one,
            (
                jnp.arange(count, dtype=jnp.uint32),
                state.packet_cell,
                state.packet_group,
                state.packet_direction,
                state.packet_energy,
                state.packet_live,
            ),
        )
        cell, direction, energy, live, deposited, escaped, ddmc = values
        material_increment = (
            jnp.zeros((self.cell_count,), dtype=energy.dtype)
            .at[state.packet_cell]
            .add(deposited)
        )
        material_candidate = state.material_energy + material_increment
        temperature_candidate = material_candidate / self.heat_capacity
        candidate = IMCDDMCState(
            cell,
            state.packet_group,
            direction,
            energy,
            live,
            material_candidate,
            temperature_candidate,
            state.time + step,
            state.accepted_step + 1,
            self.plan_id,
        )
        escaped_total = jnp.sum(escaped)
        final_total = jnp.sum(material_candidate) + jnp.sum(energy) + escaped_total
        initial_total = initial_material_energy + initial_packet_energy
        residual = final_total - initial_total
        finite = (
            jnp.isfinite(step)
            & (step > 0.0)
            & jnp.all(jnp.isfinite(material_candidate))
            & jnp.all(jnp.isfinite(energy))
            & jnp.isfinite(residual)
        )
        scale = jnp.maximum(jnp.abs(initial_total), 1.0)
        tolerance = 512.0 * jnp.finfo(energy.dtype).eps * scale
        successful = (
            finite
            & jnp.all(material_candidate >= 0.0)
            & jnp.all(energy >= 0.0)
            & (jnp.abs(residual) <= tolerance)
        )
        accepted = jax.tree.map(
            lambda proposed, prior: jnp.where(successful, proposed, prior),
            candidate,
            state,
        )
        evidence = IMCDDMCEvidence(
            initial_total,
            final_total,
            escaped_total,
            residual,
            jnp.min(material_candidate),
            jnp.min(energy),
            jnp.sum(ddmc & state.packet_live, dtype=jnp.int32),
            jnp.sum(~ddmc & state.packet_live, dtype=jnp.int32),
            finite,
            successful,
            self.plan_id,
        )
        return IMCDDMCStepResult(
            candidate,
            accepted,
            escaped,
            fleck,
            evidence,
            successful,
            self.plan_id,
        )


__all__ = [
    "HybridIMCDDMCPlan",
    "IMCDDMCEvidence",
    "IMCDDMCState",
    "IMCDDMCStepResult",
]
