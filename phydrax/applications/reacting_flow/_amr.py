#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.amr._core import (
    BlockHierarchyState,
    BlockLevelState,
)
from ...equations._chemical_mechanism import PreparedChemicalMechanism


class ReactingAMRSynchronizationEvidence(StrictModule):
    minimum_species_density: Array
    element_defect: Array
    charge_defect: Array
    enthalpy_defect: Array
    nonlinear_residual: Array
    active_cell_count: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReactingAMRSynchronizationPlan(StrictModule, NonTrainableState):
    """Post-reflux synchronized chemistry for all-species/enthalpy AMR blocks.

    Block values must end in every species mass density followed by enthalpy
    density. The specialist changes species through the prepared mechanism while
    retaining enthalpy exactly. It is intended for BlockAMRRuntimePlan's
    specialist_synchronization hook with level subcycling disabled.
    """

    mechanism: PreparedChemicalMechanism
    thermodynamic_pressure: float = eqx.field(static=True)
    correction_sweeps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_temperature_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        /,
        *,
        thermodynamic_pressure: float,
        correction_sweeps: int = 3,
        tolerance: float = 1.0e-8,
        maximum_temperature_iterations: int = 64,
    ):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        pressure = float(thermodynamic_pressure)
        sweeps = int(correction_sweeps)
        tolerance_ = float(tolerance)
        iterations = int(maximum_temperature_iterations)
        if (
            not isfinite(pressure)
            or pressure <= 0.0
            or sweeps < 1
            or not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Reacting AMR pressure or numerical controls are invalid.")
        self.mechanism = mechanism
        self.thermodynamic_pressure = pressure
        self.correction_sweeps = sweeps
        self.tolerance = tolerance_
        self.maximum_temperature_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "synchronized-amr-thermochemistry",
                "mechanism": mechanism.mechanism_id,
                "thermodynamic_pressure": pressure,
                "correction_sweeps": sweeps,
                "tolerance": tolerance_,
                "maximum_temperature_iterations": iterations,
                "subcycling": False,
            }
        )

    def _temperature(self, species_density: Array, enthalpy_density: Array, /):
        masses = self.mechanism.schema.molar_masses.astype(species_density.dtype)
        concentration = species_density / masses
        lower = jnp.full_like(
            enthalpy_density, self.mechanism.thermodynamics.minimum_temperature
        )
        upper = jnp.full_like(
            enthalpy_density, self.mechanism.thermodynamics.maximum_temperature
        )

        def enthalpy(temperature):
            species = self.mechanism.thermodynamics.evaluate(temperature)
            return contract(
                "...s,...s->...",
                concentration,
                species.molar_enthalpy,
                backend="jax",
            )

        def iteration(_, bounds):
            low, high = bounds
            midpoint = 0.5 * (low + high)
            below = enthalpy(midpoint) < enthalpy_density
            return jnp.where(below, midpoint, low), jnp.where(below, high, midpoint)

        low, high = jax.lax.fori_loop(
            0, self.maximum_temperature_iterations, iteration, (lower, upper)
        )
        temperature = 0.5 * (low + high)
        residual = enthalpy(temperature) - enthalpy_density
        return temperature, residual

    def synchronize(
        self,
        level: int,
        hierarchy: BlockHierarchyState,
        time: Array,
        step_size: Array,
        args=None,
        /,
    ) -> tuple[BlockHierarchyState, Array, ReactingAMRSynchronizationEvidence]:
        del time, args
        level_ = int(level)
        if not isinstance(hierarchy, BlockHierarchyState):
            raise TypeError("hierarchy must be BlockHierarchyState.")
        if not 0 <= level_ < len(hierarchy.levels):
            raise ValueError("Reacting AMR level is out of range.")
        step = jnp.asarray(step_size, dtype=hierarchy.levels[level_].values.dtype)
        values = hierarchy.levels[level_].values
        count = self.mechanism.schema.species_count
        if values.shape[-1] != count + 1:
            raise ValueError(
                "Reacting AMR values must end in species densities and enthalpy."
            )
        active_block = hierarchy.levels[level_].metadata.active.reshape(
            (hierarchy.levels[level_].plan.maximum_blocks,)
            + (1,) * len(hierarchy.levels[level_].plan.block_shape)
        )
        active = jnp.broadcast_to(active_block, values.shape[:-1])
        species_initial = values[..., :count]
        enthalpy = values[..., count]
        safe_species = jnp.where(active[..., None], species_initial, 1.0)
        safe_enthalpy = jnp.where(active, enthalpy, 1.0)

        def mass_rate(species_density):
            temperature, _ = self._temperature(species_density, safe_enthalpy)
            concentration = species_density / self.mechanism.schema.molar_masses.astype(
                species_density.dtype
            )
            chemistry = self.mechanism.evaluate(
                concentration,
                temperature,
                jnp.full_like(temperature, self.thermodynamic_pressure),
            )
            return (
                chemistry.species_amount_rate
                * self.mechanism.schema.molar_masses.astype(species_density.dtype),
                chemistry.successful,
            )

        initial_rate, initial_success = mass_rate(safe_species)
        candidate = safe_species + step * initial_rate
        residual = jnp.asarray(jnp.inf, dtype=values.dtype)
        chemistry_success = initial_success
        for _ in range(self.correction_sweeps):
            valid = jnp.all(candidate > 0.0, axis=-1) & jnp.all(
                jnp.isfinite(candidate), axis=-1
            )
            evaluated = jnp.where(valid[..., None], candidate, safe_species)
            candidate_rate, rate_success = mass_rate(evaluated)
            corrected = safe_species + 0.5 * step * (initial_rate + candidate_rate)
            scale = jnp.maximum(jnp.max(jnp.abs(corrected), axis=-1), 1.0)
            residual = jnp.max(jnp.abs(corrected - candidate), axis=-1) / scale
            candidate = corrected
            chemistry_success = chemistry_success & rate_success
        candidate = jnp.where(active[..., None], candidate, species_initial)
        candidate_values = values.at[..., :count].set(candidate)
        amount_initial = species_initial / self.mechanism.schema.molar_masses.astype(
            values.dtype
        )
        amount_final = candidate / self.mechanism.schema.molar_masses.astype(values.dtype)
        element_defect = self.mechanism.schema.element_amount(
            amount_final - amount_initial
        )
        charge_defect = self.mechanism.schema.charge_amount(amount_final - amount_initial)
        minimum = jnp.min(jnp.where(active[..., None], candidate, jnp.inf))
        finite = jnp.all(jnp.where(active[..., None], jnp.isfinite(candidate), True))
        scale = jnp.maximum(
            jnp.max(jnp.where(active[..., None], jnp.abs(species_initial), 0.0)), 1.0
        )
        successful = (
            jnp.isfinite(step)
            & (step > 0.0)
            & finite
            & (minimum >= 0.0)
            & jnp.all(jnp.where(active, chemistry_success, True))
            & jnp.all(jnp.where(active, residual <= self.tolerance, True))
            & (
                jnp.max(
                    jnp.abs(jnp.where(active[..., None], element_defect, 0.0)),
                    initial=0.0,
                )
                <= self.tolerance * scale
            )
            & (
                jnp.max(jnp.abs(jnp.where(active, charge_defect, 0.0)), initial=0.0)
                <= self.tolerance * scale
            )
        )
        updated_level = BlockLevelState(
            hierarchy.levels[level_].plan,
            hierarchy.levels[level_].metadata,
            candidate_values,
        )
        levels = tuple(
            updated_level if index == level_ else value
            for index, value in enumerate(hierarchy.levels)
        )
        updated = BlockHierarchyState(hierarchy.topology, levels)
        evidence = ReactingAMRSynchronizationEvidence(
            minimum,
            element_defect,
            charge_defect,
            jnp.zeros_like(enthalpy),
            residual,
            jnp.sum(active, dtype=jnp.int32),
            finite,
            successful,
            self.plan_id,
        )
        return updated, successful, evidence

    def __call__(self, level, hierarchy, end_time, interval_dt, args=None):
        updated, successful, _ = self.synchronize(
            level, hierarchy, end_time, interval_dt, args
        )
        return updated, successful


__all__ = [
    "ReactingAMRSynchronizationEvidence",
    "ReactingAMRSynchronizationPlan",
]
