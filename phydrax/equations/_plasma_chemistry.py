#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_mechanism import PreparedChemicalMechanism
from ._chemical_rates import ChemicalRateRuntime


class ReactionTemperatureSpec(StrictModule, NonTrainableState):
    """Temperature owner for one reaction rate and reverse equilibrium."""

    kind: Literal["heavy", "mode", "electron", "geometric-mean"] = eqx.field(static=True)
    mode_index: int = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["heavy", "mode", "electron", "geometric-mean"],
        /,
        *,
        mode_index: int = -1,
    ):
        index = int(mode_index)
        if kind not in ("heavy", "mode", "electron", "geometric-mean") or (
            kind == "mode" and index < 0
        ):
            raise ValueError("Reaction temperature kind or mode index is invalid.")
        if kind != "mode" and index != -1:
            raise ValueError("Only mode-controlled reactions accept mode_index.")
        self.kind = kind
        self.mode_index = index
        self.spec_id = canonical_fingerprint(
            {
                "kind": "reaction-temperature",
                "source": kind,
                "mode_index": index,
            }
        )

    def select(
        self,
        heavy_temperature: Array,
        mode_temperatures: Array,
        electron_temperature: Array,
        /,
    ) -> Array:
        if self.kind == "heavy":
            return heavy_temperature
        if self.kind == "electron":
            return electron_temperature
        if self.kind == "mode":
            if self.mode_index >= mode_temperatures.shape[-1]:
                raise ValueError("Reaction mode index exceeds mode temperatures.")
            return mode_temperatures[..., self.mode_index]
        return jnp.sqrt(heavy_temperature * electron_temperature)


class PlasmaChemicalRateEvaluation(StrictModule):
    selected_temperatures: Array
    forward_rate_constants: Array
    reverse_rate_constants: Array
    net_progress_rates: Array
    species_amount_rate: Array
    mode_energy_rate: Array
    element_residual: Array
    charge_residual: Array
    explicit_step_restriction: Array
    finite: Array
    successful: Array
    mechanism_id: str = eqx.field(static=True)


class PreparedPlasmaMechanism(StrictModule, NonTrainableState):
    """Prepared mechanism with per-reaction thermal control and mode exchange."""

    mechanism: PreparedChemicalMechanism
    temperatures: tuple[ReactionTemperatureSpec, ...]
    mode_energy_per_progress: Array
    mechanism_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        temperatures: Sequence[ReactionTemperatureSpec],
        /,
        *,
        mode_energy_per_progress: ArrayLike,
    ):
        specifications = tuple(temperatures)
        mode_energy = np.asarray(mode_energy_per_progress, dtype=float)
        if (
            not isinstance(mechanism, PreparedChemicalMechanism)
            or len(specifications) != mechanism.reaction_count
            or any(
                not isinstance(value, ReactionTemperatureSpec) for value in specifications
            )
            or mode_energy.ndim != 2
            or mode_energy.shape[0] != mechanism.reaction_count
            or np.any(~np.isfinite(mode_energy))
        ):
            raise ValueError(
                "Plasma mechanism temperatures or mode exchange are invalid."
            )
        self.mechanism = mechanism
        self.temperatures = specifications
        self.mode_energy_per_progress = jnp.asarray(mode_energy)
        self.mechanism_id = canonical_fingerprint(
            {
                "kind": "prepared-plasma-mechanism",
                "mechanism": mechanism.mechanism_id,
                "temperatures": tuple(value.spec_id for value in specifications),
                "mode_energy_per_progress": array_tree_fingerprint(
                    self.mode_energy_per_progress
                ),
            }
        )

    @property
    def mode_count(self) -> int:
        return self.mode_energy_per_progress.shape[1]

    def evaluate(
        self,
        concentrations: ArrayLike,
        heavy_temperature: ArrayLike,
        mode_temperatures: ArrayLike,
        electron_temperature: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        runtime: ChemicalRateRuntime | None = None,
    ) -> PlasmaChemicalRateEvaluation:
        concentration = jnp.asarray(concentrations)
        heavy = jnp.asarray(heavy_temperature, dtype=concentration.dtype)
        modes = jnp.asarray(mode_temperatures, dtype=concentration.dtype)
        electron = jnp.asarray(electron_temperature, dtype=concentration.dtype)
        pressure_ = jnp.asarray(pressure, dtype=concentration.dtype)
        cell_shape = concentration.shape[:-1]
        if (
            concentration.shape[-1] != self.mechanism.schema.species_count
            or heavy.shape != cell_shape
            or electron.shape != cell_shape
            or pressure_.shape != cell_shape
            or modes.shape != cell_shape + (self.mode_count,)
        ):
            raise ValueError("Plasma mechanism state shapes are incompatible.")
        runtime_ = ChemicalRateRuntime() if runtime is None else runtime
        selected = jnp.stack(
            tuple(
                specification.select(heavy, modes, electron)
                for specification in self.temperatures
            ),
            axis=-1,
        )
        forward_values = []
        reverse_values = []
        for index, reaction in enumerate(self.mechanism.reactions):
            temperature = selected[..., index]
            forward = reaction.forward_rate.evaluate(
                temperature, pressure_, concentration, runtime_
            )
            if reaction.reverse_rate is not None:
                reverse = reaction.reverse_rate.evaluate(
                    temperature, pressure_, concentration, runtime_
                )
            elif reaction.thermodynamic_reversible:
                thermodynamics = self.mechanism.thermodynamics.evaluate(temperature)
                reverse = self.mechanism._thermodynamic_reverse_rate(
                    index, forward, temperature, thermodynamics
                )
            else:
                reverse = jnp.zeros_like(forward)
            forward_values.append(jnp.broadcast_to(forward, cell_shape))
            reverse_values.append(jnp.broadcast_to(reverse, cell_shape))
        forward_constant = jnp.stack(tuple(forward_values), axis=-1)
        reverse_constant = jnp.stack(tuple(reverse_values), axis=-1)
        safe_concentration = jnp.maximum(
            concentration, jnp.finfo(concentration.dtype).tiny
        )
        forward_mass_action = jnp.exp(
            contract(
                "...s,rs->...r",
                jnp.log(safe_concentration),
                self.mechanism.forward_orders,
                backend="jax",
            )
        )
        reverse_mass_action = jnp.exp(
            contract(
                "...s,rs->...r",
                jnp.log(safe_concentration),
                self.mechanism.product_stoichiometry,
                backend="jax",
            )
        )
        net_progress = (
            forward_constant * forward_mass_action
            - reverse_constant * reverse_mass_action
        )
        species_rate = contract(
            "...r,rs->...s",
            net_progress,
            self.mechanism.net_stoichiometry,
            backend="jax",
        )
        mode_rate = contract(
            "...r,rm->...m",
            net_progress,
            self.mode_energy_per_progress,
            backend="jax",
        )
        element_residual = contract(
            "es,...s->...e",
            self.mechanism.schema.element_composition,
            species_rate,
            backend="jax",
        )
        charge_residual = contract(
            "s,...s->...",
            self.mechanism.schema.charges,
            species_rate,
            backend="jax",
        )
        restriction = jnp.min(
            jnp.where(
                species_rate < 0.0,
                concentration
                / jnp.maximum(-species_rate, jnp.finfo(concentration.dtype).tiny),
                jnp.inf,
            ),
            axis=-1,
        )
        scale = jnp.maximum(jnp.max(jnp.abs(species_rate), axis=-1), 1.0)
        tolerance = 256.0 * jnp.finfo(concentration.dtype).eps * scale
        finite = (
            jnp.all(jnp.isfinite(concentration), axis=-1)
            & jnp.all(jnp.isfinite(selected), axis=-1)
            & jnp.all(jnp.isfinite(forward_constant), axis=-1)
            & jnp.all(jnp.isfinite(reverse_constant), axis=-1)
            & jnp.all(jnp.isfinite(species_rate), axis=-1)
            & jnp.all(jnp.isfinite(mode_rate), axis=-1)
        )
        successful = (
            finite
            & jnp.all(concentration >= 0.0, axis=-1)
            & jnp.all(selected > 0.0, axis=-1)
            & jnp.all(forward_constant >= 0.0, axis=-1)
            & jnp.all(reverse_constant >= 0.0, axis=-1)
            & jnp.all(jnp.abs(element_residual) <= tolerance[..., None], axis=-1)
            & (jnp.abs(charge_residual) <= tolerance)
        )
        return PlasmaChemicalRateEvaluation(
            selected,
            forward_constant,
            reverse_constant,
            net_progress,
            species_rate,
            mode_rate,
            element_residual,
            charge_residual,
            restriction,
            finite,
            successful,
            self.mechanism_id,
        )


__all__ = [
    "PlasmaChemicalRateEvaluation",
    "PreparedPlasmaMechanism",
    "ReactionTemperatureSpec",
]
