#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._strict import StrictModule


class SITActivityModel(StrictModule):
    charges: Array
    interaction_kg_mol: Array
    debye_a: Array
    maximum_ionic_strength_mol_kg: Array

    def __init__(
        self,
        charges: ArrayLike,
        interaction_kg_mol: ArrayLike,
        /,
        *,
        debye_a: ArrayLike = 0.509,
        maximum_ionic_strength_mol_kg: ArrayLike = 4.0,
    ):
        charges_ = jnp.asarray(charges)
        interaction = jnp.asarray(interaction_kg_mol)
        if charges_.ndim != 1 or interaction.shape != (charges_.size, charges_.size):
            raise ValueError("SIT charges/interactions have incompatible shapes.")
        self.charges = eqx.error_if(
            charges_,
            jnp.any(~jnp.isfinite(charges_))
            | jnp.any(~jnp.isfinite(interaction))
            | jnp.any(jnp.abs(interaction - interaction.T) > 1e-10),
            "SIT charges and symmetric interactions must be finite.",
        )
        self.interaction_kg_mol = interaction
        parameters = jnp.broadcast_arrays(
            jnp.asarray(debye_a), jnp.asarray(maximum_ionic_strength_mol_kg)
        )
        self.debye_a = eqx.error_if(
            parameters[0],
            jnp.any(~jnp.isfinite(parameters[0]))
            | jnp.any(parameters[0] <= 0)
            | jnp.any(~jnp.isfinite(parameters[1]))
            | jnp.any(parameters[1] <= 0),
            "SIT Debye and ionic-strength limits must be positive finite.",
        )
        self.maximum_ionic_strength_mol_kg = parameters[1]

    def activity_coefficients(self, molality_mol_kg: ArrayLike, /) -> Array:
        molality = jnp.asarray(molality_mol_kg)
        if molality.shape[-1] != self.charges.size:
            raise ValueError("SIT molality trailing component axis is invalid.")
        ionic = 0.5 * jnp.sum(molality * self.charges**2, axis=-1)
        molality = eqx.error_if(
            molality,
            jnp.any(~jnp.isfinite(molality))
            | jnp.any(molality < 0)
            | jnp.any(ionic > self.maximum_ionic_strength_mol_kg),
            "SIT molality exceeds its finite declared ionic-strength domain.",
        )
        root = jnp.sqrt(ionic)
        debye = (
            -self.debye_a
            * self.charges**2
            * root[..., None]
            / (1.0 + 1.5 * root[..., None])
        )
        interaction = ein.contract("ij,...j->...i", self.interaction_kg_mol, molality)
        return 10.0 ** (debye + interaction)


class PitzerInteractionModel(StrictModule):
    charges: Array
    beta0_kg_mol: Array
    beta1_kg_mol: Array
    cphi_kg2_mol2: Array
    alpha_kg_sqrt_mol: Array
    debye_a_phi: Array
    maximum_ionic_strength_mol_kg: Array

    def __init__(
        self,
        charges: ArrayLike,
        beta0_kg_mol: ArrayLike,
        beta1_kg_mol: ArrayLike,
        cphi_kg2_mol2: ArrayLike,
        /,
        *,
        alpha_kg_sqrt_mol: ArrayLike = 2.0,
        debye_a_phi: ArrayLike = 0.392,
        maximum_ionic_strength_mol_kg: ArrayLike = 12.0,
    ):
        charges_ = jnp.asarray(charges)
        matrices = tuple(
            jnp.asarray(value) for value in (beta0_kg_mol, beta1_kg_mol, cphi_kg2_mol2)
        )
        expected = (charges_.size, charges_.size)
        if charges_.ndim != 1 or any(value.shape != expected for value in matrices):
            raise ValueError("Pitzer charges and binary interaction matrices disagree.")
        invalid = jnp.any(~jnp.isfinite(charges_))
        for value in matrices:
            invalid = (
                invalid
                | jnp.any(~jnp.isfinite(value))
                | jnp.any(jnp.abs(value - value.T) > 1e-10)
            )
        self.charges = eqx.error_if(
            charges_,
            invalid,
            "Pitzer binary interaction data must be finite and symmetric.",
        )
        self.beta0_kg_mol, self.beta1_kg_mol, self.cphi_kg2_mol2 = matrices
        parameters = jnp.broadcast_arrays(
            jnp.asarray(alpha_kg_sqrt_mol),
            jnp.asarray(debye_a_phi),
            jnp.asarray(maximum_ionic_strength_mol_kg),
        )
        self.alpha_kg_sqrt_mol = eqx.error_if(
            parameters[0],
            any(jnp.any(~jnp.isfinite(value)) for value in parameters)
            | any(jnp.any(value <= 0) for value in parameters),
            "Pitzer range, Debye, and ionic-strength limits must be positive finite.",
        )
        self.debye_a_phi, self.maximum_ionic_strength_mol_kg = parameters[1:]

    def activity_coefficients(self, molality_mol_kg: ArrayLike, /) -> Array:
        molality = jnp.asarray(molality_mol_kg)
        if molality.shape[-1] != self.charges.size:
            raise ValueError("Pitzer molality trailing component axis is invalid.")
        ionic = 0.5 * jnp.sum(molality * self.charges**2, axis=-1)
        molality = eqx.error_if(
            molality,
            jnp.any(~jnp.isfinite(molality))
            | jnp.any(molality < 0)
            | jnp.any(ionic > self.maximum_ionic_strength_mol_kg),
            "Pitzer molality exceeds its declared ionic-strength domain.",
        )
        root = jnp.sqrt(ionic)
        x = self.alpha_kg_sqrt_mol * root
        safe_x = jnp.maximum(x, 1e-8)
        resolved = 2.0 * (1.0 - (1.0 + safe_x) * jnp.exp(-safe_x)) / safe_x**2
        g = jnp.where(x > 1e-8, resolved, 1.0 - 2.0 * x / 3.0)
        binary = self.beta0_kg_mol + self.beta1_kg_mol * g[..., None, None]
        debye = (
            -self.debye_a_phi
            * self.charges**2
            * root[..., None]
            / (1.0 + 1.2 * root[..., None])
        )
        pair = 2.0 * ein.contract("...ij,...j->...i", binary, molality)
        ternary = ionic[..., None] * ein.contract(
            "ij,...j->...i", self.cphi_kg2_mol2, molality
        )
        return jnp.exp(debye + pair + ternary)


class RedoxEquilibrium(StrictModule):
    electron_count: Array
    standard_potential_V: Array

    def __init__(self, electron_count: ArrayLike, standard_potential_V: ArrayLike, /):
        electrons, potential = jnp.broadcast_arrays(
            jnp.asarray(electron_count), jnp.asarray(standard_potential_V)
        )
        self.electron_count = eqx.error_if(
            electrons,
            jnp.any(~jnp.isfinite(electrons))
            | jnp.any(electrons <= 0)
            | jnp.any(~jnp.isfinite(potential)),
            "Redox electron count and standard potential must be finite and physical.",
        )
        self.standard_potential_V = potential

    def potential(
        self,
        temperature_K: ArrayLike,
        oxidized_activity: ArrayLike,
        reduced_activity: ArrayLike,
        /,
    ) -> Array:
        temperature, oxidized, reduced = jnp.broadcast_arrays(
            jnp.asarray(temperature_K),
            jnp.asarray(oxidized_activity),
            jnp.asarray(reduced_activity),
        )
        temperature = eqx.error_if(
            temperature,
            jnp.any(~jnp.isfinite(temperature))
            | jnp.any(temperature <= 0)
            | jnp.any(~jnp.isfinite(oxidized))
            | jnp.any(oxidized <= 0)
            | jnp.any(~jnp.isfinite(reduced))
            | jnp.any(reduced <= 0),
            "Redox temperature and activities must be positive finite.",
        )
        gas_constant, faraday = 8.314462618, 96485.33212
        return self.standard_potential_V + gas_constant * temperature / (
            self.electron_count * faraday
        ) * jnp.log(oxidized / reduced)


class HenryGasEquilibrium(StrictModule):
    henry_mol_m3_Pa: Array

    def __init__(self, henry_mol_m3_Pa: ArrayLike, /):
        value = jnp.asarray(henry_mol_m3_Pa)
        self.henry_mol_m3_Pa = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)) | jnp.any(value <= 0),
            "Henry coefficient must be positive finite.",
        )

    def dissolved_concentration(self, partial_pressure_Pa: ArrayLike, /) -> Array:
        pressure = jnp.asarray(partial_pressure_Pa)
        pressure = eqx.error_if(
            pressure,
            jnp.any(~jnp.isfinite(pressure)) | jnp.any(pressure < 0),
            "Gas partial pressure must be finite and nonnegative.",
        )
        return self.henry_mol_m3_Pa * pressure


class IonExchangeEquilibrium(StrictModule):
    selectivity: Array
    charges: Array

    def __init__(self, selectivity: ArrayLike, charges: ArrayLike, /):
        selectivity, charges_ = jnp.broadcast_arrays(
            jnp.asarray(selectivity), jnp.asarray(charges)
        )
        self.selectivity = eqx.error_if(
            selectivity,
            jnp.any(~jnp.isfinite(selectivity))
            | jnp.any(selectivity <= 0)
            | jnp.any(~jnp.isfinite(charges_))
            | jnp.any(charges_ <= 0),
            "Ion-exchange selectivity and charge magnitudes must be positive finite.",
        )
        self.charges = charges_

    def equivalent_fractions(self, activities: ArrayLike, /) -> Array:
        activity = jnp.asarray(activities)
        if activity.shape[-1] != self.selectivity.size:
            raise ValueError("Ion-exchange activities do not match species.")
        weights = self.selectivity * activity ** (1.0 / self.charges)
        totals = jnp.sum(weights, axis=-1)
        weights = eqx.error_if(
            weights,
            jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0) | jnp.any(totals <= 0),
            "Ion-exchange activities must form positive finite equivalent pools.",
        )
        return weights / jnp.sum(weights, axis=-1, keepdims=True)


__all__ = [
    "HenryGasEquilibrium",
    "IonExchangeEquilibrium",
    "PitzerInteractionModel",
    "RedoxEquilibrium",
    "SITActivityModel",
]
