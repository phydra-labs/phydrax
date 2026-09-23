#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coupled structural–thermal–optical performance analysis."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class OptomechanicalSTOPResult:
    temperature_rise_k: Array
    displacement_m: Array
    optical_path_difference_m: Array
    rms_wavefront_error_m: Array
    thermal_residual_norm: Array
    mechanical_residual_norm: Array
    strain_energy_j: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class SpatialOptomechanicalSystem:
    thermal_stiffness_w_k: Array
    mechanical_stiffness_n_m: Array
    thermoelastic_load_n_k: Array
    displacement_to_opd: Array
    temperature_to_opd_m_k: Array
    tolerance: float
    optical_weights: Array

    @classmethod
    def create(
        cls,
        thermal_stiffness_w_k: ArrayLike,
        mechanical_stiffness_n_m: ArrayLike,
        thermoelastic_load_n_k: ArrayLike,
        displacement_to_opd: ArrayLike,
        temperature_to_opd_m_k: ArrayLike,
        optical_weights: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> SpatialOptomechanicalSystem:
        thermal = np.asarray(thermal_stiffness_w_k, dtype=np.float64)
        mechanical = np.asarray(mechanical_stiffness_n_m, dtype=np.float64)
        coupling = np.asarray(thermoelastic_load_n_k, dtype=np.float64)
        displacement_map = np.asarray(displacement_to_opd, dtype=np.float64)
        temperature_map = np.asarray(temperature_to_opd_m_k, dtype=np.float64)
        weights = np.asarray(optical_weights, dtype=np.float64)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("STOP tolerance must be finite and positive.")
        if not all(
            np.all(np.isfinite(value))
            for value in (
                thermal,
                mechanical,
                coupling,
                displacement_map,
                temperature_map,
                weights,
            )
        ):
            raise ValueError("STOP operators and weights must be finite.")
        if thermal.ndim != 2 or thermal.shape[0] != thermal.shape[1]:
            raise ValueError("STOP thermal stiffness must be square.")
        if mechanical.ndim != 2 or mechanical.shape[0] != mechanical.shape[1]:
            raise ValueError("STOP mechanical stiffness must be square.")
        if coupling.shape != (mechanical.shape[0], thermal.shape[0]):
            raise ValueError("STOP thermoelastic coupling has incompatible shape.")
        if displacement_map.ndim != 2 or displacement_map.shape[1] != mechanical.shape[0]:
            raise ValueError("STOP displacement optical map has incompatible shape.")
        if temperature_map.shape != (displacement_map.shape[0], thermal.shape[0]):
            raise ValueError("STOP temperature optical map has incompatible shape.")
        if weights.shape != (displacement_map.shape[0],) or np.any(weights <= 0):
            raise ValueError("STOP optical weights must be positive and sample aligned.")
        if any(
            not np.allclose(value, value.T, atol=tolerance, rtol=0)
            for value in (thermal, mechanical)
        ):
            raise ValueError("STOP thermal and mechanical stiffness must be symmetric.")
        if (
            np.min(np.linalg.eigvalsh(thermal)) <= 0
            or np.min(np.linalg.eigvalsh(mechanical)) <= 0
        ):
            raise ValueError(
                "STOP thermal and mechanical stiffness must be positive definite."
            )
        return cls(
            jnp.asarray(thermal),
            jnp.asarray(mechanical),
            jnp.asarray(coupling),
            jnp.asarray(displacement_map),
            jnp.asarray(temperature_map),
            float(tolerance),
            jnp.asarray(weights),
        )

    def solve(
        self, absorbed_power_w: ArrayLike, optical_force_n: ArrayLike, /
    ) -> OptomechanicalSTOPResult:
        power = jnp.asarray(absorbed_power_w)
        force = jnp.asarray(optical_force_n)
        if power.shape != (self.thermal_stiffness_w_k.shape[0],):
            raise ValueError("STOP absorbed power has incompatible shape.")
        if force.shape != (self.mechanical_stiffness_n_m.shape[0],):
            raise ValueError("STOP optical force has incompatible shape.")
        power = eqx.error_if(
            power,
            jnp.any(~jnp.isfinite(power) | ~jnp.isfinite(force)),
            "STOP absorbed power and optical force must be finite.",
        )
        thermal = solve(
            LinearSystem(DenseLinearOperator(self.thermal_stiffness_w_k)),
            power,
            policy=LinearSolvePolicy(DenseLU()),
        )
        mechanical_right = force + self.thermoelastic_load_n_k @ thermal.value
        mechanical = solve(
            LinearSystem(DenseLinearOperator(self.mechanical_stiffness_n_m)),
            mechanical_right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        opd = (
            self.displacement_to_opd @ mechanical.value
            + self.temperature_to_opd_m_k @ thermal.value
        )
        normalized_weights = self.optical_weights / jnp.sum(self.optical_weights)
        piston = contract("q,q->", normalized_weights, opd)
        centered = opd - piston
        rms = jnp.sqrt(contract("q,q,q->", normalized_weights, centered, centered))
        thermal_residual = self.thermal_stiffness_w_k @ thermal.value - power
        mechanical_residual = (
            self.mechanical_stiffness_n_m @ mechanical.value - mechanical_right
        )
        thermal_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(thermal_residual), thermal_residual))
        )
        mechanical_norm = jnp.sqrt(
            jnp.real(
                contract("i,i->", jnp.conj(mechanical_residual), mechanical_residual)
            )
        )
        strain_energy = 0.5 * contract(
            "i,ij,j->", mechanical.value, self.mechanical_stiffness_n_m, mechanical.value
        )
        successful = (
            thermal.successful
            & mechanical.successful
            & jnp.all(jnp.isfinite(opd))
            & jnp.isfinite(rms)
            & jnp.isfinite(strain_energy)
            & (thermal_norm <= self.tolerance * (1.0 + jnp.linalg.norm(power)))
            & (
                mechanical_norm
                <= self.tolerance * (1.0 + jnp.linalg.norm(mechanical_right))
            )
        )
        return OptomechanicalSTOPResult(
            thermal.value,
            mechanical.value,
            opd,
            rms,
            thermal_norm,
            mechanical_norm,
            strain_energy,
            successful,
        )


__all__ = ["OptomechanicalSTOPResult", "SpatialOptomechanicalSystem"]
