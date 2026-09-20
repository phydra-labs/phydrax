#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Process–structure–property evolution for spatial material states."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ._spatial import SpatialMaterialField
from ._state import MaterialState


@dataclass(frozen=True, slots=True)
class ICMEStepResult:
    field: SpatialMaterialField
    effective_properties: Array
    transformed_phase_measure: Array
    phase_balance_residual: Array


@dataclass(frozen=True, slots=True)
class SpatialICMEModel:
    """Kinetic phase evolution followed by local property homogenization.

    Each phase relaxes toward a supplied equilibrium fraction with a non-negative
    first-order rate. The exact exponential update is positivity preserving; the
    final normalization enforces the local phase simplex to roundoff.
    """

    phase_properties: Array
    kinetic_rates_s_inv: Array
    homogenization: str = "voigt"

    @classmethod
    def create(
        cls,
        phase_properties: ArrayLike,
        kinetic_rates_s_inv: ArrayLike,
        /,
        *,
        homogenization: str = "voigt",
    ) -> SpatialICMEModel:
        properties = np.asarray(phase_properties, dtype=np.float64)
        rates = np.asarray(kinetic_rates_s_inv, dtype=np.float64)
        if properties.ndim < 1 or properties.shape[0] < 1:
            raise ValueError("ICME phase properties require a leading phase axis.")
        if rates.shape != (properties.shape[0],) or np.any(rates < 0):
            raise ValueError("ICME phase kinetic rates must be non-negative and aligned.")
        if not np.all(np.isfinite(properties)) or not np.all(np.isfinite(rates)):
            raise ValueError("ICME phase data must be finite.")
        if homogenization not in ("voigt", "reuss", "hill"):
            raise ValueError("ICME homogenization must be voigt, reuss, or hill.")
        if homogenization in ("reuss", "hill") and np.any(properties <= 0):
            raise ValueError("Reuss-family homogenization requires positive properties.")
        return cls(jnp.asarray(properties), jnp.asarray(rates), homogenization)

    def advance(
        self,
        field: SpatialMaterialField,
        temperature_k: ArrayLike,
        equilibrium_phase_fractions: ArrayLike,
        step_size_s: float,
        /,
        *,
        pressure_pa: ArrayLike | None = None,
    ) -> ICMEStepResult:
        if step_size_s <= 0:
            raise ValueError("ICME step size must be positive.")
        old = field.state.phase_fractions
        equilibrium = jnp.asarray(equilibrium_phase_fractions)
        temperature = jnp.asarray(temperature_k)
        pressure = (
            field.state.pressure_pa if pressure_pa is None else jnp.asarray(pressure_pa)
        )
        if equilibrium.shape != old.shape:
            raise ValueError("Equilibrium phase fractions must match the material field.")
        if temperature.shape != field.state.temperature_k.shape:
            raise ValueError("ICME temperature must match the material field.")
        if pressure.shape != field.state.pressure_pa.shape:
            raise ValueError("ICME pressure must match the material field.")
        if not bool(
            jnp.all(equilibrium >= 0) & jnp.allclose(jnp.sum(equilibrium, axis=-1), 1)
        ):
            raise ValueError("Equilibrium phase fractions must lie on the simplex.")

        decay = jnp.exp(-float(step_size_s) * self.kinetic_rates_s_inv)
        fractions = equilibrium + (old - equilibrium) * decay
        fractions = jnp.maximum(fractions, 0)
        fractions = fractions / jnp.sum(fractions, axis=-1, keepdims=True)

        voigt = contract("qp,p...->q...", fractions, self.phase_properties)
        if self.homogenization == "voigt":
            effective = voigt
        else:
            reuss = 1 / contract("qp,p...->q...", fractions, 1 / self.phase_properties)
            effective = reuss if self.homogenization == "reuss" else 0.5 * (voigt + reuss)

        state = MaterialState(
            temperature,
            pressure,
            fractions,
            field.state.internal_variables,
        )
        evolved = SpatialMaterialField.create(
            field.coordinates_m, field.measure_weights, state
        )
        phase_measure = contract("q,qp->p", field.measure_weights, fractions)
        expected_measure = contract(
            "q,qp->p", field.measure_weights, equilibrium + (old - equilibrium) * decay
        )
        residual = phase_measure - expected_measure
        return ICMEStepResult(evolved, effective, phase_measure, residual)


__all__ = ["ICMEStepResult", "SpatialICMEModel"]
