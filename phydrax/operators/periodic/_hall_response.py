#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Intrinsic two-dimensional sheet Hall response from resolved band curvature."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_PLANCK_SI = 6.626_070_15e-34
_BOLTZMANN_SI = 1.380_649e-23


class PeriodicSheetHallPlan(StrictModule, NonTrainableState):
    energies_joule: Array
    chern_density: Array
    point_weights: Array
    charge_coulomb: float = eqx.field(static=True)
    chemical_potential_joule: float = eqx.field(static=True)
    temperature_kelvin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies_joule: ArrayLike,
        chern_density: ArrayLike,
        point_weights: ArrayLike,
        /,
        *,
        charge_coulomb: float,
        chemical_potential_joule: float,
        temperature_kelvin: float,
    ):
        energies = np.asarray(energies_joule, dtype=np.float64)
        curvature = np.asarray(chern_density, dtype=np.float64)
        weights = np.asarray(point_weights, dtype=np.float64)
        charge = float(charge_coulomb)
        chemical = float(chemical_potential_joule)
        temperature = float(temperature_kelvin)
        if (
            energies.ndim != 2
            or curvature.shape != energies.shape
            or weights.shape != energies.shape[:1]
            or np.any(~np.isfinite(energies))
            or np.any(~np.isfinite(curvature))
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(np.sum(weights), 1.0)
            or not isfinite(charge)
            or charge == 0.0
            or not isfinite(chemical)
            or not isfinite(temperature)
            or temperature < 0.0
        ):
            raise ValueError(
                "Sheet Hall energies, curvature, weights, or thermodynamics are invalid."
            )
        self.energies_joule = jnp.asarray(energies)
        self.chern_density = jnp.asarray(curvature)
        self.point_weights = jnp.asarray(weights)
        self.charge_coulomb = charge
        self.chemical_potential_joule = chemical
        self.temperature_kelvin = temperature
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-sheet-hall-plan",
                "arrays": array_tree_fingerprint(
                    {"energies": energies, "chern_density": curvature, "weights": weights}
                ),
                "charge": charge,
                "chemical_potential": chemical,
                "temperature": temperature,
            }
        )


class PeriodicSheetHallResult(StrictModule, NonTrainableState):
    band_occupations: Array
    effective_chern: Array
    sheet_conductance_siemens: Array
    quantization_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def evaluate_periodic_sheet_hall(
    plan: PeriodicSheetHallPlan, /
) -> PeriodicSheetHallResult:
    if not isinstance(plan, PeriodicSheetHallPlan):
        raise TypeError("plan must be PeriodicSheetHallPlan.")
    if plan.temperature_kelvin == 0.0:
        occupation = (plan.energies_joule < plan.chemical_potential_joule).astype(
            jnp.float64
        )
    else:
        argument = (plan.energies_joule - plan.chemical_potential_joule) / (
            _BOLTZMANN_SI * plan.temperature_kelvin
        )
        occupation = jax.nn.sigmoid(-argument)
    effective_chern = jnp.sum(
        plan.point_weights[:, None] * occupation * plan.chern_density
    )
    conductance = plan.charge_coulomb**2 / _PLANCK_SI * effective_chern
    nearest = jnp.rint(effective_chern)
    residual = jnp.abs(effective_chern - nearest)
    successful = jnp.isfinite(conductance) & jnp.isfinite(residual)
    return PeriodicSheetHallResult(
        occupation,
        effective_chern,
        conductance,
        residual,
        successful,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "periodic-sheet-hall-result",
                "plan": plan.plan_id,
                "effective_chern": float(np.asarray(effective_chern)),
            }
        ),
    )


__all__ = [
    "PeriodicSheetHallPlan",
    "PeriodicSheetHallResult",
    "evaluate_periodic_sheet_hall",
]
