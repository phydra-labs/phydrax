#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic periodic energy derivatives and defect formation energies."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


PeriodicEnergyFunction = Callable[[Array, Array], Array]


class PeriodicEnergyDerivativeResult(StrictModule, NonTrainableState):
    energy: Array
    forces: Array
    stress: Array
    force_balance_residual: Array
    stress_symmetry_residual: Array
    successful: Array
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(self, energy, forces, stress, successful, provider_id, /):
        energy_ = jnp.asarray(energy).reshape(())
        force = jnp.asarray(forces, dtype=energy_.dtype)
        stress_ = jnp.asarray(stress, dtype=energy_.dtype)
        provider = str(provider_id).strip()
        if (
            force.ndim != 2
            or force.shape[1] != 3
            or stress_.shape != (3, 3)
            or not provider
        ):
            raise ValueError("Periodic force, stress, or provider identity is invalid.")
        balance = jnp.max(jnp.abs(jnp.sum(force, axis=0)), initial=0.0)
        symmetry = jnp.max(jnp.abs(stress_ - stress_.T), initial=0.0)
        valid = (
            jnp.asarray(successful, dtype=bool)
            & jnp.isfinite(energy_)
            & jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(stress_))
        )
        self.energy = energy_
        self.forces = force
        self.stress = stress_
        self.force_balance_residual = balance
        self.stress_symmetry_residual = symmetry
        self.successful = valid
        self.provider_id = provider
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-energy-derivative-result",
                "provider": provider,
                "successful": bool(valid),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(energy_),
                        "forces": np.asarray(force),
                        "stress": np.asarray(stress_),
                    }
                ),
            }
        )


class PeriodicEnergyDerivativePlan(StrictModule, NonTrainableState):
    energy_function: PeriodicEnergyFunction = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, energy_function: PeriodicEnergyFunction, provider_id: str, /):
        if not callable(energy_function):
            raise TypeError("energy_function must be differentiable and callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.energy_function = energy_function
        self.provider_id = provider
        self.plan_id = canonical_fingerprint(
            {"kind": "periodic-energy-derivative-plan", "provider": provider}
        )

    def evaluate(
        self, positions: ArrayLike, cell_vectors: ArrayLike, /
    ) -> PeriodicEnergyDerivativeResult:
        coordinate = jnp.asarray(positions)
        cell = jnp.asarray(cell_vectors, dtype=coordinate.dtype)
        if coordinate.ndim != 2 or coordinate.shape[1] != 3 or cell.shape != (3, 3):
            raise ValueError("Periodic positions and cell have invalid shapes.")
        energy, position_gradient = jax.value_and_grad(
            lambda value: self.energy_function(value, cell)
        )(coordinate)
        volume = jnp.abs(jnp.dot(cell[0], jnp.cross(cell[1], cell[2])))

        def strained_energy(strain):
            deformation = jnp.eye(3, dtype=cell.dtype) + strain
            return self.energy_function(
                coordinate @ deformation.T,
                cell @ deformation.T,
            )

        strain_gradient = jax.grad(strained_energy)(jnp.zeros_like(cell))
        stress = 0.5 * (strain_gradient + jnp.conj(strain_gradient.T)) / volume
        successful = (
            (volume > 0.0)
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(position_gradient))
            & jnp.all(jnp.isfinite(strain_gradient))
        )
        return PeriodicEnergyDerivativeResult(
            energy, -position_gradient, stress, successful, self.provider_id
        )


class DefectFormationEnergyResult(StrictModule, NonTrainableState):
    formation_energy: Array
    chemical_potential_term: Array
    charge_term: Array
    correction: Array
    successful: Array
    result_id: str = eqx.field(static=True)


def defect_formation_energy(
    defect_energy: ArrayLike,
    bulk_energy: ArrayLike,
    stoichiometric_changes: ArrayLike,
    chemical_potentials: ArrayLike,
    charge_state: float,
    fermi_level: ArrayLike,
    valence_band_maximum: ArrayLike,
    correction: ArrayLike = 0.0,
    /,
) -> DefectFormationEnergyResult:
    defect = jnp.asarray(defect_energy)
    bulk = jnp.asarray(bulk_energy, dtype=defect.dtype)
    changes = jnp.asarray(stoichiometric_changes, dtype=defect.dtype)
    potentials = jnp.asarray(chemical_potentials, dtype=defect.dtype)
    if (
        changes.ndim != 1
        or potentials.shape != changes.shape
        or not isfinite(float(charge_state))
    ):
        raise ValueError(
            "Defect stoichiometry, chemical potentials, or charge state is invalid."
        )
    chemical = -jnp.dot(changes, potentials)
    charge = float(charge_state) * (
        jnp.asarray(fermi_level, dtype=defect.dtype)
        + jnp.asarray(valence_band_maximum, dtype=defect.dtype)
    )
    correction_ = jnp.asarray(correction, dtype=defect.dtype)
    formation = defect - bulk + chemical + charge + correction_
    successful = jnp.isfinite(formation)
    return DefectFormationEnergyResult(
        formation,
        chemical,
        charge,
        correction_,
        successful,
        canonical_fingerprint(
            {
                "kind": "defect-formation-energy-result",
                "arrays": array_tree_fingerprint(
                    {
                        "formation": np.asarray(formation),
                        "chemical": np.asarray(chemical),
                        "charge": np.asarray(charge),
                        "correction": np.asarray(correction_),
                    }
                ),
            }
        ),
    )


__all__ = [
    "DefectFormationEnergyResult",
    "PeriodicEnergyDerivativePlan",
    "PeriodicEnergyDerivativeResult",
    "defect_formation_energy",
]
