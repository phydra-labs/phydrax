#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic chemistry mean-field corrections."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...operators.periodic._orbital import PeriodicOrbitalBasisPlan
from ...units import ENERGY, UnitDefinition


class PeriodicHubbardMeanFieldPlan(StrictModule, NonTrainableState):
    """Diagonal Hubbard/reference/ionic fields, separate from the H/S pencil."""

    basis_id: str = eqx.field(static=True)
    onsite_hubbard: Array
    reference_populations: Array
    ionic_energy: Array
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: PeriodicOrbitalBasisPlan,
        onsite_hubbard: ArrayLike,
        reference_populations: ArrayLike,
        ionic_energy: ArrayLike,
        energy_unit: UnitDefinition,
        /,
    ):
        if not isinstance(basis, PeriodicOrbitalBasisPlan):
            raise TypeError("basis must be PeriodicOrbitalBasisPlan.")
        hubbard = np.asarray(onsite_hubbard)
        reference = np.asarray(reference_populations)
        ionic = np.asarray(ionic_energy)
        if (
            hubbard.shape != (basis.orbital_count,)
            or reference.shape != hubbard.shape
            or ionic.shape != ()
            or np.any(~np.isfinite(hubbard))
            or np.any(hubbard < 0.0)
            or np.any(~np.isfinite(reference))
            or np.any(reference < 0.0)
            or not np.isfinite(ionic)
        ):
            raise ValueError("Hubbard, reference, and ionic fields are invalid.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        if energy_unit.dimension != ENERGY:
            raise ValueError(
                "Periodic Hubbard mean-field energy_unit must have energy dimension."
            )
        self.basis_id = basis.basis_id
        self.onsite_hubbard = jnp.asarray(hubbard)
        self.reference_populations = jnp.asarray(reference)
        self.ionic_energy = jnp.asarray(ionic)
        self.energy_unit = energy_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-hubbard-mean-field-plan",
                "basis": basis.basis_id,
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "onsite_hubbard": hubbard,
                        "reference_populations": reference,
                        "ionic_energy": ionic,
                    }
                ),
            }
        )


__all__ = ["PeriodicHubbardMeanFieldPlan"]
