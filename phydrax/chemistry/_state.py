#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-molecule ground-state electronic-sector identities."""

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomisticSystemPlan, PreparedAtomisticSystem


def _integer(value: int, name: str, /, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


class PreparedMolecularElectronicSector(StrictModule, NonTrainableState):
    """Ground-state electron sector bound to one exact atomistic system."""

    total_charge: int = eqx.field(static=True)
    spin_multiplicity: int = eqx.field(static=True)
    electron_count: int = eqx.field(static=True)
    alpha_electron_count: int = eqx.field(static=True)
    beta_electron_count: int = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MolecularElectronicSectorPlan,
        system: AtomisticSystemPlan,
        /,
    ):
        active = np.asarray(system.active_mask, dtype=bool)
        elements = np.asarray(system.element_mask, dtype=bool)
        numbers = np.asarray(system.atomic_numbers, dtype=np.int64)
        if np.any(active & ~elements):
            raise ValueError(
                "Finite molecular electronic sectors require every active site to be an element."
            )
        if system.cell is not None and any(system.cell.periodic_axes):
            raise ValueError(
                "MolecularElectronicSectorPlan supports finite nonperiodic systems only."
            )
        nuclear_charge = int(np.sum(numbers[elements]))
        electrons = nuclear_charge - plan.total_charge
        unpaired = plan.spin_multiplicity - 1
        if electrons < 0:
            raise ValueError("Total charge produces a negative electron count.")
        if unpaired > electrons or (electrons + unpaired) % 2:
            raise ValueError(
                "Electron count and spin multiplicity have incompatible parity."
            )
        alpha = (electrons + unpaired) // 2
        beta = electrons - alpha
        self.total_charge = plan.total_charge
        self.spin_multiplicity = plan.spin_multiplicity
        self.electron_count = electrons
        self.alpha_electron_count = alpha
        self.beta_electron_count = beta
        self.system_id = system.system_id
        self.sector_id = plan.sector_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-molecular-electronic-sector",
                "system": system.system_id,
                "sector": plan.sector_id,
                "electron_count": electrons,
                "alpha_electrons": alpha,
                "beta_electrons": beta,
            }
        )


class MolecularElectronicSectorPlan(StrictModule, NonTrainableState):
    """Total charge and spin sector for one finite molecular ground state."""

    total_charge: int = eqx.field(static=True)
    spin_multiplicity: int = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)

    def __init__(
        self,
        total_charge: int,
        spin_multiplicity: int,
        /,
    ):
        charge = _integer(total_charge, "total_charge")
        multiplicity = _integer(spin_multiplicity, "spin_multiplicity", minimum=1)
        self.total_charge = charge
        self.spin_multiplicity = multiplicity
        self.sector_id = canonical_fingerprint(
            {
                "kind": "molecular-electronic-sector",
                "total_charge": charge,
                "spin_multiplicity": multiplicity,
            }
        )

    def prepare(
        self, system: AtomisticSystemPlan | PreparedAtomisticSystem, /
    ) -> PreparedMolecularElectronicSector:
        plan = system.plan if isinstance(system, PreparedAtomisticSystem) else system
        if not isinstance(plan, AtomisticSystemPlan):
            raise TypeError(
                "system must be AtomisticSystemPlan or PreparedAtomisticSystem."
            )
        return PreparedMolecularElectronicSector(self, plan)


__all__ = ["MolecularElectronicSectorPlan", "PreparedMolecularElectronicSector"]
