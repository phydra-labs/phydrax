#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite and periodic ground-state electronic-sector identities."""

from __future__ import annotations

from math import isfinite
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
    spin_magnetization: int = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MolecularElectronicSectorPlan,
        system: AtomisticSystemPlan,
        /,
    ):
        active = np.asarray(system.active_mask, dtype=np.bool_)
        elements = np.asarray(system.element_mask, dtype=np.bool_)
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
        self.spin_magnetization = alpha - beta
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
                "spin_magnetization": alpha - beta,
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


class PreparedPeriodicElectronicSector(StrictModule, NonTrainableState):
    """Periodic electron/spin population sector bound to one atomistic system."""

    total_charge: float = eqx.field(static=True)
    spin_multiplicity: float | None = eqx.field(static=True)
    electron_count: float = eqx.field(static=True)
    alpha_electron_count: float = eqx.field(static=True)
    beta_electron_count: float = eqx.field(static=True)
    spin_magnetization: float = eqx.field(static=True)
    charge_per_cell: float = eqx.field(static=True)
    background_policy: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: "PeriodicElectronicSectorPlan",
        system: AtomisticSystemPlan,
        /,
    ):
        if system.cell is None or not any(system.cell.periodic_axes):
            raise ValueError(
                "PeriodicElectronicSectorPlan requires a periodic atomistic system."
            )
        alpha = 0.5 * (plan.electron_count + plan.spin_magnetization)
        beta = 0.5 * (plan.electron_count - plan.spin_magnetization)
        self.total_charge = plan.charge_per_cell
        self.spin_multiplicity = (
            1.0 if plan.spin_magnetization == 0.0 else abs(plan.spin_magnetization) + 1.0
        )
        self.electron_count = plan.electron_count
        self.alpha_electron_count = alpha
        self.beta_electron_count = beta
        self.spin_magnetization = plan.spin_magnetization
        self.charge_per_cell = plan.charge_per_cell
        self.background_policy = plan.background_policy
        self.system_id = system.system_id
        self.sector_id = plan.sector_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-electronic-sector",
                "system": system.system_id,
                "sector": plan.sector_id,
                "electron_count": plan.electron_count,
                "alpha_electrons": alpha,
                "beta_electrons": beta,
                "spin_magnetization": plan.spin_magnetization,
                "charge_per_cell": plan.charge_per_cell,
                "background_policy": plan.background_policy,
            }
        )


class PeriodicElectronicSectorPlan(StrictModule, NonTrainableState):
    """Electron count, collinear spin, and charge-background policy per cell."""

    electron_count: float = eqx.field(static=True)
    spin_magnetization: float = eqx.field(static=True)
    charge_per_cell: float = eqx.field(static=True)
    background_policy: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)

    def __init__(
        self,
        electron_count: float,
        /,
        *,
        spin_magnetization: float = 0.0,
        charge_per_cell: float = 0.0,
        background_policy: str = "forbid-charged-cell",
    ):
        electrons = float(electron_count)
        magnetization = float(spin_magnetization)
        charge = float(charge_per_cell)
        policy = str(background_policy).strip()
        allowed_policies = {"forbid-charged-cell", "uniform-background"}
        if policy not in allowed_policies:
            raise ValueError(
                "background_policy must be 'forbid-charged-cell' or 'uniform-background'."
            )
        if any(not isfinite(value) for value in (electrons, magnetization, charge)):
            raise ValueError(
                "Periodic electron, magnetization, and charge values must be finite."
            )
        if electrons <= 0.0 or abs(magnetization) > electrons or not policy:
            raise ValueError("Periodic electronic sector is invalid.")
        if charge != 0.0 and policy == "forbid-charged-cell":
            raise ValueError(
                "Charged periodic cells require an explicit background policy."
            )
        self.electron_count = electrons
        self.spin_magnetization = magnetization
        self.charge_per_cell = charge
        self.background_policy = policy
        self.sector_id = canonical_fingerprint(
            {
                "kind": "periodic-electronic-sector",
                "electron_count": electrons,
                "spin_magnetization": magnetization,
                "charge_per_cell": charge,
                "background_policy": policy,
            }
        )

    def prepare(
        self, system: AtomisticSystemPlan | PreparedAtomisticSystem, /
    ) -> PreparedPeriodicElectronicSector:
        plan = system.plan if isinstance(system, PreparedAtomisticSystem) else system
        if not isinstance(plan, AtomisticSystemPlan):
            raise TypeError(
                "system must be AtomisticSystemPlan or PreparedAtomisticSystem."
            )
        return PreparedPeriodicElectronicSector(self, plan)


ElectronicSectorPlan = MolecularElectronicSectorPlan | PeriodicElectronicSectorPlan
PreparedElectronicSector = (
    PreparedMolecularElectronicSector | PreparedPeriodicElectronicSector
)


__all__ = [
    "ElectronicSectorPlan",
    "MolecularElectronicSectorPlan",
    "PeriodicElectronicSectorPlan",
    "PreparedElectronicSector",
    "PreparedMolecularElectronicSector",
    "PreparedPeriodicElectronicSector",
]
