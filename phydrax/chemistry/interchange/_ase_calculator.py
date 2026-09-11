#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit ASE-calculator execution without retaining foreign structure state."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure, AtomisticScaleContract
from ...units import (
    ANGSTROM,
    conversion_factor,
    DALTON,
    ELECTRONVOLT,
    ELEMENTARY_CHARGE,
)
from .._calculation import ElectronicCalculationPlan, make_electronic_evaluation
from .._model import ElectronicReferenceKind
from .._properties import ElectronicProperty
from .._provider import (
    AbstractElectronicProvider,
    AbstractPreparedElectronicCalculation,
    ElectronicProviderCapabilities,
)
from .._result import (
    ElectronicConvergenceEvidence,
    ElectronicEvaluation,
    ElectronicWorkEvidence,
)


ASESpinSemantics = Literal["multiplicity", "unpaired-electrons"]


def is_ase_calculator_available() -> bool:
    return importlib.util.find_spec("ase") is not None


def require_ase_calculator():
    if not is_ase_calculator_available():
        raise ImportError("ASE calculator execution requires optional dependency 'ase'.")
    return importlib.import_module("ase")


class ASEElectronicStateBinding(StrictModule, NonTrainableState):
    """Exact mapping of native total charge and spin into ``Atoms.info``."""

    charge_key: str | None = eqx.field(static=True)
    spin_key: str | None = eqx.field(static=True)
    spin_semantics: ASESpinSemantics = eqx.field(static=True)
    state_invariant: bool = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        charge_key: str | None,
        spin_key: str | None,
        spin_semantics: ASESpinSemantics = "multiplicity",
        state_invariant: bool = False,
    ):
        charge = None if charge_key is None else str(charge_key).strip()
        spin = None if spin_key is None else str(spin_key).strip()
        if charge_key is not None and not charge:
            raise ValueError("charge_key must be non-empty when provided.")
        if spin_key is not None and not spin:
            raise ValueError("spin_key must be non-empty when provided.")
        if spin_semantics not in ("multiplicity", "unpaired-electrons"):
            raise ValueError("Unknown ASE spin semantics.")
        invariant = bool(state_invariant)
        if invariant and (charge is not None or spin is not None):
            raise ValueError("State-invariant ASE bindings cannot also write state fields.")
        if not invariant and (charge is None or spin is None):
            raise ValueError("State-dependent ASE bindings require charge and spin keys.")
        self.charge_key = charge
        self.spin_key = spin
        self.spin_semantics = spin_semantics
        self.state_invariant = invariant
        self.binding_id = canonical_fingerprint(
            {
                "kind": "ase-electronic-state-binding",
                "charge_key": charge,
                "spin_key": spin,
                "spin_semantics": spin_semantics,
                "state_invariant": invariant,
            }
        )

    @classmethod
    def info(
        cls,
        /,
        *,
        charge_key: str = "charge",
        spin_key: str = "spin",
        spin_semantics: ASESpinSemantics = "multiplicity",
    ) -> ASEElectronicStateBinding:
        return cls(
            charge_key=charge_key,
            spin_key=spin_key,
            spin_semantics=spin_semantics,
        )

    @classmethod
    def invariant(cls) -> ASEElectronicStateBinding:
        return cls(charge_key=None, spin_key=None, state_invariant=True)

    def apply(self, atoms: Any, calculation: ElectronicCalculationPlan, /) -> None:
        if self.state_invariant:
            return
        spin = (
            calculation.state.spin_multiplicity
            if self.spin_semantics == "multiplicity"
            else calculation.state.spin_multiplicity - 1
        )
        atoms.info[self.charge_key] = calculation.state.total_charge
        atoms.info[self.spin_key] = spin


ASECalculatorFactory = Callable[[], Any]


class PreparedASECalculator(AbstractPreparedElectronicCalculation):
    calculation: ElectronicCalculationPlan
    capabilities: ElectronicProviderCapabilities
    calculator_factory: ASECalculatorFactory
    state_binding: ASEElectronicStateBinding
    provider_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        calculation: ElectronicCalculationPlan,
        capabilities: ElectronicProviderCapabilities,
        calculator_factory: ASECalculatorFactory,
        state_binding: ASEElectronicStateBinding,
        provider_id: str,
        /,
    ):
        if not isinstance(capabilities, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities.")
        self.calculation = calculation
        self.capabilities = capabilities
        self.calculator_factory = calculator_factory
        self.state_binding = state_binding
        self.provider_id = provider_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-ase-electronic-calculation",
                "calculation": calculation.calculation_id,
                "capabilities": capabilities.capabilities_id,
                "provider": provider_id,
                "state_binding": state_binding.binding_id,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> ElectronicEvaluation:
        ase = require_ase_calculator()
        system = self.calculation.system
        coordinate = np.asarray(positions, dtype=np.dtype(system.coordinate_dtype))
        expected = (int(system.particle_ids.shape[0]), 3)
        if coordinate.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        active = np.asarray(system.active_mask, dtype=bool)
        length_to_angstrom = float(
            conversion_factor(system.units.scale.length_unit, ANGSTROM)
        )
        source_cell = (
            system.cell.vectors if cell_vectors is None and system.cell is not None else cell_vectors
        )
        cell = (
            None
            if source_cell is None
            else np.asarray(source_cell, dtype=coordinate.dtype) * length_to_angstrom
        )
        periodic_axes = (
            None
            if cell is None
            else np.zeros((3,), dtype=bool)
            if system.cell is None
            else np.asarray(system.cell.periodic_axes, dtype=bool)
        )
        structure = AtomicStructure(
            np.asarray(system.atomic_numbers)[active],
            coordinate[active] * length_to_angstrom,
            np.asarray(system.masses)[active]
            * float(conversion_factor(system.units.mass_unit, DALTON)),
            AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
            particle_ids=np.asarray(system.particle_ids)[active],
            cell=cell,
            periodic_axes=periodic_axes,
            name=system.name,
        )
        atoms = ase.Atoms(
            numbers=np.asarray(structure.atomic_numbers),
            positions=np.asarray(structure.positions),
            masses=np.asarray(structure.masses),
            cell=(
                np.zeros((3, 3))
                if structure.cell is None
                else np.asarray(structure.cell)
            ),
            pbc=(
                np.zeros((3,), dtype=bool)
                if structure.periodic_axes is None
                else np.asarray(structure.periodic_axes)
            ),
        )
        self.state_binding.apply(atoms, self.calculation)
        calculator = self.calculator_factory()
        if calculator is None:
            raise TypeError("ASE calculator factory returned None.")
        atoms.calc = calculator
        energy_source = float(atoms.get_potential_energy())
        active_forces_source = np.asarray(atoms.get_forces(), dtype=float)
        if active_forces_source.shape != (int(np.count_nonzero(active)), 3):
            raise ValueError("ASE calculator returned an invalid force shape.")
        energy_factor = float(
            conversion_factor(ELECTRONVOLT, system.units.scale.energy_unit)
        )
        length_from_angstrom = float(
            conversion_factor(ANGSTROM, system.units.scale.length_unit)
        )
        force_factor = energy_factor / length_from_angstrom
        forces = np.zeros((active.size, 3), dtype=active_forces_source.dtype)
        forces[active] = active_forces_source * force_factor
        dipole = None
        if self.calculation.request.requires(ElectronicProperty.DIPOLE):
            dipole_source = np.asarray(atoms.get_dipole_moment(), dtype=float).reshape((3,))
            charge_factor = float(
                conversion_factor(ELEMENTARY_CHARGE, system.units.charge_unit)
            )
            dipole = dipole_source * charge_factor * length_from_angstrom
        calculator_id = canonical_fingerprint(
            {
                "kind": "ase-calculator-instance",
                "type": f"{type(calculator).__module__}.{type(calculator).__qualname__}",
                "ase_version": importlib.metadata.version("ase"),
            }
        )
        return make_electronic_evaluation(
            self.calculation,
            self.provider_id,
            coordinate,
            energy_source * energy_factor,
            forces=forces,
            dipole=dipole,
            cell_vectors=cell_vectors,
            convergence=ElectronicConvergenceEvidence(True, message="ase-calculator-complete"),
            work=ElectronicWorkEvidence(
                energy_evaluations=1,
                force_evaluations=1,
                property_evaluations=int(dipole is not None),
            ),
            source_unit_ids=(
                ("charge", ELEMENTARY_CHARGE.unit_id),
                ("energy", ELECTRONVOLT.unit_id),
                ("length", ANGSTROM.unit_id),
            ),
            artifact_ids=(calculator_id,),
        )


class ASECalculatorProvider(AbstractElectronicProvider):
    calculator_factory: ASECalculatorFactory
    state_binding: ASEElectronicStateBinding
    model_chemistry_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    capabilities: ElectronicProviderCapabilities

    def __init__(
        self,
        calculator_factory: ASECalculatorFactory,
        provider_name: str,
        state_binding: ASEElectronicStateBinding,
        /,
        *,
        model_chemistry_id: str,
        capabilities: ElectronicProviderCapabilities | None = None,
    ):
        if not callable(calculator_factory):
            raise TypeError("calculator_factory must be callable.")
        name = str(provider_name).strip()
        if not name:
            raise ValueError("provider_name must be non-empty.")
        if not isinstance(state_binding, ASEElectronicStateBinding):
            raise TypeError("state_binding must be ASEElectronicStateBinding.")
        model_id = str(model_chemistry_id).strip()
        if not model_id:
            raise ValueError("model_chemistry_id must be non-empty.")
        capabilities_ = (
            ElectronicProviderCapabilities(
                (ElectronicProperty.ENERGY, ElectronicProperty.FORCES),
                (
                    ElectronicReferenceKind.RESTRICTED,
                    ElectronicReferenceKind.UNRESTRICTED,
                ),
                total_charge=not state_binding.state_invariant,
                spin_multiplicity=not state_binding.state_invariant,
                conservative_forces=True,
                concurrency="serial",
            )
            if capabilities is None
            else capabilities
        )
        if not isinstance(capabilities_, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities or None.")
        if ElectronicProperty.HESSIAN in capabilities_.properties:
            raise ValueError(
                "ASE calculator Hessians must use the native force-difference workflow."
            )
        if state_binding.state_invariant and (
            capabilities_.total_charge or capabilities_.spin_multiplicity
        ):
            raise ValueError(
                "State-invariant ASE bindings must reject charged and open-shell systems."
            )
        self.calculator_factory = calculator_factory
        self.state_binding = state_binding
        self.model_chemistry_id = model_id
        self.capabilities = capabilities_
        self.provider_id = canonical_fingerprint(
            {
                "kind": "ase-calculator-provider",
                "name": name,
                "state_binding": state_binding.binding_id,
                "capabilities": capabilities_.capabilities_id,
                "model_chemistry": model_id,
            }
        )

    def prepare(self, calculation: ElectronicCalculationPlan, /) -> PreparedASECalculator:
        if not is_ase_calculator_available():
            raise ImportError("ASE calculator execution requires optional dependency 'ase'.")
        self.capabilities.require(calculation)
        if (
            calculation.model_chemistry.model_chemistry_id
            != self.model_chemistry_id
        ):
            raise ValueError("ASE calculator provider is bound to another model chemistry.")
        return PreparedASECalculator(
            calculation,
            self.capabilities,
            self.calculator_factory,
            self.state_binding,
            self.provider_id,
        )


__all__ = [
    "ASECalculatorProvider",
    "ASEElectronicStateBinding",
    "PreparedASECalculator",
    "is_ase_calculator_available",
    "require_ase_calculator",
]
