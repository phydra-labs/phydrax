#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""ASE calculator over one native atomistic model.

This module imports ASE at load time; the interchange facade resolves it
lazily, so Phydrax itself never requires ASE.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np
from ase import Atoms
from ase.calculators.calculator import (
    all_changes,
    BaseCalculator,
    CalculationFailed,
    PropertyNotImplementedError,
)
from ase.stress import full_3x3_to_voigt_6_stress

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PeriodicCell
from ...discretization.particle._verlet import ParticleImageVerletState
from ...typing import checked
from ...units import (
    ANGSTROM,
    conversion_factor,
    DALTON,
    derived_unit,
    ELECTRONVOLT,
    SI_REFERENCE_SYSTEM_ID,
)
from .._born_oppenheimer import (
    NativeAtomisticEvaluation,
    NativeAtomisticProvider,
    NativeAtomisticProviderPlan,
    NativeProviderNeighborhoodState,
)
from .._system import AtomisticSystemPlan
from .._types import AtomisticScaleContract
from .._units import AtomisticUnitSystem
from ._ase import from_ase_atoms


_ASE_SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
_ASE_PRESSURE = derived_unit("eV/Ang^3", ((ELECTRONVOLT, 1), (ANGSTROM, -3)))


class NativeASECalculatorPlan(StrictModule, NonTrainableState):
    """Bind one native provider recipe and unit system to ASE structures.

    ASE positions and cells in angstrom and masses in dalton are converted
    exactly into ``units``; ``pbc`` axes become periodic cell axes, and a
    structure without periodic axes is finite regardless of its ASE box.
    """

    provider: NativeAtomisticProviderPlan
    units: AtomisticUnitSystem
    coordinate_dtype: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        provider: NativeAtomisticProviderPlan,
        units: AtomisticUnitSystem,
        /,
        *,
        coordinate_dtype: str = "float64",
    ) -> None:
        if provider.model.scale.scale_id != units.scale.scale_id:
            raise ValueError("Model and atomistic unit system scales differ.")
        if units.scale.length_unit.reference_system_id != SI_REFERENCE_SYSTEM_ID:
            raise ValueError("ASE calculation requires SI-convertible atomistic units.")
        dtype = np.dtype(coordinate_dtype)
        if dtype.kind != "f":
            raise TypeError("coordinate_dtype must be a real floating dtype.")
        self.provider = provider
        self.units = units
        self.coordinate_dtype = dtype.name
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-ase-calculator-plan",
                "provider": provider.plan_id,
                "units": units.unit_system_id,
                "coordinate_dtype": dtype.name,
            }
        )

    @property
    def length_factor(self) -> float:
        """Angstrom to native length."""

        return float(conversion_factor(ANGSTROM, self.units.scale.length_unit))

    def prepare(self, atoms: Atoms, /) -> NativeAtomisticProvider:
        """Prepare the native system and provider for one structure."""

        structure, _ = from_ase_atoms(atoms, _ASE_SCALE, name="ase-calculator-atoms")
        length = self.length_factor
        mass = float(conversion_factor(DALTON, self.units.mass_unit))
        axes = (
            (False, False, False)
            if structure.periodic_axes is None
            else tuple(bool(axis) for axis in np.asarray(structure.periodic_axes))
        )
        cell = (
            PeriodicCell(np.asarray(structure.cell) * length, periodic_axes=axes)
            if structure.cell is not None and any(axes)
            else None
        )
        system = AtomisticSystemPlan(
            structure.particle_ids,
            structure.atomic_numbers,
            np.asarray(structure.masses) * mass,
            self.units,
            cell=cell,
            name=structure.name,
            coordinate_dtype=self.coordinate_dtype,
        ).prepare()
        return self.provider.prepare(system)


class NativeASECalculator(BaseCalculator):
    """ASE calculator evaluating one native model at eV/Angstrom boundaries.

    ``energy`` and ``free_energy`` are the same potential energy (a classical
    surface carries no electronic entropy), ``energies`` the per-atom
    decomposition summing to it, ``forces`` the conservative negative gradient,
    and ``stress`` the tensile Voigt stress ``(1/V) dE/d(strain)`` in eV/Ang^3,
    available only for fully periodic cells. An unavailable stress raises
    ``PropertyNotImplementedError``; no zero is substituted.

    Results are invalidated by ASE on position, number, cell, PBC, charge, or
    moment changes, and by ``update_plan`` on model revision or preparation
    changes. Number or PBC changes re-prepare the system; positions and cells
    advance the cached Verlet state through its certificate, and a failed
    evaluation is retried once from a fresh preparation at the current
    geometry before ``CalculationFailed`` is raised.
    """

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(
        self, plan: NativeASECalculatorPlan, /, *, artifact_id: str | None = None
    ) -> None:
        if not isinstance(plan, NativeASECalculatorPlan):
            raise TypeError("plan must be a NativeASECalculatorPlan.")
        super().__init__()
        self.plan = plan
        self.artifact_id = artifact_id
        self.provider: NativeAtomisticProvider | None = None
        self.neighborhood_state: NativeProviderNeighborhoodState | None = None
        self._structure_key: tuple[bytes, tuple[bool, ...]] | None = None
        self.preparation_count = 0
        self.provenance: dict[str, str | int | None] = {}

    def update_plan(self, plan: NativeASECalculatorPlan, /) -> None:
        """Bind a new model revision or preparation and drop every cache."""

        if not isinstance(plan, NativeASECalculatorPlan):
            raise TypeError("plan must be a NativeASECalculatorPlan.")
        self.plan = plan
        self.provider = None
        self.neighborhood_state = None
        self._structure_key = None
        self.atoms = None
        self.results = {}
        self.provenance = {}

    def _prepare(self, atoms: Atoms, /) -> NativeAtomisticProvider:
        provider = self.plan.prepare(atoms)
        self.provider = provider
        self.neighborhood_state = None
        self.preparation_count += 1
        return provider

    def _provider(self, atoms: Atoms, /) -> NativeAtomisticProvider:
        key = (
            np.asarray(atoms.numbers, dtype=np.int64).tobytes(),
            tuple(bool(axis) for axis in atoms.pbc),
        )
        if self.provider is None or key != self._structure_key:
            provider = self._prepare(atoms)
            self._structure_key = key
            return provider
        return self.provider

    def _evaluate(
        self, provider: NativeAtomisticProvider, atoms: Atoms, /
    ) -> NativeAtomisticEvaluation:
        length = self.plan.length_factor
        positions = np.asarray(atoms.positions, dtype=np.float64) * length
        cell = (
            None
            if provider.program.system.cell is None
            else np.asarray(atoms.cell.array, dtype=np.float64) * length
        )
        return provider.evaluate_state(positions, cell, self.neighborhood_state)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | tuple[str, ...] = ("energy",),
        system_changes: list[str] = all_changes,
    ) -> None:
        del system_changes
        target = self.atoms if atoms is None else atoms
        if not isinstance(target, Atoms):
            raise TypeError("NativeASECalculator requires an ase.Atoms structure.")
        provider = self._provider(target)
        value = self._evaluate(provider, target)
        if not bool(value.evaluation.successful):
            provider = self._prepare(target)
            value = self._evaluate(provider, target)
        if not bool(value.evaluation.successful):
            self.neighborhood_state = None
            state = value.request.neighborhood
            capacity = (
                bool(state.capacity_failure)
                if isinstance(state, ParticleImageVerletState)
                else None
            )
            raise CalculationFailed(
                "Native evaluation failed after a fresh preparation "
                f"(neighborhood capacity failure: {capacity})."
            )
        evaluation = value.evaluation
        if "stress" in properties and evaluation.stress is None:
            raise PropertyNotImplementedError(
                "Stress requires a fully periodic cell and a cell-differentiable model."
            )
        self.neighborhood_state = value.request.neighborhood
        scale = self.plan.units.scale
        energy_factor = float(conversion_factor(scale.energy_unit, ELECTRONVOLT))
        force_factor = float(conversion_factor(scale.force_unit, _ASE_SCALE.force_unit))
        energy = float(np.asarray(evaluation.energy, dtype=np.float64)) * energy_factor
        results: dict[str, float | np.ndarray] = {
            "energy": energy,
            "free_energy": energy,
            "energies": np.asarray(value.atom_energy, dtype=np.float64) * energy_factor,
            "forces": np.asarray(evaluation.forces, dtype=np.float64) * force_factor,
        }
        if evaluation.stress is not None:
            pressure = float(
                conversion_factor(self.plan.units.pressure_unit, _ASE_PRESSURE)
            )
            results["stress"] = full_3x3_to_voigt_6_stress(
                np.asarray(evaluation.stress, dtype=np.float64) * pressure
            )
        self.results = results
        self.provenance = {
            "plan": self.plan.plan_id,
            "model_revision": self.plan.provider.model_revision_id,
            "artifact": self.artifact_id,
            "provider": provider.provider_id,
            "neighborhood_epoch": int(np.asarray(value.request.neighborhood.epoch)),
            "preparation_count": self.preparation_count,
        }


__all__ = ["NativeASECalculator", "NativeASECalculatorPlan"]
