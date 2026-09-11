#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-molecule PySCF provider with explicit units and convergence."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...units import BOHR, conversion_factor, ELEMENTARY_CHARGE, HARTREE
from .._calculation import ElectronicCalculationPlan, make_electronic_evaluation
from .._model import (
    ElectronicMethodFamily,
    ElectronicReferenceKind,
)
from .._properties import ElectronicProperty
from .._provider import (
    AbstractElectronicProvider,
    AbstractPreparedElectronicCalculation,
    ElectronicCapabilityError,
    ElectronicProviderCapabilities,
)
from .._result import (
    ElectronicCalculationStatus,
    ElectronicConvergenceEvidence,
    ElectronicEvaluation,
    ElectronicWorkEvidence,
)
from ._qcschema import _SYMBOLS


def is_pyscf_available() -> bool:
    return importlib.util.find_spec("pyscf") is not None


def require_pyscf() -> tuple[Any, Any, Any]:
    if not is_pyscf_available():
        raise ImportError("PySCF execution requires optional dependency 'pyscf'.")
    return (
        importlib.import_module("pyscf.gto"),
        importlib.import_module("pyscf.scf"),
        importlib.import_module("pyscf.dft"),
    )


class PreparedPySCFCalculation(AbstractPreparedElectronicCalculation):
    calculation: ElectronicCalculationPlan
    capabilities: ElectronicProviderCapabilities
    convergence_tolerance: float = eqx.field(static=True)
    maximum_cycles: int = eqx.field(static=True)
    provider_version: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        calculation: ElectronicCalculationPlan,
        capabilities: ElectronicProviderCapabilities,
        convergence_tolerance: float,
        maximum_cycles: int,
        provider_version: str,
        provider_id: str,
        /,
    ):
        if not isinstance(capabilities, ElectronicProviderCapabilities):
            raise TypeError("capabilities must be ElectronicProviderCapabilities.")
        self.calculation = calculation
        self.capabilities = capabilities
        self.convergence_tolerance = convergence_tolerance
        self.maximum_cycles = maximum_cycles
        self.provider_version = provider_version
        self.provider_id = provider_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-pyscf-calculation",
                "calculation": calculation.calculation_id,
                "capabilities": capabilities.capabilities_id,
                "provider": provider_id,
                "convergence_tolerance": convergence_tolerance,
                "maximum_cycles": maximum_cycles,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
    ) -> ElectronicEvaluation:
        if cell_vectors is not None:
            raise ValueError("PySCF molecular calculations do not accept cell vectors.")
        gto, scf, dft = require_pyscf()
        system = self.calculation.system
        coordinate = np.asarray(positions, dtype=np.dtype(system.coordinate_dtype))
        expected = (int(system.particle_ids.shape[0]), 3)
        if coordinate.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        active = np.asarray(system.active_mask, dtype=bool)
        numbers = np.asarray(system.atomic_numbers, dtype=np.int64)[active]
        if np.any(numbers <= 0) or np.any(numbers >= len(_SYMBOLS)):
            raise ValueError("PySCF provider requires supported positive atomic numbers.")
        length_to_bohr = float(
            conversion_factor(system.units.scale.length_unit, BOHR)
        )
        basis = self.calculation.model_chemistry.basis
        if basis is None:
            raise ElectronicCapabilityError("PySCF calculations require a basis set.")
        molecule = gto.Mole()
        molecule.atom = [
            (_SYMBOLS[int(number)], tuple(position))
            for number, position in zip(
                numbers, coordinate[active] * length_to_bohr, strict=True
            )
        ]
        molecule.unit = "Bohr"
        molecule.basis = basis.name
        molecule.charge = self.calculation.state.total_charge
        molecule.spin = self.calculation.state.spin_multiplicity - 1
        molecule.cart = basis.convention == "cartesian"
        molecule.verbose = 0
        molecule.build()
        method = self.calculation.model_chemistry.method
        if method.family is ElectronicMethodFamily.HARTREE_FOCK:
            mean_field = {
                ElectronicReferenceKind.RESTRICTED: scf.RHF,
                ElectronicReferenceKind.UNRESTRICTED: scf.UHF,
                ElectronicReferenceKind.RESTRICTED_OPEN_SHELL: scf.ROHF,
            }[method.reference](molecule)
        else:
            mean_field = {
                ElectronicReferenceKind.RESTRICTED: dft.RKS,
                ElectronicReferenceKind.UNRESTRICTED: dft.UKS,
                ElectronicReferenceKind.RESTRICTED_OPEN_SHELL: dft.ROKS,
            }[method.reference](molecule)
            mean_field.xc = method.method
        mean_field.conv_tol = self.convergence_tolerance
        mean_field.max_cycle = self.maximum_cycles
        energy_source = float(mean_field.kernel())
        converged = bool(mean_field.converged) and np.isfinite(energy_source)
        status = (
            ElectronicCalculationStatus.SUCCESS
            if converged
            else ElectronicCalculationStatus.ELECTRONIC_NOT_CONVERGED
        )
        energy_factor = float(
            conversion_factor(HARTREE, system.units.scale.energy_unit)
        )
        length_from_bohr = float(
            conversion_factor(BOHR, system.units.scale.length_unit)
        )
        forces = None
        request = self.calculation.request
        if request.requires(ElectronicProperty.FORCES):
            gradient = np.asarray(mean_field.nuc_grad_method().kernel(), dtype=float)
            if gradient.shape != (int(np.count_nonzero(active)), 3):
                raise ValueError("PySCF returned an invalid nuclear-gradient shape.")
            forces = np.zeros((active.size, 3), dtype=gradient.dtype)
            forces[active] = -gradient * energy_factor / length_from_bohr
        dipole = None
        if request.requires(ElectronicProperty.DIPOLE):
            dipole_source = np.asarray(
                mean_field.dip_moment(unit="AU", verbose=0), dtype=float
            ).reshape((3,))
            dipole = (
                dipole_source
                * float(conversion_factor(ELEMENTARY_CHARGE, system.units.charge_unit))
                * length_from_bohr
            )
        artifact_id = canonical_fingerprint(
            {
                "kind": "pyscf-electronic-evaluation",
                "provider_version": self.provider_version,
                "calculation": self.calculation.calculation_id,
                "energy_hartree": energy_source,
                "converged": converged,
            }
        )
        return make_electronic_evaluation(
            self.calculation,
            self.provider_id,
            coordinate,
            energy_source * energy_factor,
            forces=forces,
            dipole=dipole,
            convergence=ElectronicConvergenceEvidence(
                converged,
                message="pyscf-converged" if converged else "pyscf-not-converged",
            ),
            work=ElectronicWorkEvidence(
                energy_evaluations=1,
                force_evaluations=int(forces is not None),
                property_evaluations=int(dipole is not None),
            ),
            status=status,
            source_unit_ids=(
                ("charge", ELEMENTARY_CHARGE.unit_id),
                ("energy", HARTREE.unit_id),
                ("length", BOHR.unit_id),
            ),
            artifact_ids=(artifact_id,),
        )


class PySCFProvider(AbstractElectronicProvider):
    convergence_tolerance: float = eqx.field(static=True)
    maximum_cycles: int = eqx.field(static=True)
    provider_version: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    capabilities: ElectronicProviderCapabilities

    def __init__(
        self,
        *,
        convergence_tolerance: float = 1.0e-10,
        maximum_cycles: int = 100,
    ):
        tolerance = float(convergence_tolerance)
        cycles = int(maximum_cycles)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be finite and positive.")
        if cycles <= 0:
            raise ValueError("maximum_cycles must be positive.")
        available = is_pyscf_available()
        version = importlib.metadata.version("pyscf") if available else "unavailable"
        capabilities = ElectronicProviderCapabilities(
            (
                ElectronicProperty.ENERGY,
                ElectronicProperty.FORCES,
                ElectronicProperty.DIPOLE,
            ),
            (
                ElectronicReferenceKind.RESTRICTED,
                ElectronicReferenceKind.UNRESTRICTED,
                ElectronicReferenceKind.RESTRICTED_OPEN_SHELL,
            ),
            finite_geometry=True,
            conservative_forces=True,
            concurrency="process-isolated",
        )
        self.convergence_tolerance = tolerance
        self.maximum_cycles = cycles
        self.provider_version = version
        self.capabilities = capabilities
        self.provider_id = canonical_fingerprint(
            {
                "kind": "pyscf-provider",
                "version": version,
                "convergence_tolerance": tolerance,
                "maximum_cycles": cycles,
                "capabilities": capabilities.capabilities_id,
            }
        )

    def prepare(self, calculation: ElectronicCalculationPlan, /) -> PreparedPySCFCalculation:
        if not is_pyscf_available():
            raise ImportError("PySCF execution requires optional dependency 'pyscf'.")
        self.capabilities.require(calculation)
        model = calculation.model_chemistry
        if model.method.family not in (
            ElectronicMethodFamily.HARTREE_FOCK,
            ElectronicMethodFamily.KOHN_SHAM_DFT,
        ):
            raise ElectronicCapabilityError("PySCF provider supports HF and Kohn-Sham DFT.")
        if model.basis is None:
            raise ElectronicCapabilityError("PySCF provider requires a basis set.")
        if model.environment.kind != "vacuum":
            raise ElectronicCapabilityError(
                "PySCF provider currently supports the vacuum environment only."
            )
        if (
            model.method.definition_ids
            or model.correction_ids
            or model.relativistic_id is not None
            or model.numerical_definition_ids
            or model.model_artifact is not None
        ):
            raise ElectronicCapabilityError(
                "PySCF provider does not map additional method, correction, "
                "relativistic, numerical-definition, or model-artifact semantics."
            )
        if calculation.request.requires(ElectronicProperty.HESSIAN):
            raise ElectronicCapabilityError(
                "Use the native molecular Hessian workflow with PySCF forces."
            )
        return PreparedPySCFCalculation(
            calculation,
            self.capabilities,
            self.convergence_tolerance,
            self.maximum_cycles,
            self.provider_version,
            self.provider_id,
        )


__all__ = [
    "PreparedPySCFCalculation",
    "PySCFProvider",
    "is_pyscf_available",
    "require_pyscf",
]
