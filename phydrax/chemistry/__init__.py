#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed molecular computational chemistry and provider-neutral workflows."""

from . import interchange
from ._atomistic import (
    AtomisticPotentialEnergySurface,
    ExternalAtomisticPotentialEnergySurface,
    SurfaceExternalAtomisticProvider,
)
from ._calculation import (
    electronic_geometry_id,
    ElectronicCalculationPlan,
    make_electronic_evaluation,
)
from ._derivatives import MolecularHessianPlan, MolecularHessianResult
from ._lifecycle import (
    chemistry_lifecycle,
    ChemistryRunEnvelope,
    read_electronic_result_archive,
    write_electronic_result_archive,
)
from ._model import (
    BasisSetReference,
    ElectronicEnvironmentPlan,
    ElectronicMethodFamily,
    ElectronicMethodPlan,
    ElectronicModelChemistryPlan,
    ElectronicReferenceKind,
)
from ._optimization import (
    MolecularGeometryConvergencePlan,
    MolecularGeometryOptimizationPlan,
    MolecularGeometryOptimizationResult,
)
from ._properties import ElectronicProperty, ElectronicPropertyRequest
from ._provider import (
    AbstractElectronicProvider,
    AbstractPreparedElectronicCalculation,
    CallableElectronicProvider,
    CallablePreparedElectronicCalculation,
    ElectronicCapabilityError,
    ElectronicConcurrencyKind,
    ElectronicExecutionKind,
    ElectronicProviderCapabilities,
    ElectronicProviderUnavailableError,
)
from ._result import (
    ElectronicCalculationStatus,
    ElectronicConvergenceEvidence,
    ElectronicEnergyEvaluation,
    ElectronicEnergyForceEvaluation,
    ElectronicEnergyForceHessianEvaluation,
    ElectronicEvaluation,
    ElectronicEvaluationHeader,
    ElectronicGroundStatePropertyEvaluation,
    ElectronicWorkEvidence,
)
from ._spectroscopy import IRSpectrumPlan, IRSpectrumResult
from ._state import MolecularElectronicStatePlan, PreparedMolecularElectronicState
from ._surface import (
    AbstractPreparedPotentialEnergySurface,
    CallablePotentialEnergySurface,
    CompositePotentialEnergySurface,
    ElectronicPotentialEnergySurface,
    PotentialEnergySurfaceCapabilities,
    PotentialEnergySurfaceEvaluation,
)
from ._thermochemistry import (
    HarmonicThermochemistryPlan,
    MolarThermochemistryResult,
    MolecularThermochemistryResult,
    to_molar_thermochemistry,
)
from ._units import (
    angular_frequency_to_wavenumber,
    ChemistryPhysicalConstants,
    dipole_derivative_unit,
    dipole_unit,
    entropy_unit,
    hessian_unit,
)
from ._vibration import (
    StationaryPointKind,
    VibrationalAnalysisPlan,
    VibrationalAnalysisResult,
)


__all__ = [
    "AbstractElectronicProvider",
    "AbstractPreparedElectronicCalculation",
    "AbstractPreparedPotentialEnergySurface",
    "AtomisticPotentialEnergySurface",
    "BasisSetReference",
    "CallableElectronicProvider",
    "CallablePotentialEnergySurface",
    "CallablePreparedElectronicCalculation",
    "ChemistryPhysicalConstants",
    "ChemistryRunEnvelope",
    "CompositePotentialEnergySurface",
    "ElectronicCalculationPlan",
    "ElectronicCalculationStatus",
    "ElectronicCapabilityError",
    "ElectronicConcurrencyKind",
    "ElectronicConvergenceEvidence",
    "ElectronicEnergyEvaluation",
    "ElectronicEnergyForceEvaluation",
    "ElectronicEnergyForceHessianEvaluation",
    "ElectronicEnvironmentPlan",
    "ElectronicEvaluation",
    "ElectronicEvaluationHeader",
    "ElectronicExecutionKind",
    "ElectronicGroundStatePropertyEvaluation",
    "ElectronicMethodFamily",
    "ElectronicMethodPlan",
    "ElectronicModelChemistryPlan",
    "ElectronicPotentialEnergySurface",
    "ElectronicProperty",
    "ElectronicPropertyRequest",
    "ElectronicProviderCapabilities",
    "ElectronicProviderUnavailableError",
    "ElectronicReferenceKind",
    "ElectronicWorkEvidence",
    "ExternalAtomisticPotentialEnergySurface",
    "HarmonicThermochemistryPlan",
    "IRSpectrumPlan",
    "IRSpectrumResult",
    "MolarThermochemistryResult",
    "MolecularElectronicStatePlan",
    "MolecularGeometryConvergencePlan",
    "MolecularGeometryOptimizationPlan",
    "MolecularGeometryOptimizationResult",
    "MolecularHessianPlan",
    "MolecularHessianResult",
    "MolecularThermochemistryResult",
    "PotentialEnergySurfaceCapabilities",
    "PotentialEnergySurfaceEvaluation",
    "PreparedMolecularElectronicState",
    "StationaryPointKind",
    "SurfaceExternalAtomisticProvider",
    "VibrationalAnalysisPlan",
    "VibrationalAnalysisResult",
    "angular_frequency_to_wavenumber",
    "chemistry_lifecycle",
    "dipole_derivative_unit",
    "dipole_unit",
    "electronic_geometry_id",
    "entropy_unit",
    "hessian_unit",
    "interchange",
    "make_electronic_evaluation",
    "read_electronic_result_archive",
    "to_molar_thermochemistry",
    "write_electronic_result_archive",
]
