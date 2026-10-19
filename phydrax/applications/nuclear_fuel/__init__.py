#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bound nuclear fuel performance on Phydrax-native substrates.

Public names resolve lazily from `_EXPORTS`, so qualification discovery does
not load constitutive models, nonlinear execution, or observation providers.
The `TYPE_CHECKING` imports declare the same names for static tooling.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from ..._jax_config import process_numerics


if TYPE_CHECKING:
    from ._acceptance import (
        FuelAcceptancePolicy as FuelAcceptancePolicy,
        FuelBalanceEvidence as FuelBalanceEvidence,
        FuelCouplingEvidence as FuelCouplingEvidence,
        FuelLinearEvidence as FuelLinearEvidence,
        FuelNonlinearEvidence as FuelNonlinearEvidence,
        FuelPhysicalEvidence as FuelPhysicalEvidence,
        FuelSolverEvidence as FuelSolverEvidence,
        FuelSourceSupport as FuelSourceSupport,
        FuelStepEvidence as FuelStepEvidence,
        FuelStepStatus as FuelStepStatus,
    )
    from ._bundles import (
        BoreEmissivity as BoreEmissivity,
        CoulombFriction as CoulombFriction,
        EnergyDeposition as EnergyDeposition,
        FuelApplicabilityPolicy as FuelApplicabilityPolicy,
        FuelModelBundle as FuelModelBundle,
        HeliumSource as HeliumSource,
        PlenumTemperatureClosure as PlenumTemperatureClosure,
        RelocationHistory as RelocationHistory,
    )
    from ._calibration import (
        FuelCalibrationEvaluation as FuelCalibrationEvaluation,
        FuelCalibrationParameters as FuelCalibrationParameters,
        FuelCalibrationPlan as FuelCalibrationPlan,
        FuelCalibrationRod as FuelCalibrationRod,
        FuelDenseSignalJacobian as FuelDenseSignalJacobian,
        FuelSignalDerivatives as FuelSignalDerivatives,
        FuelSignalJacobianActions as FuelSignalJacobianActions,
        prepare_fuel_least_squares as prepare_fuel_least_squares,
        prepare_fuel_posterior as prepare_fuel_posterior,
    )
    from ._cladding_deformation import (
        ZircaloyCreep as ZircaloyCreep,
        ZircaloyPlasticity as ZircaloyPlasticity,
    )
    from ._coolant import (
        PrescribedWallTemperature as PrescribedWallTemperature,
        SingleChannelCoolant as SingleChannelCoolant,
    )
    from ._coupled_coolant import (
        CoupledChannelCoolant as CoupledChannelCoolant,
    )
    from ._ensembles import (
        FuelEnsembleMember as FuelEnsembleMember,
        FuelEnsemblePlan as FuelEnsemblePlan,
        FuelEnsembleResult as FuelEnsembleResult,
    )
    from ._execution import (
        fuel_numerical_execution_id as fuel_numerical_execution_id,
        FuelNumericalBinding as FuelNumericalBinding,
        FuelNumericalExecution as FuelNumericalExecution,
    )
    from ._fission_gas import (
        AthermalRelease as AthermalRelease,
        FissionGasModel as FissionGasModel,
    )
    from ._gap import (
        GapConductance as GapConductance,
        GapConductanceLaw as GapConductanceLaw,
        GasAccommodationLaw as GasAccommodationLaw,
    )
    from ._gas_block import (
        advance_fuel_gas as advance_fuel_gas,
        FuelGasBlockResult as FuelGasBlockResult,
    )
    from ._gas_state import (
        FISSION_GAS_TRACER_SPECIES as FISSION_GAS_TRACER_SPECIES,
        FissionGasDiagnostics as FissionGasDiagnostics,
        FissionGasSiteState as FissionGasSiteState,
        FissionGasSpeciesState as FissionGasSpeciesState,
        FissionGasSpeciesStepResult as FissionGasSpeciesStepResult,
        FissionGasSpeciesTransfers as FissionGasSpeciesTransfers,
        FissionGasStepResult as FissionGasStepResult,
        FissionGasTransfers as FissionGasTransfers,
    )
    from ._grain_boundary import (
        AinscoughGrainGrowth as AinscoughGrainGrowth,
        GrainBoundaryMicrocracking as GrainBoundaryMicrocracking,
        GrainFaceBubbles as GrainFaceBubbles,
    )
    from ._hbs import (
        HighBurnupStructureFormation as HighBurnupStructureFormation,
        HighBurnupStructurePores as HighBurnupStructurePores,
    )
    from ._helium import (
        HeliumGasModel as HeliumGasModel,
        HeliumGasSiteState as HeliumGasSiteState,
        HeliumGasStepResult as HeliumGasStepResult,
        HeliumTransportTable as HeliumTransportTable,
    )
    from ._identifiability import (
        evaluate_fuel_calibration_identifiability as evaluate_fuel_calibration_identifiability,
        evaluate_fuel_identifiability as evaluate_fuel_identifiability,
        FuelIdentifiabilityReport as FuelIdentifiabilityReport,
    )
    from ._identity import (
        AxialInteraction as AxialInteraction,
        CladdingAlloy as CladdingAlloy,
        CladdingHeatTreatment as CladdingHeatTreatment,
        CoolantBoundaryKind as CoolantBoundaryKind,
        FUEL_GAS_SPECIES as FUEL_GAS_SPECIES,
        FuelForm as FuelForm,
        FuelGasSpecies as FuelGasSpecies,
        FuelGeometryReduction as FuelGeometryReduction,
    )
    from ._intragranular import (
        SpeightTrapping as SpeightTrapping,
        TurnbullDiffusivity as TurnbullDiffusivity,
    )
    from ._inventory import (
        FuelEnergyLedger as FuelEnergyLedger,
        FuelGasLedger as FuelGasLedger,
    )
    from ._irradiation import (
        IrradiationConditions as IrradiationConditions,
        IrradiationHistory as IrradiationHistory,
    )
    from ._layout import (
        FuelSourceSlot as FuelSourceSlot,
    )
    from ._observables import (
        FuelGasAncestry as FuelGasAncestry,
        FuelObservableKind as FuelObservableKind,
        FuelObservationPlan as FuelObservationPlan,
        FuelObservationResult as FuelObservationResult,
        FuelSensor as FuelSensor,
        FuelSensorBody as FuelSensorBody,
        FuelSensorParameters as FuelSensorParameters,
        thermocouple_response as thermocouple_response,
    )
    from ._problem import (
        admit_fuel_rod as admit_fuel_rod,
        FuelRodAdmission as FuelRodAdmission,
        FuelRodCandidateResult as FuelRodCandidateResult,
        prepare_fuel_rod as prepare_fuel_rod,
        PreparedFuelRod as PreparedFuelRod,
    )
    from ._qualification import (
        nuclear_fuel_execution_profile as nuclear_fuel_execution_profile,
        nuclear_fuel_profiles as nuclear_fuel_profiles,
    )
    from ._radial_power import (
        RadialPowerProfileModel as RadialPowerProfileModel,
        RadialPowerState as RadialPowerState,
    )
    from ._result import (
        FuelRodResult as FuelRodResult,
    )
    from ._sensitivity import (
        fuel_parameterization as fuel_parameterization,
        FuelFixedGrid as FuelFixedGrid,
        FuelParameterization as FuelParameterization,
        FuelParameterName as FuelParameterName,
        FuelReplayResult as FuelReplayResult,
        FuelSensitivityPlan as FuelSensitivityPlan,
        FuelSensitivityResult as FuelSensitivityResult,
        replay_fuel_grid as replay_fuel_grid,
    )
    from ._simulation import (
        FuelRodSimulation as FuelRodSimulation,
    )
    from ._specification import (
        CladdingSpecification as CladdingSpecification,
        FuelRodSpecification as FuelRodSpecification,
        PelletSpecification as PelletSpecification,
        RodSpecification as RodSpecification,
    )
    from ._state import (
        FuelRodFields as FuelRodFields,
        FuelRodState as FuelRodState,
    )


# Public name -> defining submodule; `__all__` follows this order.
_EXPORTS: dict[str, str] = {
    "AinscoughGrainGrowth": "_grain_boundary",
    "AthermalRelease": "_fission_gas",
    "AxialInteraction": "_identity",
    "BoreEmissivity": "_bundles",
    "CladdingAlloy": "_identity",
    "CladdingHeatTreatment": "_identity",
    "CladdingSpecification": "_specification",
    "CoolantBoundaryKind": "_identity",
    "CoulombFriction": "_bundles",
    "CoupledChannelCoolant": "_coupled_coolant",
    "EnergyDeposition": "_bundles",
    "FISSION_GAS_TRACER_SPECIES": "_gas_state",
    "FUEL_GAS_SPECIES": "_identity",
    "FissionGasDiagnostics": "_gas_state",
    "FissionGasModel": "_fission_gas",
    "FissionGasSiteState": "_gas_state",
    "FissionGasSpeciesState": "_gas_state",
    "FissionGasSpeciesStepResult": "_gas_state",
    "FissionGasSpeciesTransfers": "_gas_state",
    "FissionGasStepResult": "_gas_state",
    "FissionGasTransfers": "_gas_state",
    "FuelAcceptancePolicy": "_acceptance",
    "FuelApplicabilityPolicy": "_bundles",
    "FuelBalanceEvidence": "_acceptance",
    "FuelCalibrationEvaluation": "_calibration",
    "FuelCalibrationParameters": "_calibration",
    "FuelCalibrationPlan": "_calibration",
    "FuelCalibrationRod": "_calibration",
    "FuelCouplingEvidence": "_acceptance",
    "FuelDenseSignalJacobian": "_calibration",
    "FuelEnergyLedger": "_inventory",
    "FuelEnsembleMember": "_ensembles",
    "FuelEnsemblePlan": "_ensembles",
    "FuelEnsembleResult": "_ensembles",
    "FuelFixedGrid": "_sensitivity",
    "FuelForm": "_identity",
    "FuelGasAncestry": "_observables",
    "FuelGasBlockResult": "_gas_block",
    "FuelGasLedger": "_inventory",
    "FuelGasSpecies": "_identity",
    "FuelGeometryReduction": "_identity",
    "FuelIdentifiabilityReport": "_identifiability",
    "FuelLinearEvidence": "_acceptance",
    "FuelModelBundle": "_bundles",
    "FuelNonlinearEvidence": "_acceptance",
    "FuelNumericalBinding": "_execution",
    "FuelNumericalExecution": "_execution",
    "FuelObservableKind": "_observables",
    "FuelObservationPlan": "_observables",
    "FuelObservationResult": "_observables",
    "FuelParameterName": "_sensitivity",
    "FuelParameterization": "_sensitivity",
    "FuelPhysicalEvidence": "_acceptance",
    "FuelReplayResult": "_sensitivity",
    "FuelRodAdmission": "_problem",
    "FuelRodCandidateResult": "_problem",
    "FuelRodFields": "_state",
    "FuelRodResult": "_result",
    "FuelRodSimulation": "_simulation",
    "FuelRodSpecification": "_specification",
    "FuelRodState": "_state",
    "FuelSensitivityPlan": "_sensitivity",
    "FuelSensitivityResult": "_sensitivity",
    "FuelSensor": "_observables",
    "FuelSensorBody": "_observables",
    "FuelSensorParameters": "_observables",
    "FuelSignalDerivatives": "_calibration",
    "FuelSignalJacobianActions": "_calibration",
    "FuelSolverEvidence": "_acceptance",
    "FuelSourceSlot": "_layout",
    "FuelSourceSupport": "_acceptance",
    "FuelStepEvidence": "_acceptance",
    "FuelStepStatus": "_acceptance",
    "GapConductance": "_gap",
    "GapConductanceLaw": "_gap",
    "GasAccommodationLaw": "_gap",
    "GrainBoundaryMicrocracking": "_grain_boundary",
    "GrainFaceBubbles": "_grain_boundary",
    "HeliumGasModel": "_helium",
    "HeliumGasSiteState": "_helium",
    "HeliumGasStepResult": "_helium",
    "HeliumSource": "_bundles",
    "HeliumTransportTable": "_helium",
    "HighBurnupStructureFormation": "_hbs",
    "HighBurnupStructurePores": "_hbs",
    "IrradiationConditions": "_irradiation",
    "IrradiationHistory": "_irradiation",
    "PelletSpecification": "_specification",
    "PlenumTemperatureClosure": "_bundles",
    "PreparedFuelRod": "_problem",
    "PrescribedWallTemperature": "_coolant",
    "RadialPowerProfileModel": "_radial_power",
    "RadialPowerState": "_radial_power",
    "RelocationHistory": "_bundles",
    "RodSpecification": "_specification",
    "SingleChannelCoolant": "_coolant",
    "SpeightTrapping": "_intragranular",
    "TurnbullDiffusivity": "_intragranular",
    "ZircaloyCreep": "_cladding_deformation",
    "ZircaloyPlasticity": "_cladding_deformation",
    "admit_fuel_rod": "_problem",
    "advance_fuel_gas": "_gas_block",
    "evaluate_fuel_calibration_identifiability": "_identifiability",
    "evaluate_fuel_identifiability": "_identifiability",
    "fuel_numerical_execution_id": "_execution",
    "fuel_parameterization": "_sensitivity",
    "nuclear_fuel_execution_profile": "_qualification",
    "nuclear_fuel_profiles": "_qualification",
    "prepare_fuel_least_squares": "_calibration",
    "prepare_fuel_posterior": "_calibration",
    "prepare_fuel_rod": "_problem",
    "replay_fuel_grid": "_sensitivity",
    "thermocouple_response": "_observables",
}


__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    with process_numerics():
        if name not in _EXPORTS:
            raise AttributeError(f"{__name__!r} has no public capability {name!r}.")
        module = import_module(f"{__name__}.{_EXPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
