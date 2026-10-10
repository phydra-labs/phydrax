#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ._calibration import calibrate_linear_material, LinearMaterialCalibrationResult
from ._fe2 import FE2Plan, FE2Result, LinearFE2Plan
from ._fft_homogenization import scalar_fft_effective_conductivity
from ._history import MaterialHistory
from ._homogenization import (
    HomogenizationBound,
    homogenize_scalar,
    reuss_tensor,
    TensorHomogenizationResult,
    voigt_tensor,
)
from ._icme import ICMEStepResult, SpatialICMEModel
from ._microstructure import GrainStructure
from ._precipitation import kwn_population_rate
from ._profiles import materials_candidate_profiles
from ._property import TabulatedProperty
from ._record import MaterialRecord
from ._rve import RepresentativeVolumeElement
from ._spatial import ConservativeMaterialTransfer, SpatialMaterialField
from ._state import MaterialState
from ._thermodynamics import (
    CallableEquilibriumProvider,
    EquilibriumProblem,
    EquilibriumResult,
)
from ._transfer import conservative_transfer_error, transfer_material_field
from ._transformation import jmak_fraction, koistinen_marburger_fraction
from ._validity import PropertyValidityRange, ValidityExceedance
from ._water import (
    iapws97_liquid_enthalpy,
    IAPWS97_LIQUID_RANGE,
    iapws97_liquid_state,
    iapws97_liquid_temperature,
    iapws97_region1_contains,
    iapws97_saturation_pressure,
    iapws97_saturation_temperature,
    iapws2008_viscosity,
    WaterLiquidState,
)


__all__ = [
    "IAPWS97_LIQUID_RANGE",
    "CallableEquilibriumProvider",
    "EquilibriumProblem",
    "EquilibriumResult",
    "ConservativeMaterialTransfer",
    "FE2Plan",
    "FE2Result",
    "GrainStructure",
    "ICMEStepResult",
    "LinearFE2Plan",
    "HomogenizationBound",
    "LinearMaterialCalibrationResult",
    "MaterialHistory",
    "MaterialRecord",
    "MaterialState",
    "PropertyValidityRange",
    "RepresentativeVolumeElement",
    "SpatialICMEModel",
    "SpatialMaterialField",
    "TensorHomogenizationResult",
    "TabulatedProperty",
    "ValidityExceedance",
    "WaterLiquidState",
    "calibrate_linear_material",
    "conservative_transfer_error",
    "homogenize_scalar",
    "jmak_fraction",
    "koistinen_marburger_fraction",
    "iapws2008_viscosity",
    "iapws97_liquid_enthalpy",
    "iapws97_liquid_state",
    "iapws97_liquid_temperature",
    "iapws97_region1_contains",
    "iapws97_saturation_pressure",
    "iapws97_saturation_temperature",
    "kwn_population_rate",
    "materials_candidate_profiles",
    "reuss_tensor",
    "scalar_fft_effective_conductivity",
    "transfer_material_field",
    "voigt_tensor",
]
