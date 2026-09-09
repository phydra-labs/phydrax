#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native acausal thermofluid components and fixed-grid spatial design workflows."""

from ._heat import (
    ConstantCOPHeatPumpLaw,
    heat_conversion_component,
    HeatConversionEvaluation,
    HeatConversionLaw,
    HeatPortBridge,
    ResistiveHeatingLaw,
    temperature_boundary_component,
    thermal_capacitance_component,
    thermal_conductor_component,
)
from ._material import (
    homogeneous_fluid_heat_exchanger_component,
    material_boundary_component,
    material_mixer_component,
)
from ._process import (
    fixed_material_boundary_component,
    HeatFlowOrientation,
    isenthalpic_valve_component,
    MaterialFlowDirection,
    ThermofluidComponent,
    ThermofluidConnection,
    ThermofluidPortKind,
    ThermofluidPortSpec,
    ThermofluidProcessPlan,
)
from ._topology_design import (
    ThermofluidMaterial,
    ThermofluidTopologyDesign,
    ThermofluidTopologyEvidence,
    ThermofluidTopologyReanalysis,
)
from ._turbomachinery import (
    CompressorDesignArtifact,
    CompressorEvaluation,
    CompressorMapEvaluation,
    CompressorMapPlan,
    CompressorPlan,
    GasStation,
)


__all__ = [
    "ThermofluidMaterial",
    "ThermofluidTopologyDesign",
    "ThermofluidTopologyEvidence",
    "ThermofluidTopologyReanalysis",
    "CompressorDesignArtifact",
    "CompressorEvaluation",
    "CompressorMapEvaluation",
    "CompressorMapPlan",
    "CompressorPlan",
    "GasStation",
    "ConstantCOPHeatPumpLaw",
    "HeatConversionEvaluation",
    "HeatConversionLaw",
    "HeatFlowOrientation",
    "HeatPortBridge",
    "ResistiveHeatingLaw",
    "MaterialFlowDirection",
    "ThermofluidComponent",
    "ThermofluidConnection",
    "ThermofluidPortKind",
    "ThermofluidPortSpec",
    "ThermofluidProcessPlan",
    "fixed_material_boundary_component",
    "heat_conversion_component",
    "homogeneous_fluid_heat_exchanger_component",
    "material_boundary_component",
    "material_mixer_component",
    "temperature_boundary_component",
    "thermal_capacitance_component",
    "thermal_conductor_component",
    "isenthalpic_valve_component",
]
