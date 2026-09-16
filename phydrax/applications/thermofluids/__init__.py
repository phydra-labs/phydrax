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
from ._hydraulics import (
    hydraulic_channel_component,
    hydraulic_compliance_component,
    hydraulic_flow_boundary_component,
    hydraulic_inertance_component,
    hydraulic_junction_component,
    hydraulic_pressure_boundary_component,
    HydraulicChannelPlan,
    HydraulicFluidProperties,
    HydraulicLawEvaluation,
    HydraulicPressureDropPlan,
    HydraulicReason,
    HydraulicReducedResponsePlan,
    MonotoneHydraulicResponsePlan,
)
from ._material import (
    homogeneous_fluid_heat_exchanger_component,
    material_boundary_component,
    material_mixer_component,
)
from ._process import (
    fixed_material_boundary_component,
    HeatFlowOrientation,
    HydraulicPortSpec,
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
    "HydraulicChannelPlan",
    "HydraulicFluidProperties",
    "HydraulicLawEvaluation",
    "HydraulicPortSpec",
    "HydraulicPressureDropPlan",
    "HydraulicReason",
    "HydraulicReducedResponsePlan",
    "MonotoneHydraulicResponsePlan",
    "ResistiveHeatingLaw",
    "MaterialFlowDirection",
    "ThermofluidComponent",
    "ThermofluidConnection",
    "ThermofluidPortKind",
    "ThermofluidPortSpec",
    "ThermofluidProcessPlan",
    "fixed_material_boundary_component",
    "heat_conversion_component",
    "hydraulic_channel_component",
    "hydraulic_compliance_component",
    "hydraulic_flow_boundary_component",
    "hydraulic_inertance_component",
    "hydraulic_junction_component",
    "hydraulic_pressure_boundary_component",
    "homogeneous_fluid_heat_exchanger_component",
    "material_boundary_component",
    "material_mixer_component",
    "temperature_boundary_component",
    "thermal_capacitance_component",
    "thermal_conductor_component",
    "isenthalpic_valve_component",
]
