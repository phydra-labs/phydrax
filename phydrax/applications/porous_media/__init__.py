#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative porous flow, heat, transport, chemistry, and surface exchange."""

from ._advanced_chemistry import (
    HenryGasEquilibrium,
    IonExchangeEquilibrium,
    PitzerInteractionModel,
    RedoxEquilibrium,
    SITActivityModel,
)
from ._boundaries import PorousBoundaryConditions
from ._coupled import CoupledWaterHeatPlan
from ._cryosphere import (
    AtmosphericExchangePlan,
    AtmosphericExchangeResult,
    FreezeThawMaterial,
    VaporEquilibrium,
)
from ._fracture_exchange import ExchangeStep, FractureMatrixExchange
from ._fracture_network import (
    FractureNetworkState,
    FractureNetworkStepResult,
    MixedDimensionalFractureNetworkPlan,
)
from ._hysteresis import (
    BrooksCoreyRetention,
    DynamicCapillaryPressure,
    HysteresisState,
    HystereticRetentionPlan,
)
from ._materials import PorousMaterial
from ._mesh_qualification import (
    QualifiedVoroCrustPorousMesh,
    qualify_vorocrust_porous_mesh,
)
from ._monolithic_reactive import (
    MonolithicReactiveTransportPlan,
    MonolithicReactiveTransportResult,
)
from ._multiphase import (
    MultiphaseComponentState,
    MultiphaseConservationPlan,
    MultiphaseConservationResidual,
    MultiphaseFaceFluxes,
)
from ._phase_equilibrium import (
    CompositionalFlashPlan,
    PhaseEquilibriumResult,
    RachfordRiceFlashPlan,
)
from ._reactions import (
    MineralKinetics,
    MineralReactionResult,
    reactive_transport_step,
    ReactiveTransportStep,
)
from ._retention import VanGenuchtenMualem
from ._richards import RichardsPlan
from ._speciation import MassActionSystem, SpeciationResult
from ._state import PorousFluxes, PorousState, PorousStepResult, WaterHeatState
from ._surface_coupled import SurfacePorousResult, SurfaceRichardsPlan
from ._surface_exchange import (
    OrthogonalDiffusiveWaveSurfacePlan,
    SurfaceExchangeResult,
    SurfaceWaterState,
)
from ._surface_flow import (
    SurfaceFlowState,
    SurfaceFlowStepResult,
    UnstructuredShallowWaterPlan,
)
from ._surface_thermal_coupled import SurfaceWaterHeatPlan, SurfaceWaterHeatResult
from ._thermal import PorousHeatFluxes, PorousThermalMaterial
from ._transport import ComponentTransport, TransportBoundary, TransportStep
from ._wells import WellCompletionPlan, WellControl, WellResult


__all__ = [
    "ComponentTransport",
    "CoupledWaterHeatPlan",
    "ExchangeStep",
    "FractureMatrixExchange",
    "MassActionSystem",
    "MineralKinetics",
    "MineralReactionResult",
    "PorousBoundaryConditions",
    "PorousFluxes",
    "PorousHeatFluxes",
    "PorousMaterial",
    "PorousState",
    "PorousStepResult",
    "PorousThermalMaterial",
    "QualifiedVoroCrustPorousMesh",
    "ReactiveTransportStep",
    "RichardsPlan",
    "SpeciationResult",
    "SurfaceExchangeResult",
    "SurfacePorousResult",
    "SurfaceRichardsPlan",
    "OrthogonalDiffusiveWaveSurfacePlan",
    "SurfaceWaterHeatPlan",
    "SurfaceWaterHeatResult",
    "SurfaceWaterState",
    "TransportBoundary",
    "TransportStep",
    "VanGenuchtenMualem",
    "WaterHeatState",
    "qualify_vorocrust_porous_mesh",
    "reactive_transport_step",
    "AtmosphericExchangePlan",
    "AtmosphericExchangeResult",
    "FreezeThawMaterial",
    "VaporEquilibrium",
    "BrooksCoreyRetention",
    "DynamicCapillaryPressure",
    "HysteresisState",
    "HystereticRetentionPlan",
    "MultiphaseComponentState",
    "MultiphaseConservationPlan",
    "MultiphaseConservationResidual",
    "MultiphaseFaceFluxes",
    "CompositionalFlashPlan",
    "PhaseEquilibriumResult",
    "RachfordRiceFlashPlan",
    "SurfaceFlowState",
    "SurfaceFlowStepResult",
    "UnstructuredShallowWaterPlan",
    "WellCompletionPlan",
    "WellControl",
    "WellResult",
    "HenryGasEquilibrium",
    "IonExchangeEquilibrium",
    "PitzerInteractionModel",
    "RedoxEquilibrium",
    "SITActivityModel",
    "FractureNetworkState",
    "FractureNetworkStepResult",
    "MixedDimensionalFractureNetworkPlan",
    "MonolithicReactiveTransportPlan",
    "MonolithicReactiveTransportResult",
]
