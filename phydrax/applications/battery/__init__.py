#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Battery equation models with explicit development and authenticated release admission."""

from ._admission import (
    battery_release_deployment_record,
    BatteryExecutionAdmission,
    issue_battery_execution_admission,
    load_battery_execution_admission,
)
from ._circuit_ecm import (
    CircuitConnectedEcmAdapter,
    CircuitConnectedEcmInitialCondition,
    CircuitConnectedEcmLedger,
    CircuitConnectedEcmPlan,
    CircuitConnectedEcmStateView,
    CircuitEcmImplicitLaw,
    PreparedCircuitConnectedEcm,
)
from ._ecm import (
    PreparedThermalEquivalentCircuit,
    ThermalEquivalentCircuitAdapter,
    ThermalEquivalentCircuitInitialCondition,
    ThermalEquivalentCircuitLedger,
    ThermalEquivalentCircuitParameters,
    ThermalEquivalentCircuitPlan,
    ThermalEquivalentCircuitState,
)
from ._experiment import (
    BatteryDAESolvePlan,
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
    prepare_battery_experiment,
    PreparedBatteryExperiment,
    run_battery_experiment,
)
from ._particle import BatteryParticlePlan, PreparedBatteryParticle
from ._properties import (
    ConcentrationTemperaturePropertyLaw,
    ConstantPropertyLaw,
    TabulatedPropertyLaw,
)
from ._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    BatteryTerminalConvention,
    CurrentStepPlan,
    CurrentStopGuard,
    PASSIVE_TERMINAL_CONVENTION,
    RestStepPlan,
    StoichiometryStopGuard,
    TemperatureStopGuard,
    VoltageStopGuard,
)
from ._qualification import (
    build_thermal_ecm_release_profile,
    CIRCUIT_ECM_CANDIDATE,
    CIRCUIT_ECM_SUPPORT,
    MARQUIS_2019_SPME_CANDIDATE,
    MARQUIS_2019_SPME_SUPPORT,
    require_released_battery_profile,
    THERMAL_ECM_CANDIDATE,
    THERMAL_ECM_SUPPORT,
)
from ._release import BatteryReleaseBundle, BatteryReleaseRecord, build_battery_release
from ._results import (
    BatteryDAEReplayEvidence,
    BatteryDAERestartEvidence,
    BatteryDAESolution,
    BatteryExperimentResult,
    BatteryRunStatus,
    BatteryTermination,
)
from ._spm import PrescribedCurrentSpmPlan, SpmInitialCondition, SpmParameters
from ._spme_marquis2019 import (
    Marquis2019SpmeAdapter,
    Marquis2019SpmeInitialCondition,
    Marquis2019SpmeLedger,
    Marquis2019SpmeParameters,
    Marquis2019SpmePlan,
    Marquis2019SpmeState,
    PreparedMarquis2019Spme,
)
from ._through_cell import PreparedThroughCellMesh, ThroughCellRegionPlan
from ._validity import (
    BatteryValidityEnvelope,
    CIRCUIT_ECM_ENVELOPE,
    MARQUIS_2019_SPME_ENVELOPE,
    THERMAL_ECM_ENVELOPE,
)


__all__ = [
    "BatteryDAEReplayEvidence",
    "BatteryDAERestartEvidence",
    "BatteryDAESolution",
    "BatteryDAESolvePlan",
    "BatteryExecutionAdmission",
    "BatteryParticlePlan",
    "BatteryReleaseBundle",
    "BatteryReleaseRecord",
    "BatteryValidityEnvelope",
    "CIRCUIT_ECM_CANDIDATE",
    "CIRCUIT_ECM_ENVELOPE",
    "CIRCUIT_ECM_SUPPORT",
    "CircuitConnectedEcmAdapter",
    "CircuitConnectedEcmInitialCondition",
    "CircuitConnectedEcmLedger",
    "CircuitConnectedEcmPlan",
    "CircuitConnectedEcmStateView",
    "CircuitEcmImplicitLaw",
    "ConcentrationTemperaturePropertyLaw",
    "MARQUIS_2019_SPME_CANDIDATE",
    "MARQUIS_2019_SPME_ENVELOPE",
    "MARQUIS_2019_SPME_SUPPORT",
    "Marquis2019SpmeAdapter",
    "Marquis2019SpmeInitialCondition",
    "Marquis2019SpmeLedger",
    "Marquis2019SpmeParameters",
    "Marquis2019SpmePlan",
    "Marquis2019SpmeState",
    "PreparedBatteryParticle",
    "PreparedCircuitConnectedEcm",
    "PreparedMarquis2019Spme",
    "PreparedThroughCellMesh",
    "PrescribedCurrentSpmPlan",
    "SpmInitialCondition",
    "SpmParameters",
    "StoichiometryStopGuard",
    "THERMAL_ECM_ENVELOPE",
    "ThroughCellRegionPlan",
    "battery_release_deployment_record",
    "build_battery_release",
    "issue_battery_execution_admission",
    "load_battery_execution_admission",
    "PASSIVE_TERMINAL_CONVENTION",
    "THERMAL_ECM_CANDIDATE",
    "THERMAL_ECM_SUPPORT",
    "BatteryDiffraxSolvePlan",
    "BatteryExperimentPlan",
    "BatteryExperimentResult",
    "BatteryOutputPlan",
    "BatteryProtocolPlan",
    "BatteryProtocolValues",
    "BatteryRunStatus",
    "BatteryTerminalConvention",
    "BatteryTermination",
    "ConstantPropertyLaw",
    "CurrentStepPlan",
    "CurrentStopGuard",
    "PreparedBatteryExperiment",
    "PreparedThermalEquivalentCircuit",
    "RestStepPlan",
    "TabulatedPropertyLaw",
    "TemperatureStopGuard",
    "ThermalEquivalentCircuitAdapter",
    "ThermalEquivalentCircuitInitialCondition",
    "ThermalEquivalentCircuitLedger",
    "ThermalEquivalentCircuitParameters",
    "ThermalEquivalentCircuitPlan",
    "ThermalEquivalentCircuitState",
    "VoltageStopGuard",
    "build_thermal_ecm_release_profile",
    "prepare_battery_experiment",
    "require_released_battery_profile",
    "run_battery_experiment",
]
