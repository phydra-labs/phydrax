# Battery applications

The public namespace exposes thermal ECM, canonical isothermal SPMe, and one-cell
circuit ECM construction. Availability is not release authorization. Production
execution requires the exact authenticated distribution/support/envelope and
runtime attestation; an explicit candidate profile is development-only.
See [thermal ECM equations](../../guides_battery.md) and
[production admission](../../guides_battery_production.md).
Positive current enters the positive terminal and physical inputs use SI units.
DFN and pack construction remain behind authenticated entry decisions; no
safety, lifetime, fast-charge, or arbitrary-cell predictive claim is made.

## Terminal convention and protocol

::: phydrax.applications.battery.BatteryTerminalConvention

---

::: phydrax.applications.battery.PASSIVE_TERMINAL_CONVENTION

---

::: phydrax.applications.battery.CurrentStepPlan

---

::: phydrax.applications.battery.RestStepPlan

---

::: phydrax.applications.battery.VoltageStopGuard

---

::: phydrax.applications.battery.CurrentStopGuard

---

::: phydrax.applications.battery.TemperatureStopGuard

---

::: phydrax.applications.battery.BatteryProtocolPlan

---

::: phydrax.applications.battery.BatteryProtocolValues

## Property laws

::: phydrax.applications.battery.ConstantPropertyLaw

---

::: phydrax.applications.battery.TabulatedPropertyLaw

## Thermal equivalent circuit

::: phydrax.applications.battery.ThermalEquivalentCircuitPlan

---

::: phydrax.applications.battery.PreparedThermalEquivalentCircuit

---

::: phydrax.applications.battery.ThermalEquivalentCircuitParameters

---

::: phydrax.applications.battery.ThermalEquivalentCircuitInitialCondition

---

::: phydrax.applications.battery.ThermalEquivalentCircuitState

---

::: phydrax.applications.battery.ThermalEquivalentCircuitAdapter

---

::: phydrax.applications.battery.ThermalEquivalentCircuitLedger

## Circuit-connected ECM

::: phydrax.applications.battery.CircuitConnectedEcmPlan

::: phydrax.applications.battery.PreparedCircuitConnectedEcm

::: phydrax.applications.battery.CircuitEcmImplicitLaw

::: phydrax.applications.battery.CircuitConnectedEcmInitialCondition

::: phydrax.applications.battery.CircuitConnectedEcmStateView

::: phydrax.applications.battery.CircuitConnectedEcmAdapter

::: phydrax.applications.battery.CircuitConnectedEcmLedger

## Canonical isothermal SPMe

::: phydrax.applications.battery.Marquis2019SpmePlan

::: phydrax.applications.battery.PreparedMarquis2019Spme

::: phydrax.applications.battery.Marquis2019SpmeParameters

::: phydrax.applications.battery.Marquis2019SpmeInitialCondition

::: phydrax.applications.battery.Marquis2019SpmeState

::: phydrax.applications.battery.Marquis2019SpmeAdapter

::: phydrax.applications.battery.Marquis2019SpmeLedger

::: phydrax.applications.battery.SpmParameters

::: phydrax.applications.battery.ConcentrationTemperaturePropertyLaw

## Experiment execution and results

::: phydrax.applications.battery.BatteryOutputPlan

---

::: phydrax.applications.battery.BatteryDiffraxSolvePlan

::: phydrax.applications.battery.BatteryDAESolvePlan

::: phydrax.applications.battery.BatteryDAESolution

::: phydrax.applications.battery.BatteryDAERestartEvidence

::: phydrax.applications.battery.BatteryDAEReplayEvidence

---

::: phydrax.applications.battery.BatteryExperimentPlan

---

::: phydrax.applications.battery.PreparedBatteryExperiment

---

::: phydrax.applications.battery.prepare_battery_experiment

---

::: phydrax.applications.battery.run_battery_experiment

---

::: phydrax.applications.battery.BatteryExperimentResult

---

::: phydrax.applications.battery.BatteryRunStatus

---

::: phydrax.applications.battery.BatteryTermination

## Exact support and release admission

Persisted criterion, campaign, observation, evidence, and gate times are UTC
Unix-epoch nanoseconds. Any monotonic timing used by the campaign runner is
process-local validation state and is not serialized.

::: phydrax.applications.battery.THERMAL_ECM_SUPPORT

---

::: phydrax.applications.battery.THERMAL_ECM_CANDIDATE

---

::: phydrax.applications.battery.build_thermal_ecm_release_profile

---

::: phydrax.applications.battery.require_released_battery_profile

::: phydrax.applications.battery.BatteryValidityEnvelope

::: phydrax.applications.battery.BatteryReleaseBundle

::: phydrax.applications.battery.BatteryReleaseRecord

::: phydrax.applications.battery.build_battery_release

::: phydrax.applications.battery.BatteryExecutionAdmission

::: phydrax.applications.battery.issue_battery_execution_admission

::: phydrax.applications.battery.load_battery_execution_admission

::: phydrax.qualification.RuntimeDistributionAttestation

::: phydrax.qualification.QualificationRoleTrust

::: phydrax.qualification.AsymmetricReleaseTrustPolicy

::: phydrax.qualification.PromotionState

::: phydrax.qualification.PromotionRepository
