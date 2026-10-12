# Nuclear fuel performance

`phydrax.applications.nuclear_fuel` simulates one UO₂/Zircaloy-4 fuel rod
through a prescribed irradiation history. The capability is
candidate-unreleased; its verification is synthetic and numerical. See
[the workflow guide](../../guides_nuclear_fuel.md) and
[model sources](../../nuclear_fuel_sources.md).

## Specification and fabrication

::: phydrax.applications.nuclear_fuel.FuelRodSpecification

---

::: phydrax.applications.nuclear_fuel.PelletSpecification

---

::: phydrax.applications.nuclear_fuel.CladdingSpecification

---

::: phydrax.applications.nuclear_fuel.RodSpecification

---

::: phydrax.applications.nuclear_fuel.FuelForm

---

::: phydrax.applications.nuclear_fuel.CladdingAlloy

---

::: phydrax.applications.nuclear_fuel.CladdingHeatTreatment

---

::: phydrax.applications.nuclear_fuel.FuelGasSpecies

---

::: phydrax.applications.nuclear_fuel.FUEL_GAS_SPECIES

## Irradiation history

::: phydrax.applications.nuclear_fuel.IrradiationHistory

---

::: phydrax.applications.nuclear_fuel.IrradiationConditions

## Coolant boundaries

::: phydrax.applications.nuclear_fuel.PrescribedWallTemperature

---

::: phydrax.applications.nuclear_fuel.SingleChannelCoolant

---

::: phydrax.applications.nuclear_fuel.CoupledChannelCoolant

---

::: phydrax.applications.nuclear_fuel.CoolantBoundaryKind

## Model bundle

::: phydrax.applications.nuclear_fuel.FuelModelBundle

---

::: phydrax.applications.nuclear_fuel.FuelApplicabilityPolicy

---

::: phydrax.applications.nuclear_fuel.RadialPowerProfileModel

---

::: phydrax.applications.nuclear_fuel.RadialPowerState

---

::: phydrax.applications.nuclear_fuel.GapConductanceLaw

---

::: phydrax.applications.nuclear_fuel.GasAccommodationLaw

---

::: phydrax.applications.nuclear_fuel.GapConductance

---

::: phydrax.applications.nuclear_fuel.ZircaloyCreep

---

::: phydrax.applications.nuclear_fuel.ZircaloyPlasticity

### Closure records

::: phydrax.applications.nuclear_fuel.HeliumSource

---

::: phydrax.applications.nuclear_fuel.RelocationHistory

---

::: phydrax.applications.nuclear_fuel.BoreEmissivity

---

::: phydrax.applications.nuclear_fuel.PlenumTemperatureClosure

---

::: phydrax.applications.nuclear_fuel.EnergyDeposition

---

::: phydrax.applications.nuclear_fuel.CoulombFriction

## Fission gas and helium

::: phydrax.applications.nuclear_fuel.FissionGasModel

---

::: phydrax.applications.nuclear_fuel.AthermalRelease

---

::: phydrax.applications.nuclear_fuel.TurnbullDiffusivity

---

::: phydrax.applications.nuclear_fuel.SpeightTrapping

---

::: phydrax.applications.nuclear_fuel.GrainFaceBubbles

---

::: phydrax.applications.nuclear_fuel.AinscoughGrainGrowth

---

::: phydrax.applications.nuclear_fuel.GrainBoundaryMicrocracking

---

::: phydrax.applications.nuclear_fuel.HighBurnupStructureFormation

---

::: phydrax.applications.nuclear_fuel.HighBurnupStructurePores

---

::: phydrax.applications.nuclear_fuel.FissionGasSiteState

---

::: phydrax.applications.nuclear_fuel.FissionGasStepResult

---

::: phydrax.applications.nuclear_fuel.FissionGasTransfers

---

::: phydrax.applications.nuclear_fuel.FissionGasDiagnostics

---

::: phydrax.applications.nuclear_fuel.FISSION_GAS_TRACER_SPECIES

---

::: phydrax.applications.nuclear_fuel.FissionGasSpeciesState

---

::: phydrax.applications.nuclear_fuel.FissionGasSpeciesTransfers

---

::: phydrax.applications.nuclear_fuel.FissionGasSpeciesStepResult

---

::: phydrax.applications.nuclear_fuel.HeliumGasModel

---

::: phydrax.applications.nuclear_fuel.HeliumTransportTable

---

::: phydrax.applications.nuclear_fuel.HeliumGasSiteState

---

::: phydrax.applications.nuclear_fuel.HeliumGasStepResult

---

::: phydrax.applications.nuclear_fuel.advance_fuel_gas

---

::: phydrax.applications.nuclear_fuel.FuelGasBlockResult

## Admission and preparation

::: phydrax.applications.nuclear_fuel.admit_fuel_rod

---

::: phydrax.applications.nuclear_fuel.FuelRodAdmission

---

::: phydrax.applications.nuclear_fuel.FuelGeometryReduction

---

::: phydrax.applications.nuclear_fuel.AxialInteraction

---

::: phydrax.applications.nuclear_fuel.prepare_fuel_rod

---

::: phydrax.applications.nuclear_fuel.PreparedFuelRod

---

::: phydrax.applications.nuclear_fuel.FuelRodCandidateResult

## Simulation and results

::: phydrax.applications.nuclear_fuel.FuelRodSimulation

---

::: phydrax.applications.nuclear_fuel.FuelAcceptancePolicy

---

::: phydrax.applications.nuclear_fuel.FuelRodResult

---

::: phydrax.applications.nuclear_fuel.FuelRodState

---

::: phydrax.applications.nuclear_fuel.FuelRodFields

---

::: phydrax.applications.nuclear_fuel.FuelGasLedger

---

::: phydrax.applications.nuclear_fuel.FuelEnergyLedger

---

::: phydrax.applications.nuclear_fuel.FuelNumericalExecution

---

::: phydrax.applications.nuclear_fuel.FuelNumericalBinding

---

::: phydrax.applications.nuclear_fuel.fuel_numerical_execution_id

### Step evidence

::: phydrax.applications.nuclear_fuel.FuelStepEvidence

---

::: phydrax.applications.nuclear_fuel.FuelStepStatus

---

::: phydrax.applications.nuclear_fuel.FuelSolverEvidence

---

::: phydrax.applications.nuclear_fuel.FuelNonlinearEvidence

---

::: phydrax.applications.nuclear_fuel.FuelLinearEvidence

---

::: phydrax.applications.nuclear_fuel.FuelCouplingEvidence

---

::: phydrax.applications.nuclear_fuel.FuelPhysicalEvidence

---

::: phydrax.applications.nuclear_fuel.FuelBalanceEvidence

---

::: phydrax.applications.nuclear_fuel.FuelSourceSupport

---

::: phydrax.applications.nuclear_fuel.FuelSourceSlot

## Observations

::: phydrax.applications.nuclear_fuel.FuelObservationPlan

---

::: phydrax.applications.nuclear_fuel.FuelSensor

---

::: phydrax.applications.nuclear_fuel.FuelObservableKind

---

::: phydrax.applications.nuclear_fuel.FuelSensorBody

---

::: phydrax.applications.nuclear_fuel.FuelSensorParameters

---

::: phydrax.applications.nuclear_fuel.FuelGasAncestry

---

::: phydrax.applications.nuclear_fuel.FuelObservationResult

---

::: phydrax.applications.nuclear_fuel.thermocouple_response

## Sensitivities

::: phydrax.applications.nuclear_fuel.FuelFixedGrid

---

::: phydrax.applications.nuclear_fuel.replay_fuel_grid

---

::: phydrax.applications.nuclear_fuel.FuelReplayResult

---

::: phydrax.applications.nuclear_fuel.fuel_parameterization

---

::: phydrax.applications.nuclear_fuel.FuelParameterName

---

::: phydrax.applications.nuclear_fuel.FuelParameterization

---

::: phydrax.applications.nuclear_fuel.FuelSensitivityPlan

---

::: phydrax.applications.nuclear_fuel.FuelSensitivityResult

## Calibration and identifiability

::: phydrax.applications.nuclear_fuel.FuelCalibrationPlan

---

::: phydrax.applications.nuclear_fuel.FuelCalibrationRod

---

::: phydrax.applications.nuclear_fuel.FuelCalibrationParameters

---

::: phydrax.applications.nuclear_fuel.FuelCalibrationEvaluation

---

::: phydrax.applications.nuclear_fuel.prepare_fuel_least_squares

---

::: phydrax.applications.nuclear_fuel.prepare_fuel_posterior

---

::: phydrax.applications.nuclear_fuel.FuelSignalDerivatives

---

::: phydrax.applications.nuclear_fuel.FuelSignalJacobianActions

---

::: phydrax.applications.nuclear_fuel.FuelDenseSignalJacobian

---

::: phydrax.applications.nuclear_fuel.evaluate_fuel_calibration_identifiability

---

::: phydrax.applications.nuclear_fuel.evaluate_fuel_identifiability

---

::: phydrax.applications.nuclear_fuel.FuelIdentifiabilityReport

## Ensembles

::: phydrax.applications.nuclear_fuel.FuelEnsemblePlan

---

::: phydrax.applications.nuclear_fuel.FuelEnsembleMember

---

::: phydrax.applications.nuclear_fuel.FuelEnsembleResult

## Qualification profiles

::: phydrax.applications.nuclear_fuel.nuclear_fuel_profiles

---

::: phydrax.applications.nuclear_fuel.nuclear_fuel_execution_profile
