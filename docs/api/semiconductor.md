# Semiconductor applications

The [device guide](../guides_semiconductor.md) defines physical signs, units,
model/mesh admission, storage semantics, qualification, and examples.

## Support and physical device definition

::: phydrax.applications.semiconductor.TransportSupport

::: phydrax.applications.semiconductor.DevicePlan

::: phydrax.applications.semiconductor.MaterialBinding

::: phydrax.applications.semiconductor.OhmicContact

::: phydrax.applications.semiconductor.GateContact

## Materials and constitutive closures

::: phydrax.applications.semiconductor.SemiconductorMaterial

::: phydrax.applications.semiconductor.DielectricMaterial

::: phydrax.applications.semiconductor.ConstantMobility

::: phydrax.applications.semiconductor.DopingDependentMobility

::: phydrax.applications.semiconductor.srh_recombination

### Dimensional thermodynamics and state

::: phydrax.applications.semiconductor.BandThermodynamics

::: phydrax.applications.semiconductor.IncompleteIonization

::: phydrax.applications.semiconductor.SemiconductorStateLayout

### Material interfaces

::: phydrax.applications.semiconductor.MaterialInterface

::: phydrax.applications.semiconductor.ThermionicInterface

::: phydrax.applications.semiconductor.interface_electrostatics

::: phydrax.applications.semiconductor.InterfaceExchange

### High-field, thermal, carrier-energy, tunneling, and traps

::: phydrax.applications.semiconductor.LocalVelocitySaturation

::: phydrax.applications.semiconductor.LocalImpactIonization

::: phydrax.applications.semiconductor.ConstantLatticeHeatCapacity

::: phydrax.applications.semiconductor.ThermalConductance

::: phydrax.applications.semiconductor.ThermalBoundaryExchange

::: phydrax.applications.semiconductor.CarrierEnergyTransport

::: phydrax.applications.semiconductor.CarrierEnergyRelaxation

::: phydrax.applications.semiconductor.DynamicTrap

::: phydrax.applications.semiconductor.WKBBarrierPath

::: phydrax.applications.semiconductor.NonlocalTunnelingPath

## Conservative continuum execution

::: phydrax.applications.semiconductor.PreparedSemiconductorDevice

::: phydrax.applications.semiconductor.SemiconductorOperatingPoint

::: phydrax.applications.semiconductor.SemiconductorEvidence

::: phydrax.applications.semiconductor.SemiconductorSweepResult

## Physical example devices

::: phydrax.applications.semiconductor.pn_junction

::: phydrax.applications.semiconductor.mos_capacitor

::: phydrax.applications.semiconductor.bipolar_transistor

## Transient, small-signal, and implicit derivative analyses

::: phydrax.applications.semiconductor.semiconductor_transient

::: phydrax.applications.semiconductor.SemiconductorTransientResult

::: phydrax.applications.semiconductor.semiconductor_small_signal

::: phydrax.applications.semiconductor.SemiconductorSmallSignalResult

::: phydrax.applications.semiconductor.semiconductor_sensitivity

::: phydrax.applications.semiconductor.SemiconductorSensitivityResult

::: phydrax.applications.semiconductor.SemiconductorLinearEvidence

## Native circuit coupling

::: phydrax.applications.semiconductor.SemiconductorCircuitLaw

::: phydrax.applications.semiconductor.semiconductor_circuit_operating_point

::: phydrax.applications.semiconductor.semiconductor_circuit_transient

## Conservative mesh transitions

::: phydrax.applications.semiconductor.semiconductor_reprepare

::: phydrax.applications.semiconductor.SemiconductorTransferResult

::: phydrax.applications.semiconductor.SemiconductorTransferEvidence

::: phydrax.applications.semiconductor.semiconductor_adaptation_indicators

::: phydrax.applications.semiconductor.SemiconductorAdaptationIndicators

## Quantum electrostatics and transport

::: phydrax.applications.semiconductor.quantum.QuantumResources

::: phydrax.applications.semiconductor.quantum.EffectiveMass1D

::: phydrax.applications.semiconductor.quantum.DensityGradient1D

::: phydrax.applications.semiconductor.quantum.solve_schrodinger

::: phydrax.applications.semiconductor.quantum.QuantumPoisson1D

::: phydrax.applications.semiconductor.quantum.solve_schrodinger_poisson

::: phydrax.applications.semiconductor.quantum.SemiInfiniteLead

::: phydrax.applications.semiconductor.quantum.CoherentDevice

::: phydrax.applications.semiconductor.quantum.integrate_coherent

::: phydrax.applications.semiconductor.quantum.OpticalPhononBath

::: phydrax.applications.semiconductor.quantum.PhononEnergyGrid

::: phydrax.applications.semiconductor.quantum.solve_phonon_transport

::: phydrax.applications.semiconductor.quantum.QuantumInitialState

::: phydrax.applications.semiconductor.quantum.QuantumPulse

::: phydrax.applications.semiconductor.quantum.solve_quantum_transient

::: phydrax.applications.semiconductor.quantum.coherent_low_frequency_noise

::: phydrax.applications.semiconductor.quantum.QuantumCapacitance

::: phydrax.applications.semiconductor.quantum.finite_frequency_quantum_response

::: phydrax.applications.semiconductor.quantum.QuantumClassicalInterface

::: phydrax.applications.semiconductor.quantum.solve_quantum_classical_interface

## Inference and qualification

::: phydrax.applications.semiconductor.SemiconductorCalibrationDomain

::: phydrax.applications.semiconductor.SemiconductorMeasurementCase

::: phydrax.applications.semiconductor.SemiconductorCalibrationCampaign

::: phydrax.applications.semiconductor.SemiconductorParameterBinding

::: phydrax.applications.semiconductor.prepare_semiconductor_calibration

::: phydrax.applications.semiconductor.calibrate_semiconductor

::: phydrax.applications.semiconductor.qualify_semiconductor_calibration

::: phydrax.applications.semiconductor.archive_semiconductor_calibration
