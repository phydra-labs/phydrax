# Geophysics applications

The namespace contains shared atmosphere/ocean geophysical bindings and the
modality-specific solid-Earth and subsurface packages below. See the
[native geophysics guide](../../guides_geophysics.md) for the capability matrix,
derivative scope, and explicit non-claims.

## Shared geophysical data and observation contracts

::: phydrax.applications.geophysics.GeophysicalFieldBinding

::: phydrax.applications.geophysics.GeophysicalQuantity

::: phydrax.applications.geophysics.GeophysicalTimeSpec

::: phydrax.applications.geophysics.GeophysicalData

::: phydrax.applications.geophysics.GeophysicalObservationOperator

::: phydrax.applications.geophysics.GeophysicalObservationPolicy

::: phydrax.applications.geophysics.PreparedGeophysicalObservations

## Electrical and induced-polarization models

::: phydrax.applications.geophysics.electrical
    options:
      show_root_heading: true
      members: true

## Electromagnetic models

::: phydrax.applications.geophysics.electromagnetics
    options:
      show_root_heading: true
      members: true

## Gravity and magnetic potential fields

::: phydrax.applications.geophysics.potential_fields
    options:
      show_root_heading: true
      members: true

## Seismic models, imaging, and source workflows

::: phydrax.applications.geophysics.seismic
    options:
      show_root_heading: true
      members: true

## Petrophysical and geological composition

::: phydrax.applications.geophysics.petrophysics
    options:
      show_root_heading: true
      members: true

## Deformation, faults, and geodynamics

::: phydrax.applications.geophysics.deformation
    options:
      show_root_heading: true
      members: true

## Joint inference

::: phydrax.applications.geophysics.IndependentModalityTerm

::: phydrax.applications.geophysics.StructuralCrossGradientCoupling

::: phydrax.applications.geophysics.PetrophysicalDiscrepancyCoupling

::: phydrax.applications.geophysics.SharedInterfaceCoupling

::: phydrax.applications.geophysics.MultimodalJointInferencePlan

::: phydrax.applications.geophysics.MatrixFreeMAPPlan

::: phydrax.applications.geophysics.EnsembleKalmanInversionPlan

::: phydrax.applications.geophysics.PCNSampler

## Time-lapse monitoring

::: phydrax.applications.geophysics.MonitoringEpoch

::: phydrax.applications.geophysics.TimeLapseParameterization

::: phydrax.applications.geophysics.MonitoringState

::: phydrax.applications.geophysics.MonitoringStepResult

::: phydrax.applications.geophysics.SequentialMonitoringPlan

## Planetary extensions

::: phydrax.applications.geophysics.RadialBodyModel

::: phydrax.applications.geophysics.SphericalRayPlan

::: phydrax.applications.geophysics.PlanetaryPotentialPlan

::: phydrax.applications.geophysics.RadialThermalConductionPlan

## Capability, resource, and restart evidence

::: phydrax.applications.geophysics.GeophysicalCapabilityEvidence

::: phydrax.applications.geophysics.GeophysicalResourceEstimate

::: phydrax.applications.geophysics.GeophysicalResourcePolicy

::: phydrax.applications.geophysics.GeophysicalContinuationState

::: phydrax.applications.geophysics.GeophysicalCheckpointPlan
