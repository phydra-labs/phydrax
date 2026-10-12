# Geophysics applications

The namespace contains shared atmosphere/ocean geophysical bindings and the
modality-specific solid-Earth and subsurface packages below. See the
[native geophysics guide](../../guides_geophysics.md) for the capability matrix,
derivative scope, and explicit non-claims.

## Shared geophysical data and observation contracts

`GeophysicalFieldBinding.value_port()` derives the `ValuePort` of a state-layout
binding: its semantic identity is the bound quantity ID, its component IDs are the
selected state components, and every component carries the quantity's declared
dimension. Field-space and operator-task bindings take their ports from the native
storage owner.

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

`TravelTimeGraphPlan` constructs a symmetrized directed relation and delegates
`length / wavespeed` costs to `ShortestPathFieldPlan(method="host-dijkstra")`.
This method has an intentional SciPy host synchronization through
`jax.pure_callback`, including under JIT; it is not a device-native eikonal
solver. Travel-time gradients use the selected acyclic forest's device VJP.
`converged` means certified success; `derivative_available` also requires every
reachable inherited path margin to exceed `path_margin_tolerance_s`. Equal
incoming candidates are retained in the alternative-path comparison, so an
exact tie, or ambiguity inherited from a parent, suppresses the regularity
claim while preserving the deterministic selected Clarke subgradient.
The source is fixed, so its returned `path_margin_s` is `+inf` before applying
the tolerance; incoming cycles at this root do not constrain descendant margins.
`TravelTimeResult` preserves `field_status`, `bellman_residual`, `dual_residual`,
and `reachable` from the owning field certificate rather than reducing failure
evidence to a Boolean. The field plan applies work and peak host-byte admission
before its callback, including when invoked through this travel-time adapter.


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
