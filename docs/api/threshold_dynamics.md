# Threshold dynamics

`phydrax.threshold_dynamics` advances hard multiphase label fields by heat-kernel
thresholding (MBO / Esedoglu–Otto / Salvador–Esedoglu), with exact integer label
volumes by capacitated auction, a sparse candidate-label route for many labels,
a mesh heat-action route, and gas-diffusive dry-foam coarsening. See the
[threshold dynamics guide](../guides_threshold_dynamics.md) for the model,
accuracy, statuses and nonclaims. The pairwise tension and mobility contracts
(`InterfaceTensionMatrix`, `InterfaceMobilityMatrix`) are documented with
[interfacial transport](surface_thin_films.md); the capacitated auction with
[combinatorial optimization](combinatorial.md).

## Plan, routes and state

::: phydrax.threshold_dynamics.ThresholdDynamicsPlan

::: phydrax.threshold_dynamics.PreparedThresholdDynamics

`ThresholdRoute` is an `AbstractThresholdHeatKernel` or `SparseLabelGrid`.

::: phydrax.threshold_dynamics.AbstractThresholdHeatKernel

::: phydrax.threshold_dynamics.PeriodicGridHeatKernel

::: phydrax.threshold_dynamics.MeshHeatKernel

::: phydrax.threshold_dynamics.SparseLabelGrid

::: phydrax.threshold_dynamics.LabelFieldState

::: phydrax.threshold_dynamics.ThresholdPotentials

`ThresholdKernelForm` is the closed selector `"single-gaussian" | "two-gaussian"`.

::: phydrax.threshold_dynamics.ThresholdKernelDecomposition

::: phydrax.threshold_dynamics.decompose_threshold_kernel

::: phydrax.threshold_dynamics.ThresholdDynamicsResourcePolicy

## Explicit-surface seeding

The threshold facade re-exports the geometry-owned extraction contracts for a
one-way C-to-E seed/repair transaction. Full API details are under
[multiregion surfaces](multiregion_surfaces.md#hard-label-extraction).

::: phydrax.threshold_dynamics.LabelFieldSurfaceExtractionPlan

::: phydrax.threshold_dynamics.PreparedLabelFieldSurfaceExtraction

::: phydrax.threshold_dynamics.LabelFieldSurfaceExtractionResult

## Volume constraints and coarsening

::: phydrax.threshold_dynamics.LabelVolumeConstraint

::: phydrax.threshold_dynamics.GasDiffusionCoarsening

::: phydrax.threshold_dynamics.GasDiffusionRunResult

::: phydrax.threshold_dynamics.GasDiffusionEvidence

## Results, evidence and status

::: phydrax.threshold_dynamics.ThresholdDynamicsStepResult

::: phydrax.threshold_dynamics.ThresholdDynamicsRunResult

::: phydrax.threshold_dynamics.ThresholdDynamicsEvidence

::: phydrax.threshold_dynamics.HeatActionEvidence

::: phydrax.threshold_dynamics.VolumeConstraintEvidence

::: phydrax.threshold_dynamics.SparseCandidateEvidence

::: phydrax.threshold_dynamics.ThresholdDynamicsStatus

::: phydrax.threshold_dynamics.threshold_dynamics_status_message

## Qualification measurements and profiles

::: phydrax.threshold_dynamics.label_contacts

::: phydrax.threshold_dynamics.label_neighbor_counts

::: phydrax.threshold_dynamics.triple_junction_angles

::: phydrax.threshold_dynamics.von_neumann_mullins_fit

::: phydrax.threshold_dynamics.threshold_dynamics_candidate_profiles
