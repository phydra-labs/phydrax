# Reduced-order models

PhydraX reduced models compose canonical decomposition, vector-space, operator,
solver, fidelity, lifecycle, and identification substrates. There is no generic
profile trainer or truth-backed online evaluator.

## Basis and projection

::: phydrax.rom.ReducedBasisArtifact

::: phydrax.rom.TrialTestReduction

::: phydrax.rom.reduced_basis_from_subspace_model

::: phydrax.rom.trial_test_reduction_from_bases

## Affine linear reduction

::: phydrax.rom.AbstractAffineCoefficientMap

::: phydrax.rom.ArrayAffineCoefficientMap

::: phydrax.rom.AffineLinearROMProblem

::: phydrax.rom.AffineLinearROMPlan

::: phydrax.rom.PreparedAffineLinearROM

::: phydrax.rom.AffineLinearROMEvaluation

::: phydrax.rom.prepare_affine_linear_rom

::: phydrax.rom.AffineLinearROMFidelityEvaluator

## Audit and certification

::: phydrax.rom.audit_affine_linear_rom

::: phydrax.rom.ResidualDualNormArtifact

::: phydrax.rom.ArrayAffineStabilityBound

::: phydrax.rom.AffineROMCertification

::: phydrax.rom.prepare_residual_dual_norm

::: phydrax.rom.certify_affine_rom_evaluation

## Identified dynamics

::: phydrax.rom.IdentifiedReducedDynamics

::: phydrax.rom.project_trajectory_data

## Nonlinear projection and hyperreduction

::: phydrax.rom.FullResidualGalerkin

::: phydrax.rom.ReducedLSPGProblem

::: phydrax.rom.DEIMArtifact

::: phydrax.rom.GNATLSPGProblem

::: phydrax.rom.ECSWArtifact

::: phydrax.rom.prepare_deim

::: phydrax.rom.prepare_gnat

::: phydrax.rom.prepare_ecsw

## Empirical interpolation

`prepare_empirical_interpolation` derives deterministic interpolation nodes from a
role-explicit `ReducedBasisArtifact`. The artifact binds source, support, measure,
geometry, conditioning, and maximum basis-reproduction defect. State/ROQ EIM does
not imply nonlinear DEIM.

::: phydrax.rom.EmpiricalInterpolationPlan

::: phydrax.rom.EmpiricalInterpolationArtifact

::: phydrax.rom.PreparedEmpiricalInterpolation

::: phydrax.rom.prepare_empirical_interpolation

## Persistence

::: phydrax.rom.write_reduced_basis_artifact

::: phydrax.rom.read_reduced_basis_artifact

::: phydrax.rom.write_affine_linear_rom

::: phydrax.rom.read_affine_linear_rom

::: phydrax.rom.write_empirical_interpolation_artifact

::: phydrax.rom.read_empirical_interpolation_artifact

## Production governance and data

::: phydrax.rom.ROMResourcePolicy

::: phydrax.rom.ROMAdmissionEvidence

::: phydrax.rom.ROMCapabilityDeclaration

::: phydrax.rom.rom_capability_catalog

::: phydrax.rom.SnapshotManifest

::: phydrax.ml.decomposition.PhysicalPODPlan

## Lifts, transient and mixed systems

::: phydrax.rom.ReducedLiftArtifact

::: phydrax.rom.AffineEvolutionROMProblem

::: phydrax.rom.PreparedAffineEvolutionROM

::: phydrax.rom.RectangularLinearROMProblem

::: phydrax.rom.ReducedInfSupEvidence

::: phydrax.rom.IndexOneDescriptorReduction

## Greedy and goal-oriented reduction

::: phydrax.rom.EstimatorGreedyPlan

::: phydrax.rom.SuccessiveConstraintArtifact

::: phydrax.rom.PrimalDualOutputBound

## Selected execution

::: phydrax.rom.SelectedEntitySet

::: phydrax.rom.SelectedEvaluationPlan

::: phydrax.rom.ThinGNATArtifact

## Geometry and state charts

::: phydrax.rom.ReferencePhysicalRepresentation

::: phydrax.rom.ReducedBasisAtlasArtifact

::: phydrax.rom.QuadraticStateChart

::: phydrax.rom.CoordinateConditionedStateChart

## Sensing and assimilation

::: phydrax.rom.SensorConfiguration

::: phydrax.rom.ObservationHistory

::: phydrax.rom.LinearSensorHistoryEstimator

::: phydrax.rom.ReducedKalmanAssimilator

## Adaptation and distribution

::: phydrax.rom.ROMGeneration

::: phydrax.rom.ActiveLearningPlan

::: phydrax.rom.EnrichmentTransaction

::: phydrax.rom.DistributedBasisArtifact

## Structure-preserving reduction

::: phydrax.rom.SymplecticReduction

::: phydrax.rom.PortHamiltonianReduction

## Deployment bundles

::: phydrax.rom.ROMDeploymentBundle

::: phydrax.rom.write_rom_deployment_bundle

::: phydrax.rom.read_rom_deployment_bundle

See the [reduced-order modeling guide](../guides_reduced_order_modeling.md), the
[multi-fidelity guide](../guides_multifidelity.md), and the
[gravitational-wave inference guide](../guides_gravitational_wave_inference.md#qualified-acceleration).
