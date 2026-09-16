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

See the [reduced-order modeling guide](../guides_reduced_order_modeling.md), the
[multi-fidelity guide](../guides_multifidelity.md), and the
[gravitational-wave inference guide](../guides_gravitational_wave_inference.md#qualified-acceleration).
