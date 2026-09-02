# Structured finite volume

## Geometry and methods

::: phydrax.discretization.FiniteVolumePlan

---

::: phydrax.discretization.FiniteVolumeDiscretization

---

::: phydrax.discretization.MappedFiniteVolumePlan

---

::: phydrax.discretization.MappedFiniteVolumeDiscretization
---

::: phydrax.discretization.MappedPeriodicSeamPlan

---

::: phydrax.discretization.MappedPeriodicSeam


---

::: phydrax.discretization.FiniteVolumeMethodPlan

---

::: phydrax.discretization.PreparedFiniteVolumeDynamics

---

::: phydrax.discretization.FiniteVolumeResidualDiagnostics

---

::: phydrax.discretization.FiniteVolumeEntropyDiagnostics

---

::: phydrax.discretization.integrated_finite_volume_relative_entropy
---

::: phydrax.discretization.FiniteVolumeEntropyProductionDiagnostics

---

::: phydrax.discretization.evaluate_content_form_entropy_diagnostics


## Reconstruction and limiting

::: phydrax.discretization.PiecewiseConstantReconstruction

---

::: phydrax.discretization.MUSCLReconstruction

---

::: phydrax.discretization.MinmodLimiter

---

::: phydrax.discretization.MCLimiter

---

::: phydrax.discretization.VanLeerLimiter

---

::: phydrax.discretization.SuperbeeLimiter

---

::: phydrax.discretization.WENOReconstructionPlan

---

::: phydrax.discretization.HighResolutionReconstructionPlan

---

::: phydrax.discretization.NonuniformWENOReconstructionPlan

---

::: phydrax.discretization.CharacteristicReconstructionPlan

---

::: phydrax.discretization.ConvexStateLimiterPlan

## Numerical fluxes and waves

::: phydrax.discretization.RusanovFluxPlan

---

::: phydrax.discretization.HLLFluxPlan

---

::: phydrax.discretization.HLLCFluxPlan

---

::: phydrax.discretization.RoeFluxPlan

---

::: phydrax.discretization.EntropyConservativeEulerFluxPlan

---

::: phydrax.discretization.EntropyStableEulerFluxPlan
---

::: phydrax.discretization.EntropyStableFluxPlan


---

::: phydrax.discretization.RoeWavePropagationPlan

---

::: phydrax.discretization.ShallowWaterHydrostaticHLLPlan

---

::: phydrax.discretization.ShallowWaterWetDryPolicy

---

::: phydrax.discretization.PreparedShallowWaterBathymetry
---

::: phydrax.discretization.ShallowWaterBathymetryPlan

---

::: phydrax.discretization.ShallowWaterEquilibriumWENOZPlan

---

::: phydrax.discretization.GeostrophicBalancePlan

---

::: phydrax.discretization.ShallowWaterShorelineEvent
---

::: phydrax.discretization.PreparedBalancedShallowWaterLowering

---

::: phydrax.discretization.lower_triangle_unstructured_shallow_water

---

::: phydrax.discretization.lower_sbp_shallow_water

---

::: phydrax.discretization.lower_global_spectral_shallow_water

---

::: phydrax.discretization.lower_dgsem_shallow_water



---

::: phydrax.discretization.ShallowWaterObservables

---

::: phydrax.discretization.WaveFamilyLimiterPlan

---

::: phydrax.discretization.TransverseWaveSolverPlan

## Boundaries

::: phydrax.discretization.FiniteVolumeBoundarySet

---

::: phydrax.discretization.FiniteVolumeBoundaryPair

---

::: phydrax.discretization.ExtrapolationBoundary

---

::: phydrax.discretization.ConstantStateBoundary

---

::: phydrax.discretization.PrescribedStateBoundary

---

::: phydrax.discretization.PrescribedNormalFluxBoundary
---

::: phydrax.discretization.ShallowWaterNormalDischargeBoundary

---

::: phydrax.discretization.ShallowWaterCharacteristicOpenBoundary

---

::: phydrax.discretization.ReflectiveBoundary
---

::: phydrax.discretization.SlipWallBoundary

---

::: phydrax.discretization.NoSlipAdiabaticWallBoundary

---

::: phydrax.discretization.NoSlipIsothermalWallBoundary

---

::: phydrax.discretization.SupersonicInflowBoundary

---

::: phydrax.discretization.SupersonicOutflowBoundary

---

::: phydrax.discretization.CharacteristicInflowBoundary

---

::: phydrax.discretization.CharacteristicOutflowBoundary

---

::: phydrax.discretization.FarFieldBoundary

## Halo, positivity, and distribution

::: phydrax.discretization.FiniteVolumeHaloPlan

---

::: phydrax.discretization.PreparedFiniteVolumeHaloPlan

---

::: phydrax.discretization.EinfeldtHLLFluxPlan

---

::: phydrax.discretization.FluxPositivityPlan

---

::: phydrax.discretization.FiniteVolumeDecompositionPlan

---

::: phydrax.discretization.PreparedFiniteVolumeDecomposition

## Diffusion, viscosity, and projection

::: phydrax.discretization.FaceCoefficientPlan

---

::: phydrax.discretization.ConservativeDiffusionPlan

---

::: phydrax.discretization.ViscousFluxPlan

---

MAC pressure route eligibility belongs to the prepared solver, not to the grid alone.
The transform-diagonal route requires a uniform constant-coefficient
periodic/Neumann tensor. The hybrid route additionally requires rank three, an
all-Neumann pressure closure, one explicitly nonperiodic physical line, and two
uniform transform-compatible transverse axes. It transforms the transverse axes in
axis order, solves the nonuniform Neumann tridiagonal lines, and synthesizes in reverse
order. Preparation checks the represented physical action and resource budget.

The hybrid all-zero transverse mode uses volume compatibility and a zero-mean volume
gauge; pinning one physical-line row is only a factorization device. Supplying a
runtime inverse-momentum diagonal forces iterative projection. Variable-density,
mixed/open-closure, distributed, and sharded-line execution are not hybrid routes.
MAC also has no fixed-bulk-flux controller.

::: phydrax.discretization.MACOperatorPlan

---

::: phydrax.discretization.PreparedMACOperators

---

::: phydrax.discretization.MACOperatorReport

---

::: phydrax.discretization.MACDiffuseSDFGeometryPlan

---

::: phydrax.discretization.MACExactSDFMeasurePlan

---

::: phydrax.geometry.QualifiedSharpGeometry

---

::: phydrax.geometry.SharpGeometryEvidence

---

::: phydrax.discretization.MACBoundaryPlan

---

::: phydrax.discretization.MACMomentumPlan

---

::: phydrax.discretization.PreparedMACMomentumOperators

---

::: phydrax.discretization.MACMomentumReport

---

::: phydrax.discretization.MACMomentumDiagnostics

---

### MAC plane/wall statistics and production

`MACPlaneWallStatisticsPlan` supports one nonperiodic wall axis and periodic
homogeneous axes in two or three dimensions. It centers face components to cells for
volume-weighted plane moments, forms face-measure-weighted normal means from native
boundary faces, and reports separate signed shears from a one-sided
zero-wall-velocity derivative. It does not inspect boundary plans or subtract moving
tangential wall velocity. It is
instantaneous; temporal windows and block uncertainty are owned by
`StructuredMACProductionPlan`.
`MACConstantPressureGradientForcing` is fixed compiler-bound acceleration, not
fixed-flux feedback.

::: phydrax.applications.incompressible_flow.MACPlaneWallStatisticsPlan

---

::: phydrax.applications.incompressible_flow.MACPlaneWallStatistics

---

::: phydrax.applications.incompressible_flow.MACConstantPressureGradientForcing

---

::: phydrax.applications.incompressible_flow.StructuredMACProductionPlan

---

::: phydrax.applications.incompressible_flow.PreparedStructuredMACProduction

---

::: phydrax.discretization.MACScalarProblem

---

::: phydrax.discretization.PreparedMACScalarTransport

---

::: phydrax.discretization.MACPassiveTracerMacCormackPlan

---

::: phydrax.discretization.PreparedMACPassiveTracerMacCormack

---

::: phydrax.discretization.MACPassiveTracerMacCormackResult

---

::: phydrax.discretization.MACVariableDensityPlan

---

::: phydrax.discretization.MACMarkerTransferPlan

---

::: phydrax.discretization.MACMarkerKernelPlan

---

::: phydrax.discretization.MACMarkerRouteState

---

::: phydrax.discretization.MappedMACMarkerTransferPlan

---

::: phydrax.discretization.CompositeMACMarkerTransferPlan

---

::: phydrax.discretization.DistributedMarkerOwnershipPlan

---

::: phydrax.discretization.DistributedMACMarkerTransfer

---

::: phydrax.discretization.MarkerEpochTransferPlan

---

::: phydrax.discretization.ResolvedLubricationCorrectionPlan

---

::: phydrax.discretization.MACDistributedTopologyPlan

---

::: phydrax.discretization.MappedMACGeometryPlan

## Multiblock and AMR

::: phydrax.discretization.ConservativeMultiblockInterfacePlan

---

::: phydrax.discretization.ConservativeMultiblockFluxResult

---

::: phydrax.discretization.ConservativeAMRSubcyclingPlan

---

::: phydrax.discretization.ConservativeAMRSynchronizationPlan

---

::: phydrax.discretization.FluxRegister

## Equation systems and compilation

::: phydrax.equations.AbstractConservationSystem

---

::: phydrax.equations.ScalarConservationSystem

---

::: phydrax.equations.EulerSystem
---

::: phydrax.equations.CompressibleNavierStokesSystem

---

::: phydrax.equations.IdealGasMaterial

---

::: phydrax.equations.StiffenedGasMaterial

---

::: phydrax.equations.ConstantTransport

---

::: phydrax.equations.SutherlandTransport

---

::: phydrax.equations.MultispeciesEulerSystem

---

::: phydrax.equations.ShallowWaterSystem

---
::: phydrax.equations.ShallowWaterCoriolisSource

---


::: phydrax.equations.IdealMHDSystem

---

::: phydrax.equations.ConservationProblemIR

---

::: phydrax.equations.CompiledConservationProblem

---

::: phydrax.equations.compile_conservation_problem

## Time execution

::: phydrax.solver.UnsplitFiniteVolumeSSPRK3Plan

---

::: phydrax.solver.DirectionalSplitFiniteVolumePlan

---

::: phydrax.solver.FiniteVolumeStepResult

## Runtime and persistence

`FiniteVolumePrecisionPolicy` is owned by the prepared discretization/runtime,
not attached as case metadata after preparation. It independently places stored
cell averages, reconstruction, interface fluxes, conservative reductions and
decisions, returned snapshots, and checkpoints. SSPRK combinations, CFL and
positivity decisions, flux registers, conservation diagnostics, AMR
synchronization, rollout retention, HDF5 output, and checkpoint restore all
consume the same policy and retain content-addressed evidence.

::: phydrax.discretization.FiniteVolumePrecisionPolicy

---

::: phydrax.solver.FiniteVolumeRuntimeState

---

::: phydrax.solver.FiniteVolumeStepPolicy

---

::: phydrax.solver.PreparedFiniteVolumeRuntime

---

::: phydrax.solver.FiniteVolumeCaseSpec

---

::: phydrax.solver.FiniteVolumeCheckpointPlan

---

::: phydrax.solver.FiniteVolumeOutputPlan

---

::: phydrax.solver.AdaptiveFiniteVolumeRolloutPlan

---

::: phydrax.solver.ScheduledFiniteVolumeRolloutPlan

---

::: phydrax.solver.FiniteVolumeReplayPolicy

---

::: phydrax.solver.FiniteVolumeGradientReport
