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

::: phydrax.discretization.FiniteVolumeMethodPlan

---

::: phydrax.discretization.PreparedFiniteVolumeDynamics

---

::: phydrax.discretization.FiniteVolumeResidualDiagnostics

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

::: phydrax.discretization.RoeWavePropagationPlan

---

::: phydrax.discretization.FWaveShallowWaterPlan

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

::: phydrax.discretization.MACPressureProjectionPlan

---

::: phydrax.discretization.FunctionalPressureCorrectionPlan

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

::: phydrax.solver.FiniteVolumeRolloutPlan

---

::: phydrax.solver.FiniteVolumeGradientReport
