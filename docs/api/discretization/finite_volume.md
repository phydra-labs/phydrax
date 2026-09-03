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
`MACPressureOperatorSpec` freezes the positive coefficient and the closure-aware
matrix-free action `A p = -D(beta G_h p)`. Static Robin traces use
`alpha p + beta_r dp/dn = value`. Preparation records coefficient contrast and
structure, affine lift, symmetry, JVP/VJP, resource, and geometry-epoch evidence.

`transform` is an exact uniform constant-coefficient tensor route. `hybrid` is an exact
three-dimensional all-Neumann transform-line route for constant or line-structured
coefficients with one named nonperiodic line. Both require the matching prepared
`direct_solve` callback at execution. General positive coefficients and symmetric
mixed/Robin closures use PCG with a frozen constant preconditioner; stabilized
nonsymmetric traction uses FGMRES. Distributed projection remains a separate
collective PCG plan, and partition-aware line solvers do not silently make the hybrid
route multi-device.

`MACFlowControlTarget` separately declares pressure-gradient, bulk-velocity, or
frozen-density mass-flux control. `MACFlowControlPlan.prepare()` binds SSPRK,
IMEX-Euler, or SBDF2 method-stage response maps. A prepared step solves only the finite
control response system and atomically rejects rank/conditioning, target, response,
boundary, projection, pressure, resource, or underlying-method failures.

::: phydrax.solver.MACPressureRobinSide

---

::: phydrax.solver.MACPressureOperatorSpec

---

::: phydrax.solver.PreparedMACPressureOperator

---

::: phydrax.solver.MACPressureSolveResult

---

::: phydrax.applications.incompressible_flow.MACFlowControlTarget

---

::: phydrax.applications.incompressible_flow.MACFlowControlPlan

---

::: phydrax.applications.incompressible_flow.PreparedMACFlowControl

---

::: phydrax.applications.incompressible_flow.MACFlowControlStepResult

---

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

`ConservativeMultiblockInterfacePlan` supports conforming and nested 2:1
upper-to-lower interfaces. It evaluates one fine-mortar flux, returns equal-and-opposite
integrated block fluxes, and reports the compensated conservation defect.
`FiniteVolumeMultiblockRuntimePlan.limit_stage()` applies one secondary positivity
factor to every block candidate and shared mortar integral; if the monotone fallback is
inadmissible, all blocks retain their base states.

::: phydrax.discretization.ConservativeMultiblockInterfacePlan

---

::: phydrax.discretization.ConservativeMultiblockFluxResult

::: phydrax.discretization.FiniteVolumeMultiblockRuntimePlan

---

::: phydrax.discretization.MultiblockPositivityResult

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

::: phydrax.equations.HomogeneousMixtureEulerSystem

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

## Compressible and reacting application facades

`phydrax.applications.compressible_flow` binds smooth DG, high-resolution FV,
all-speed, shock, finite-x boundary, and frozen slow-growth application contracts.
Every profile is route-exact; `dns-candidate` does not make `claims_dns` true.

::: phydrax.applications.compressible_flow.CompressibleFlowCaseSpec

---

::: phydrax.applications.compressible_flow.AllSpeedCompressiblePolicy

---

::: phydrax.applications.compressible_flow.AllSpeedHLLFluxPlan

---
::: phydrax.applications.compressible_flow.ShockAwareAllSpeedFluxPlan

---


::: phydrax.applications.compressible_flow.ShockResolvingPolicy

---

::: phydrax.applications.compressible_flow.SmoothCompressibleProductionPlan

---

::: phydrax.applications.compressible_flow.NodalDGCompressibleProductionPlan

---

::: phydrax.applications.compressible_flow.StructuredFVCompressibleProductionPlan

---

::: phydrax.applications.compressible_flow.PreparedCompressibleProduction

---

::: phydrax.applications.compressible_flow.CompressiblePlaneBaseflowPlan

---

::: phydrax.applications.compressible_flow.TemporalSlowGrowthModelPlan

---

::: phydrax.applications.compressible_flow.SpatialSlowGrowthModelPlan

---

::: phydrax.applications.compressible_flow.PreparedSlowGrowthSource

---

::: phydrax.applications.compressible_flow.SlowGrowthFiniteXEvidence

`phydrax.applications.reacting_flow` consumes the canonical all-species
`HomogeneousHelmholtzPlan`, `HomogeneousMixtureEulerSystem`, and
`PreparedChemicalMechanism`. It owns transport, Strang/IMEX scheduling, low-Mach
constraint, statistics/closure targets, and host-only Cantera boundaries—not a second
EOS, state layout, mechanism compiler, Euler system, or FV runtime.

::: phydrax.equations.HomogeneousMixtureEulerSystem

---

::: phydrax.equations.HomogeneousMixtureCompressibleNavierStokesSystem

---

::: phydrax.applications.reacting_flow.ReactiveStrangPlan

---

::: phydrax.applications.reacting_flow.ReactiveIMEXPlan

---

::: phydrax.applications.reacting_flow.LowMachReactingFormulation

---

::: phydrax.applications.reacting_flow.CanteraYAMLAdapter

---

::: phydrax.applications.reacting_flow.CanteraReferenceAdapter

---

::: phydrax.applications.reacting_flow.ReactiveClosureTargetPlan

---

::: phydrax.applications.reacting_flow.ReactiveFlowStatisticsPlan

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
