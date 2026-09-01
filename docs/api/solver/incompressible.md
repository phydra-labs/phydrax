# Incompressible flow

## Spectral problems and periodic projection

::: phydrax.equations.IncompressibleFlowProblem

---

::: phydrax.discretization.PeriodicLerayProjector

---

::: phydrax.equations.compile_periodic_incompressible_flow

---

::: phydrax.equations.CompiledIncompressibleSpectralDynamics

---

::: phydrax.discretization.IncompressibleSpectralDiagnostics

## Fourier–Chebyshev–Fourier channel flow

::: phydrax.discretization.ChannelMeanConstraint

---

::: phydrax.discretization.ChannelStokesPlan

---

::: phydrax.discretization.PreparedChannelStokesSolver

---

::: phydrax.discretization.ChannelStokesSolveResult

---

::: phydrax.equations.compile_channel_flow

---

::: phydrax.equations.CompiledChannelFlowDynamics

---

::: phydrax.equations.ChannelVelocityDiagnostics

---

::: phydrax.solver.ChannelSBDF2Method

---

::: phydrax.solver.solve_channel_sbdf2

---

::: phydrax.solver.ChannelFlowSolution

## Structured finite-volume MAC dynamics and projection

::: phydrax.discretization.MACOperatorPlan

---

::: phydrax.discretization.PreparedMACOperators

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

::: phydrax.solver.MACPressureProjectionPlan

---

::: phydrax.solver.MACPressureProjectionResult

---

::: phydrax.solver.MACRateProjectionResult

---

::: phydrax.equations.compile_mac_incompressible_flow

---

::: phydrax.equations.CompiledMACIncompressibleDynamics

---

::: phydrax.equations.MACIncompressibleDiagnostics

---

::: phydrax.equations.MACStepRestriction

## Scalar and variable-density MAC dynamics

::: phydrax.discretization.MACScalarProblem

---

::: phydrax.discretization.PreparedMACScalarTransport

---

::: phydrax.equations.MACBuoyancyLaw

---

::: phydrax.equations.compile_mac_scalar_buoyancy

---

::: phydrax.equations.CompiledMACScalarBuoyancyDynamics

---

::: phydrax.discretization.MACVariableDensityPlan

---

::: phydrax.solver.MACVariableDensityProjectionPlan

---

::: phydrax.equations.compile_mac_variable_density_flow

---

::: phydrax.equations.CompiledMACVariableDensityDynamics

## Implicit, adaptive, and sensitivity execution

::: phydrax.solver.MACHelmholtzSolvePlan

---

::: phydrax.solver.MACIMEXEulerMethod

---

::: phydrax.solver.MACSBDF2Method

---

::: phydrax.solver.MACAdaptiveRolloutPlan

---

::: phydrax.solver.MACFrozenGridReplayPlan

---

::: phydrax.solver.MACFixedGridSensitivityPlan

---

::: phydrax.solver.MACSegmentedShadowingPlan

## Resolved, distributed, and moving-geometry execution

::: phydrax.discretization.MACMarkerTransferPlan

---

::: phydrax.solver.MACImmersedBoundaryProjectionPlan

---

::: phydrax.solver.MACImmersedBoundaryIMEXEulerMethod

---

::: phydrax.solver.MACImmersedBoundarySBDF2Method

---

::: phydrax.equations.MACPenaltyIBCFDEMCouplingPlan

---

::: phydrax.solver.advance_mac_penalty_ib_cfd_dem_window

---

::: phydrax.solver.MACRigidImmersedEulerMethod

---

::: phydrax.solver.MACDeformableImmersedBackwardEulerMethod

---

::: phydrax.discretization.MACDistributedTopologyPlan

---

::: phydrax.solver.MACDistributedProjectionPlan

---

::: phydrax.discretization.MappedMACGeometryPlan

---

::: phydrax.solver.MACALEGeometryPlan

---

::: phydrax.solver.MACRemeshEpochPlan

## Marker-flow stage, mechanics, and topology closure

::: phydrax.solver.MACHelmholtzStageInverseMomentum

---

::: phydrax.solver.MACVariableDensityStageInverseMomentum

---

::: phydrax.solver.MACVariableViscosityStagePlan

---

::: phydrax.solver.MACRigidImmersedBackwardEulerMethod

---

::: phydrax.solver.MACRigidImmersedMidpointMethod

---

::: phydrax.solver.MACRigidImmersedContactMethod

---

::: phydrax.solver.MACRigidImmersedJointMethod

---

::: phydrax.solver.MACDeformableImmersedNewmarkMethod

---

::: phydrax.solver.DeformableContactResidualPlan

---

::: phydrax.discretization.ResolvedLubricationCorrectionPlan

---

::: phydrax.discretization.CompositeMACMarkerTransferPlan

---

::: phydrax.solver.CompositeMACProjectionPlan

---

::: phydrax.discretization.DistributedMACMarkerTransfer

---

::: phydrax.discretization.MarkerEpochTransferPlan

## Divergence-free, sharp, and stochastic families

::: phydrax.solver.MACDivergenceFreeMarkerTransfer

---

::: phydrax.solver.MACDFIBProjectionPlan

---

::: phydrax.solver.MACSharpInterfaceProjectionPlan

---

::: phydrax.solver.MACMovingSharpInterfaceEpochPlan

---

::: phydrax.solver.MACImmersedInterfaceProjectionPlan

---

::: phydrax.solver.MACInterfaceMethodSelector

---

::: phydrax.solver.MACDiscreteStochasticStressPlan

---

::: phydrax.solver.MACFluctuatingHydrodynamicsPlan

---

::: phydrax.solver.MACInertialStochasticStepPlan

---

::: phydrax.solver.FIBOverdampedPlan

## Marker-flow runtime and evidence

::: phydrax.solver.MarkerFlowCheckpointPlan

---

::: phydrax.solver.MarkerFlowReplayPlan

---

::: phydrax.solver.MarkerFlowOutputPlan

---

::: phydrax.solver.HydrodynamicLoadPlan

---

::: phydrax.solver.MarkerFlowAdaptiveStepPlan

---

::: phydrax.solver.MarkerFlowTrajectoryAdapter

---

::: phydrax.solver.MarkerFlowCompiledExportPlan

---

::: phydrax.solver.MarkerFlowQualificationPlan
