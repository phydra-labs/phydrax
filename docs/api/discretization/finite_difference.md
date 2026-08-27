# Finite-difference solver substrate

## Support and locations

::: phydrax.discretization.PreparedTensorGrid

---

::: phydrax.discretization.GridLocation

---

::: phydrax.discretization.StructuredAxis

---

::: phydrax.discretization.TensorEntityLayout

## Coefficients, stencils, and evidence

::: phydrax.discretization.StencilCoefficientPlan

---

::: phydrax.discretization.DerivativeRequest

---

::: phydrax.discretization.LinearStencil

---

::: phydrax.discretization.BoundaryStencilSet

---

::: phydrax.discretization.PreparedStencilOperator

---

::: phydrax.discretization.FDConsistencyReport

---

::: phydrax.discretization.FDAdjointReport

---

::: phydrax.discretization.FDConservationReport

---

::: phydrax.discretization.FDStabilityReport

---

::: phydrax.equations.ManufacturedPDECase

---

::: phydrax.equations.ManufacturedConvergencePlan

## Boundaries, interfaces, and halos

::: phydrax.discretization.BoundaryStageContext

---

::: phydrax.discretization.CellGhostBoundary

---

::: phydrax.discretization.NodalBoundaryRuntime

---


::: phydrax.equations.PreparedFDBoundaryProgram

---

::: phydrax.equations.PreparedFDInterface

---

::: phydrax.discretization.BoundaryAffineMap

---

::: phydrax.discretization.HaloPlan

## Lifecycle and compact execution

::: phydrax.discretization.FiniteDifferencePlan

---

::: phydrax.discretization.PreparedFiniteDifferenceDiscretization

---

::: phydrax.discretization.StencilExecutionPlan

---

::: phydrax.discretization.PreparedStencilExecutionOperator

---

::: phydrax.discretization.StencilExecutionReport

---

::: phydrax.discretization.StencilAssignment

---

::: phydrax.discretization.StencilProgramPlan

---

::: phydrax.discretization.FDPipelineReport

---

::: phydrax.equations.CompiledFiniteDifferenceDynamics

## Compact implicit line calculus

::: phydrax.discretization.CompactDerivativePlan

---

::: phydrax.discretization.CompactInterpolationPlan

---

::: phydrax.discretization.PreparedCompactOperator

---

::: phydrax.discretization.CompactOperatorReport


## Conservative face operators

Cell-to-face conservative diffusion and advection now belong to the
[structured finite-volume API](finite_volume.md). Finite-difference equation lowering
reuses those prepared flux operators where the requested expression is conservative.

## SBP-SAT and mapped geometry

::: phydrax.discretization.SBPFamily

---

::: phydrax.discretization.SBPDerivativePlan

---

::: phydrax.discretization.PreparedSBPOperator

---

::: phydrax.discretization.CompatibleSBPSecondDerivative

---

::: phydrax.discretization.SATBoundaryPlan

---

::: phydrax.discretization.SATInterfacePlan

## Periodic SBP flux differencing

Periodic SBP conservation diagnostics use a twofold compensated reduction for the
quadrature-weighted residual in the prepared grid dtype. The antisymmetric pairwise
flux construction, entropy rates, and evolution residual are unchanged.


::: phydrax.discretization.TensorSBPPlan

---

::: phydrax.discretization.TensorSBPDiscretization

---

::: phydrax.discretization.SBPFluxDifferencingMethodPlan

---

::: phydrax.discretization.PreparedSBPConservationDynamics

---

::: phydrax.discretization.SBPFluxDifferencingDiagnostics

---

::: phydrax.discretization.SBPFluxDifferencingReport


---

::: phydrax.discretization.MappedTensorGridPlan

---

::: phydrax.discretization.PreparedMappedTensorGrid

---

::: phydrax.discretization.MappedMetricIdentityReport

---

::: phydrax.discretization.MappedDiffusionOperator

---

::: phydrax.discretization.evaluate_mapped_metrics

## Multiblock and multigrid

::: phydrax.discretization.MultiblockGridPlan

---

::: phydrax.discretization.BlockInterface

---

::: phydrax.discretization.InterfaceOrientation

---

::: phydrax.discretization.NormCompatibleInterpolationPlan

---

::: phydrax.discretization.MultiblockSATCoupling

---

::: phydrax.discretization.StructuredTransferPlan

---

::: phydrax.discretization.StructuredMultigridPlan

---

::: phydrax.discretization.PreparedStructuredMultigrid

## Certified transform-direct Laplacians

::: phydrax.discretization.diagonalize_fd_laplacian

---

::: phydrax.discretization.FDLaplacianDiagonalization

---

::: phydrax.discretization.FDLaplacianSolvePlan

---

::: phydrax.discretization.solve_fd_laplacian

## High-resolution hyperbolic methods

Cell-average reconstruction, numerical fluxes, physical conservation systems, and
positivity policies belong to the [structured finite-volume API](finite_volume.md).

## AMR and distributed execution

::: phydrax.discretization.FDAMRHaloPlan

---

::: phydrax.discretization.AMREntityTransferPlan

---

::: phydrax.discretization.ConservativeAMRSubcyclingPlan

---

::: phydrax.discretization.FDRegridPlan

---

::: phydrax.discretization.AMRMigrationPlan

---

::: phydrax.discretization.DistributedHaloSchedule

---

`FDExecutionPrecisionPolicy` is executable: coefficient banks use
`coefficient_dtype`, field spaces/halo payloads/checkpoints use `field_dtype`,
stencil contractions reduce in `accumulation_dtype`, and host/device numerical
certification uses `certification_dtype`. The policy identity is included in
prepared operators, AMR/subcycling/reflux plans, distributed schedules,
multigrid hierarchies, adjoints, checkpoints, and preflight estimates. Resource
preflight derives byte counts from the same policy's explicit item-size
assumptions and rejects operators prepared under a different policy.

::: phydrax.discretization.FDExecutionPreflightPlan

---

::: phydrax.discretization.FDExecutionPrecisionPolicy

---

::: phydrax.discretization.FDResourceEstimate

## Checkpointing, adjoints, and compatible systems

::: phydrax.discretization.FDCheckpointPlan

---

::: phydrax.discretization.FDCheckpoint

---

::: phydrax.discretization.FDActionAdjointPlan

---

::: phydrax.discretization.CheckpointedFDAdjointPlan

---

::: phydrax.discretization.StructuredCochainBridge

---

::: phydrax.solver.CompatibleMaxwellPlan

---

::: phydrax.solver.PreparedCompatibleMaxwell

---

::: phydrax.solver.CompatibleMaxwellState

---

::: phydrax.solver.CompatibleMaxwellDiagnostics

### Maxwell materials, boundaries, observers, and adjoints

::: phydrax.solver.maxwell.DiagonalMaxwellConstitutivePlan

---

::: phydrax.solver.maxwell.MatrixMaxwellConstitutivePlan

---

::: phydrax.solver.maxwell.ConductiveMaxwellConstitutivePlan

---

::: phydrax.solver.maxwell.LorentzDrudeMaxwellConstitutivePlan

---

::: phydrax.solver.maxwell.KerrPockelsMaxwellConstitutivePlan

---

::: phydrax.solver.maxwell.MaxwellBoundaryPlan

---

::: phydrax.solver.maxwell.BlochCochainCalculus

---

::: phydrax.solver.maxwell.MaxwellCPMLPlan

---

::: phydrax.solver.maxwell.FieldProbePlan

---

::: phydrax.solver.maxwell.DFTObserverPlan

---

::: phydrax.solver.maxwell.FrequencyMaxwellOperator

---

::: phydrax.solver.maxwell.PyTreeCheckpointedAdjointPlan

---

::: phydrax.solver.maxwell.MaxwellReversibleAdjointPlan

---

::: phydrax.solver.maxwell.UnstructuredMaxwellPlan

### Point-cloud strong-form calculus

::: phydrax.discretization.PointCloudPlan

---

::: phydrax.discretization.PreparedPointCloudDiscretization

---

::: phydrax.discretization.DissipativePointDiffusion

---

::: phydrax.discretization.solve_point_cloud_poisson

---

::: phydrax.solver.CompatibleElasticityDynamics

---

::: phydrax.solver.CompatibleIncompressibleProjection

---

::: phydrax.solver.CompatibleVariableDensityProjection

---

::: phydrax.solver.CompatibleIdealMHDInductionDynamics

---

::: phydrax.solver.CompatiblePoroelasticDynamics

---

::: phydrax.solver.CompatibleThermoelasticDynamics

## Staggered acoustic reference solver

::: phydrax.solver.SplitFieldPMLPlan

---

::: phydrax.solver.StaggeredAcousticPlan

---

::: phydrax.solver.PreparedStaggeredAcoustics
