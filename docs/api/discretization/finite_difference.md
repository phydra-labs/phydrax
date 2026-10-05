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

### Boundary norms and traces

`PreparedFiniteDifferenceDiscretization.prepare_side_trace` restricts the nodal
field to boundary-node facets measured by the tangential factors of a declared
`SBPGridNorm`; `boundary_face_selection` and `integration_domain` select faces.

::: phydrax.discretization.SBPGridNorm

---

::: phydrax.discretization.SBPClosureEvidence

---

::: phydrax.discretization.SBPNormKind

---

::: phydrax.discretization.SBPNormLayout

## Periodic SBP flux differencing

Periodic SBP conservation diagnostics use a twofold compensated reduction for the
quadrature-weighted residual in the prepared grid dtype. The antisymmetric pairwise
flux construction, entropy rates, and evolution residual are unchanged.

`SBPFluxDifferencingMethodPlan.volume_flux` is a dynamic child typed by the neutral
numerical-flux slot, so a learned symmetric two-point flux keeps its parameters.
The optional source uses the same `(time, state, coordinates, args)` conservation
source ABI as the finite-volume owners.


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

`PreparedFDAMRHierarchy` prepares host topology selection, field-specific
conservative topology transitions, source-classified cell-centered FillPatch routes,
and stencil-footprint validation. It does not advance solver time. FillPatch reads
caller-supplied coarse old/new states and times; physical boundary values remain
caller-owned and are exposed as explicit requests.

::: phydrax.discretization.FDAMRHierarchyPlan

---

::: phydrax.discretization.PreparedFDAMRHierarchy

---
::: phydrax.discretization.FDAMRFillPatchPlan

---


::: phydrax.discretization.FDAMRFillPatchResult

---

::: phydrax.discretization.FDAMRFillPatchWorkspace

---

::: phydrax.discretization.FDAMRPhysicalBoundaryRequest

---

::: phydrax.discretization.FillPatchSource

---

::: phydrax.discretization.AMREntityTransferPlan

---

::: phydrax.discretization.DistributedHaloSchedule

---

`FDExecutionPrecisionPolicy` is executable: coefficient banks use
`coefficient_dtype`, field spaces/FillPatch payloads/checkpoints use
`field_dtype`, stencil contractions reduce in `accumulation_dtype`, and
host/device numerical certification uses `certification_dtype`. The policy
identity is included in prepared operators, block-AMR FillPatch plans,
distributed schedules, multigrid hierarchies, adjoints, checkpoints, and
preflight estimates. Resource preflight derives byte counts from the same
policy's explicit item-size assumptions and rejects operators prepared under a
different policy.

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

::: phydrax.exterior.bloch_coefficient_system

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

### Meshfree and point-cloud calculus

See the [meshfree API](meshfree.md) for prepared local approximation, point-cloud
field views, sparse diffusion, and reusable elliptic solves.

### Compatible PDE consumers

::: phydrax.solver.CompatibleElasticityDynamics

---

Compatible pressure projections accept the canonical `CochainDiscretization`
(degrees 0 and 1, including a meshfree positive one-complex from
`PreparedMeshfreeExteriorCalculus.to_cochain()`) or a `StructuredCochainBridge`.
The pressure Poisson problem `δ(w d p) = δu - s` is a prepared native
`ProjectedPCG` solve in the degree-zero Hodge pairing. Its exact kernel is the
per-component constant subspace of `d0`, and the minimum-norm gauge fixes the
pressure. No dense pseudoinverse or size budget is involved. Tolerances, step
limits, resources, preconditioning, and precision come from an optional
`LinearSolvePolicy`, which must select `ProjectedPCG`.
`CompatibleVariableDensityProjection` interpolates `w = ρₑ⁻¹` from the
arithmetic mean of endpoint densities. It requires a diagonal degree-one Hodge
and rebinds the prepared solve on every call.

Both projections take `preconditioner: CompatiblePressurePreconditioner`,
either `"none"` (the default) or `"smoothed-aggregation"`. The latter assembles
`δd` at unit edge weights from the oriented active incidence by native sparse
assembly and prepares the native smoothed-aggregation V-cycle once, on the host,
at construction. The per-component constants are its near-nullspace candidates,
every graph coupling counts as strong, and its transfers respect the Hodge
pairing. Levels smooth with damped Jacobi (ω = 1/2), and the coarsest level
uses a symmetric Gauss–Seidel sweep. The result is a fixed, self-adjoint left
preconditioner for `ProjectedPCG` in the solve's own degree-zero space. Its
positive definiteness is asserted on coarse levels, where the contraction bound
of the fine graph Laplacian does not carry over; the independent residual
acceptance still decides `SUCCESS`. The structural kernel certificate is
unchanged. It requires diagonal degree-zero and degree-one Hodges and no
isolated active vertex. It replaces any preconditioning in `solve_policy`,
which must not declare one; violations raise `ValueError`. The
variable-density projection keeps the unit-weight hierarchy frozen across
densities, so the call stays traceable under `jit`, but iteration counts grow
with the edge-coefficient contrast. The native multigrid numeric refresh is
host-only. Iterations stay near 15 from 289 to 16641 structured unknowns,
while unpreconditioned counts grow from 77 to 565. The warm solve is faster
from about 4000 unknowns, but host setup costs 20–35 s. `"none"` stays the
default because the smoothed-aggregation route is not admissible for every
cochain (for example, sparse Hodges). See
`benchmarks/compatible_projection_scaling.py`.

`IncompressibleProjectionResult` reports divergence before and after, the
original pressure residual, the compatibility and gauge residuals, kernel
validity, nullity, the native `LinearSolveResult` (status, iterations,
precision, derivative contract), the selected `preconditioner` with its native
`multigrid` setup evidence (level dimensions, grid and operator complexity),
and a `CompatibleProjectionStatus`.
A target divergence `s` with a net source on a closed component is refused as
`INCOMPATIBLE_SOURCE`. A failed native solve is refused as `SOLVE_FAILED`.
In both cases the committed velocity is the unchanged input, the committed
pressure is NaN, and the rejected candidate is kept. A graph solenoidal
projection enforces a discrete constraint. It is not a qualified
Navier–Stokes velocity.

::: phydrax.solver.CompatibleIncompressibleProjection

---

::: phydrax.solver.CompatibleVariableDensityProjection

---

::: phydrax.solver.IncompressibleProjectionResult

---

::: phydrax.solver.CompatibleProjectionStatus

---

::: phydrax.solver.CompatibleIdealMHDInductionDynamics

---

::: phydrax.solver.CompatiblePoroelasticDynamics

---

::: phydrax.solver.CompatibleThermoelasticDynamics

### Field views

Finite-difference nodal values become field views only through an explicit
interpolation policy; point clouds use explicitly selected polynomial or positive
reconstruction with conditioning and support evidence.

::: phydrax.discretization.prepare_finite_difference_field_reconstruction

---

::: phydrax.discretization.MultilinearGridInterpolation

---

::: phydrax.discretization.BSplineGridInterpolation

---

::: phydrax.discretization.FiniteDifferenceFieldReconstructionKernel

Point-cloud reconstruction is documented in the [meshfree API](meshfree.md).

## Staggered acoustic reference solver

::: phydrax.solver.SplitFieldPMLPlan

---

::: phydrax.solver.StaggeredAcousticPlan

---

::: phydrax.solver.PreparedStaggeredAcoustics
