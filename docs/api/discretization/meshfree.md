# Meshfree and point-cloud solvers

See [Meshfree solvers](../../guides_meshfree.md) for numerical contracts,
preparation/resource boundaries, metric admission, and migration.

## Neighborhoods and local approximation

::: phydrax.discretization.meshfree.MeshfreeNeighborhoodPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeNeighborhood

---

::: phydrax.discretization.meshfree.SmoothSupportEnvelope

---

::: phydrax.discretization.meshfree.MeshfreeEdgeRelationPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeEdgeRelation

---

::: phydrax.discretization.meshfree.LocalStencilPolicy

---

::: phydrax.discretization.meshfree.MeshfreeFunctional

---

::: phydrax.discretization.meshfree.prepare_local_stencils

---

::: phydrax.discretization.meshfree.prepare_chart_stencils

---

::: phydrax.discretization.meshfree.PreparedLocalStencils

---

::: phydrax.discretization.meshfree.LocalStencilEvidence

---

::: phydrax.discretization.meshfree.LocalStencilReport

---

::: phydrax.discretization.meshfree.fit_chart_stencils

---

::: phydrax.discretization.meshfree.refresh_local_stencils

---

::: phydrax.discretization.meshfree.refresh_chart_stencils

---

::: phydrax.discretization.meshfree.LocalStencilRefresh

---

::: phydrax.discretization.meshfree.LocalStencilRefreshStatus

---

::: phydrax.discretization.spatial.MortonRadiusShellWitnessPlan

---

::: phydrax.discretization.spatial.MortonRadiusShellResult

---

::: phydrax.discretization.spatial.MortonRadiusShellEvidence

---

::: phydrax.discretization.meshfree.MeshfreeOperator

## Point-cloud calculus and observations

::: phydrax.discretization.PointCloudPlan

---

::: phydrax.discretization.PreparedPointCloudDiscretization

---

::: phydrax.discretization.PointCloudRefresh

---

::: phydrax.discretization.PointCloudCoordinateSensitivity

---

::: phydrax.discretization.prepare_point_cloud_field_reconstruction

---

::: phydrax.discretization.point_cloud_reconstruction_sensitivity

---

::: phydrax.discretization.PointCloudFieldReconstructionKernel

## Sparse elliptic solves

::: phydrax.discretization.PointDiffusionOperator

---

::: phydrax.discretization.PointDiffusivityEvidence

---

::: phydrax.discretization.PointBoundaryCondition

---

::: phydrax.discretization.PointBoundaryPlan

---

::: phydrax.discretization.PointCloudPoissonPlan

---

::: phydrax.discretization.PointCollocationPlan

---

::: phydrax.discretization.PreparedPointCollocation

---

::: phydrax.discretization.PreparedPointCloudPoisson

---

::: phydrax.discretization.PointCloudPoissonResult

---

::: phydrax.discretization.point_sbp_report

---

::: phydrax.discretization.PointSBPReport

---

::: phydrax.discretization.meshfree.prepare_point_sbp_derivatives

---

::: phydrax.discretization.meshfree.PointSBPDerivatives

---

::: phydrax.discretization.meshfree.prepare_tensor_point_sbp

The constrained preparation exposes SBP identities and algebraic solves, not a
global stability or accuracy guarantee. The tensor bridge instead binds actual
prepared native point-primary `SBPDerivativePlan` families to explicit cloud
coordinates, grid row IDs, `SBPGridNorm` volume weights, and tensor-face measures.
`PointSBPDerivatives.stable_realization` records this distinct native authority;
measured continuum error remains a separate diagnostic.


## Side-labeled support, interfaces, and boundary samples

::: phydrax.discretization.meshfree.PointSideSupportPlan

---

::: phydrax.discretization.meshfree.PreparedPointSideSupport

---

::: phydrax.discretization.meshfree.PointSideAdmissionEvidence

---

::: phydrax.discretization.meshfree.PointDerivativeFamily

---

::: phydrax.discretization.meshfree.PointInterfaceCondition

---

::: phydrax.discretization.meshfree.PointGhostLayerPlan

---

::: phydrax.discretization.meshfree.PreparedPointGhostLayer

---

::: phydrax.discretization.meshfree.PointGhostLayerEvidence

---

::: phydrax.discretization.meshfree.PointBoundarySamples

---

::: phydrax.discretization.meshfree.sample_boundary_atlas

---

::: phydrax.discretization.meshfree.sample_cubature_atlas

---

`PointBoundaryCharts` is the geometry authority for coupling boundaries of a
2-D cloud: straight atlas edges whose Gauss–Lobatto sites must be cloud
points. It publishes outward normals, the lumped boundary measure per point, an
exterior-facet domain, and exact-transpose value traces for
`MeshfreeTraceComponent`.

::: phydrax.discretization.meshfree.PointBoundaryCharts

## Coupled block systems

::: phydrax.discretization.meshfree.PointBlockSystemPlan

---

::: phydrax.discretization.meshfree.PreparedPointBlockSystem

---

::: phydrax.discretization.meshfree.PointBlockSystemResult

---

::: phydrax.discretization.meshfree.PointBlockNonlinearResult

---

::: phydrax.discretization.meshfree.PointCouplingEvidence

---

::: phydrax.discretization.meshfree.isotropic_elasticity_coefficients

## Solid mechanics

Small-strain elasticity, finite-strain neo-Hookean equilibrium with load
stepping and rollback, and the generalized Stokes / Herrmann mixed form with
energy, work, force-balance, and volumetric evidence.
See the [guide](../../guides_meshfree.md#meshfree-solid-mechanics).

::: phydrax.discretization.meshfree.MechanicsStatus

---

::: phydrax.discretization.meshfree.MeshfreeElasticityPlan

---

::: phydrax.discretization.meshfree.MeshfreeElasticityPreconditioner

---

::: phydrax.discretization.meshfree.MeshfreeTractionRoute

---

::: phydrax.discretization.meshfree.PreparedMeshfreeElasticity

---

::: phydrax.discretization.meshfree.MeshfreeElasticityResult

---

::: phydrax.discretization.meshfree.MeshfreeHyperelasticPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeHyperelastic

---

::: phydrax.discretization.meshfree.MeshfreeHyperelasticResult

---

::: phydrax.discretization.meshfree.MeshfreeGeneralizedStokesPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeGeneralizedStokes

---

::: phydrax.discretization.meshfree.MeshfreeMixedResult

---

::: phydrax.discretization.meshfree.displacement_gradient

## Stabilization

::: phydrax.discretization.meshfree.HyperviscosityPlan

---

::: phydrax.discretization.meshfree.PreparedHyperviscosity

---

::: phydrax.discretization.meshfree.HyperviscosityEvidence

## Multilevel preparation

::: phydrax.discretization.meshfree.MeshfreeCoarseningPolicy

---

::: phydrax.discretization.meshfree.MeshfreeHierarchyPlan

---

::: phydrax.discretization.meshfree.MeshfreeComponentSpace

---

::: phydrax.discretization.meshfree.MeshfreeNearNullspace

---

::: phydrax.discretization.meshfree.MeshfreeAuxiliaryStiffness

---

::: phydrax.discretization.meshfree.meshfree_auxiliary_stiffness

---

::: phydrax.discretization.meshfree.meshfree_auxiliary_builder

---

::: phydrax.discretization.meshfree.PreparedMeshfreeHierarchy

---

::: phydrax.discretization.meshfree.MeshfreeHierarchyEvidence

---

::: phydrax.discretization.meshfree.meshfree_multigrid_builder

---

::: phydrax.linalg.ProjectedPseudoinversePreconditionerBuilder

---

::: phydrax.linalg.ProjectedPseudoinversePreconditioner

## Overlapping Schwarz patches

`MeshfreeSchwarzPlan` partitions a point graph into contiguous stable-ID Morton
cores, grows each core by whole hops of the symmetrized adjacency, and prepares
restrictions, prolongations and a partition of unity. It returns native
`SubspaceCorrectionTerm` values only: the additive or multiplicative
composition, local and coarse factorizations, Krylov iteration and refresh stay
with `phydrax.linalg`, and convergence is judged on the original system.
`prolongation="partition_of_unity"` (restricted Schwarz with `ownership`) is
generally nonsymmetric and needs GMRES or FGMRES; `prolongation="adjoint"` is
classical additive Schwarz and certifies self-adjointness when its local solves
do. `meshfree_coarse_correction_term` supplies an exact Galerkin coarse term at
an explicit hierarchy level; placing it first in a multiplicative sweep gives a
hybrid two-level method. Patch, overlap and transfer capacities refuse
preparation instead of silently growing.

```python
schwarz = MeshfreeSchwarzPlan(cloud.plan.points, cloud.relation).prepare(plan.solve_space)
coarse = meshfree_coarse_correction_term(hierarchy, level=1)
policy = la.LinearSolvePolicy(
    la.GMRES(restart=100),
    preconditioning=la.PreconditioningPolicy(
        la.MultiplicativeSubspaceCorrectionBuilder((coarse,) + schwarz.terms())
    ),
)
```

::: phydrax.discretization.meshfree.MeshfreeSchwarzPlan

---

::: phydrax.discretization.meshfree.MeshfreeSchwarzPolicy

---

::: phydrax.discretization.meshfree.PreparedMeshfreeSchwarz

---

::: phydrax.discretization.meshfree.MeshfreeSchwarzEvidence

---

::: phydrax.discretization.meshfree.meshfree_coarse_correction_term

## Intrinsic surface preparation

::: phydrax.discretization.meshfree.ImplicitSurfaceGeometry

---

::: phydrax.discretization.meshfree.ChartSurfaceGeometry

---

::: phydrax.discretization.meshfree.SampledSurfaceGeometry

---

::: phydrax.discretization.meshfree.SurfaceBoundary

---

::: phydrax.discretization.meshfree.SurfaceGeometryEvidence

---

::: phydrax.discretization.meshfree.SurfaceGeometryEvaluation

---

::: phydrax.discretization.meshfree.SurfaceQuadraturePolicy

---

::: phydrax.discretization.meshfree.SurfaceQuadratureEvidence

---

::: phydrax.discretization.meshfree.chart_box_atlas

---

::: phydrax.discretization.meshfree.SurfacePointCloudPlan

---

::: phydrax.discretization.meshfree.PreparedSurfacePointCloud

---

::: phydrax.discretization.meshfree.SurfaceChartDerivatives

---

::: phydrax.discretization.meshfree.SurfaceRefreshResult

---

::: phydrax.discretization.meshfree.SurfaceFieldReconstructionKernel

---

::: phydrax.discretization.meshfree.SurfacePatchInterface

---

::: phydrax.discretization.meshfree.SurfaceEllipticSystem

---

::: phydrax.discretization.meshfree.SurfaceTangentCalculus

### Surface Stokes in strain form

Closed-surface Stokes/Brinkman flow with the deformation-rate viscous stress;
see the [meshfree guide](../../guides_meshfree.md#surface-stokes-in-strain-form).

::: phydrax.solver.MeshfreeSurfaceStokesPlan

---

::: phydrax.solver.MeshfreeSurfaceStokesResult

## Conservative metrics and transport

::: phydrax.discretization.meshfree.MeshfreeExteriorCalculusPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeExteriorCalculus

---

::: phydrax.discretization.meshfree.MeshfreeMetricPolicy

---

::: phydrax.discretization.meshfree.MetricSolver

---

::: phydrax.discretization.meshfree.BoundaryClosure

---

::: phydrax.discretization.meshfree.MeshfreeMetricGeometry

---

::: phydrax.discretization.meshfree.PreparedMeshfreeMetric

---

::: phydrax.discretization.meshfree.MeshfreeMetricResult

---

::: phydrax.discretization.meshfree.MeshfreeDiffusionOperator

---

::: phydrax.discretization.meshfree.MeshfreeStiffnessEvidence

---

::: phydrax.discretization.meshfree.MeshfreeCoercivityPolicy

---

::: phydrax.discretization.meshfree.CoercivityAssessment

---

::: phydrax.discretization.meshfree.MeshfreeExteriorRefreshResult

---

::: phydrax.discretization.meshfree.MeshfreeExteriorRefreshStatus

---

::: phydrax.discretization.meshfree.MeshfreeBoundaryQuadrature

---

::: phydrax.discretization.meshfree.MeshfreeAdvection

---

::: phydrax.discretization.meshfree.MeshfreeAdvectionResult

---

::: phydrax.discretization.meshfree.edge_upwind_content

---

::: phydrax.discretization.meshfree.MeshfreeAdvectionRate

---

::: phydrax.discretization.meshfree.ConservativeTransport

---

::: phydrax.discretization.meshfree.TransportScheme

---

::: phydrax.discretization.meshfree.TransportVolumeFlux

---

::: phydrax.discretization.meshfree.TransportRate

---

::: phydrax.discretization.meshfree.TransportStatus

---

::: phydrax.discretization.meshfree.TransportCFL

---

::: phydrax.discretization.meshfree.TransportRefresh

---

::: phydrax.discretization.meshfree.TransportRefreshStatus

---

::: phydrax.discretization.meshfree.minimum_image_edge_charts

---

::: phydrax.discretization.meshfree.MeshfreeEvolutionPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeEvolution

---

::: phydrax.discretization.meshfree.MeshfreeDiffusionLaw

---

::: phydrax.discretization.meshfree.MeshfreeReactionLaw

---

::: phydrax.discretization.meshfree.ReactionTreatment

---

::: phydrax.discretization.meshfree.InflowTreatment

---

::: phydrax.discretization.meshfree.MeshfreeDirichletRows

---

::: phydrax.discretization.meshfree.MeshfreeMotion

---

::: phydrax.discretization.meshfree.MotionKind

---

::: phydrax.discretization.meshfree.MaterialParticleMeasure

---

::: phydrax.discretization.meshfree.MeshfreeEvolutionFields

---

::: phydrax.discretization.meshfree.MeshfreeEvolutionAdmission

---

::: phydrax.discretization.meshfree.MeshfreeEvolutionStatus

---

::: phydrax.discretization.meshfree.MeshfreeEvolutionSSPMethod

---

::: phydrax.discretization.meshfree.MeshfreeEvolutionStepEvidence

---

::: phydrax.discretization.meshfree.MeshfreeEvolutionCapacity

---

::: phydrax.discretization.meshfree.MeshfreeSpectralEstimate

---

::: phydrax.discretization.meshfree.EvolutionLayout

---

::: phydrax.discretization.meshfree.EvolutionMeasure

---

::: phydrax.discretization.meshfree.EvolutionSSPMethod

---

::: phydrax.linalg.SparseRowRankPolicy

---

::: phydrax.linalg.SparseRowRankEvidence

---

::: phydrax.linalg.SparseRowRankRefusal

---

::: phydrax.linalg.prepare_sparse_row_rank

## Higher exterior degrees

Geometry-authorized k-form realization over a supplied oriented simplicial
complex; see the [meshfree guide](../../guides_meshfree.md#higher-exterior-degrees).
The radius-clique route is abstract research only.

::: phydrax.discretization.meshfree.ComplexDomainIdentity

---

::: phydrax.discretization.meshfree.ComplexGeometryRole

---

::: phydrax.discretization.meshfree.ComplexGeometryAuthority

---

::: phydrax.discretization.meshfree.MeshfreeComplexPolicy

---

::: phydrax.discretization.meshfree.MeshfreeCellComplexPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeCellComplex

---

::: phydrax.discretization.meshfree.MeshfreeComplexEvidence

---

::: phydrax.discretization.meshfree.MeshfreeComplexDegreeEvidence

---

::: phydrax.discretization.meshfree.MeshfreeComplexAdmissionError

---

::: phydrax.discretization.meshfree.ComplexFidelity

---

::: phydrax.discretization.meshfree.RadiusCliquePolicy

---

::: phydrax.discretization.meshfree.RadiusCliqueComplex

---

::: phydrax.discretization.meshfree.radius_clique_complex

## Moving surfaces and epoch transfer

::: phydrax.discretization.meshfree.MovingSurfacePlan

---

::: phydrax.discretization.meshfree.MovingSurfaceState

---

::: phydrax.discretization.meshfree.MovingGeometryRefresh

---

::: phydrax.discretization.meshfree.MovingSurfaceEvidence

---

::: phydrax.discretization.meshfree.MovingSurfaceStepResult

---

::: phydrax.discretization.meshfree.MovingSurfaceCheckpoint

---

::: phydrax.discretization.meshfree.MovingSurfaceEpochResult

---

::: phydrax.discretization.meshfree.MovingGCLPolicy

---

::: phydrax.discretization.meshfree.MovingStageAdmission

---

::: phydrax.discretization.meshfree.AbstractSurfaceMotionLaw

---

::: phydrax.discretization.meshfree.PrescribedVelocityMotion

---

::: phydrax.discretization.meshfree.ChartMotion

---

::: phydrax.discretization.meshfree.LevelSetMotion

---

::: phydrax.discretization.meshfree.MeanCurvatureMotion

---

::: phydrax.discretization.meshfree.BulkDrivenMotion

---

::: phydrax.discretization.meshfree.SurfaceMotion

---

::: phydrax.discretization.meshfree.SurfaceGeometryProvider

---

::: phydrax.discretization.meshfree.MeshfreeCapacityPolicy

---

::: phydrax.discretization.meshfree.MeshfreeCapacityMap

---

::: phydrax.discretization.meshfree.SurfaceShiftPolicy

---

::: phydrax.discretization.meshfree.SurfaceMeshShift

---

::: phydrax.discretization.meshfree.SurfaceResamplingPolicy

---

::: phydrax.discretization.meshfree.SurfaceResamplingResult

---

::: phydrax.discretization.meshfree.SurfaceQualityEvidence

---

::: phydrax.discretization.meshfree.SurfaceTransferPlan

## Point transfers and meshfree epochs

Declared conservative/positive/joint transfers with minimum-change native
solves and host audits, and one staged composition transaction per meshfree
epoch change (sample repair or committed physical surface event).

::: phydrax.discretization.meshfree.PointTransferRequest

---

::: phydrax.discretization.meshfree.PointTransferPlan

---

::: phydrax.discretization.meshfree.PreparedPointTransfer

---

::: phydrax.discretization.meshfree.PointTransferEvidence

---

::: phydrax.discretization.meshfree.PointTransferStatus

---

::: phydrax.discretization.meshfree.MeshfreeEpochChange

---

::: phydrax.discretization.meshfree.MeshfreeEventLineage

---

::: phydrax.discretization.meshfree.stage_meshfree_epoch

---

::: phydrax.discretization.meshfree.commit_meshfree_epoch

---

::: phydrax.discretization.meshfree.MeshfreeEpochCandidate

---

::: phydrax.discretization.meshfree.MeshfreeEpochReceipt

---

::: phydrax.discretization.meshfree.remap_live_histories

---

::: phydrax.discretization.meshfree.MeshfreeHistoryRemap

---

::: phydrax.discretization.meshfree.surface_event_epoch

---

::: phydrax.discretization.meshfree.MeshfreeSurfaceEventEpoch

---

::: phydrax.discretization.meshfree.MeshfreeSheetSupport

## Adaptivity

Indicators (never error bounds), stable-ID marking, bounded adaptation
proposals, their conservative transfer and the explicit epoch acceptance
boundary. See the [guide](../../guides_meshfree.md#adaptivity).

::: phydrax.discretization.meshfree.MeshfreeErrorIndicator

---

::: phydrax.discretization.meshfree.MeshfreeProbeJet

---

::: phydrax.discretization.meshfree.probe_residual_indicator

---

::: phydrax.discretization.meshfree.flux_jump_indicator

---

::: phydrax.discretization.meshfree.degree_difference_indicator

---

::: phydrax.discretization.meshfree.support_quality

---

::: phydrax.discretization.meshfree.MeshfreeSupportQuality

---

::: phydrax.discretization.meshfree.MeshfreeMarkingPolicy

---

::: phydrax.discretization.meshfree.mark_points

---

::: phydrax.discretization.meshfree.MeshfreeMarking

---

::: phydrax.discretization.meshfree.MeshfreeAdaptiveSupport

---

::: phydrax.discretization.meshfree.MeshfreeAdaptationPolicy

---

::: phydrax.discretization.meshfree.propose_adaptation

---

::: phydrax.discretization.meshfree.MeshfreeAdaptationProposal

---

::: phydrax.discretization.meshfree.MeshfreeAdaptationStatus

---

::: phydrax.discretization.meshfree.meshfree_fill_measures

---

::: phydrax.discretization.meshfree.prepare_adaptation_transfer

---

::: phydrax.discretization.meshfree.adaptation_acceptance

---

::: phydrax.discretization.meshfree.MeshfreeAdaptationAcceptance

## Incompressible flow

Transient incompressible flow on a closed exterior graph: IMEX transport and
viscous diffusion, compatible projection, moment-consistent velocity
reconstruction, cycle classification, and native fixed-step evidence.
See the [guide](../../guides_meshfree.md#incompressible-meshfree-flow).

::: phydrax.solver.MeshfreeIncompressibleFlowPlan

---

::: phydrax.solver.PreparedMeshfreeIncompressibleFlow

---

::: phydrax.solver.IncompressibleDensityModel

---

::: phydrax.solver.MeshfreeIncompressibleState

---

::: phydrax.solver.MeshfreeIncompressibleStepResult

---

::: phydrax.solver.MeshfreeFlowEvidence

---

::: phydrax.solver.MeshfreeFlowStatus

---

::: phydrax.solver.MeshfreeCycleEvidence

---

::: phydrax.solver.MeshfreeVelocityReconstruction

## Lagrangian particle flow

Material GMLS points with an explicit `EvolutionMeasure`, a weak dissipative
pressure projection on the refreshed or re-prepared cloud, transactional
acceptance, measure-aware point transfers, and SPH reconstruction exchange.
See the [guide](../../guides_meshfree.md#lagrangian-particle-flow).

::: phydrax.solver.MeshfreeLagrangianFlowPlan

---

::: phydrax.solver.PreparedMeshfreeLagrangianFlow

---

::: phydrax.solver.MeshfreeLagrangianFlowState

---

::: phydrax.solver.MeshfreeLagrangianStepResult

---

::: phydrax.solver.MeshfreeLagrangianEvidence

---

::: phydrax.solver.MeshfreeLagrangianStatus

---

::: phydrax.solver.MeshfreeMeasureTransferPlan

---

::: phydrax.solver.PreparedMeshfreeMeasureTransfer

---

::: phydrax.solver.MeshfreeMeasureTransferResult

---

::: phydrax.solver.MeshfreeSPHReconstruction

---

::: phydrax.solver.MeshfreeSPHComparison

---

::: phydrax.solver.MeshfreeMeasureConversion

## Constitutive laws and implicit solves

::: phydrax.discretization.meshfree.EdgeFeatureField

---

::: phydrax.discretization.meshfree.EdgeFrameFeatures

---

::: phydrax.discretization.meshfree.AbstractEdgeConstitutiveLaw

---

::: phydrax.discretization.meshfree.MonotoneEdgeConductance

---

::: phydrax.discretization.meshfree.LipschitzEdgeFlux

---

::: phydrax.discretization.meshfree.EdgeModelLipschitzCertificate

---

::: phydrax.discretization.meshfree.AbstractCoupledEdgeConstitutiveLaw

---

::: phydrax.discretization.meshfree.O3EdgeInvariants

---

::: phydrax.discretization.meshfree.MonotoneCoupledEdgeFlux

---

::: phydrax.discretization.meshfree.InvariantConvexEdgeCertificate

---

::: phydrax.discretization.meshfree.O3EdgeNetwork

---

::: phydrax.discretization.meshfree.LipschitzCoupledEdgeFlux

---

::: phydrax.discretization.meshfree.EdgeFeatureCoverage

---

::: phydrax.discretization.meshfree.EdgeCoverageAssessment

---

::: phydrax.discretization.meshfree.MeshfreeConservationProblem

---

::: phydrax.discretization.meshfree.prepare_meshfree_conservation_solve

---

::: phydrax.discretization.meshfree.PreparedMeshfreeConservationSolve

---

::: phydrax.discretization.meshfree.MeshfreeConservationResult

---

::: phydrax.discretization.meshfree.MeshfreeConservationAdjoint

---

::: phydrax.discretization.meshfree.MeshfreeConservationLedger

---

::: phydrax.discretization.meshfree.MeshfreeConstitutiveEvidence

---

::: phydrax.discretization.meshfree.MeshfreeCoupledConservationProblem

---

::: phydrax.discretization.meshfree.prepare_meshfree_coupled_conservation_solve

---

::: phydrax.discretization.meshfree.PreparedMeshfreeCoupledConservationSolve

---

::: phydrax.discretization.meshfree.MeshfreeCoupledConservationResult

---

::: phydrax.discretization.meshfree.MeshfreeCoupledConservationAdjoint

---

::: phydrax.discretization.meshfree.MeshfreeComponentConservationLedger

## Learned corrections

Moment-exact projection of learned metric candidates with explicit sign,
coercivity and witness evidence, and parity-audited oriented flux corrections.
See the [guide](../../guides_meshfree.md#learned-corrections).

::: phydrax.discretization.meshfree.MeshfreeMetricCorrectionPolicy

---

::: phydrax.discretization.meshfree.MeshfreeMetricCorrectionPlan

---

::: phydrax.discretization.meshfree.MeshfreeMetricProjection

---

::: phydrax.discretization.meshfree.MeshfreeMetricCorrection

---

::: phydrax.discretization.meshfree.MeshfreeMetricCorrectionEvidence

---

::: phydrax.discretization.meshfree.MeshfreeEdgeFluxCandidate

---

::: phydrax.discretization.meshfree.MeshfreeEdgeFluxCorrectionPlan

---

::: phydrax.discretization.meshfree.MeshfreeEdgeFluxProjection

---

::: phydrax.discretization.meshfree.MeshfreeEdgeFluxCorrection

---

::: phydrax.discretization.meshfree.MeshfreeEdgeFluxCorrectionEvidence

---

::: phydrax.discretization.meshfree.MeshfreeCorrectionStatus

## Distributed operators and precision

`DistributedMeshfreeOperator.bind` partitions a prepared
`SparseCoordinateOperator` (row or edge relation, scalar or block fibers) over
the owned target rows of a `DistributedPointLayout`; every route belongs to its
target row's owner exactly once. Sources on other owners become deduplicated
halo columns. `apply` gathers halo columns and reduces owned routes,
`transpose_apply` returns halo contributions to their owners exactly once, and
`adjoint_apply` composes the transpose with the bound Euclidean or diagonal
pairings. `from_neighbor_rows` binds distributed neighbor rows and per-row
coefficients without any global host relation. Binding refuses an incomplete
halo, and a binding is valid only for the layout epochs in its evidence; rebind
after migration. `distributed_sum`, `distributed_inner`, and `distributed_norm`
reduce owned active slots once (owner-local slot order, then the collective
all-reduce), reproducible for a fixed partition and backend. Forced CPU devices
demonstrate functional parity with the single-device operator only.

`MeshfreePrecisionPolicy` resolves geometry, coefficient, fit, compute,
accumulation, residual, certification, communication, checkpoint, and output
dtypes (float32 or float64) through the native precision request and evidence.
Certification is never narrower than geometry, fit, or residual; residual and
accumulation are never narrower than compute; communication never narrows
exchanged fields; checkpoints never narrow restartable state. Requested float64
is refused when JAX x64 is disabled, and `cast("certification", ...)` refuses
wider inputs instead of rounding near ties. Pass the policy as `precision=` to
`MeshfreeNeighborhoodPlan`/`MeshfreeEdgeRelationPlan`: geometry stores the
coordinates, certification decides neighbor identity and distances, and the
prepared neighborhood carries it to `prepare_local_stencils` (fit role for the
local solve, coefficient role for stored weights, effective dtypes in
`LocalStencilReport.precision`) and to `MeshfreeOperator` (compute, output and
accumulation roles).

::: phydrax.discretization.meshfree.DistributedMeshfreeOperator

---

::: phydrax.discretization.meshfree.DistributedMeshfreeEvidence

---

::: phydrax.discretization.meshfree.distributed_sum

---

::: phydrax.discretization.meshfree.distributed_inner

---

::: phydrax.discretization.meshfree.distributed_norm

---

::: phydrax.discretization.meshfree.MeshfreePrecisionPolicy

## Native coupled publication

::: phydrax.solver.coupling.MeshfreeComponent

---

::: phydrax.solver.coupling.MeshfreeCapacity

---

::: phydrax.solver.coupling.SurfaceExchangeLaw

---

::: phydrax.solver.coupling.SurfaceExchangeEvidence

---

::: phydrax.solver.coupling.SurfaceDeposition

---

::: phydrax.solver.coupling.SurfaceEpochRelocation

---

::: phydrax.solver.coupling.LangmuirAdsorptionFlux

---

::: phydrax.solver.coupling.MeshfreeBulkSurfaceMethod
