# Meshfree and point-cloud solvers

See [Meshfree solvers](../../guides_meshfree.md) for numerical contracts,
preparation/resource boundaries, metric admission, and migration.

## Neighborhoods and local approximation

::: phydrax.discretization.meshfree.MeshfreeNeighborhoodPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeNeighborhood

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

::: phydrax.discretization.meshfree.MeshfreeOperator

## Point-cloud calculus and observations

::: phydrax.discretization.PointCloudPlan

---

::: phydrax.discretization.PreparedPointCloudDiscretization

---

::: phydrax.discretization.prepare_point_cloud_field_reconstruction

---

::: phydrax.discretization.PointCloudFieldReconstructionKernel

## Sparse elliptic solves and stabilization

::: phydrax.discretization.PointDiffusionOperator

---

::: phydrax.discretization.PointBoundaryPlan

---

::: phydrax.discretization.PointCloudPoissonPlan

---

::: phydrax.discretization.PreparedPointCloudPoisson

---

::: phydrax.discretization.PointCloudPoissonResult

---

::: phydrax.discretization.point_sbp_report

---

::: phydrax.discretization.PointSBPReport

---

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

::: phydrax.discretization.meshfree.PreparedMeshfreeHierarchy

---

::: phydrax.discretization.meshfree.MeshfreeHierarchyEvidence

---

::: phydrax.discretization.meshfree.meshfree_multigrid_builder

---

::: phydrax.linalg.ProjectedPseudoinversePreconditionerBuilder

---

::: phydrax.linalg.ProjectedPseudoinversePreconditioner

## Intrinsic surface preparation

::: phydrax.discretization.meshfree.ImplicitSurfaceGeometry

---

::: phydrax.discretization.meshfree.SampledSurfaceGeometry

---

::: phydrax.discretization.meshfree.SurfaceGeometryEvidence

---

::: phydrax.discretization.meshfree.SurfaceGeometryEvaluation

---

::: phydrax.discretization.meshfree.SurfaceQuadraturePolicy

---

::: phydrax.discretization.meshfree.SurfaceQuadratureEvidence

---

::: phydrax.discretization.meshfree.SurfacePointCloudPlan

---

::: phydrax.discretization.meshfree.PreparedSurfacePointCloud

---

::: phydrax.discretization.meshfree.SurfaceRefreshResult

---

::: phydrax.discretization.meshfree.SurfaceFieldReconstructionKernel

## Conservative metrics and transport

::: phydrax.discretization.meshfree.MeshfreeExteriorCalculusPlan

---

::: phydrax.discretization.meshfree.PreparedMeshfreeExteriorCalculus

---

::: phydrax.discretization.meshfree.MeshfreeMetricPolicy

---

::: phydrax.discretization.meshfree.PreparedMeshfreeMetric

---

::: phydrax.discretization.meshfree.MeshfreeMetricResult

---

::: phydrax.discretization.meshfree.MeshfreeDiffusionOperator

---

::: phydrax.discretization.meshfree.MeshfreeStiffnessEvidence

---

::: phydrax.discretization.meshfree.MeshfreeBoundaryQuadrature

---

::: phydrax.discretization.meshfree.MeshfreeAdvection

---

::: phydrax.discretization.meshfree.MeshfreeAdvectionResult

---

::: phydrax.discretization.meshfree.edge_upwind_content

---

::: phydrax.linalg.SparseRowRankPolicy

---

::: phydrax.linalg.SparseRowRankEvidence

---

::: phydrax.linalg.SparseRowRankRefusal

---

::: phydrax.linalg.prepare_sparse_row_rank

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

::: phydrax.discretization.meshfree.MeshfreeCapacityPolicy

---

::: phydrax.discretization.meshfree.MeshfreeCapacityMap

---

::: phydrax.discretization.meshfree.SurfaceShiftPolicy

---

::: phydrax.discretization.meshfree.SurfaceShiftResult

---

::: phydrax.discretization.meshfree.surface_relative_advection

---

::: phydrax.discretization.meshfree.SurfaceRelativeAdvectionResult

---

::: phydrax.discretization.meshfree.SurfaceResamplingPolicy

---

::: phydrax.discretization.meshfree.SurfaceResamplingResult

---

::: phydrax.discretization.meshfree.SurfaceQualityEvidence

---

::: phydrax.discretization.meshfree.SurfaceTransferPlan

---

::: phydrax.discretization.meshfree.PreparedSurfaceTransfer

---

::: phydrax.discretization.meshfree.SurfaceTransferEvidence

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

## Native coupled publication

::: phydrax.solver.coupling.MeshfreeComponent

---

::: phydrax.solver.coupling.MeshfreeCapacity

---

::: phydrax.solver.coupling.SurfaceExchangeLaw

---

::: phydrax.solver.coupling.SurfaceExchangeEvidence

---

::: phydrax.solver.coupling.LangmuirAdsorptionFlux

---

::: phydrax.solver.coupling.MeshfreeBulkSurfaceMethod
