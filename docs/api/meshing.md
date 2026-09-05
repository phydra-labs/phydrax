# Meshing

See the [meshing guide](../guides_meshing.md) for identity, provider,
certification, and topology-transition contracts.

::: phydrax.meshing
    options:
      members:
        - MeshingScope
        - CellMeshingTarget
        - CellFamilyPolicy
        - SurfaceMeshingSpec
        - SurfaceRemeshingSpec
        - VolumeMeshingSpec
        - MeshingLimits
        - MeshPatch
        - MeshZone
        - MeshLabel
        - MeshAttribute
        - UniformSizeControl
        - CurvatureSizeControl
        - ProximitySizeControl
        - ResolvedSizeField
        - MeshMetricField
        - resolve_size_controls
        - normalize_mesh_metric
        - CellMeshingResult
        - CellMeshAuditPolicy
        - CellMeshAuditReport
        - GeometryAssociation
        - MeshingComplianceReport
        - MeshingTrace
        - MeshingFailure
        - certify_cell_mesh
        - evaluate_cell_quality
        - import_cell_mesh
        - export_cell_mesh
        - export_mesh_array_artifact
        - MeshLineage
        - CellMeshTransition
        - VertexInterpolationStencil
        - refine_triangle_mesh
        - TargetMatrixOptimizationPlan
        - optimize_cell_mesh
        - optimize_cell_geometry_coordinates
        - GmshProvider
        - GmshOptions
        - GmshSession
        - NativeImplicitProvider
        - ManifoldProvider
        - SurfaceBooleanOperation
        - MmgProvider
        - MmgOptions
        - FTetWildProvider
        - FTetWildOptions
        - OrientedPointCloud
        - PoissonProvider
        - PoissonReconstructionSpec
        - OpenVDBProvider
        - OpenVDBMeshingSpec
        - OmegaHProvider
        - VoroCrustProvider
        - VoroCrustOptions
        - TiogaProvider
        - TiogaOptions
        - MeshPart
        - MeshAssembly
        - MeshDistribution
        - ConformalCoupling
        - PeriodicCoupling
        - ContactCoupling
        - OversetCoupling
        - MeshMarkingProposal
        - MeshSizeProposal
        - MeshMetricProposal
        - MeshCoordinateProposal
        - MeshProposalSafetyPolicy
        - MeshProposalTransaction
        - project_mesh_proposal
        - prepare_mesh_proposal

::: phydrax.discretization.CellGeometrySpec

::: phydrax.SpatialCoordinateContract

::: phydrax.discretization.CellPartition

::: phydrax.interchange.MeshArrayArtifact

::: phydrax.interchange.MeshArraySelection
