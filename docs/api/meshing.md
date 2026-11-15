# Meshing

See the [meshing guide](../guides_meshing.md) for identity, provider,
certification, and topology-transition contracts.

## Coupled tetrahedral relocation

The local coupled model uses a sparse native quadratic program with explicitly
identified source, carrier, orbit, metric, and native rows. Structural support
comes from topology, not coefficient values; supported zero coefficients remain
in the prepared pattern. A search retains at most one primary and one certificate
template, reused only when their structure and policy match. Actual sparse input,
factorization, symbolic-work, and workspace limits govern admission.
The structural namespace identifies the source/frame/generation, ordered
vertices, row roles and entity identities (including affected edges and
periodic equalities), ordered supports, capacities, and dtype. Selected target
edges and source-cell choices may change coefficients without changing those
structural meanings. Their complete immutable binding and each fresh numeric
provenance remain attached to the actual solve; historical identities are not
reused.

The primary result is retained unchanged. An optimal result or an already
valid infeasibility ray needs no second program. Otherwise, an unresolved primary
may use one bounded native Farkas LP: equality multipliers are free within
`[-1, 1]`, and inequality multipliers are in `[0, 1]`. Its candidate is audited
against the original conic program, not the certificate LP. The primary and
certificate results and demands remain separate; iteration work is their actual
sum, not a relabeled primary count. A local certificate does not establish global
infeasibility.

`TetraMetricProposalError` retains the primary `result`, `policy`, `demand`, and
`binding`, and, when available, `certificate_result`, `certificate_demand`,
`ray_audit`, and `certificate_error` for certificate preparation or execution
failure. Existing source admissibility and native geometry guards remain in force.
Its iteration properties retain separate and aggregate work. If certificate
execution started but returned no receipt, its iteration count and the aggregate
are unknown (`None`), not a fabricated zero. Numeric-binding provenance is
constructed outside the already-compiled native solve.

::: phydrax.meshing
    options:
      members:
        - MeshingScope
        - MeshingEntityKind
        - CellMeshingTarget
        - CellFamilyPolicy
        - SurfaceMeshingSpec
        - SurfaceRemeshingSpec
        - VolumeMeshingSpec
        - MeshPatch
        - MeshZoneRole
        - RegionRole
        - MeshZone
        - MeshLabel
        - MeshAttribute
        - RegionControl
        - PatchControl
        - LayerSchedule
        - BoundaryLayerRoute
        - BoundaryLayerCollisionPolicy
        - BoundaryLayerCornerPolicy
        - BoundaryLayerControl
        - BoundaryLayerPolicy
        - BoundaryLayerEvidence
        - BoundaryLayerMesh
        - prepare_boundary_layers
        - BoundaryLayerExtrusion
        - prepare_boundary_layer_extrusion
        - ProtectedFeature
        - BackgroundMetricMode
        - BackgroundMetricControl
        - SurfaceReconstructionControl
        - PlanarBandControl
        - PlanarBandPlan
        - PlanarBandResult
        - prepare_planar_bands
        - SizeCombinationPolicy
        - SizeCompliancePolicy
        - UniformSizeControl
        - CurvatureSizeControl
        - ProximitySizeControl
        - ResolvedSizeField
        - resolve_size_controls
        - size_field_metric
        - MeshMetricField
        - MeshMetricSamples
        - MetricNormalizationPolicy
        - MetricNormalizationEvidence
        - MetricComplexityStatus
        - normalize_mesh_metric
        - MetricGradationKind
        - MetricGradationPolicy
        - MetricGradationEvidence
        - MetricGradationStatus
        - MetricGradationError
        - grade_mesh_metric
        - MetricCombinationEvidence
        - MetricCombinationResult
        - combine_mesh_metrics
        - interpolate_mesh_metric
        - metric_edge_lengths
        - HessianMetricEvidence
        - lp_metric_from_hessian
        - CellMeshingResult
        - CellMeshAuditDisposition
        - CellMeshAuditPolicy
        - CellMeshAuditReport
        - GeometryAssociation
        - GeometryAssociationKind
        - GeometryAssociationProvenance
        - AssociationPropagationPolicy
        - AssociationPropagationError
        - BRepAssociationTransfer
        - NativeLayerCoreAssociationTransfer
        - associate_mesh_vertices
        - associate_mesh_entities
        - propagate_association
        - rederive_association
        - MeshingComplianceReport
        - MeshingTrace
        - MeshingDiagnostic
        - MeshingFailure
        - MeshingFailureEvidence
        - certify_cell_mesh
        - evaluate_cell_quality
        - FiniteVolumeQualityEvaluation
        - evaluate_finite_volume_quality
        - import_cell_mesh
        - export_cell_mesh
        - export_mesh_array_artifact
        - MeshLineage
        - CellMeshTransition
        - VertexInterpolationStencil
        - MeshAdaptationRoute
        - MeshAdaptationStatus
        - MeshAdaptationPolicy
        - MarkedMeshAdaptation
        - MetricMeshAdaptation
        - RelocationMeshAdaptation
        - PreparedMeshAdaptation
        - MeshAdaptationResult
        - prepare_mesh_adaptation
        - execute_mesh_adaptation
        - BisectionCompatibility
        - BisectionHierarchy
        - BisectionEvidence
        - LocalMetricEvidence
        - PreparedAdaptiveSimplex
        - prepare_adaptive_simplex
        - commit_adaptive_simplex
        - PartitionedAdaptiveSimplex
        - partition_adaptive_simplex
        - commit_partitioned_adaptive_simplex
        - PreparedDeviceMetricAdaptation
        - DeviceMetricLayout
        - DeviceMetricState
        - DeviceMetricReport
        - DeviceMetricUpdate
        - DeviceMetricEvidence
        - prepare_device_metric_adaptation
        - adapt_device_metric
        - commit_device_metric_adaptation
        - project_hp_lineage
        - MeshQualityObjective
        - MeshUntanglingPolicy
        - MeshUntanglingEvidence
        - TargetMatrixOptimizationPlan
        - MeshOptimizationStatus
        - MeshOptimizationResult
        - optimize_cell_mesh
        - CellGeometryOptimizationResult
        - optimize_cell_geometry_coordinates
        - HighOrderCurvingPolicy
        - HighOrderCurvingStatus
        - HighOrderCurvingResult
        - CurvedGeometryEvidence
        - curve_cell_mesh
        - verify_curved_geometry
        - MeshMotionDecision
        - MeshMotionMonitorPolicy
        - MeshMotionMonitor
        - MeshMotionAssessment
        - MeshMotionAdvance
        - advance_mesh_motion
        - GmshProvider
        - GmshOptions
        - GmshSurfaceAlgorithm
        - GmshVolumeAlgorithm
        - GmshHighOrderOptimization
        - GmshMeshingPlan
        - GmshRemeshingPlan
        - GmshSession
        - NativeMeshingProvider
        - NativeMeshingPlan
        - NativeMeshingOptions
        - NativeVolumeSchedule
        - PlcSearchBudget
        - PLC_SEARCH_BUDGET
        - NativeMeshingRoute
        - NativeMeshingSource
        - SurfaceEnvelopeSource
        - ManifoldProvider
        - SurfaceBooleanOperation
        - MmgProvider
        - MmgOptions
        - MmgAdaptationPlan
        - MmgAdaptationResult
        - MmgLevelSet
        - MmgLagrangianMotion
        - MmgLagrangianMode
        - MmgFieldTransfer
        - MmgReference
        - MmgReferenceRetention
        - MmgSessionEvidence
        - FTetWildProvider
        - FTetWildOptions
        - OrientedPointCloud
        - PoissonProvider
        - PoissonReconstructionSpec
        - PoissonBoundaryCondition
        - OpenVDBProvider
        - OpenVDBMeshingSpec
        - OpenVDBLevelSetRebuild
        - OmegaHProvider
        - OmegaHOptions
        - OmegaHField
        - OmegaHFieldTransfer
        - OmegaHClassification
        - OmegaHAdaptationResult
        - OmegaHAdaptationEvidence
        - OmegaHFieldEvidence
        - OmegaHTransferredField
        - OmegaHPartition
        - VoroCrustProvider
        - VoroCrustOptions
        - TiogaProvider
        - TiogaOptions
        - TiogaRegistration
        - MeshPart
        - MeshAssembly
        - MeshInterfaceAttachment
        - MeshDistribution
        - MeshPartitionKind
        - MeshPartitionPolicy
        - MeshPartitionEvidence
        - prepare_mesh_distribution
        - MeshDistributionTransition
        - prepare_distribution_transition
        - MetisUnavailableError
        - MetisPartitionError
        - ConformalCoupling
        - PeriodicCoupling
        - ContactCoupling
        - OversetCoupling
        - CouplingSearchStatus
        - CouplingSearchEvidence
        - CouplingSearchError
        - MeshProposalFeatures
        - AdaptationAction
        - ErrorQuantity
        - PhysicalErrorEvidence
        - MeasuredAdaptationCost
        - RouteFeasibility
        - MeshAdaptationCandidate
        - MeshAdaptationDecision
        - MeshAdaptationObjective
        - FixedEpochDerivativeEvidence
        - MeshMarkingProposal
        - MeshSizeProposal
        - MeshMetricProposal
        - MeshCoordinateProposal
        - MeshProposalSafetyPolicy
        - MeshProposalTransaction
        - AbstractMeshProposer
        - LearnedMeshProposer
        - project_mesh_proposal
        - prepare_mesh_proposal

::: phydrax.discretization.CellGeometrySpec

::: phydrax.discretization.CellValidityPolicy

::: phydrax.discretization.CellValidityCertificate

::: phydrax.discretization.CellValidityStatus

::: phydrax.discretization.certify_cell_geometry_validity

::: phydrax.SpatialCoordinateContract

::: phydrax.discretization.CellPartition

::: phydrax.interchange.MeshArrayArtifact

::: phydrax.interchange.MeshArraySelection
