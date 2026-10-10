# Meshing

See the [meshing guide](../guides_meshing.md) for identity, provider,
certification, and topology-transition contracts.

The symbols below are implementation/API documentation, not a release index.
Exact source × route × family × order × control × placement × derivative ×
transfer combinations are recorded by the generated capability catalog.
Independent flags must not be combined into an unsupported request. The
[current closure table](../guides_meshing.md#evidence-vocabulary-and-current-closure-state)
keeps implemented, tested, qualified, released, optional-provider, refused, and
open research states distinct.

For an exact scalar zero set with sound value and derivative enclosures,
`NativeMeshingOptions("implicit_adaptive_tetrahedral")` combines the canonical
adaptive discovery policy with the native volume schedule. It publishes only
after complete regular topology discovery, independently bracketed cycle-root
witnesses, exact coverage of the extracted boundary, global embedding, and
continuous two-sided source-fidelity certification. Source-distance bounds come
from the exhaustive enclosure cover and actual root witnesses, not from treating
an arbitrary scalar field as a signed distance. Unknown, hidden, tangential, or
singular boxes remain explicit failures within the authored resource bounds.
`"implicit_restricted_delaunay"` retains its distinct established-reach analytic
specialization; the adaptive route never substitutes that specialization for a
general field. Multiple material/junction sources use the shared image/interface
complex, rather than assigning a tetrahedron from one scalar sample.


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
        - CurveMeshingSpec
        - CurveJunction
        - CurveEnd
        - MeshQualityTarget
        - MeshingLimits
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
        - associate_mesh_vertices
        - associate_mesh_entities
        - propagate_association
        - rederive_association
        - MeshingComplianceReport
        - MeshingTrace
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
        - NativeMeshingRoute
        - NativeCurveSchedule
        - NativeMeshingSource
        - NativePlanarSource
        - NativeImplicitSource
        - NativeCurveSource
        - ManifoldProvider
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
        - AdaptationAction
        - ComposedAssociationTransfer
        - ImplicitAssociationTransfer
        - MappedReferenceAssociationTransfer
        - BisectionUniformRefinement
        - BlockInterfaceControl
        - DecisionBudget
        - DeviceGenerationCandidates
        - DeviceGenerationEvidence
        - DeviceGenerationLayout
        - DeviceGenerationStatus
        - DeviceGenerationUpdate
        - PreparedDeviceGeneration
        - prepare_device_generation
        - evaluate_device_generation_candidates
        - execute_device_generation_round
        - DeviceTetraMetricEvidence
        - DeviceTetraMetricLayout
        - DeviceTetraMetricReport
        - DeviceTetraMetricState
        - DeviceTetraMetricUpdate
        - prepare_device_tetra_metric
        - adapt_device_tetra_metric
        - PreparedDistributedSurfaceGeneration
        - prepare_distributed_surface_generation
        - ErrorQuantity
        - FixedEpochDerivativeEvidence
        - MeasuredAdaptationCost
        - PhysicalErrorEvidence
        - RouteFeasibility
        - SolverAwareCandidate
        - SolverAwareDecision
        - GeometrySourceEntityRole
        - GeometryRealizationMeshAdaptation
        - LevelSetEvidence
        - LevelSetMeshAdaptation
        - MeshCertificationCheck
        - MeshCertificationInputs
        - MeshCertificationOutcome
        - MeshCertificationPreparedEvidence
        - MeshCertificationReport
        - MeshCertificationRoute
        - MeshCertificationSchedule
        - MeshingEvidenceBinding
        - acceptance_stage_report
        - certify_meshing_acceptance
        - declared_plc_domain
        - MeshProposalFeatures
        - MixedAdaptationHierarchy
        - MixedLayerColumns
        - MultiblockConstruction
        - StructuredConstruction
        - TransfiniteBlock
        - TransfiniteCurveControl
        - TransfiniteSurfaceControl
        - assemble_independent_blocks
        - generate_structured_block
        - glue_structured_blocks
        - SweepConstruction
        - SweepControl
        - SweepMapKind
        - generate_sweep
        - PiecewiseLinearComplex
        - NativeVolumeSchedule
        - NativeLayerCoreSource
        - NativeHexGridRoute
        - NativeHexGridSchedule
        - NativeMappedHexSource
        - NativePeriodicSource
        - NativePlcSource
        - NativePolyhedralSchedule
        - NativePolyhedralSource
        - NativeStructuredSchedule
        - NativeStructuredSource
        - NativeSurfaceEnvelopeSource
        - NativeSurfaceSchedule
        - NativeSurfaceSource
        - NativeSweepSource
        - NativeMeshingPhase
        - NativeMeshingPhaseMeasurement
        - NativeMeshingPhaseRecorder
        - PeriodicAssociationTransfer
        - PeriodicMeshConstruction
        - PeriodicPointOrbits
        - PeriodicQuotientEvidence
        - PeriodicRefinement
        - periodic_cell_from_constraints
        - periodic_delaunay_mesh
        - publish_periodic_simplices
        - refine_periodic_mesh
        - PolyhedralAdaptationOperation
        - PolyhedralMeshAdaptation
        - OversetCellStatus
        - OversetConnectivity
        - OversetConnectivityError
        - OversetDonorPacket
        - OversetPartBlanking
        - OversetPartSpec
        - OversetPolicy
        - OversetReceptorEvidence
        - OversetRegistration
        - OversetVertexStatus
        - PreparedOversetFieldTransfer
        - PreparedOversetMotion
        - prepare_overset_connectivity
        - prepare_overset_conservative_remap
        - prepare_overset_conservative_state_transport
        - prepare_overset_field_transfer
        - prepare_overset_motion_rebind
        - prepare_native_layer_association_transfer
        - SourceProximityGapEvidence
        - RegionBoundaryEvidence
        - RegionMeshingEvidence
        - metric_simplex_quality
        - transition_interface_attachment

::: phydrax.discretization.CellGeometrySpec

::: phydrax.discretization.CellValidityPolicy

::: phydrax.discretization.CellValidityCertificate

::: phydrax.discretization.CellValidityStatus

::: phydrax.discretization.certify_cell_geometry_validity

::: phydrax.SpatialCoordinateContract

::: phydrax.discretization.CellPartition

::: phydrax.interchange.MeshArrayArtifact

::: phydrax.interchange.MeshArraySelection

## Lifecycle and solver transition records

`MeshingAcceptedEpoch` binds one accepted carrier to complete named field banks;
it does not expose the qualification-only source-closure codec as a generic
serializer. `refinement_parent_cells` returns a nesting witness only when every
target cell is preserved or refined from exactly one source cell.
`MaterialTopologyTransferResult` retains the actual material transaction and
the prepared conservative remap owners used to produce it.


::: phydrax.solver.MaterialTopologyTransferResult

---

::: phydrax.solver.refinement_parent_cells
