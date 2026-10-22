# Meshing

See the [meshing guide](../guides_meshing.md) for identity, provider,
certification, and topology-transition contracts.

The capability catalog records admitted source, route, family, order, control,
placement, derivative, and transfer combinations. Use the
[evidence guide](../guides_meshing.md#evidence-and-resource-limits) to interpret
request-bound certification and qualification records.

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

Layer/core accepted-state archives retain every epoch, including coefficient
fields, finite-volume material density, the solved harmonic field and damage
history. A cold continuation restores the penultimate epoch and replays the
final inverse coarsening against the archived warm final epoch; it does not
initialize replacement fields from the restored geometry.
The original producer binds capacity to the native resource limits and
authors the requested material densities and material-site identity in the
initial accepted state. Its resolution and core-size selectors admit only
the original source values; varying them refuses rather than silently
substituting another source or claiming a resolution-scaled benchmark.

Within one request, algebraic physical cell measures may reuse the same
immutable element/coefficient-bank binding and outward publication error.
Every reuse rechecks the caller's current integration tolerance. Layer/core DG
and compatible-cochain transfers share their issued rooted nested-reference
preparation, while retaining the independent field functionals, commuting
checks and original work/storage allowances.
An exact mixed refinement reuses the source's total physical measure only
after the canonical nested-reference owner proves an oriented full tiling of
every source cell. Inverse coarsening still independently measures both maps.
Exact form moments retain unchanged affine reference arguments locally, keyed
by the complete execution views and their source-owned rational banks.


## Requests and controls

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
        - prepare_native_layer_association_transfer
        - SourceProximityGapEvidence
        - RegionBoundaryEvidence
        - RegionMeshingEvidence
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

## Results, certification, and associations

::: phydrax.meshing
    options:
      members:
        - CellMeshingResult
        - CellMeshAuditDisposition
        - CellMeshAuditPolicy
        - CellMeshAuditReport
        - GeometryAssociation
        - GeometryAssociationKind
        - GeometryAssociationSourceBank
        - GeometryAssociationSourceIdentities
        - ComposedAssociationTransfer
        - ImplicitAssociationTransfer
        - MappedReferenceAssociationTransfer
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
        - evaluate_cell_quality
        - FiniteVolumeQualityEvaluation
        - evaluate_finite_volume_quality
        - import_cell_mesh
        - export_cell_mesh
        - export_mesh_array_artifact

## Adaptation and optimization

`MeshLineage.validate_complete(source_mesh, target_mesh)` validates the bound
source and target topology and accounts for every retained entity. It returns
host-only `LineageAccounting`: `lineage_id` identifies the checked lineage;
`dimensions` stores dimension-ordered `EntityLineageAccounting` records with
`relations`, `created`, and `deleted` counts. Missing or contradictory entity
accounting refuses instead of producing a completeness record. `to_record()`
serializes the validated counts without retaining either mesh.

::: phydrax.meshing
    options:
      members:
        - MeshLineage
        - LineageAccounting
        - EntityLineageAccounting
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
        - MeshProposalFeatures
        - MixedAdaptationHierarchy
        - MixedLayerColumns
        - GeometryRealizationMeshAdaptation
        - LevelSetEvidence
        - LevelSetMeshAdaptation
        - PolyhedralAdaptationOperation
        - PolyhedralMeshAdaptation
        - BisectionCompatibility
        - BisectionHierarchy
        - BisectionEvidence
        - BisectionUniformRefinement
        - LocalMetricEvidence
        - PreparedAdaptiveSimplex
        - prepare_adaptive_simplex
        - commit_adaptive_simplex
        - PartitionedAdaptiveSimplex
        - partition_adaptive_simplex
        - commit_partitioned_adaptive_simplex
        - PartitionedMeshAdaptationResult
        - CollectiveMeshIdentity
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

## Providers and native construction

::: phydrax.meshing
    options:
      members:
        - GmshProvider
        - GmshOptions
        - GmshSurfaceAlgorithm
        - GmshVolumeAlgorithm
        - GmshHighOrderOptimization
        - GmshMeshingPlan
        - GmshRemeshingPlan
        - GmshSession
        - NativeMeshingProvider
        - NativeMeshingSession
        - NativeMeshingPlan
        - NativeMeshingOptions
        - NativeMeshingRoute
        - NativeCurveSchedule
        - NativeMeshingSource
        - NativePlanarSource
        - NativeImplicitSource
        - NativeCurveSource
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
        - BlockInterfaceControl
        - declared_plc_domain
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

## Distribution, periodicity, and coupling

::: phydrax.meshing
    options:
      members:
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
        - PeriodicAssociationTransfer
        - PeriodicMeshConstruction
        - PeriodicPointOrbits
        - PeriodicQuotientEvidence
        - PeriodicRefinement
        - periodic_cell_from_constraints
        - periodic_delaunay_mesh
        - publish_periodic_simplices
        - refine_periodic_mesh
        - ContactCoupling
        - OversetCoupling
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
        - CouplingSearchStatus
        - CouplingSearchEvidence
        - CouplingSearchError

## Proposals and solver-aware decisions

::: phydrax.meshing
    options:
      members:
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

## Qualification preparation and evidence

The qualification owner prepares authored sources, binds physical fields to
accepted carriers, and produces scientific continuation and benchmark evidence.
Its preparation entry points consume domain limits; resource scopes and tickets
remain inside the package operations. Repository tools use these operations for
CLI workflows and serialize their results at the reporting boundary.

`declared_geometry_edge_lengths` evaluates affine source edges after validating
their published coordinate carriers. Authenticated retained coefficients and
once-rounded descendants follow their declared publication laws; inconsistent
source maps and unsupported curved maps refuse through the supplied
`failure_factory`. This check is shared by product tests and CLI qualification
without depending on tool-owned scientific logic.

`validate_envelope_acceptance` requires keyword-only `limits=` from the
caller-owned meshing specification. Acceptance must use that request's limits,
not an independently constructed allowance. This API requirement does not
establish that envelope qualification or retained-evidence replay has passed.

::: phydrax.meshing.qualification
    options:
      members:
        - AuthoredPeriodicLayerSource
        - prepare_corrected_periodic_layer_source
        - periodic_layer_source_identity_record
        - LayerCoreLifecyclePlans
        - LayerCoreLifecycleReport
        - LayerCoreAcceptedState
        - initial_layer_core_state
        - continue_layer_core_state
        - warm_layer_core_initial_state
        - warm_layer_core_step
        - layer_core_accepted_epoch
        - restore_layer_core_state
        - layer_core_continuation_defects
        - layer_core_restoration_defects
        - layer_core_lineage_records
        - continue_layer_core_lifecycle
        - continue_layer_core_fields
        - continue_layer_core_finite_volume
        - continue_layer_core_history
        - rigid_planar_successor
        - deform_planar_interior
        - native_planar_part
        - field_plans
        - field_declarations
        - prepare_poisson_update
        - poisson_update
        - restart_native
        - PreparedEnvelopeSource
        - prepare_envelope_source
        - prepare_envelope_association_transfer
        - validate_envelope_acceptance
        - prepare_envelope_accepted_epoch
        - prepare_envelope_solver
        - SurfaceAcceptedEpoch
        - prepare_surface_fields
        - prepare_surface_checkpoint
        - prepare_periodic_accepted_epoch
        - FamilyQualityMeasurements
        - prepare_family_association_transfer
        - prepare_family_quality_measurements
        - write_family_oracle
        - read_family_oracle
        - DistributedCheckpointSourceArrays
        - prepare_distributed_checkpoint_source_closure
        - bind_distributed_checkpoint_source_arrays
        - QualifiedSimplexArrays
        - qualify_simplex_resources
        - declared_geometry_edge_lengths
        - prepare_plc_associations
        - PeriodicSourceQuadrature
        - prepare_periodic_source_quadrature
        - PredicateFamilyQualification
        - reference_predicate_signs
        - qualify_predicate_families
        - qualify_bvh_scaling
        - qualify_overlap_pair_scaling
        - prepare_benchmark_provider_input
