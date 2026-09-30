# Block AMR

The public block-AMR surface separates immutable hierarchy geometry, one realized
`TopologyEpoch`, canonical stable-block metadata, numeric hierarchy payloads,
FillPatch routes, finite-volume dynamics, solver scheduling, composite elliptic
operators, and optional distributed ownership. Topology compilation is host-only;
numeric kernels are JAX transformations over one fixed epoch.

## Hierarchy, topology, and field transition

::: phydrax.discretization.BlockLevelPlan

---

::: phydrax.discretization.BlockHierarchyPlan

---

::: phydrax.discretization.BlockMetadata

---

::: phydrax.discretization.BlockHierarchyTopology

---

::: phydrax.discretization.BlockLevelState

---

::: phydrax.discretization.BlockHierarchyState

---

::: phydrax.discretization.BlockTopologyCompiler

---

::: phydrax.discretization.BlockTopologyCompileResult

---

::: phydrax.discretization.BlockTopologyCompileStatus

---

::: phydrax.discretization.BlockTopologyCompileEvidence

---

::: phydrax.discretization.BlockTopologyRouteGraph

---

::: phydrax.discretization.BlockFieldTopologyTransition

---

::: phydrax.discretization.BlockFieldTopologyTransitionResult

## Cell-centered preparation and FillPatch

`AMRAxisEntity` is the public literal axis-entity kind `"point"` or `"interval"` used by degree-aware transfer plans.


::: phydrax.discretization.AMREntityTransferPlan

---

::: phydrax.discretization.AMREntityTransferReport

---

::: phydrax.discretization.FDAMRHierarchyPlan

---

::: phydrax.discretization.PreparedFDAMRHierarchy

---

::: phydrax.discretization.FillPatchSource

---

::: phydrax.discretization.FDAMRPhysicalBoundaryRequest

---

::: phydrax.discretization.FDAMRFillPatchPlan

---

::: phydrax.discretization.FDAMRFillPatchWorkspace

---

::: phydrax.discretization.FDAMRFillPatchResult

## Finite volume and N-level time execution

::: phydrax.discretization.BlockAMRFiniteVolumePlan

---

::: phydrax.discretization.PreparedBlockAMRFiniteVolumeDynamics

---

::: phydrax.discretization.BlockAMRFiniteVolumeStageResult

---

::: phydrax.discretization.BlockAMRConservationPlan

---

::: phydrax.discretization.FluxRegister

---

::: phydrax.solver.AMRTimeSchedulePlan

---

::: phydrax.solver.BlockAMRRuntimePlan

---

::: phydrax.solver.PreparedBlockAMRRuntime

---

::: phydrax.solver.BlockAMRRuntimeState

---

::: phydrax.solver.BlockAMRAdvanceResult

---

::: phydrax.solver.BlockAMRAdvancePhase

---

::: phydrax.solver.multirate_amr_schedule_plan

## Composite elliptic operators

::: phydrax.discretization.CompositeAMRCellLayout

---

::: phydrax.discretization.CompositeAMRDiffusionPlan

---

::: phydrax.discretization.PreparedCompositeAMRDiffusion

---

::: phydrax.discretization.composite_amr_multigrid_builder

## Distributed ownership

::: phydrax.discretization.BlockAMRPartitionPlan

---

::: phydrax.discretization.PreparedDistributedBlockAMRHierarchy

---

::: phydrax.discretization.BlockAMRStableIDMigrationPlan

---

::: phydrax.discretization.DistributedBlockAMRResourceEvidence

## Persistence and output

`FiniteVolumeCheckpointPlan` and `FiniteVolumeOutputPlan` remain the fixed-block
runtime owners. The multivalued path uses `MultivaluedBlockAMRCheckpointPlan`
because its archive must reconstruct body-tagged component topology, while
retaining the same pickle-free array-archive substrate. Partition identity is
deliberately excluded from the portable multivalued checkpoint.

::: phydrax.solver.FiniteVolumeCheckpointPlan

---

::: phydrax.solver.FiniteVolumeCheckpoint

---

::: phydrax.solver.write_finite_volume_checkpoint

---

::: phydrax.solver.read_finite_volume_checkpoint

---

::: phydrax.solver.FiniteVolumeOutputPlan

## Variable patches and bounded entities

::: phydrax.discretization.LogicalPatchBox

---

::: phydrax.discretization.PatchShapeSignature

---

::: phydrax.discretization.PatchBucketPlan

---

::: phydrax.discretization.VariablePatchHierarchyPlan

---

::: phydrax.discretization.VariablePatchTopologyCompiler

---

::: phydrax.discretization.VariablePatchHierarchyTopology

---

::: phydrax.discretization.VariablePatchFieldState

---

::: phydrax.discretization.VariablePatchFillPatchPlan

---

::: phydrax.discretization.BlockHierarchyCapacityPlan

---

::: phydrax.discretization.VariablePatchEntityComplexPlan

---

::: phydrax.discretization.VariablePatchEntityExecutionPlan

---

::: phydrax.discretization.CompatibleEntityTransferFamily

## Mapped, ALE, and embedded geometry

::: phydrax.discretization.VariablePatchGeometryPlan

---

::: phydrax.discretization.VariablePatchALEPlan

---

Metric revisions use `CochainDiscretization.with_metric(hodges, numeric_revision=...)`.
Hodges remain metric-only; fixed-pattern numerical refresh preserves binding
identity for compiled execution. Compatible transfers expose `ComplexMap` and
commutation evidence rather than a second cochain metric-state carrier.

::: phydrax.discretization.CochainDiscretization

---

::: phydrax.discretization.VariablePatchEmbeddedBoundaryPlan

---

::: phydrax.solver.MovingEmbeddedBoundaryEventPlan

## Variable-patch placement and persistence

::: phydrax.discretization.VariablePatchPartitionPlan

---

::: phydrax.solver.VariablePatchCheckpointPlan

---

::: phydrax.solver.write_variable_patch_checkpoint

---

::: phydrax.solver.read_variable_patch_checkpoint

## Canonical production hierarchy and resources

::: phydrax.discretization.CanonicalPatchHierarchy

---

::: phydrax.discretization.canonicalize_patch_hierarchy

---

::: phydrax.discretization.BlockAMRResourcePlan

---

::: phydrax.discretization.BlockAMRResourceEvidence

---

::: phydrax.discretization.PatchClusteringPolicy

---

::: phydrax.discretization.BlockAMRReferenceParityPlan

## High-order mapped geometry and mortars

::: phydrax.discretization.PatchCoordinateMapSet

---

::: phydrax.discretization.CanonicalMappedGeometryPlan

---

::: phydrax.discretization.CanonicalMappedGeometryState

---

::: phydrax.discretization.MappedMortarPlan

---

::: phydrax.discretization.MappedMortarGeometry

---

::: phydrax.discretization.MappedMortarFluxPlan

## Two- and three-dimensional multivalued cut complexes

::: phydrax.discretization.EmbeddedLevelSetBody

---

::: phydrax.discretization.EmbeddedLevelSetBodySet

---

::: phydrax.discretization.CertifiedImplicitBody

---

::: phydrax.discretization.AdaptiveImplicitSamplingPlan

---

::: phydrax.discretization.PreparedAdaptiveImplicitSampling

---

::: phydrax.discretization.MultivaluedCutCellPlan

---

::: phydrax.discretization.MultivaluedCutCellComplex

---

::: phydrax.discretization.MultivaluedCutCellEvidence

---

::: phydrax.discretization.MultivaluedCutCell2DPlan

---

::: phydrax.discretization.MultivaluedCutCell2DComplex

---

::: phydrax.discretization.MultivaluedCutCellTransition

---

::: phydrax.discretization.MultivaluedCutCellDiffusionPlan

---

::: phydrax.discretization.ConservativeSmallCellRedistributionPlan.from_multivalued_cut_complex

## Compatible cut-cell entities and constrained transport

::: phydrax.discretization.CutCellCochainPlan

---

`CutCellCochainPlan.prepare()` returns a `CochainDiscretization` with retained boundary masks and an explicit numerical revision.

---

::: phydrax.discretization.CutCellCochainTransferPlan

---

::: phydrax.solver.advanced.CutCellCochainSynchronizationPlan

## Dynamic executable signatures

::: phydrax.discretization.PatchSignaturePolicy

---

::: phydrax.discretization.PatchExecutableSignature

---

::: phydrax.discretization.PatchExecutableCachePlan

---

::: phydrax.discretization.PatchExecutableCacheState

## Distributed multivalued execution

::: phydrax.discretization.DistributedCutCellPartitionPlan

---

::: phydrax.discretization.PreparedDistributedCutCellComplex

---

::: phydrax.discretization.PreparedDistributedCutCellComplex.pack_process_local

---

::: phydrax.discretization.DistributedCutCellState

## Differentiation

::: phydrax.discretization.BlockAMRDerivativePolicy

---

::: phydrax.discretization.FrozenCutCellTransitionDerivativePlan

---

::: phydrax.discretization.EventAwareCutCellDerivativePlan

---

::: phydrax.discretization.RelaxedHierarchyBlendPlan

---

::: phydrax.discretization.MappedGeometryDerivativePlan

## Moving topology, restart, and output

::: phydrax.solver.MovingMultivaluedCutCellPlan

---

::: phydrax.solver.MovingCutCellState

---

::: phydrax.solver.MovingTopologyLocalizationPlan

---

::: phydrax.solver.LocalizedMovingCutCellResult

---

::: phydrax.solver.MultivaluedBlockAMRCheckpointPlan

---

::: phydrax.solver.CutCellRestartRegistry

---

::: phydrax.solver.write_multivalued_block_amr_checkpoint

---

::: phydrax.solver.read_multivalued_block_amr_checkpoint

---

::: phydrax.solver.write_multivalued_cut_cell_output

## Forest AMR

Quadtree/octree forests refine the cells of one uniform root grid (a brick of
optionally periodic and optionally mapped roots). Leaves are the canonical sorted
Morton prefix of fixed-capacity worksets with stable tree-path IDs, 2:1 balance
over a face, edge, or corner stencil, per-level worksets, and power-of-two capacity
buckets, so compiled kernels are reused across adaptation cycles inside one
bucket. `ForestBlockLowering` maps a forest exactly onto a block hierarchy, which
the mapped-geometry and multivalued cut-complex owners consume unchanged.

::: phydrax.discretization.AMRBalanceStencil

---

::: phydrax.discretization.ForestPlan

---

::: phydrax.discretization.ForestHierarchyTopology

---

::: phydrax.discretization.ForestLeafWorkset

---

::: phydrax.discretization.ForestWorksetSignature

---

::: phydrax.discretization.ForestFaceKind

---

::: phydrax.discretization.ForestTopologyCompiler

---

::: phydrax.discretization.ForestAdaptResult

---

::: phydrax.discretization.ForestAdaptStatus

---

::: phydrax.discretization.ForestAdaptEvidence

---

::: phydrax.discretization.forest_common_refinement

---

::: phydrax.discretization.ForestBlockLowering

---

::: phydrax.discretization.ForestLeafGeometry

---

::: phydrax.discretization.forest_leaf_geometry

---

::: phydrax.discretization.ForestCutComplex

---

::: phydrax.discretization.prepare_forest_cut_complex

### Forest transfer and reflux

::: phydrax.discretization.ForestFieldTransition

---

::: phydrax.discretization.ForestTransferRoutes

---

::: phydrax.discretization.ForestTransferResult

---

::: phydrax.discretization.ForestRefluxRoutes

---

::: phydrax.discretization.ForestVertexLayout

---

::: phydrax.discretization.forest_vertex_interpolation

---

::: phydrax.discretization.ForestCochainComplex

---

::: phydrax.discretization.ForestCochainTransfer

### Forest partitions and migration

::: phydrax.discretization.ForestPartitionPlan

---

::: phydrax.discretization.PreparedForestPartition

---

::: phydrax.discretization.ForestPartitionEvidence

---

::: phydrax.discretization.ForestMigrationPlan
