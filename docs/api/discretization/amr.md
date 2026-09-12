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

## Cell-centred preparation and FillPatch

::: phydrax.discretization.AMRAxisEntity

---

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

`FiniteVolumeCheckpointPlan` and `FiniteVolumeOutputPlan` accept a
`PreparedBlockAMRRuntime`; an optional `PreparedDistributedBlockAMRHierarchy`
binds the exact partition identity. They remain the ordinary finite-volume
persistence owners rather than creating a second AMR store or schema family.

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

::: phydrax.discretization.CochainMetricPlan

---

::: phydrax.discretization.CochainMetricState

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
