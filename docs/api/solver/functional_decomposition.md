# Functional domain decomposition

Public symbols are also re-exported from `phydrax.solver`.

Block, Schwarz, and asynchronous execution accept an `IterationSession`.
Records are emitted only after a complete sweep/update commits its local
parameters and trace exchange. Host control stops before the next sweep and
returns the exact accepted `FunctionalDecompositionState`; joint execution
supports terminal sinks but rejects host stopping because it has no host-safe
internal boundary.


::: phydrax.solver.functional_decomposition.FunctionalDecompositionProblem
    options:
        members:
            - __init__
            - partition_of_unity
            - broken
            - training_terms
            - diagnostic_terms

---

::: phydrax.solver.functional_decomposition.ScopedFunctionalTerm

---

::: phydrax.solver.functional_decomposition.PatchScope

---

::: phydrax.solver.functional_decomposition.PairScope

---

::: phydrax.solver.functional_decomposition.GlobalScope

---

::: phydrax.solver.functional_decomposition.JointDecompositionTraining

---

::: phydrax.solver.functional_decomposition.BlockDecompositionTraining

---

::: phydrax.solver.functional_decomposition.SchwarzDecompositionTraining

---

::: phydrax.solver.functional_decomposition.FunctionalDecompositionPlan

---

::: phydrax.solver.functional_decomposition.PreparedFunctionalDecomposition

---

::: phydrax.solver.functional_decomposition.prepare_functional_decomposition

---

::: phydrax.solver.functional_decomposition.solve_functional_decomposition

---

::: phydrax.solver.functional_decomposition.FunctionalDecompositionState

---

::: phydrax.solver.functional_decomposition.FunctionalDecompositionEvidence

---

::: phydrax.solver.functional_decomposition.FunctionalDecompositionResult

---

::: phydrax.solver.functional_decomposition.FunctionalDecompositionIterationMetrics

---

::: phydrax.solver.functional_decomposition.save_functional_decomposition_checkpoint

---

::: phydrax.solver.functional_decomposition.load_functional_decomposition_checkpoint

---

::: phydrax.solver.functional_decomposition.FunctionalCoarseCorrection

---

::: phydrax.solver.functional_decomposition.TraceExchangeState

---

::: phydrax.solver.functional_decomposition.SchwarzTraceState

---

::: phydrax.solver.functional_decomposition.capture_schwarz_trace_state

---

::: phydrax.terms.MortarInterfacePenalty

---

::: phydrax.terms.NitscheInterfaceFunctional

---

::: phydrax.terms.AugmentedValueConstraint

---

::: phydrax.terms.LocalTestSpace

---

::: phydrax.terms.LocalizedResidualNorm

---

::: phydrax.solver.functional_decomposition.AugmentedInterfacePlan

---

::: phydrax.solver.functional_decomposition.AugmentedInterfaceResult

---

::: phydrax.solver.functional_decomposition.solve_augmented_interface

---

::: phydrax.solver.functional_decomposition.LocalCurvaturePlan

---

::: phydrax.solver.functional_decomposition.LocalCurvatureResult

---

::: phydrax.solver.functional_decomposition.dense_local_curvature_step

---

::: phydrax.solver.functional_decomposition.FunctionalHierarchyPlan

---

::: phydrax.solver.functional_decomposition.FunctionalHierarchyResult

---

::: phydrax.solver.functional_decomposition.train_functional_hierarchy

---

::: phydrax.solver.functional_decomposition.AdaptiveRefinementPlan

---

::: phydrax.solver.functional_decomposition.AdaptiveTopologyEvidence

---

::: phydrax.solver.functional_decomposition.AdaptiveTopologyTransaction

---

::: phydrax.solver.functional_decomposition.prepare_adaptive_topology_transaction

---

::: phydrax.solver.functional_decomposition.refine_axis_partition

---

::: phydrax.solver.functional_decomposition.FunctionalDecompositionShardingPlan

---

::: phydrax.solver.functional_decomposition.DecompositionShardingEvidence

---

::: phydrax.solver.functional_decomposition.place_local_field_family

---

::: phydrax.solver.functional_decomposition.place_schwarz_trace_state

---

::: phydrax.solver.functional_decomposition.FunctionalPatchParticipant

---

::: phydrax.solver.functional_decomposition.FixedPatchParticipant

---

::: phydrax.solver.functional_decomposition.HybridFunctionalDecomposition

---

::: phydrax.solver.functional_decomposition.DecompositionDeploymentArtifact

---

::: phydrax.solver.functional_decomposition.save_decomposition_artifact

---

::: phydrax.solver.functional_decomposition.load_decomposition_artifact

---

::: phydrax.solver.functional_decomposition.SchwarzTraceQuantity

---

`SchwarzDecompositionTraining(relaxation=phydrax.nonlinear.AitkenRelaxation())`
uses bounded Irons–Tuck relaxation of a complete raw sweep. Intermediate
Gauss–Seidel and colored captures use unit relaxation; the final targets alone
receive the scalar update. Exchanged traces must share one nonempty `quantity_id`
and have commensurate or nondimensionalized coordinates. Physical interface
defect, rather than coefficient convergence, remains the convergence criterion.

`FunctionalDecompositionState.previous_increment` and `.omega` preserve Aitken
history across chunked execution and checkpoints. Both are `None` on the fixed
float path, whose numerical behavior is unchanged. Checkpoint loading requires
a matching `state_like` template, including optional-field presence, shapes,
dtypes, completed sweeps, and patch kernels. The canonical reconstruction field
set changed; old checkpoints have no compatibility shim.

`AitkenTracePlan`, `AitkenTraceState`, and `aitken_relax_trace_state` are removed.
Migrate helper callers to the shared nonlinear policy through the training
strategy. Asynchronous Schwarz does not consume strategy relaxation.

---

::: phydrax.solver.functional_decomposition.AsynchronousSchwarzPlan

---

::: phydrax.solver.functional_decomposition.solve_asynchronous_schwarz

---

::: phydrax.solver.functional_decomposition.DecompositionKFACResult

---

::: phydrax.solver.functional_decomposition.solve_local_kfac

---

::: phydrax.solver.functional_decomposition.solve_overlap_kfac

---

::: phydrax.solver.functional_decomposition.MatrixFreeGaussNewtonPlan

---

::: phydrax.solver.functional_decomposition.matrix_free_gauss_newton_step

---

::: phydrax.solver.functional_decomposition.FunctionalCyclePlan

---

::: phydrax.solver.functional_decomposition.train_functional_cycles

---

::: phydrax.solver.functional_decomposition.TrainableAxisPartition

---

::: phydrax.solver.functional_decomposition.coarsen_axis_partition

---

::: phydrax.solver.functional_decomposition.DistributedCollectiveEvidence

---

::: phydrax.solver.functional_decomposition.distributed_pou_collective

---

::: phydrax.solver.functional_decomposition.distributed_schwarz_exchange
