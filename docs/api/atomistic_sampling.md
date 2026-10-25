# Atomistic sampling and free energy

## Collective variables

CVs evaluate one `CollectiveVariableGeometry` (wrapped and material
coordinates, current cell, bound neighborhood views) and report `successful`,
`regular` and an explicit `CollectiveVariableStatus`. Point-pair vectors use the
cell's certified minimum image; groups use the accepted unwrapped branch, a
bond-tree reconstruction or an extent-certified anchor.

::: phydrax.atomistic.sampling.CollectiveVariableGeometry

::: phydrax.atomistic.sampling.CollectiveVariableImageBranch

::: phydrax.atomistic.sampling.CollectiveVariableStatus

::: phydrax.atomistic.sampling.CollectiveVariableKind

::: phydrax.atomistic.sampling.AbstractCollectiveVariablePlan

::: phydrax.atomistic.sampling.AbstractCollectiveVariableProgram

::: phydrax.atomistic.sampling.CollectiveVariableMetric

::: phydrax.atomistic.sampling.CollectiveVariablePlan

::: phydrax.atomistic.sampling.PreparedCollectiveVariable

::: phydrax.atomistic.sampling.CollectiveVariableProgram

::: phydrax.atomistic.sampling.ModelCollectiveVariableProgram

::: phydrax.atomistic.sampling.CollectiveVariableEvaluation

::: phydrax.atomistic.sampling.CollectiveVariableProgramEvaluation

::: phydrax.atomistic.sampling.PreparedGroupBranch

::: phydrax.atomistic.sampling.aligned_deviation

::: phydrax.atomistic.sampling.AlignmentEvaluation

## Biases as program terms

Biases are conservative terms of the scalar program (`AtomisticBiasTerm`), so
forces, constraint projection, stress, Monte Carlo cell trials and
reduced-potential cross-evaluation all see one definition. Controls and
histories are read-only term states bound by exact `term_id`. They change only
at committed boundaries of `PreparedBiasedDynamics` or multistate iterations.
The row `coupling` scales a term's energy, and `0` is the unbiased row of the
same term. Cell-dependent biases contribute stress through the shared strain map: a
volume umbrella `k (V - V0)^2 / 2` contributes the tension-positive strain stress
`sym(dE/dstrain) / V = k (V - V0) I`, a term stress that is not a pressure.

::: phydrax.atomistic.sampling.AtomisticBiasTerm

::: phydrax.atomistic.sampling.PreparedAtomisticBiasTerm

::: phydrax.atomistic.sampling.AtomisticBiasTermState

::: phydrax.atomistic.sampling.AtomisticBiasEvidence

::: phydrax.atomistic.sampling.with_bias_controls

::: phydrax.atomistic.sampling.with_bias_history

::: phydrax.atomistic.sampling.AbstractBiasForm

::: phydrax.atomistic.sampling.HarmonicBiasForm

::: phydrax.atomistic.sampling.FlatBottomBiasForm

::: phydrax.atomistic.sampling.WallBiasForm

::: phydrax.atomistic.sampling.WallSide

::: phydrax.atomistic.sampling.HillHistoryBiasForm

::: phydrax.atomistic.sampling.IntegratedProfileBiasForm

::: phydrax.atomistic.sampling.LearnedFreeEnergyBiasForm

## Adaptive histories, ABF and biased dynamics

Deposits happen only at committed boundaries and are all-or-nothing against the
pre-event history. Terminal storage refuses the event that would overflow. In
`PreparedBiasedDynamics` that refusal is latched (`BiasedStepReason.HISTORY_EXHAUSTED`),
and nothing is committed until `grow_bias_history_capacity` rebinds a larger buffer.
FIFO storage disqualifies the usual profile formula. `FrozenBiasHistory.declaration`
(`"frozen:<digest>"`) is the only row declaration of fixed content, and the content is
always authenticated, never the label. `warm_start_bias_history` declares a new run's
initial condition; exact resume goes through `write_biased_dynamics_checkpoint` and
`read_biased_dynamics_checkpoint`.

`PreparedABF.sample` returns an `ABFSample` whose `dual_solve` (and, for a constrained
closure, `fixman_solve`) is an `ABFLinearSolveEvidence` with the native solve `status`,
`derivative_valid`, `diagnostics` and `provenance`. An irregular sample has a NaN
`mean_force` and is never recorded. Force-only ABF is a typed nonconservative force port
with no energy, and such runtimes refuse energy-based moves. A frozen, integrated ABF
profile is an ordinary conservative term. `abf_profile` and `metadynamics_profile`
return diagnostic `AdaptiveProfileResult` values, never equilibrium free energies.

::: phydrax.atomistic.sampling.AtomisticBiasHistoryPlan

::: phydrax.atomistic.sampling.AtomisticBiasHistoryState

::: phydrax.atomistic.sampling.BiasHistoryOwnership

::: phydrax.atomistic.sampling.BiasHistoryStorage

::: phydrax.atomistic.sampling.HillKernel

::: phydrax.atomistic.sampling.deposit_bias_history

::: phydrax.atomistic.sampling.AtomisticBiasDeposition

::: phydrax.atomistic.sampling.grow_bias_history

::: phydrax.atomistic.sampling.FrozenBiasHistory

::: phydrax.atomistic.sampling.warm_start_bias_history

::: phydrax.atomistic.sampling.AtomisticABFPlan

::: phydrax.atomistic.sampling.ABFBaseMeasure

::: phydrax.atomistic.sampling.PreparedABF

::: phydrax.atomistic.sampling.AtomisticABFState

::: phydrax.atomistic.sampling.ABFSample

::: phydrax.atomistic.sampling.ABFLinearSolveEvidence

::: phydrax.atomistic.sampling.AtomisticABFForcePort

::: phydrax.atomistic.sampling.abf_profile

::: phydrax.atomistic.sampling.metadynamics_profile

::: phydrax.atomistic.sampling.AdaptiveProfileResult

::: phydrax.atomistic.sampling.AdaptiveProfileEstimator

::: phydrax.atomistic.sampling.PreparedBiasedDynamics

::: phydrax.atomistic.sampling.BiasedDynamicsState

::: phydrax.atomistic.sampling.BiasedStepEvaluation

::: phydrax.atomistic.sampling.BiasedStepReason

::: phydrax.atomistic.sampling.grow_bias_history_capacity

::: phydrax.atomistic.sampling.BiasedDynamicsCheckpoint

::: phydrax.atomistic.sampling.write_biased_dynamics_checkpoint

::: phydrax.atomistic.sampling.read_biased_dynamics_checkpoint

::: phydrax.atomistic.sampling.BiasFormEvaluation

::: phydrax.atomistic.sampling.AtomisticBiasHistoryEvaluation

## Restrained mean force

::: phydrax.atomistic.sampling.RestrainedMeanForcePlan

::: phydrax.atomistic.sampling.estimate_restrained_free_energy_gradient

::: phydrax.atomistic.sampling.MeanForceData

::: phydrax.atomistic.sampling.LearnedFreeEnergyCommittee

`fit_free_energy_model` runs each update as one attempt of the shared training
kernel with `MODEL` root authority. A training evaluation that is not `valid` (no
active window or nonfinite predictions) carries no support and is rejected, and a
nonfinite one rolls back. Neither is committed: training stops with a `FAILURE`
event, `valid=False`, and the last accepted model.

::: phydrax.atomistic.sampling.fit_free_energy_model

::: phydrax.atomistic.sampling.FreeEnergyTrainingPolicy

::: phydrax.atomistic.sampling.FreeEnergyFitResult


## Thermodynamic states and multistate execution

Every history term is bound by the table rows with one common declaration:
`"adaptive"` (deposits at committed iterations; its draws are never
`sample_active`) or `"frozen:<digest>"`. Segment results carry per-draw
`bias_adapting`, `bias_deposited`, `bias_exhausted`, `bias_owner_keys` and
`bias_history_revision` evidence. `bias_history_digests()` authenticates the
bound content at the segment end, and the checkpoint manifest and UQ conversion
repeat that check.

::: phydrax.atomistic.AtomisticPhaseSpaceMeasurePlan

::: phydrax.atomistic.AtomisticThermodynamicStatePlan

::: phydrax.atomistic.sampling.AtomisticBiasRowBinding

::: phydrax.atomistic.PreparedThermodynamicStateTable

::: phydrax.atomistic.sampling.AtomisticCanonicalSamplingQualification


::: phydrax.atomistic.sampling.AtomisticReplicaExchangePlan

::: phydrax.atomistic.sampling.AtomisticSAMSPlan

::: phydrax.atomistic.sampling.AtomisticSAMSState

::: phydrax.atomistic.sampling.AtomisticMultistatePlan

::: phydrax.atomistic.sampling.BiasHistoryLabel

::: phydrax.atomistic.sampling.PreparedAtomisticMultistate

::: phydrax.atomistic.sampling.AtomisticMultistateState

::: phydrax.atomistic.sampling.AtomisticMultistateIteration

::: phydrax.atomistic.sampling.AtomisticReducedPotentialEvaluation

::: phydrax.atomistic.sampling.AtomisticMultistateSegmentPlan

::: phydrax.atomistic.sampling.AtomisticMultistateSegmentResult

::: phydrax.atomistic.sampling.AtomisticMultistateEquilibriumSamples

::: phydrax.atomistic.sampling.AtomisticDrawConfigurationCapture

::: phydrax.atomistic.sampling.AtomisticMultistateCheckpointPlan

::: phydrax.atomistic.sampling.AtomisticMultistateCheckpoint

::: phydrax.atomistic.sampling.write_atomistic_multistate_checkpoint

::: phydrax.atomistic.sampling.read_atomistic_multistate_checkpoint

## Free-energy observations and estimators

::: phydrax.uq.ReducedPotentialDataset

::: phydrax.uq.ReducedWorkDataset

::: phydrax.uq.ThermodynamicDerivativeDataset

::: phydrax.uq.FreeEnergySelectionPlan

::: phydrax.uq.FreeEnergySelectionEvidence

::: phydrax.uq.FreeEnergyResult

::: phydrax.uq.reduced_potential_dataset_from_multistate

::: phydrax.uq.reduced_potential_dataset_from_alchemical_evaluation

::: phydrax.uq.reduced_work_dataset_from_alchemical_switching

::: phydrax.uq.free_energy_perturbation

::: phydrax.uq.thermodynamic_integration

::: phydrax.uq.bennett_acceptance_ratio

::: phydrax.uq.multistate_bennett_acceptance_ratio

## Free-energy networks

::: phydrax.uq.FreeEnergyEdgeObservation

::: phydrax.uq.FreeEnergyNetworkPlan

::: phydrax.uq.FreeEnergyNetworkResult

::: phydrax.uq.analyze_free_energy_network

::: phydrax.uq.SparseReducedPotentialDataset

::: phydrax.uq.SparsePairwiseFreeEnergyNetworkResult

::: phydrax.uq.sparse_pairwise_free_energy_network


## Targeted maps


::: phydrax.uq.TargetedMapPlan

::: phydrax.uq.TargetedFreeEnergyProblem

::: phydrax.uq.evaluate_targeted_work

::: phydrax.uq.fit_targeted_free_energy_map
