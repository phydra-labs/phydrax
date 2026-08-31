# Probabilistic graphical models

`phydrax.pgm` provides immutable finite-discrete factor graphs, sparse and open factor
kernels, bounded exact inference, scheduled/accelerated belief propagation, advanced
Gibbs-family samplers, MAP bounds, checkpointing, and composable training objectives.

## Variables, factors, and graph structure

::: phydrax.pgm.DiscreteVariableGroup

---

::: phydrax.pgm.VariableSelection

---

::: phydrax.pgm.DiscreteFactorGraph

---

::: phydrax.pgm.DenseTableFactorGroup

---

::: phydrax.pgm.EnumeratedFactorGroup

---

::: phydrax.pgm.IsingFactorGroup

---

::: phydrax.pgm.PottsFactorGroup

---

::: phydrax.pgm.LogicalFactorGroup

---

::: phydrax.pgm.BinaryCardinalityFactorGroup

---

::: phydrax.pgm.KernelFactorGroup

---

::: phydrax.pgm.AbstractDiscreteFactorKernel

---

::: phydrax.pgm.CallableFactorKernel

---

::: phydrax.pgm.FactorKernelCapabilities

---

::: phydrax.pgm.FactorGraphPrecisionPolicy

---

::: phydrax.pgm.FactorGraphResourcePolicy

---

::: phydrax.pgm.FactorExecutionEvidence

::: phydrax.pgm.factor_graph_log_score

---

::: phydrax.pgm.pack_assignments

---

::: phydrax.pgm.pack_evidence

## Exact finite-state inference

::: phydrax.pgm.enumerate_factor_graph

---

::: phydrax.pgm.ExactFactorGraphResult

---

::: phydrax.pgm.ExactFactorGraphStatus

---

::: phydrax.pgm.VariableEliminationMethod

---

::: phydrax.pgm.plan_variable_elimination

---

::: phydrax.pgm.variable_elimination

---

::: phydrax.pgm.VariableEliminationResult

---

::: phydrax.pgm.plan_junction_tree

---

::: phydrax.pgm.junction_tree_calibrate

---

::: phydrax.pgm.NormalizedFactorGraphLaw

---

::: phydrax.pgm.SmoothDualLP

---

::: phydrax.pgm.solve_smooth_dual_lp

---

::: phydrax.pgm.perturb_and_map_log_normalizer

## Belief propagation

::: phydrax.pgm.SumProductBeliefPropagation

---

::: phydrax.pgm.MaxProductBeliefPropagation

---

::: phydrax.pgm.PreparedBeliefPropagation

---

::: phydrax.pgm.BeliefPropagationState

---

::: phydrax.pgm.prepare_belief_propagation

---

::: phydrax.pgm.refresh_belief_propagation

---

::: phydrax.pgm.initialize_belief_propagation

---

::: phydrax.pgm.run_belief_propagation

---

::: phydrax.pgm.SumProductBeliefPropagationResult

---

::: phydrax.pgm.MaxProductBeliefPropagationResult

---

::: phydrax.pgm.BeliefPropagationStatus

---

::: phydrax.pgm.BeliefPropagationSchedulePolicy

---

::: phydrax.pgm.run_accelerated_belief_propagation

---

::: phydrax.pgm.run_implicit_belief_propagation

## Factor-graph batching

::: phydrax.pgm.BatchedBeliefPropagationState

---

::: phydrax.pgm.batch_belief_propagation

---

::: phydrax.pgm.PackedFactorGraphBatch

---

::: phydrax.pgm.pack_factor_graphs

---

::: phydrax.pgm.FactorGraphShardingPolicy

## Chromatic Gibbs sampling

::: phydrax.pgm.ChromaticGibbs

---

::: phydrax.pgm.GibbsSchedule

---

::: phydrax.pgm.PreparedChromaticGibbs

---

::: phydrax.pgm.GibbsState

---

::: phydrax.pgm.prepare_chromatic_gibbs

---

::: phydrax.pgm.refresh_chromatic_gibbs

---

::: phydrax.pgm.initialize_gibbs

---

::: phydrax.pgm.gibbs_sweep

---

::: phydrax.pgm.sample_gibbs

---

::: phydrax.pgm.GibbsSampleResult

---

::: phydrax.pgm.GibbsDiagnostics

---

::: phydrax.pgm.GibbsScanPolicy

---

::: phydrax.pgm.gibbs_sweep_with_policy

---

::: phydrax.pgm.JointDiscreteBlock

---

::: phydrax.pgm.joint_block_sweep

---

::: phydrax.pgm.ParallelTempering

---

::: phydrax.pgm.parallel_tempering_step

---

::: phydrax.pgm.wolff_cluster_step

---

::: phydrax.pgm.reduce_gibbs_chain

## Structured constructors and training

::: phydrax.pgm.ising_factor_graph

---

::: phydrax.pgm.potts_factor_graph

---

::: phydrax.pgm.exact_factor_graph_negative_log_likelihood

---

::: phydrax.pgm.contrastive_divergence_loss

---

::: phydrax.pgm.factor_graph_moments

---

::: phydrax.pgm.FactorGraphTrainingDiagnostics

---

::: phydrax.pgm.pseudolikelihood_loss

---

::: phydrax.pgm.bethe_negative_log_likelihood

---

::: phydrax.pgm.initialize_persistent_training

---

::: phydrax.pgm.persistent_contrastive_divergence_step

---

::: phydrax.pgm.stochastic_maximum_likelihood_step

---

::: phydrax.pgm.expectation_maximization_step

---

::: phydrax.pgm.write_factor_graph_checkpoint

---

::: phydrax.pgm.read_factor_graph_checkpoint
