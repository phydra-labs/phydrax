# Probabilistic graphical models

`phydrax.pgm` provides immutable finite-discrete factor graphs, exact enumeration,
belief propagation, chromatic Gibbs sampling, structured Ising/Potts/logical factors,
and composable training objectives.

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
