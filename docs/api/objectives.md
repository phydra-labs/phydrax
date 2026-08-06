# Objectives

Raw scalar objective terms for `FunctionalSolver`. Unlike constraints, these terms do
not impose squared-residual semantics or nonnegativity.

`IntegralFunctional` uses the same target, plan, and realization contract as
[`phydrax.integration`](integration.md).

## Sampled objective batches

`AbstractSamplingObjectiveTerm` separates expensive random-label construction from
the differentiated scalar loss. `FunctionalSolver` calls each sampled objective's
`sample(key=...)` exactly once per optimizer update, before the compiled
value-and-gradient call, and threads that immutable PyTree batch through every
same-update evaluation. Ordinary objectives receive no batch.

Use a fixed batch for common-random-number optimization and deterministic replay.
Use `sampling_mode="resample"` only when target refresh is part of the estimator.
Different sampled objectives receive deterministic distinct subkeys. The sampling
policy must remain static during a compiled training run; changing probe count, path
count, or chunk shape requires a separate compilation.

## Feynman--Kac regression

`FeynmanKacRegressionObjective` fits value and optional control fields to frozen
global-in-time labels generated from a `BSDEProblem`. Targets and their reported
Monte Carlo uncertainty are stopped before differentiation. Value and control weights
are explicit and zero control weight does not require a control model or labels.

::: phydrax.objectives.FeynmanKacRegressionObjective

---

::: phydrax.objectives.FeynmanKacRegressionDiagnostics

## Randomized residual objectives

`RandomizedResidualObjective` consumes raw residual realizations rather than a
pre-averaged stochastic operator estimate. This distinction is required for
estimator-aware squaring:

- `loss_mode="u_statistic"` estimates a squared mean from distinct probe pairs and
  is unbiased, but an individual batch may be negative;
- `loss_mode="independent_product"` multiplies two independently generated residual
  ensembles;
- `loss_mode="plug_in"` squares the sample mean and is nonnegative but biased upward
  by estimator variance.

Signed unbiased objectives are incompatible with naive `keep_best=True` model
selection: a more negative noisy batch is not a better nonnegative mathematical loss.
Use fixed probes or independent validation for selection, and inspect
`RandomizedResidualDiagnostics` rather than treating the training scalar as a
certificate of PDE error.

::: phydrax.objectives.RandomizedResidualObjective

---

::: phydrax.objectives.RandomizedResidualBatch

---

::: phydrax.objectives.RandomizedResidualDiagnostics

## Particle score matching

`ScoreMatchingObjective` learns a score field
\(s_\theta(t,x)\approx\nabla_x\log p_t(x)\) from state-time particles without
normalizing or reconstructing a density. `method="exact"` computes the score
divergence exactly, `method="implicit"` estimates it with JVP probes, and
`method="sliced"` uses projected score matching. The score output must have exactly
the state shape.

Trajectory masks, per-node weights, path identities, and time coverage are retained.
Reported path uncertainty reduces over independent path clusters, not flattened
state-time nodes. A score field is the delivered quantity; normalized-density
reconstruction is a separate problem.

::: phydrax.objectives.ScoreMatchingObjective

---

::: phydrax.objectives.ScoreMatchingPolicy

---

::: phydrax.objectives.ScoreMatchingDiagnostics

## Supporting contracts

`BatchSampler` and `ResidualEvaluator` are the callable protocols used by randomized
residual objectives. `RandomizedResidualLossMode` and
`RandomizedResidualSamplingMode` are the corresponding literal policy types.
`LabelProvider` is the Feynman--Kac label callback contract.
`ScoreSampleProvider`, `ScoreMatchingMethod`, and `ScoreMatchingSamplingMode` provide
the equivalent score-matching contracts.

::: phydrax.objectives.RandomizedResidualSamples

---

::: phydrax.objectives.ScoreMatchingBatch

## Base objective types

::: phydrax.objectives.AbstractObjectiveTerm

::: phydrax.objectives.AbstractSamplingObjectiveTerm

::: phydrax.objectives.IntegralFunctional
