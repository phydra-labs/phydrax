# Scalar terms

Raw scalar terms for `FunctionalSolver`. Unlike penalties built from conditions and
integration sources, these terms do not require squared-residual semantics or
nonnegativity.

`IntegralFunctional` uses the same target, plan, and realization contract as
[`phydrax.integration`](integration.md).

## Sampled term batches

`AbstractSamplingTerm` separates expensive random-label construction from the
differentiated scalar loss. `FunctionalSolver` calls each sampled term's
`sample(key=...)` exactly once per optimizer update, before the compiled
value-and-gradient call, and threads that immutable PyTree batch through every
same-update evaluation. Ordinary terms receive no batch.

Use a fixed batch for common-random-number optimization and deterministic replay.
Use `sampling_mode="resample"` only when target refresh is part of the estimator.
Different sampled terms receive deterministic distinct subkeys. The sampling
policy must remain static during a compiled training run; changing probe count, path
count, or chunk shape requires a separate compilation.

## Feynman--Kac regression

`FeynmanKacRegressionTerm` fits value and optional control fields to frozen
global-in-time labels generated from a `BSDEProblem`. Targets and their reported
Monte Carlo uncertainty are stopped before differentiation. Value and control weights
are explicit and zero control weight does not require a control model or labels.

::: phydrax.terms.FeynmanKacRegressionTerm

---

::: phydrax.terms.FeynmanKacRegressionDiagnostics

## Randomized residual terms

`RandomizedResidualTerm` consumes raw residual realizations rather than a
pre-averaged stochastic operator estimate. This distinction is required for
estimator-aware squaring:

- `loss_mode="u_statistic"` estimates a squared mean from distinct probe pairs and
  is unbiased, but an individual batch may be negative;
- `loss_mode="independent_product"` multiplies two independently generated residual
  ensembles;
- `loss_mode="plug_in"` squares the sample mean and is nonnegative but biased upward
  by estimator variance.

Signed unbiased terms are incompatible with `keep_best=True`: a more negative
sampled value is not a better nonnegative mathematical loss. `FunctionalSolver`
rejects that combination. Train with `keep_best=False`, use fixed probes or an
independent deterministic realization for selection, and inspect
`RandomizedResidualDiagnostics` rather than treating the training scalar as a
certificate of PDE error.

::: phydrax.terms.RandomizedResidualTerm

---

::: phydrax.terms.RandomizedResidualBatch

---

::: phydrax.terms.RandomizedResidualDiagnostics

## Randomized moment penalties

`MomentPenalty` is reserved for deterministic per-step integration, fixed
realizations, and caller-supplied realizations. `RandomizedMomentPenalty`
materializes independent integration realizations and applies the same
U-statistic, independent-product, or explicit plug-in policy after integration.
This avoids silently adding parameter-dependent estimator variance to a squared
moment objective.

::: phydrax.terms.RandomizedMomentPenalty

---

::: phydrax.terms.RandomizedMomentBatch

---

::: phydrax.terms.RandomizedMomentDiagnostics


## Endpoint flow matching

`FlowMatchingTerm` learns a state-shaped velocity field from explicitly coupled
source/target endpoints. Its interpolant supplies a conditional state and velocity;
the term never infers a density, inverse, or stochastic-process realization.

Fixed endpoint samples retain common random numbers while resampling interpolation
times on every optimizer update. With `sampling_mode="resample"`, the endpoint
provider itself is called exactly once per update. Masks and log weights are
normalized over valid pairs only.

`EuclideanFlowMatchingMetric` uses the squared norm of the complete event. The
`normalize_event` option is explicit rather than silently changing loss scale.
Fixed-query fields should instead use
`phydrax.nn.operator.training.OperatorFlowMatchingMetric`, which applies query
quadrature, masks, and channel geometry.

::: phydrax.terms.FlowMatchingTerm

---

::: phydrax.terms.FlowMatchingPolicy

---

::: phydrax.terms.FlowMatchingDiagnostics

---

::: phydrax.terms.EuclideanFlowMatchingMetric

---

::: phydrax.terms.AbstractFlowMatchingMetric

## Particle score matching

`ScoreMatchingTerm` learns a score field
\(s_\theta(t,x)\approx\nabla_x\log p_t(x)\) from state-time particles without
normalizing or reconstructing a density. `method="exact"` computes the score
divergence exactly, `method="implicit"` estimates it with JVP probes, and
`method="sliced"` uses projected score matching. The score output must have exactly
the state shape.

Trajectory masks, per-node weights, path identities, and time coverage are retained.
Reported path uncertainty reduces over independent path clusters, not flattened
state-time nodes. A score field is the delivered quantity; normalized-density
reconstruction is a separate problem.

::: phydrax.terms.ScoreMatchingTerm

---

::: phydrax.terms.ScoreMatchingPolicy

---

::: phydrax.terms.ScoreMatchingDiagnostics

## Transport functionals

Transport terms compare complete physical measures or empirical events instead of
reducing pointwise discrepancies first. Balanced Sinkhorn terms retain native
convergence diagnostics and reject a failed solve. The unbalanced spatial term is
reserved for physical intensity or count measures with meaningful unequal mass and
also rejects transported-mass collapse. Prepared references reuse only the fixed
target self term. Sliced terms retain their projection design.

`SoftQuantileFunctional` is a regularized order objective, not an exact sample-quantile
penalty. Its diagnostics expose the effective solver epsilon and mark exact `q=0` and
`q=1` endpoints. Interior squared penalties inherit the finite Sinkhorn map's
regularity. Exact endpoints are only almost-everywhere differentiable, and absolute
discrepancy has a kink at zero residual. A supplied balanced transport solver overrides
the convenience `epsilon`; convenience diagnostics do not retain complete solver
evidence.

::: phydrax.terms.SpatialSinkhornDivergenceTerm

---

::: phydrax.terms.SpatialUnbalancedSinkhornDivergenceTerm

---

::: phydrax.terms.EmpiricalSinkhornDivergenceTerm

---

::: phydrax.terms.SlicedWassersteinTerm

---

::: phydrax.terms.SoftQuantileFunctional


## Supporting contracts

`BatchSampler` and `ResidualEvaluator` are the callable protocols used by randomized
residual terms. `RandomizedResidualLossMode` and
`RandomizedResidualSamplingMode` are the corresponding literal policy types.
`LabelProvider` is the Feynman--Kac label callback contract.
`ScoreSampleProvider`, `ScoreMatchingMethod`, and `ScoreMatchingSamplingMode` provide
the equivalent score-matching contracts.
`FlowEndpointProvider` and `FlowMatchingSamplingMode` provide the corresponding
endpoint-provider and refresh contracts for flow matching.

::: phydrax.terms.RandomizedResidualSamples

---

::: phydrax.terms.ScoreMatchingBatch

## Base term types

::: phydrax.terms.AbstractScalarTerm

::: phydrax.terms.AbstractSamplingTerm

::: phydrax.terms.IntegralFunctional

## Adaptive residual collocation

Adaptive collocation is a sampling policy for pointwise-capable terms, not a
condition type. Attach a policy through
`integration.adaptive(target, initial_plan, policy)` and pass that source to a
`ResidualPenalty`. The solver owns the resulting population lifecycle.

::: phydrax.sampling.collocation.AbstractCollocationPolicy

---

::: phydrax.sampling.collocation.CollocationPolicy

---

::: phydrax.sampling.collocation.CoresetCollocationPolicy

---

::: phydrax.sampling.collocation.SeparableCollocationPolicy

---

::: phydrax.sampling.collocation.HierarchicalAxisPolicy

---

::: phydrax.sampling.collocation.ControlledCollocationPolicy

---

::: phydrax.sampling.collocation.RefreshSchedule

---

::: phydrax.sampling.collocation.ResidualMonitor

---

::: phydrax.sampling.collocation.RefreshGuard

---

::: phydrax.sampling.collocation.AdaptationBudget

---

::: phydrax.sampling.collocation.CoverageAnchors

---

::: phydrax.sampling.collocation.collocation_policy_support
