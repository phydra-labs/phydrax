# Causal inference

`phydrax.causal` separates causal semantics from numerical inference. A result is causal only relative to an explicit study design, target population, intervention contrast, available data law, causal structure or randomized design, and recorded assumptions.

## Three distinct workflows

### Identified effects from observed data

The observed-data route is:

1. Build a role-free `CausalDataset`.
2. Declare assignment, exposure, source population, interference, and assumptions in `CausalStudyDesign`.
3. Declare a query-specific treatment contrast, outcome, and target population.
4. Establish that the required target law is available.
5. Identify an observational functional from randomization or a causal graph.
6. Issue an immutable certificate for one selected functional.
7. Fit nuisance functions out of fold, assess overlap, and evaluate the functional.

An estimator cannot accept a discovery result or arbitrary covariate list in place of an identification certificate.

```python
import phydrax as phx

schema = phx.causal.CausalSchema(
    tuple(phx.causal.CausalVariable(name=name) for name in ("z", "t", "y"))
)
graph = phx.causal.CausalDAG(
    schema=schema,
    directed_edges=(("z", "t"), ("z", "y")),
)
assert phx.causal.d_separated(graph, ("t",), ("y",), ("z",)).separated
```

### Structural causal models

An `SCMPlan` binds a DAG to exactly one mechanism per endogenous variable. Parent order, mechanism semantic/numeric identities, and the exogenous-noise realization are part of model identity. The current executor requires an explicit independent-exogenous declaration; correlated exogenous laws are rejected rather than silently factorized.

A perfect intervention replaces a mechanism. It is not conditioning. An external randomized intervention also severs the target's former causes. Policy and soft interventions declare their retained inputs and produce a new derived graph.

Counterfactuals use abduction → action → prediction. Ordinary conditional sampling is insufficient: the same abducted exogenous state must be reused across worlds.

### Causal discovery

PC-Stable produces a CPDAG only when the result is a valid completed equivalence-class graph. Conservative FCI returns a PAG and retains unresolved endpoint circles. Bounded GES searches score-equivalent classes. None constructs an SCM automatically.

A class-wide identified functional remains conditional on the discovered equivalence class. Finite-sample structure uncertainty is not erased by using a separate estimation split.

## Conditioning is not intervention

For a confounded treatment `T`, these distributions generally differ:

```text
P(Y | T=t)                 observational conditioning
P(Y | do(T=t))             mechanism replacement
P(Y_t | factual evidence)  counterfactual abduction and prediction
```

Phydrax represents these with different types and execution paths.

## Assumptions and diagnostics

Identification certificates certify a derivation conditional on assumptions; they do not prove those assumptions. Positivity is a population assumption. Empirical propensity overlap is only finite-sample evidence and may block an estimator, but cannot prove or disprove population positivity.

Diagnostics use rejected, not-rejected, inconclusive, unsupported, not-applicable, and not-run outcomes. A not-rejected graph implication never validates a graph or absence of unmeasured confounding.

## Supported boundaries

The observed-data estimators currently support point treatments, scalar outcomes, finite treatment arms, fully observed required variables, a declared no-interference design, and the sample or a predeclared subgroup target. Unknown transport, nontrivial missing-data recovery, time-varying treatment, interference, continuous dose, IV/LATE, and natural mediation effects are rejected rather than approximated silently.

General ID and conditional IDC are executable against a normalized finite observed law. Continuous general-ID functionals can be represented only when a compatible evaluator is supplied; adjustment estimands use native g-computation, IPW, or cross-fitted AIPW.

## Complete examples

- `examples/causal_adjustment.py`: known-graph AIPW.
- `examples/causal_scm.py`: intervention and counterfactual worlds.
- `examples/causal_discovery.py`: CPDAG discovery without SCM promotion.

See the [causal API overview](api/causal/index.md) for the public contracts.
