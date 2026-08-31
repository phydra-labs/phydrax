# Finite-discrete probabilistic graphical models

`phydrax.pgm` represents an unnormalized finite-discrete law as a sum of local
factor log potentials. Variables are scalar category indices stored as `int32`;
variable groups retain user-facing tensor shapes while execution uses deterministic
flat variable and variable-state layouts.

## Build one graph

```python
import jax.numpy as jnp
import phydrax as phx

spins = phx.pgm.DiscreteVariableGroup("spin", shape=(4,), num_states=2)
field = phx.pgm.IsingFactorGroup(
    (phx.pgm.VariableSelection.all(spins),),
    jnp.asarray([0.2, -0.1, 0.05, 0.0]),
)
edge = phx.pgm.IsingFactorGroup(
    (
        phx.pgm.VariableSelection(spins, [0, 1, 2]),
        phx.pgm.VariableSelection(spins, [1, 2, 3]),
    ),
    jnp.asarray([0.4, -0.25, 0.3]),
)
graph = phx.pgm.DiscreteFactorGraph((spins,), (field, edge))
```

The Ising state convention is `s = 2*x - 1`, with the stored category `x` in
`{0, 1}`. `factor_graph_log_score` returns an unnormalized score, not a normalized
log probability. Exact `-inf` values represent impossible configurations; NaN and
positive infinity are rejected.

Every graph also exposes a typed bipartite `GraphIR` through `graph.topology`.
Topology, cardinalities, factor scopes, and configuration support determine the
stable `structure_id`. Trainable numeric factor values do not, so compatible values
can be refreshed without recoloring or rebuilding message routes.

## Exact enumeration

```python
exact = phx.pgm.enumerate_factor_graph(graph, max_configurations=4096)
```

Enumeration returns the exact log normalizer, variable and factor marginals, a
lexicographically tie-broken MAP assignment, and the number of feasible assignments.
The complete configuration count is checked before allocation. An all-impossible graph
returns `INFEASIBLE` and zero marginal mass rather than NaNs.

Enumeration is the preferred oracle for small graphs and qualification tests. It is
not a fallback for a model that exceeds the caller's explicit resource cap.

## Sum-product belief propagation

```python
prepared = phx.pgm.prepare_belief_propagation(
    graph,
    phx.pgm.SumProductBeliefPropagation(
        maximum_steps=200,
        relaxation=0.7,
    ),
)
state = phx.pgm.initialize_belief_propagation(prepared)
result = phx.pgm.run_belief_propagation(prepared, state)
```

Messages use one flat entry per represented incidence state; variables are not padded
to a global maximum cardinality. Hard support is preserved exactly. Sum-product
relaxation mixes normalized probability messages, allowing support to recover from an
initial message without replacing impossibility by a large finite sentinel.

For a factor-graph forest, the prepared plan runs enough undamped propagation passes to
obtain exact variable/factor marginals and an exact log normalizer. The result declares
`marginals_exact` and `log_normalizer_exact`.

For a loopy graph, `SUCCESS` means the normalized message fixed-point residual reached
the requested tolerance. It does not certify marginal accuracy. `log_normalizer` is a
Bethe estimate and `log_normalizer_kind == "bethe"`.

Unary evidence uses the same flat variable-state layout:

```python
evidence = phx.pgm.pack_evidence(
    graph,
    jnp.asarray([0.0, -jnp.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)
state = phx.pgm.initialize_belief_propagation(prepared, evidence=evidence)
```

## Max-product and MAP semantics

```python
prepared = phx.pgm.prepare_belief_propagation(
    graph,
    phx.pgm.MaxProductBeliefPropagation(),
)
result = phx.pgm.run_belief_propagation(
    prepared,
    phx.pgm.initialize_belief_propagation(prepared),
)
```

On a forest, max-product backtracks through factor configurations and returns a
consistent exact MAP assignment with `map_available=True` and `optimal=True`.

On a loopy graph, `local_modes` contains independent variable max-marginal modes.
Those modes need not form a globally consistent MAP assignment, so `map_available` and
`optimal` remain false. Max-product is nonsmooth at ties and carries no smooth-gradient
guarantee.

## Chromatic Gibbs sampling

```python
import jax.random as jr

prepared = phx.pgm.prepare_chromatic_gibbs(graph)
state = phx.pgm.initialize_gibbs(
    prepared,
    jnp.asarray(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ]
    ),
)
result = phx.pgm.sample_gibbs(
    prepared,
    state,
    key=jr.key(0),
    schedule=phx.pgm.GibbsSchedule(
        warmup_sweeps=100,
        num_draws=1000,
        sweeps_per_draw=2,
    ),
)
```

Preparation constructs a deterministic strong coloring: no two variables in one
factor scope may share a color. A supplied color vector is fully validated. Every
color stage reads one immutable pre-stage state and commits its independent updates
together.

Each scalar update evaluates the exact conditional over all states of that variable.
If every candidate is impossible, the previous state is retained and the transition
is marked `INFEASIBLE_CONDITIONAL`; no arbitrary category is sampled.

`GibbsState` is persistent. Semantic random-key addresses include chain, sweep, stage,
and variable identity, so adding chains preserves the existing chain prefix.

A shared clamped mask keeps selected sites fixed while allowing their states to affect
neighbor conditionals. The mask may change without recoloring because every graph
variable participates in the prepared coloring.

When at least two chains and four draws are available, `GibbsDiagnostics` reports
rank-normalized R-hat and bulk/tail ESS. Exact Gibbs transitions are rejection-free;
state-change fraction and invalid-conditional count remain separate because acceptance
one does not imply mixing. Sampling completion itself is not a convergence certificate.

Gibbs results lower through `phx.integration.markov_chain_measure`. The resulting
empirical target retains chain/draw axes and `independent=False`; integration never
invents an IID standard error for correlated draws.

## Structured factors

- `DenseTableFactorGroup`: full equal-signature tables.
- `EnumeratedFactorGroup`: common supported configurations; omitted configurations
  are exactly impossible.
- `IsingFactorGroup`: arbitrary fixed-arity products of binary spins.
- `PottsFactorGroup`: unary or pairwise categorical tables.
- `LogicalFactorGroup`: exact OR/AND parent-to-child relations.
- `BinaryCardinalityFactorGroup`: a score indexed by the number of active variables.

Belief-propagation and Gibbs preparation explicitly cap dense factor-configuration
work. A high-arity factor that exceeds the cap is rejected rather than silently
materialized.

## Parameter refresh and training

`refresh_belief_propagation` and `refresh_chromatic_gibbs` replace factor tables only
when structure and parameter signatures match. Scope, cardinality, kernel, shape, or
dtype changes require explicit re-preparation.

Exact maximum-likelihood training accepts only exact enumeration or exact forest
sum-product normalizers:

```python
observed_assignments = jnp.asarray(
    [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
    ]
)
loss, diagnostics = phx.pgm.exact_factor_graph_negative_log_likelihood(
    graph,
    observed_assignments,
    exact,
)
```

A loopy Bethe estimate is never relabeled as an exact likelihood.

`contrastive_divergence_loss` accepts separately materialized positive and negative
states. Negative states are stop-gradient by default; differentiation through discrete
Gibbs trajectories is not implied. The function returns a pure scalar and diagnostics
for composition with Equinox/Optax and existing PhydraX training controllers.

## Boundaries

The current substrate intentionally excludes continuous/PyTree-valued nodes, joint
dependent Gibbs blocks, junction-tree inference, hardware execution contracts, and
denoising thermodynamic model training. Existing finite-state state-space filtering and
Viterbi remain the canonical time-series API; chain factor graphs are a compatible
generalization, not a replacement for their physical-time and masking contracts.
