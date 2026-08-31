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

## Exact inference and normalized laws

Full enumeration remains the preferred oracle for small graphs:

```python
exact = phx.pgm.enumerate_factor_graph(graph, max_configurations=4096)
```

It returns the exact log normalizer, variable and factor marginals, a
lexicographically tie-broken MAP assignment, and the number of feasible assignments.
The complete configuration count is checked before allocation. An all-impossible graph
returns `INFEASIBLE` and zero marginal mass rather than NaNs.

For larger low-treewidth graphs, plan bounded variable elimination before execution:

```python
resources = phx.pgm.FactorGraphResourcePolicy(
    maximum_treewidth=12,
    maximum_elimination_elements=1_000_000,
)
elimination = phx.pgm.plan_variable_elimination(graph, resources=resources)
eliminated = phx.pgm.variable_elimination(elimination)
junction_tree = phx.pgm.plan_junction_tree(elimination)
calibrated = phx.pgm.junction_tree_calibrate(junction_tree)
law = phx.pgm.NormalizedFactorGraphLaw(elimination, eliminated)
```

Planning uses deterministic min-fill by default and rejects excessive induced
treewidth or workspace before execution. `NormalizedFactorGraphLaw` exposes exact
`log_prob`, support, and sampling semantics to PhydraX probability-law consumers.
Neither enumeration nor elimination silently falls back after crossing a resource cap.

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

For a factor-graph forest, the default directed schedule evaluates every
factor-to-variable edge once in inward/outward tree order. This is linear in the
incidence count for fixed factor arity and returns exact variable/factor marginals and
an exact log normalizer. The result declares `marginals_exact` and
`log_normalizer_exact`.

For a loopy graph, `SUCCESS` means the normalized message fixed-point residual reached
the requested tolerance. It does not certify marginal accuracy. `log_normalizer` is a
Bethe estimate and `log_normalizer_kind == "bethe"`. Update order is explicit:

```python
result = phx.pgm.run_belief_propagation(
    prepared,
    state,
    schedule=phx.pgm.BeliefPropagationSchedulePolicy("asynchronous"),
)
accelerated = phx.pgm.run_accelerated_belief_propagation(prepared, state)
```

Asynchronous execution is Gauss--Seidel over message blocks. Accelerated execution
uses the native nonlinear fixed-point substrate. `run_implicit_belief_propagation`
is restricted to finite, fixed-support sum-product systems and returns the nonlinear
root evidence used by its implicit derivative policy.

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

## Open, sparse, and structured factors

- `DenseTableFactorGroup`: full equal-signature tables.
- `EnumeratedFactorGroup`: common supported configurations; omitted configurations
  are exactly impossible. BP and Gibbs operate on represented support directly.
- `IsingFactorGroup`: arbitrary fixed-arity products of binary spins with parity BP.
- `PottsFactorGroup`: unary or pairwise categorical tables.
- `LogicalFactorGroup`: exact OR/AND relations with non-enumerating BP updates.
- `BinaryCardinalityFactorGroup`: active-count scores with dynamic-programming BP.
- `KernelFactorGroup`: an open `AbstractDiscreteFactorKernel` plus parameter PyTree.

Custom kernels declare stable identity and `FactorKernelCapabilities`; planners reject
unsupported sum-product, max-product, or scalar-conditional requests. Every prepared
BP/Gibbs plan retains `FactorExecutionEvidence` with represented configurations, dense
elements, message entries, workspace, and work estimates.

`FactorGraphPrecisionPolicy` controls evaluation, accumulation, certification, and
output dtypes. `FactorGraphResourcePolicy` caps configuration, dense-table, message,
elimination, coloring, and retained-sample resources. These policies are part of plan
identity; exceeding one is an error, never an implicit algorithm substitution.

Same-topology evidence batches use `BatchedBeliefPropagationState` and
`batch_belief_propagation`. `pack_factor_graphs` builds a block-diagonal `GraphIR` for
heterogeneous independent graphs while preserving per-graph semantic models and
ownership offsets.

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

Additional objectives make approximation explicit:

- `pseudolikelihood_loss` uses exact scalar conditionals.
- `bethe_negative_log_likelihood` accepts only a result labeled `\"bethe\"`.
- persistent contrastive divergence and stochastic maximum likelihood retain negative
  `GibbsState` chains across optimizer steps.
- `expectation_maximization_step` runs an exact elimination E-step, requires a
  structure-preserving caller M-step, and reports observed-objective monotonicity.

For MAP, `solve_smooth_dual_lp` returns a relaxed upper bound, decoded discrete lower
bound, and their gap. `perturb_and_map_log_normalizer` reports its unary-Gumbel
estimate and Monte Carlo standard error without presenting samples as exact.

Checkpointing uses `write_factor_graph_checkpoint` and
`read_factor_graph_checkpoint`. The archive is pickle-free and checksum-validated.
Built-in factor groups and optional BP/Gibbs states round-trip; callable kernels are
rejected because executable code cannot be reconstructed from a neutral archive.

## Advanced sampling and discrete transport

`GibbsScanPolicy` selects systematic, random-site, or randomized-color execution.
`JointDiscreteBlock` enumerates an explicitly capped dependent conditional.
`ParallelTempering` maintains replica temperatures and alternating neighboring swaps.
`wolff_cluster_step` is deliberately qualified to eager, zero-field, ferromagnetic
pairwise Ising graphs. `reduce_gibbs_chain` computes moments or a best state online
without retaining the complete chain.

Arbitrary PyTree conditional programs live in `phx.sampling.conditional`. Named
variable groups, interaction groups, stateful conditional kernels, and immutable
parallel stages generalize Gibbs-style execution beyond finite scalar nodes without
weakening `phx.pgm`'s finite-discrete invariants.

`phx.transport.discrete` provides retain-or-uniform forward noising,
factor-graph-backed reverse Gibbs kernels, multilayer denoising processes, recovery
objectives, adaptive mixing penalties, and explicit encoder/decoder embeddings.

## Boundaries

`phx.pgm` remains finite-discrete: category-valued variables, finite factor support,
and explicit resource bounds. General continuous or PyTree-valued conditional state
belongs in `phx.sampling.conditional`; it is not coerced into discrete factor tables.
Existing finite-state state-space filtering and Viterbi remain the canonical
physical-time API. Chain factor graphs are a compatible generalization, not a
replacement for their masking and temporal provenance contracts.
