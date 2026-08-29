# Native combinatorial optimization

`phydrax.combinatorial` solves fixed finite decision problems whose objective is
linear in an explicit feature representation. Every method in this namespace is
JAX-native, batched, deterministic at ties, and independently certified. No host
solver or callback is selected implicitly.

## Decision and feature spaces

A logical decision and its objective features are separate objects. A shortest
path is represented logically by its ordered vertices and edges, while its
objective features are an edge-incidence vector. An assignment is represented
by one selected column per row, while its objective features are a binary
assignment matrix.

For a fixed feasible set `Y`, costs `c`, and feature map `φ`, every declared
problem has objective

```text
y*(c) = argmin y in Y  <c, φ(y)>.
```

`LinearCombinatorialProblem` validates that cost and feature PyTrees have the
same structure and trailing shapes. Leading cost axes are independent batch
axes. `with_costs` replaces only numeric costs and rejects shape, dtype, batch,
or structure changes.

```python
import jax.numpy as jnp
import phydrax as phx

space = phx.combinatorial.CardinalitySpace(size=5, count=2)
problem = phx.combinatorial.LinearCombinatorialProblem(
    space,
    jnp.asarray([3.0, -1.0, 2.0, 0.0, 4.0]),
)
result = phx.combinatorial.solve_combinatorial(
    problem,
    phx.combinatorial.StableCardinalityOracle(),
)

assert result.success
assert result.decision.indices.tolist() == [1, 3]
```

Static contract errors raise before execution. Numerical failures are represented
by `CombinatorialStatus` and fixed-shape invalid results. `success` means that
the decision is feasible, its objective is independently reproducible, and its
method-specific optimality certificate passed.

## Planning and resource bounds

`plan_combinatorial` validates method/space compatibility without solving. The
returned `CombinatorialPlan` records static work and workspace estimates,
method configuration, certificate kind, tie policy, and content-addressed
problem identity. Resource limits are method-specific rather than expressed in
ambiguous generic work units.

```python
plan = phx.combinatorial.plan_combinatorial(
    problem,
    phx.combinatorial.StableCardinalityOracle(maximum_items=100_000),
)
assert plan.capabilities.exact
assert plan.capabilities.jax_native
```

## Explicit finite decision sets

`ExplicitDecisionSpace` stores independent decision and feature catalogs with a
common leading candidate axis. `ExhaustiveLinearOracle` streams the catalog in
bounded batches, selects the lowest canonical candidate index at exact ties,
and reports the exact second-best margin.

Use this method for genuinely explicit finite feasible sets and as a reference
oracle for small structured problems. It never materializes a Cartesian product
that was not explicitly declared.

## Fixed cardinality

`CardinalitySpace(n, k)` declares binary decisions with exactly `k` selected
valid items. `StableCardinalityOracle` uses stable native sorting, supports
signed costs and item masks, and certifies optimality from the selected/unselected
cost boundary. `k=0` and `k=n` are valid one-decision spaces.

## Bipartite assignment

`BipartiteAssignmentSpace(rows, columns)` assigns every row exactly once and
each column at most once. A boolean matrix masks forbidden edges.
`HungarianAssignment` implements the primal-dual shortest-augmenting-path
Hungarian method with bounded JAX loops and a primal-dual certificate.

Partial assignment is not implicit. Add explicit dummy columns and declared
costs when unmatched rows are part of the model.

```python
space = phx.combinatorial.BipartiteAssignmentSpace(
    3,
    3,
    valid=jnp.asarray(
        [
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ]
    ),
)
problem = phx.combinatorial.LinearCombinatorialProblem(
    space,
    jnp.asarray(
        [
            [4.0, 1.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 2.0, 2.0],
        ]
    ),
)
result = phx.combinatorial.solve_combinatorial(
    problem,
    phx.combinatorial.HungarianAssignment(),
)
assert result.success
assert result.certificate.dual_available
```

## Directed acyclic shortest paths

`ShortestPathSpace` uses a fixed `phydrax.sparse.EdgeRelation`. Its constructor
computes deterministic static topology and incoming-edge tables.
`DAGShortestPath` supports arbitrary signed finite edge costs because acyclicity
precludes negative cycles. The result contains a fixed-capacity ordered path and
edge-incidence features. Distance potentials certify the primal path objective.

A cyclic relation is a valid `ShortestPathSpace`, but planning
`DAGShortestPath` for it fails explicitly. Future graph methods can share the
same space without changing its decision semantics.

## Hard solutions and gradients

`solve_combinatorial` returns stopped-gradient decisions and features. Hard
finite argmin maps are locally constant almost everywhere; silently assigning a
surrogate derivative would make ordinary `jax.grad` calls misleading.

`BlackboxInterpolation` is an explicit opt-in loss-dependent surrogate. Given a
forward optimum `y`, incoming feature cotangent `g`, and positive scale `λ`, its
pullback solves once at perturbed costs:

```text
c' = c + λ g
y' = argmin <c', φ(y)>
surrogate cost gradient = [φ(y') - φ(y)] / λ.
```

`estimate_blackbox_pullback` exposes both solves and complete evidence.
`blackbox_solution` provides the same rule through a custom VJP for ergonomic
first-order reverse-mode training.

```python
policy = phx.combinatorial.BlackboxInterpolation(1.0)
features = phx.combinatorial.blackbox_solution(
    problem,
    phx.combinatorial.HungarianAssignment(),
    policy=policy,
)
```

This is not the VJP of a fixed classical Jacobian: the backward map is generally
nonlinear in the incoming cotangent. Consequently:

- only objective costs receive a surrogate gradient;
- topology, masks, constraints, method configuration, and `λ` are fixed;
- both forward and perturbed solves must be exact and certified;
- additive interpolation requires signed-cost support;
- no clipping or projection is performed;
- forward-mode JVP is unsupported;
- higher-order derivatives have no classical interpretation;
- deterministic ties select one declared interpolation branch.

`λ` is a bias/gradient-density parameter, not a step that should automatically
approach zero. Inspect `relative_perturbation`, `feature_change_norm`, and
`zero_gradient` when tuning it.

## Relationship to other Phydrax substrates

- `phydrax.optim.FiniteProductSpace` describes arbitrary candidate catalogs for
  black-box domain searches; it does not declare a linear feature map.
- `phydrax.optim` owns continuous, constrained, stochastic, and derivative-free
  optimization.
- `phydrax.transport` owns continuous transport plans and differentiable soft
  assignment/order relaxations.
- `phydrax.graph` owns graph data and learned graph operators.
- `phydrax.sparse.EdgeRelation` supplies fixed sparse topology reused by native
  path spaces.

## API

::: phydrax.combinatorial.AbstractCombinatorialSpace

---

::: phydrax.combinatorial.LinearCombinatorialProblem

---

::: phydrax.combinatorial.CombinatorialPlan

---

::: phydrax.combinatorial.CombinatorialResult

---

::: phydrax.combinatorial.CombinatorialCertificate

---

::: phydrax.combinatorial.CombinatorialStatus

---

::: phydrax.combinatorial.ExplicitDecisionSpace

---

::: phydrax.combinatorial.ExhaustiveLinearOracle

---

::: phydrax.combinatorial.CardinalitySpace

---

::: phydrax.combinatorial.StableCardinalityOracle

---

::: phydrax.combinatorial.BipartiteAssignmentSpace

---

::: phydrax.combinatorial.HungarianAssignment

---

::: phydrax.combinatorial.ShortestPathSpace

---

::: phydrax.combinatorial.DAGShortestPath

---

::: phydrax.combinatorial.BlackboxInterpolation

---

::: phydrax.combinatorial.estimate_blackbox_pullback

---

::: phydrax.combinatorial.blackbox_solution
