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

space = phx.combinatorial.CardinalitySpace(5, 2)
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

## Boundable linear oracles

`AbstractBoundableCombinatorialSpace` declares compact feature bounds and the
coordinates that are integral. `CombinatorialFeatureRestriction` narrows those
bounds immutably at one search node. A method must declare
`bound_restrictions=True` and implement
`AbstractBoundableLinearCombinatorialMethod`; unsupported methods cannot ignore
a restriction.

`solve_restricted_combinatorial` independently checks the returned decision
against both the original space and the requested bounds. Cardinality,
Hungarian assignment, and exact set packing support required and forbidden
features. Shortest-path and flow restrictions remain unsupported until their
required-edge/flow semantics have dedicated certificates.

These restricted oracles power `phydrax.optim.solve_integer_hull`. The
combinatorial layer owns exact linear minimization and logical decisions; the
optimization layer owns convex objective evaluation, Frank–Wolfe gaps, and the
global tree.

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

## Capacitated auction

`CapacitatedAuctionPlan(sites, labels, width)` assigns every site to one label
through a fixed-width candidate relation (slot `(x, c)` names label
`candidate_labels[x, c]`) under integer label-count bounds
`lower_i <= n_i <= upper_i`; `lower == upper` prescribes exact counts. Values
are maximized. The plan owns a static, strictly decreasing epsilon schedule
(`epsilon_scale="value-range"` multiplies it by the valid value range,
`"absolute"` does not), a per-phase round cap, and a route budget: plans with
`sites * width > maximum_routes` are refused.

Each epsilon stage runs a Jacobi forward auction (unassigned sites bid; labels
accept by sorted bids up to their upper counts; prices only rise) followed by a
Jacobi reverse auction (labels below their lower count, or below their upper
count at a positive price, lower their prices and make offers; prices only
fall). Both phases are bounded `lax.while_loop`s and the whole solve is
traceable, so it can run inside `jax.jit` or `lax.scan`.

```python
plan = phx.combinatorial.CapacitatedAuctionPlan(4, 2, 2)
prepared = plan.prepare(jnp.asarray([[0, 1]] * 4))
result = prepared.solve(
    jnp.asarray([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),
    jnp.asarray([2, 2]),
    jnp.asarray([2, 2]),
)
assert result.success
assert result.labels.tolist() == [0, 0, 1, 1]
```

`prepare` host-validates a fixed topology once (duplicate, out-of-range, or
non-int32-representable labels on valid slots raise `ValueError`).
`CapacitatedAuctionPlan.solve` accepts candidates built on device instead and
checks labels, count bounds, and warm-start slots in their original integer
dtype before narrowing. Inconsistent candidates, bounds, or warm starts yield
`INFEASIBLE`; the corresponding `evidence.candidates_consistent`,
`evidence.bounds_consistent`, or `evidence.warm_start_consistent` flag is false.

The returned `prices` are numerical dual variables of the count constraints,
not physical pressures. `CapacitatedAuctionEvidence` reports the primal value
`P`, the dual value
`D = sum_x max_c (a - p) + sum_i max(upper_i p_i, lower_i p_i)`, and their gap:
`P <= optimum <= D` always, and `D - P <= sites * epsilon` for a certified
result, so integer values with `sites * epsilon < 1` are solved exactly. Status
precedence is `NONFINITE_INPUT`, `INFEASIBLE` (inconsistent candidates, bounds,
or warm start, or price divergence), `MAXIMUM_STEPS_REACHED`,
`CERTIFICATION_FAILED`, then `OPTIMAL` (gap within the default certification
tolerance) or `FEASIBLE` (certified epsilon-optimal). Failed solves return
labels and slots `-1`, zero counts, and NaN prices; no partial assignment is
exposed. Warm starts pass the previous `prices` as `initial_prices`; optional
`initial_slots` values must be representable in signed int32.

`CapacitatedAssignmentSpace` and `CapacitatedAuction` expose the same kernel
through `solve_combinatorial`, minimizing `costs = -values` with the chosen-slot
one-hot matrix as objective features.

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

::: phydrax.combinatorial.AbstractBoundableCombinatorialSpace

---

::: phydrax.combinatorial.CombinatorialFeatureRestriction

---

::: phydrax.combinatorial.solve_restricted_combinatorial

---

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

::: phydrax.combinatorial.CapacitatedAuctionPlan

---

::: phydrax.combinatorial.PreparedCapacitatedAuction

---

::: phydrax.combinatorial.CapacitatedAuctionResult

---

::: phydrax.combinatorial.CapacitatedAuctionEvidence

---

::: phydrax.combinatorial.CapacitatedAssignmentSpace

---

::: phydrax.combinatorial.CapacitatedAssignmentDecision

---

::: phydrax.combinatorial.CapacitatedAuction

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
