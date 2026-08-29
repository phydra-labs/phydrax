# Exact combinatorial decisions and surrogate learning

This workflow solves and certifies a hard assignment, then differentiates a
separate DAG path loss through one exact perturbed solve. Both numerical methods
remain native JAX computations.

```python
import jax
import jax.numpy as jnp
import phydrax as phx
```

## Certify a hard assignment

The assignment space requires every row to select one distinct valid column.
Forbidden entries are structural, not large objective penalties.

```python
assignment_space = phx.combinatorial.BipartiteAssignmentSpace(
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
assignment_costs = jnp.asarray(
    [
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ]
)
assignment_problem = phx.combinatorial.LinearCombinatorialProblem(
    assignment_space,
    assignment_costs,
    problem_id="cookbook-assignment",
)
assignment = phx.combinatorial.solve_combinatorial(
    assignment_problem,
    phx.combinatorial.HungarianAssignment(),
)

if not assignment.success:
    raise RuntimeError(
        phx.combinatorial.combinatorial_status_message(assignment.status)
    )

assert assignment.decision.columns.tolist() == [1, 0, 2]
assert assignment.objective_value == 5.0
assert assignment.certificate.absolute_gap <= 1e-12
```

`assignment.features` is the binary matrix dual to `assignment_costs`.
`assignment.decision.columns` is the compact logical decision. The primal-dual
certificate is independent of the returned matrix encoding.

## Learn edge costs through a hard DAG path

Declare a fixed directed graph. Edge index is the objective-feature coordinate.
Acyclicity permits signed costs in both the forward and perturbed solves.

```python
edges = phx.sparse.EdgeRelation(
    jnp.asarray([0, 0, 1, 2, 1], dtype=jnp.int32),
    jnp.asarray([1, 2, 2, 3, 3], dtype=jnp.int32),
    source_size=4,
    target_size=4,
)
path_space = phx.combinatorial.ShortestPathSpace(edges, 0, 3)
path_method = phx.combinatorial.DAGShortestPath()
interpolation = phx.combinatorial.BlackboxInterpolation(1.0)

target_incidence = jnp.asarray([0.0, 1.0, 0.0, 1.0, 0.0])


def decision_loss(edge_costs):
    problem = phx.combinatorial.LinearCombinatorialProblem(
        path_space,
        edge_costs,
        problem_id="learned-dag-path",
    )
    chosen_incidence = phx.combinatorial.blackbox_solution(
        problem,
        path_method,
        policy=interpolation,
    )
    return 0.5 * jnp.sum((chosen_incidence - target_incidence) ** 2)


initial_costs = jnp.asarray([2.0, 5.0, -1.0, 1.0, 10.0])
loss, surrogate_gradient = jax.value_and_grad(decision_loss)(initial_costs)
```

The returned gradient is the blackbox interpolation rule, not the derivative of
the locally constant hard path map. Inspect the explicit evidence before choosing
`λ`:

```python
hard_problem = phx.combinatorial.LinearCombinatorialProblem(
    path_space,
    initial_costs,
    problem_id="learned-dag-path",
)
hard = phx.combinatorial.solve_combinatorial(hard_problem, path_method)
cotangent = hard.features - target_incidence
pullback = phx.combinatorial.estimate_blackbox_pullback(
    hard_problem,
    path_method,
    cotangent,
    policy=interpolation,
)

assert pullback.valid
assert jnp.allclose(surrogate_gradient, pullback.gradient)
```

Useful diagnostics:

- `pullback.forward` and `pullback.perturbed`: both certified executions;
- `pullback.relative_perturbation`: cost-relative perturbation magnitude;
- `pullback.feature_change_norm`: discrete structural change;
- `pullback.zero_gradient`: whether the perturbed optimum stayed unchanged;
- `pullback.exact_theory_applicable`: all exactness assumptions passed.

Do not differentiate changing edge topology, validity masks, source/target roles,
or other feasible-set data through this estimator. Use a continuous relaxation
or another explicitly declared estimator when those quantities are trainable.
