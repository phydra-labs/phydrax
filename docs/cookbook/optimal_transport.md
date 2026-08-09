# Optimal transport

This recipe builds balanced finite-measure transport problems, uses exact and
regularized distances, adds transport objectives to SciML training, compares
whole-function predictive laws, and transforms weighted particles. Read the
[transport guide](../guides_transport.md) first for measure, scaling, convergence, and
method-selection semantics.

```python
import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx
```

## 1. Transport two weighted point clouds

Use integration targets as the public measure representation. `normalized=True`
means that each weight vector is normalized over active atoms and represents a
probability law.

```python
def probability_measure(points, weights, *, provenance):
    return phx.integration.discrete(
        jnp.asarray(points),
        cx.Field(jnp.asarray(weights), dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance=provenance,
    )


source = probability_measure(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    [0.2, 0.5, 0.3],
    provenance="source-cloud",
)
target = probability_measure(
    [[0.2, 0.1], [0.8, 0.2], [0.1, 0.9]],
    [0.4, 0.4, 0.2],
    provenance="target-cloud",
)
problem = phx.transport.discrete_problem(
    source,
    target,
    cost=phx.transport.SquaredEuclideanCost(),
)
solver = phx.transport.Sinkhorn(
    0.5,
    max_iterations=500,
    tolerance=1e-6,
    check_every=5,
)
result = phx.transport.require_converged(solver(problem))

transport_cost = result.transport_cost
regularized_cost = result.regularized_cost
source_residual = result.source_marginal() - problem.source_weights
target_residual = result.target_marginal() - problem.target_weights
```

Inspect `result.diagnostics.normalized_marginal_residual`,
`result.diagnostics.primal_dual_gap`, `result.diagnostics.status`, and
`result.provenance` before reporting the objective.

## 2. Apply the coupling without forming it

Plan actions support arbitrary trailing payload dimensions.

```python
source_payload = jnp.asarray(
    [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]
)
transported_mass = result.apply_source_to_target(source_payload)
target_conditioned = result.barycentric_source_to_target(source_payload)
```

`transported_mass` is a physical coupling action. `target_conditioned` divides by the
target marginal and is the target-conditioned barycentric payload. Materialize
`result.dense_plan()` only for diagnostics that truly require every coupling entry.

## 3. Use periodic or component-scaled geometry

Use component scales for commensurate nonperiodic coordinates. For two angular
coordinates with periods `2*pi` and `pi`, use a periodic cost:

```python
scaled_cost = phx.transport.WeightedSquaredEuclideanCost(
    jnp.asarray([2.0 * jnp.pi, 0.25])
)
periodic_cost = phx.transport.PeriodicSquaredEuclideanCost(
    jnp.asarray([2.0 * jnp.pi, jnp.pi])
)
```

Use one coherent cost for a problem. For mixed periodic/nonperiodic semantics, define
a small custom `AbstractGroundCost`; a large artificial period is not a nonperiodic
declaration. Scales and periods are part of the scientific model and must match the
event feature size.

## 4. Keep masked padding inert

A mask is defined on the atom axis. Masked coordinates may carry padding values, but
active coordinates must be finite and active mass must remain positive.

```python
mask = cx.Field(jnp.asarray([True, True, False, False]), dims=("atom",))
padded = phx.integration.discrete(
    jnp.asarray([[0.0], [1.0], [999.0], [999.0]]),
    cx.Field(jnp.asarray([0.4, 0.6, 0.0, 0.0]), dims=("atom",)),
    axes="atom",
    mask=mask,
    normalized=True,
    provenance="masked-cloud",
)
```

Do not mask a positive-weight physical atom to force two distributions to have equal
mass. Choose the measure contract explicitly.

## 5. Compute and reuse Sinkhorn divergence

```python
divergence = phx.transport.sinkhorn_divergence(problem, solver)

reference = phx.transport.prepare_sinkhorn_reference(
    target,
    cost=phx.transport.SquaredEuclideanCost(),
    solver=solver,
)
repeated = phx.transport.sinkhorn_divergence_against(source, reference)
```

`divergence.converged` requires the cross and both self solves to converge. A prepared
reference reuses the target self solve; it is appropriate when the target, cost,
solver, encoder semantics, and mass tolerance are fixed across evaluations.

## 6. Use exact scalar Wasserstein distance

```python
exact = phx.transport.wasserstein_distance_1d(
    jnp.asarray([0.0, 1.0, 3.0]),
    jnp.asarray([0.5, 2.0]),
    source_weights=jnp.asarray([0.2, 0.5, 0.3]),
    target_weights=jnp.asarray([0.6, 0.4]),
    p=2.0,
)
```

This is exact for the supplied weighted one-dimensional empirical measures and may use
different atom counts. It is preferable to regularized Sinkhorn when the event is
truly scalar.

## 7. Replay sliced Wasserstein projections

```python
source_events = jnp.asarray(
    [[0.0, 0.0], [1.0, 0.5], [0.2, 1.0], [0.8, 0.9]]
)
target_events = source_events + jnp.asarray([0.3, -0.1])

sliced = phx.transport.sliced_wasserstein_distance(
    source_events,
    target_events,
    p=2.0,
    num_projections=64,
    key=jr.key(0),
)
replay = phx.transport.sliced_wasserstein_distance(
    source_events,
    target_events,
    p=2.0,
    projections=sliced.projections,
)
```

Store `sliced.projections` with an experiment. The projection count and design are
part of the estimator, not an incidental random seed.

## 8. Build differentiable order objectives

```python
values = cx.Field(
    jnp.asarray([[3.0, 1.0, 4.0, 2.0], [5.0, -1.0, 1.0, 0.0]]),
    dims=("case", "member"),
)
soft_sorted = phx.transport.soft_sort(values, axis="member", epsilon=0.05)
soft_ranks = phx.transport.soft_rank(values, axis="member", epsilon=0.05)
median = phx.transport.soft_quantile(
    values,
    0.5,
    axis="member",
    epsilon=0.05,
)
top_two_mask = phx.transport.soft_topk_mask(
    values,
    2,
    axis="member",
    epsilon=0.05,
)
```

The field dimension order is preserved. Compare against hard sort, rank, quantile, and
top-k outputs at the selected `epsilon`; these operators are smooth approximations.

## 9. Add transport to a functional solver

Prepare fixed reference data once, then make the model-to-measure map explicit.

```python
reference_points = jnp.linspace(0.0, 1.0, 64)[:, None]
reference_measure = probability_measure(
    reference_points,
    jnp.ones((64,)),
    provenance="reference-density",
)
reference = phx.transport.prepare_sinkhorn_reference(
    reference_measure,
    cost=phx.transport.SquaredEuclideanCost(),
    solver=phx.transport.Sinkhorn(0.5, max_iterations=500, tolerance=1e-6),
)


def model_measure(functions):
    log_density = functions["log_density"](reference_points)
    return phx.integration.discrete(
        reference_points,
        cx.Field(jnp.exp(log_density), dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance="model-density",
    )


transport_term = phx.terms.SpatialSinkhornDivergenceTerm(
    model_measure,
    reference,
    objective_vars=("log_density",),
    weight=1.0,
    label="density-transport",
)
```

Pass `transport_term` with the other terms to `FunctionalSolver`. Its
`TermEvaluation.diagnostics` retains the three Sinkhorn solves. The term raises if any
solve does not converge.

For sample generators use `EmpiricalSinkhornDivergenceTerm`; for a projection-based
whole-event objective use `SlicedWassersteinTerm`; for a quantile constraint use
`SoftQuantileFunctional`.

## 10. Compare predictive and operator-output laws

Raw empirical predictive samples use one leading sample axis and treat every trailing
coordinate as one event:

```python
source_draws = jnp.asarray(
    [[0.0, 1.0], [0.2, 0.8], [-0.1, 1.1], [0.1, 0.9]]
)
target_draws = source_draws + jnp.asarray([0.3, -0.2])
predictive_metric = phx.uq.predictive_sinkhorn_divergence(
    source_draws,
    target_draws,
    epsilon=0.2,
)
```

For neural operators, first construct `OperatorPredictiveField` objects through
`operator_predictive_from_samples` or `sample_operator_predictive`. Then preserve
physical cases and query geometry:

```py
operator_metric = phx.uq.operator_ensemble_sinkhorn_divergence(
    left_predictive,
    right_predictive,
    measure="quadrature",
    reduction="none",
    epsilon=0.5,
)
operator_sliced = phx.uq.operator_ensemble_sliced_wasserstein(
    left_predictive,
    right_predictive,
    measure="quadrature",
    reduction="none",
    num_projections=64,
    key=jr.key(1),
)
```

`per_case` retains physical case shape. Query masks and quadrature are applied before
whole-field events are compared; incompatible output/query contracts are rejected.

## 11. Penalize distributional semigroup inconsistency

For an `AbstractProbabilisticOperatorModel` whose uncertainty source is `"process"`:

```py
semigroup = phx.nn.operator.training.SinkhornDistributionalSemigroupObjective(
    num_samples=16,
    measure="quadrature",
    reduction="mean",
    epsilon=1.0,
    key_mode="fold_in",
)
loss = semigroup(
    model,
    transition_batch,
    dt1,
    dt2,
    condition_batch,
    advance_batch,
    key=jr.key(2),
)
```

The direct and independently composed predictive laws use controlled independent
subkeys. This is a marginal process-law objective. Use pathwise cocycle objectives
when both paths share one explicit driver realization.

## 12. Transform weighted particles deterministically

```python
particles = jnp.asarray(
    [[-1.0, 0.0], [0.0, 0.5], [1.0, 2.0], [2.0, 3.0]]
)
weights = jnp.asarray([0.05, 0.15, 0.30, 0.50])
transform = phx.uq.optimal_transport_ensemble_transform(
    particles,
    weights,
    particle_axis=0,
    epsilon=1.0,
)
equal_weight_particles = transform.particles
mean_error = transform.mean_error
```

The output has the same particle count and axis order. Check transport convergence and
`mean_error`; do not substitute this transform into a categorical-resampling API whose
genealogy or randomness is part of its contract.

## 13. Run smoke benchmarks

```bash
python -m tools.transport_benchmarks --smoke
python -m tools.soft_transport_benchmarks --smoke
python -m tools.transport_scientific_benchmarks --smoke
```

Use the non-smoke size and repeat flags for performance claims. The smoke mode proves
execution and emits deterministic JSON records; it is not a stable cross-machine
performance baseline.
