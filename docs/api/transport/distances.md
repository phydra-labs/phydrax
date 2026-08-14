# Exact and sliced Wasserstein distances

## Exact one-dimensional distance

`wasserstein_distance_1d` computes the exact weighted empirical Wasserstein distance
for finite one-dimensional measures. Source and target atom counts may differ. The
implementation sorts both supports and integrates the quantile mismatch; it does not
solve a regularized transport problem.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

source_values = jnp.asarray([0.0, 1.0, 3.0])
target_values = jnp.asarray([0.5, 2.0])
source_probabilities = jnp.asarray([0.2, 0.5, 0.3])
target_probabilities = jnp.asarray([0.6, 0.4])
value = phx.transport.wasserstein_distance_1d(
    source_values,
    target_values,
    source_weights=source_probabilities,
    target_weights=target_probabilities,
    p=2.0,
)
```

`p` must be finite and at least one. Weights must be finite, nonnegative, and have
positive total mass; each side is normalized to a probability measure. The returned
quantity is the Wasserstein distance, not its `p`th power.

::: phydrax.transport.wasserstein_distance_1d

## Sliced Wasserstein distance

`sliced_wasserstein_distance` projects vector events onto one-dimensional directions,
evaluates the exact weighted distance for each projection, and aggregates the
projection costs before taking the `p`th root. It retains the actual normalized
projections so a stochastic estimate can be replayed exactly.

Pass either `key=` with `num_projections=` or an explicit `projections=` array. These
modes are mutually exclusive. Reusing the returned projections makes comparisons and
gradient checks independent of PRNG consumption.

```python
source_events = jnp.asarray([[0.0, 0.0], [1.0, 0.5], [0.2, 1.0], [0.8, 0.9]])
target_events = source_events + jnp.asarray([0.3, -0.1])
estimate = phx.transport.sliced_wasserstein_distance(
    source_events,
    target_events,
    p=2.0,
    num_projections=128,
    key=jr.key(0),
)
replay = phx.transport.sliced_wasserstein_distance(
    source_events,
    target_events,
    p=2.0,
    projections=estimate.projections,
)
```

::: phydrax.transport.SlicedWassersteinResult

---

::: phydrax.transport.sliced_wasserstein_distance

## Method choice

- Use exact one-dimensional Wasserstein when the scientific event is scalar.
- Use sliced Wasserstein when full event dimension or atom count makes a dense
  coupling undesirable and projection variance is acceptable.
- Use Sinkhorn divergence when the chosen multivariate ground geometry and a smooth
  finite coupling are central to the objective.

Sliced distance is a finite-projection estimator, not the full multivariate
Wasserstein distance. Report the projection design alongside the value.
