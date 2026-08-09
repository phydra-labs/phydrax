# Optimal transport

`phydrax.transport` is the native finite-measure transport subsystem. It consumes
`phydrax.integration.DiscreteMeasureTarget` and `WeightedSampleTarget` objects rather
than introducing a second measure hierarchy. Coordinate axes, event structure,
masks, support validity, physical mass, normalization, and provenance remain explicit
through lowering, solving, and plan application.

The first release covers balanced finite-mass transport with equal source and target
mass, stabilized entropic Sinkhorn solves, debiased Sinkhorn divergence, exact
one-dimensional Wasserstein distance, sliced Wasserstein distance, and differentiable
order operators. It does not silently normalize unequal physical masses and does not
implement unbalanced transport, Gromov--Wasserstein, barycenter solvers, or low-rank
transport.

## Basic workflow

```python
import coordax as cx
import jax.numpy as jnp
import phydrax as phx

source = phx.integration.discrete(
    jnp.asarray([[0.0], [1.0], [2.0]]),
    cx.Field(jnp.asarray([1.0, 2.0, 1.0]), dims=("atom",)),
    axes="atom",
    normalized=True,
)
target = phx.integration.discrete(
    jnp.asarray([[0.5], [1.5]]),
    cx.Field(jnp.asarray([1.0, 1.0]), dims=("atom",)),
    axes="atom",
    normalized=True,
)
problem = phx.transport.discrete_problem(
    source,
    target,
    cost=phx.transport.SquaredEuclideanCost(),
)
solver = phx.transport.Sinkhorn(
    0.2,
    max_iterations=500,
    tolerance=1e-7,
    check_every=5,
)
result = phx.transport.require_converged(solver(problem))
```

`result` retains potentials, transport and regularization components, primal and dual
values, marginal residuals, status, iteration counts, and numerical provenance. Its
plan actions multiply payloads without materializing the coupling. Call
`result.dense_plan()` only when the complete source-by-target matrix is genuinely
needed.

## Measure and problem contract

- `normalized=True` defines a probability measure. Input weights are normalized over
  active atoms, while the represented physical mass is one.
- `normalized=False` preserves the supplied physical mass. Balanced transport requires
  source and target masses to agree within `mass_tolerance`.
- Zero-weight and masked atoms remain represented but are inert. Positive-infinite and
  NaN weights, active non-finite coordinates, empty active support, and mismatched
  event encodings are rejected.
- A structured event PyTree must lower to one array leaf or provide an explicit
  `encoder=`. The encoder is part of the scientific model, not serialization detail.
- Ground-cost units determine the units of `epsilon`; coordinate scaling therefore
  changes both geometry and the appropriate regularization scale.

::: phydrax.transport.DiscreteTransportProblem

---

::: phydrax.transport.TransportProblemProvenance

---

::: phydrax.transport.discrete_problem

## Result status

`TransportStatus.CONVERGED` means the configured marginal tolerance was met. Reaching
an iteration budget, numerical stagnation, or a non-finite iterate remains visible in
`SinkhornDiagnostics`; no failed result is clipped or repaired. Training integrations
call `require_converged` and fail rather than optimize through an invalid solve.

::: phydrax.transport.TransportStatus

---

::: phydrax.transport.status_message

## Related pages

- [Ground costs](costs.md)
- [Sinkhorn solving and divergence](sinkhorn.md)
- [Exact and sliced distances](distances.md)
- [Differentiable ordering](soft.md)
- [Guide: transport semantics and method choice](../../guides_transport.md)
- [Cookbook: optimal transport workflows](../../cookbook/optimal_transport.md)
