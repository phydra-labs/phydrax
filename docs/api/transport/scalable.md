# Positive-feature scalable transport

Positive-feature transport is an explicitly approximate balanced backend. It applies
to the entropic kernel induced by `SquaredEuclideanCost` and does not introduce a
second measure hierarchy: build the same `DiscreteTransportProblem` used by exact
`Sinkhorn`.

## Approximation contract

`GaussianPositiveFeatures` draws positive Gaussian random features from its explicit
JAX key. Reusing the key, rank, problem, and `epsilon` replays the same factors and
probe pairs. The resulting `PositiveKernelFactors` validates finite nonnegative
factors and retains row log scales rather than clipping or silently normalizing an
invalid approximation.

`PositiveKernelApproximationDiagnostics` records:

- the key data, requested rank, and probe count;
- source and target probe indices;
- exact and approximate kernel values and relative errors;
- zero source and target kernel-row counts;
- a fixed `KernelApproximationStatus`.

A non-finite feature construction, zero kernel row, non-finite probe, or exceeded
probe tolerance remains a failed approximation. It is not repaired and
`PositiveFeatureSinkhornResult.converged` is false.

```python
import coordax as cx
import jax.numpy as jnp
import jax.random as jr
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

feature_map = phx.transport.GaussianPositiveFeatures(
    jr.key(2026),
    512,
    num_probes=64,
    probe_tolerance=0.2,
)
solver = phx.transport.PositiveFeatureSinkhorn(
    1.0,
    feature_map,
    max_iterations=500,
    tolerance=1e-7,
)
result = solver(problem)
checked = phx.transport.require_converged(result)
```

## Objective and plan semantics

The solver scales the represented factorized kernel without constructing the full
source-by-target matrix. Marginals and plan actions preserve physical mass and
arbitrary trailing payload event shape. `dense_plan()` is an explicit full-matrix
request.

`regularized_cost` (also exposed as `surrogate_regularized_cost`) is the entropic
objective of the represented surrogate kernel. Request
`solver(problem, exact_ground_cost=True)` to additionally compute
`exact_transport_cost`, the exact ground-cost statistic of that approximate plan.
That statistic uses compiled blocks and does not make the plan or solve exact.

`TransportProvenance.approximation` records the positive-factor method and rank;
the fixed approximation diagnostics retain probe count and values. Balanced
divergence, prepared references, UQ transport metrics,
transport terms, the distributional semigroup objective, and the particle transform
consume the common `AbstractBalancedTransportSolver` and
`AbstractBalancedTransportPlan` contracts, retain result provenance, and reject
nonconvergence where a scientific training value is consumed.

Blockwise exact Sinkhorn evaluates compiled array blocks. It is not host-callback
streaming or an out-of-core API. No Python callback, signed factor, or generic
Nyström contract is part of this feature.

---

::: phydrax.transport.AbstractBalancedTransportPlan
    options:
        show_root_heading: true
        show_source: false

---

::: phydrax.transport.AbstractBalancedTransportSolver
    options:
        show_root_heading: true
        show_source: false

---

::: phydrax.transport.PositiveKernelFactors
    options:
        show_root_heading: true
        show_source: false

---

::: phydrax.transport.PositiveKernelApproximationDiagnostics
    options:
        show_root_heading: true
        show_source: false

---

::: phydrax.transport.KernelApproximationStatus
    options:
        show_root_heading: true
        show_source: false

---

::: phydrax.transport.GaussianPositiveFeatures
    options:
        show_root_heading: true
        show_source: false

---

::: phydrax.transport.PositiveFeatureSinkhorn
    options:
        show_root_heading: true
        show_source: false

---

::: phydrax.transport.PositiveFeatureSinkhornResult
    options:
        show_root_heading: true
        show_source: false
