# Wasserstein barycenters

PhydraX provides separate native contracts for fixed- and free-support balanced
barycenters. Both consume the finite measure contracts already owned by
`phydrax.integration`; neither introduces a public transport-measure hierarchy.
Physical mass, masks, padded atoms, event shape, measure weights, convergence, and
provenance remain explicit.

## Fixed support

`FixedSupportBarycenterProblem` accepts a nonempty tuple of
`DiscreteMeasureTarget`, `WeightedSampleTarget`, or their external-measure
`IntegrationRealization`, plus a declared finite support. Every measure and the
support must have common physical mass. `measure_weights` must be finite, strictly
positive, and sum to one; the constructor never repairs or normalizes them.
Encoders may differ, but their encoded feature sizes must agree.

```python
import coordax as cx
import jax.numpy as jnp
import phydrax as phx


def law(points, weights, provenance):
    return phx.integration.discrete(
        jnp.asarray(points),
        cx.Field(jnp.asarray(weights), dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance=provenance,
    )


first = law([[-1.0], [1.0]], [0.5, 0.5], "ensemble-a")
second = law([[0.0], [2.0]], [0.25, 0.75], "ensemble-b")
support = law([[-0.5], [0.5], [1.5]], [0.2, 0.5, 0.3], "declared-grid")
problem = phx.transport.fixed_support_barycenter_problem(
    (first, second),
    support,
    measure_weights=jnp.asarray([0.4, 0.6]),
    cost=phx.transport.SquaredEuclideanCost(),
)
solver = phx.transport.SinkhornBarycenter(
    0.2,
    max_iterations=500,
    tolerance=1e-8,
    check_every=5,
    store_history=True,
)
result = phx.transport.require_barycenter_converged(solver(problem))
barycenter = result.as_target()
```

The per-measure objective uses the standard finite entropic-barycenter convention
`sum(coupling * cost) + epsilon * sum(coupling * log(coupling))`, scaled by
physical mass. This is stated separately from pairwise `SinkhornResult`, whose
regularization is reported relative to its declared product measure.

The log-domain solve pads unequal input atom counts and masks every padded entry.
`block_size=` changes only the exact reduction schedule: dense and blockwise execution
solve the same declared finite problem and both report `approximate=False`.
`BarycenterResult` retains the common support probabilities, every measure and support
potential, padded or individually unpadded physical couplings, per-measure transport
and regularization objectives, their weighted aggregate, residual histories, statuses,
and execution provenance.

::: phydrax.transport.FixedSupportBarycenterProblem

---

::: phydrax.transport.fixed_support_barycenter_problem

---

::: phydrax.transport.SinkhornBarycenter

---

::: phydrax.transport.BarycenterResult

---

::: phydrax.transport.BarycenterDiagnostics

---

::: phydrax.transport.BarycenterProblemProvenance

---

::: phydrax.transport.BarycenterProvenance

## Free support

`FreeSupportBarycenter` is an explicit outer alternating solver. The support in the
problem is the required initialization; it is never synthesized from the inputs. Each
outer step solves the fixed-support problem and applies the coupling-weighted quadratic
barycentric coordinate update. Consequently free support is restricted to
`SquaredEuclideanCost` and `WeightedSquaredEuclideanCost`.

```python
outer = phx.transport.FreeSupportBarycenter(
    solver,
    max_iterations=20,
    tolerance=1e-6,
    collapse_tolerance=1e-10,
)
local_result = outer(problem)
```

The result is a local optimum, not a globally certified support search. Its provenance
records the explicit initialization and local-optimization method. `inner_results`
retains every fixed-support solve, while outer objective, displacement, inner-status,
stagnation, and support-collapse histories remain available. A collapsed support stops
with `TransportStatus.SUPPORT_COLLAPSE`; atoms are not merged, jittered, clipped, or
silently deleted.

::: phydrax.transport.FreeSupportBarycenter

---

::: phydrax.transport.FreeSupportBarycenterResult

---

::: phydrax.transport.FreeSupportBarycenterDiagnostics

---

::: phydrax.transport.FreeSupportBarycenterProvenance

---

::: phydrax.transport.require_barycenter_converged

## Scientific aggregation and objectives

`phydrax.uq.aggregate_transport_barycenter` and
`aggregate_free_support_transport_barycenter` return the aggregate
`DiscreteMeasureTarget` together with complete native transport results. They are
appropriate when several finite predictive or posterior laws have a scientifically
meaningful Wasserstein aggregate.

`phydrax.terms.BarycenterObjectiveTerm` is the composable scalar training integration.
Its builder returns a complete `FixedSupportBarycenterProblem`, so model code controls
which laws and support are compared. The term rejects nonconverged inner transport
instead of optimizing a failed numerical result.

::: phydrax.uq.aggregate_transport_barycenter

---

::: phydrax.uq.aggregate_free_support_transport_barycenter

---

::: phydrax.uq.TransportBarycenterAggregationResult

---

::: phydrax.uq.FreeSupportTransportBarycenterAggregationResult

---

::: phydrax.terms.BarycenterObjectiveTerm
