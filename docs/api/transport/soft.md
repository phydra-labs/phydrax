# Differentiable ordering

The soft-order family builds one entropic monotone coupling between observed values and
ordered target mass bins. Every operation is derived from that coupling. It is a
separate relaxed model, not a hard operator with substituted gradients.

For arrays, `axis` is an integer. For `coordax.Field`, it is a named dimension and the
returned field preserves caller dimension order. Weights must broadcast exactly under
the selected axis contract. Zero-weight atoms remain shape-present but are inert.

## Core coupling

::: phydrax.transport.soft_order_transport

### Canonical dimensionless geometry

Soft ordering does not transport values in their raw physical units. For normalized
source probabilities `p`, it computes the weighted center `μ`, the weighted scale
`σ = sqrt(weighted_variance + machine_epsilon)`, and source locations
`sigmoid((value - μ) / σ)`. Target locations are cumulative-probability bin midpoints.
The source provenance is
`soft-order-source:weighted-standardize-sigmoid`; the target provenance is
`soft-order-target:probability-midpoints`.

This canonicalization makes the order map translation-equivariant and, within floating
point tolerance, positive-affine-equivariant. Negative scaling reverses order.
`epsilon` is dimensionless in this canonical order geometry. Do not interpret it as the
ground-cost regularization for raw physical coordinates. Changing one candidate changes
the weighted center and scale, and can therefore change every relaxed location and
gradient.

There is deliberately no numerical “unsquash.” The returned result retains the
canonical source and target supports, masses, provenance, and coupling; physical
sorted values are recovered by applying that coupling to the original values as a
payload through `result.barycentric_source_to_target(values)`. Without the original
payload, the result makes no physical-value reconstruction claim and does not attempt
an unstable inverse sigmoid.

The convenience `epsilon=` value configures the default solver. When `solver=` is
supplied, that solver owns the effective epsilon, iteration budget, tolerance,
block size, and differentiation behavior.

## Sorting and ranking

::: phydrax.transport.soft_sort

---

::: phydrax.transport.soft_rank

---

::: phydrax.transport.soft_sort_by

`soft_rank` returns zero-based ascending barycentric ranks. Its weighting and
coupling-conservation semantics are not interchangeable with one-based pairwise
logistic ranks used by `phydrax.ml.soft_ranks`.

## Fast unweighted ordering

::: phydrax.transport.fast_soft_sort

---

::: phydrax.transport.fast_soft_rank

The `fast_soft_*` family is a separate unweighted algorithm, not a `method=` option on
the Sinkhorn API. It follows the L2 permutahedron relaxation and pool-adjacent-violators
construction of [Blondel et al. (2020)](https://arxiv.org/abs/2002.08871). Sorting
dominates its asymptotic cost: it uses `O(n log n)` work and `O(n)` temporary state
rather than constructing an `n`-by-`n` transport problem.

Each row is centered and variance-standardized before relaxation. `temperature` is
therefore dimensionless; larger values are smoother and smaller values approach hard
ordering. Constant rows are handled without dividing by zero. The operators preserve
permutation symmetry and positive-affine equivariance within floating-point tolerance.
`fast_soft_sort` preserves the unweighted sum and returns monotone values.
`fast_soft_rank` returns zero-based ascending ranks in `[0, n - 1]` whose unweighted
sum is `n * (n - 1) / 2`.

The relaxation is continuous and piecewise smooth, not globally smooth. Its PAV active
partition can change at chamber boundaries, and sufficiently separated values can
enter an exactly ordered region with zero rank derivative. Tune `temperature` against
both hard-order error and gradient utility on the actual data.

Use this family only when every atom has equal importance and only sorted values or
ranks are needed. It has no weights, transport plan, barycentric payload map,
convergence iterations, or marginal-residual diagnostics. Use the Sinkhorn family
when any of those semantics matter. Both raw arrays with integer axes and
`coordax.Field` values with named axes are supported.

```python
import jax.numpy as jnp

import phydrax as phx

values = jnp.asarray([3.0, 1.0, 4.0, 2.0])

sorted_values = phx.transport.fast_soft_sort(values, temperature=0.5)
zero_based_ranks = phx.transport.fast_soft_rank(values, temperature=2.0)
```

## Top-k relaxations

::: phydrax.transport.soft_topk_mask

---

::: phydrax.transport.soft_topk_values

`soft_topk_mask(values, k)` maps the indicator of the largest `k`
equal-probability target bins back to source atoms. With uniform source weights, its
fractional memberships sum to `k`. With normalized nonuniform source weights `p`, the
conserved statement is `sum(p * mask) = k / axis_size`, not an unweighted cardinality
claim. The boundary cases are exact: `k=0` returns zeros and `k=axis_size` returns
ones. `soft_topk_values` returns values in the largest `k` ordered bins, with the
selected axis retained at size `k`.

## Quantiles and quantization

::: phydrax.transport.soft_quantile

---

::: phydrax.transport.soft_quantile_normalize

---

::: phydrax.transport.soft_quantize

Quantiles may be scalar or array-valued and must lie in the closed unit interval.
Caller quantile order is preserved. Interior quantiles interpolate relaxed sorted
values. Exact endpoints return the active sample minimum and maximum, so their
derivatives are only almost-everywhere defined and are nondifferentiable at relevant
ties. `soft_quantile_normalize` maps through the same relaxed empirical order;
`soft_quantize` maps normalized levels through learned ordered barycentric levels.

## Regularization, diagnostics, and method choice

Smaller `epsilon` approaches harder order but makes Sinkhorn iterations more
ill-conditioned; larger `epsilon` broadens the coupling and usually distributes
gradients across more atoms, at the cost of greater approximation bias.
`soft_order_transport` returns the full `SinkhornResult`, including convergence,
marginal residuals, potentials, and transport provenance. Convenience functions return
only transformed values, but still fail explicitly when the solve does not converge.
They do not retain full solver evidence for later inspection.

Do not report a soft sort, rank, quantile, or top-k output as exact. Validate the
chosen regularization with hard-order approximation error, monotonicity, range,
finite-gradient, convergence, and final scientific metrics on the actual workload.

Use pairwise logistic ranks in `phydrax.ml` for lightweight unweighted ML losses. Use
this Sinkhorn family when nonuniform empirical weights, zero-mass support, named
scientific axes, or a shared monotone coupling are part of the model. Use hard JAX
ordering for exact reporting and guarantee-bearing statistics. Hardening is terminal:
Phydrax installs no straight-through derivative.
