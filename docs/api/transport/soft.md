# Differentiable ordering

The soft-order family builds one entropic monotone coupling between observed values and
ordered target mass bins. Every operation is derived from that coupling; none replaces
the hard combinatorial operator in the forward pass.

For arrays, `axis` is an integer. For `coordax.Field`, it may be a named dimension and
the returned field preserves caller dimension order. Weights must broadcast exactly
under the selected axis contract.

## Core coupling

::: phydrax.transport.soft_order_transport

## Sorting and ranking

::: phydrax.transport.soft_sort

---

::: phydrax.transport.soft_rank

---

::: phydrax.transport.soft_sort_by

## Top-k relaxations

::: phydrax.transport.soft_topk_mask

---

::: phydrax.transport.soft_topk_values

`soft_topk_mask(values, k)` returns fractional membership with total mass `k`. The
boundary cases are exact: `k=0` returns zeros and `k=axis_size` returns ones.
`soft_topk_values` returns values in the largest `k` ordered bins, with the selected
axis retained at size `k`.

## Quantiles and quantization

::: phydrax.transport.soft_quantile

---

::: phydrax.transport.soft_quantile_normalize

---

::: phydrax.transport.soft_quantize

Quantiles may be scalar or array-valued and must lie in the closed unit interval.
Caller quantile order is preserved. Exact endpoints return the sample minimum and
maximum. `soft_quantile_normalize` maps through a differentiable empirical CDF-like
rank; `soft_quantize` maps those normalized levels onto supplied target levels.

## Regularization and gradients

Smaller `epsilon` approaches harder order but makes Sinkhorn iterations more
ill-conditioned; larger `epsilon` smooths gradients and increases approximation bias.
Do not report a soft sort, rank, quantile, or top-k output as exact. Validate the
chosen regularization with hard-order approximation error, monotonicity, range,
finite-gradient, and convergence diagnostics on the workload's actual scale.
