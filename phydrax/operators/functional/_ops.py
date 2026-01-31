#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ...domain._components import DomainComponent, DomainComponentUnion
from ...domain._function import DomainFunction
from ...domain._structure import CoordSeparableBatch, PointsBatch, QuadratureBatch
from ..integral._batch_ops import integral, mean


def spatial_mean(
    u: DomainFunction,
    batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...],
    /,
    *,
    component: DomainComponent | DomainComponentUnion,
    quadrature: QuadratureBatch | tuple[QuadratureBatch | None, ...] | None = None,
    over: str | tuple[str, ...] | None = "x",
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> cx.Field:
    r"""Estimate the spatial mean of a field.

    Computes the Monte Carlo / quadrature estimate of

    $$
    \langle u \rangle = \frac{1}{\mu(\Omega)}\int_{\Omega} u(x)\,d\mu(x),
    $$

    where $\Omega$ is the subset described by `component` and the integration axes
    are selected by `over` (default `"x"`).

    **Arguments:**

    - `u`: Integrand `DomainFunction`.
    - `batch`: `PointsBatch`/`CoordSeparableBatch` used for sampling (or a tuple for unions).
    - `component`: `DomainComponent` or `DomainComponentUnion` defining the integration region.
    - `quadrature`: Optional `QuadratureBatch` for paired sampling.
    - `over`: Which axes/labels to reduce over (defaults to `"x"`).
    - `key`: PRNG key forwarded to callables.
    - `kwargs`: Extra keyword arguments forwarded to `u` and component callables.

    **Returns:**

    - A `coordax.Field` containing the mean value (with remaining named axes, if any).
    """
    return mean(
        u, batch, component=component, quadrature=quadrature, over=over, key=key, **kwargs
    )


def spatial_inner_product(
    u: DomainFunction,
    v: DomainFunction,
    batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...],
    /,
    *,
    component: DomainComponent | DomainComponentUnion,
    quadrature: QuadratureBatch | tuple[QuadratureBatch | None, ...] | None = None,
    over: str | tuple[str, ...] | None = "x",
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> cx.Field:
    r"""Estimate an $L^2$-type inner product.

    Forms the pointwise Euclidean/Frobenius product and integrates:

    $$
    \langle u, v \rangle = \int_{\Omega} u(x)\cdot v(x)\,d\mu(x),
    $$

    where `u(x)·v(x)` is implemented as `sum(u * v)` over the value dimensions.
    Domains are joined before evaluation.

    **Arguments:**

    - `u`, `v`: Input `DomainFunction`s.
    - `batch`, `component`, `quadrature`, `over`, `key`, `kwargs`: As in `spatial_mean`.

    **Returns:**

    - A `coordax.Field` containing the scalar inner product value.
    """
    joined = u.domain.join(v.domain)
    u2 = u.promote(joined)
    v2 = v.promote(joined)

    deps = tuple(lbl for lbl in joined.labels if (lbl in u2.deps) or (lbl in v2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    v_pos = tuple(idx[lbl] for lbl in v2.deps)

    def _ip(*args, key=None, **kw):
        u_args = [args[i] for i in u_pos]
        v_args = [args[i] for i in v_pos]
        uu = jnp.asarray(u2.func(*u_args, key=key, **kw))
        vv = jnp.asarray(v2.func(*v_args, key=key, **kw))
        return jnp.sum(uu * vv)

    ip = DomainFunction(domain=joined, deps=deps, func=_ip, metadata={})
    return integral(
        ip,
        batch,
        component=component,
        quadrature=quadrature,
        over=over,
        key=key,
        **kwargs,
    )


def spatial_lp_norm(
    u: DomainFunction,
    batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...],
    /,
    *,
    p: float = 2.0,
    component: DomainComponent | DomainComponentUnion,
    quadrature: QuadratureBatch | tuple[QuadratureBatch | None, ...] | None = None,
    over: str | tuple[str, ...] | None = "x",
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> cx.Field:
    r"""Estimate an $L^p$ norm over space.

    Computes

    $$
    \|u\|_{L^p(\Omega)} = \left(\int_{\Omega} \|u(x)\|_2^p\,d\mu(x)\right)^{1/p},
    $$

    where $\|u(x)\|_2$ is the Euclidean norm of the (possibly vector/tensor) value at
    $x$, flattened.

    **Arguments:**

    - `u`: Input `DomainFunction`.
    - `batch`, `component`, `quadrature`, `over`, `key`, `kwargs`: As in `spatial_mean`.
    - `p`: Norm exponent $p>0$.

    **Returns:**

    - A `coordax.Field` containing the scalar $L^p$ norm value.
    """
    if p <= 0:
        raise ValueError("p must be positive.")
    p_ = float(p)

    def _pow_norm(*args, key=None, **kw):
        val = jnp.asarray(u.func(*args, key=key, **kw))
        flat = val.reshape((-1,))
        return jnp.power(jnp.linalg.norm(flat), p_)

    integrand = DomainFunction(domain=u.domain, deps=u.deps, func=_pow_norm, metadata={})
    val = integral(
        integrand,
        batch,
        component=component,
        quadrature=quadrature,
        over=over,
        key=key,
        **kwargs,
    )
    return cx.Field(jnp.power(jnp.asarray(val.data), 1.0 / p_), dims=val.dims)


def spatial_l2_norm(
    u: DomainFunction,
    batch: PointsBatch | CoordSeparableBatch | tuple[PointsBatch, ...],
    /,
    *,
    component: DomainComponent | DomainComponentUnion,
    quadrature: QuadratureBatch | tuple[QuadratureBatch | None, ...] | None = None,
    over: str | tuple[str, ...] | None = "x",
    key: Key[Array, ""] = DOC_KEY0,
    **kwargs: Any,
) -> cx.Field:
    r"""Estimate the $L^2$ norm over space.

    Equivalent to `spatial_lp_norm(..., p=2)`.
    """
    return spatial_lp_norm(
        u,
        batch,
        p=2.0,
        component=component,
        quadrature=quadrature,
        over=over,
        key=key,
        **kwargs,
    )


__all__ = [
    "spatial_inner_product",
    "spatial_l2_norm",
    "spatial_lp_norm",
    "spatial_mean",
]
