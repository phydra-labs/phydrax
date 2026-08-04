#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax.numpy as jnp

from ...domain._function import DomainFunction
from ..integral._batch_ops import integral, mean


def spatial_mean(
    u: DomainFunction,
    target_or_realization: Any,
    plan: Any = None,
    /,
    **kwargs: Any,
) -> cx.Field:
    """Integrate a field under normalized target-measure semantics."""
    return mean(u, target_or_realization, plan, **kwargs)


def spatial_inner_product(
    u: DomainFunction,
    v: DomainFunction,
    target_or_realization: Any,
    plan: Any = None,
    /,
    **kwargs: Any,
) -> cx.Field:
    """Integrate the pointwise Euclidean/Frobenius product of two fields."""
    joined = u.domain.join(v.domain)
    u2 = u.promote(joined)
    v2 = v.promote(joined)
    deps = tuple(
        label for label in joined.labels if label in u2.deps or label in v2.deps
    )
    indices = {label: index for index, label in enumerate(deps)}
    u_positions = tuple(indices[label] for label in u2.deps)
    v_positions = tuple(indices[label] for label in v2.deps)

    def _inner(*args, key=None, **inner_kwargs):
        left = jnp.asarray(
            u2.func(*(args[index] for index in u_positions), key=key, **inner_kwargs)
        )
        right = jnp.asarray(
            v2.func(*(args[index] for index in v_positions), key=key, **inner_kwargs)
        )
        return jnp.sum(jnp.conj(left) * right)

    integrand = DomainFunction(domain=joined, deps=deps, func=_inner, metadata={})
    return integral(integrand, target_or_realization, plan, **kwargs)


def spatial_lp_norm(
    u: DomainFunction,
    target_or_realization: Any,
    plan: Any = None,
    /,
    *,
    p: float = 2.0,
    **kwargs: Any,
) -> cx.Field:
    """Integrate the pointwise Euclidean norm to obtain an L-p norm."""
    if p <= 0:
        raise ValueError("p must be positive.")
    exponent = float(p)

    def _power(*args, key=None, **inner_kwargs):
        value = jnp.asarray(u.func(*args, key=key, **inner_kwargs))
        return jnp.power(jnp.linalg.norm(value.reshape((-1,))), exponent)

    integrand = DomainFunction(
        domain=u.domain,
        deps=u.deps,
        func=_power,
        metadata={},
    )
    value = integral(integrand, target_or_realization, plan, **kwargs)
    return cx.Field(
        jnp.power(jnp.asarray(value.data), 1.0 / exponent), dims=value.dims
    )


def spatial_l2_norm(
    u: DomainFunction,
    target_or_realization: Any,
    plan: Any = None,
    /,
    **kwargs: Any,
) -> cx.Field:
    """Compute the L-2 norm under a typed integration execution."""
    return spatial_lp_norm(
        u, target_or_realization, plan, p=2.0, **kwargs
    )


__all__ = [
    "spatial_inner_product",
    "spatial_l2_norm",
    "spatial_lp_norm",
    "spatial_mean",
]
