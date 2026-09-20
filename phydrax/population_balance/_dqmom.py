#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def quadrature_moment_rates(
    nodes: ArrayLike,
    weights: ArrayLike,
    node_rates: ArrayLike,
    weight_rates: ArrayLike,
    orders: ArrayLike,
    /,
):
    x = jnp.asarray(nodes)
    w = jnp.asarray(weights)
    dx = jnp.asarray(node_rates)
    dw = jnp.asarray(weight_rates)
    k = jnp.asarray(orders)
    return jnp.sum(
        dw[None, :] * x[None, :] ** k[:, None]
        + jnp.where(
            k[:, None] > 0,
            k[:, None] * w[None, :] * x[None, :] ** (k[:, None] - 1) * dx[None, :],
            0,
        ),
        axis=1,
    )


__all__ = ["quadrature_moment_rates"]
