#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
from jaxtyping import Array, ArrayLike


def squared_relu(x: ArrayLike, /) -> Array:
    r"""Squared rectified linear unit $\max(x, 0)^2$.

    It is convex and nondecreasing, $C^1$, and piecewise quadratic.
    """
    positive = jax.nn.relu(x)
    return positive * positive
