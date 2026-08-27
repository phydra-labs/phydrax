#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._algebra_spaces import AlgebraArraySpace
from ._operators import FunctionLinearOperator


AlgebraActionSide: TypeAlias = Literal["left", "right"]


class _BoundAlgebraAction(StrictModule):
    product: Any
    multiplier: Array
    side: AlgebraActionSide = eqx.field(static=True)

    def __init__(self, product: Any, multiplier: Array, side: AlgebraActionSide, /):
        self.product = product
        self.multiplier = multiplier
        self.side = side

    def __call__(self, value: Array, /) -> Array:
        if self.side == "left":
            return self.product(self.multiplier, value)
        return self.product(value, self.multiplier)


def _broadcast_multiplier(
    multiplier: ArrayLike,
    space: AlgebraArraySpace,
    /,
) -> Array:
    value = jnp.asarray(multiplier)
    if jnp.iscomplexobj(value):
        raise TypeError("Algebra regular-action multipliers must use real coordinates.")
    if not np.issubdtype(np.dtype(value.dtype), np.floating):
        raise TypeError("Algebra regular-action multipliers must use floating dtype.")
    if np.dtype(value.dtype) != space.dtype:
        raise TypeError(
            f"Algebra regular-action multiplier must have dtype {space.dtype}; "
            f"got {value.dtype}."
        )
    dimension = space.algebra.coordinate_dimension
    if value.shape == (dimension,):
        shape = [1] * len(space.shape)
        shape[space.algebra_axis] = dimension
        value = value.reshape(tuple(shape))
    elif value.ndim != len(space.shape) or value.shape[space.algebra_axis] != dimension:
        raise ValueError(
            "Algebra regular-action multiplier must be one algebra element or expose "
            "the space's declared algebra axis."
        )
    if jnp.broadcast_shapes(value.shape, space.shape) != space.shape:
        raise ValueError(
            "Algebra regular-action multiplier does not broadcast to the declared space."
        )
    return value


def algebra_regular_action_operator(
    product: Any,
    multiplier: ArrayLike,
    space: AlgebraArraySpace,
    /,
    *,
    side: AlgebraActionSide,
) -> FunctionLinearOperator:
    """Bind left or right pointwise multiplication as a real-linear operator."""
    from ..metrix.algebra import AlgebraProductPlan

    if not isinstance(product, AlgebraProductPlan):
        raise TypeError("product must be an AlgebraProductPlan.")
    if not isinstance(space, AlgebraArraySpace):
        raise TypeError("space must be an AlgebraArraySpace.")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    product.algebra.require_compatible(space.algebra)
    product_axis = product.layout.algebra_axis
    if product_axis < 0:
        product_axis += len(space.shape)
    if product_axis != space.algebra_axis:
        raise ValueError("Product layout and algebra space use different algebra axes.")
    multiplier_ = _broadcast_multiplier(multiplier, space)
    action = _BoundAlgebraAction(product, multiplier_, side)
    operator_id = canonical_fingerprint(
        {
            "kind": "algebra-regular-action-v1",
            "product": product.plan_id,
            "space": space.space_id,
            "side": side,
            "multiplier": array_tree_fingerprint(multiplier_),
        }
    )
    return FunctionLinearOperator(
        action,
        source=space,
        target=space,
        operator_id=operator_id,
    )


__all__ = ["AlgebraActionSide", "algebra_regular_action_operator"]
