#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..stochastic._signature import PrimitiveBasis


RoughVectorFields: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
LiftedRoughVectorFields: TypeAlias = Callable[[Array, Array, Any], ArrayLike]


def lift_rough_vector_fields(
    vector_fields: RoughVectorFields,
    basis: PrimitiveBasis,
    time: ArrayLike,
    state: ArrayLike,
    args: Any,
    /,
    *,
    explicit_fields: LiftedRoughVectorFields | None = None,
) -> Array:
    """Evaluate standard Lyndon field brackets in the Davie word orientation.

    A tensor word ``ij`` acts as ``D V_j V_i``. Consequently the standard
    primitive bracket ``[u, v] = uv - vu`` is lifted as
    ``D V_v V_u - D V_u V_v``.
    """
    if not callable(vector_fields):
        raise TypeError("vector_fields must be callable.")
    if not isinstance(basis, PrimitiveBasis):
        raise TypeError("basis must be a PrimitiveBasis.")
    value = jnp.asarray(state)
    time_value = jnp.asarray(time)
    if explicit_fields is not None:
        if not callable(explicit_fields):
            raise TypeError("explicit_fields must be callable or None.")
        lifted = jnp.asarray(explicit_fields(time_value, value, args))
        expected = value.shape + (basis.size,)
        if lifted.shape != expected:
            raise ValueError(
                f"explicit_fields must return shape {expected}; got {lifted.shape}."
            )
        return lifted

    def field(index: int, argument: Array) -> Array:
        word = basis.words[index]
        children = basis.children[index]
        if children is None:
            fields = jnp.asarray(vector_fields(time_value, argument, args))
            expected = argument.shape + (basis.dimension,)
            if fields.shape != expected:
                raise ValueError(
                    f"vector_fields must return shape {expected}; got {fields.shape}."
                )
            return fields[..., word[0]]
        left_index, right_index = children
        left_value = field(left_index, argument)
        right_value = field(right_index, argument)
        right_after_left = jax.jvp(
            lambda point: field(right_index, point),
            (argument,),
            (left_value,),
        )[1]
        left_after_right = jax.jvp(
            lambda point: field(left_index, point),
            (argument,),
            (right_value,),
        )[1]
        return right_after_left - left_after_right

    return jnp.stack(tuple(field(index, value) for index in range(basis.size)), axis=-1)


__all__ = [
    "LiftedRoughVectorFields",
    "lift_rough_vector_fields",
]
