#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ._algebra_spaces import AlgebraArraySpace
from ._operators import AbstractLinearOperator, FunctionLinearOperator
from ._spaces import ArraySpace


def apply_real_map_componentwise(action, value: ArrayLike, /) -> Array:
    """Extend one real-coordinate action to a native-complex value."""
    array = jnp.asarray(value)
    if not jnp.iscomplexobj(array):
        return jnp.asarray(action(array))
    return jax.lax.complex(
        jnp.asarray(action(jnp.real(array))),
        jnp.asarray(action(jnp.imag(array))),
    ).astype(array.dtype)


def lift_real_operator_to_algebra(
    operator: AbstractLinearOperator,
    algebra_space: AlgebraArraySpace,
    /,
) -> FunctionLinearOperator:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must implement AbstractLinearOperator.")
    if not isinstance(algebra_space, AlgebraArraySpace):
        raise TypeError("algebra_space must be AlgebraArraySpace.")
    if not isinstance(operator.source, ArraySpace) or not isinstance(
        operator.target, ArraySpace
    ):
        raise TypeError(
            "Algebra coordinate lifting initially requires ArraySpace operators."
        )
    if operator.source.shape != algebra_space.base_shape:
        raise ValueError("Operator source shape must match the algebra base shape.")
    target_space = AlgebraArraySpace(
        operator.target.shape,
        algebra_space.algebra,
        algebra_axis=algebra_space.algebra_axis,
        dtype=algebra_space.dtype,
    )
    axis = algebra_space.algebra_axis

    def action(value):
        coordinates = jnp.moveaxis(value, axis, 0)
        applied = jax.vmap(operator.mv)(coordinates)
        return jnp.moveaxis(applied, 0, axis)

    def transpose_action(value):
        coordinates = jnp.moveaxis(value, axis, 0)
        applied = jax.vmap(operator.transpose_mv)(coordinates)
        return jnp.moveaxis(applied, 0, axis)

    return FunctionLinearOperator(
        action,
        source=algebra_space,
        target=target_space,
        transpose_action=transpose_action,
        operator_id=canonical_fingerprint(
            {
                "kind": "algebra-coordinate-lifted-operator-v1",
                "operator": operator.operator_id,
                "algebra": algebra_space.algebra.algebra_id,
                "source": algebra_space.space_id,
                "target": target_space.space_id,
            }
        ),
    )


def complexify_real_operator(
    operator: AbstractLinearOperator,
    /,
    *,
    complex_dtype: Any,
) -> FunctionLinearOperator:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must implement AbstractLinearOperator.")
    if not isinstance(operator.source, ArraySpace) or not isinstance(
        operator.target, ArraySpace
    ):
        raise TypeError("Complexification initially requires ArraySpace operators.")
    dtype = jnp.dtype(complex_dtype)
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        raise TypeError("complex_dtype must be complex64 or complex128.")
    source = ArraySpace(operator.source.shape, dtype=dtype)
    target = ArraySpace(operator.target.shape, dtype=dtype)

    def action(value):
        return apply_real_map_componentwise(operator.mv, value)

    def transpose_action(value):
        return apply_real_map_componentwise(operator.transpose_mv, value)

    return FunctionLinearOperator(
        action,
        source=source,
        target=target,
        transpose_action=transpose_action,
        operator_id=canonical_fingerprint(
            {
                "kind": "complexified-real-operator-v1",
                "operator": operator.operator_id,
                "dtype": str(dtype),
            }
        ),
    )


__all__ = [
    "apply_real_map_componentwise",
    "complexify_real_operator",
    "lift_real_operator_to_algebra",
]
