#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


def _right_solve(matrix, right):
    a = jnp.asarray(matrix)
    b = jnp.asarray(right)
    n = a.shape[-1]
    space = ArraySpace((n,), dtype=a.dtype)
    op = DenseLinearOperator(jnp.swapaxes(a, -1, -2), source=space, target=space)
    cols = tuple(
        solve(
            LinearSystem(op),
            jnp.swapaxes(b, -1, -2)[:, i],
            policy=LinearSolvePolicy(DenseLU()),
        ).value
        for i in range(n)
    )
    return jnp.swapaxes(jnp.stack(cols, axis=1), -1, -2)


def s_to_z(scattering: ArrayLike, reference_impedance_ohm: float, /):
    s = jnp.asarray(scattering)
    identity = jnp.eye(s.shape[-1], dtype=s.dtype)
    return float(reference_impedance_ohm) * _right_solve(identity - s, identity + s)


def z_to_s(impedance: ArrayLike, reference_impedance_ohm: float, /):
    z = jnp.asarray(impedance)
    identity = jnp.eye(z.shape[-1], dtype=z.dtype)
    return _right_solve(
        z + float(reference_impedance_ohm) * identity,
        z - float(reference_impedance_ohm) * identity,
    )


__all__ = ["s_to_z", "z_to_s"]
