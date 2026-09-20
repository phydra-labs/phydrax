#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


@dataclass(frozen=True, slots=True)
class PoleResidueModel:
    poles: Array
    residues: Array
    direct: Array

    def evaluate(self, angular_frequency_rad_s: ArrayLike):
        s = 1j * jnp.asarray(angular_frequency_rad_s)
        return self.direct + jnp.sum(self.residues / (s[..., None] - self.poles), axis=-1)


def fit_fixed_poles(
    angular_frequency_rad_s: ArrayLike, response: ArrayLike, poles: ArrayLike, /
) -> PoleResidueModel:
    omega = jnp.asarray(angular_frequency_rad_s)
    values = jnp.asarray(response)
    poles_ = jnp.asarray(poles)
    design = jnp.concatenate(
        (
            1 / (1j * omega[:, None] - poles_[None, :]),
            jnp.ones((omega.size, 1), dtype=complex),
        ),
        axis=1,
    )
    normal = jnp.conj(design).T @ design
    right = jnp.conj(design).T @ values
    space = ArraySpace((normal.shape[0],), dtype=normal.dtype)
    coeff = solve(
        LinearSystem(DenseLinearOperator(normal, source=space, target=space)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    ).value
    return PoleResidueModel(poles_, coeff[:-1], coeff[-1])


__all__ = ["PoleResidueModel", "fit_fixed_poles"]
