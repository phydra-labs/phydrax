#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class ComplexMode:
    eigenvalue: Array
    right: Array
    left: Array

    def biorthogonal_normalize(self):
        scale = jnp.vdot(self.left, self.right)
        return ComplexMode(self.eigenvalue, self.right / scale, self.left)


def modal_overlap(left_modes: ArrayLike, right_modes: ArrayLike):
    left = jnp.asarray(left_modes)
    right = jnp.asarray(right_modes)
    return jnp.abs(jnp.conj(left).T @ right) ** 2 / (
        jnp.sum(jnp.abs(left) ** 2, axis=0)[:, None]
        * jnp.sum(jnp.abs(right) ** 2, axis=0)[None, :]
    )


__all__ = ["ComplexMode", "modal_overlap"]
