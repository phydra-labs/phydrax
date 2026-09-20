#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-crystal molecular-to-laboratory orientation."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ..atomic_quantum._angular_momentum import cartesian_rotation_zyz


class SingleCrystalOrientation(StrictModule):
    """Active ZYZ orientation ``Rz(alpha) Ry(beta) Rz(gamma)`` in radians."""

    euler_angles_rad: Array
    rotation: Array
    orthogonality_residual: Array
    determinant_residual: Array
    orientation_id: str = eqx.field(static=True)

    def __init__(
        self,
        euler_angles_rad: ArrayLike = (0.0, 0.0, 0.0),
        /,
        *,
        orientation_id: str = "single-crystal",
    ):
        angles = np.asarray(euler_angles_rad, dtype=np.float64)
        if angles.shape != (3,) or np.any(~np.isfinite(angles)):
            raise ValueError("euler_angles_rad must be finite with shape (3,).")
        identifier = str(orientation_id)
        if not identifier:
            raise ValueError("orientation_id must be nonempty.")
        angles_array = jnp.asarray(angles)
        rotation = cartesian_rotation_zyz(angles_array)
        identity = jnp.eye(3, dtype=rotation.dtype)
        self.euler_angles_rad = angles_array
        self.rotation = rotation
        self.orthogonality_residual = jnp.max(jnp.abs(rotation @ rotation.T - identity))
        determinant = (
            rotation[0, 0]
            * (rotation[1, 1] * rotation[2, 2] - rotation[1, 2] * rotation[2, 1])
            - rotation[0, 1]
            * (rotation[1, 0] * rotation[2, 2] - rotation[1, 2] * rotation[2, 0])
            + rotation[0, 2]
            * (rotation[1, 0] * rotation[2, 1] - rotation[1, 1] * rotation[2, 0])
        )
        self.determinant_residual = jnp.abs(determinant - 1.0)
        self.orientation_id = identifier

    def vector_to_lab(self, vector_molecular: ArrayLike, /) -> Array:
        """Actively rotate one molecular-frame Cartesian vector into the lab."""

        vector = jnp.asarray(vector_molecular)
        if vector.shape != (3,):
            raise ValueError("vector_molecular must have shape (3,).")
        return ein.contract("ab,b->a", self.rotation, vector, backend="jax")

    def tensor_to_lab(self, tensor_molecular: ArrayLike, /) -> Array:
        """Actively rotate a molecular-frame second-rank Cartesian tensor."""

        tensor = jnp.asarray(tensor_molecular)
        if tensor.shape != (3, 3):
            raise ValueError("tensor_molecular must have shape (3, 3).")
        return ein.contract(
            "ai,ij,bj->ab",
            self.rotation,
            tensor,
            self.rotation,
            backend="jax",
        )


__all__ = ["SingleCrystalOrientation"]
