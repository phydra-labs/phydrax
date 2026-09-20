#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class GrainStructure:
    grain_id: Array
    orientation_quaternion: Array
    phase_id: Array

    @classmethod
    def create(
        cls, grain_id: ArrayLike, orientation_quaternion: ArrayLike, phase_id: ArrayLike
    ):
        value = cls(
            jnp.asarray(grain_id, dtype=jnp.int32),
            jnp.asarray(orientation_quaternion),
            jnp.asarray(phase_id, dtype=jnp.int32),
        )
        if (
            value.grain_id.shape != value.phase_id.shape
            or value.orientation_quaternion.shape
            != (
                *value.grain_id.shape,
                4,
            )
        ):
            raise ValueError("Grain arrays do not align.")
        if not bool(
            jnp.allclose(jnp.linalg.norm(value.orientation_quaternion, axis=-1), 1)
        ):
            raise ValueError("Orientations must be unit quaternions.")
        return value


__all__ = ["GrainStructure"]
