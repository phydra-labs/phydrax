#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


Z4C_CHANNEL_COUNT = 25
SYMMETRIC_COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def pack_symmetric(matrix: ArrayLike, /) -> Array:
    """Pack component-first symmetric 3x3 tensors as ``xx,xy,xz,yy,yz,zz``."""

    value = jnp.asarray(matrix)
    if value.ndim < 2 or value.shape[:2] != (3, 3):
        raise ValueError("matrix must have component-first shape (3,3,...).")
    return jnp.stack(tuple(value[i, j] for i, j in SYMMETRIC_COMPONENTS), axis=0)


def unpack_symmetric(packed: ArrayLike, /) -> Array:
    """Unpack ``xx,xy,xz,yy,yz,zz`` into a component-first symmetric tensor."""

    value = jnp.asarray(packed)
    if value.ndim < 1 or value.shape[0] != 6:
        raise ValueError("packed symmetric tensors must have shape (6,...).")
    result = jnp.zeros((3, 3) + value.shape[1:], dtype=value.dtype)
    for channel, (i, j) in enumerate(SYMMETRIC_COMPONENTS):
        result = result.at[i, j].set(value[channel])
        result = result.at[j, i].set(value[channel])
    return result


class Z4cState(StrictModule):
    """Fixed-shape packed conformal Z4c state on one grid topology.

    The leading 25 channels are chi, six conformal-metric components, K-hat,
    six conformal trace-free extrinsic-curvature components, Theta, three
    conformal connection functions, lapse, shift, and Gamma-driver fields.
    Spatial axes are always the last three axes.
    """

    values: Array
    grid_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, /, *, grid_id: str):
        value = jnp.asarray(values)
        identifier = str(grid_id)
        if value.ndim != 4 or value.shape[0] != Z4C_CHANNEL_COUNT:
            raise ValueError("Z4c values must have shape (25,nx,ny,nz).")
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("Z4c values must have a real floating-point dtype.")
        if not identifier:
            raise ValueError("grid_id must be non-empty.")
        self.values = value
        self.grid_id = identifier

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.values.shape[1:]

    @property
    def chi(self) -> Array:
        return self.values[0]

    @property
    def conformal_metric(self) -> Array:
        return unpack_symmetric(self.values[1:7])

    @property
    def k_hat(self) -> Array:
        return self.values[7]

    @property
    def conformal_extrinsic_curvature(self) -> Array:
        return unpack_symmetric(self.values[8:14])

    @property
    def theta(self) -> Array:
        return self.values[14]

    @property
    def conformal_connection(self) -> Array:
        return self.values[15:18]

    @property
    def lapse(self) -> Array:
        return self.values[18]

    @property
    def shift(self) -> Array:
        return self.values[19:22]

    @property
    def shift_driver(self) -> Array:
        return self.values[22:25]

    @property
    def trace_extrinsic_curvature(self) -> Array:
        return self.k_hat + 2.0 * self.theta

    @property
    def physical_metric(self) -> Array:
        return self.conformal_metric / self.chi[None, None, ...]

    @property
    def physical_extrinsic_curvature(self) -> Array:
        trace_term = (
            self.conformal_metric * self.trace_extrinsic_curvature[None, None, ...] / 3.0
        )
        return (self.conformal_extrinsic_curvature + trace_term) / self.chi[
            None, None, ...
        ]

    def with_values(self, values: ArrayLike, /) -> Z4cState:
        return Z4cState(values, grid_id=self.grid_id)


def make_z4c_state(
    chi: ArrayLike,
    conformal_metric: ArrayLike,
    k_hat: ArrayLike,
    conformal_extrinsic_curvature: ArrayLike,
    theta: ArrayLike,
    conformal_connection: ArrayLike,
    lapse: ArrayLike,
    shift: ArrayLike,
    shift_driver: ArrayLike,
    /,
    *,
    grid_id: str,
) -> Z4cState:
    """Validate fields and assemble the canonical packed Z4c representation."""

    chi_ = jnp.asarray(chi)
    if chi_.ndim != 3:
        raise ValueError("chi must have shape (nx,ny,nz).")
    shape = chi_.shape
    scalar_fields = {
        "k_hat": jnp.asarray(k_hat),
        "theta": jnp.asarray(theta),
        "lapse": jnp.asarray(lapse),
    }
    for name, value in scalar_fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
    tensor_fields = {
        "conformal_metric": jnp.asarray(conformal_metric),
        "conformal_extrinsic_curvature": jnp.asarray(conformal_extrinsic_curvature),
    }
    for name, value in tensor_fields.items():
        if value.shape != (3, 3) + shape:
            raise ValueError(f"{name} must have shape {(3, 3) + shape}.")
    vector_fields = {
        "conformal_connection": jnp.asarray(conformal_connection),
        "shift": jnp.asarray(shift),
        "shift_driver": jnp.asarray(shift_driver),
    }
    for name, value in vector_fields.items():
        if value.shape != (3,) + shape:
            raise ValueError(f"{name} must have shape {(3,) + shape}.")
    arrays = (
        chi_,
        *scalar_fields.values(),
        *tensor_fields.values(),
        *vector_fields.values(),
    )
    if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in arrays):
        raise TypeError("Z4c fields must be real; complex values are not supported.")
    dtype = jnp.result_type(*arrays)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.result_type(dtype, jnp.float32)
    values = jnp.concatenate(
        (
            chi_.astype(dtype)[None, ...],
            pack_symmetric(tensor_fields["conformal_metric"]).astype(dtype),
            scalar_fields["k_hat"].astype(dtype)[None, ...],
            pack_symmetric(tensor_fields["conformal_extrinsic_curvature"]).astype(dtype),
            scalar_fields["theta"].astype(dtype)[None, ...],
            vector_fields["conformal_connection"].astype(dtype),
            scalar_fields["lapse"].astype(dtype)[None, ...],
            vector_fields["shift"].astype(dtype),
            vector_fields["shift_driver"].astype(dtype),
        ),
        axis=0,
    )
    return Z4cState(values, grid_id=grid_id)


def flat_z4c_state(
    grid_shape: tuple[int, int, int],
    /,
    *,
    grid_id: str,
    dtype=jnp.float32,
) -> Z4cState:
    """Return exact Cartesian Minkowski data in the packed Z4c representation."""

    shape = tuple(grid_shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("grid_shape must contain three positive extents.")
    scalar = jnp.ones(shape, dtype=dtype)
    zero = jnp.zeros(shape, dtype=dtype)
    identity = jnp.broadcast_to(
        jnp.eye(3, dtype=dtype)[..., None, None, None], (3, 3) + shape
    )
    vector = jnp.zeros((3,) + shape, dtype=dtype)
    tensor = jnp.zeros((3, 3) + shape, dtype=dtype)
    return make_z4c_state(
        scalar,
        identity,
        zero,
        tensor,
        zero,
        vector,
        scalar,
        vector,
        vector,
        grid_id=grid_id,
    )


__all__ = [
    "SYMMETRIC_COMPONENTS",
    "Z4C_CHANNEL_COUNT",
    "Z4cState",
    "flat_z4c_state",
    "make_z4c_state",
    "pack_symmetric",
    "unpack_symmetric",
]
