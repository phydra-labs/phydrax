#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from functools import partial
from math import ceil
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

from ._types import (
    BackendAvailability,
    BackendCapabilities,
    BackendDistributionCapabilities,
)


_SpatialDistanceBackend: TypeAlias = Literal["jax", "pallas"]


PALLAS_SPATIAL_CAPABILITIES = BackendCapabilities(
    backend="pallas-spatial",
    problem_kinds=(
        "spatial.distance_squared",
        "spatial.morton_knn_leaf",
        "spatial.morton_radius_leaf",
        "particle_fmm.p2p",
    ),
    execution="device",
    host_only=False,
    supports_matrix_free=True,
    supports_assembled=False,
    coordinate_dtypes=("float32", "float64"),
    distribution=BackendDistributionCapabilities(
        array_model="global",
        scopes=("single_device",),
        platforms=("cpu_interpret", "cuda", "tpu"),
        transformations=("jit", "jvp", "vjp"),
        communicator_model="jax",
    ),
)


def pallas_spatial_availability(*, interpret: bool = False) -> BackendAvailability:
    """Return whether the built-in Pallas spatial kernel can execute here."""
    platforms = tuple(device.platform for device in jax.devices())
    available = bool(interpret) or any(
        platform in ("cuda", "gpu", "tpu") for platform in platforms
    )
    reason = (
        "Pallas interpret mode is explicitly enabled"
        if interpret
        else "a Pallas GPU or TPU device is available"
        if available
        else "Pallas compilation requires CUDA/TPU; use interpret mode on CPU"
    )
    return BackendAvailability(
        capabilities=PALLAS_SPATIAL_CAPABILITIES,
        available=available,
        requirement="JAX Pallas on CUDA/TPU, or explicit CPU interpret mode",
        reason=reason,
        versions=(("jax", jax.__version__),),
    )


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _pallas_squared_norm(
    relative: jax.Array,
    block_size: int,
    interpret: bool,
) -> jax.Array:
    from jax.experimental import pallas as pl

    values = jnp.asarray(relative)
    dimension = int(values.shape[-1])
    leading_shape = tuple(int(size) for size in values.shape[:-1])
    item_count = values.size // dimension
    blocks = ceil(item_count / block_size)
    padded_count = blocks * block_size
    flattened = values.reshape((item_count, dimension))
    padded = jnp.pad(flattened, ((0, padded_count - item_count), (0, 0)))

    def kernel(value_ref, output_ref):
        value = value_ref[...]
        output_ref[...] = jnp.sum(value * value, axis=1)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded_count,), values.dtype),
        grid=(blocks,),
        in_specs=(pl.BlockSpec((block_size, dimension), lambda block: (block, 0)),),
        out_specs=pl.BlockSpec((block_size,), lambda block: (block,)),
        interpret=bool(interpret),
        name="phydrax_spatial_squared_norm",
    )
    return call(padded)[:item_count].reshape(leading_shape)


@_pallas_squared_norm.defjvp
def _pallas_squared_norm_jvp(
    block_size: int,
    interpret: bool,
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    (relative,) = primals
    (relative_tangent,) = tangents
    value = _pallas_squared_norm(relative, block_size, interpret)
    tangent = 2 * jnp.sum(relative * relative_tangent, axis=-1)
    return value, tangent


def _reference_pair_acceleration(
    relative: jax.Array,
    source_mass: jax.Array,
    valid: jax.Array,
    /,
    *,
    softening: float,
    coefficient: float,
) -> jax.Array:
    squared = jnp.sum(relative * relative, axis=-1) + softening**2
    return jnp.where(
        valid[..., None],
        coefficient * source_mass[..., None] * relative / squared[..., None] ** 1.5,
        0.0,
    )


@partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6))
def _pallas_pair_acceleration(
    relative: jax.Array,
    source_mass: jax.Array,
    valid: jax.Array,
    softening: float,
    coefficient: float,
    block_size: int,
    interpret: bool,
) -> jax.Array:
    from jax.experimental import pallas as pl

    dimension = int(relative.shape[-1])
    leading_shape = relative.shape[:-1]
    item_count = source_mass.size
    blocks = ceil(item_count / block_size)
    padded_count = blocks * block_size
    flat_displacement = jnp.pad(
        relative.reshape((item_count, dimension)),
        ((0, padded_count - item_count), (0, 0)),
    )
    flat_mass = jnp.pad(
        source_mass.reshape((item_count,)),
        (0, padded_count - item_count),
    )
    flat_mask = jnp.pad(
        valid.reshape((item_count,)),
        (0, padded_count - item_count),
    )

    def kernel(relative_ref, mass_ref, valid_ref, output_ref):
        value = relative_ref[...]
        squared = jnp.sum(value * value, axis=1) + softening**2
        output_ref[...] = jnp.where(
            valid_ref[...][:, None],
            coefficient * mass_ref[...][:, None] * value / squared[:, None] ** 1.5,
            0.0,
        )

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded_count, dimension), relative.dtype),
        grid=(blocks,),
        in_specs=(
            pl.BlockSpec((block_size, dimension), lambda block: (block, 0)),
            pl.BlockSpec((block_size,), lambda block: (block,)),
            pl.BlockSpec((block_size,), lambda block: (block,)),
        ),
        out_specs=pl.BlockSpec((block_size, dimension), lambda block: (block, 0)),
        interpret=bool(interpret),
        name="phydrax_particle_fmm_pair_acceleration",
    )
    return call(flat_displacement, flat_mass, flat_mask)[:item_count].reshape(
        leading_shape + (dimension,)
    )


@_pallas_pair_acceleration.defjvp
def _pallas_pair_acceleration_jvp(
    softening,
    coefficient,
    block_size,
    interpret,
    primals,
    tangents,
):
    relative, source_mass, valid = primals
    relative_tangent, mass_tangent, _ = tangents
    value = _pallas_pair_acceleration(
        relative,
        source_mass,
        valid,
        softening,
        coefficient,
        block_size,
        interpret,
    )
    _, tangent = jax.jvp(
        lambda current_relative, current_mass: _reference_pair_acceleration(
            current_relative,
            current_mass,
            valid,
            softening=softening,
            coefficient=coefficient,
        ),
        (relative, source_mass),
        (relative_tangent, mass_tangent),
    )
    return value, tangent


def spatial_squared_norm(
    relative: jax.Array,
    /,
    *,
    backend: _SpatialDistanceBackend = "jax",
    pallas_interpret: bool = False,
    block_size: int = 128,
) -> jax.Array:
    """Squared trailing-axis norm with an explicit portable backend policy."""
    values = jnp.asarray(relative)
    if values.ndim < 1 or int(values.shape[-1]) < 1:
        raise ValueError("relative must have one non-empty trailing coordinate axis.")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("relative must have floating dtype.")
    if values.size == 0:
        return jnp.sum(values * values, axis=-1)
    if backend == "jax":
        return jnp.sum(values * values, axis=-1)
    if backend != "pallas":
        raise ValueError("backend must be 'jax' or 'pallas'.")
    size = int(block_size)
    if size < 1:
        raise ValueError("block_size must be positive.")
    pallas_spatial_availability(interpret=pallas_interpret).require(
        "spatial.distance_squared"
    )
    return _pallas_squared_norm(values, size, bool(pallas_interpret))


def spatial_pair_acceleration(
    relative: jax.Array,
    source_mass: jax.Array,
    valid: jax.Array,
    /,
    *,
    softening: float,
    coefficient: float,
    backend: _SpatialDistanceBackend = "jax",
    pallas_interpret: bool = False,
    block_size: int = 128,
) -> jax.Array:
    """Evaluate softened inverse-cube pair acceleration with native gradients."""
    displacement = jnp.asarray(relative)
    mass = jnp.asarray(source_mass, dtype=displacement.dtype)
    mask = jnp.asarray(valid, dtype=bool)
    if displacement.shape[:-1] != mass.shape or mass.shape != mask.shape:
        raise ValueError("Pair acceleration arrays have incompatible shapes.")

    if backend == "jax" or displacement.size == 0:
        return _reference_pair_acceleration(
            displacement,
            mass,
            mask,
            softening=softening,
            coefficient=coefficient,
        )
    if backend != "pallas":
        raise ValueError("backend must be 'jax' or 'pallas'.")
    size = int(block_size)
    if size < 1:
        raise ValueError("block_size must be positive.")
    pallas_spatial_availability(interpret=pallas_interpret).require("particle_fmm.p2p")
    return _pallas_pair_acceleration(
        displacement,
        mass,
        mask,
        float(softening),
        float(coefficient),
        size,
        bool(pallas_interpret),
    )


__all__ = [
    "PALLAS_SPATIAL_CAPABILITIES",
    "pallas_spatial_availability",
    "spatial_squared_norm",
    "spatial_pair_acceleration",
]
