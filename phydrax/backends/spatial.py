#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from functools import partial
from math import ceil
from typing import assert_never, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._execution_device import device_platform
from ..sparse._execution import KeyGroupAccumulation
from ..sparse._streamed import StreamedFragment
from ..typing import parse
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
    platforms = tuple(device_platform(device) for device in jax.devices())
    available = bool(interpret) or any(
        platform in ("cuda", "tpu") for platform in platforms
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
    dimension = values.shape[-1]
    padded_dimension = 1 << (dimension - 1).bit_length()
    block_size = 1 << (block_size - 1).bit_length()
    leading_shape = values.shape[:-1]
    item_count = values.size // dimension
    blocks = ceil(item_count / block_size)
    padded_count = blocks * block_size
    flattened = values.reshape((item_count, dimension))
    padded = jnp.pad(
        flattened, ((0, padded_count - item_count), (0, padded_dimension - dimension))
    )

    def kernel(value_ref: jax.Ref, output_ref: jax.Ref) -> None:
        value = value_ref[...]
        output_ref[...] = jnp.sum(value * value, axis=1)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded_count,), values.dtype),
        grid=(blocks,),
        in_specs=(
            pl.BlockSpec((block_size, padded_dimension), lambda block: (block, 0)),
        ),
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

    dimension = relative.shape[-1]
    padded_dimension = 1 << (dimension - 1).bit_length()
    block_size = 1 << (block_size - 1).bit_length()
    leading_shape = relative.shape[:-1]
    item_count = source_mass.size
    blocks = ceil(item_count / block_size)
    padded_count = blocks * block_size
    flat_displacement = jnp.pad(
        relative.reshape((item_count, dimension)),
        ((0, padded_count - item_count), (0, padded_dimension - dimension)),
    )
    flat_mass = jnp.pad(
        source_mass.reshape((item_count,)),
        (0, padded_count - item_count),
    )
    flat_mask = jnp.pad(
        valid.reshape((item_count,)),
        (0, padded_count - item_count),
    )

    def kernel(
        relative_ref: jax.Ref,
        mass_ref: jax.Ref,
        valid_ref: jax.Ref,
        output_ref: jax.Ref,
    ) -> None:
        value = relative_ref[...]
        squared = jnp.sum(value * value, axis=1) + softening**2
        output_ref[...] = jnp.where(
            valid_ref[...][:, None],
            coefficient * mass_ref[...][:, None] * value / squared[:, None] ** 1.5,
            0.0,
        )

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded_count, padded_dimension), relative.dtype),
        grid=(blocks,),
        in_specs=(
            pl.BlockSpec((block_size, padded_dimension), lambda block: (block, 0)),
            pl.BlockSpec((block_size,), lambda block: (block,)),
            pl.BlockSpec((block_size,), lambda block: (block,)),
        ),
        out_specs=pl.BlockSpec((block_size, padded_dimension), lambda block: (block, 0)),
        interpret=bool(interpret),
        name="phydrax_particle_fmm_pair_acceleration",
    )
    return call(flat_displacement, flat_mass, flat_mask)[:item_count, :dimension].reshape(
        leading_shape + (dimension,)
    )


@_pallas_pair_acceleration.defjvp
def _pallas_pair_acceleration_jvp(
    softening: float,
    coefficient: float,
    block_size: int,
    interpret: bool,
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
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
    if values.ndim < 1 or values.shape[-1] < 1:
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
    mask = jnp.asarray(valid, dtype=jnp.bool_)
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


def spatial_leaf_pair_fragment(
    sources: tuple[Array, Array, Array, Array],
    receivers: tuple[Array, Array, Array],
    fragment: StreamedFragment,
    seed: KeyGroupAccumulation,
    /,
    *,
    softening: float,
    coefficient: float,
    short_range_scale: float | None = None,
    short_range_cutoff: float | None = None,
    backend: _SpatialDistanceBackend = "jax",
    pallas_interpret: bool = False,
) -> KeyGroupAccumulation:
    """Accumulate bounded leaf-pair events as gradient xyz and potential.

    Sources are global leaf rows; receivers are already gathered fragment
    slots. Equal storage IDs exclude the diagonal. The split potential is
    ``coefficient * strength * erfc(radius / (2 * scale)) / radius``, with
    softened radius and inclusive cutoff. Pallas executes one receiver program
    and writes its seeded high/correction once; its derivatives use the same
    bounded JAX reference, never a replacement primal.

    Geometry, strengths, and seeds must share float32 or float64. Compiled
    execution supports CUDA only; CPU requires explicit interpret mode.
    """
    from ._spatial_fragments import (
        pallas_leaf_pair_fragment,
        reference_leaf_pair_fragment,
    )

    backend = parse(backend, _SpatialDistanceBackend, "backend")
    if not isinstance(fragment, StreamedFragment):
        raise TypeError("fragment must be a StreamedFragment.")
    if not isinstance(seed, KeyGroupAccumulation):
        raise TypeError("seed must be a KeyGroupAccumulation.")
    x, q, source_mask, source_ids = sources
    y, receiver_mask, receiver_ids = receivers
    if x.ndim != 3 or x.shape[-1] != 3 or y.ndim != 3 or y.shape[-1] != 3:
        raise ValueError("Leaf positions must have shape [leaves, L, 3].")
    leaf_size = x.shape[1]
    receiver_count = y.shape[0]
    if y.shape[1] != leaf_size or q.shape != x.shape[:2]:
        raise ValueError("Source strengths and receiver leaf extents must agree.")
    if source_mask.shape != q.shape or source_ids.shape != q.shape:
        raise ValueError("Source masks and IDs must match source leaf rows.")
    if receiver_mask.shape != y.shape[:2] or receiver_ids.shape != y.shape[:2]:
        raise ValueError("Receiver masks and IDs must match gathered receiver rows.")
    if source_mask.dtype != jnp.bool_ or receiver_mask.dtype != jnp.bool_:
        raise TypeError("Leaf masks must have bool dtype.")
    if not jnp.issubdtype(source_ids.dtype, jnp.integer) or not jnp.issubdtype(
        receiver_ids.dtype, jnp.integer
    ):
        raise TypeError("Leaf storage IDs must have integer dtype.")
    if source_ids.dtype != receiver_ids.dtype:
        raise TypeError("Source and receiver storage IDs must have the same dtype.")
    if x.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("Leaf fragments support float32 and float64 only.")
    if any(value.dtype != x.dtype for value in (q, y, seed.high, seed.correction)):
        raise TypeError("Leaf geometry, strengths, and seeds must share one dtype.")
    if seed.high.shape != (receiver_count, leaf_size, 4):
        raise ValueError("Fragment seeds must have shape [R, L, 4].")
    if (
        fragment.receiver_ids.shape != (receiver_count,)
        or fragment.lane_sources.ndim != 1
    ):
        raise ValueError("Routing must describe one fragment with R receiver slots.")
    epsilon, scale = float(softening), float(coefficient)
    split = None if short_range_scale is None else float(short_range_scale)
    cutoff = None if short_range_cutoff is None else float(short_range_cutoff)
    if not np.isfinite(epsilon) or epsilon < 0 or not np.isfinite(scale):
        raise ValueError("softening must be finite/nonnegative and coefficient finite.")
    if split is not None and (not np.isfinite(split) or split <= 0):
        raise ValueError("short_range_scale must be finite and positive.")
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff < 0):
        raise ValueError("short_range_cutoff must be finite and nonnegative.")
    match backend:
        case "jax":
            execute = reference_leaf_pair_fragment
        case "pallas":
            platforms = tuple(device_platform(device) for device in jax.devices())
            if not pallas_interpret and "cuda" not in platforms:
                raise ValueError(
                    "Pallas leaf fragments require CUDA or explicit CPU interpret mode; "
                    "TPU and HIP are unsupported."
                )
            pallas_spatial_availability(interpret=pallas_interpret).require(
                "particle_fmm.p2p"
            )
            if not receiver_count or not leaf_size or not x.shape[0]:
                return seed
            return pallas_leaf_pair_fragment(
                sources,
                receivers,
                fragment,
                seed,
                epsilon,
                scale,
                split,
                cutoff,
                bool(pallas_interpret),
            )
        case _:
            assert_never(backend)
    if not receiver_count or not leaf_size or not x.shape[0]:
        return seed
    return execute(sources, receivers, fragment, seed, epsilon, scale, split, cutoff)


__all__ = [
    "PALLAS_SPATIAL_CAPABILITIES",
    "pallas_spatial_availability",
    "spatial_pair_acceleration",
    "spatial_leaf_pair_fragment",
    "spatial_squared_norm",
]
