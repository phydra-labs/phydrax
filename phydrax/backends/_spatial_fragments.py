#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from functools import partial
from typing import assert_never

import jax
import jax.numpy as jnp
from jax import Array
from jax._src.core import ShapedArray
from jax.extend import core
from jax.interpreters import mlir

from .._numerics._compensated import two_sum
from ..sparse._execution import KeyGroupAccumulation
from ..sparse._streamed import StreamedFragment


# Triton has no erfc lowering. A kernel-local primitive calls CUDA libdevice
# directly, retaining float64 and the erfc tail instead of subtracting erf from 1.
# The ordinary JAX implementation is used only by explicit CPU interpret mode.
_erfc = core.Primitive("phydrax_spatial_fragment_erfc")
_erfc.def_impl(jax.lax.erfc)


@_erfc.def_abstract_eval
def _erfc_shape(value: ShapedArray) -> ShapedArray:
    return value


mlir.register_lowering(_erfc, mlir.lower_fun(jax.lax.erfc, multiple_results=False))


def _register_erfc_lowering() -> None:
    from jax._src.lib.mlir import ir
    from jax._src.lib.triton import dialect
    from jax._src.pallas.triton import lowering

    @lowering.register_lowering(_erfc)
    def lower(ctx: lowering.LoweringRuleContext, value: ir.Value) -> ir.Value:
        dtype = ctx.avals_in[0].dtype
        symbol = "__nv_erfcf" if dtype == jnp.dtype(jnp.float32) else "__nv_erfc"
        return dialect.extern_elementwise(
            value.type, [value], libname="libdevice", libpath="", symbol=symbol, pure=True
        )


_register_erfc_lowering()


def _pair_value(
    relative: Array,
    strength: Array,
    valid: Array,
    softening: float,
    coefficient: float,
    split: float | None,
    cutoff: float | None,
    *,
    kernel: bool = False,
) -> Array:
    channels = jnp.arange(4, dtype=jnp.int32)
    dtype = relative.dtype
    zero = jnp.zeros((), dtype=dtype)
    epsilon_squared = jnp.asarray(softening**2, dtype=dtype)
    coefficient_value = jnp.asarray(coefficient, dtype=dtype)
    relative = jnp.where(valid[..., None], relative, zero)
    squared = jnp.sum(jnp.where(channels < 3, relative * relative, zero), axis=-1)
    squared = jnp.where(valid, squared + epsilon_squared, jnp.ones((), dtype=dtype))
    radius = jnp.sqrt(squared)
    potential_factor = jnp.ones_like(radius)
    gradient_factor = jnp.ones_like(radius)
    if split is not None:
        split_value = jnp.asarray(split, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        sqrt_pi = jnp.sqrt(jnp.asarray(jnp.pi, dtype=dtype))
        argument = radius / (two * split_value)
        potential_factor = _erfc.bind(argument) if kernel else jax.lax.erfc(argument)
        gradient_factor = potential_factor + radius / (split_value * sqrt_pi) * jnp.exp(
            -(argument * argument)
        )
    if cutoff is not None:
        valid = valid & (radius <= jnp.asarray(cutoff, dtype=dtype))
    amplitude = coefficient_value * jnp.where(valid, strength, zero)
    gradient = (
        amplitude[..., None]
        * relative
        * (gradient_factor / (squared * radius))[..., None]
    )
    potential = amplitude * potential_factor / radius
    return jnp.where(channels < 3, gradient, potential[..., None])


def reference_leaf_pair_fragment(
    sources: tuple[Array, Array, Array, Array],
    receivers: tuple[Array, Array, Array],
    fragment: StreamedFragment,
    seed: KeyGroupAccumulation,
    softening: float,
    coefficient: float,
    split: float | None,
    cutoff: float | None,
) -> KeyGroupAccumulation:
    source_positions, strengths, source_mask, source_ids = sources
    receiver_positions, receiver_mask, receiver_ids = receivers
    leaf_size = source_positions.shape[1]
    lane_positions = receiver_positions[fragment.lane_slots]
    lane_source = fragment.lane_sources
    lane_mask = receiver_mask[fragment.lane_slots] & fragment.lane_use[:, None]
    lane_ids = receiver_ids[fragment.lane_slots]

    def add_source(index: int, total: Array) -> Array:
        valid = (
            lane_mask
            & source_mask[lane_source, index, None]
            & (lane_ids != source_ids[lane_source, index, None])
        )
        relative = jnp.pad(
            source_positions[lane_source, index, None, :] - lane_positions,
            ((0, 0), (0, 0), (0, 1)),
        )
        value = _pair_value(
            relative,
            strengths[lane_source, index, None],
            valid,
            softening,
            coefficient,
            split,
            cutoff,
        )
        return total + value

    values = jax.lax.fori_loop(
        0,
        leaf_size,
        add_source,
        jnp.zeros((fragment.lane_sources.shape[0], leaf_size, 4), source_positions.dtype),
    )
    return fragment.reduce(values, seed)


@partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6, 7, 8))
def pallas_leaf_pair_fragment(
    sources: tuple[Array, Array, Array, Array],
    receivers: tuple[Array, Array, Array],
    fragment: StreamedFragment,
    seed: KeyGroupAccumulation,
    softening: float,
    coefficient: float,
    split: float | None,
    cutoff: float | None,
    interpret: bool,
) -> KeyGroupAccumulation:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    source_positions, strengths, source_mask, source_ids = sources
    receiver_positions, receiver_mask, receiver_ids = receivers
    receiver_count, leaf_size = receiver_mask.shape
    target_width = 1 << (leaf_size - 1).bit_length()
    mode = fragment.accumulation

    def kernel(
        positions_ref: jax.Ref,
        strength_ref: jax.Ref,
        mask_ref: jax.Ref,
        ids_ref: jax.Ref,
        targets_ref: jax.Ref,
        target_mask_ref: jax.Ref,
        target_ids_ref: jax.Ref,
        offsets_ref: jax.Ref,
        counts_ref: jax.Ref,
        lanes_ref: jax.Ref,
        use_ref: jax.Ref,
        high_ref: jax.Ref,
        correction_ref: jax.Ref,
        output_high_ref: jax.Ref,
        output_correction_ref: jax.Ref,
    ) -> None:
        slot = pl.program_id(0)
        targets = jnp.arange(target_width, dtype=jnp.int32)
        channels = jnp.arange(4, dtype=jnp.int32)
        coords = jnp.minimum(channels, 2)
        target_position = plt.load(
            targets_ref.at[slot, targets[:, None], coords[None, :]],
            mask=targets[:, None] < leaf_size,
            other=0.0,
        )
        target_mask = plt.load(
            target_mask_ref.at[slot, targets], mask=targets < leaf_size, other=False
        )
        target_ids = plt.load(
            target_ids_ref.at[slot, targets], mask=targets < leaf_size, other=0
        )
        initial_high = plt.load(
            high_ref.at[slot, targets[:, None], channels[None, :]],
            mask=targets[:, None] < leaf_size,
            other=0.0,
        )
        initial_correction = plt.load(
            correction_ref.at[slot, targets[:, None], channels[None, :]],
            mask=targets[:, None] < leaf_size,
            other=0.0,
        )
        offset, count = offsets_ref[slot], counts_ref[slot]

        def add_lane(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            high, correction = carry
            lane = offset + index
            source = lanes_ref[lane]
            usable = use_ref[lane]

            def add_source(point: int, total: Array) -> Array:
                source_position = positions_ref[source, point, coords]
                valid = (
                    target_mask
                    & usable
                    & mask_ref[source, point]
                    & (target_ids != ids_ref[source, point])
                )
                value = _pair_value(
                    source_position[None, :] - target_position,
                    strength_ref[source, point],
                    valid,
                    softening,
                    coefficient,
                    split,
                    cutoff,
                    kernel=True,
                )
                return total + value

            value = jax.lax.fori_loop(0, leaf_size, add_source, jnp.zeros_like(high))
            match mode:
                case "fast" | "deterministic":
                    next_high = high + value
                    return jnp.where(value != 0, next_high, high), correction
                case "compensated":
                    next_high, error = two_sum(high, value)
                    return (
                        jnp.where(value != 0, next_high, high),
                        jnp.where(value != 0, correction + error, correction),
                    )
                case _:
                    assert_never(mode)

        # Fast sparse reduction first sums the events, then adds the seed.
        high = jnp.zeros_like(initial_high) if mode == "fast" else initial_high
        high, correction = jax.lax.fori_loop(
            0, count, add_lane, (high, initial_correction)
        )
        if mode == "fast":
            high = jnp.where(high != 0, initial_high + high, initial_high)
        plt.store(
            output_high_ref.at[slot, targets[:, None], channels[None, :]],
            high,
            mask=targets[:, None] < leaf_size,
        )
        plt.store(
            output_correction_ref.at[slot, targets[:, None], channels[None, :]],
            correction,
            mask=targets[:, None] < leaf_size,
        )

    shape = jax.ShapeDtypeStruct(seed.high.shape, seed.high.dtype)
    high, correction = pl.pallas_call(
        kernel,
        out_shape=(shape, shape),
        grid=(receiver_count,),
        interpret=interpret,
        name="phydrax_spatial_leaf_pair_fragment",
    )(
        source_positions,
        strengths,
        source_mask,
        source_ids,
        receiver_positions,
        receiver_mask,
        receiver_ids,
        fragment.receiver_lane_offsets,
        fragment.receiver_lane_counts,
        fragment.lane_sources,
        fragment.lane_use,
        seed.high,
        seed.correction,
    )
    return KeyGroupAccumulation(high, correction)


@pallas_leaf_pair_fragment.defjvp
def _fragment_jvp(
    softening: float,
    coefficient: float,
    split: float | None,
    cutoff: float | None,
    interpret: bool,
    primals: tuple[
        tuple[Array, Array, Array, Array],
        tuple[Array, Array, Array],
        StreamedFragment,
        KeyGroupAccumulation,
    ],
    tangents: tuple[
        tuple[Array, Array, Array, Array],
        tuple[Array, Array, Array],
        StreamedFragment,
        KeyGroupAccumulation,
    ],
) -> tuple[KeyGroupAccumulation, KeyGroupAccumulation]:
    sources, receivers, fragment, seed = primals
    ds, dr, _, dseed = tangents
    value = pallas_leaf_pair_fragment(
        sources,
        receivers,
        fragment,
        seed,
        softening,
        coefficient,
        split,
        cutoff,
        interpret,
    )

    def reference(
        x: Array, q: Array, y: Array, high: Array, correction: Array
    ) -> KeyGroupAccumulation:
        return reference_leaf_pair_fragment(
            (x, q, sources[2], sources[3]),
            (y, receivers[1], receivers[2]),
            fragment,
            KeyGroupAccumulation(high, correction),
            softening,
            coefficient,
            split,
            cutoff,
        )

    _, tangent = jax.jvp(
        reference,
        (sources[0], sources[1], receivers[0], seed.high, seed.correction),
        (ds[0], ds[1], dr[0], dseed.high, dseed.correction),
    )
    return value, tangent
