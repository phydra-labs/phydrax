from __future__ import annotations

import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from ._material_schedule import MaterialChainSchedule


_ZERO_EXPONENT = -(1 << 60)


def _multiply(
    left_mantissa: Array,
    left_exponent: Array,
    right_mantissa: Array,
    right_exponent: Array,
    /,
) -> tuple[Array, Array]:
    product, shift = jnp.frexp(left_mantissa * right_mantissa)
    exponent = left_exponent + right_exponent + shift.astype(jnp.int64)
    return product, jnp.where(product == 0.0, _ZERO_EXPONENT, exponent)


def _map_action(
    schedule: MaterialChainSchedule,
    values: Array,
    /,
    *,
    transpose: bool,
) -> Array:
    source_rows = schedule.target_count if transpose else schedule.source_count
    target_rows = schedule.source_count if transpose else schedule.target_count
    if values.ndim == 0 or values.shape[0] != source_rows:
        raise ValueError(
            f"Material chain values must start with {source_rows} endpoint rows; got {values.shape}."
        )
    payload = tuple(values.shape[1:])
    original_dtype = values.dtype
    if jnp.issubdtype(original_dtype, jnp.complexfloating):
        channels = jnp.stack((jnp.real(values), jnp.imag(values)), axis=-1)
        channels = channels.reshape((source_rows, -1))
    else:
        channels = values.reshape((source_rows, -1))
    compute_dtype = jnp.result_type(channels.dtype, jnp.float64)
    channels = channels.astype(compute_dtype)
    input_mantissas, input_exponents = jnp.frexp(channels)
    input_exponents = input_exponents.astype(jnp.int64)
    slots = schedule.slot_count
    state_size = slots + schedule.maximum_rows
    channel_count = channels.shape[1]
    state_mantissas = jnp.zeros((state_size, channel_count), dtype=compute_dtype)
    state_exponents = jnp.full(
        (state_size, channel_count), _ZERO_EXPONENT, dtype=jnp.int64
    )
    input_slots = schedule.final_slots if transpose else schedule.initial_slots
    state_mantissas = state_mantissas.at[input_slots].set(input_mantissas)
    state_exponents = state_exponents.at[input_slots].set(
        jnp.where(input_mantissas == 0.0, _ZERO_EXPONENT, input_exponents)
    )

    def action(index: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        mantissas, exponents = state
        map_index = schedule.map_count - 1 - index if transpose else index
        edge_sources = schedule.edge_source_slots[map_index]
        edge_targets = schedule.edge_target_slots[map_index]
        valid = schedule.edge_valid[map_index]
        source_slots = edge_targets if transpose else edge_sources
        target_slots = edge_sources if transpose else edge_targets
        safe_sources = jnp.where(valid, source_slots, 0)
        safe_targets = jnp.where(valid, target_slots, 0)
        source_mantissas = mantissas[safe_sources]
        source_exponents = exponents[safe_sources]
        weighted_mantissas, weighted_exponents = _multiply(
            source_mantissas,
            source_exponents,
            schedule.edge_weight_mantissas[map_index, :, None].astype(compute_dtype),
            schedule.edge_weight_exponents[map_index, :, None],
        )
        contribution_valid = valid[:, None] & (weighted_mantissas != 0.0)
        weighted_exponents = jnp.where(
            contribution_valid, weighted_exponents, _ZERO_EXPONENT
        )
        maximum = jnp.full((state_size, channel_count), _ZERO_EXPONENT, dtype=jnp.int64)
        maximum = maximum.at[safe_targets].max(weighted_exponents)
        shift = weighted_exponents - maximum[safe_targets]
        aligned = jnp.ldexp(weighted_mantissas, shift)
        aligned = jnp.where(contribution_valid, aligned, 0.0)
        sums = jnp.zeros((state_size, channel_count), dtype=compute_dtype)
        sums = sums.at[safe_targets].add(aligned)
        output_mantissas, output_shifts = jnp.frexp(sums)
        output_exponents = maximum + output_shifts.astype(jnp.int64)
        output_exponents = jnp.where(sums == 0.0, _ZERO_EXPONENT, output_exponents)
        row_slots = (
            schedule.source_row_slots[map_index]
            if transpose
            else schedule.target_row_slots[map_index]
        )
        row_valid = (
            schedule.source_row_valid[map_index]
            if transpose
            else schedule.target_row_valid[map_index]
        )
        safe_rows = jnp.where(
            row_valid,
            row_slots,
            slots + jnp.arange(schedule.maximum_rows, dtype=jnp.int64),
        )
        prior_mantissas = mantissas[safe_rows]
        prior_exponents = exponents[safe_rows]
        mantissas = mantissas.at[safe_rows].set(
            jnp.where(row_valid[:, None], output_mantissas[safe_rows], prior_mantissas)
        )
        exponents = exponents.at[safe_rows].set(
            jnp.where(row_valid[:, None], output_exponents[safe_rows], prior_exponents)
        )
        return mantissas, exponents

    state_mantissas, state_exponents = lax.fori_loop(
        0,
        schedule.map_count,
        action,
        (state_mantissas, state_exponents),
    )
    output_slots = schedule.initial_slots if transpose else schedule.final_slots
    result = jnp.ldexp(state_mantissas[output_slots], state_exponents[output_slots])
    if jnp.issubdtype(original_dtype, jnp.complexfloating):
        paired = result.reshape((target_rows, *payload, 2))
        restored = lax.complex(paired[..., 0], paired[..., 1])
        return restored.astype(original_dtype)
    return result.reshape((target_rows, *payload)).astype(original_dtype)


def material_chain_apply(
    schedule: MaterialChainSchedule,
    values: ArrayLike,
    /,
) -> Array:
    """Apply the scheduled chronological extensive-content action."""
    if not isinstance(schedule, MaterialChainSchedule):
        raise TypeError("schedule must be a MaterialChainSchedule.")
    return _map_action(schedule, jnp.asarray(values), transpose=False)


def material_chain_transpose(
    schedule: MaterialChainSchedule,
    cotangents: ArrayLike,
    /,
) -> Array:
    """Apply the algebraic transpose of the scheduled extensive-content action."""
    if not isinstance(schedule, MaterialChainSchedule):
        raise TypeError("schedule must be a MaterialChainSchedule.")
    return _map_action(schedule, jnp.asarray(cotangents), transpose=True)


__all__ = ["material_chain_apply", "material_chain_transpose"]
