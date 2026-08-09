#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from .._layout import StateLayout
from .._trajectory import TrajectoryData


def _delay_valid(data: TrajectoryData, delay: int, /):
    indices = jnp.maximum(jnp.arange(data.capacity) - delay, 0)
    valid = jnp.take(data.sample_valid, indices, axis=-1)
    valid = valid & (jnp.arange(data.capacity) >= delay)
    connected = []
    for target in range(data.capacity):
        source = max(0, target - delay)
        value = jnp.ones(data.case_shape, dtype=bool)
        for transition in range(source, target):
            value = value & data.transition_valid[..., transition]
        connected.append(value)
    return valid & jnp.stack(connected, axis=-1), indices


def delay_embed(data: TrajectoryData, delays: Sequence[int], /) -> TrajectoryData:
    """Delay-embed states without crossing padding, discontinuities, or resets."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    resolved = tuple(int(delay) for delay in delays)
    if not resolved or any(delay < 0 for delay in resolved):
        raise ValueError("delays must contain nonnegative integers.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("delays must be unique.")
    time_axis = len(data.case_shape)
    flattened = []
    masks = []
    for delay in resolved:
        valid, indices = _delay_valid(data, delay)
        values = jnp.take(data.states, indices, axis=time_axis).reshape(
            data.case_shape + (data.capacity, data.state_layout.size)
        )
        flattened.append(values)
        masks.append(valid)
    embedded = jnp.concatenate(tuple(flattened), axis=-1)
    embedded_valid = jnp.all(jnp.stack(tuple(masks), axis=-1), axis=-1)
    embedded = jnp.where(
        embedded_valid[..., None], embedded, jnp.full_like(embedded, jnp.nan)
    )
    component_names = tuple(
        f"{name}[t-{delay}]"
        for delay in resolved
        for name in data.state_layout.component_names
    )
    layout = StateLayout(
        (len(component_names),),
        axes=("delay_state",),
        component_names=component_names,
    )
    transitions = (
        data.transition_valid & embedded_valid[..., :-1] & embedded_valid[..., 1:]
    )
    input_support = embedded_valid if data.input_alignment == "samples" else transitions
    return TrajectoryData(
        data.coordinates,
        embedded,
        state_layout=layout,
        sample_valid=embedded_valid,
        transition_valid=transitions,
        reset_mask=data.reset_mask,
        weights=jnp.where(embedded_valid, data.weights, 0.0),
        inputs=data.inputs,
        input_layout=data.input_layout,
        input_valid=(
            None if data.input_valid is None else data.input_valid & input_support
        ),
        input_alignment=(
            "transitions" if data.input_alignment is None else data.input_alignment
        ),
        case_axes=data.case_axes,
        case_axis_roles=data.case_axis_roles,
        coordinate_id=data.coordinate_id,
        source_id=f"{data.source_id}:delay-embedding:{resolved}",
    )


__all__ = ["delay_embed"]
