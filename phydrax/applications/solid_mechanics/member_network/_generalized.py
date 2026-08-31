#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....metrix import AbstractStateGeometry, EuclideanStateGeometry


class GeneralizedDOFChannel(StrictModule, NonTrainableState):
    """One named shaped generalized-coordinate channel."""

    channel_id: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    constrained: Array
    prescribed_indices: Array
    free_indices: Array
    geometry: AbstractStateGeometry
    size: int = eqx.field(static=True)
    reduced_size: int = eqx.field(static=True)

    def __init__(
        self,
        channel_id: str,
        shape: Sequence[int],
        /,
        *,
        constrained: ArrayLike | None = None,
        geometry: AbstractStateGeometry | None = None,
    ):
        identifier = str(channel_id)
        shape_ = tuple(int(value) for value in shape)
        if not identifier or not shape_ or any(value <= 0 for value in shape_):
            raise ValueError("Generalized channel ID and shape must be nonempty.")
        size = prod(shape_)
        mask = (
            jnp.zeros((size,), dtype=bool)
            if constrained is None
            else jnp.asarray(constrained, dtype=bool).reshape((-1,))
        )
        if mask.shape != (size,):
            raise ValueError("Generalized channel constraint mask has the wrong size.")
        geometry_ = EuclideanStateGeometry() if geometry is None else geometry
        if not isinstance(geometry_, AbstractStateGeometry):
            raise TypeError("geometry must be an AbstractStateGeometry.")
        self.channel_id = identifier
        self.shape = shape_
        self.constrained = mask.reshape(shape_)
        self.prescribed_indices = jnp.flatnonzero(mask, size=size, fill_value=-1)[
            : jnp.sum(mask)
        ]
        self.free_indices = jnp.flatnonzero(~mask, size=size, fill_value=-1)[
            : jnp.sum(~mask)
        ]
        self.geometry = geometry_
        self.size = size
        self.reduced_size = int(jnp.sum(~mask))

    @property
    def prescribed_size(self) -> int:
        return self.size - self.reduced_size


class GeneralizedDOFLayout(StrictModule, NonTrainableState):
    """Static ordered product of named generalized-coordinate channels."""

    channels: tuple[GeneralizedDOFChannel, ...]
    reduced_offsets: tuple[int, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, channels: Sequence[GeneralizedDOFChannel], /):
        channels_ = tuple(channels)
        if not channels_ or len({value.channel_id for value in channels_}) != len(
            channels_
        ):
            raise ValueError("Generalized channels must be nonempty with unique IDs.")
        offsets = [0]
        for channel in channels_:
            offsets.append(offsets[-1] + channel.reduced_size)
        self.channels = channels_
        self.reduced_offsets = tuple(offsets)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "generalized-dof-layout",
                "channels": [
                    {
                        "id": value.channel_id,
                        "shape": list(value.shape),
                        "geometry": value.geometry.geometry_id,
                        "constraints": value.constrained.tolist(),
                    }
                    for value in channels_
                ],
            }
        )

    @property
    def reduced_size(self) -> int:
        return self.reduced_offsets[-1]

    def reduce(self, values: Mapping[str, ArrayLike], /) -> Array:
        reduced = []
        for channel in self.channels:
            if channel.channel_id not in values:
                raise KeyError(f"Missing generalized channel {channel.channel_id!r}.")
            value = jnp.asarray(values[channel.channel_id])
            if value.shape != channel.shape:
                raise ValueError(f"Channel {channel.channel_id!r} has the wrong shape.")
            reduced.append(value.reshape((-1,))[channel.free_indices])
        return jnp.concatenate(tuple(reduced))

    def expand(
        self,
        reduced: ArrayLike,
        prescribed: Mapping[str, ArrayLike],
        /,
    ) -> GeneralizedKinematics:
        vector = jnp.asarray(reduced)
        if vector.shape != (self.reduced_size,):
            raise ValueError("Reduced generalized coordinates have the wrong shape.")
        values: dict[str, Array] = {}
        for channel, left, right in zip(
            self.channels,
            self.reduced_offsets[:-1],
            self.reduced_offsets[1:],
            strict=True,
        ):
            fixed = jnp.asarray(prescribed[channel.channel_id])
            if fixed.shape != (channel.prescribed_size,):
                raise ValueError(
                    f"Prescribed channel {channel.channel_id!r} has the wrong shape."
                )
            full = jnp.zeros((channel.size,), dtype=vector.dtype)
            full = full.at[channel.prescribed_indices].set(fixed, unique_indices=True)
            full = full.at[channel.free_indices].set(
                vector[left:right], unique_indices=True
            )
            values[channel.channel_id] = full.reshape(channel.shape)
        return GeneralizedKinematics(values, self.layout_id)


class GeneralizedKinematics(StrictModule):
    """Named generalized coordinates under one immutable layout identity."""

    values: dict[str, Array]
    layout_id: str = eqx.field(static=True)

    def channel(self, channel_id: str, /) -> Array:
        if channel_id not in self.values:
            raise KeyError(f"Unknown generalized channel {channel_id!r}.")
        return self.values[channel_id]


__all__ = [
    "GeneralizedDOFChannel",
    "GeneralizedDOFLayout",
    "GeneralizedKinematics",
]
