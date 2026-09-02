#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class PlayerControlPartition(StrictModule):
    """Ordered contiguous ownership of a flattened joint-control vector."""

    player_ids: tuple[str, ...] = eqx.field(static=True)
    control_sizes: tuple[int, ...] = eqx.field(static=True)
    control_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    control_owner: tuple[int, ...] = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    joint_control_size: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        player_ids: Sequence[str],
        control_sizes: Sequence[int],
        /,
    ):
        if isinstance(player_ids, str):
            raise TypeError("player_ids must be a sequence of player identifiers.")
        resolved_ids = tuple(player_ids)
        if not resolved_ids:
            raise ValueError("PlayerControlPartition requires at least one player.")
        if any(
            not isinstance(player_id, str) or not player_id for player_id in resolved_ids
        ):
            raise ValueError("Player identifiers must be non-empty strings.")
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("Player identifiers must be unique.")

        if isinstance(control_sizes, (str, bytes)):
            raise TypeError("control_sizes must be a sequence of positive integers.")
        raw_sizes = tuple(control_sizes)
        if len(raw_sizes) != len(resolved_ids):
            raise ValueError("control_sizes must provide one size per player.")
        if any(isinstance(size, bool) for size in raw_sizes):
            raise TypeError("Control sizes must be positive integers, not booleans.")
        resolved_sizes = tuple(index(size) for size in raw_sizes)
        if any(size <= 0 for size in resolved_sizes):
            raise ValueError("Control sizes must be positive integers.")

        offsets = [0]
        for size in resolved_sizes:
            offsets.append(offsets[-1] + size)
        resolved_slices = tuple(pairwise(offsets))
        resolved_owner = tuple(
            player for player, size in enumerate(resolved_sizes) for _ in range(size)
        )
        payload = [
            {"player_id": player_id, "control_size": size}
            for player_id, size in zip(resolved_ids, resolved_sizes, strict=True)
        ]

        self.player_ids = resolved_ids
        self.control_sizes = resolved_sizes
        self.control_slices = resolved_slices
        self.control_owner = resolved_owner
        self.num_players = len(resolved_ids)
        self.joint_control_size = offsets[-1]
        self.partition_id = f"player-controls:{canonical_fingerprint(payload)}"

    def split_controls(self, joint_controls, /) -> tuple[jax.Array, ...]:
        """Split the trailing joint-control axis in declared player order."""
        array = jnp.asarray(joint_controls)
        if array.ndim < 1 or array.shape[-1] != self.joint_control_size:
            raise ValueError(
                "joint_controls must have trailing shape "
                f"({self.joint_control_size},); got {array.shape}."
            )
        return tuple(array[..., start:stop] for start, stop in self.control_slices)

    def join_controls(self, player_controls: Sequence, /) -> jax.Array:
        """Join player-local trailing control axes in declared player order."""
        arrays = tuple(jnp.asarray(value) for value in player_controls)
        if len(arrays) != self.num_players:
            raise ValueError("player_controls must provide one array per player.")
        leading_shape = arrays[0].shape[:-1] if arrays[0].ndim >= 1 else None
        for player_id, size, array in zip(
            self.player_ids,
            self.control_sizes,
            arrays,
            strict=True,
        ):
            if array.ndim < 1 or array.shape[-1] != size:
                raise ValueError(
                    f"Player {player_id!r} controls must have trailing shape ({size},); "
                    f"got {array.shape}."
                )
            if array.shape[:-1] != leading_shape:
                raise ValueError("All player control arrays must share leading axes.")
        return jnp.concatenate(arrays, axis=-1)

    def split_feedback_gain(self, gain, /) -> tuple[jax.Array, ...]:
        """Split the penultimate control-row axis of a joint feedback gain."""
        array = jnp.asarray(gain)
        if array.ndim < 2 or array.shape[-2] != self.joint_control_size:
            raise ValueError(
                "gain must have penultimate control axis of size "
                f"{self.joint_control_size}; got {array.shape}."
            )
        return tuple(array[..., start:stop, :] for start, stop in self.control_slices)
