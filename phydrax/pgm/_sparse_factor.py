#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..graph import segment_logsumexp
from ._model import EnumeratedFactorGroup


def enumerated_joint_scores(
    group: EnumeratedFactorGroup,
    log_potentials: Array,
    incoming: tuple[Array, ...],
    /,
) -> Array:
    """Return supported-configuration log scores without dense state materialization."""
    scores = log_potentials
    for position, values in enumerate(incoming):
        indices = jnp.broadcast_to(
            group.configurations[:, position][None, :],
            (int(values.shape[0]), int(group.configurations.shape[0])),
        )
        scores = scores + jnp.take_along_axis(values, indices, axis=-1)
    return scores


def enumerated_factor_messages(
    group: EnumeratedFactorGroup,
    log_potentials: Array,
    incoming: tuple[Array, ...],
    cardinalities: tuple[int, ...],
    /,
    *,
    mode: Literal["sum", "max"],
) -> tuple[Array, ...]:
    """Compute sparse sum/max-product messages in represented configuration work."""
    outputs = []
    for target, cardinality in enumerate(cardinalities):
        scores = log_potentials
        for position, values in enumerate(incoming):
            if position == target:
                continue
            indices = jnp.broadcast_to(
                group.configurations[:, position][None, :],
                (int(values.shape[0]), int(group.configurations.shape[0])),
            )
            scores = scores + jnp.take_along_axis(values, indices, axis=-1)
        state_ids = group.configurations[:, target]

        def one_factor(
            values,
            state_ids=state_ids,
            cardinality=cardinality,
        ):
            if mode == "sum":
                return segment_logsumexp(values, state_ids, cardinality)
            return jax.ops.segment_max(values, state_ids, cardinality)

        reduced = jax.vmap(one_factor)(scores)
        maxima = jnp.max(reduced, axis=-1, keepdims=True)
        outputs.append(jnp.where(jnp.isfinite(maxima), reduced - maxima, -jnp.inf))
    return tuple(outputs)


__all__ = ["enumerated_factor_messages", "enumerated_joint_scores"]
