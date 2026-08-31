#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array

from ._model import (
    BinaryCardinalityFactorGroup,
    IsingFactorGroup,
    LogicalFactorGroup,
)


def _reduce(values: Array, mode: Literal["sum", "max"], /, *, axis=-1) -> Array:
    return (
        jsp.special.logsumexp(values, axis=axis)
        if mode == "sum"
        else jnp.max(values, axis=axis)
    )


def _normalize(values: Array) -> Array:
    maximum = jnp.max(values, axis=-1, keepdims=True)
    return jnp.where(jnp.isfinite(maximum), values - maximum, -jnp.inf)


def _parity_distribution(
    messages: list[Array],
    count: int,
    mode: Literal["sum", "max"],
    dtype,
):
    negative = jnp.full((count,), -jnp.inf, dtype=dtype)
    positive = jnp.zeros((count,), dtype=dtype)
    for values in messages:
        next_negative = _reduce(
            jnp.stack(
                [negative + values[:, 1], positive + values[:, 0]],
                axis=-1,
            ),
            mode,
        )
        next_positive = _reduce(
            jnp.stack(
                [positive + values[:, 1], negative + values[:, 0]],
                axis=-1,
            ),
            mode,
        )
        negative, positive = next_negative, next_positive
    return jnp.stack([negative, positive], axis=-1)


def ising_factor_messages(
    group: IsingFactorGroup,
    incoming: tuple[Array, ...],
    /,
    *,
    mode: Literal["sum", "max"],
) -> tuple[Array, ...]:
    """Compute p-spin messages through a two-state parity dynamic program."""
    outputs = []
    spins = jnp.asarray([-1.0, 1.0], dtype=incoming[0].dtype)
    weights = jnp.asarray(group.weights, dtype=incoming[0].dtype)
    for target in range(len(incoming)):
        parity = _parity_distribution(
            [value for index, value in enumerate(incoming) if index != target],
            int(incoming[target].shape[0]),
            mode,
            incoming[target].dtype,
        )
        scores = []
        for target_state in range(2):
            potential = weights[:, None] * spins[target_state] * spins[None, :]
            scores.append(_reduce(parity + potential, mode))
        outputs.append(_normalize(jnp.stack(scores, axis=-1)))
    return tuple(outputs)


def _logsubexp(total: Array, excluded: Array) -> Array:
    ratio = jnp.exp(jnp.minimum(excluded - total, 0.0))
    return jnp.where(total > excluded, total + jnp.log1p(-ratio), -jnp.inf)


def logical_factor_messages(
    group: LogicalFactorGroup,
    incoming: tuple[Array, ...],
    /,
    *,
    mode: Literal["sum", "max"],
) -> tuple[Array, ...]:
    """Compute OR/AND messages without enumerating parent configurations."""
    parents = incoming[:-1]
    child = incoming[-1]
    outputs = []
    trigger_state = 1 if group.kind == "or" else 0
    neutral_state = 1 - trigger_state
    child_trigger = 1 if group.kind == "or" else 0
    child_neutral = 1 - child_trigger

    for target in range(len(parents)):
        others = [value for index, value in enumerate(parents) if index != target]
        neutral = sum(
            (value[:, neutral_state] for value in others),
            start=jnp.zeros_like(child[:, 0]),
        )
        if mode == "sum":
            total = sum(
                (jsp.special.logsumexp(value, axis=-1) for value in others),
                start=jnp.zeros_like(child[:, 0]),
            )
            triggered = _logsubexp(total, neutral)
        else:
            total = sum(
                (jnp.max(value, axis=-1) for value in others),
                start=jnp.zeros_like(child[:, 0]),
            )
            # Max with at least one trigger: force each candidate trigger once and maximize.
            candidates = [
                value[:, trigger_state]
                + sum(
                    (
                        jnp.max(other, axis=-1)
                        for other_index, other in enumerate(others)
                        if other_index != index
                    ),
                    start=jnp.zeros_like(child[:, 0]),
                )
                for index, value in enumerate(others)
            ]
            triggered = (
                jnp.max(jnp.stack(candidates), axis=0)
                if candidates
                else jnp.full_like(total, -jnp.inf)
            )
        neutral_score = _reduce(
            jnp.stack(
                [neutral + child[:, child_neutral], triggered + child[:, child_trigger]],
                axis=-1,
            ),
            mode,
        )
        trigger_score = total + child[:, child_trigger]
        values = jnp.stack([neutral_score, trigger_score], axis=-1)
        if trigger_state == 0:
            values = values[:, ::-1]
        outputs.append(_normalize(values))

    all_neutral = sum(
        (value[:, neutral_state] for value in parents),
        start=jnp.zeros_like(child[:, 0]),
    )
    if mode == "sum":
        all_total = sum(
            (jsp.special.logsumexp(value, axis=-1) for value in parents),
            start=jnp.zeros_like(child[:, 0]),
        )
        any_trigger = _logsubexp(all_total, all_neutral)
    else:
        all_total = sum(
            (jnp.max(value, axis=-1) for value in parents),
            start=jnp.zeros_like(child[:, 0]),
        )
        candidates = []
        for index, value in enumerate(parents):
            candidates.append(
                value[:, trigger_state]
                + sum(
                    (
                        jnp.max(other, axis=-1)
                        for other_index, other in enumerate(parents)
                        if other_index != index
                    ),
                    start=jnp.zeros_like(child[:, 0]),
                )
            )
        any_trigger = jnp.max(jnp.stack(candidates), axis=0)
    child_values = jnp.stack([all_neutral, any_trigger], axis=-1)
    if child_trigger == 0:
        child_values = child_values[:, ::-1]
    outputs.append(_normalize(child_values))
    return tuple(outputs)


def _count_convolution(left: Array, right: Array, mode: Literal["sum", "max"]):
    output = []
    for count in range(int(left.shape[-1] + right.shape[-1] - 1)):
        terms = []
        for left_count in range(int(left.shape[-1])):
            right_count = count - left_count
            if 0 <= right_count < int(right.shape[-1]):
                terms.append(left[:, left_count] + right[:, right_count])
        output.append(_reduce(jnp.stack(terms, axis=-1), mode))
    return jnp.stack(output, axis=-1)


def cardinality_factor_messages(
    group: BinaryCardinalityFactorGroup,
    incoming: tuple[Array, ...],
    /,
    *,
    mode: Literal["sum", "max"],
) -> tuple[Array, ...]:
    """Compute binary cardinality messages by count dynamic programming."""
    outputs = []
    factor_count = int(incoming[0].shape[0])
    for target in range(len(incoming)):
        distribution = jnp.zeros((factor_count, 1), dtype=incoming[target].dtype)
        for index, values in enumerate(incoming):
            if index != target:
                distribution = _count_convolution(distribution, values, mode)
        scores = []
        for target_state in range(2):
            potentials = jnp.asarray(
                group.log_count_potentials,
                dtype=incoming[target].dtype,
            )[:, target_state : target_state + distribution.shape[-1]]
            scores.append(_reduce(distribution + potentials, mode))
        outputs.append(_normalize(jnp.stack(scores, axis=-1)))
    return tuple(outputs)


__all__ = [
    "cardinality_factor_messages",
    "ising_factor_messages",
    "logical_factor_messages",
]
