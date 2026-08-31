#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ._strict import StrictModule
from .domain import DomainFunction


class StateTimeScoreField(StrictModule):
    """Validated state-shaped score field with optional time dependence."""
    function: DomainFunction
    state_label: str = eqx.field(static=True)
    time_label: str = eqx.field(static=True)
    context_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        function: DomainFunction,
        /,
        *,
        state_label: str,
        time_label: str,
        context_labels=(),
    ):
        if not isinstance(function, DomainFunction):
            raise TypeError("score field must be a DomainFunction.")
        if not state_label or not time_label or state_label == time_label:
            raise ValueError("state_label and time_label must be distinct and non-empty.")
        contexts = tuple(str(label) for label in context_labels)
        if any(not label for label in contexts) or len(set(contexts)) != len(contexts):
            raise ValueError("context_labels must contain unique non-empty labels.")
        if state_label in contexts or time_label in contexts:
            raise ValueError("Context labels must be distinct from state and time labels.")
        allowed = {state_label, time_label, *contexts}
        unknown = tuple(label for label in function.deps if label not in allowed)
        if unknown or state_label not in function.deps:
            raise ValueError(
                "score field must depend on state and only declared time/context labels."
            )
        self.function = function
        self.state_label = str(state_label)
        self.time_label = str(time_label)
        self.context_labels = contexts

    def __call__(
        self,
        state: ArrayLike,
        time: ArrayLike,
        /,
        *,
        key: Key[Array, ""] | None = None,
        context: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        state_array = jnp.asarray(state)
        time_array = jnp.asarray(time)
        if time_array.shape != ():
            raise ValueError("One score-field evaluation requires scalar time.")
        supplied = {} if context is None else dict(context)
        required_context = tuple(
            dependency
            for dependency in self.function.deps
            if dependency not in (self.state_label, self.time_label)
        )
        if set(supplied) != set(required_context):
            raise ValueError(
                f"Score context keys must be exactly {required_context}; got {tuple(supplied)}."
            )
        arguments = tuple(
            state_array
            if dependency == self.state_label
            else (
                time_array
                if dependency == self.time_label
                else jnp.asarray(supplied[dependency])
            )
            for dependency in self.function.deps
        )
        value = jnp.asarray(self.function.func(*arguments, key=key))
        if value.shape != state_array.shape:
            raise ValueError(
                "score field output must have the same shape as the state; "
                f"got {value.shape} and {state_array.shape}."
            )
        return value


def require_score_field(
    functions: Mapping[str, DomainFunction],
    name: str,
    /,
    *,
    state_label: str,
    time_label: str,
    context_labels=(),
) -> StateTimeScoreField:
    if name not in functions:
        raise KeyError(f"Missing score field {name!r}.")
    return StateTimeScoreField(
        functions[name],
        state_label=state_label,
        time_label=time_label,
        context_labels=context_labels,
    )


__all__ = ["StateTimeScoreField", "require_score_field"]
