#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


PhysicalityStatus: TypeAlias = Literal["valid", "invalid", "unknown"]


class ApproximationAxis(StrictModule):
    name: str = eqx.field(static=True)
    value: Array
    parent_value: Array | None
    units: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        value: ArrayLike,
        /,
        *,
        parent_value: ArrayLike | None = None,
        units: str = "dimensionless",
    ):
        identifier = str(name)
        if not identifier:
            raise ValueError("Approximation-axis name must be non-empty.")
        self.name = identifier
        self.value = jnp.asarray(value)
        self.parent_value = None if parent_value is None else jnp.asarray(parent_value)
        self.units = str(units)


class OpenSystemApproximationEvidence(StrictModule):
    axes: tuple[ApproximationAxis, ...]
    local_error: Array
    statistical_error: Array
    valid: Array
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        representation_id: str,
        axes: Sequence[ApproximationAxis],
        /,
        *,
        local_error: ArrayLike = 0.0,
        statistical_error: ArrayLike = 0.0,
        valid: ArrayLike = True,
    ):
        self.representation_id = str(representation_id)
        self.axes = tuple(axes)
        self.local_error = jnp.asarray(local_error)
        self.statistical_error = jnp.asarray(statistical_error)
        self.valid = jnp.asarray(valid, dtype=bool)


class OpenSystemPhysicalityEvidence(StrictModule):
    trace_residual: Array
    hermiticity_residual: Array
    positivity_margin: Array
    channel_cp_margin: Array
    valid: Array
    status: PhysicalityStatus = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        trace_residual: ArrayLike = jnp.nan,
        hermiticity_residual: ArrayLike = jnp.nan,
        positivity_margin: ArrayLike = jnp.nan,
        channel_cp_margin: ArrayLike = jnp.nan,
        status: PhysicalityStatus = "unknown",
    ):
        if status not in ("valid", "invalid", "unknown"):
            raise ValueError("Unknown physicality status.")
        self.trace_residual = jnp.asarray(trace_residual)
        self.hermiticity_residual = jnp.asarray(hermiticity_residual)
        self.positivity_margin = jnp.asarray(positivity_margin)
        self.channel_cp_margin = jnp.asarray(channel_cp_margin)
        self.status = status
        self.valid = jnp.asarray(status == "valid")


class QuantumGeneratorAction(StrictModule):
    action_function: Callable[[Array, Array, Any], Array]
    representation_id: str = eqx.field(static=True)
    generator_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable[[Array, Array, Any], Array],
        /,
        *,
        representation_id: str,
        generator_id: str,
    ):
        if not callable(action):
            raise TypeError("Generator action must be callable.")
        self.action_function = action
        self.representation_id = str(representation_id)
        self.generator_id = str(generator_id)

    def __call__(self, time: Array, state: Array, args: Any = None, /) -> Array:
        result = jnp.asarray(self.action_function(time, state, args))
        if result.shape != state.shape:
            raise ValueError("Generator action must preserve the state shape.")
        return result


class QuantumObservablePlan(StrictModule):
    reducer: Callable[[Any], Array]
    observable_id: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)

    def __init__(
        self,
        reducer: Callable[[Any], Array],
        /,
        *,
        observable_id: str,
        exact: bool,
    ):
        if not callable(reducer):
            raise TypeError("Observable reducer must be callable.")
        self.reducer = reducer
        self.observable_id = str(observable_id)
        self.exact = bool(exact)

    def __call__(self, state: Any, /) -> Array:
        return jnp.asarray(self.reducer(state))


class OpenSystemRefinement(StrictModule):
    coarse_representation_id: str = eqx.field(static=True)
    fine_representation_id: str = eqx.field(static=True)
    axis: ApproximationAxis
    state_embedding: Callable[[Any], Any]

    def __init__(
        self,
        coarse_representation_id: str,
        fine_representation_id: str,
        axis: ApproximationAxis,
        state_embedding: Callable[[Any], Any],
        /,
    ):
        if not callable(state_embedding):
            raise TypeError("state_embedding must be callable.")
        self.coarse_representation_id = str(coarse_representation_id)
        self.fine_representation_id = str(fine_representation_id)
        self.axis = axis
        self.state_embedding = state_embedding

    def embed(self, state: Any, /) -> Any:
        return self.state_embedding(state)


__all__ = [
    "ApproximationAxis",
    "OpenSystemApproximationEvidence",
    "OpenSystemPhysicalityEvidence",
    "OpenSystemRefinement",
    "PhysicalityStatus",
    "QuantumGeneratorAction",
    "QuantumObservablePlan",
]
