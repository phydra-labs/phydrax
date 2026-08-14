#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from ._layout import InputLayout, StateLayout


AutonomousContinuousVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
InputContinuousVectorField: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
AutonomousDiscreteTransition: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
InputDiscreteTransition: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
SystemVectorField: TypeAlias = (
    AutonomousContinuousVectorField | InputContinuousVectorField
)
SystemTransition: TypeAlias = AutonomousDiscreteTransition | InputDiscreteTransition


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


class ContinuousSystem(StrictModule):
    """A continuous-time local evolution law independent of numerical integration."""

    vector_field: Callable[..., ArrayLike]
    state_layout: StateLayout
    input_layout: InputLayout | None
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: SystemVectorField,
        /,
        *,
        state_layout: StateLayout,
        input_layout: InputLayout | None = None,
        system_id: str,
    ):
        if not callable(vector_field):
            raise TypeError("ContinuousSystem vector_field must be callable.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        self.vector_field = vector_field
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.system_id = _identifier(system_id, "ContinuousSystem system_id")

    def evaluate(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        state_array = _inexact(state)
        if state_array.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_array.shape}."
            )
        time_array = jnp.asarray(time)
        if time_array.shape != ():
            raise ValueError("time must be scalar for one ContinuousSystem evaluation.")
        if self.input_layout is None:
            if inputs is not None:
                raise ValueError("An autonomous ContinuousSystem does not accept inputs.")
            value = self.vector_field(time_array, state_array, args)
        else:
            if inputs is None:
                raise ValueError("This ContinuousSystem requires explicit inputs.")
            input_array = _inexact(inputs)
            if input_array.shape != self.input_layout.shape:
                raise ValueError(
                    f"inputs must have shape {self.input_layout.shape}; got {input_array.shape}."
                )
            value = self.vector_field(time_array, state_array, input_array, args)
        output = _inexact(value)
        if output.shape != self.state_layout.shape:
            raise ValueError(
                "ContinuousSystem vector_field returned shape "
                f"{output.shape}; expected {self.state_layout.shape}."
            )
        return output

    def __call__(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        return self.evaluate(time, state, args, inputs=inputs)


class DiscreteSystem(StrictModule):
    """A discrete transition law independent of rollout and analysis policy."""

    transition: Callable[..., ArrayLike]
    state_layout: StateLayout
    input_layout: InputLayout | None
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition: SystemTransition,
        /,
        *,
        state_layout: StateLayout,
        input_layout: InputLayout | None = None,
        system_id: str,
    ):
        if not callable(transition):
            raise TypeError("DiscreteSystem transition must be callable.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        self.transition = transition
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.system_id = _identifier(system_id, "DiscreteSystem system_id")

    def evaluate(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        state_array = _inexact(state)
        if state_array.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_array.shape}."
            )
        coordinate_array = jnp.asarray(coordinate)
        if coordinate_array.shape != ():
            raise ValueError(
                "coordinate must be scalar for one DiscreteSystem evaluation."
            )
        if self.input_layout is None:
            if inputs is not None:
                raise ValueError("An autonomous DiscreteSystem does not accept inputs.")
            value = self.transition(coordinate_array, state_array, args)
        else:
            if inputs is None:
                raise ValueError("This DiscreteSystem requires explicit inputs.")
            input_array = _inexact(inputs)
            if input_array.shape != self.input_layout.shape:
                raise ValueError(
                    f"inputs must have shape {self.input_layout.shape}; got {input_array.shape}."
                )
            value = self.transition(coordinate_array, state_array, input_array, args)
        output = _inexact(value)
        if output.shape != self.state_layout.shape:
            raise ValueError(
                "DiscreteSystem transition returned shape "
                f"{output.shape}; expected {self.state_layout.shape}."
            )
        return output

    def __call__(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        return self.evaluate(coordinate, state, args, inputs=inputs)


class AbstractInputPolicy(StrictModule):
    """State-aware input policy bound into a pathwise evolution."""

    input_layout: AbstractAttribute[InputLayout]
    policy_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError

    def __call__(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        return self.evaluate(coordinate, state, args)


class CallableInputPolicy(AbstractInputPolicy):
    """Callable input policy with an explicit layout and identity."""

    policy: Callable[[Array, Array, Any], ArrayLike]
    input_layout: InputLayout
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: Callable[[Array, Array, Any], ArrayLike],
        /,
        *,
        input_layout: InputLayout,
        policy_id: str,
    ):
        if not callable(policy):
            raise TypeError("CallableInputPolicy policy must be callable.")
        if not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout.")
        self.policy = policy
        self.input_layout = input_layout
        self.policy_id = _identifier(policy_id, "CallableInputPolicy policy_id")

    def evaluate(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        result = _inexact(self.policy(jnp.asarray(coordinate), _inexact(state), args))
        if result.shape != self.input_layout.shape:
            raise ValueError(
                f"Input policy returned shape {result.shape}; expected {self.input_layout.shape}."
            )
        return result


__all__ = [
    "AbstractInputPolicy",
    "AutonomousContinuousVectorField",
    "AutonomousDiscreteTransition",
    "CallableInputPolicy",
    "ContinuousSystem",
    "DiscreteSystem",
    "InputContinuousVectorField",
    "InputDiscreteTransition",
    "SystemTransition",
    "SystemVectorField",
]
