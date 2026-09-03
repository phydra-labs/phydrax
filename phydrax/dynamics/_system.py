#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from ._layout import InputLayout, StateLayout


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


class DiscreteTransitionResult(StrictModule):
    """Candidate and accepted states with explicit transition termination evidence."""

    candidate_state: Array
    accepted_state: Array
    successful: Array
    status: Array

    def __init__(
        self,
        candidate_state: ArrayLike,
        accepted_state: ArrayLike,
        successful: ArrayLike,
        status: ArrayLike,
    ):
        candidate = _inexact(candidate_state)
        accepted = _inexact(accepted_state)
        successful_array = jnp.asarray(successful, dtype=bool)
        status_array = jnp.asarray(status, dtype=jnp.int32)
        if candidate.shape != accepted.shape:
            raise ValueError(
                "Discrete transition candidate_state and accepted_state must "
                "have matching shapes."
            )
        if successful_array.shape != ():
            raise ValueError("Discrete transition successful must be scalar.")
        if status_array.shape != ():
            raise ValueError("Discrete transition status must be scalar.")
        self.candidate_state = candidate
        self.accepted_state = accepted
        self.successful = successful_array
        self.status = status_array


AutonomousContinuousVectorField: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
InputContinuousVectorField: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
AutonomousDiscreteTransition: TypeAlias = Callable[
    ["DiscreteStepContext", Array, Any], ArrayLike | DiscreteTransitionResult
]
InputDiscreteTransition: TypeAlias = Callable[
    ["DiscreteStepContext", Array, Array, Any],
    ArrayLike | DiscreteTransitionResult,
]
SystemVectorField: TypeAlias = (
    AutonomousContinuousVectorField | InputContinuousVectorField
)
SystemTransition: TypeAlias = AutonomousDiscreteTransition | InputDiscreteTransition


class DiscreteStepContext(StrictModule):
    """Canonical source/target interval and traced step index for one transition."""

    source: Array
    target: Array
    step_index: Array

    def __init__(
        self,
        source: ArrayLike,
        target: ArrayLike,
        step_index: ArrayLike,
        /,
    ):
        source_array = jnp.asarray(source)
        target_array = jnp.asarray(target)
        index_array = jnp.asarray(step_index, dtype=jnp.int32)
        if (
            source_array.shape != ()
            or target_array.shape != ()
            or index_array.shape != ()
        ):
            raise ValueError("Discrete step context values must be scalar.")
        self.source = source_array
        self.target = target_array
        self.step_index = index_array

    @property
    def duration(self) -> Array:
        return self.target - self.source


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

    transition: Callable[..., ArrayLike | DiscreteTransitionResult] = eqx.field(
        static=True
    )
    state_layout: StateLayout
    input_layout: InputLayout | None
    system_id: str = eqx.field(static=True)
    step_size: float | None = eqx.field(static=True)
    step_rtol: float = eqx.field(static=True)
    step_atol: float = eqx.field(static=True)
    minimum_step_size: float | None = eqx.field(static=True)
    maximum_step_size: float | None = eqx.field(static=True)

    def __init__(
        self,
        transition: SystemTransition,
        /,
        *,
        state_layout: StateLayout,
        input_layout: InputLayout | None = None,
        system_id: str,
        step_size: float | None = None,
        step_rtol: float = 1e-7,
        step_atol: float = 1e-12,
        minimum_step_size: float | None = None,
        maximum_step_size: float | None = None,
    ):
        if not callable(transition):
            raise TypeError("DiscreteSystem transition must be callable.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        resolved_step = None if step_size is None else float(step_size)
        relative_tolerance = float(step_rtol)
        absolute_tolerance = float(step_atol)
        minimum_step = None if minimum_step_size is None else float(minimum_step_size)
        maximum_step = None if maximum_step_size is None else float(maximum_step_size)
        if resolved_step is not None and (
            not np.isfinite(resolved_step) or resolved_step <= 0.0
        ):
            raise ValueError("step_size must be finite and positive or None.")
        if (
            not np.isfinite(relative_tolerance)
            or relative_tolerance < 0.0
            or not np.isfinite(absolute_tolerance)
            or absolute_tolerance < 0.0
        ):
            raise ValueError("step_rtol and step_atol must be finite and nonnegative.")
        if minimum_step is not None and (
            not np.isfinite(minimum_step) or minimum_step <= 0.0
        ):
            raise ValueError("minimum_step_size must be finite and positive.")
        if maximum_step is not None and (
            not np.isfinite(maximum_step) or maximum_step <= 0.0
        ):
            raise ValueError("maximum_step_size must be finite and positive.")
        if (
            minimum_step is not None
            and maximum_step is not None
            and minimum_step > maximum_step
        ):
            raise ValueError("minimum_step_size must not exceed maximum_step_size.")
        self.transition = transition
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.system_id = _identifier(system_id, "DiscreteSystem system_id")
        self.step_size = resolved_step
        self.step_rtol = relative_tolerance
        self.step_atol = absolute_tolerance
        self.minimum_step_size = minimum_step
        self.maximum_step_size = maximum_step

    def evaluate_result(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> DiscreteTransitionResult:
        """Evaluate one transition without discarding candidate/status evidence."""
        state_array = _inexact(state)
        if state_array.shape != self.state_layout.shape:
            raise ValueError(
                f"state must have shape {self.state_layout.shape}; got {state_array.shape}."
            )
        if not isinstance(context, DiscreteStepContext):
            raise TypeError("DiscreteSystem evaluation requires DiscreteStepContext.")
        duration = context.duration
        duration_valid = jnp.isfinite(context.source) & jnp.isfinite(context.target)
        duration_valid = duration_valid & (duration > 0.0)
        if self.step_size is not None:
            duration_valid = duration_valid & jnp.isclose(
                duration,
                self.step_size,
                rtol=self.step_rtol,
                atol=self.step_atol,
            )
        if self.minimum_step_size is not None:
            duration_valid = duration_valid & (duration >= self.minimum_step_size)
        if self.maximum_step_size is not None:
            duration_valid = duration_valid & (duration <= self.maximum_step_size)
        checked_source = eqx.error_if(
            context.source,
            ~duration_valid,
            "Discrete step interval is invalid for the declared step_size or system bounds.",
        )
        context = DiscreteStepContext(
            checked_source,
            context.target,
            context.step_index,
        )
        if self.input_layout is None:
            if inputs is not None:
                raise ValueError("An autonomous DiscreteSystem does not accept inputs.")
            value = self.transition(context, state_array, args)
        else:
            if inputs is None:
                raise ValueError("This DiscreteSystem requires explicit inputs.")
            input_array = _inexact(inputs)
            if input_array.shape != self.input_layout.shape:
                raise ValueError(
                    f"inputs must have shape {self.input_layout.shape}; got {input_array.shape}."
                )
            value = self.transition(context, state_array, input_array, args)
        if isinstance(value, DiscreteTransitionResult):
            if value.candidate_state.shape != self.state_layout.shape:
                raise ValueError(
                    "DiscreteSystem transition candidate_state has shape "
                    f"{value.candidate_state.shape}; expected {self.state_layout.shape}."
                )
            if value.accepted_state.shape != self.state_layout.shape:
                raise ValueError(
                    "DiscreteSystem transition accepted_state has shape "
                    f"{value.accepted_state.shape}; expected {self.state_layout.shape}."
                )
            return value
        output = _inexact(value)
        if output.shape != self.state_layout.shape:
            raise ValueError(
                "DiscreteSystem transition returned shape "
                f"{output.shape}; expected {self.state_layout.shape}."
            )
        return DiscreteTransitionResult(
            output,
            output,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def evaluate(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        return self.evaluate_result(
            context,
            state,
            args,
            inputs=inputs,
        ).accepted_state

    def __call__(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        args: Any = None,
        /,
        *,
        inputs: ArrayLike | None = None,
    ) -> Array:
        return self.evaluate(context, state, args, inputs=inputs)


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

    @abc.abstractmethod
    def evaluate_step(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        if not isinstance(context, DiscreteStepContext):
            raise TypeError("evaluate_step requires DiscreteStepContext.")
        return self.evaluate(context.source, state, args)

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

    policy: Callable[..., ArrayLike]
    input_layout: InputLayout
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: Callable[..., ArrayLike],
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

    def evaluate_step(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        result = _inexact(self.policy(context, _inexact(state), args))
        if result.shape != self.input_layout.shape:
            raise ValueError(
                f"Input policy returned shape {result.shape}; expected {self.input_layout.shape}."
            )
        return result


class HeldInputPolicy(AbstractInputPolicy):
    """State-independent interval values on one strictly increasing time grid."""

    times: Array
    values: Array
    input_layout: InputLayout
    node_side: Literal["left", "right"] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        input_layout: InputLayout,
        node_side: Literal["left", "right"] = "left",
        policy_id: str,
    ):
        if not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout.")
        if node_side not in ("left", "right"):
            raise ValueError("node_side must be 'left' or 'right'.")
        times_ = _inexact(times)
        values_ = _inexact(values)
        if times_.ndim != 1 or int(times_.size) < 2:
            raise ValueError("HeldInputPolicy times must contain at least two nodes.")
        expected = (int(times_.size) - 1,) + input_layout.shape
        if values_.shape != expected:
            raise ValueError(
                f"HeldInputPolicy values must have shape {expected}; got {values_.shape}."
            )
        times_ = eqx.error_if(
            times_,
            jnp.any(~jnp.isfinite(times_)) | jnp.any(jnp.diff(times_) <= 0.0),
            "HeldInputPolicy times must be finite and strictly increasing.",
        )
        values_ = eqx.error_if(
            values_,
            jnp.any(~jnp.isfinite(values_)),
            "HeldInputPolicy values must be finite.",
        )
        self.times = times_
        self.values = values_
        self.input_layout = input_layout
        self.node_side = node_side
        self.policy_id = _identifier(policy_id, "HeldInputPolicy policy_id")

    def evaluate(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        del state, args
        time = jnp.asarray(coordinate, dtype=self.times.dtype)
        if time.shape != ():
            raise ValueError("HeldInputPolicy coordinate must be scalar.")
        time = eqx.error_if(
            time,
            ~jnp.isfinite(time) | (time < self.times[0]) | (time > self.times[-1]),
            "HeldInputPolicy coordinate lies outside its time grid.",
        )
        side = "left" if self.node_side == "left" else "right"
        index = jnp.searchsorted(self.times, time, side=side) - 1
        return self.values[jnp.clip(index, 0, int(self.values.shape[0]) - 1)]

    def evaluate_step(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        return self.evaluate(context.source, state, args)


__all__ = [
    "AbstractInputPolicy",
    "AutonomousContinuousVectorField",
    "AutonomousDiscreteTransition",
    "CallableInputPolicy",
    "HeldInputPolicy",
    "ContinuousSystem",
    "DiscreteStepContext",
    "DiscreteSystem",
    "DiscreteTransitionResult",
    "InputContinuousVectorField",
    "InputDiscreteTransition",
    "SystemTransition",
    "SystemVectorField",
]
