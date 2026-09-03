#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from math import prod
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...dynamics._system import (
    DiscreteStepContext,
    DiscreteSystem,
    DiscreteTransitionResult,
)


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _shape(value: tuple[int, ...], owner: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _key_data(key: ArrayLike, /) -> Array:
    data = jnp.asarray(jax.random.key_data(key), dtype=jnp.uint32)
    if data.shape != (2,):
        raise ValueError("Robot environment PRNG keys must have shape (2,).")
    return data

def _type_id(value: Any, /) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _provenance_value(value: Any, owner: str, /) -> Any:
    if eqx.is_array(value):
        return {"array": array_tree_fingerprint(value)}
    if isinstance(value, np.generic):
        return _provenance_value(value.item(), owner)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{owner} contains nonfinite numeric configuration.")
        return value
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"{owner} contains nonfinite numeric configuration.")
        return {"type": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.dtype):
        return {"type": "dtype", "value": value.str}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _provenance_value(item, f"{owner}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _provenance_value(item, f"{owner}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, Mapping):
        entries = [
            [
                _provenance_value(key, f"{owner} key"),
                _provenance_value(item, f"{owner}[{key!r}]"),
            ]
            for key, item in value.items()
        ]
        entries.sort(key=lambda entry: canonical_fingerprint(entry[0]))
        return {"type": _type_id(value), "entries": entries}
    if isinstance(value, (set, frozenset)):
        items = [_provenance_value(item, owner) for item in value]
        items.sort(key=canonical_fingerprint)
        return {"type": _type_id(value), "items": items}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": _type_id(value),
            "fields": [
                {
                    "name": field.name,
                    "value": _provenance_value(
                        value.__dict__[field.name],
                        f"{owner}.{field.name}",
                    ),
                }
                for field in fields(value)
            ],
        }
    if isinstance(value, type):
        return {"type": "class", "value": _type_id(value)}
    if callable(value):
        return {"callable": _callable_provenance(value, owner)}
    raise TypeError(
        f"{owner} contains unsupported provenance value of type {_type_id(value)}."
    )

def _callable_provenance(value: Callable[..., Any], owner: str, /) -> Any:
    if isinstance(value, (FunctionType, BuiltinFunctionType)):
        return {
            "type": "function",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    if isinstance(value, MethodType):
        bound_owner = value.__self__
        function = cast(FunctionType, value.__func__)
        return {
            "type": "method",
            "module": function.__module__,
            "qualname": function.__qualname__,
            "owner": (
                _provenance_value(bound_owner, f"{owner} owner")
                if is_dataclass(bound_owner)
                else _type_id(bound_owner)
            ),
        }
    return {"type": _type_id(value)}


def _tree_all_finite(tree: Any, owner: str, /) -> Array:
    finite: list[Array] = []
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if not (
            jnp.issubdtype(array.dtype, jnp.number)
            or jnp.issubdtype(array.dtype, jnp.bool_)
        ):
            raise TypeError(f"{owner} must be a PyTree of numeric array leaves.")
        finite.append(jnp.all(jnp.isfinite(array)))
    if not finite:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(tuple(finite)))


def _finite_or_zero_tree(tree: Any, /) -> Any:
    return jax.tree.map(
        lambda value: jnp.where(
            jnp.isfinite(value),
            value,
            jnp.zeros_like(value),
        ),
        tree,
    )

def _error_if_tree(tree: Any, predicate: Array, message: str, /) -> Any:
    return jax.tree.map(
        lambda value: eqx.error_if(value, predicate, message),
        tree,
    )


def _select_tree(predicate: Array, candidate: Any, source: Any, /) -> Any:
    return jax.tree.map(
        lambda proposed, previous: jnp.where(predicate, proposed, previous),
        candidate,
        source,
    )


class RobotTaskEvaluation(StrictModule):
    """Task-owned outputs at one accepted plant and task state."""

    observation: Array
    terminated: Array
    descriptor: Array


class RobotTaskTransition(StrictModule):
    """Task-owned candidate state and outputs for one plant transition."""

    task_state: Any
    observation: Array
    reward_components: Array
    terminated: Array
    descriptor: Array


class AbstractRobotTask(StrictModule):
    """Immutable task contract independent of plant mechanics and wrappers."""

    __strict_abstract__ = True
    task_id: AbstractAttribute[str]
    observation_shape: AbstractAttribute[tuple[int, ...]]
    reward_component_names: AbstractAttribute[tuple[str, ...]]
    descriptor_shape: AbstractAttribute[tuple[int, ...]]

    @abstractmethod
    def initialize(self, plant_state: Array, key: Array, /) -> Any:
        """Return fixed-structure task state for a freshly initialized plant."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        plant_state: Array,
        task_state: Any,
        /,
    ) -> RobotTaskEvaluation:
        """Observe and classify one accepted task state without changing it."""
        raise NotImplementedError

    @abstractmethod
    def transition(
        self,
        context: DiscreteStepContext,
        source_plant_state: Array,
        accepted_plant_state: Array,
        action: Array,
        task_state: Any,
        key: Array,
        /,
    ) -> RobotTaskTransition:
        """Evaluate reward and termination for one mechanics-accepted plant step."""
        raise NotImplementedError


class RobotEnvironmentWrapperTransition(StrictModule):
    """One wrapper's candidate state and administrative truncation decision."""

    wrapper_state: Any
    truncated: Array


class AbstractRobotEnvironmentWrapper(StrictModule):
    """Ordered immutable owner of repetition and episode administration."""

    __strict_abstract__ = True
    wrapper_id: AbstractAttribute[str]
    action_repeat: AbstractAttribute[int]
    horizon: AbstractAttribute[int | None]
    auto_reset: AbstractAttribute[bool]

    @abstractmethod
    def initialize(
        self,
        plant_state: Array,
        task_state: Any,
        key: Array,
        /,
    ) -> Any:
        """Return this wrapper's fixed-structure state at environment reset."""
        raise NotImplementedError

    @abstractmethod
    def transition(
        self,
        context: DiscreteStepContext,
        wrapper_state: Any,
        plant_state: Array,
        task_state: Any,
        observation: Array,
        terminated: Array,
        key: Array,
        /,
    ) -> RobotEnvironmentWrapperTransition:
        """Advance administrative state and report non-domain truncation."""
        raise NotImplementedError


class RobotEnvironmentState(StrictModule):
    """Complete accepted runtime state for one immutable robot environment."""

    plant_state: Array
    key: Array
    clock: Array
    step_index: Array
    task_state: Any
    wrapper_states: tuple[Any, ...]
    environment_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class RobotEnvironmentReset(StrictModule):
    """Fresh environment state and its task-owned outputs."""

    state: RobotEnvironmentState
    observation: Array
    terminated: Array
    descriptor: Array
    environment_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class RobotEnvironmentEvidence(StrictModule):
    """Fixed-work mechanics and atomic-commit evidence for one external step."""

    attempted: Array
    repeat_successful: Array
    repeat_status: Array
    accepted: Array
    rollback_applied: Array
    mechanics_successful: Array
    mechanics_status: Array
    source_step_index: Array
    candidate_step_index: Array
    environment_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    task_id: str = eqx.field(static=True)
    wrapper_ids: tuple[str, ...] = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class RobotEnvironmentTransition(StrictModule):
    """Candidate, accepted, and optional reset products of one external step."""

    candidate_state: RobotEnvironmentState
    accepted_state: RobotEnvironmentState
    reset_state: RobotEnvironmentState
    candidate_observation: Array
    observation: Array
    final_observation: Array
    total_reward: Array
    reward_components: Array
    terminated: Array
    truncated: Array
    descriptor: Array
    reset_performed: Array
    evidence: RobotEnvironmentEvidence
    reward_component_names: tuple[str, ...] = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted

    @property
    def mechanics_successful(self) -> Array:
        return self.evidence.mechanics_successful

    @property
    def mechanics_status(self) -> Array:
        return self.evidence.mechanics_status

    @property
    def provenance_id(self) -> str:
        return self.evidence.provenance_id


class _RobotSubstep(StrictModule):
    candidate_state: RobotEnvironmentState
    commit_state: RobotEnvironmentState
    task: RobotTaskTransition
    wrapper_truncated: Array
    outputs_finite: Array
    mechanics_successful: Array
    mechanics_status: Array


class PreparedRobotEnvironment(StrictModule, NonTrainableState):
    """Prepared composition of one array-state plant, task, and wrapper stack."""

    system: DiscreteSystem
    initializer: Callable[[Array], ArrayLike]
    task: AbstractRobotTask
    wrappers: tuple[AbstractRobotEnvironmentWrapper, ...]
    step_size: float = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)
    initializer_id: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: DiscreteSystem,
        initializer: Callable[[Array], ArrayLike],
        task: AbstractRobotTask,
        wrappers: tuple[AbstractRobotEnvironmentWrapper, ...] = (),
        /,
        *,
        initializer_id: str,
        step_size: float | None = None,
        environment_id: str | None = None,
    ):
        if not isinstance(system, DiscreteSystem):
            raise TypeError("system must be a DiscreteSystem.")
        if system.input_layout is None:
            raise ValueError("Robot environments require a system InputLayout.")
        if len(system.state_layout.shape) != 1:
            raise ValueError("Robot plant state must use a rank-1 StateLayout.")
        if len(system.input_layout.shape) != 1:
            raise ValueError("Robot actions must use a rank-1 InputLayout.")
        if any(role != "control" for role in system.input_layout.roles):
            raise ValueError("Every robot environment input must have role 'control'.")
        if not callable(initializer):
            raise TypeError("initializer must be callable.")
        if not isinstance(task, AbstractRobotTask):
            raise TypeError("task must be an AbstractRobotTask.")
        wrapper_tuple = tuple(wrappers)
        if any(
            not isinstance(wrapper, AbstractRobotEnvironmentWrapper)
            for wrapper in wrapper_tuple
        ):
            raise TypeError(
                "wrappers must contain only AbstractRobotEnvironmentWrapper instances."
            )

        task_id = _identifier(task.task_id, "task_id")
        observation_shape = _shape(task.observation_shape, "observation_shape")
        descriptor_shape = _shape(task.descriptor_shape, "descriptor_shape")
        reward_names = tuple(str(name) for name in task.reward_component_names)
        if not reward_names or any(not name for name in reward_names):
            raise ValueError("reward_component_names must contain non-empty names.")
        if len(set(reward_names)) != len(reward_names):
            raise ValueError("reward_component_names must be unique.")
        del observation_shape, descriptor_shape

        wrapper_ids: list[str] = []
        repeats: list[int] = []
        for wrapper in wrapper_tuple:
            wrapper_ids.append(_identifier(wrapper.wrapper_id, "wrapper_id"))
            repeat = int(wrapper.action_repeat)
            if repeat <= 0:
                raise ValueError("Wrapper action_repeat must be positive.")
            repeats.append(repeat)
            if wrapper.horizon is not None and int(wrapper.horizon) <= 0:
                raise ValueError("Wrapper horizon must be positive or None.")
            if not isinstance(wrapper.auto_reset, bool):
                raise TypeError("Wrapper auto_reset must be bool.")

        resolved_step = system.step_size if step_size is None else float(step_size)
        if resolved_step is None:
            raise ValueError(
                "step_size is required when the DiscreteSystem has no fixed step_size."
            )
        resolved_step = float(resolved_step)
        if not np.isfinite(resolved_step) or resolved_step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if system.step_size is not None and not np.isclose(
            resolved_step,
            system.step_size,
            rtol=system.step_rtol,
            atol=system.step_atol,
        ):
            raise ValueError("step_size must match the DiscreteSystem fixed step_size.")

        initialization_id = _identifier(initializer_id, "initializer_id")
        resolved_repeat = prod(repeats) if repeats else 1
        resolved_auto_reset = any(
            wrapper.auto_reset for wrapper in wrapper_tuple
        )
        provenance_id = canonical_fingerprint(
            {
                "system": {
                    "system_id": system.system_id,
                    "transition": _callable_provenance(
                        system.transition,
                        "system transition",
                    ),
                    "step_size": system.step_size,
                    "step_rtol": system.step_rtol,
                    "step_atol": system.step_atol,
                    "minimum_step_size": system.minimum_step_size,
                    "maximum_step_size": system.maximum_step_size,
                },
                "state_layout": _provenance_value(
                    system.state_layout,
                    "state_layout",
                ),
                "input_layout": _provenance_value(
                    system.input_layout,
                    "input_layout",
                ),
                "initializer": {
                    "initializer_id": initialization_id,
                    "callable": _callable_provenance(
                        initializer,
                        "initializer",
                    ),
                },
                "task": _provenance_value(task, "task"),
                "wrappers": _provenance_value(wrapper_tuple, "wrappers"),
                "step_size": resolved_step,
                "action_repeat": resolved_repeat,
                "horizons": [
                    wrapper.horizon for wrapper in wrapper_tuple
                ],
                "auto_reset": resolved_auto_reset,
                "prng_representation": {
                    "format": "legacy-key-data",
                    "shape": [2],
                    "dtype": "uint32",
                    "implementation": str(
                        jax.config.jax_default_prng_impl  # ty: ignore[unresolved-attribute]
                    ),
                },
            }
        )
        generated_id = f"robot-environment:{provenance_id}"

        self.system = system
        self.initializer = initializer
        self.task = task
        self.wrappers = wrapper_tuple
        self.step_size = resolved_step
        self.action_repeat = resolved_repeat
        self.auto_reset = resolved_auto_reset
        self.initializer_id = initialization_id
        self.environment_id = (
            generated_id
            if environment_id is None
            else _identifier(environment_id, "environment_id")
        )
        self.provenance_id = provenance_id

    def _check_task_evaluation(
        self,
        evaluation: RobotTaskEvaluation,
        /,
    ) -> RobotTaskEvaluation:
        if not isinstance(evaluation, RobotTaskEvaluation):
            raise TypeError("task.evaluate must return RobotTaskEvaluation.")
        observation = jnp.asarray(evaluation.observation)
        terminated = jnp.asarray(evaluation.terminated, dtype=bool)
        descriptor = jnp.asarray(evaluation.descriptor)
        if observation.shape != self.task.observation_shape:
            raise ValueError(
                "Task observation shape does not match task.observation_shape."
            )
        if terminated.shape != ():
            raise ValueError("Task terminated must be scalar.")
        if descriptor.shape != self.task.descriptor_shape:
            raise ValueError("Task descriptor shape does not match task.descriptor_shape.")
        return RobotTaskEvaluation(observation, terminated, descriptor)

    def _check_task_transition(
        self,
        transition: RobotTaskTransition,
        /,
    ) -> RobotTaskTransition:
        if not isinstance(transition, RobotTaskTransition):
            raise TypeError("task.transition must return RobotTaskTransition.")
        observation = jnp.asarray(transition.observation)
        rewards = jnp.asarray(transition.reward_components)
        terminated = jnp.asarray(transition.terminated, dtype=bool)
        descriptor = jnp.asarray(transition.descriptor)
        if observation.shape != self.task.observation_shape:
            raise ValueError(
                "Task observation shape does not match task.observation_shape."
            )
        if rewards.shape != (len(self.task.reward_component_names),):
            raise ValueError(
                "Task reward_components must have one scalar per declared name."
            )
        if terminated.shape != ():
            raise ValueError("Task terminated must be scalar.")
        if descriptor.shape != self.task.descriptor_shape:
            raise ValueError("Task descriptor shape does not match task.descriptor_shape.")
        return RobotTaskTransition(
            transition.task_state,
            observation,
            rewards,
            terminated,
            descriptor,
        )

    def _reset(self, key: Array, /) -> RobotEnvironmentReset:
        keys = jax.random.split(key, len(self.wrappers) + 3)
        plant_state = jnp.asarray(self.initializer(keys[1]))
        if not jnp.issubdtype(plant_state.dtype, jnp.inexact):
            plant_state = plant_state.astype(float)
        if plant_state.shape != self.system.state_layout.shape:
            raise ValueError(
                "initializer returned shape "
                f"{plant_state.shape}; expected {self.system.state_layout.shape}."
            )
        plant_state = _error_if_tree(
            plant_state,
            ~_tree_all_finite(plant_state, "initializer output"),
            "Robot environment initializer output must be finite.",
        )
        task_state = self.task.initialize(plant_state, keys[2])
        raw_evaluation = self.task.evaluate(plant_state, task_state)
        evaluation = self._check_task_evaluation(raw_evaluation)
        wrapper_states = tuple(
            wrapper.initialize(plant_state, task_state, wrapper_key)
            for wrapper, wrapper_key in zip(self.wrappers, keys[3:], strict=True)
        )
        reset_outputs_finite = _tree_all_finite(
            (task_state, wrapper_states, raw_evaluation),
            "robot environment reset outputs",
        )
        message = "Robot environment reset outputs must be finite."
        plant_state = _error_if_tree(
            plant_state,
            ~reset_outputs_finite,
            message,
        )
        task_state = _error_if_tree(
            task_state,
            ~reset_outputs_finite,
            message,
        )
        wrapper_states = _error_if_tree(
            wrapper_states,
            ~reset_outputs_finite,
            message,
        )
        evaluation = _error_if_tree(
            evaluation,
            ~reset_outputs_finite,
            message,
        )
        state = RobotEnvironmentState(
            plant_state,
            keys[0],
            jnp.asarray(0.0, dtype=plant_state.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            task_state,
            wrapper_states,
            self.environment_id,
            self.provenance_id,
        )
        return RobotEnvironmentReset(
            state,
            evaluation.observation,
            evaluation.terminated,
            evaluation.descriptor,
            self.environment_id,
            self.provenance_id,
        )

    def reset(self, key: ArrayLike, /) -> RobotEnvironmentReset:
        """Initialize the plant, task, wrappers, PRNG stream, and episode clock."""
        return self._reset(_key_data(key))

    def _substep(
        self,
        state: RobotEnvironmentState,
        action: Array,
        args: Any,
        /,
    ) -> _RobotSubstep:
        target_clock = state.clock + self.step_size
        clock_finite = _tree_all_finite(
            (state.clock, target_clock),
            "robot environment clock",
        )
        context = DiscreteStepContext(
            jnp.where(clock_finite, state.clock, jnp.zeros_like(state.clock)),
            jnp.where(
                clock_finite,
                target_clock,
                jnp.asarray(self.step_size, dtype=target_clock.dtype),
            ),
            state.step_index,
        )
        plant = self.system.evaluate_result(
            context,
            state.plant_state,
            args,
            inputs=action,
        )
        if not isinstance(plant, DiscreteTransitionResult):
            raise TypeError("DiscreteSystem.evaluate_result returned an invalid result.")
        accepted_plant_finite = _tree_all_finite(
            plant.accepted_state,
            "mechanics accepted state",
        )
        safe_accepted_plant = jnp.where(
            accepted_plant_finite,
            plant.accepted_state,
            jnp.zeros_like(plant.accepted_state),
        )
        keys = jax.random.split(state.key, len(self.wrappers) + 2)
        raw_task = self.task.transition(
            context,
            state.plant_state,
            safe_accepted_plant,
            action,
            state.task_state,
            keys[1],
        )
        task = self._check_task_transition(raw_task)
        task_outputs_finite = _tree_all_finite(
            raw_task,
            "task transition outputs",
        )
        safe_task = _finite_or_zero_tree(task)
        outputs_finite = (
            clock_finite & accepted_plant_finite & task_outputs_finite
        )
        wrapper_states: list[Any] = []
        wrapper_truncated = jnp.asarray(False)
        for wrapper, wrapper_state, wrapper_key in zip(
            self.wrappers,
            state.wrapper_states,
            keys[2:],
            strict=True,
        ):
            update = wrapper.transition(
                context,
                wrapper_state,
                safe_accepted_plant,
                safe_task.task_state,
                safe_task.observation,
                safe_task.terminated,
                wrapper_key,
            )
            if not isinstance(update, RobotEnvironmentWrapperTransition):
                raise TypeError(
                    "wrapper.transition must return "
                    "RobotEnvironmentWrapperTransition."
                )
            outputs_finite = outputs_finite & _tree_all_finite(
                update,
                "wrapper transition outputs",
            )
            truncated = jnp.asarray(update.truncated, dtype=bool)
            if truncated.shape != ():
                raise ValueError("Wrapper truncated must be scalar.")
            wrapper_states.append(update.wrapper_state)
            wrapper_truncated = wrapper_truncated | truncated
            if wrapper.horizon is not None:
                wrapper_truncated = wrapper_truncated | (
                    context.step_index + 1 >= int(wrapper.horizon)
                )

        runtime_state = (
            keys[0],
            target_clock,
            context.step_index + 1,
            task.task_state,
            tuple(wrapper_states),
            self.environment_id,
            self.provenance_id,
        )
        candidate = RobotEnvironmentState(
            plant.candidate_state,
            *runtime_state,
        )
        commit = RobotEnvironmentState(
            plant.accepted_state,
            *runtime_state,
        )
        return _RobotSubstep(
            candidate,
            commit,
            task,
            wrapper_truncated,
            outputs_finite,
            plant.successful,
            plant.status,
        )

    def step(
        self,
        state: RobotEnvironmentState,
        action: ArrayLike,
        args: Any = None,
        /,
    ) -> RobotEnvironmentTransition:
        """Propose one fixed-work repeated action and atomically commit or roll back."""
        if not isinstance(state, RobotEnvironmentState):
            raise TypeError("state must be a RobotEnvironmentState.")
        if state.environment_id != self.environment_id:
            raise ValueError("RobotEnvironmentState belongs to a different environment.")
        if state.provenance_id != self.provenance_id:
            raise ValueError(
                "RobotEnvironmentState provenance belongs to a different environment."
            )
        if len(state.wrapper_states) != len(self.wrappers):
            raise ValueError("RobotEnvironmentState wrapper state count is incompatible.")
        if state.plant_state.shape != self.system.state_layout.shape:
            raise ValueError("RobotEnvironmentState plant_state shape is incompatible.")
        if state.key.shape != (2,) or state.key.dtype != jnp.dtype(jnp.uint32):
            raise ValueError(
                "RobotEnvironmentState key must use uint32 key data with shape (2,)."
            )
        if state.clock.shape != () or state.step_index.shape != ():
            raise ValueError("RobotEnvironmentState clock and step_index must be scalar.")

        input_layout = self.system.input_layout
        assert input_layout is not None

        action_array = jnp.asarray(action)
        if not jnp.issubdtype(action_array.dtype, jnp.inexact):
            action_array = action_array.astype(float)
        if action_array.shape != input_layout.shape:
            raise ValueError(
                f"action must have shape {input_layout.shape}; got {action_array.shape}."
            )

        source_state_finite = _tree_all_finite(
            (
                state.plant_state,
                state.key,
                state.clock,
                state.step_index,
                state.task_state,
                state.wrapper_states,
            ),
            "RobotEnvironmentState",
        )
        action_finite = _tree_all_finite(action_array, "action")
        safe_state = _finite_or_zero_tree(state)
        safe_action = jnp.where(
            action_finite,
            action_array,
            jnp.zeros_like(action_array),
        )
        raw_source_evaluation = self.task.evaluate(
            safe_state.plant_state,
            safe_state.task_state,
        )
        source_evaluation = self._check_task_evaluation(raw_source_evaluation)
        source_outputs_finite = _tree_all_finite(
            raw_source_evaluation,
            "task evaluation outputs",
        )
        initial_valid = (
            source_state_finite & action_finite & source_outputs_finite
        )

        working = safe_state
        last_candidate = state
        candidate_observation = source_evaluation.observation
        candidate_descriptor = source_evaluation.descriptor
        accumulated_rewards = jnp.zeros(
            (len(self.task.reward_component_names),),
            dtype=jnp.result_type(state.plant_state, action_array),
        )
        active = initial_valid
        all_mechanics_successful = jnp.asarray(True)
        all_outputs_finite = jnp.asarray(True)
        terminated = jnp.asarray(False)
        truncated = jnp.asarray(False)
        last_status = jnp.asarray(0, dtype=jnp.int32)
        attempted_values: list[Array] = []
        successful_values: list[Array] = []
        status_values: list[Array] = []

        for _ in range(self.action_repeat):
            attempted = active
            proposal = self._substep(working, safe_action, args)
            mechanics_successful = jnp.asarray(
                proposal.mechanics_successful, dtype=bool
            )
            mechanics_status = jnp.asarray(proposal.mechanics_status, dtype=jnp.int32)
            if mechanics_successful.shape != () or mechanics_status.shape != ():
                raise ValueError("Mechanics successful and status outputs must be scalar.")

            last_candidate = _select_tree(
                attempted,
                proposal.candidate_state,
                last_candidate,
            )
            commit_substep = (
                attempted & mechanics_successful & proposal.outputs_finite
            )
            working = _select_tree(
                commit_substep,
                proposal.commit_state,
                working,
            )
            candidate_observation = jnp.where(
                attempted,
                proposal.task.observation,
                candidate_observation,
            )
            candidate_descriptor = jnp.where(
                attempted,
                proposal.task.descriptor,
                candidate_descriptor,
            )
            accumulated_rewards = accumulated_rewards + jnp.where(
                commit_substep,
                proposal.task.reward_components,
                jnp.zeros_like(proposal.task.reward_components),
            )
            terminated_now = commit_substep & proposal.task.terminated
            truncated_now = (
                commit_substep
                & ~proposal.task.terminated
                & proposal.wrapper_truncated
            )
            terminated = terminated | terminated_now
            truncated = truncated | truncated_now
            all_mechanics_successful = all_mechanics_successful & (
                ~attempted | mechanics_successful
            )
            all_outputs_finite = all_outputs_finite & (
                ~attempted | proposal.outputs_finite
            )
            last_status = jnp.where(attempted, mechanics_status, last_status)
            attempted_values.append(attempted)
            successful_values.append(attempted & mechanics_successful)
            status_values.append(
                jnp.where(attempted, mechanics_status, jnp.asarray(0, jnp.int32))
            )
            active = (
                active
                & mechanics_successful
                & proposal.outputs_finite
                & ~terminated_now
                & ~truncated_now
            )

        candidate_total_reward = jnp.sum(accumulated_rewards)
        aggregate_outputs_finite = _tree_all_finite(
            (
                working.plant_state,
                working.key,
                working.clock,
                working.step_index,
                working.task_state,
                working.wrapper_states,
                candidate_observation,
                candidate_total_reward,
                accumulated_rewards,
                candidate_descriptor,
            ),
            "robot environment transition outputs",
        )
        accepted = (
            initial_valid
            & all_mechanics_successful
            & all_outputs_finite
            & aggregate_outputs_finite
        )
        accepted_state = _select_tree(accepted, working, state)
        final_observation = jnp.where(
            accepted,
            candidate_observation,
            source_evaluation.observation,
        )
        descriptor = jnp.where(
            accepted,
            candidate_descriptor,
            source_evaluation.descriptor,
        )
        reward_components = jnp.where(
            accepted,
            accumulated_rewards,
            jnp.zeros_like(accumulated_rewards),
        )
        terminated = accepted & terminated
        truncated = accepted & truncated
        total_reward = jnp.sum(reward_components)

        reset_performed = accepted & (terminated | truncated) & self.auto_reset
        if self.auto_reset:
            no_reset = RobotEnvironmentReset(
                accepted_state,
                final_observation,
                terminated,
                descriptor,
                self.environment_id,
                self.provenance_id,
            )
            fresh = jax.lax.cond(
                reset_performed,
                lambda key: self._reset(key),
                lambda key: no_reset,
                accepted_state.key,
            )
            reset_state = fresh.state
            observation = fresh.observation
        else:
            reset_state = accepted_state
            observation = final_observation

        attempted_array = jnp.stack(tuple(attempted_values))
        successful_array = jnp.stack(tuple(successful_values))
        status_array = jnp.stack(tuple(status_values))
        evidence = RobotEnvironmentEvidence(
            attempted_array,
            successful_array,
            status_array,
            accepted,
            ~accepted,
            all_mechanics_successful,
            last_status,
            state.step_index,
            last_candidate.step_index,
            self.environment_id,
            self.system.system_id,
            self.task.task_id,
            tuple(wrapper.wrapper_id for wrapper in self.wrappers),
            self.provenance_id,
        )
        return RobotEnvironmentTransition(
            last_candidate,
            accepted_state,
            reset_state,
            candidate_observation,
            observation,
            final_observation,
            total_reward,
            reward_components,
            terminated,
            truncated,
            descriptor,
            reset_performed,
            evidence,
            self.task.reward_component_names,
            self.environment_id,
        )


__all__ = [
    "AbstractRobotEnvironmentWrapper",
    "AbstractRobotTask",
    "PreparedRobotEnvironment",
    "RobotEnvironmentEvidence",
    "RobotEnvironmentReset",
    "RobotEnvironmentState",
    "RobotEnvironmentTransition",
    "RobotEnvironmentWrapperTransition",
    "RobotTaskEvaluation",
    "RobotTaskTransition",
]
