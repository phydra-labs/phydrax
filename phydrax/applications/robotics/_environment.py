#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from math import prod
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_tree import ArrayPyTreeSchema
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...dynamics._plant import (
    AbstractDiscretePlant,
    ArrayDiscreteSystemPlant,
    PlantParameters,
    PlantRuntimeState,
    PlantStepContext,
)
from ...dynamics._system import DiscreteSystem


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _shape(value: tuple[int, ...], owner: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _case_shape(value: Sequence[int], ndim: int, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if len(shape) != ndim:
        raise ValueError(f"case_shape must contain exactly {ndim} dimensions.")
    if any(size <= 0 for size in shape):
        raise ValueError("case_shape dimensions must be positive.")
    return shape


def _key_data(key: ArrayLike, case_shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(key)
    if array.dtype == jnp.dtype(jnp.uint32) and array.shape == case_shape + (2,):
        return array
    data = jnp.asarray(jax.random.key_data(key), dtype=jnp.uint32)
    if data.shape != case_shape + (2,):
        raise ValueError(
            f"Robot environment PRNG key data must have shape {case_shape + (2,)}."
        )
    return data


def _split_key_data(key: Array, count: int, /) -> tuple[Array, ...]:
    if count <= 0:
        raise ValueError("PRNG split count must be positive.")
    case_shape = key.shape[:-1]
    typed = jax.random.wrap_key_data(key)
    flat = jnp.reshape(typed, (-1,))
    split = jax.vmap(lambda item: jax.random.split(item, count))(flat)
    data = jnp.reshape(
        jax.random.key_data(split),
        case_shape + (count, 2),
    )
    return tuple(data[..., index, :] for index in range(count))


def _select_keys(
    selector: Array,
    candidate: Array,
    source: Array,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    candidate_data = jax.random.key_data(candidate)
    source_data = jax.random.key_data(source)
    selected = jnp.where(selector[..., None], candidate_data, source_data)
    return jax.random.wrap_key_data(selected) if source.shape == case_shape else selected


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


def _tree_case_all_finite(
    tree: Any,
    case_shape: tuple[int, ...],
    owner: str,
    /,
) -> Array:
    finite = jnp.ones(case_shape, dtype=bool)
    case_ndim = len(case_shape)
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if not (
            jnp.issubdtype(array.dtype, jnp.number)
            or jnp.issubdtype(array.dtype, jnp.bool_)
        ):
            raise TypeError(f"{owner} must be a PyTree of numeric array leaves.")
        if array.shape[:case_ndim] != case_shape:
            raise ValueError(f"{owner} leaves must begin with the plant case shape.")
        axes = tuple(range(case_ndim, array.ndim))
        finite = finite & jnp.all(jnp.isfinite(array), axis=axes)
    return finite


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
        lambda value: eqx.error_if(value, jnp.any(predicate), message),
        tree,
    )


def _select_tree(
    predicate: Array,
    candidate: Any,
    source: Any,
    case_shape: tuple[int, ...],
    /,
) -> Any:
    if jax.tree.structure(candidate) != jax.tree.structure(source):
        raise ValueError("Candidate and source environment PyTrees must match.")
    case_ndim = len(case_shape)

    def select(proposed: Any, previous: Any) -> Array:
        proposed_array = jnp.asarray(proposed)
        previous_array = jnp.asarray(previous)
        if proposed_array.shape != previous_array.shape:
            raise ValueError("Candidate and source environment leaf shapes must match.")
        if proposed_array.shape[:case_ndim] != case_shape:
            raise ValueError("Environment state leaves must begin with the case shape.")
        expanded = jnp.reshape(
            predicate,
            case_shape + (1,) * (proposed_array.ndim - case_ndim),
        )
        return jnp.where(expanded, proposed_array, previous_array)

    return jax.tree.map(select, candidate, source)


def _select_plant_state(
    plant: AbstractDiscretePlant,
    predicate: Array,
    candidate: PlantRuntimeState,
    source: PlantRuntimeState,
    case_shape: tuple[int, ...],
    /,
) -> PlantRuntimeState:
    return PlantRuntimeState(
        plant.state_schema.select_cases(
            predicate,
            candidate.payload,
            source.payload,
        ),
        jnp.where(predicate, candidate.time, source.time),
        jnp.where(predicate, candidate.step_index, source.step_index),
        _select_keys(predicate, candidate.key, source.key, case_shape),
        source.semantic_provenance_id,
        source.numeric_revision_id,
        source.state_schema_id,
        source.execution_signature_id,
    )


def _plant_ids(plant: AbstractDiscretePlant, /) -> tuple[str, str, str, str]:
    return (
        plant.semantic_provenance.semantic_id,
        plant.numeric_revision.revision_id,
        plant.state_schema.schema_id,
        plant.execution_signature.signature_id,
    )


class RobotTaskEvaluation(StrictModule):
    """Task-owned outputs at one accepted complete plant state."""

    observation: Array
    terminated: Array
    descriptor: Array


class RobotTaskTransition(StrictModule):
    """Task-owned candidate state and outputs for one accepted plant transition."""

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
    def initialize(self, plant_state: PlantRuntimeState, key: Array, /) -> Any:
        """Return fixed-structure task state for a freshly accepted plant state."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        plant_state: PlantRuntimeState,
        task_state: Any,
        /,
    ) -> RobotTaskEvaluation:
        """Observe and classify one accepted complete plant state."""
        raise NotImplementedError

    @abstractmethod
    def transition(
        self,
        context: PlantStepContext,
        source_plant_state: PlantRuntimeState,
        accepted_plant_state: PlantRuntimeState,
        action: Any,
        task_state: Any,
        key: Array,
        /,
    ) -> RobotTaskTransition:
        """Evaluate one mechanics-accepted complete plant transition."""
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
        plant_state: PlantRuntimeState,
        task_state: Any,
        key: Array,
        /,
    ) -> Any:
        """Return this wrapper's fixed-structure state at environment reset."""
        raise NotImplementedError

    @abstractmethod
    def transition(
        self,
        context: PlantStepContext,
        wrapper_state: Any,
        plant_state: PlantRuntimeState,
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

    plant_state: PlantRuntimeState
    key: Array
    episode_step_index: Array
    task_state: Any
    wrapper_states: tuple[Any, ...]
    environment_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class RobotEnvironmentReset(StrictModule):
    """Fresh environment state and task-owned outputs."""

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
    repeat_backend_status: Array
    plant_evidence: tuple[Any, ...]
    accepted: Array
    rollback_applied: Array
    mechanics_successful: Array
    mechanics_status: Array
    source_episode_step_index: Array
    candidate_episode_step_index: Array
    environment_id: str = eqx.field(static=True)
    plant_semantic_provenance_id: str = eqx.field(static=True)
    plant_numeric_revision_id: str = eqx.field(static=True)
    plant_state_schema_id: str = eqx.field(static=True)
    plant_execution_signature_id: str = eqx.field(static=True)
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
    plant_attempted: Array
    mechanics_successful: Array
    mechanics_status: Array
    backend_status: Array
    plant_evidence: Any


class PreparedRobotEnvironment(StrictModule, NonTrainableState):
    """Prepared composition of one complete-state plant, task, and wrappers."""

    plant: AbstractDiscretePlant
    parameters: PlantParameters
    task: AbstractRobotTask
    wrappers: tuple[AbstractRobotEnvironmentWrapper, ...]
    step_size: float = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        plant: AbstractDiscretePlant,
        parameters: PlantParameters,
        task: AbstractRobotTask,
        wrappers: tuple[AbstractRobotEnvironmentWrapper, ...] = (),
        /,
        *,
        step_size: float,
        environment_id: str | None = None,
    ):
        if not isinstance(plant, AbstractDiscretePlant):
            raise TypeError("plant must be an AbstractDiscretePlant.")
        if plant.control_schema is None:
            raise ValueError("Robot environments require a plant control_schema.")
        if not isinstance(parameters, PlantParameters):
            raise TypeError("parameters must be PlantParameters.")
        if parameters.schema_id != plant.parameter_schema.schema_id:
            raise ValueError("PlantParameters schema_id does not match the plant.")
        if (
            parameters.numeric_revision.semantic_id
            != plant.semantic_provenance.semantic_id
        ):
            raise ValueError(
                "PlantParameters semantic provenance does not match the plant."
            )
        if parameters.numeric_revision.revision_id != plant.numeric_revision.revision_id:
            raise ValueError("PlantParameters numeric revision does not match the plant.")
        plant.parameter_schema.validate(parameters.values)
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

        _identifier(task.task_id, "task_id")
        _shape(task.observation_shape, "observation_shape")
        _shape(task.descriptor_shape, "descriptor_shape")
        reward_names = tuple(str(name) for name in task.reward_component_names)
        if not reward_names or any(not name for name in reward_names):
            raise ValueError("reward_component_names must contain non-empty names.")
        if len(set(reward_names)) != len(reward_names):
            raise ValueError("reward_component_names must be unique.")

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

        resolved_step = float(step_size)
        if not np.isfinite(resolved_step) or resolved_step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        resolved_repeat = prod(repeats) if repeats else 1
        resolved_auto_reset = any(wrapper.auto_reset for wrapper in wrapper_tuple)
        plant_ids = _plant_ids(plant)
        provenance_id = canonical_fingerprint(
            {
                "plant": {
                    "semantic_provenance_id": plant_ids[0],
                    "numeric_revision_id": plant_ids[1],
                    "state_schema_id": plant_ids[2],
                    "execution_signature_id": plant_ids[3],
                },
                "parameter_schema_id": parameters.schema_id,
                "task": _provenance_value(task, "task"),
                "wrappers": _provenance_value(wrapper_tuple, "wrappers"),
                "step_size": resolved_step,
                "action_repeat": resolved_repeat,
                "horizons": [wrapper.horizon for wrapper in wrapper_tuple],
                "auto_reset": resolved_auto_reset,
                "prng_representation": {
                    "format": "legacy-key-data",
                    "dtype": "uint32",
                    "implementation": str(
                        jax.config.jax_default_prng_impl  # ty: ignore[unresolved-attribute]
                    ),
                },
            }
        )
        generated_id = f"robot-environment:{provenance_id}"

        self.plant = plant
        self.parameters = parameters
        self.task = task
        self.wrappers = wrapper_tuple
        self.step_size = resolved_step
        self.action_repeat = resolved_repeat
        self.auto_reset = resolved_auto_reset
        self.environment_id = (
            generated_id
            if environment_id is None
            else _identifier(environment_id, "environment_id")
        )
        self.provenance_id = provenance_id

    def _check_task_evaluation(
        self,
        evaluation: RobotTaskEvaluation,
        case_shape: tuple[int, ...],
        /,
    ) -> RobotTaskEvaluation:
        if not isinstance(evaluation, RobotTaskEvaluation):
            raise TypeError("task.evaluate must return RobotTaskEvaluation.")
        observation = jnp.asarray(evaluation.observation)
        terminated = jnp.asarray(evaluation.terminated, dtype=bool)
        descriptor = jnp.asarray(evaluation.descriptor)
        if observation.shape != case_shape + self.task.observation_shape:
            raise ValueError(
                "Task observation shape does not match the case and observation shapes."
            )
        if terminated.shape != case_shape:
            raise ValueError("Task terminated must have the plant case shape.")
        if descriptor.shape != case_shape + self.task.descriptor_shape:
            raise ValueError(
                "Task descriptor shape does not match the case and descriptor shapes."
            )
        return RobotTaskEvaluation(observation, terminated, descriptor)

    def _check_task_transition(
        self,
        transition: RobotTaskTransition,
        case_shape: tuple[int, ...],
        /,
    ) -> RobotTaskTransition:
        if not isinstance(transition, RobotTaskTransition):
            raise TypeError("task.transition must return RobotTaskTransition.")
        observation = jnp.asarray(transition.observation)
        rewards = jnp.asarray(transition.reward_components)
        terminated = jnp.asarray(transition.terminated, dtype=bool)
        descriptor = jnp.asarray(transition.descriptor)
        if observation.shape != case_shape + self.task.observation_shape:
            raise ValueError(
                "Task observation shape does not match the case and observation shapes."
            )
        if rewards.shape != case_shape + (len(self.task.reward_component_names),):
            raise ValueError(
                "Task reward_components must have one scalar per declared name and case."
            )
        if terminated.shape != case_shape:
            raise ValueError("Task terminated must have the plant case shape.")
        if descriptor.shape != case_shape + self.task.descriptor_shape:
            raise ValueError(
                "Task descriptor shape does not match the case and descriptor shapes."
            )
        return RobotTaskTransition(
            transition.task_state,
            observation,
            rewards,
            terminated,
            descriptor,
        )

    def _validate_plant_state(self, state: PlantRuntimeState, /) -> tuple[int, ...]:
        if not isinstance(state, PlantRuntimeState):
            raise TypeError(
                "RobotEnvironmentState plant_state must be PlantRuntimeState."
            )
        observed = (
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
        )
        for name, observed_id, expected_id in zip(
            (
                "semantic provenance",
                "numeric revision",
                "state schema",
                "execution signature",
            ),
            observed,
            _plant_ids(self.plant),
            strict=True,
        ):
            if observed_id != expected_id:
                raise ValueError(
                    f"RobotEnvironmentState plant {name} does not match this plant."
                )
        case_shape = self.plant.state_schema.validate(state.payload)
        if state.time.shape != case_shape or state.step_index.shape != case_shape:
            raise ValueError("Plant time and step_index must have the plant case shape.")
        _key_data(state.key, case_shape)
        return case_shape

    def _safe_plant_state(
        self,
        state: PlantRuntimeState,
        valid: Array,
        case_shape: tuple[int, ...],
        /,
    ) -> PlantRuntimeState:
        payload = self.plant.state_schema.select_cases(
            valid,
            state.payload,
            self.plant.state_schema.zeros(case_shape),
        )
        return PlantRuntimeState(
            payload,
            jnp.where(valid, state.time, jnp.zeros_like(state.time)),
            jnp.where(valid, state.step_index, jnp.zeros_like(state.step_index)),
            state.key,
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
        )

    def _select_environment_state(
        self,
        predicate: Array,
        candidate: RobotEnvironmentState,
        source: RobotEnvironmentState,
        case_shape: tuple[int, ...],
        /,
    ) -> RobotEnvironmentState:
        return RobotEnvironmentState(
            _select_plant_state(
                self.plant,
                predicate,
                candidate.plant_state,
                source.plant_state,
                case_shape,
            ),
            _select_keys(predicate, candidate.key, source.key, case_shape),
            jnp.where(
                predicate,
                candidate.episode_step_index,
                source.episode_step_index,
            ),
            _select_tree(
                predicate,
                candidate.task_state,
                source.task_state,
                case_shape,
            ),
            _select_tree(
                predicate,
                candidate.wrapper_states,
                source.wrapper_states,
                case_shape,
            ),
            source.environment_id,
            source.provenance_id,
        )

    def _reset(
        self,
        plant_key: Array,
        environment_key: Array,
        case_shape: tuple[int, ...],
        /,
    ) -> RobotEnvironmentReset:
        plant_reset = self.plant.reset(
            plant_key,
            self.parameters,
            case_shape=case_shape,
            initial_time=jnp.zeros(case_shape),
        )
        plant_state = plant_reset.accepted_state
        keys = _split_key_data(
            _key_data(environment_key, case_shape),
            len(self.wrappers) + 2,
        )
        task_state = self.task.initialize(plant_state, keys[1])
        raw_evaluation = self.task.evaluate(plant_state, task_state)
        evaluation = self._check_task_evaluation(raw_evaluation, case_shape)
        wrapper_states = tuple(
            wrapper.initialize(plant_state, task_state, wrapper_key)
            for wrapper, wrapper_key in zip(self.wrappers, keys[2:], strict=True)
        )
        reset_outputs_finite = _tree_case_all_finite(
            (task_state, wrapper_states, raw_evaluation),
            case_shape,
            "robot environment reset outputs",
        )
        message = "Robot environment reset outputs must be finite."
        task_state = _error_if_tree(task_state, ~reset_outputs_finite, message)
        wrapper_states = _error_if_tree(
            wrapper_states,
            ~reset_outputs_finite,
            message,
        )
        evaluation = _error_if_tree(evaluation, ~reset_outputs_finite, message)
        state = RobotEnvironmentState(
            plant_state,
            keys[0],
            jnp.zeros(case_shape, dtype=jnp.int32),
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

    def reset(
        self,
        key: ArrayLike,
        /,
        *,
        case_shape: Sequence[int] = (),
    ) -> RobotEnvironmentReset:
        """Initialize separate plant and episode transactions from one root key."""
        resolved_case_shape = _case_shape(
            case_shape,
            self.plant.state_schema.case_ndim,
        )
        plant_key, environment_key = _split_key_data(
            _key_data(key, resolved_case_shape),
            2,
        )
        return self._reset(plant_key, environment_key, resolved_case_shape)

    def _substep(
        self,
        state: RobotEnvironmentState,
        plant_action: Any,
        task_action: Any,
        case_shape: tuple[int, ...],
        /,
    ) -> _RobotSubstep:
        target_time = state.plant_state.time + self.step_size
        context = PlantStepContext(
            state.plant_state.time,
            target_time,
            state.plant_state.step_index,
        )
        plant_result = self.plant.step(
            context,
            state.plant_state,
            plant_action,
            self.parameters,
        )
        accepted_plant = plant_result.accepted_state
        keys = _split_key_data(state.key, len(self.wrappers) + 2)
        raw_task = self.task.transition(
            context,
            state.plant_state,
            accepted_plant,
            task_action,
            state.task_state,
            keys[1],
        )
        task = self._check_task_transition(raw_task, case_shape)
        task_outputs_finite = _tree_case_all_finite(
            raw_task,
            case_shape,
            "task transition outputs",
        )
        safe_task = _finite_or_zero_tree(task)
        outputs_finite = task_outputs_finite
        wrapper_states: list[Any] = []
        wrapper_truncated = jnp.zeros(case_shape, dtype=bool)
        for wrapper, wrapper_state, wrapper_key in zip(
            self.wrappers,
            state.wrapper_states,
            keys[2:],
            strict=True,
        ):
            update = wrapper.transition(
                context,
                wrapper_state,
                accepted_plant,
                safe_task.task_state,
                safe_task.observation,
                safe_task.terminated,
                wrapper_key,
            )
            if not isinstance(update, RobotEnvironmentWrapperTransition):
                raise TypeError(
                    "wrapper.transition must return RobotEnvironmentWrapperTransition."
                )
            outputs_finite = outputs_finite & _tree_case_all_finite(
                update,
                case_shape,
                "wrapper transition outputs",
            )
            truncated = jnp.asarray(update.truncated, dtype=bool)
            if truncated.shape != case_shape:
                raise ValueError("Wrapper truncated must have the plant case shape.")
            wrapper_states.append(update.wrapper_state)
            wrapper_truncated = wrapper_truncated | truncated
            if wrapper.horizon is not None:
                wrapper_truncated = wrapper_truncated | (
                    state.episode_step_index + 1 >= int(wrapper.horizon)
                )

        runtime_state = (
            keys[0],
            state.episode_step_index + jnp.asarray(1, dtype=jnp.int32),
            task.task_state,
            tuple(wrapper_states),
            self.environment_id,
            self.provenance_id,
        )
        candidate = RobotEnvironmentState(
            plant_result.candidate_state,
            *runtime_state,
        )
        commit = RobotEnvironmentState(
            accepted_plant,
            *runtime_state,
        )
        return _RobotSubstep(
            candidate,
            commit,
            task,
            wrapper_truncated,
            outputs_finite,
            plant_result.attempted,
            plant_result.attempted & plant_result.successful,
            plant_result.status,
            plant_result.backend_status,
            plant_result.evidence,
        )

    def step(
        self,
        state: RobotEnvironmentState,
        action: Any,
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
        case_shape = self._validate_plant_state(state.plant_state)
        if state.key.shape != case_shape + (2,) or state.key.dtype != jnp.dtype(
            jnp.uint32
        ):
            raise ValueError(
                "RobotEnvironmentState episode key must use uint32 key data with "
                "the plant case shape."
            )
        if state.episode_step_index.shape != case_shape:
            raise ValueError(
                "RobotEnvironmentState episode_step_index must have the plant case shape."
            )

        control_schema = self.plant.control_schema
        assert control_schema is not None
        action_case_shape = control_schema.validate(action)
        if action_case_shape not in ((), case_shape):
            raise ValueError("Action case shape must be shared or match the plant cases.")
        safe_action = _finite_or_zero_tree(action)

        payload_finite = (
            self.plant.state_schema.finite_mask(state.plant_state.payload)
            if self.plant.require_finite_state
            else jnp.ones(case_shape, dtype=bool)
        )
        plant_runtime_finite = (
            payload_finite
            & jnp.isfinite(state.plant_state.time)
            & (state.plant_state.step_index >= 0)
        )
        environment_state_finite = _tree_case_all_finite(
            (
                state.key,
                state.episode_step_index,
                state.task_state,
                state.wrapper_states,
            ),
            case_shape,
            "RobotEnvironmentState",
        ) & (state.episode_step_index >= 0)
        safe_plant = self._safe_plant_state(
            state.plant_state,
            plant_runtime_finite,
            case_shape,
        )
        safe_state = RobotEnvironmentState(
            safe_plant,
            _finite_or_zero_tree(state.key),
            jnp.where(
                state.episode_step_index >= 0,
                state.episode_step_index,
                jnp.zeros_like(state.episode_step_index),
            ),
            _finite_or_zero_tree(state.task_state),
            _finite_or_zero_tree(state.wrapper_states),
            state.environment_id,
            state.provenance_id,
        )
        raw_source_evaluation = self.task.evaluate(
            safe_state.plant_state,
            safe_state.task_state,
        )
        source_evaluation = self._check_task_evaluation(
            raw_source_evaluation,
            case_shape,
        )
        source_outputs_finite = _tree_case_all_finite(
            raw_source_evaluation,
            case_shape,
            "task evaluation outputs",
        )
        initial_valid = (
            plant_runtime_finite & environment_state_finite & source_outputs_finite
        )

        working = safe_state
        last_candidate = state
        candidate_observation = source_evaluation.observation
        candidate_descriptor = source_evaluation.descriptor
        result_dtype = jnp.result_type(
            *jax.tree.leaves(state.plant_state.payload),
            *jax.tree.leaves(action),
        )
        accumulated_rewards = jnp.zeros(
            case_shape + (len(self.task.reward_component_names),),
            dtype=result_dtype,
        )
        active = initial_valid
        all_mechanics_successful = jnp.ones(case_shape, dtype=bool)
        all_outputs_finite = jnp.ones(case_shape, dtype=bool)
        terminated = jnp.zeros(case_shape, dtype=bool)
        truncated = jnp.zeros(case_shape, dtype=bool)
        last_status = jnp.zeros(case_shape, dtype=jnp.int32)
        attempted_values: list[Array] = []
        successful_values: list[Array] = []
        status_values: list[Array] = []
        backend_status_values: list[Array] = []
        plant_evidence_values: list[Any] = []

        for _ in range(self.action_repeat):
            proposal = self._substep(working, action, safe_action, case_shape)
            attempted = active & proposal.plant_attempted
            mechanics_successful = jnp.asarray(
                proposal.mechanics_successful,
                dtype=bool,
            )
            mechanics_status = jnp.asarray(
                proposal.mechanics_status,
                dtype=jnp.int32,
            )
            backend_status = jnp.asarray(
                proposal.backend_status,
                dtype=jnp.int32,
            )
            for name, value in (
                ("mechanics successful", mechanics_successful),
                ("mechanics status", mechanics_status),
                ("backend status", backend_status),
            ):
                if value.shape != case_shape:
                    raise ValueError(f"Plant {name} must have the plant case shape.")

            last_candidate = self._select_environment_state(
                active,
                proposal.candidate_state,
                last_candidate,
                case_shape,
            )
            commit_substep = attempted & mechanics_successful & proposal.outputs_finite
            working = self._select_environment_state(
                commit_substep,
                proposal.commit_state,
                working,
                case_shape,
            )
            observation_mask = jnp.reshape(
                active,
                case_shape + (1,) * len(self.task.observation_shape),
            )
            descriptor_mask = jnp.reshape(
                active,
                case_shape + (1,) * len(self.task.descriptor_shape),
            )
            candidate_observation = jnp.where(
                observation_mask,
                proposal.task.observation,
                candidate_observation,
            )
            candidate_descriptor = jnp.where(
                descriptor_mask,
                proposal.task.descriptor,
                candidate_descriptor,
            )
            accumulated_rewards = accumulated_rewards + jnp.where(
                commit_substep[..., None],
                proposal.task.reward_components,
                jnp.zeros_like(proposal.task.reward_components),
            )
            terminated_now = commit_substep & proposal.task.terminated
            truncated_now = (
                commit_substep & ~proposal.task.terminated & proposal.wrapper_truncated
            )
            terminated = terminated | terminated_now
            truncated = truncated | truncated_now
            all_mechanics_successful = all_mechanics_successful & (
                ~active | (proposal.plant_attempted & mechanics_successful)
            )
            all_outputs_finite = all_outputs_finite & (~active | proposal.outputs_finite)
            last_status = jnp.where(active, mechanics_status, last_status)
            attempted_values.append(attempted)
            successful_values.append(attempted & mechanics_successful)
            status_values.append(
                jnp.where(active, mechanics_status, jnp.asarray(0, jnp.int32))
            )
            backend_status_values.append(
                jnp.where(active, backend_status, jnp.asarray(0, jnp.int32))
            )
            plant_evidence_values.append(proposal.plant_evidence)
            active = (
                active
                & proposal.plant_attempted
                & mechanics_successful
                & proposal.outputs_finite
                & ~terminated_now
                & ~truncated_now
            )

        candidate_total_reward = jnp.sum(accumulated_rewards, axis=-1)
        aggregate_outputs_finite = _tree_case_all_finite(
            (
                working.plant_state.time,
                working.plant_state.step_index,
                working.key,
                working.episode_step_index,
                working.task_state,
                working.wrapper_states,
                candidate_observation,
                candidate_total_reward,
                accumulated_rewards,
                candidate_descriptor,
            ),
            case_shape,
            "robot environment transition outputs",
        )
        accepted = (
            initial_valid
            & all_mechanics_successful
            & all_outputs_finite
            & aggregate_outputs_finite
        )
        accepted_state = self._select_environment_state(
            accepted,
            working,
            state,
            case_shape,
        )
        observation_mask = jnp.reshape(
            accepted,
            case_shape + (1,) * len(self.task.observation_shape),
        )
        descriptor_mask = jnp.reshape(
            accepted,
            case_shape + (1,) * len(self.task.descriptor_shape),
        )
        final_observation = jnp.where(
            observation_mask,
            candidate_observation,
            source_evaluation.observation,
        )
        descriptor = jnp.where(
            descriptor_mask,
            candidate_descriptor,
            source_evaluation.descriptor,
        )
        reward_components = jnp.where(
            accepted[..., None],
            accumulated_rewards,
            jnp.zeros_like(accumulated_rewards),
        )
        terminated = accepted & terminated
        truncated = accepted & truncated
        total_reward = jnp.sum(reward_components, axis=-1)

        reset_performed = accepted & (terminated | truncated) & self.auto_reset
        if self.auto_reset:
            fresh = self._reset(
                accepted_state.plant_state.key,
                accepted_state.key,
                case_shape,
            )
            reset_state = self._select_environment_state(
                reset_performed,
                fresh.state,
                accepted_state,
                case_shape,
            )
            reset_observation_mask = jnp.reshape(
                reset_performed,
                case_shape + (1,) * len(self.task.observation_shape),
            )
            observation = jnp.where(
                reset_observation_mask,
                fresh.observation,
                final_observation,
            )
        else:
            reset_state = accepted_state
            observation = final_observation

        plant_ids = _plant_ids(self.plant)
        evidence = RobotEnvironmentEvidence(
            jnp.stack(tuple(attempted_values)),
            jnp.stack(tuple(successful_values)),
            jnp.stack(tuple(status_values)),
            jnp.stack(tuple(backend_status_values)),
            tuple(plant_evidence_values),
            accepted,
            ~accepted,
            all_mechanics_successful,
            last_status,
            state.episode_step_index,
            last_candidate.episode_step_index,
            self.environment_id,
            *plant_ids,
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


def prepare_array_robot_environment(
    system: DiscreteSystem,
    initializer: Callable[[Array], ArrayLike],
    task: AbstractRobotTask,
    wrappers: tuple[AbstractRobotEnvironmentWrapper, ...] = (),
    /,
    *,
    initializer_id: str,
    reset_fallback: ArrayLike,
    parameter_values: Any | None = None,
    parameter_schema: ArrayPyTreeSchema | None = None,
    semantic_provenance: SemanticProvenance | None = None,
    numeric_revision: NumericRevision | None = None,
    execution_signature: ExecutableSignature | None = None,
    case_ndim: int = 0,
    control_dtype: Any | None = None,
    step_size: float | None = None,
    environment_id: str | None = None,
) -> PreparedRobotEnvironment:
    """Adapt one legacy array ``DiscreteSystem`` to the complete plant lifecycle."""
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
    initialization_id = _identifier(initializer_id, "initializer_id")
    fallback = jnp.asarray(reset_fallback)
    if fallback.shape != system.state_layout.shape:
        raise ValueError("reset_fallback must match the DiscreteSystem state shape.")

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

    semantics = (
        SemanticProvenance(
            {
                "kind": "array-discrete-system-robot-plant",
                "system_id": system.system_id,
                "transition": _callable_provenance(
                    system.transition,
                    "system transition",
                ),
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
                    "callable": _callable_provenance(initializer, "initializer"),
                },
                "step_size": system.step_size,
                "step_rtol": system.step_rtol,
                "step_atol": system.step_atol,
                "minimum_step_size": system.minimum_step_size,
                "maximum_step_size": system.maximum_step_size,
            }
        )
        if semantic_provenance is None
        else semantic_provenance
    )
    if not isinstance(semantics, SemanticProvenance):
        raise TypeError("semantic_provenance must be SemanticProvenance or None.")

    values = (
        () if parameter_values is None and parameter_schema is None else parameter_values
    )
    resolved_parameter_schema = parameter_schema
    if resolved_parameter_schema is None and parameter_values is not None:
        resolved_parameter_schema = ArrayPyTreeSchema.from_tree(values, case_ndim=0)
    revision = (
        NumericRevision(
            semantics,
            {
                "parameter_values": values,
                "reset_fallback": fallback,
            },
        )
        if numeric_revision is None
        else numeric_revision
    )
    if not isinstance(revision, NumericRevision):
        raise TypeError("numeric_revision must be NumericRevision or None.")
    executable = (
        ExecutableSignature(
            shapes={
                "state": system.state_layout.shape,
                "control": system.input_layout.shape,
            },
            dtypes={
                "state": fallback.dtype,
                "control": fallback.dtype if control_dtype is None else control_dtype,
            },
            algorithm_facts={
                "case_ndim": int(case_ndim),
                "step_size": resolved_step,
                "system_id": system.system_id,
            },
        )
        if execution_signature is None
        else execution_signature
    )
    if not isinstance(executable, ExecutableSignature):
        raise TypeError("execution_signature must be ExecutableSignature or None.")

    plant = ArrayDiscreteSystemPlant(
        system,
        initializer,
        reset_fallback=fallback,
        semantic_provenance=semantics,
        numeric_revision=revision,
        execution_signature=executable,
        parameter_schema=resolved_parameter_schema,
        case_ndim=case_ndim,
        control_dtype=control_dtype,
    )
    parameters = PlantParameters(
        values,
        plant.parameter_schema.schema_id,
        revision,
    )
    return PreparedRobotEnvironment(
        plant,
        parameters,
        task,
        wrappers,
        step_size=resolved_step,
        environment_id=environment_id,
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
    "prepare_array_robot_environment",
]
