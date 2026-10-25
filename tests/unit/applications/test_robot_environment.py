import functools
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_tree import ArrayPyTreeSchema
from phydrax._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from phydrax._strict import StrictModule
from phydrax.applications.contact._articulated_step import ArticulatedActuationMode
from phydrax.applications.robotics._articulated_plant import (
    ArticulatedPlantCommand,
    NativeArticulatedPlantPlan,
    prepare_articulated_plant,
    PreparedArticulatedPlant,
)
from phydrax.applications.robotics._environment import (
    AbstractRobotEnvironmentWrapper,
    AbstractRobotTask,
    prepare_array_robot_environment,
    PreparedRobotEnvironment,
    RobotEnvironmentWrapperTransition,
    RobotTaskEvaluation,
    RobotTaskTransition,
)
from phydrax.dynamics import DiscreteStepContext, DiscreteSystem, InputLayout, StateLayout
from phydrax.dynamics._plant import (
    AbstractDiscretePlant,
    ArrayDiscreteSystemPlant,
    PlantParameters,
    PlantProposal,
    PlantRuntimeState,
    PlantStepContext,
)
from phydrax.dynamics._plant_codec import ControlVectorCodec
from phydrax.dynamics._system import DiscreteTransitionResult
from tests._support.assertions import assert_tree_equal
from tests._support.native_articulated import (
    LBVH_TRAVERSAL_VISITS,
    route_budget,
    slider_model,
    standard_gravity,
    step_policy,
)


class _ThresholdTask(AbstractRobotTask):
    threshold: jax.Array
    task_id: str = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    reward_component_names: tuple[str, ...] = eqx.field(static=True)
    descriptor_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, threshold: Any) -> None:
        self.threshold = jnp.asarray(threshold)
        self.task_id = "threshold-task"
        self.observation_shape = (2,)
        self.reward_component_names = ("progress", "effort")
        self.descriptor_shape = (1,)

    def initialize(self, plant_state: Any, key: Any, /) -> Any:
        del plant_state
        return jnp.asarray([0.0, key[0].astype(jnp.float32)])

    def evaluate(self, plant_state: Any, task_state: Any, /) -> Any:
        payload = plant_state.payload
        return RobotTaskEvaluation(
            jnp.asarray([payload[0], task_state[0]]),
            payload[0] >= self.threshold,
            payload[:1],
        )

    def transition(
        self,
        context: Any,
        source_plant_state: Any,
        accepted_plant_state: Any,
        action: Any,
        task_state: Any,
        key: Any,
        /,
    ) -> Any:
        del context
        source = source_plant_state.payload
        accepted = accepted_plant_state.payload
        candidate_task_state = task_state.at[0].add(1.0)
        candidate_task_state = candidate_task_state.at[1].set(
            key[0].astype(task_state.dtype)
        )
        return RobotTaskTransition(
            candidate_task_state,
            jnp.asarray([accepted[0], candidate_task_state[0]]),
            jnp.asarray(
                [
                    accepted[0] - source[0],
                    -(action[0] * action[0]),
                ]
            ),
            accepted[0] >= self.threshold,
            accepted[:1],
        )


class _EpisodeWrapper(AbstractRobotEnvironmentWrapper):
    wrapper_id: str = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    horizon: int | None = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)

    def __init__(
        self, *, action_repeat: Any, horizon: Any = None, auto_reset: Any = False
    ) -> None:
        self.wrapper_id = "episode-wrapper"
        self.action_repeat = action_repeat
        self.horizon = horizon
        self.auto_reset = auto_reset

    def initialize(self, plant_state: Any, task_state: Any, key: Any, /) -> Any:
        del plant_state, task_state
        return jnp.asarray([0.0, key[0].astype(jnp.float32)])

    def transition(
        self,
        context: Any,
        wrapper_state: Any,
        plant_state: Any,
        task_state: Any,
        observation: Any,
        terminated: Any,
        key: Any,
        /,
    ) -> Any:
        del context, plant_state, task_state, observation, terminated
        candidate = wrapper_state.at[0].add(1.0)
        candidate = candidate.at[1].set(key[0].astype(wrapper_state.dtype))
        return RobotEnvironmentWrapperTransition(candidate, jnp.asarray(False))


class _PlantRecordingWrapper(AbstractRobotEnvironmentWrapper):
    wrapper_id: str = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    horizon: int | None = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)

    def __init__(
        self, *, action_repeat: Any = 1, horizon: Any = None, auto_reset: Any = False
    ) -> None:
        self.wrapper_id = "plant-recording-wrapper"
        self.action_repeat = action_repeat
        self.horizon = horizon
        self.auto_reset = auto_reset

    def initialize(self, plant_state: Any, task_state: Any, key: Any, /) -> Any:
        del plant_state, task_state, key
        return jnp.zeros((1,))

    def transition(
        self,
        context: Any,
        wrapper_state: Any,
        plant_state: Any,
        task_state: Any,
        observation: Any,
        terminated: Any,
        key: Any,
        /,
    ) -> Any:
        del context, wrapper_state, task_state, observation, terminated, key
        return RobotEnvironmentWrapperTransition(
            plant_state.payload[:1],
            jnp.asarray(False),
        )


def _initial_state(key: Any) -> Any:
    del key
    return jnp.zeros((2,))


def _unit_initial_state(key: Any) -> Any:
    del key
    return jnp.ones((2,))


def _bounded_transition(
    context: DiscreteStepContext,
    state: Any,
    action: Any,
    args: Any,
) -> Any:
    del context
    limit = jnp.asarray(1.0 if args is None else args)
    candidate = state.at[0].add(action[0])
    successful = candidate[0] <= limit
    accepted = jnp.where(successful, candidate, state)
    status = jnp.where(successful, 0, 7).astype(jnp.int32)
    return DiscreteTransitionResult(candidate, accepted, successful, status)


def _projected_transition(context: Any, state: Any, action: Any, args: Any) -> Any:
    del context, args
    candidate = state.at[0].add(10.0 * action[0])
    accepted = state.at[0].add(action[0])
    return DiscreteTransitionResult(
        candidate,
        accepted,
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
    )


class _GainTransition(StrictModule):
    gain: jax.Array

    def __init__(self, gain: Any) -> None:
        self.gain = jnp.asarray(gain)

    def __call__(self, context: Any, state: Any, action: Any, args: Any) -> Any:
        del context, args
        accepted = state.at[0].add(self.gain * action[0])
        return DiscreteTransitionResult(
            accepted,
            accepted,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        )


def _halve(value: Any) -> Any:
    return 0.5 * value


def _double(value: Any) -> Any:
    return 2.0 * value


class _ShapedWrapper(AbstractRobotEnvironmentWrapper):
    shaping: Callable
    wrapper_id: str = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    horizon: int | None = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)

    def __init__(self, shaping: Any) -> None:
        self.shaping = shaping
        self.wrapper_id = "shaped-wrapper"
        self.action_repeat = 1
        self.horizon = None
        self.auto_reset = False

    def initialize(self, plant_state: Any, task_state: Any, key: Any, /) -> Any:
        del plant_state, task_state, key
        return jnp.ones((1,))

    def transition(
        self,
        context: Any,
        wrapper_state: Any,
        plant_state: Any,
        task_state: Any,
        observation: Any,
        terminated: Any,
        key: Any,
        /,
    ) -> Any:
        del context, plant_state, task_state, observation, terminated, key
        return RobotEnvironmentWrapperTransition(
            self.shaping(wrapper_state),
            jnp.asarray(False),
        )


def _environment(
    *,
    threshold: Any = 100.0,
    repeat: Any = 1,
    horizon: Any = None,
    auto_reset: Any = False,
    transition: Any = _bounded_transition,
    state_layout: Any = None,
    input_layout: Any = None,
    system_id: Any = "bounded-plant",
    initializer: Any = _initial_state,
    environment_id: Any = None,
    **callable_ids: Any,
) -> Any:
    system = DiscreteSystem(
        transition,
        state_layout=StateLayout((2,)) if state_layout is None else state_layout,
        input_layout=InputLayout((1,)) if input_layout is None else input_layout,
        system_id=system_id,
        step_size=1.0,
    )
    return prepare_array_robot_environment(
        system,
        initializer,
        _ThresholdTask(threshold),
        (
            _EpisodeWrapper(
                action_repeat=repeat,
                horizon=horizon,
                auto_reset=auto_reset,
            ),
        ),
        **callable_ids,
        reset_fallback=jnp.zeros((2,)),
        parameter_values=jnp.asarray(1.0),
        environment_id=environment_id,
    )


def test_robot_environment_scenario_1() -> None:
    environment = _environment(repeat=2)
    reset = environment.reset(jax.random.key(4))
    assert isinstance(environment.plant, ArrayDiscreteSystemPlant)
    assert isinstance(environment.parameters, PlantParameters)
    assert not hasattr(environment, "system")
    assert not hasattr(environment, "initializer")

    result = environment.step(reset.state, jnp.asarray([1.0]))

    assert not result.evidence.accepted
    assert result.evidence.rollback_applied
    assert not result.evidence.mechanics_successful
    assert result.evidence.mechanics_status == 7
    assert jnp.array_equal(result.evidence.attempted, jnp.asarray([True, True]))
    assert jnp.array_equal(
        result.evidence.repeat_successful,
        jnp.asarray([True, False]),
    )
    assert result.candidate_state.plant_state.payload[0] == 2.0
    assert result.evidence.plant_evidence[1].legacy_candidate_state[0] == 2.0
    assert result.candidate_state.plant_state.time == 2.0
    assert result.candidate_state.plant_state.step_index == 2
    assert result.candidate_state.episode_step_index == 2
    _assert_state_equal(result.accepted_state, reset.state)
    assert result.total_reward == 0.0
    assert not result.terminated
    assert not result.truncated
    environment = _environment(transition=_projected_transition)
    environment = PreparedRobotEnvironment(
        environment.plant,
        environment.parameters,
        environment.task,
        (_PlantRecordingWrapper(action_repeat=2),),
        step_size=environment.step_size,
    )
    reset = environment.reset(jax.random.key(10))

    result = environment.step(reset.state, jnp.asarray([1.0]))

    assert result.accepted
    assert result.accepted_state.plant_state.payload[0] == 2.0
    assert result.candidate_state.plant_state.payload[0] == 11.0
    assert result.evidence.plant_evidence[1].legacy_candidate_state[0] == 11.0
    assert result.accepted_state.task_state[0] == 2.0
    assert result.accepted_state.wrapper_states[0][0] == 2.0
    assert result.candidate_observation[0] == 2.0
    assert jnp.array_equal(result.reward_components, jnp.asarray([2.0, -2.0]))
    terminated_environment = _environment(threshold=1.0, repeat=3, horizon=1)
    terminated_reset = terminated_environment.reset(jax.random.key(5))
    terminated = terminated_environment.step(
        terminated_reset.state,
        jnp.asarray([1.0]),
    )

    assert terminated.accepted
    assert terminated.terminated
    assert not terminated.truncated
    assert jnp.array_equal(
        terminated.evidence.attempted,
        jnp.asarray([True, False, False]),
    )
    assert terminated.accepted_state.episode_step_index == 1
    assert terminated.accepted_state.plant_state.step_index == 1

    horizon_environment = _environment(repeat=3, horizon=2)
    horizon_reset = horizon_environment.reset(jax.random.key(6))
    horizon = horizon_environment.step(horizon_reset.state, jnp.asarray([0.25]))

    assert horizon.accepted
    assert not horizon.terminated
    assert horizon.truncated
    assert jnp.array_equal(
        horizon.evidence.attempted,
        jnp.asarray([True, True, False]),
    )
    assert horizon.accepted_state.episode_step_index == 2
    assert horizon.accepted_state.plant_state.time == 2.0
    environment = _environment(threshold=1.0, repeat=3, auto_reset=True)
    reset = environment.reset(jax.random.key(7))

    result = environment.step(reset.state, jnp.asarray([1.0]))

    assert result.reset_performed
    assert result.terminated
    assert result.final_observation[0] == 1.0
    assert result.observation[0] == 0.0
    assert result.accepted_state.plant_state.payload[0] == 1.0
    assert result.accepted_state.plant_state.step_index == 1
    assert result.accepted_state.episode_step_index == 1
    assert result.reset_state.plant_state.payload[0] == 0.0
    assert result.reset_state.plant_state.step_index == 0
    assert result.reset_state.plant_state.time == 0.0
    assert result.reset_state.episode_step_index == 0
    assert not jnp.array_equal(
        result.reset_state.plant_state.key,
        result.accepted_state.plant_state.key,
    )
    assert not jnp.array_equal(result.reset_state.key, result.accepted_state.key)


def test_robot_environment_scenario_2() -> None:
    display_id = "shared-display-id"
    baseline = _environment(threshold=1.0, environment_id=display_id)
    changed_task = _environment(threshold=2.0, environment_id=display_id)
    changed_initializer = _environment(
        threshold=1.0,
        initializer=_unit_initial_state,
        environment_id=display_id,
    )
    reset = baseline.reset(jax.random.key(40))
    result = baseline.step(reset.state, jnp.asarray([0.25]))

    assert changed_task.provenance_id != baseline.provenance_id
    assert changed_initializer.provenance_id != baseline.provenance_id
    assert result.evidence.plant_semantic_provenance_id == (
        baseline.plant.semantic_provenance.semantic_id
    )
    assert result.evidence.plant_numeric_revision_id == (
        baseline.plant.numeric_revision.revision_id
    )
    assert result.evidence.plant_state_schema_id == baseline.plant.state_schema.schema_id
    assert result.evidence.plant_execution_signature_id == (
        baseline.plant.execution_signature.signature_id
    )
    with pytest.raises(ValueError, match="provenance"):
        changed_task.step(reset.state, jnp.asarray([0.25]))
    environment = _mixed_environment(reset_successful=False)
    keys = jax.random.split(jax.random.key(59), 2)
    reset = environment.reset(keys, case_shape=(2,))

    assert jnp.array_equal(reset.attempted, jnp.asarray([True, True]))
    assert jnp.array_equal(reset.successful, jnp.asarray([False, False]))
    assert jnp.array_equal(reset.status, jnp.asarray([7, 7]))
    assert jnp.array_equal(reset.backend_status, jnp.asarray([17, 17]))
    assert reset.evidence == ()
    first = _environment(
        initializer=lambda key: jnp.zeros((2,)),
        initializer_semantic_id="zero-initializer",
        initializer_numeric_id="zero-initializer:rev-1",
    )
    second = _environment(
        initializer=lambda key: jnp.zeros((2,)),
        initializer_semantic_id="other-initializer",
        initializer_numeric_id="other-initializer:rev-1",
    )
    revised = _environment(
        initializer=lambda key: jnp.zeros((2,)),
        initializer_semantic_id="zero-initializer",
        initializer_numeric_id="zero-initializer:rev-2",
    )

    assert first.provenance_id != second.provenance_id
    assert (
        first.plant.semantic_provenance.semantic_id
        != second.plant.semantic_provenance.semantic_id
    )
    assert (
        revised.plant.semantic_provenance.semantic_id
        == first.plant.semantic_provenance.semantic_id
    )
    assert (
        revised.plant.numeric_revision.revision_id
        != first.plant.numeric_revision.revision_id
    )
    slow = _environment(transition=_GainTransition(1.0))
    fast = _environment(transition=_GainTransition(2.0))

    assert (
        slow.plant.semantic_provenance.semantic_id
        == fast.plant.semantic_provenance.semantic_id
    )
    assert slow.plant.numeric_revision.revision_id != (
        fast.plant.numeric_revision.revision_id
    )
    assert slow.provenance_id != fast.provenance_id
    with pytest.raises(ValueError, match="StrictModule"):
        _environment(
            transition=_GainTransition(1.0),
            transition_semantic_id="gain",
            transition_numeric_id="gain:rev-1",
        )


def test_opaque_environment_callables_without_declared_ids_are_refused() -> None:
    def closure_transition(context: Any, state: Any, action: Any, args: Any) -> Any:
        return _bounded_transition(context, state, action, args)

    with pytest.raises(TypeError, match="Opaque callables"):
        _environment(initializer=lambda key: jnp.zeros((2,)))
    with pytest.raises(TypeError, match="Opaque callables"):
        _environment(initializer=functools.partial(_initial_state))
    with pytest.raises(TypeError, match="Opaque callables"):
        _environment(transition=closure_transition)
    with pytest.raises(TypeError, match="Opaque callables"):
        _environment(
            initializer=lambda key: jnp.zeros((2,)),
            initializer_semantic_id="zero-initializer",
        )

    baseline = _environment()
    with pytest.raises(TypeError, match="Opaque callables"):
        PreparedRobotEnvironment(
            baseline.plant,
            baseline.parameters,
            baseline.task,
            (_ShapedWrapper(lambda value: 0.5 * value),),
            step_size=1.0,
        )


def test_wrapper_callable_fields_bind_function_content() -> None:
    baseline = _environment()

    def prepared(shaping: Any) -> Any:
        return PreparedRobotEnvironment(
            baseline.plant,
            baseline.parameters,
            baseline.task,
            (_ShapedWrapper(shaping),),
            step_size=1.0,
        )

    assert prepared(_halve).provenance_id == prepared(_halve).provenance_id
    assert prepared(_halve).provenance_id != prepared(_double).provenance_id


class _BatchedTask(AbstractRobotTask):
    task_id: str = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    reward_component_names: tuple[str, ...] = eqx.field(static=True)
    descriptor_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self) -> None:
        self.task_id = "batched-complete-state-task"
        self.observation_shape = (1,)
        self.reward_component_names = ("progress",)
        self.descriptor_shape = (1,)

    def initialize(self, plant_state: Any, key: Any, /) -> Any:
        del key
        return {"accepted_position": plant_state.payload["position"][..., 0]}

    def evaluate(self, plant_state: Any, task_state: Any, /) -> Any:
        position = plant_state.payload["position"]
        return RobotTaskEvaluation(
            position,
            position[..., 0] >= 1.0,
            task_state["accepted_position"][..., None],
        )

    def transition(
        self,
        context: Any,
        source_plant_state: Any,
        accepted_plant_state: Any,
        action: Any,
        task_state: Any,
        key: Any,
        /,
    ) -> Any:
        del context, action, task_state, key
        source = source_plant_state.payload["position"][..., 0]
        accepted = accepted_plant_state.payload["position"][..., 0]
        return RobotTaskTransition(
            {"accepted_position": accepted},
            accepted[..., None],
            (accepted - source)[..., None],
            accepted >= 1.0,
            accepted[..., None],
        )


class _BatchedEpisodeWrapper(AbstractRobotEnvironmentWrapper):
    wrapper_id: str = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    horizon: int | None = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)

    def __init__(self) -> None:
        self.wrapper_id = "batched-auto-reset"
        self.action_repeat = 1
        self.horizon = None
        self.auto_reset = True

    def initialize(self, plant_state: Any, task_state: Any, key: Any, /) -> Any:
        del task_state, key
        return {"count": jnp.zeros_like(plant_state.step_index, dtype=jnp.int32)}

    def transition(
        self,
        context: Any,
        wrapper_state: Any,
        plant_state: Any,
        task_state: Any,
        observation: Any,
        terminated: Any,
        key: Any,
        /,
    ) -> Any:
        del context, task_state, observation, terminated, key
        return RobotEnvironmentWrapperTransition(
            {"count": wrapper_state["count"] + 1},
            jnp.zeros_like(plant_state.step_index, dtype="bool"),
        )


class _MixedBatchedPlant(AbstractDiscretePlant):
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: object
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    reset_successful: bool = eqx.field(static=True)

    def __init__(
        self, semantic_tag: Any = "mixed", *, reset_successful: Any = True
    ) -> None:
        fallback = {
            "position": jnp.zeros((1,), dtype=jnp.float32),
            "memory": {
                "mode": jnp.asarray(0, dtype=jnp.int32),
                "history": jnp.zeros((2,), dtype=jnp.float32),
            },
        }
        semantics = SemanticProvenance(
            {
                "kind": "mixed-test-plant",
                "semantic_tag": semantic_tag,
                "reset_successful": bool(reset_successful),
            }
        )
        self.state_schema = ArrayPyTreeSchema.from_tree(
            jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), fallback),
            case_ndim=1,
        )
        self.control_schema = ArrayPyTreeSchema.from_tree(
            {"drive": jnp.zeros((2, 1), dtype=jnp.float32)},
            case_ndim=1,
        )
        self.parameter_schema = ArrayPyTreeSchema.from_tree((), case_ndim=0)
        self.reset_fallback = fallback
        self.semantic_provenance = semantics
        self.numeric_revision = NumericRevision(semantics, ())
        self.execution_signature = ExecutableSignature(
            shapes={"position": (1,), "history": (2,), "drive": (1,)},
            dtypes={"position": jnp.float32, "mode": jnp.int32},
        )
        self.require_finite_state = True
        self.require_finite_controls = True
        self.reset_successful = bool(reset_successful)
        self.require_finite_parameters = True

    def propose_reset(
        self,
        keys: Any,
        parameters: Any,
        /,
        *,
        case_shape: Any,
        initial_time: Any,
    ) -> Any:
        del keys, parameters, initial_time
        payload = self.state_schema.zeros(case_shape)
        return PlantProposal(
            payload,
            payload,
            jnp.ones(case_shape, dtype="bool"),
            jnp.full(case_shape, self.reset_successful, dtype="bool"),
            jnp.full(
                case_shape,
                0 if self.reset_successful else 7,
                dtype=jnp.int32,
            ),
            jnp.full(
                case_shape,
                0 if self.reset_successful else 17,
                dtype=jnp.int32,
            ),
            (),
        )

    def propose_step(
        self, context: Any, source: Any, commands: Any, parameters: Any, keys: Any, /
    ) -> Any:
        del context, parameters, keys
        drive = commands["drive"][..., 0]
        candidate_position = source["position"][..., 0] + drive
        successful = candidate_position <= 1.0
        payload = {
            "position": candidate_position[..., None],
            "memory": {
                "mode": source["memory"]["mode"] + 1,
                "history": source["memory"]["history"] + drive[..., None],
            },
        }
        status = jnp.where(successful, 0, 9).astype(jnp.int32)
        return PlantProposal(
            payload,
            payload,
            jnp.ones(successful.shape, dtype="bool"),
            successful,
            status,
            status,
            (),
        )


def _mixed_environment(
    *, semantic_tag: Any = "mixed", reset_successful: Any = True
) -> Any:
    plant = _MixedBatchedPlant(semantic_tag, reset_successful=reset_successful)
    parameters = PlantParameters(
        (),
        plant.parameter_schema.schema_id,
        plant.numeric_revision,
    )
    return PreparedRobotEnvironment(
        plant,
        parameters,
        _BatchedTask(),
        (_BatchedEpisodeWrapper(),),
        step_size=0.5,
        environment_id="mixed-display-id",
    )


def test_robot_environment_scenario_3() -> None:
    environment = _mixed_environment()
    keys = jax.random.split(jax.random.key(60), 2)
    reset = environment.reset(keys, case_shape=(2,))
    source = reset.state

    result = environment.step(
        source,
        {"drive": jnp.asarray([[1.0], [2.0]], dtype=jnp.float32)},
    )

    assert jnp.array_equal(result.accepted, jnp.asarray([True, False]))
    assert jnp.array_equal(result.evidence.attempted, jnp.asarray([[True, True]]))
    assert jnp.array_equal(result.evidence.repeat_status, jnp.asarray([[0, 9]]))
    assert result.candidate_state.plant_state.payload["position"][1, 0] == 2.0
    assert result.candidate_state.plant_state.payload["memory"]["mode"][1] == 1
    assert jnp.array_equal(
        result.candidate_state.plant_state.payload["memory"]["history"][1],
        jnp.asarray([2.0, 2.0]),
    )
    assert result.candidate_state.task_state["accepted_position"][1] == 0.0
    assert result.candidate_observation[1, 0] == 0.0

    assert result.accepted_state.plant_state.payload["position"][0, 0] == 1.0
    assert result.accepted_state.plant_state.payload["position"][1, 0] == 0.0
    assert result.accepted_state.plant_state.payload["memory"]["mode"][0] == 1
    assert result.accepted_state.plant_state.payload["memory"]["mode"][1] == 0
    assert jnp.array_equal(
        result.accepted_state.plant_state.payload["memory"]["history"],
        jnp.asarray([[1.0, 1.0], [0.0, 0.0]]),
    )
    assert jnp.array_equal(
        result.accepted_state.task_state["accepted_position"],
        jnp.asarray([1.0, 0.0]),
    )
    assert result.accepted_state.plant_state.time[0] == 0.5
    assert result.accepted_state.plant_state.time[1] == 0.0
    assert result.accepted_state.plant_state.step_index[0] == 1
    assert result.accepted_state.plant_state.step_index[1] == 0
    assert jnp.array_equal(
        result.accepted_state.plant_state.key[1],
        source.plant_state.key[1],
    )
    assert jnp.array_equal(result.accepted_state.key[1], source.key[1])
    assert result.accepted_state.episode_step_index[1] == 0
    assert result.accepted_state.wrapper_states[0]["count"][1] == 0
    assert jnp.array_equal(
        result.accepted_state.wrapper_states[0]["count"],
        jnp.asarray([1, 0]),
    )
    assert jnp.array_equal(
        result.reward_components,
        jnp.asarray([[1.0], [0.0]]),
    )
    assert jnp.array_equal(result.terminated, jnp.asarray([True, False]))
    assert jnp.array_equal(result.truncated, jnp.asarray([False, False]))

    assert jnp.array_equal(result.reset_performed, jnp.asarray([True, False]))
    assert result.reset_state.plant_state.payload["position"][0, 0] == 0.0
    assert result.reset_state.plant_state.step_index[0] == 0
    assert result.reset_state.episode_step_index[0] == 0
    assert jnp.array_equal(
        result.reset_state.plant_state.key[1],
        source.plant_state.key[1],
    )
    assert jnp.array_equal(result.reset_state.key[1], source.key[1])
    assert result.reset_state.wrapper_states[0]["count"][1] == 0
    environment = _mixed_environment()
    stale_environment = _mixed_environment(semantic_tag="stale-mixed")
    keys = jax.random.split(jax.random.key(61), 2)
    state = environment.reset(keys, case_shape=(2,)).state
    stale = PlantRuntimeState(
        state.plant_state.payload,
        state.plant_state.time,
        state.plant_state.step_index,
        state.plant_state.key,
        stale_environment.plant.semantic_provenance.semantic_id,
        state.plant_state.numeric_revision_id,
        state.plant_state.state_schema_id,
        state.plant_state.execution_signature_id,
    )
    stale_state = eqx.tree_at(lambda item: item.plant_state, state, stale)

    with pytest.raises(ValueError, match="plant semantic provenance"):
        environment.step(
            stale_state,
            {"drive": jnp.zeros((2, 1), dtype=jnp.float32)},
        )


class _CountingWrapper(AbstractRobotEnvironmentWrapper):
    wrapper_id: str = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    horizon: int | None = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)

    def __init__(self, *, action_repeat: int = 1, auto_reset: bool = False) -> None:
        self.wrapper_id = "counting-wrapper"
        self.action_repeat = action_repeat
        self.horizon = None
        self.auto_reset = auto_reset

    def initialize(self, plant_state: Any, task_state: Any, key: Any, /) -> Any:
        del task_state, key
        return {"count": jnp.zeros(plant_state.step_index.shape, dtype=jnp.int32)}

    def transition(
        self,
        context: Any,
        wrapper_state: Any,
        plant_state: Any,
        task_state: Any,
        observation: Any,
        terminated: Any,
        key: Any,
        /,
    ) -> Any:
        del context, task_state, observation, terminated, key
        return RobotEnvironmentWrapperTransition(
            {"count": wrapper_state["count"] + 1},
            jnp.zeros(plant_state.step_index.shape, dtype=jnp.bool_),
        )


class _AlternatingHistoryPlant(AbstractDiscretePlant):
    """Actuator memory and a boolean mode decide the next position."""

    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: dict[str, jax.Array]
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)

    def __init__(self) -> None:
        fallback = {
            "position": jnp.zeros((1,), dtype=jnp.float32),
            "memory": jnp.zeros((1,), dtype=jnp.float32),
            "mode": jnp.asarray(False),
        }
        self.state_schema = ArrayPyTreeSchema.from_tree(
            jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), fallback),
            case_ndim=1,
        )
        self.control_schema = ArrayPyTreeSchema.from_tree(
            jnp.zeros((2, 1), dtype=jnp.float32), case_ndim=1
        )
        self.parameter_schema = ArrayPyTreeSchema.from_tree((), case_ndim=0)
        self.reset_fallback = fallback
        self.semantic_provenance = SemanticProvenance(
            {"kind": "alternating-actuator-memory-environment-plant"}
        )
        self.numeric_revision = NumericRevision(self.semantic_provenance, ())
        self.execution_signature = ExecutableSignature(
            shapes={"position": (1,), "memory": (1,), "command": (1,)},
            dtypes={"position": jnp.float32, "mode": jnp.bool_},
        )
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True

    def propose_reset(
        self,
        keys: Any,
        parameters: Any,
        /,
        *,
        case_shape: Any,
        initial_time: Any,
    ) -> Any:
        del keys, parameters, initial_time
        payload = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, case_shape + leaf.shape),
            self.reset_fallback,
        )
        valid = jnp.ones(case_shape, dtype=jnp.bool_)
        status = jnp.zeros(case_shape, dtype=jnp.int32)
        return PlantProposal(payload, payload, valid, valid, status, status, ())

    def propose_step(
        self, context: Any, source: Any, commands: Any, parameters: Any, keys: Any, /
    ) -> Any:
        del parameters, keys
        duration = context.duration.astype(jnp.float32)[..., None]
        candidate = {
            "position": source["position"] + duration * (commands + source["memory"]),
            "memory": jnp.where(source["mode"][..., None], -commands, commands),
            "mode": ~source["mode"],
        }
        successful = commands[..., 0] >= -2.0
        status = jnp.where(successful, 0, 37).astype(jnp.int32)
        return PlantProposal(
            candidate,
            candidate,
            jnp.ones(successful.shape, dtype=jnp.bool_),
            successful,
            status,
            status,
            (),
        )


class _PositionTask(AbstractRobotTask):
    task_id: str = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    reward_component_names: tuple[str, ...] = eqx.field(static=True)
    descriptor_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self) -> None:
        self.task_id = "position-progress-task"
        self.observation_shape = (1,)
        self.reward_component_names = ("progress",)
        self.descriptor_shape = (1,)

    def initialize(self, plant_state: Any, key: Any, /) -> Any:
        del key
        return {"progress": jnp.zeros_like(plant_state.payload["position"][..., 0])}

    def evaluate(self, plant_state: Any, task_state: Any, /) -> Any:
        del task_state
        position = plant_state.payload["position"]
        return RobotTaskEvaluation(
            position,
            jnp.zeros(position.shape[:-1], dtype=jnp.bool_),
            plant_state.payload["memory"],
        )

    def transition(
        self,
        context: Any,
        source_plant_state: Any,
        accepted_plant_state: Any,
        action: Any,
        task_state: Any,
        key: Any,
        /,
    ) -> Any:
        del context, action, key
        accepted = accepted_plant_state.payload["position"]
        progress = accepted - source_plant_state.payload["position"]
        return RobotTaskTransition(
            {"progress": task_state["progress"] + progress[..., 0]},
            accepted,
            progress,
            jnp.zeros(accepted.shape[:-1], dtype=jnp.bool_),
            accepted_plant_state.payload["memory"],
        )


def _history_environment() -> PreparedRobotEnvironment:
    plant = _AlternatingHistoryPlant()
    parameters = PlantParameters(
        (), plant.parameter_schema.schema_id, plant.numeric_revision
    )
    return PreparedRobotEnvironment(
        plant,
        parameters,
        _PositionTask(),
        (_CountingWrapper(action_repeat=2),),
        step_size=0.5,
    )


def _case(tree: Any, index: int) -> Any:
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _assert_state_equal(left: Any, right: Any) -> None:
    """Compare complete states exactly, including typed plant PRNG keys."""

    def raw(leaf: Any) -> Any:
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            return jax.random.key_data(leaf)
        return leaf

    assert_tree_equal(jax.tree.map(raw, left), jax.tree.map(raw, right))


def _manual_steps(
    environment: PreparedRobotEnvironment,
    state: PlantRuntimeState,
    command: Any,
    count: int,
) -> PlantRuntimeState:
    for _ in range(count):
        context = PlantStepContext(
            state.time,
            state.time + jnp.asarray(environment.step_size, dtype=state.time.dtype),
            state.step_index,
        )
        result = environment.plant.step(context, state, command, environment.parameters)
        assert bool(jnp.all(result.successful))
        state = result.accepted_state
    return state


def test_history_with_equal_positions_steps_differently_and_rolls_back_casewise() -> None:
    environment = _history_environment()
    keys = jax.random.split(jax.random.key(80), 2)
    reset = environment.reset(keys, case_shape=(2,))
    source = eqx.tree_at(
        lambda state: (
            state.plant_state.payload["memory"],
            state.plant_state.payload["mode"],
        ),
        reset.state,
        (jnp.asarray([[0.0], [1.0]], dtype=jnp.float32), jnp.asarray([False, True])),
    )
    command = jnp.ones((2, 1), dtype=jnp.float32)

    result = environment.step(source, command)
    manual = _manual_steps(environment, source.plant_state, command, 2)

    assert np.asarray(result.accepted).tolist() == [True, True]
    assert environment.plant.state_digest(
        result.accepted_state.plant_state
    ) == environment.plant.state_digest(manual)
    assert np.asarray(result.accepted_state.plant_state.payload["position"]).tolist() == [
        [1.5],
        [1.0],
    ]
    assert np.asarray(result.accepted_state.plant_state.payload["memory"]).tolist() == [
        [-1.0],
        [1.0],
    ]
    assert np.asarray(result.accepted_state.plant_state.payload["mode"]).tolist() == [
        False,
        True,
    ]
    assert np.asarray(result.accepted_state.wrapper_states[0]["count"]).tolist() == [2, 2]
    assert np.asarray(result.accepted_state.task_state["progress"]).tolist() == [1.5, 1.0]

    advanced = result.accepted_state
    rejected = environment.step(advanced, jnp.asarray([[1.0], [-3.0]], dtype=jnp.float32))

    assert np.asarray(rejected.accepted).tolist() == [True, False]
    assert np.asarray(rejected.evidence.attempted).tolist() == [
        [True, True],
        [True, False],
    ]
    assert np.asarray(rejected.evidence.repeat_status).tolist() == [[0, 37], [0, 0]]
    _assert_state_equal(_case(rejected.accepted_state, 1), _case(advanced, 1))
    assert rejected.candidate_state.plant_state.payload["mode"][1] == jnp.asarray(False)
    assert rejected.reward_components[1, 0] == 0.0
    assert int(rejected.accepted_state.plant_state.step_index[0]) == 4
    # Episode and wrapper counters advance once per accepted repeated substep.
    assert int(rejected.accepted_state.episode_step_index[0]) == 4
    assert int(rejected.accepted_state.wrapper_states[0]["count"][0]) == 4


class _OffsetInitializer(StrictModule):
    offset: jax.Array

    def __init__(self, offset: Any) -> None:
        self.offset = jnp.asarray(offset)

    def __call__(self, key: Any) -> Any:
        del key
        return jnp.stack((self.offset, jnp.zeros_like(self.offset)))


def test_task_and_initializer_arrays_remain_dynamic_under_compilation() -> None:
    environment = _environment(
        threshold=1.0, initializer=_OffsetInitializer(jnp.zeros(()))
    )
    threshold_dtype = environment.task.threshold.dtype
    offset_dtype = environment.plant.initializer.offset.dtype
    action = jnp.asarray([0.25])

    def run(threshold: jax.Array, offset: jax.Array) -> Any:
        prepared = eqx.tree_at(
            lambda value: (value.task.threshold, value.plant.initializer.offset),
            environment,
            (threshold, offset),
        )
        reset = prepared.reset(jax.random.key(90))
        result = prepared.step(reset.state, action)
        return (
            reset.state.plant_state.payload[0],
            result.terminated,
            result.accepted_state.plant_state.payload[0],
        )

    # Batched task/initializer arrays are traced data of one compiled program.
    initial, terminated, accepted = jax.jit(jax.vmap(run))(
        jnp.asarray([1.0, 0.5], dtype=threshold_dtype),
        jnp.asarray([0.0, 0.5], dtype=offset_dtype),
    )

    assert np.asarray(initial).tolist() == [0.0, 0.5]
    assert np.asarray(terminated).tolist() == [False, True]
    assert np.asarray(accepted).tolist() == [0.25, 0.75]


@functools.cache
def _native_plant(mode: ArticulatedActuationMode) -> PreparedArticulatedPlant:
    model = slider_model()
    plan = NativeArticulatedPlantPlan(
        step_policy(),
        gravity=standard_gravity(),
        actuation_mode=mode,
        initial_lifted_configuration=[0.002],
        initial_velocity=[0.0],
        case_ndim=1,
        activation_distance=0.05,
        search_backend="lbvh",
        route_capacities=route_budget(model),
        maximum_traversal_visits=LBVH_TRAVERSAL_VISITS,
    )
    return prepare_articulated_plant(model, plan)


def _command_codec(
    plant: PreparedArticulatedPlant,
    *,
    size: int = 1,
    case_ndim: int = 0,
    numeric_revision: NumericRevision | None = None,
) -> ControlVectorCodec:
    dtype = plant.initial_state.command.dtype
    template = ArticulatedPlantCommand(jnp.zeros((1,) * case_ndim + (size,), dtype=dtype))
    return ControlVectorCodec(
        ArrayPyTreeSchema.from_tree(template, case_ndim=case_ndim),
        semantic_provenance=plant.semantic_provenance,
        numeric_revision=(
            plant.numeric_revision if numeric_revision is None else numeric_revision
        ),
        executable_signature=plant.execution_signature,
    )


class _NativeHeightTask(AbstractRobotTask):
    ceiling: jax.Array
    task_id: str = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    reward_component_names: tuple[str, ...] = eqx.field(static=True)
    descriptor_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, ceiling: float) -> None:
        self.ceiling = jnp.asarray(ceiling)
        self.task_id = "native-slider-height-task"
        self.observation_shape = (2,)
        self.reward_component_names = ("rise",)
        self.descriptor_shape = (1,)

    def _evaluation(self, plant_state: Any) -> RobotTaskEvaluation:
        payload = plant_state.payload
        height = payload.configuration[..., 0]
        return RobotTaskEvaluation(
            jnp.stack((height, payload.velocity[..., 0]), axis=-1),
            height >= self.ceiling,
            payload.sensors.frame_transforms[..., 0, 2, 3:4],
        )

    def initialize(self, plant_state: Any, key: Any, /) -> Any:
        del key
        return {"steps": jnp.zeros(plant_state.step_index.shape, dtype=jnp.int32)}

    def evaluate(self, plant_state: Any, task_state: Any, /) -> Any:
        del task_state
        return self._evaluation(plant_state)

    def transition(
        self,
        context: Any,
        source_plant_state: Any,
        accepted_plant_state: Any,
        action: Any,
        task_state: Any,
        key: Any,
        /,
    ) -> Any:
        del context, action, key
        evaluation = self._evaluation(accepted_plant_state)
        rise = (
            accepted_plant_state.payload.configuration
            - source_plant_state.payload.configuration
        )
        return RobotTaskTransition(
            {"steps": task_state["steps"] + 1},
            evaluation.observation,
            rise,
            evaluation.terminated,
            evaluation.descriptor,
        )


def _native_environment(
    *,
    mode: ArticulatedActuationMode = ArticulatedActuationMode.GENERALIZED_EFFORT,
    repeat: int = 1,
    auto_reset: bool = False,
    ceiling: float = 1.0,
) -> PreparedRobotEnvironment:
    plant = _native_plant(mode)
    return PreparedRobotEnvironment(
        plant,
        plant.bind_parameters(),
        _NativeHeightTask(ceiling),
        (_CountingWrapper(action_repeat=repeat, auto_reset=auto_reset),),
        step_size=0.01,
        control_codec=(
            None if mode == ArticulatedActuationMode.AUTONOMOUS else _command_codec(plant)
        ),
    )


def _native_reset(environment: PreparedRobotEnvironment) -> Any:
    reset = environment.reset(jax.random.split(jax.random.key(70), 2), case_shape=(2,))
    assert np.asarray(reset.successful).tolist() == [True, True]
    # The native plant admits only typed keys; the environment keeps raw data.
    assert jax.dtypes.issubdtype(reset.state.plant_state.key.dtype, jax.dtypes.prng_key)
    assert reset.state.key.dtype == jnp.uint32
    return reset


def _native_vector(values: Any) -> jax.Array:
    plant = _native_plant(ArticulatedActuationMode.GENERALIZED_EFFORT)
    return jnp.asarray(values, dtype=plant.initial_state.command.dtype)


def test_native_plant_environment_matches_manual_complete_transactions() -> None:
    environment = _native_environment(repeat=2)
    plant = environment.plant
    assert isinstance(plant, PreparedArticulatedPlant)
    state = _native_reset(environment).state
    manual = state.plant_state
    actions = _native_vector([[0.0], [5.0]])
    command = ArticulatedPlantCommand(actions)
    expected_time = state.plant_state.time

    for _ in range(2):
        result = environment.step(state, actions)
        assert np.asarray(result.accepted).tolist() == [True, True]
        manual = _manual_steps(environment, manual, command, 2)
        state = result.accepted_state
        assert plant.state_digest(state.plant_state) == plant.state_digest(manual)
        for _ in range(2):
            expected_time = expected_time + jnp.asarray(0.01, dtype=expected_time.dtype)

    payload = state.plant_state.payload
    observation = plant.observe(state.plant_state)
    assert np.array_equal(np.asarray(state.plant_state.time), np.asarray(expected_time))
    assert np.asarray(state.plant_state.step_index).tolist() == [4, 4]
    assert np.asarray(state.episode_step_index).tolist() == [4, 4]
    assert np.asarray(state.wrapper_states[0]["count"]).tolist() == [4, 4]
    assert np.asarray(state.task_state["steps"]).tolist() == [4, 4]
    assert np.asarray(payload.epoch).tolist() == [4, 4]
    assert np.asarray(observation.fresh).tolist() == [True, True]
    assert np.asarray(payload.command).tolist() == [[0.0], [5.0]]
    assert payload.contact_history is not None
    assert bool(jnp.any(payload.contact_history.active[0]))
    assert float(jnp.sum(payload.sensors.contact_normal_impulse[0])) > 0.0

    stale = eqx.tree_at(
        lambda value: value.plant_state.payload.sensor_epoch,
        state,
        payload.sensor_epoch - 1,
    )
    assert np.asarray(plant.observe(stale.plant_state).fresh).tolist() == [False, False]
    refreshed = environment.step(stale, actions)
    assert np.asarray(refreshed.accepted).tolist() == [True, True]
    assert np.asarray(
        plant.observe(refreshed.accepted_state.plant_state).fresh
    ).tolist() == [
        True,
        True,
    ]


def test_native_environment_failure_rolls_back_complete_case_state() -> None:
    environment = _native_environment(repeat=2)
    plant = environment.plant
    reset = _native_reset(environment)
    source = environment.step(reset.state, _native_vector([[0.0], [0.0]])).accepted_state

    result = environment.step(source, _native_vector([[30.0], [jnp.nan]]))

    assert np.asarray(result.accepted).tolist() == [True, False]
    assert np.asarray(result.evidence.rollback_applied).tolist() == [False, True]
    assert np.asarray(result.evidence.attempted).tolist() == [[True, True], [True, False]]
    assert int(result.evidence.repeat_status[0, 1]) != 0
    _assert_state_equal(_case(result.accepted_state, 1), _case(source, 1))
    _assert_state_equal(_case(result.reset_state, 1), _case(source, 1))
    assert np.array_equal(
        np.asarray(result.final_observation[1]),
        np.asarray(
            environment.task.evaluate(source.plant_state, source.task_state).observation[
                1
            ]
        ),
    )
    assert result.reward_components[1, 0] == 0.0
    assert not bool(result.terminated[1]) and not bool(result.truncated[1])
    assert int(result.accepted_state.plant_state.step_index[0]) == 4
    assert float(result.accepted_state.plant_state.payload.velocity[0, 0]) > float(
        source.plant_state.payload.velocity[0, 0]
    )
    assert np.asarray(result.accepted_state.plant_state.payload.command[0]).tolist() == [
        30.0
    ]
    assert plant.state_digest(
        jax.tree.map(lambda leaf: leaf[1:], result.accepted_state.plant_state)
    ) == plant.state_digest(jax.tree.map(lambda leaf: leaf[1:], source.plant_state))


def test_native_auto_reset_separates_terminal_and_reset_observations() -> None:
    environment = _native_environment(repeat=2, auto_reset=True, ceiling=-1.0)
    reset = _native_reset(environment)

    result = environment.step(reset.state, _native_vector([[0.0], [10.0]]))

    assert np.asarray(result.accepted).tolist() == [True, True]
    assert np.asarray(result.terminated).tolist() == [True, True]
    assert np.asarray(result.reset_performed).tolist() == [True, True]
    assert np.asarray(result.reset_successful).tolist() == [True, True]
    assert np.asarray(result.reset_status).tolist() == [0, 0]
    assert np.asarray(result.evidence.attempted).tolist() == [
        [True, True],
        [False, False],
    ]
    assert np.asarray(result.accepted_state.plant_state.step_index).tolist() == [1, 1]
    assert np.array_equal(
        np.asarray(result.final_observation[..., 0]),
        np.asarray(result.accepted_state.plant_state.payload.configuration[..., 0]),
    )
    assert np.array_equal(np.asarray(result.observation), np.asarray(reset.observation))
    assert not np.array_equal(
        np.asarray(result.observation), np.asarray(result.final_observation)
    )
    assert_tree_equal(
        result.reset_state.plant_state.payload, reset.state.plant_state.payload
    )
    assert np.asarray(result.reset_state.plant_state.step_index).tolist() == [0, 0]
    assert np.asarray(result.reset_state.plant_state.time).tolist() == [0.0, 0.0]
    assert np.asarray(result.reset_state.episode_step_index).tolist() == [0, 0]
    assert not np.array_equal(
        np.asarray(jax.random.key_data(result.reset_state.plant_state.key)),
        np.asarray(jax.random.key_data(result.accepted_state.plant_state.key)),
    )
    assert not np.array_equal(
        np.asarray(result.reset_state.key), np.asarray(result.accepted_state.key)
    )


def test_native_environment_history_replays_from_plant_checkpoint() -> None:
    environment = _native_environment()
    plant = environment.plant
    initial = _native_reset(environment).state
    checkpoint = plant.checkpoint(initial.plant_state)
    actions = (
        _native_vector([[0.0], [0.0]]),
        _native_vector([[25.0], [5.0]]),
        _native_vector([[0.0], [40.0]]),
    )
    contexts: list[PlantStepContext] = []
    digests: list[str] = []
    state = initial
    for action in actions:
        plant_state = state.plant_state
        contexts.append(
            PlantStepContext(
                plant_state.time,
                plant_state.time + jnp.asarray(0.01, dtype=plant_state.time.dtype),
                plant_state.step_index,
            )
        )
        result = environment.step(state, action)
        assert np.asarray(result.accepted).tolist() == [True, True]
        state = result.accepted_state
        digests.append(plant.state_digest(state.plant_state))

    replay = plant.replay(
        checkpoint,
        contexts,
        [ArticulatedPlantCommand(action) for action in actions],
        environment.parameters,
        expected_digests=digests,
    )
    repeated = initial
    for action in actions:
        repeated = environment.step(repeated, action).accepted_state

    assert replay.matched
    assert bool(jnp.all(replay.successful))
    _assert_state_equal(repeated, state)


def test_autonomous_plant_environment_and_command_codec_binding() -> None:
    controlled = _native_plant(ArticulatedActuationMode.GENERALIZED_EFFORT)
    environment = _native_environment(mode=ArticulatedActuationMode.AUTONOMOUS)
    reset = _native_reset(environment)

    result = environment.step(reset.state, None)

    assert np.asarray(result.accepted).tolist() == [True, True]
    assert np.asarray(result.accepted_state.plant_state.step_index).tolist() == [1, 1]
    with pytest.raises(ValueError, match="action=None"):
        environment.step(reset.state, _native_vector([[0.0], [0.0]]))

    def prepared(plant: Any, codec: ControlVectorCodec) -> PreparedRobotEnvironment:
        return PreparedRobotEnvironment(
            plant,
            plant.bind_parameters(),
            _NativeHeightTask(1.0),
            step_size=0.01,
            control_codec=codec,
        )

    with pytest.raises(ValueError, match="Autonomous"):
        prepared(environment.plant, _command_codec(controlled))
    with pytest.raises(ValueError, match="unbatched"):
        prepared(controlled, _command_codec(controlled, case_ndim=1))
    with pytest.raises(ValueError, match="complete plant command"):
        prepared(controlled, _command_codec(controlled, size=2))
    stale_revision = NumericRevision(controlled.semantic_provenance, {"stale": 1})
    with pytest.raises(ValueError, match="identities"):
        prepared(controlled, _command_codec(controlled, numeric_revision=stale_revision))

    bound = _native_environment()
    unbound = PreparedRobotEnvironment(
        controlled,
        controlled.bind_parameters(),
        _NativeHeightTask(1.0),
        (_CountingWrapper(),),
        step_size=0.01,
    )
    assert bound.provenance_id != unbound.provenance_id
    bound_state = _native_reset(bound).state
    unbound_state = _native_reset(unbound).state
    vector = _native_vector([[3.0], [0.0]])
    by_vector = bound.step(bound_state, vector)
    by_command = unbound.step(unbound_state, ArticulatedPlantCommand(vector))
    assert controlled.state_digest(
        by_vector.accepted_state.plant_state
    ) == controlled.state_digest(by_command.accepted_state.plant_state)
    with pytest.raises(ValueError, match="codec size"):
        bound.step(bound_state, _native_vector([[3.0, 1.0], [0.0, 1.0]]))
