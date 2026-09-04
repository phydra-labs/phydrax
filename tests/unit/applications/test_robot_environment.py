import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax._array_tree import ArrayPyTreeSchema
from phydrax._identity import ExecutableSignature, NumericRevision, SemanticProvenance
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
)
from phydrax.dynamics._system import DiscreteTransitionResult


class _ThresholdTask(AbstractRobotTask):
    threshold: jax.Array
    task_id: str = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    reward_component_names: tuple[str, ...] = eqx.field(static=True)
    descriptor_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, threshold):
        self.threshold = jnp.asarray(threshold)
        self.task_id = "threshold-task"
        self.observation_shape = (2,)
        self.reward_component_names = ("progress", "effort")
        self.descriptor_shape = (1,)

    def initialize(self, plant_state, key, /):
        del plant_state
        return jnp.asarray([0.0, key[0].astype(jnp.float32)])

    def evaluate(self, plant_state, task_state, /):
        payload = plant_state.payload
        return RobotTaskEvaluation(
            jnp.asarray([payload[0], task_state[0]]),
            payload[0] >= self.threshold,
            payload[:1],
        )

    def transition(
        self,
        context,
        source_plant_state,
        accepted_plant_state,
        action,
        task_state,
        key,
        /,
    ):
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

    def __init__(self, *, action_repeat, horizon=None, auto_reset=False):
        self.wrapper_id = "episode-wrapper"
        self.action_repeat = action_repeat
        self.horizon = horizon
        self.auto_reset = auto_reset

    def initialize(self, plant_state, task_state, key, /):
        del plant_state, task_state
        return jnp.asarray([0.0, key[0].astype(jnp.float32)])

    def transition(
        self,
        context,
        wrapper_state,
        plant_state,
        task_state,
        observation,
        terminated,
        key,
        /,
    ):
        del context, plant_state, task_state, observation, terminated
        candidate = wrapper_state.at[0].add(1.0)
        candidate = candidate.at[1].set(key[0].astype(wrapper_state.dtype))
        return RobotEnvironmentWrapperTransition(candidate, jnp.asarray(False))


class _PlantRecordingWrapper(AbstractRobotEnvironmentWrapper):
    wrapper_id: str = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    horizon: int | None = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)

    def __init__(self, *, action_repeat=1, horizon=None, auto_reset=False):
        self.wrapper_id = "plant-recording-wrapper"
        self.action_repeat = action_repeat
        self.horizon = horizon
        self.auto_reset = auto_reset

    def initialize(self, plant_state, task_state, key, /):
        del plant_state, task_state, key
        return jnp.zeros((1,))

    def transition(
        self,
        context,
        wrapper_state,
        plant_state,
        task_state,
        observation,
        terminated,
        key,
        /,
    ):
        del context, wrapper_state, task_state, observation, terminated, key
        return RobotEnvironmentWrapperTransition(
            plant_state.payload[:1],
            jnp.asarray(False),
        )


def _initial_state(key):
    del key
    return jnp.zeros((2,))


def _bounded_transition(
    context: DiscreteStepContext,
    state,
    action,
    args,
):
    del context
    limit = jnp.asarray(1.0 if args is None else args)
    candidate = state.at[0].add(action[0])
    successful = candidate[0] <= limit
    accepted = jnp.where(successful, candidate, state)
    status = jnp.where(successful, 0, 7).astype(jnp.int32)
    return DiscreteTransitionResult(candidate, accepted, successful, status)


def _projected_transition(context, state, action, args):
    del context, args
    candidate = state.at[0].add(10.0 * action[0])
    accepted = state.at[0].add(action[0])
    return DiscreteTransitionResult(
        candidate,
        accepted,
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _environment(
    *,
    threshold=100.0,
    repeat=1,
    horizon=None,
    auto_reset=False,
    transition=_bounded_transition,
    state_layout=None,
    input_layout=None,
    system_id="bounded-plant",
    initializer_id="zero-initializer",
    environment_id=None,
):
    system = DiscreteSystem(
        transition,
        state_layout=StateLayout((2,)) if state_layout is None else state_layout,
        input_layout=InputLayout((1,)) if input_layout is None else input_layout,
        system_id=system_id,
        step_size=1.0,
    )
    return prepare_array_robot_environment(
        system,
        _initial_state,
        _ThresholdTask(threshold),
        (
            _EpisodeWrapper(
                action_repeat=repeat,
                horizon=horizon,
                auto_reset=auto_reset,
            ),
        ),
        initializer_id=initializer_id,
        reset_fallback=jnp.zeros((2,)),
        parameter_values=jnp.asarray(1.0),
        environment_id=environment_id,
    )


def _assert_tree_equal(left, right):
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        assert jnp.array_equal(left_leaf, right_leaf)


def test_array_system_bridge_preserves_repetition_and_atomic_rollback():
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
    _assert_tree_equal(result.accepted_state, reset.state)
    assert result.total_reward == 0.0
    assert not result.terminated
    assert not result.truncated


def test_array_adapter_accepted_state_drives_task_wrappers_and_repetition():
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


def test_domain_termination_and_horizon_use_environment_episode_index():
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


def test_auto_reset_keeps_terminal_outputs_and_resets_both_state_domains():
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


def test_environment_provenance_binds_all_plant_identities_and_task_content():
    display_id = "shared-display-id"
    baseline = _environment(threshold=1.0, environment_id=display_id)
    changed_task = _environment(threshold=2.0, environment_id=display_id)
    changed_initializer = _environment(
        threshold=1.0,
        initializer_id="different-initializer",
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


class _BatchedTask(AbstractRobotTask):
    task_id: str = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    reward_component_names: tuple[str, ...] = eqx.field(static=True)
    descriptor_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self):
        self.task_id = "batched-complete-state-task"
        self.observation_shape = (1,)
        self.reward_component_names = ("progress",)
        self.descriptor_shape = (1,)

    def initialize(self, plant_state, key, /):
        del key
        return {"accepted_position": plant_state.payload["position"][..., 0]}

    def evaluate(self, plant_state, task_state, /):
        position = plant_state.payload["position"]
        return RobotTaskEvaluation(
            position,
            position[..., 0] >= 1.0,
            task_state["accepted_position"][..., None],
        )

    def transition(
        self,
        context,
        source_plant_state,
        accepted_plant_state,
        action,
        task_state,
        key,
        /,
    ):
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

    def __init__(self):
        self.wrapper_id = "batched-auto-reset"
        self.action_repeat = 1
        self.horizon = None
        self.auto_reset = True

    def initialize(self, plant_state, task_state, key, /):
        del task_state, key
        return {"count": jnp.zeros_like(plant_state.step_index, dtype=jnp.int32)}

    def transition(
        self,
        context,
        wrapper_state,
        plant_state,
        task_state,
        observation,
        terminated,
        key,
        /,
    ):
        del context, task_state, observation, terminated, key
        return RobotEnvironmentWrapperTransition(
            {"count": wrapper_state["count"] + 1},
            jnp.zeros_like(plant_state.step_index, dtype=bool),
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

    def __init__(self, semantic_tag="mixed-v1"):
        fallback = {
            "position": jnp.zeros((1,), dtype=jnp.float32),
            "memory": {
                "mode": jnp.asarray(0, dtype=jnp.int32),
                "history": jnp.zeros((2,), dtype=jnp.float32),
            },
        }
        semantics = SemanticProvenance(
            {"kind": "mixed-test-plant", "semantic_tag": semantic_tag}
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
        self.require_finite_parameters = True

    def propose_reset(
        self,
        keys,
        parameters,
        /,
        *,
        case_shape,
        initial_time,
    ):
        del keys, parameters, initial_time
        payload = self.state_schema.zeros(case_shape)
        return PlantProposal(
            payload,
            payload,
            jnp.ones(case_shape, dtype=bool),
            jnp.ones(case_shape, dtype=bool),
            jnp.zeros(case_shape, dtype=jnp.int32),
            jnp.zeros(case_shape, dtype=jnp.int32),
            (),
        )

    def propose_step(self, context, source, commands, parameters, keys, /):
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
            jnp.ones(successful.shape, dtype=bool),
            successful,
            status,
            status,
            (),
        )


def _mixed_environment(*, semantic_tag="mixed-v1"):
    plant = _MixedBatchedPlant(semantic_tag)
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


def test_mixed_pytree_cases_never_expose_failed_candidate_and_roll_back_all_leaves():
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


def test_same_shape_stale_plant_provenance_is_rejected_before_transition():
    environment = _mixed_environment()
    stale_environment = _mixed_environment(semantic_tag="mixed-v2")
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
