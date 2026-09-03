import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.robotics._environment import (
    AbstractRobotEnvironmentWrapper,
    AbstractRobotTask,
    PreparedRobotEnvironment,
    RobotEnvironmentWrapperTransition,
    RobotTaskEvaluation,
    RobotTaskTransition,
)
from phydrax.dynamics import DiscreteStepContext, DiscreteSystem, InputLayout, StateLayout
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
        return RobotTaskEvaluation(
            jnp.asarray([plant_state[0], task_state[0]]),
            plant_state[0] >= self.threshold,
            plant_state[:1],
        )

    def transition(
        self,
        context,
        source_plant_state,
        candidate_plant_state,
        action,
        task_state,
        key,
        /,
    ):
        del context
        candidate_task_state = task_state.at[0].add(1.0)
        candidate_task_state = candidate_task_state.at[1].set(
            key[0].astype(task_state.dtype)
        )
        return RobotTaskTransition(
            candidate_task_state,
            jnp.asarray([candidate_plant_state[0], candidate_task_state[0]]),
            jnp.asarray(
                [
                    candidate_plant_state[0] - source_plant_state[0],
                    -(action[0] * action[0]),
                ]
            ),
            candidate_plant_state[0] >= self.threshold,
            candidate_plant_state[:1],
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


def _initial_state(key):
    del key
    return jnp.zeros((2,))


def _bounded_transition(
    context: DiscreteStepContext,
    state,
    action,
    args,
):
    limit = jnp.asarray(1.0 if args is None else args)
    candidate = state.at[0].add(action[0])
    successful = candidate[0] <= limit
    accepted = jnp.where(successful, candidate, state)
    status = jnp.where(successful, 0, 7).astype(jnp.int32)
    return DiscreteTransitionResult(candidate, accepted, successful, status)


def _environment(*, threshold=100.0, repeat=1, horizon=None, auto_reset=False):
    system = DiscreteSystem(
        _bounded_transition,
        state_layout=StateLayout((2,)),
        input_layout=InputLayout((1,)),
        system_id="bounded-plant",
        step_size=1.0,
    )
    return PreparedRobotEnvironment(
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
        initializer_id="zero-initializer",
    )


def test_failed_repeat_rolls_back_every_leaf_and_retains_candidate_diagnostics():
    environment = _environment(repeat=2)
    reset = environment.reset(jax.random.key(4))

    result = environment.step(reset.state, jnp.asarray([1.0]))

    assert not result.evidence.accepted
    assert result.evidence.rollback_applied
    assert not result.evidence.mechanics_successful
    assert result.evidence.mechanics_status == 7
    assert not result.mechanics_successful
    assert result.mechanics_status == 7
    assert result.provenance_id == environment.provenance_id
    assert jnp.array_equal(result.evidence.attempted, jnp.asarray([True, True]))
    assert jnp.array_equal(
        result.evidence.repeat_successful, jnp.asarray([True, False])
    )
    assert result.candidate_state.plant_state[0] == 2.0
    assert result.candidate_state.task_state[0] == 2.0
    assert result.candidate_state.wrapper_states[0][0] == 2.0
    assert result.candidate_observation[0] == 2.0
    assert jnp.array_equal(result.accepted_state.plant_state, reset.state.plant_state)
    assert jnp.array_equal(result.accepted_state.key, reset.state.key)
    assert result.accepted_state.clock == reset.state.clock
    assert result.accepted_state.step_index == reset.state.step_index
    assert jnp.array_equal(result.accepted_state.task_state, reset.state.task_state)
    assert jnp.array_equal(
        result.accepted_state.wrapper_states[0], reset.state.wrapper_states[0]
    )
    assert result.total_reward == 0.0
    assert not result.terminated
    assert not result.truncated


def test_domain_termination_stops_repeat_without_becoming_truncation():
    environment = _environment(threshold=1.0, repeat=3, horizon=1)
    reset = environment.reset(jax.random.key(5))

    result = environment.step(reset.state, jnp.asarray([1.0]))

    assert result.evidence.accepted
    assert result.terminated
    assert not result.truncated
    assert jnp.array_equal(
        result.evidence.attempted, jnp.asarray([True, False, False])
    )
    assert result.accepted_state.step_index == 1
    assert result.accepted_state.plant_state[0] == 1.0
    assert result.reward_component_names == ("progress", "effort")
    assert jnp.array_equal(result.reward_components, jnp.asarray([1.0, -1.0]))
    assert result.total_reward == 0.0


def test_horizon_truncates_administratively_and_masks_remaining_repeats():
    environment = _environment(repeat=3, horizon=2)
    reset = environment.reset(jax.random.key(6))

    result = environment.step(reset.state, jnp.asarray([0.25]), 100.0)

    assert result.evidence.accepted
    assert not result.terminated
    assert result.truncated
    assert jnp.array_equal(
        result.evidence.attempted, jnp.asarray([True, True, False])
    )
    assert result.accepted_state.step_index == 2
    assert result.accepted_state.clock == 2.0
    assert result.accepted_state.plant_state[0] == 0.5


def test_auto_reset_keeps_terminal_observation_and_returns_reset_state():
    environment = _environment(threshold=1.0, repeat=3, auto_reset=True)
    reset = environment.reset(jax.random.key(7))

    result = environment.step(reset.state, jnp.asarray([1.0]))

    assert result.reset_performed
    assert result.terminated
    assert not result.truncated
    assert result.final_observation[0] == 1.0
    assert result.observation[0] == 0.0
    assert result.accepted_state.plant_state[0] == 1.0
    assert result.accepted_state.step_index == 1
    assert result.reset_state.plant_state[0] == 0.0
    assert result.reset_state.step_index == 0
    assert result.reset_state.clock == 0.0
    assert not jnp.array_equal(result.reset_state.key, result.accepted_state.key)


def test_fixed_shape_environment_is_deterministic_under_jit_and_vmap():
    environment = _environment(repeat=2)
    reset = environment.reset(jax.random.key(8))
    action = jnp.asarray([0.25])
    compiled_step = jax.jit(lambda state, value: environment.step(state, value, 100.0))

    eager = environment.step(reset.state, action, 100.0)
    compiled = compiled_step(reset.state, action)
    repeated = compiled_step(reset.state, action)

    assert jnp.array_equal(compiled.accepted_state.plant_state, eager.accepted_state.plant_state)
    assert jnp.array_equal(compiled.accepted_state.key, eager.accepted_state.key)
    assert jnp.array_equal(compiled.reward_components, eager.reward_components)
    assert jnp.array_equal(repeated.accepted_state.key, compiled.accepted_state.key)

    keys = jax.random.split(jax.random.key(9), 4)
    states = jax.vmap(lambda key: environment.reset(key).state)(keys)
    batched = jax.jit(
        jax.vmap(lambda state: environment.step(state, action, 100.0))
    )(states)

    assert batched.accepted_state.plant_state.shape == (4, 2)
    assert batched.accepted_state.key.shape == (4, 2)
    assert batched.reward_components.shape == (4, 2)
    assert jnp.all(batched.evidence.accepted)
