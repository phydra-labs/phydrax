import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

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
        accepted_plant_state,
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
            jnp.asarray([accepted_plant_state[0], candidate_task_state[0]]),
            jnp.asarray(
                [
                    accepted_plant_state[0] - source_plant_state[0],
                    -(action[0] * action[0]),
                ]
            ),
            accepted_plant_state[0] >= self.threshold,
            accepted_plant_state[:1],
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
    record_nonfinite: bool = eqx.field(static=True)
    wrapper_id: str = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    horizon: int | None = eqx.field(static=True)
    auto_reset: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        action_repeat=1,
        record_nonfinite=False,
        horizon=None,
        auto_reset=False,
    ):
        self.record_nonfinite = record_nonfinite
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
        value = jnp.where(
            self.record_nonfinite,
            jnp.asarray(jnp.nan, dtype=plant_state.dtype),
            plant_state[0],
        )
        return RobotEnvironmentWrapperTransition(
            jnp.reshape(value, (1,)),
            jnp.asarray(False),
        )


class _NonfiniteTask(AbstractRobotTask):
    task_id: str = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    reward_component_names: tuple[str, ...] = eqx.field(static=True)
    descriptor_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self):
        self.task_id = "nonfinite-task"
        self.observation_shape = (2,)
        self.reward_component_names = ("progress", "effort")
        self.descriptor_shape = (1,)

    def initialize(self, plant_state, key, /):
        del plant_state, key
        return jnp.zeros((1,))

    def evaluate(self, plant_state, task_state, /):
        return RobotTaskEvaluation(
            jnp.asarray([plant_state[0], task_state[0]]),
            jnp.asarray(False),
            plant_state[:1],
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
        del context, source_plant_state, action, key
        return RobotTaskTransition(
            task_state.at[0].set(jnp.nan),
            jnp.asarray([accepted_plant_state[0], jnp.nan]),
            jnp.asarray([jnp.nan, 0.0]),
            jnp.asarray(False),
            jnp.asarray([jnp.nan]),
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


def _ignoring_action_transition(context, state, action, args):
    del context, action, args
    return state.at[0].add(0.25)


def _nonfinite_legacy_transition(context, state, action, args):
    del context, action, args
    return state.at[0].set(jnp.nan)


def _nonfinite_accepted_transition(context, state, action, args):
    del context, action, args
    return DiscreteTransitionResult(
        state.at[0].add(1.0),
        state.at[0].set(jnp.nan),
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
    task=None,
    wrappers=None,
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
    task_value = _ThresholdTask(threshold) if task is None else task
    wrapper_values = (
        (
            _EpisodeWrapper(
                action_repeat=repeat,
                horizon=horizon,
                auto_reset=auto_reset,
            ),
        )
        if wrappers is None
        else tuple(wrappers)
    )
    return PreparedRobotEnvironment(
        system,
        _initial_state,
        task_value,
        wrapper_values,
        initializer_id=initializer_id,
        environment_id=environment_id,
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
    assert result.candidate_observation[0] == 1.0
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


def test_mechanics_accepted_state_drives_task_wrappers_and_repetition():
    environment = _environment(
        transition=_projected_transition,
        wrappers=(_PlantRecordingWrapper(action_repeat=2),),
    )
    reset = environment.reset(jax.random.key(10))

    result = environment.step(reset.state, jnp.asarray([1.0]))

    assert result.accepted
    assert result.accepted_state.plant_state[0] == 2.0
    assert result.candidate_state.plant_state[0] == 11.0
    assert result.accepted_state.task_state[0] == 2.0
    assert result.accepted_state.wrapper_states[0][0] == 2.0
    assert result.candidate_observation[0] == 2.0
    assert result.final_observation[0] == 2.0
    assert jnp.array_equal(
        result.reward_components,
        jnp.asarray([2.0, -2.0]),
    )


def test_nonfinite_controls_and_mechanics_outputs_fail_closed():
    cases = (
        (
            _environment(transition=_ignoring_action_transition),
            jnp.asarray([jnp.nan]),
        ),
        (
            _environment(transition=_nonfinite_legacy_transition),
            jnp.asarray([0.25]),
        ),
        (
            _environment(transition=_nonfinite_accepted_transition),
            jnp.asarray([0.25]),
        ),
    )

    for index, (environment, action) in enumerate(cases):
        reset = environment.reset(jax.random.key(20 + index))
        result = environment.step(reset.state, action)

        assert not result.accepted
        assert result.evidence.rollback_applied
        assert result.evidence.mechanics_successful
        assert result.evidence.mechanics_status == 0
        for accepted_leaf, source_leaf in zip(
            jax.tree.leaves(result.accepted_state),
            jax.tree.leaves(reset.state),
            strict=True,
        ):
            assert jnp.array_equal(accepted_leaf, source_leaf)


def test_nonfinite_task_and_wrapper_outputs_roll_back_atomically():
    environment = _environment(
        transition=_projected_transition,
        task=_NonfiniteTask(),
        wrappers=(
            _PlantRecordingWrapper(
                record_nonfinite=True,
            ),
        ),
    )
    reset = environment.reset(jax.random.key(30))

    result = environment.step(reset.state, jnp.asarray([0.25]))

    assert not result.accepted
    assert result.evidence.rollback_applied
    assert result.evidence.mechanics_successful
    assert jnp.isnan(result.candidate_state.task_state[0])
    assert jnp.isnan(result.candidate_state.wrapper_states[0][0])
    assert jnp.isnan(result.candidate_observation[1])
    assert jnp.array_equal(
        result.accepted_state.plant_state,
        reset.state.plant_state,
    )
    assert jnp.array_equal(
        result.accepted_state.key,
        reset.state.key,
    )
    assert result.accepted_state.clock == reset.state.clock
    assert result.accepted_state.step_index == reset.state.step_index
    assert jnp.array_equal(
        result.accepted_state.task_state,
        reset.state.task_state,
    )
    assert jnp.array_equal(
        result.accepted_state.wrapper_states[0],
        reset.state.wrapper_states[0],
    )
    assert jnp.array_equal(result.reward_components, jnp.zeros((2,)))
    assert jnp.array_equal(result.descriptor, reset.descriptor)


def test_environment_provenance_rejects_equal_display_id_semantic_collisions():
    display_id = "shared-display-id"
    baseline = _environment(
        threshold=1.0,
        repeat=2,
        horizon=5,
        environment_id=display_id,
    )
    semantic_variants = (
        _environment(
            threshold=2.0,
            repeat=2,
            horizon=5,
            environment_id=display_id,
        ),
        _environment(
            threshold=1.0,
            repeat=3,
            horizon=5,
            environment_id=display_id,
        ),
        _environment(
            threshold=1.0,
            repeat=2,
            horizon=6,
            environment_id=display_id,
        ),
        _environment(
            threshold=1.0,
            repeat=2,
            horizon=5,
            auto_reset=True,
            environment_id=display_id,
        ),
        _environment(
            threshold=1.0,
            repeat=2,
            horizon=5,
            initializer_id="different-initializer",
            environment_id=display_id,
        ),
        _environment(
            threshold=1.0,
            repeat=2,
            horizon=5,
            transition=_projected_transition,
            environment_id=display_id,
        ),
    )
    reset = baseline.reset(jax.random.key(40))

    assert reset.environment_id == display_id
    assert reset.provenance_id == baseline.provenance_id
    assert reset.state.environment_id == display_id
    assert reset.state.provenance_id == baseline.provenance_id
    assert all(
        variant.environment_id == display_id for variant in semantic_variants
    )
    assert all(
        variant.provenance_id != baseline.provenance_id
        for variant in semantic_variants
    )
    with pytest.raises(ValueError, match="provenance"):
        semantic_variants[0].step(reset.state, jnp.asarray([0.25]), 100.0)


def test_provenance_covers_full_layout_configuration_not_only_layout_ids():
    state_layout_id = "shared-state-layout"
    input_layout_id = "shared-input-layout"
    baseline = _environment(
        state_layout=StateLayout(
            (2,),
            component_names=("position", "velocity"),
            layout_id=state_layout_id,
        ),
        input_layout=InputLayout(
            (1,),
            component_names=("torque",),
            layout_id=input_layout_id,
        ),
        environment_id="shared-layout-environment",
    )
    changed_state_layout = _environment(
        state_layout=StateLayout(
            (2,),
            component_names=("coordinate", "rate"),
            layout_id=state_layout_id,
        ),
        input_layout=InputLayout(
            (1,),
            component_names=("torque",),
            layout_id=input_layout_id,
        ),
        environment_id="shared-layout-environment",
    )
    changed_input_layout = _environment(
        state_layout=StateLayout(
            (2,),
            component_names=("position", "velocity"),
            layout_id=state_layout_id,
        ),
        input_layout=InputLayout(
            (1,),
            component_names=("force",),
            layout_id=input_layout_id,
        ),
        environment_id="shared-layout-environment",
    )

    assert changed_state_layout.provenance_id != baseline.provenance_id
    assert changed_input_layout.provenance_id != baseline.provenance_id


def test_replay_from_accepted_checkpoint_is_deterministic():
    environment = _environment(repeat=2)
    reset = environment.reset(jax.random.key(50))
    checkpoint = environment.step(
        reset.state,
        jnp.asarray([0.25]),
        100.0,
    ).accepted_state

    first = environment.step(checkpoint, jnp.asarray([0.125]), 100.0)
    replay = environment.step(checkpoint, jnp.asarray([0.125]), 100.0)

    for first_leaf, replay_leaf in zip(
        jax.tree.leaves(first),
        jax.tree.leaves(replay),
        strict=True,
    ):
        assert jnp.array_equal(first_leaf, replay_leaf)
