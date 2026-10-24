"""Control must evolve discrete actuator history, not a frozen codec sidecar."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._array_tree import ArrayPyTreeSchema
from phydrax._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from phydrax.control import (
    ControlProblem,
    PiecewiseConstantControlParameterization,
    PlantControlDynamics,
    PlantHistoryRetention,
)
from phydrax.dynamics import StateLayout, TimeGrid
from phydrax.dynamics._plant import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantRuntimeState,
    PlantStepContext,
)
from phydrax.dynamics._plant_codec import ControlVectorCodec, PlantStateVectorCodec
from phydrax.linalg import ArraySpace


class _HistoryPlant(AbstractDiscretePlant):
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: dict[str, Array]
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    parameters: PlantParameters
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)

    def __init__(self, case_ndim: int = 0) -> None:
        fallback = {
            "position": jnp.zeros((1,), dtype=jnp.float64),
            "memory": jnp.zeros((1,), dtype=jnp.float64),
            "mode": jnp.asarray(False, dtype=jnp.bool_),
        }
        template = jax.tree.map(lambda x: x.reshape((1,) * case_ndim + x.shape), fallback)
        self.state_schema = ArrayPyTreeSchema.from_tree(template, case_ndim=case_ndim)
        self.control_schema = ArrayPyTreeSchema.from_tree(
            jnp.zeros((1,) * case_ndim + (1,), dtype=jnp.float64), case_ndim=case_ndim
        )
        parameter_values = {"gain": jnp.asarray(1.0, dtype=jnp.float64)}
        self.parameter_schema = ArrayPyTreeSchema.from_tree(parameter_values, case_ndim=0)
        self.reset_fallback = fallback
        self.semantic_provenance = SemanticProvenance(
            {"kind": "alternating-actuator-memory"}
        )
        self.numeric_revision = NumericRevision(
            self.semantic_provenance, parameter_values
        )
        self.execution_signature = ExecutableSignature(
            shapes={"state": (2,), "control": (1,)},
            dtypes={"state": np.float64, "mode": np.bool_},
            algorithm_facts={"case_ndim": case_ndim},
        )
        self.parameters = PlantParameters(
            parameter_values, self.parameter_schema.schema_id, self.numeric_revision
        )
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True

    def propose_reset(
        self,
        keys: Array,
        parameters: dict[str, Array],
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        del keys, parameters, initial_time
        payload = jax.tree.map(
            lambda value: jnp.broadcast_to(value, case_shape + value.shape),
            self.reset_fallback,
        )
        valid = jnp.ones(case_shape, dtype=jnp.bool_)
        status = jnp.zeros(case_shape, dtype=jnp.int32)
        return PlantProposal(payload, payload, valid, valid, status, status, status)

    def propose_step(
        self,
        context: PlantStepContext,
        source: dict[str, Array],
        commands: Array | None,
        parameters: dict[str, Array],
        keys: Array,
        /,
    ) -> PlantProposal:
        del keys
        if commands is None:
            raise TypeError("Actuator command is required.")
        candidate = {
            "position": source["position"]
            + context.duration[..., None]
            * (parameters["gain"] * commands + source["memory"]),
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
            status,
        )


def _setup(
    *,
    cases: tuple[int, ...] = (),
    retention: PlantHistoryRetention = "full",
    budget: int = 1_000_000,
) -> tuple[
    _HistoryPlant,
    PlantControlDynamics,
    PlantRuntimeState,
    TimeGrid,
    PiecewiseConstantControlParameterization,
]:
    plant = _HistoryPlant(len(cases))
    schema = ArrayPyTreeSchema.from_tree(plant.reset_fallback, case_ndim=0)
    layout = StateLayout(
        (2,),
        local_space=ArraySpace((2,), dtype=np.float64),
        tangent_space=ArraySpace((2,), dtype=np.float64),
    )
    state_codec = PlantStateVectorCodec(
        schema,
        layout,
        plant.reset_fallback,
        semantic_provenance=plant.semantic_provenance,
        numeric_revision=plant.numeric_revision,
        executable_signature=plant.execution_signature,
    )
    control_codec = ControlVectorCodec(
        ArrayPyTreeSchema.from_tree(jnp.zeros((1,), dtype=jnp.float64), case_ndim=0),
        semantic_provenance=plant.semantic_provenance,
        numeric_revision=plant.numeric_revision,
        executable_signature=plant.execution_signature,
    )
    dynamics = PlantControlDynamics(
        plant,
        state_codec,
        control_codec,
        parameters=plant.parameters,
        retention=retention,
        maximum_retained_bytes=budget,
    )
    key = jax.random.key(51)
    keys = jax.random.split(key, cases) if cases else key
    initial = plant.reset(
        keys,
        plant.parameters,
        case_shape=cases,
        initial_time=jnp.zeros(cases, dtype=jnp.float64),
    ).accepted_state
    grid = TimeGrid(jnp.arange(4, dtype=jnp.float64), time_id="history-control-grid")
    parameterization = PiecewiseConstantControlParameterization(
        grid, (1,), parameterization_id="history-controls"
    )
    return plant, dynamics, initial, grid, parameterization


def _equal_runtime(actual: PlantRuntimeState, expected: PlantRuntimeState) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(left, right)


def test_rollout_evolves_modes_and_retains_complete_replay() -> None:
    plant, dynamics, initial, grid, parameterization = _setup()
    controls = jnp.ones((3, 1), dtype=jnp.float64)
    problem = ControlProblem(dynamics, grid, initial, problem_id="history-control")
    trajectory = problem.rollout(parameterization, controls)
    assert bool(trajectory.successful)
    np.testing.assert_array_equal(trajectory.states[:, 1], [0.0, 1.0, 3.0, 3.0])
    history = trajectory.plant_history
    assert history is not None
    expected = initial
    for index in range(3):
        expected = plant.step(
            PlantStepContext(expected.time, grid.times[index + 1], expected.step_index),
            expected,
            controls[index],
            plant.parameters,
        ).accepted_state
        _equal_runtime(history.state_at(index + 1), expected)
    _equal_runtime(history.final_state, expected)
    restored = plant.restore(plant.checkpoint(history.state_at(2)))
    continued = plant.step(
        PlantStepContext(restored.time, grid.times[3], restored.step_index),
        restored,
        controls[2],
        plant.parameters,
    ).accepted_state
    _equal_runtime(continued, history.final_state)


def test_same_continuous_initial_view_keeps_distinct_modes() -> None:
    _, dynamics, initial, grid, parameterization = _setup()
    other = eqx.tree_at(lambda state: state.payload["mode"], initial, jnp.asarray(True))
    np.testing.assert_array_equal(
        dynamics.state_vector(initial), dynamics.state_vector(other)
    )
    controls = jnp.ones((3, 1), dtype=jnp.float64)
    first = ControlProblem(dynamics, grid, initial, problem_id="mode-a").rollout(
        parameterization, controls
    )
    second = ControlProblem(dynamics, grid, other, problem_id="mode-b").rollout(
        parameterization, controls
    )
    np.testing.assert_array_equal(first.states[:, 1], [0.0, 1.0, 3.0, 3.0])
    np.testing.assert_array_equal(second.states[:, 1], [0.0, 1.0, 1.0, 3.0])


def test_batched_failure_preserves_complete_clock_key_and_history() -> None:
    plant, dynamics, initial, grid, parameterization = _setup(cases=(2,))
    controls = jnp.asarray(
        [[[1.0], [-3.0], [1.0]], [[1.0], [1.0], [1.0]]], dtype=jnp.float64
    )
    result = ControlProblem(
        dynamics, grid, initial, problem_id="casewise-history"
    ).rollout(parameterization, controls)
    history = result.plant_history
    assert history is not None
    np.testing.assert_array_equal(
        result.valid, [[True, True, False, False], [True, True, True, True]]
    )
    np.testing.assert_array_equal(result.backend_status, [37, 0])
    np.testing.assert_array_equal(history.final_state.time, [1.0, 3.0])
    np.testing.assert_array_equal(history.final_state.step_index, [1, 3])
    before = history.state_at(1)
    for before_leaf, after_leaf in zip(
        jax.tree.leaves(before), jax.tree.leaves(history.final_state), strict=True
    ):
        if jax.dtypes.issubdtype(before_leaf.dtype, jax.dtypes.prng_key):
            before_leaf, after_leaf = (
                jax.random.key_data(before_leaf),
                jax.random.key_data(after_leaf),
            )
        np.testing.assert_array_equal(before_leaf[0], after_leaf[0])
    assert plant.state_schema.validate(history.final_state.payload) == (2,)


def test_endpoint_retention_does_not_claim_intermediate_state() -> None:
    _, dynamics, initial, grid, parameterization = _setup(retention="endpoint")
    result = ControlProblem(
        dynamics, grid, initial, problem_id="endpoint-history"
    ).rollout(parameterization, jnp.ones((3, 1), dtype=jnp.float64))
    assert result.plant_history is not None
    np.testing.assert_array_equal(
        result.plant_history.final_state.payload["position"], [3.0]
    )
    with pytest.raises(ValueError, match="Endpoint-only"):
        result.plant_history.state_at(1)


def test_history_budget_refuses_before_rollout() -> None:
    _, dynamics, initial, grid, parameterization = _setup(budget=0)
    with pytest.raises(ValueError, match="retention requires"):
        dynamics.rollout(
            grid,
            initial,
            parameterization,
            jnp.ones((3, 1), dtype=jnp.float64),
            problem_id="refused-history",
        )


def test_clock_precision_and_foreign_state_are_refused() -> None:
    _, dynamics, initial, grid, parameterization = _setup()
    foreign = eqx.tree_at(
        lambda state: state.time, initial, jnp.asarray(0.0, dtype=jnp.float32)
    )
    with pytest.raises(TypeError, match="precision"):
        dynamics.rollout(
            grid,
            foreign,
            parameterization,
            jnp.ones((3, 1), dtype=jnp.float64),
            problem_id="clock",
        )
    shifted = eqx.tree_at(
        lambda state: state.time, initial, jnp.asarray(0.5, dtype=jnp.float64)
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="clock"):
        dynamics.rollout(
            grid,
            shifted,
            parameterization,
            jnp.ones((3, 1), dtype=jnp.float64),
            problem_id="clock",
        )


def test_generic_control_argument_is_not_silently_discarded() -> None:
    plant, dynamics, initial, grid, parameterization = _setup()
    changed = eqx.tree_at(
        lambda params: params.values["gain"],
        plant.parameters,
        jnp.asarray(2.0, dtype=jnp.float64),
    )
    result = ControlProblem(
        dynamics, grid, initial, args=changed, problem_id="dynamic-gain"
    ).rollout(parameterization, jnp.ones((3, 1), dtype=jnp.float64))
    np.testing.assert_array_equal(result.states[:, 1], [0.0, 2.0, 5.0, 6.0])


def _terminal_position_cost(time: Array, state: Array, args: object) -> Array:
    del time, args
    return jnp.square(state[1])


def test_sampling_mpc_selected_replay_keeps_complete_plant_state() -> None:
    from phydrax.control import (
        initialize_sampling_mpc,
        plan_sampling_mpc,
        solve_sampling_mpc,
    )

    plant, dynamics, initial, grid, parameterization = _setup()
    problem = ControlProblem(
        dynamics,
        grid,
        initial,
        terminal_cost=_terminal_position_cost,
        problem_id="complete-sampling-control",
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=4,
        iteration_count=1,
        elite_count=2,
    )
    proposal = initialize_sampling_mpc(
        plan,
        jnp.ones((3, 1), dtype=jnp.float64),
        jnp.asarray(0.2, dtype=jnp.float64),
    )
    result = solve_sampling_mpc(plan, proposal, jax.random.key(9))
    assert bool(result.successful)
    history = result.replay.plant_history
    assert history is not None
    expected = initial
    for index in range(grid.num_steps):
        expected = plant.step(
            PlantStepContext(expected.time, grid.times[index + 1], expected.step_index),
            expected,
            result.controls[index],
            plant.parameters,
        ).accepted_state
        _equal_runtime(history.state_at(index + 1), expected)
    _equal_runtime(history.final_state, expected)


def test_array_state_optimizer_refuses_complete_plant_without_mode_contract() -> None:
    from phydrax.control import plan_ilqr

    _, dynamics, initial, grid, _ = _setup()
    problem = ControlProblem(
        dynamics,
        grid,
        initial,
        terminal_cost=_terminal_position_cost,
        problem_id="mode-admission",
    )
    with pytest.raises(ValueError):
        plan_ilqr(problem)


def test_batched_control_codec_is_refused_before_rollout() -> None:
    plant, dynamics, _, _, _ = _setup(cases=(2,))
    batched_codec = ControlVectorCodec(
        plant.control_schema,
        semantic_provenance=plant.semantic_provenance,
        numeric_revision=plant.numeric_revision,
        executable_signature=plant.execution_signature,
    )
    with pytest.raises(ValueError, match="unbatched"):
        PlantControlDynamics(
            plant,
            dynamics.state_codec,
            batched_codec,
            parameters=plant.parameters,
        )


def _bind_realization_parameters(
    base: PlantParameters, realization: dict[str, Array]
) -> PlantParameters:
    return eqx.tree_at(lambda value: value.values["gain"], base, realization["gain"])


@pytest.mark.parametrize("retention", ["full", "endpoint"])
def test_realization_replay_restores_each_complete_plant_trajectory(
    retention: PlantHistoryRetention,
) -> None:
    from phydrax.control import plan_sampling_mpc, SamplingMPCRealizations

    plant, dynamics, initial, grid, parameterization = _setup(retention=retention)
    problem = ControlProblem(
        dynamics,
        grid,
        initial,
        terminal_cost=_terminal_position_cost,
        args=plant.parameters,
        problem_id="complete-realization-replay",
    )
    gains = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    realizations = SamplingMPCRealizations(
        {"gain": gains},
        ("unit-gain", "double-gain"),
        posterior_id="actuator-gain-posterior",
        campaign_id="complete-history-replay",
    )
    plan = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=1,
        iteration_count=1,
        update="predictive",
        realizations=realizations,
        realization_binding=_bind_realization_parameters,
        realization_binding_id="complete-plant-gain-binding",
    )
    controls = jnp.ones((grid.num_steps, 1), dtype=jnp.float64)
    result = plan.solve(plan.initialize(controls, 0.0), jax.random.key(19))
    assert bool(result.successful)
    history = result.replay.plant_history
    assert history is not None
    if retention == "full":
        assert history.accepted is not None
        assert history.accepted.node_count == grid.num_times
    with pytest.raises(ValueError, match="realization"):
        history.state_at(1)
    for realization in range(realizations.count):
        selected = history.select_realization(realization)
        parameters = _bind_realization_parameters(
            plant.parameters, {"gain": gains[realization]}
        )
        expected = initial
        for step in range(grid.num_steps):
            expected = plant.step(
                PlantStepContext(
                    expected.time, grid.times[step + 1], expected.step_index
                ),
                expected,
                controls[step],
                parameters,
            ).accepted_state
            if retention == "full":
                _equal_runtime(selected.state_at(step + 1), expected)
        _equal_runtime(selected.final_state, expected)
        if retention == "full":
            restored = plant.restore(plant.checkpoint(selected.state_at(2)))
            continued = plant.step(
                PlantStepContext(restored.time, grid.times[3], restored.step_index),
                restored,
                controls[2],
                parameters,
            ).accepted_state
            _equal_runtime(continued, selected.final_state)
        else:
            assert selected.accepted is None and selected.candidates is None
            with pytest.raises(ValueError, match="Endpoint-only"):
                selected.state_at(1)
    with pytest.raises(eqx.EquinoxRuntimeError, match="Realization index"):
        history.select_realization(realizations.count)


def test_realization_replay_budget_covers_all_retained_models() -> None:
    from phydrax.control import plan_sampling_mpc, SamplingMPCRealizations

    plant, dynamics, initial, grid, parameterization = _setup()
    one_model_bytes = dynamics.retained_storage_bytes(initial, grid.num_steps)
    limited = PlantControlDynamics(
        plant,
        dynamics.state_codec,
        dynamics.control_codec,
        parameters=plant.parameters,
        retention="full",
        maximum_retained_bytes=2 * one_model_bytes - 1,
    )
    problem = ControlProblem(
        limited,
        grid,
        initial,
        args=plant.parameters,
        terminal_cost=_terminal_position_cost,
        problem_id="bounded-realization-replay",
    )
    realizations = SamplingMPCRealizations(
        {"gain": jnp.asarray([1.0, 2.0], dtype=jnp.float64)},
        ("unit-gain", "double-gain"),
        posterior_id="bounded-actuator-gain",
        campaign_id="bounded-plant-history",
    )
    with pytest.raises(ValueError, match="realization replay retention"):
        plan_sampling_mpc(
            problem,
            parameterization,
            candidate_count=1,
            iteration_count=1,
            update="predictive",
            realizations=realizations,
            realization_binding=_bind_realization_parameters,
            realization_binding_id="bounded-gain-binding",
        )
