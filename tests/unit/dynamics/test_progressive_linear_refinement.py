from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel


class _IdentityModel(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 1
        self.out_size = 1

    def __call__(self, state, /, *, key=None):
        del key
        return state


def _trajectory():
    return phx.dynamics.TrajectoryData(
        jnp.asarray((0.0, 1.0, 2.0, 3.0), dtype=jnp.float32),
        jnp.ones((4, 1), dtype=jnp.float32),
        state_layout=phx.dynamics.StateLayout((1,)),
        source_id="refinement-test",
    )


def test_progressive_policy_refines_on_plateau_then_stops_without_improvement():
    policy = phx.dynamics.identification.ProgressiveLinearRefinementPolicy(
        initial_steps=1,
        step_increment=2,
        maximum_steps=7,
        evaluation_steps=9,
        smoothing=0.0,
        grace_validations=2,
        minimum_validations=3,
        plateau_relative_improvement=0.01,
        stop_relative_improvement=0.01,
    )
    state = policy.initialize()
    records = []
    for step, metric in enumerate((1.0, 1.0, 1.0), start=1):
        state, record = policy.observe(state, metric, validation_step=step)
        records.append(record)

    assert state.current_steps == 3
    assert state.refinement_count == 1
    assert records[-1].refined
    assert policy.training_control(state).maximum_steps == 3
    assert policy.evaluation_control().maximum_steps == 9

    for step, metric in enumerate((1.0, 1.0, 1.0), start=4):
        state, record = policy.observe(state, metric, validation_step=step)

    assert state.stopped
    assert state.current_steps == 3
    assert not record.refined
    assert record.stopped


def test_progressive_policy_rejects_invalid_metrics_and_unsupported_transitions():
    policy = phx.dynamics.identification.ProgressiveLinearRefinementPolicy(
        initial_steps=1,
        step_increment=1,
        maximum_steps=3,
        grace_validations=1,
        minimum_validations=2,
    )
    with pytest.raises(ValueError, match="metric"):
        policy.observe(policy.initialize(), -1.0, validation_step=0)

    data = _trajectory()
    with pytest.raises(ValueError, match="does not support linear refinement"):
        phx.dynamics.identification.fit_discrete_model(
            _IdentityModel(),
            data,
            validation=data,
            state_layout=data.state_layout,
            system_id="direct-no-refinement",
            step_size=1.0,
            transition=phx.dynamics.identification.DirectDiscreteModelRolloutTransition(
                data.state_layout,
                step_size=1.0,
            ),
            rollout_policy=phx.dynamics.identification.DiscreteModelRolloutPolicy(
                max_horizon=1
            ),
            linear_refinement=policy,
            steps=0,
        )


class _ZeroMACRateModel(AbstractArrayModel):
    scale: jnp.ndarray
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, size):
        self.scale = jnp.asarray(0.0)
        self.in_size = int(size)
        self.out_size = int(size)

    def __call__(self, state, /, *, key=None):
        del key
        return self.scale * state


def _mac_refinement_case():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    linear_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-8,
            absolute=1e-10,
            max_steps=8,
        ),
        differentiation=phx.linalg.DifferentiationPolicy("algorithmic"),
    )
    projection = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="iterative",
        tolerance=1e-8,
        maximum_iterations=8,
        linear_policy=linear_policy,
    )
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        momentum,
        projection,
    )
    layout = phx.dynamics.StateLayout(dynamics.state_shape)
    transition = phx.applications.incompressible_flow.MACLearnedRateRolloutTransition(
        dynamics,
        state_layout=layout,
        step_size=0.1,
    )
    return dynamics, layout, transition


def test_mac_transition_consumes_dynamic_krylov_control_and_full_fidelity_evaluation(
    tmp_path,
):
    dynamics, layout, transition = _mac_refinement_case()
    model = _ZeroMACRateModel(layout.size)
    state = jnp.zeros(
        layout.shape, dtype=dynamics.momentum.operators.pressure_space.dtype
    )
    context = phx.dynamics.DiscreteStepContext(
        jnp.asarray(0.0),
        jnp.asarray(0.1),
        jnp.asarray(0, dtype=jnp.int32),
    )
    result = transition.evaluate(
        model,
        context,
        state,
        None,
        key=None,
        iteration=jnp.asarray(0),
        control=phx.linalg.LinearSolveControl(maximum_steps=1),
    )

    assert bool(result.training_usable)
    assert int(result.iterations) <= 1
    assert bool(result.physically_converged)

    coordinates = jnp.asarray((0.0, 0.1, 0.2), dtype=state.dtype)
    trajectory = phx.dynamics.TrajectoryData(
        coordinates,
        jnp.zeros((3,) + layout.shape, dtype=state.dtype),
        state_layout=layout,
        source_id="mac-linear-refinement",
    )
    policy = phx.dynamics.identification.ProgressiveLinearRefinementPolicy(
        initial_steps=1,
        step_increment=1,
        maximum_steps=4,
        grace_validations=1,
        minimum_validations=2,
    )
    fitted = phx.dynamics.identification.fit_discrete_model(
        model,
        trajectory,
        validation=trajectory,
        state_layout=layout,
        system_id="mac-refined-rate",
        model_id="zero-mac-rate",
        step_size=0.1,
        transition=transition,
        rollout_policy=phx.dynamics.identification.DiscreteModelRolloutPolicy(
            max_horizon=1
        ),
        linear_refinement=policy,
        steps=0,
        checkpoint_path=tmp_path / "mac-refinement",
        shuffle=False,
    )

    assert fitted.linear_refinement_state is not None
    assert fitted.linear_refinement_state.validation_count == 1
    assert len(fitted.linear_refinement_records) == 1
    resumed = phx.dynamics.identification.fit_discrete_model(
        model,
        trajectory,
        validation=trajectory,
        state_layout=layout,
        system_id="mac-refined-rate",
        model_id="zero-mac-rate",
        step_size=0.1,
        transition=transition,
        rollout_policy=phx.dynamics.identification.DiscreteModelRolloutPolicy(
            max_horizon=1
        ),
        linear_refinement=policy,
        steps=0,
        shuffle=False,
        checkpoint_path=tmp_path / "mac-refinement",
        resume=True,
    )
    assert resumed.linear_refinement_state == fitted.linear_refinement_state
    assert resumed.linear_refinement_records == fitted.linear_refinement_records
