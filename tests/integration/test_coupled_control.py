#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Control of a coupled finite-element/virtual-element transient.

The plant is the FE-VEM heat transient of ``examples/_coupled_heat_transient.py``
with the right-wall heat flux bound as the P6 control parameter and three
temperature sensors as P6 observation bindings. ``prepare_coupled_transition``
exposes one native implicit-Euler step through the ``DiscreteSystem`` ABI.

Independent references are host NumPy/SciPy: the implicit-Euler step operator of
the semi-discrete system ``C x' + K x = b + B g`` and the sensor rows ``H``
assembled on the host from the two meshes (``host_semidiscrete``: P1 and
degree-1 VEM operators weighted by each owner's ``rho c``; only the verified
native coordinate chart is read from the prepared transient), and the exact
bound-constrained least-squares optimum (BVLS) of the finite-horizon
sensor-tracking problem.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from examples import _coupled_heat_transient as heat, coupled_control as ex
from phydrax.solver import coupling as cpl


@dataclass(frozen=True)
class _Plant:
    transient: cpl.PreparedCoupledTransient
    dynamics: phx.control.DiscreteControlDynamics
    port: cpl.CoupledObservationPort
    grid: phx.dynamics.TimeGrid
    problem: phx.control.ControlProblem
    policy: phx.control.PiecewiseConstantControlParameterization
    reference: heat.HostSemidiscrete
    step: tuple[np.ndarray, np.ndarray, np.ndarray]


@pytest.fixture(scope="module")
def plant() -> _Plant:
    transient, dynamics, port = ex.plant(ex.CELLS)
    size = dynamics.state_shape[0]
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, ex.HORIZON * ex.STEP, ex.HORIZON + 1), time_id="control-grid"
    )
    problem = phx.control.ControlProblem(
        dynamics, grid, jnp.zeros((size,)), problem_id="coupled-heat-control"
    )
    policy = phx.control.PiecewiseConstantControlParameterization(
        grid, (2,), parameterization_id="held-wall-flux"
    )
    reference = heat.host_semidiscrete(transient, ex.CELLS)
    return _Plant(
        transient,
        dynamics,
        port,
        grid,
        problem,
        policy,
        reference,
        ex.host_step(reference, ex.STEP),
    )


def test_coupled_rollout_is_the_host_implicit_euler_recursion(plant: _Plant) -> None:
    fluxes = np.random.default_rng(11).uniform(-1.0, 1.0, (ex.HORIZON, 2))
    trajectory = plant.problem.rollout(plant.policy, jnp.asarray(fluxes))
    assert int(trajectory.status) == phx.control.CONTROL_SUCCESS
    evidence = trajectory.transition_evidence
    assert evidence is not None
    assert bool(jnp.all(evidence.successful))
    np.testing.assert_array_equal(np.asarray(evidence.status), 0)
    matrix, control, bias = plant.step
    state = np.zeros(matrix.shape[0])
    expected = [state]
    for stage in range(ex.HORIZON):
        state = matrix @ state + control @ fluxes[stage] + bias
        expected.append(state)
    np.testing.assert_allclose(
        np.asarray(trajectory.states), np.stack(expected), rtol=0.0, atol=1e-11
    )


def test_matrix_free_linearization_follows_the_step_context(plant: _Plant) -> None:
    size = plant.dynamics.state_shape[0]
    rng = np.random.default_rng(5)
    state = jnp.asarray(0.2 * rng.standard_normal(size))
    flux = jnp.asarray([0.3, -0.4])
    direction = rng.standard_normal(size + 2)
    for span in (1, 2):
        linearization = phx.control.prepare_control_linearization(
            plant.dynamics,
            plant.grid.times[3],
            state,
            flux,
            target_time=plant.grid.times[3 + span],
            step_index=3,
            output=plant.port,
        )
        assert bool(linearization.valid)
        matrix, control, bias = ex.host_step(plant.reference, span * ex.STEP)
        np.testing.assert_allclose(
            np.asarray(linearization.dynamics_value),
            matrix @ np.asarray(state) + control @ np.asarray(flux) + bias,
            rtol=0.0,
            atol=1e-11,
        )
        expected = matrix @ direction[:size] + control @ direction[size:]
        action = np.asarray(linearization.dynamics_jacobian.mv(jnp.asarray(direction)))
        np.testing.assert_allclose(action, expected, rtol=0.0, atol=1e-10)
    with pytest.raises(ValueError, match="requires target_time and step_index"):
        phx.control.prepare_control_linearization(
            plant.dynamics, plant.grid.times[0], state, flux
        )


def test_dense_mpc_matches_the_host_bounded_optimum_and_replays_on_the_coupled_plant(
    plant: _Plant,
) -> None:
    output = ex.sensor_matrix(plant.dynamics, plant.port, plant.grid)
    # The port's linear sensor map is the host point evaluation of the owners.
    np.testing.assert_allclose(output, plant.reference.sensors, rtol=0.0, atol=1e-14)
    specification = ex.tracking_problem(plant.dynamics, plant.grid, output)
    result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=ex.HORIZON,
        terminal_policy="global",
        policy=ex.QP_POLICY,
    )
    assert bool(jnp.all(result.successful))
    size = plant.dynamics.state_shape[0]
    optimal = ex.host_optimal_controls(
        plant.step, plant.reference.sensors, np.zeros(size)
    )
    assert np.any(np.isclose(np.abs(optimal), ex.FLUX_BOUND, atol=1e-9))
    assert np.any(np.abs(optimal) < ex.FLUX_BOUND - 1e-3)
    np.testing.assert_allclose(np.asarray(result.controls), optimal, rtol=0.0, atol=1e-6)
    replay = plant.problem.rollout(result.policy, result.parameters)
    assert bool(jnp.all(replay.valid))
    np.testing.assert_allclose(
        np.asarray(replay.states), np.asarray(result.states), rtol=0.0, atol=1e-10
    )


def test_mpc_sensitivity_refuses_a_weakly_active_flux_bound(plant: _Plant) -> None:
    output = ex.sensor_matrix(plant.dynamics, plant.port, plant.grid)
    tracking = ex.tracking_problem(plant.dynamics, plant.grid, output)
    size = plant.dynamics.state_shape[0]
    unconstrained = ex.host_optimal_controls(
        plant.step, plant.reference.sensors, np.zeros(size), bound=np.inf
    )
    # The first flux sits exactly on its bound: active with a zero multiplier.
    upper = np.full((ex.HORIZON, 2), 10.0)
    upper[0, 0] = unconstrained[0, 0]
    tied = phx.control.LinearQuadraticControlProblem(
        tracking.dynamics_matrices,
        tracking.control_matrices,
        tracking.initial_state,
        tracking.state_costs,
        tracking.control_costs,
        tracking.terminal_state_cost,
        dynamics_bias=tracking.dynamics_bias,
        state_linear=tracking.state_linear,
        terminal_linear=tracking.terminal_linear,
        control_lower_bounds=jnp.full((ex.HORIZON, 2), -10.0),
        control_upper_bounds=jnp.asarray(upper),
        time_grid=tracking.time_grid,
        problem_id="tied-flux-bound",
    )
    controller = phx.control.RecedingHorizonMPC(
        tied,
        prediction_horizon=ex.HORIZON,
        terminal_policy="global",
        policy=ex.QP_POLICY,
    )
    sensitivity = phx.control.prepare_receding_horizon_mpc_sensitivity(controller)
    assert sensitivity.refusal is not None
    assert "nonregular" in sensitivity.refusal
    assert not bool(sensitivity.regular)
    with pytest.raises(ValueError, match="MPC sensitivity is refused"):
        sensitivity.jvp(tied)


def test_failed_transition_is_never_repaired(plant: _Plant) -> None:
    starved = phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-30,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=1,
    )
    policy = heat.implicit_euler_policy()
    failing = cpl.prepare_coupled_transition(
        plant.transient,
        step=ex.STEP,
        policy=phx.solver.DAESolvePolicy(
            method=policy.method,
            nonlinear_method=policy.nonlinear_method,
            initialization_method=policy.initialization_method,
            nonlinear_termination=starved,
            initialization_termination=heat.stage_termination(),
        ),
        parameters={heat.CONTROL: jnp.zeros((2,))},
        control=heat.CONTROL,
    )
    dynamics = phx.control.DiscreteControlDynamics(failing.discrete_system())
    size = dynamics.state_shape[0]
    source = np.random.default_rng(17).uniform(0.1, 0.5, size)
    problem = phx.control.ControlProblem(
        dynamics, plant.grid, jnp.asarray(source), problem_id="starved-plant"
    )
    trajectory = problem.rollout(plant.policy, jnp.zeros((ex.HORIZON, 2)))
    assert int(trajectory.status) == phx.control.CONTROL_DYNAMICS_FAILED
    evidence = trajectory.transition_evidence
    assert evidence is not None
    assert not bool(evidence.successful[0])
    assert cpl.coupled_transition_status_name(int(evidence.status[0])) == "native-failure"
    # The failed step rolls back to its (nonzero) source state exactly and keeps
    # the rejected candidate as evidence instead of overwriting it.
    np.testing.assert_array_equal(np.asarray(evidence.accepted_states[0]), source)
    candidate = np.asarray(evidence.candidate_states[0])
    assert np.all(np.isfinite(candidate))
    assert np.max(np.abs(candidate - source)) > 1e-3
    with pytest.raises(
        ValueError, match="failed accepted-state transitions are not repaired"
    ):
        phx.control.linear_quadratic_problem_from_discrete_dynamics(
            dynamics,
            plant.grid,
            jnp.zeros((ex.HORIZON, size)),
            jnp.zeros((ex.HORIZON, 2)),
            jnp.zeros((size,)),
            jnp.broadcast_to(jnp.eye(size), (ex.HORIZON, size, size)),
            jnp.broadcast_to(jnp.eye(2), (ex.HORIZON, 2, 2)),
            jnp.eye(size),
            materialization=ex.BUDGET,
            problem_id="starved-lq",
        )


def test_refined_full_order_plant_exceeds_the_dense_budget() -> None:
    _, dynamics, _ = ex.plant(ex.REFINED_CELLS)
    size = dynamics.state_shape[0]
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, ex.HORIZON * ex.STEP, ex.HORIZON + 1), time_id="refined"
    )
    with pytest.raises(ValueError, match="materializ"):
        phx.control.linear_quadratic_problem_from_discrete_dynamics(
            dynamics,
            grid,
            jnp.zeros((ex.HORIZON, size)),
            jnp.zeros((ex.HORIZON, 2)),
            jnp.zeros((size,)),
            jnp.broadcast_to(jnp.eye(size), (ex.HORIZON, size, size)),
            jnp.broadcast_to(jnp.eye(2), (ex.HORIZON, 2, 2)),
            jnp.eye(size),
            materialization=ex.BUDGET,
            problem_id="refined-lq",
        )


def test_control_must_enter_the_coupled_rows(plant: _Plant) -> None:
    with pytest.raises(ValueError, match="not a refresh parameter binding"):
        cpl.prepare_coupled_transition(
            plant.transient,
            step=ex.STEP,
            policy=heat.implicit_euler_policy(),
            parameters={heat.CONTROL: jnp.zeros((2,))},
            control="conductivity",
        )


def _ramped_transient(gain: float) -> cpl.PreparedCoupledTransient:
    """Wall flux ``gain * t * u``: the control enters only away from ``t = 0``."""
    prepared = heat.coupled_heat_problem(ex.CELLS)

    def arguments(time: jax.Array, parameters: object) -> Mapping[str, object]:
        if not isinstance(parameters, Mapping):
            raise TypeError("Transient parameters are the bound parameter values.")
        control = jnp.asarray(parameters[heat.CONTROL])
        ramped = {**parameters, heat.CONTROL: gain * time * control}
        return prepared.bind_arguments(parameters=ramped).arguments

    return cpl.prepare_coupled_transient(
        prepared,
        fields=tuple(
            cpl.TransientField(name, "u", capacity=value)
            for name, value in heat.CAPACITY.items()
        ),
        arguments=arguments,
        arguments_id=f"ramped-wall-heat-flux-{gain}",
        parameters={heat.CONTROL: jnp.zeros((2,))},
    )


def test_control_entering_through_time_is_admitted_and_a_silent_one_refused() -> None:
    def prepare(gain: float) -> cpl.PreparedCoupledTransition:
        return cpl.prepare_coupled_transition(
            _ramped_transient(gain),
            step=ex.STEP,
            policy=heat.implicit_euler_policy(),
            parameters={heat.CONTROL: jnp.zeros((2,))},
            control=heat.CONTROL,
        )

    # Every row is independent of the flux at t = 0, yet the flux drives the
    # transient at every later time of the window.
    assert prepare(1.0).control == heat.CONTROL
    with pytest.raises(ValueError, match=r"components \[0, 1\] do not enter"):
        prepare(0.0)
