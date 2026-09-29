#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Model-predictive control of a finite-element/virtual-element transient.

The controlled plant is the coupled heat-conduction transient of
``examples/_coupled_heat_transient.py``: P1 triangles next to degree-1 virtual
elements, a matching transmission law, each owner's own capacity operator, and
the right-wall heat flux ``g = (g_lower, g_upper)`` bound as the P6 control
parameter. ``prepare_coupled_transition`` prepares one native implicit-Euler
step between two physical times (the DAE solve is prepared once and rebound to
every interval); ``discrete_system`` exposes it through the ``DiscreteSystem``
ABI, so ``DiscreteControlDynamics``/``ControlProblem`` roll it out with the
step context ``(t_k, t_{k+1}, k)``, and the three temperature sensors (P6
observation bindings) are the output port.

The example prints

1. the rollout of random wall fluxes against an independent reference: the
   implicit-Euler recursion of the semi-discrete system assembled on the host
   from the two meshes (``_coupled_heat_transient.host_semidiscrete``);
2. matrix-free ``PreparedControlLinearization`` actions at a non-uniform step
   (``t -> t + 2 dt``) against the host step operator;
3. a dense receding-horizon MPC that drives the sensor temperatures to a
   target under flux bounds, built by ``linear_quadratic_problem_from_discrete_dynamics``
   under an explicit materialization budget, compared with the host
   bound-constrained least-squares solution of the same finite-horizon problem
   (with the host sensor rows), and replayed through the coupled transition;
4. the refusal of the same budget for the refined full-order plant.

It raises if any accepted-step or agreement check fails.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

import phydrax as phx
from examples import _coupled_heat_transient as heat
from phydrax.solver import coupling as cpl


jax.config.update("jax_enable_x64", True)

CELLS = 4
REFINED_CELLS = 8
STEP = 0.1
HORIZON = 8
TARGET = np.asarray([0.3, 0.12, 0.12])
SENSOR_WEIGHT = 100.0
CONTROL_WEIGHT = 1.0e-2
FLUX_BOUND = 0.6
BUDGET = phx.linalg.MaterializationPolicy(max_entries=40_000, max_bytes=320_000)
# The dense window QP holds every stage state: (N + 1) n + N m variables, N n
# dynamics equalities, and 2 N m bounds (KKT dimension 800 for n = 40, N = 8).
QP_POLICY = phx.optim.ConvexSolvePolicy(
    phx.optim.DensePrimalDualQP(max_kkt_dimension=1024)
)


def plant(
    cells: int, /
) -> tuple[
    cpl.PreparedCoupledTransient,
    phx.control.DiscreteControlDynamics,
    cpl.CoupledObservationPort,
]:
    """The coupled transient, its controlled discrete dynamics, and sensor port."""
    transient = heat.coupled_heat_transient(cells)
    transition = cpl.prepare_coupled_transition(
        transient,
        step=STEP,
        policy=heat.implicit_euler_policy(),
        parameters={heat.CONTROL: jnp.zeros((2,))},
        control=heat.CONTROL,
    )
    dynamics = phx.control.DiscreteControlDynamics(
        transition.discrete_system(system_id="coupled-heat-plant")
    )
    return transient, dynamics, transition.observation_port(heat.SENSOR_BINDINGS)


def host_step(
    semidiscrete: heat.HostSemidiscrete, step: float, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``x+ = A x + B g + c`` of one implicit-Euler step of ``C x' + K x = b + B g``."""
    matrix = semidiscrete.capacity + step * semidiscrete.stiffness
    return (
        np.linalg.solve(matrix, semidiscrete.capacity),
        np.linalg.solve(matrix, step * semidiscrete.flux),
        np.linalg.solve(matrix, step * semidiscrete.source),
    )


def sensor_matrix(
    dynamics: phx.control.DiscreteControlDynamics,
    port: cpl.CoupledObservationPort,
    grid: phx.dynamics.TimeGrid,
    /,
) -> np.ndarray:
    """The port's dense sensor matrix ``H`` from the linearization (zero lift: linear)."""
    size = dynamics.state_shape[0]
    linearization = phx.control.linearize_discrete_dynamics(
        dynamics,
        grid.times[0],
        jnp.zeros((size,)),
        jnp.zeros((2,)),
        materialization=BUDGET,
        output=port,
        target_time=grid.times[1],
        step_index=0,
    )
    return np.asarray(linearization.output_matrix)


def tracking_problem(
    dynamics: phx.control.DiscreteControlDynamics,
    grid: phx.dynamics.TimeGrid,
    output: np.ndarray,
    /,
) -> phx.control.LinearQuadraticControlProblem:
    """Sensor tracking ``(H x - y*)ᵀ W (H x - y*) / 2 + gᵀ R g / 2`` under flux bounds."""
    size = dynamics.state_shape[0]
    weight = SENSOR_WEIGHT * np.eye(TARGET.size)
    state_cost = output.T @ weight @ output
    state_linear = -output.T @ weight @ TARGET
    return phx.control.linear_quadratic_problem_from_discrete_dynamics(
        dynamics,
        grid,
        jnp.zeros((HORIZON, size)),
        jnp.zeros((HORIZON, 2)),
        jnp.zeros((size,)),
        jnp.broadcast_to(state_cost, (HORIZON, size, size)),
        jnp.broadcast_to(CONTROL_WEIGHT * jnp.eye(2), (HORIZON, 2, 2)),
        state_cost,
        materialization=BUDGET,
        state_linear=jnp.broadcast_to(state_linear, (HORIZON, size)),
        terminal_linear=state_linear,
        control_lower_bounds=jnp.full((HORIZON, 2), -FLUX_BOUND),
        control_upper_bounds=jnp.full((HORIZON, 2), FLUX_BOUND),
        problem_id="coupled-heat-sensor-tracking",
    )


def host_optimal_controls(
    operator: tuple[np.ndarray, np.ndarray, np.ndarray],
    output: np.ndarray,
    initial: np.ndarray,
    /,
    *,
    bound: float = FLUX_BOUND,
) -> np.ndarray:
    """Exact bound-constrained optimum of the finite-horizon problem (host BVLS).

    ``x_t = F_t + G_t g`` is affine in the stacked controls, so the objective is
    a bounded linear least-squares problem in ``g``.
    """
    matrix, control, bias = operator
    size, width = matrix.shape[0], control.shape[1]
    free = np.zeros((size, HORIZON * width))
    offset = np.asarray(initial, dtype=np.float64)
    rows = []
    targets = []
    root = np.sqrt(SENSOR_WEIGHT)
    for stage in range(HORIZON):
        free = matrix @ free
        free[:, stage * width : (stage + 1) * width] += control
        offset = matrix @ offset + bias
        rows.append(root * output @ free)
        targets.append(root * (TARGET - output @ offset))
    rows.append(np.sqrt(CONTROL_WEIGHT) * np.eye(HORIZON * width))
    targets.append(np.zeros(HORIZON * width))
    solution = scipy.optimize.lsq_linear(
        np.concatenate(rows),
        np.concatenate(targets),
        bounds=(-bound, bound),
        method="bvls",
        tol=1e-14,
    )
    if not solution.success:
        raise RuntimeError("Host bounded least squares did not converge.")
    return solution.x.reshape((HORIZON, width))


def check(condition: bool, message: str, /) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    started = time.perf_counter()
    transient, dynamics, port = plant(CELLS)
    size = dynamics.state_shape[0]
    print(
        f"plant: {size} native coordinates "
        f"({', '.join('/'.join(path) for path in transient.paths)}), "
        f"prepared in {time.perf_counter() - started:.1f} s"
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, HORIZON * STEP, HORIZON + 1), time_id="control-grid"
    )
    problem = phx.control.ControlProblem(
        dynamics, grid, jnp.zeros((size,)), problem_id="coupled-heat-control"
    )
    policy = phx.control.PiecewiseConstantControlParameterization(
        grid, (2,), parameterization_id="held-wall-flux"
    )
    semidiscrete = heat.host_semidiscrete(transient, CELLS)
    matrix, control, bias = host_step(semidiscrete, STEP)

    fluxes = np.random.default_rng(20260928).uniform(-1.0, 1.0, (HORIZON, 2))
    trajectory = problem.rollout(policy, jnp.asarray(fluxes))
    evidence = trajectory.transition_evidence
    if evidence is None:
        raise RuntimeError("A discrete rollout records its transition evidence.")
    check(bool(jnp.all(trajectory.valid)), "The coupled rollout was not accepted.")
    reference = np.zeros(size)
    rollout_error = 0.0
    for stage in range(HORIZON):
        reference = matrix @ reference + control @ fluxes[stage] + bias
        rollout_error = max(
            rollout_error,
            float(np.max(np.abs(np.asarray(trajectory.states[stage + 1]) - reference))),
        )
    scale = float(np.max(np.abs(reference)))
    print(
        f"1. rollout of held random fluxes: every step accepted "
        f"({int(jnp.sum(evidence.successful))}/{HORIZON}); "
        f"max |x - x_host| / max|x_host| = {rollout_error / scale:.2e}"
    )
    check(rollout_error <= 1e-9 * scale, "Rollout differs from the host recursion.")

    stage = 3
    linearization = phx.control.prepare_control_linearization(
        dynamics,
        grid.times[stage],
        trajectory.states[stage],
        jnp.asarray(fluxes[stage]),
        target_time=grid.times[stage + 2],
        step_index=stage,
        output=port,
    )
    double = host_step(semidiscrete, 2.0 * STEP)
    direction = np.random.default_rng(3).standard_normal(size + 2)
    action = np.asarray(linearization.dynamics_jacobian.mv(jnp.asarray(direction)))
    expected = double[0] @ direction[:size] + double[1] @ direction[size:]
    linear_error = float(np.max(np.abs(action - expected)) / np.max(np.abs(expected)))
    print(
        f"2. matrix-free [A B] action over t -> t + 2 dt (step index {stage}): "
        f"valid={bool(linearization.valid)}, relative error vs host {linear_error:.2e}"
    )
    check(bool(linearization.valid), "The local linearization is not valid.")
    check(linear_error <= 1e-9, "Matrix-free linearization differs from the host.")

    output = sensor_matrix(dynamics, port, grid)
    check(
        bool(np.allclose(output, semidiscrete.sensors, rtol=0.0, atol=1e-14)),
        "The sensor port differs from the host point evaluation.",
    )
    specification = tracking_problem(dynamics, grid, output)
    result = phx.control.solve_receding_horizon_mpc(
        specification,
        prediction_horizon=HORIZON,
        terminal_policy="global",
        policy=QP_POLICY,
    )
    check(bool(jnp.all(result.successful)), "An MPC window was not OPTIMAL.")
    optimal = host_optimal_controls(
        (matrix, control, bias), semidiscrete.sensors, np.zeros(size)
    )
    control_error = float(np.max(np.abs(np.asarray(result.controls) - optimal)))
    active = int(np.sum(np.isclose(np.abs(optimal), FLUX_BOUND, atol=1e-9)))
    replay = problem.rollout(result.policy, result.parameters)
    check(bool(jnp.all(replay.valid)), "The MPC replay was not accepted.")
    replay_error = float(jnp.max(jnp.abs(replay.states - result.states)))
    final = np.asarray(
        port.values(
            grid.times[-1],
            replay.states[-1],
            {heat.CONTROL: result.parameters[-1]},
        )
    )
    print(
        f"3. dense MPC ({HORIZON} windows, {active} of {optimal.size} flux bounds active): "
        f"max |g_mpc - g_host| = {control_error:.2e}; replay through the coupled "
        f"transition max |x - x_mpc| = {replay_error:.2e}; final sensors "
        f"{np.array2string(final, precision=3)} (target {TARGET})"
    )
    check(control_error <= 1e-5, "MPC controls differ from the host optimum.")
    check(replay_error <= 1e-9, "MPC handoff differs from the coupled transition.")

    _, refined_dynamics, _ = plant(REFINED_CELLS)
    refined_size = refined_dynamics.state_shape[0]
    try:
        phx.control.linear_quadratic_problem_from_discrete_dynamics(
            refined_dynamics,
            grid,
            jnp.zeros((HORIZON, refined_size)),
            jnp.zeros((HORIZON, 2)),
            jnp.zeros((refined_size,)),
            jnp.broadcast_to(
                jnp.eye(refined_size), (HORIZON, refined_size, refined_size)
            ),
            jnp.broadcast_to(jnp.eye(2), (HORIZON, 2, 2)),
            jnp.eye(refined_size),
            materialization=BUDGET,
            problem_id="refined-coupled-heat",
        )
    except ValueError as error:
        print(f"4. refined plant ({refined_size} coordinates) refused: {error}")
    else:
        raise RuntimeError("The refined full-order plant must exceed the budget.")


if __name__ == "__main__":
    main()
