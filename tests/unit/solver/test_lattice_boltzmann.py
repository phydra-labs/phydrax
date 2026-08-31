#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _compiled_accelerated_flow():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    method = phx.discretization.LatticeBoltzmannMethodPlan(
        phx.discretization.BGKCollisionPlan(),
        forcing=phx.discretization.GuoForcingPlan(),
    )
    problem = phx.equations.LatticeBoltzmannProblem(
        "gradient-flow",
        2,
        acceleration=lambda time, coordinates, parameters: jnp.asarray((parameters, 0.0)),
        acceleration_id="scalar-x-acceleration",
    )
    return phx.equations.compile_lattice_boltzmann_problem(
        problem,
        discretization,
        method,
        phx.discretization.LatticeBoltzmannBoundaryPlan(),
        time_step=0.02,
    )


def test_lbm_adapter_fails_closed_on_time_step_mismatch():
    compiled = _compiled_accelerated_flow()
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(
        0.05, force_parameters=jnp.asarray(0.001)
    )
    state = compiled.initialize_state(1.0, jnp.zeros((2,)), parameters)
    method = phx.solver.LatticeBoltzmannFixedStepMethod(compiled.dynamics)
    result = method.step(
        jnp.asarray(0),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.01),
        parameters,
    )

    assert not result.successful
    np.testing.assert_array_equal(result.accepted_state, state)


def test_lbm_short_rollout_force_gradient_matches_finite_difference():
    compiled = _compiled_accelerated_flow()
    method = phx.solver.LatticeBoltzmannFixedStepMethod(compiled.dynamics)
    rollout = phx.solver.FixedStepRolloutPlan(
        retention="final", replay=phx.solver.FixedStepReplayPolicy("step")
    )

    def objective(amplitude):
        parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(
            0.05, force_parameters=amplitude
        )
        initial = compiled.initialize_state(1.0, jnp.zeros((2,)), parameters)
        problem = phx.solver.FixedStepProblem(
            method,
            initial,
            t0=0.0,
            t1=0.08,
            step_size=0.02,
            args=parameters,
        )
        final = rollout.rollout(problem).final_state
        return jnp.mean(
            compiled.macroscopic_state(0.08, final, parameters).velocity[..., 0]
        )

    amplitude = jnp.asarray(0.001)
    gradient = jax.grad(objective)(amplitude)
    epsilon = 1e-6
    finite_difference = (
        objective(amplitude + epsilon) - objective(amplitude - epsilon)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-8)
