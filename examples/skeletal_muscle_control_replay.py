#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.motor_units import PotvinFuglevand2017Plan
from phydrax.applications.skeletal_muscle.personalization import (
    SkeletalReplayObservationOperator,
    SkeletalSurrogateReplayPlan,
)
from phydrax.control import (
    ControlProblem,
    DiscreteControlDynamics,
    PiecewiseConstantControlParameterization,
    plan_sampling_mpc,
    solve_sampling_mpc,
)
from phydrax.dynamics import TimeGrid
from phydrax.optim import Bounds


def main() -> None:
    model_plan = PotvinFuglevand2017Plan(
        central_adaptation=False, peripheral_fatigue=False
    )
    runtime = model_plan.prepare()
    grid = TimeGrid(jnp.asarray((0.0, 0.1, 0.2)), time_id="example-control-time")
    parameterization = PiecewiseConstantControlParameterization(
        grid, (1,), parameterization_id="example-motor-unit-excitation"
    )
    target_force = runtime.evaluate(runtime.initialize(), 20.0).total_force

    def running_cost(time, state, control, parameters):
        del time, parameters
        force = runtime.evaluate(
            runtime.unpack_state(state), control[0]
        ).total_force
        return ((force - target_force) / target_force) ** 2

    problem = ControlProblem(
        DiscreteControlDynamics(model_plan.as_discrete_system()),
        grid,
        runtime.pack_state(runtime.initialize()),
        running_cost=running_cost,
        args=runtime.parameters,
        problem_id="example-hard-motor-unit-control",
    )
    mpc = plan_sampling_mpc(
        problem,
        parameterization,
        candidate_count=64,
        iteration_count=3,
        elite_count=8,
        bounds=Bounds(0.0, 67.0),
        minimum_standard_deviation=0.1,
    )
    result = solve_sampling_mpc(
        mpc,
        mpc.initialize(jnp.full((2, 1), 10.0), jnp.full((2, 1), 8.0)),
        jax.random.key(7),
    )

    def exact_force(trajectory):
        return jax.vmap(
            lambda state, control: runtime.evaluate(
                runtime.unpack_state(state), control[0]
            ).total_force
        )(trajectory.states[:-1], trajectory.controls)
    exact_force_operator = SkeletalReplayObservationOperator(
        exact_force, "potvin-fuglevand-relative-force-observation"
    )

    replay = SkeletalSurrogateReplayPlan(
        problem,
        parameterization,
        exact_force_operator,
        jnp.ones((2,), dtype=bool),
        "constant-state-force-surrogate",
        "relative_muscle_force",
        absolute_tolerance=1.0e-8,
        relative_tolerance=1.0e-8,
    )
    surrogate = jax.vmap(
        lambda control: runtime.evaluate(runtime.initialize(), control[0]).total_force
    )(result.controls)
    replay_result = replay.evaluate(result.controls, surrogate)
    payload = {
        "sampling_mpc_successful": bool(result.successful),
        "action": result.action.tolist(),
        "objective": float(result.objective),
        "causal_replay_accepted": bool(replay_result.accepted),
        "observation_operator_id": replay_result.observation_operator_id,
        "source_problem_id": replay_result.source_problem_id,
        "maximum_replay_error": float(replay_result.maximum_absolute_error),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
