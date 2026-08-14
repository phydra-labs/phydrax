#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _infinite_memory_problem(*, stochastic=False):
    lags = jnp.asarray([0.2, 1.0, 10.0])
    tail = phx.solver.FunctionalDelay(
        "tail",
        lambda time, state, history, args: jnp.mean(history.values(lags), axis=0),
        (0.1, jnp.inf),
    )
    noise = (
        (
            phx.solver.DelayWienerTerm(
                "driver",
                lambda time, state, memory, args: jnp.zeros(state.shape + (1,)),
                (1,),
                structure="additive",
                basis_id="infinite-memory-driver",
            ),
        )
        if stochastic
        else ()
    )
    return phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["tail"],
        lambda time, args: jnp.ones((1,)),
        (tail,),
        t0=0.0,
        t1=0.1,
        wiener_terms=noise,
    )


def test_full_history_supports_infinite_memory_functionals():
    problem = _infinite_memory_problem()
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.0, 0.05, 0.1]),
        solver=dfx.Euler(),
        dt0=0.01,
        max_steps=32,
    )

    assert problem.maximum_delay is None
    assert problem.delay_terms[0].infinite_memory
    assert jnp.allclose(solution.states[:, 0], jnp.asarray([1.0, 1.05, 1.1]))
    assert solution.stats["infinite_memory"]
    assert solution.metadata["infinite_memory"]
    assert jnp.isinf(
        solution.metadata["functional_delay_contracts"][0]["lag_interval"][1]
    )


def test_stochastic_full_history_accepts_infinite_memory_functionals():
    problem = _infinite_memory_problem(stochastic=True)
    realization = phx.stochastic.WienerRealization(
        jr.key(31),
        (1,),
        support=(0.0, 0.1),
        tolerance=1e-5,
        noise_id=problem.noise_id,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.0, 0.05, 0.1]),
        solver=dfx.Euler(),
        realization=realization,
        dt0=0.01,
        max_steps=32,
    )

    assert jnp.allclose(solution.states[..., 0], jnp.asarray([1.0, 1.05, 1.1]))
    assert solution.metadata["infinite_memory"]


@pytest.mark.parametrize("execution", ["rolling", "segmented"])
def test_infinite_memory_rejects_bounded_history_execution(execution):
    problem = _infinite_memory_problem()

    with pytest.raises(ValueError, match="finite maximum"):
        if execution == "rolling":
            phx.solver.solve_diffrax_delay(
                problem,
                save_times=jnp.asarray([0.1]),
                solver=dfx.Euler(),
                dt0=0.01,
                history_mode="rolling",
                history_capacity=16,
            )
        else:
            phx.solver.solve_diffrax_delay_segmented(
                problem,
                save_times=jnp.asarray([0.1]),
                solver=dfx.Euler(),
                dt0=0.01,
                max_steps_per_segment=8,
            )
