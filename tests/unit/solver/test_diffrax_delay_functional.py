#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _functional_problem():
    lags = jnp.linspace(0.2, 0.5, 5)
    weights = jnp.asarray([1.0, 2.0, 2.0, 2.0, 1.0])
    normalization = jnp.sum(weights * jnp.exp(-lags))
    window = phx.solver.FunctionalDelay(
        "window",
        lambda time, state, history, args: jnp.tensordot(
            weights,
            history.values(lags),
            axes=((0,), (0,)),
        )
        / normalization,
        (0.2, 0.5),
        discontinuity_lags=jnp.asarray([0.2, 0.5]),
    )
    return phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["window"],
        lambda time, args: jnp.exp(time) * jnp.ones((1,)),
        (window,),
        t0=0.0,
        t1=1.0,
        problem_id="functional-delay:exponential-window",
    )


def test_functional_delay_replays_full_rolling_and_segmented_history():
    problem = _functional_problem()
    common = {
        "save_times": jnp.linspace(0.0, 1.0, 6),
        "solver": dfx.Euler(),
        "dt0": 0.01,
        "dense": True,
    }
    full = phx.solver.solve_diffrax_delay(problem, max_steps=256, **common)
    rolling = phx.solver.solve_diffrax_delay(
        problem,
        history_mode="rolling",
        max_steps=None,
        **common,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        max_steps_per_segment=11,
        **common,
    )
    query = jnp.asarray([0.57, 0.64, 0.93])

    assert jnp.allclose(full.states[:, 0], jnp.exp(full.times), rtol=0.0, atol=0.011)
    assert jnp.array_equal(rolling.states, full.states)
    assert jnp.array_equal(segmented.states, full.states)
    assert jnp.array_equal(rolling.evaluate(query), full.evaluate(query))
    assert jnp.array_equal(segmented.evaluate(query), full.evaluate(query))
    assert full.metadata["delay_mode"] == "declared-functional-retarded"
    assert segmented.metadata["delay_mode"] == "segmented-functional-retarded"
    assert full.stats["functional_tracking"] == "declared-lag-translations"
    contract = full.metadata["functional_delay_contracts"][0]
    assert contract["name"] == "window"
    assert contract["output_kind"] == "ambient"
    assert jnp.array_equal(contract["discontinuity_lags"], jnp.asarray([0.2, 0.5]))
    assert rolling.stats["history_capacity"] > rolling.stats["history_max_occupancy"]


def test_functional_delay_enforces_declared_query_window_at_initialization():
    term = phx.solver.FunctionalDelay(
        "invalid-window",
        lambda time, state, history, args: history(0.1),
        (0.2, 0.5),
    )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="outside its declared lag interval",
    ):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: memory["invalid-window"],
            lambda time, args: jnp.ones((1,)),
            (term,),
            t0=0.0,
            t1=0.5,
        )



def test_functional_delay_routes_through_stochastic_whole_and_segmented_backends():
    lags = jnp.asarray([0.1, 0.15, 0.2])
    functional = phx.solver.FunctionalDelay(
        "window",
        lambda time, state, history, args: jnp.mean(history.values(lags), axis=0),
        (0.1, 0.2),
        discontinuity_lags=lags,
    )
    noise = phx.solver.DelayWienerTerm(
        "driver",
        lambda time, state, memory, args: (0.2 * memory["window"])[..., None],
        (1,),
        structure="general",
        basis_id="functional-delay-noise",
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: 0.1 * memory["window"],
        lambda time, args: jnp.ones((1,)),
        (functional,),
        t0=0.0,
        t1=0.4,
        wiener_terms=(noise,),
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(17),
        (1,),
        support=(0.0, 0.4),
        tolerance=1e-5,
        noise_id=problem.noise_id,
    )
    common = {
        "save_times": jnp.linspace(0.0, 0.4, 5),
        "realization": realization,
        "dt0": 0.025,
        "dense": True,
    }
    full = phx.solver.solve_diffrax_delay(problem, max_steps=64, **common)
    rolling = phx.solver.solve_diffrax_delay(
        problem,
        history_mode="rolling",
        max_steps=None,
        **common,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        max_steps_per_segment=5,
        **common,
    )

    assert jnp.array_equal(rolling.states, full.states)
    assert jnp.allclose(segmented.states, full.states, rtol=0.0, atol=5e-10)
    assert full.metadata["delay_mode"] == (
        "declared-functional-retarded-stochastic"
    )
    assert segmented.metadata["delay_mode"] == "segmented-functional-retarded"
    assert full.stats["functional_tracking"] == "declared-lag-translations"
    assert full.realization is realization