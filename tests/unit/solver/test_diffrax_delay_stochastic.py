import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _problem(
    *,
    drift=None,
    diffusion=None,
    t1=0.2,
    delay=0.5,
    interpretation="ito",
    structure="general",
    basis_id="delay-basis",
):
    if drift is None:
        drift = lambda time, state, memory, args: 2.0 * memory[0]
    if diffusion is None:
        diffusion = lambda time, state, memory, args: 3.0 * jnp.ones(state.shape + (1,))
    return phx.solver.DelayDifferentialProblem(
        drift,
        lambda time, args: jnp.ones((1,)),
        (phx.solver.ConstantDelay("past", delay),),
        t0=0.0,
        t1=t1,
        wiener_terms=(
            phx.solver.DelayWienerTerm(
                "driver",
                diffusion,
                (1,),
                structure=structure,
                basis_id=basis_id,
            ),
        ),
        interpretation=interpretation,
        problem_id="stochastic-delay-test",
    )


def _realization(
    problem,
    seed=0,
    *,
    support=(0.0, 1.0),
    sample_shape=(),
    tolerance=1e-4,
    levy_area="brownian",
    noise_shape=None,
    noise_id="use-problem",
):
    resolved_noise_id = problem.noise_id if noise_id == "use-problem" else noise_id
    return phx.stochastic.WienerRealization(
        jr.key(seed),
        problem.noise_shape if noise_shape is None else noise_shape,
        support=support,
        sample_shape=sample_shape,
        tolerance=tolerance,
        levy_area=levy_area,
        noise_id=resolved_noise_id,
    )


def test_ito_euler_maruyama_matches_manufactured_one_step_and_provenance():
    problem = _problem(structure="additive")
    realization = _realization(problem, 3)
    increment = realization.increments(jnp.asarray(0.0), jnp.asarray(0.2))[0]

    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        solver=dfx.Euler(),
        dt0=0.2,
        dense=True,
    )

    assert jnp.allclose(solution.states[0, 0], 1.0 + 0.4 + 3.0 * increment)
    assert solution.realization is realization
    assert solution.metadata["noise_id"] == problem.noise_id
    assert solution.metadata["basis_ids"] == ("delay-basis",)
    assert solution.metadata["wiener_term_slices"]["driver"] == (0, 1)
    assert solution.metadata["levy_area"] == "brownian"
    assert solution.metadata["noise_structures"] == ("additive",)
    assert solution.stats["controller_mode"] == "fixed"
    assert solution.stats["continuous_extension"] == "euler-maruyama-wiener-path"
    assert solution.has_dense_interpolation


def test_stochastic_delay_preserves_selected_levy_area_path():
    problem = _problem(structure="additive")
    realization = _realization(
        problem,
        4,
        levy_area="space_time_time",
    )
    first = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        dt0=0.2,
    )
    replay = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        dt0=0.2,
    )

    assert first.realization is realization
    assert first.metadata["levy_area"] == "space_time_time"
    assert jnp.array_equal(first.states, replay.states)


def test_stratonovich_euler_heun_matches_manufactured_one_step():
    sigma = 0.7
    problem = _problem(
        drift=lambda time, state, memory, args: 0.4 * memory["past"],
        diffusion=lambda time, state, memory, args: sigma * state[..., None],
        interpretation="stratonovich",
    )
    realization = _realization(problem, 5)
    increment = realization.increments(jnp.asarray(0.0), jnp.asarray(0.2))[0]
    predictor = 1.0 + sigma * increment
    expected = 1.0 + 0.4 * 0.2 + 0.5 * (sigma + sigma * predictor) * increment

    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        solver=dfx.EulerHeun(),
        dt0=0.2,
    )

    assert jnp.allclose(solution.states[0, 0], expected)
    assert solution.metadata["interpretation"] == "stratonovich"
    assert solution.stats["continuous_extension"] == "euler-heun-wiener-path"


def test_dense_history_uses_the_same_wiener_path_inside_an_accepted_step():
    problem = _problem(t1=0.4, structure="additive")
    realization = _realization(problem, 7, support=(0.0, 0.4))
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.4]),
        realization=realization,
        dt0=0.2,
        dense=True,
    )
    query = jnp.asarray([0.03, 0.13, 0.19])
    increments = realization.increments(jnp.zeros_like(query), query)[..., 0]
    expected = 1.0 + 2.0 * query + 3.0 * increments

    assert jnp.allclose(solution.evaluate(query)[..., 0], expected)
    history = solution.interpolation.history.computed_history
    assert int(history.size) == int(solution.stats["num_accepted_steps"])

def test_batched_stochastic_rolling_history_replays_full_path_solution():
    problem = _problem(t1=0.8, delay=0.2, structure="additive")
    realization = _realization(
        problem,
        8,
        support=(0.0, 0.8),
        sample_shape=(2,),
    )
    common = {
        "save_times": jnp.linspace(0.0, 0.8, 9),
        "realization": realization,
        "solver": dfx.Euler(),
        "dt0": 0.05,
        "dense": True,
    }
    full = phx.solver.solve_diffrax_delay(problem, max_steps=32, **common)
    rolling = phx.solver.solve_diffrax_delay(
        problem,
        history_mode="rolling",
        max_steps=None,
        **common,
    )
    query = jnp.asarray([0.65, 0.8])

    assert jnp.array_equal(rolling.states, full.states)
    assert jnp.array_equal(rolling.evaluate(query), full.evaluate(query))
    assert rolling.stats["retained_history_interval"].shape == (2, 2)
    assert jnp.all(rolling.stats["num_history_evictions"] > 0)
    assert rolling.stats["active_history_bytes"] > 0
    with pytest.raises(eqx.EquinoxRuntimeError, match="every solution interval"):
        rolling.evaluate(jnp.asarray([0.1]))


def test_delayed_diffusion_reads_the_path_consistent_accepted_history():
    delay = 0.08
    problem = _problem(
        drift=lambda time, state, memory, args: jnp.zeros_like(state),
        diffusion=lambda time, state, memory, args: memory["past"][..., None],
        t1=0.15,
        delay=delay,
    )
    realization = _realization(problem, 9, tolerance=1e-5)
    grid = jnp.asarray([0.0, 0.05, 0.1, 0.15])
    increments = realization.increments(grid[:-1], grid[1:])[..., 0]
    delayed_time = grid[2] - delay
    delayed_increment = realization.increments(
        jnp.asarray(0.0),
        delayed_time,
    )[0]
    expected = (
        1.0 + increments[0] + increments[1] + (1.0 + delayed_increment) * increments[2]
    )

    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.15]),
        realization=realization,
        dt0=0.05,
        initial_discontinuities=jnp.asarray([]),
    )

    assert jnp.allclose(solution.states[0, 0], expected)


def test_realization_replay_antithetic_sign_and_prefixes_are_preserved():
    problem = _problem(structure="additive")
    realization = _realization(problem, 11)
    first = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        dt0=0.2,
    )
    replay = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        dt0=0.2,
    )
    changed = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=_realization(problem, 12),
        dt0=0.2,
    )
    assert jnp.array_equal(first.states, replay.states)
    assert not jnp.array_equal(first.states, changed.states)

    antithetic = phx.stochastic.WienerRealization.antithetic(
        jr.key(13),
        problem.noise_shape,
        support=(0.0, 1.0),
        num_pairs=1,
        tolerance=1e-4,
        noise_id=problem.noise_id,
    )
    paired = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=antithetic,
        dt0=0.2,
    )
    assert jnp.allclose(jnp.sum(paired.states[:, 0, 0]), 2.0 * (1.0 + 0.4))

    small_realization = _realization(problem, 17, sample_shape=(2,))
    large_realization = _realization(problem, 17, sample_shape=(4,))
    small = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=small_realization,
        dt0=0.2,
    )
    large = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=large_realization,
        dt0=0.2,
    )
    assert jnp.array_equal(small.states, large.states[:2])


def test_stochastic_delay_preserves_realization_sample_and_state_shapes():
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: jnp.ones((2, 2)),
        (phx.solver.ConstantDelay("past", 0.5),),
        t0=0.0,
        t1=0.2,
        wiener_terms=(
            phx.solver.DelayWienerTerm(
                "driver",
                lambda time, state, memory, args: jnp.ones(state.shape + (1,)),
                (1,),
                structure="additive",
                basis_id="matrix-basis",
            ),
        ),
    )
    realization = _realization(problem, 19, sample_shape=(2, 3))
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.1, 0.2]),
        realization=realization,
        dt0=0.1,
        dense=True,
    )

    assert solution.sample_shape == (2, 3)
    assert solution.state_shape == (2, 2)
    assert solution.states.shape == (2, 3, 2, 2, 2)
    assert solution.valid.shape == (2, 3, 2)
    query = jnp.asarray([[0.05, 0.15]])
    dense = solution.evaluate(query)
    assert dense.shape == (2, 3, 1, 2, 2, 2)
    assert jnp.allclose(dense[..., 0, 0], dense[..., 1, 1])


def test_stochastic_delay_is_jittable_vectorizable_and_differentiable():
    template = _problem(structure="additive")
    realization = _realization(template, 23)
    increment = realization.increments(jnp.asarray(0.0), jnp.asarray(0.2))[0]

    def terminal(rate):
        problem = _problem(
            drift=lambda time, state, memory, args: rate * memory[0],
            diffusion=lambda time, state, memory, args: jnp.ones(state.shape + (1,)),
            structure="additive",
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.2]),
            realization=realization,
            dt0=0.2,
            max_steps=8,
        ).states[0, 0]

    rate = jnp.asarray(0.6)
    expected = 1.0 + 0.2 * rate + increment
    assert jnp.allclose(jax.jit(terminal)(rate), expected)
    assert jnp.allclose(jax.grad(terminal)(rate), 0.2)
    rates = jnp.asarray([0.2, 0.6, 0.9])
    expected_batch = 1.0 + 0.2 * rates + increment
    assert jnp.allclose(jax.vmap(terminal)(rates), expected_batch)


class _WrongInterpolationEuler(dfx.Euler):
    interpolation_cls = dfx.ThirdOrderHermitePolynomialInterpolation


def test_stochastic_delay_rejects_wrong_interpretation_solver_and_interpolation():
    ito = _problem()
    ito_realization = _realization(ito)
    with pytest.raises(ValueError, match="Itô.*diffrax.Euler"):
        phx.solver.solve_diffrax_delay(
            ito,
            save_times=jnp.asarray([0.2]),
            realization=ito_realization,
            solver=dfx.EulerHeun(),
            dt0=0.2,
        )
    with pytest.raises(ValueError, match="interpolation.*certified"):
        phx.solver.solve_diffrax_delay(
            ito,
            save_times=jnp.asarray([0.2]),
            realization=ito_realization,
            solver=_WrongInterpolationEuler(),
            dt0=0.2,
        )

    stratonovich = _problem(interpretation="stratonovich")
    with pytest.raises(ValueError, match="Stratonovich.*diffrax.EulerHeun"):
        phx.solver.solve_diffrax_delay(
            stratonovich,
            save_times=jnp.asarray([0.2]),
            realization=_realization(stratonovich),
            solver=dfx.Euler(),
            dt0=0.2,
        )
    with pytest.raises(ValueError, match="ConstantStepSize"):
        phx.solver.solve_diffrax_delay(
            ito,
            save_times=jnp.asarray([0.2]),
            realization=ito_realization,
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-5),
            dt0=0.2,
        )


def test_stochastic_delay_requires_realization_and_step_and_accepts_advanced_delays():
    problem = _problem()
    with pytest.raises(ValueError, match="WienerRealization"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.2]),
            dt0=0.2,
        )
    with pytest.raises(ValueError, match="explicit dt0"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.2]),
            realization=_realization(problem),
        )

    advanced_terms = (
        phx.solver.StateDependentDelay(
            "state",
            lambda time, state, args: jnp.asarray(0.15),
            minimum_delay=0.1,
            maximum_delay=0.2,
        ),
        phx.solver.DistributedDelay(
            "distributed",
            lambda time, lag, state, args: jnp.asarray(1.0),
            (0.1, 0.2),
        ),
    )
    for term in advanced_terms:
        advanced = phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args, name=term.name: memory[name],
            lambda time, args: jnp.ones((1,)),
            (term,),
            t0=0.0,
            t1=0.2,
            wiener_terms=(
                phx.solver.DelayWienerTerm(
                    "driver",
                    lambda time, state, memory, args: jnp.ones((1, 1)),
                    (1,),
                    structure="additive",
                    basis_id="delay-basis",
                ),
            ),
        )
        solution = phx.solver.solve_diffrax_delay(
            advanced,
            save_times=jnp.asarray([0.2]),
            realization=_realization(advanced),
            dt0=0.05,
        )
        assert bool(solution.successful)
        assert jnp.all(jnp.isfinite(solution.states))


def test_stochastic_state_dependent_and_distributed_delays_replay_all_history_modes():
    state_delay = phx.solver.StateDependentDelay(
        "past",
        lambda time, state, args: jnp.asarray(0.2),
        minimum_delay=0.2,
        maximum_delay=0.2,
    )
    distributed_delay = phx.solver.DistributedDelay(
        "spread",
        lambda time, lag, state, args: jnp.asarray(1.0),
        (0.15, 0.25),
    )
    noise = phx.solver.DelayWienerTerm(
        "driver",
        lambda time, state, memory, args: (
            0.2 * memory["past"] + 0.1 * memory["spread"]
        )[..., None],
        (1,),
        structure="general",
        basis_id="advanced-delay-basis",
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: 0.1 * memory["spread"],
        lambda time, args: jnp.ones((1,)),
        (state_delay, distributed_delay),
        t0=0.0,
        t1=0.6,
        wiener_terms=(noise,),
    )
    realization = _realization(
        problem,
        4,
        support=(0.0, 0.6),
        tolerance=1e-5,
    )
    common = {
        "save_times": jnp.linspace(0.0, 0.6, 7),
        "realization": realization,
        "dt0": 0.025,
        "dense": True,
    }
    full = phx.solver.solve_diffrax_delay(problem, max_steps=128, **common)
    rolling = phx.solver.solve_diffrax_delay(
        problem,
        history_mode="rolling",
        max_steps=None,
        **common,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        max_steps_per_segment=7,
        **common,
    )
    query = jnp.asarray([0.41, 0.53, 0.6])

    assert jnp.array_equal(rolling.states, full.states)
    assert jnp.array_equal(segmented.states, full.states)
    assert jnp.array_equal(rolling.evaluate(query), full.evaluate(query))
    assert jnp.array_equal(segmented.evaluate(query), full.evaluate(query))
    assert full.stats["state_dependent_tracking"] == (
        "first-order-pathwise-untracked"
    )
    assert segmented.stats["state_dependent_tracking"] == (
        "first-order-pathwise-untracked"
    )
    quadrature = full.metadata["distributed_delay_quadrature"]
    assert quadrature[0]["name"] == "spread"
    assert rolling.stats["history_capacity"] > rolling.stats["history_max_occupancy"]


def test_stochastic_delay_rejects_wrong_noise_basis_support_and_capability():
    problem = _problem()
    solve = lambda realization, **kwargs: phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        dt0=0.2,
        **kwargs,
    )
    with pytest.raises(ValueError, match="noise_shape"):
        solve(_realization(problem, noise_shape=(2,)))
    with pytest.raises(ValueError, match="noise_id"):
        solve(_realization(problem, noise_id="wrong-basis"))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="support"):
        solve(_realization(problem, support=(0.0, 0.1)))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="tolerance"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.2]),
            realization=_realization(problem, tolerance=0.2),
            dt0=0.2,
        )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="causal delay step bound",
    ):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.2]),
            realization=_realization(problem),
            dt0=0.6,
        )


def test_coupled_fine_step_reduces_strong_error_on_the_same_global_paths():
    problem = _problem(
        drift=lambda time, state, memory, args: 0.3 * memory[0],
        diffusion=lambda time, state, memory, args: 0.5 * state[..., None],
        t1=0.8,
        delay=0.2,
    )
    realization = _realization(
        problem,
        31,
        sample_shape=(32,),
        support=(0.0, 0.8),
        tolerance=1e-4,
    )

    def terminal(step):
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.8]),
            realization=realization,
            dt0=step,
            max_steps=256,
        ).states[:, 0, 0]

    coarse = terminal(0.1)
    fine = terminal(0.05)
    reference = terminal(0.0125)
    coarse_error = jnp.sqrt(jnp.mean((coarse - reference) ** 2))
    fine_error = jnp.sqrt(jnp.mean((fine - reference) ** 2))

    assert fine_error < coarse_error
