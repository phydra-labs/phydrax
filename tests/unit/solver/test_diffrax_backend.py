import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _geometric_problem(rate=0.7):
    return phx.solver.DifferentialProblem(
        lambda t, state, value: value * state,
        jnp.asarray([2.0]),
        t0=0.0,
        t1=1.0,
        args=jnp.asarray(rate),
    )


def _brownian_problem(
    *,
    interpretation="ito",
    structure="general",
    t0=0.0,
    t1=1.0,
    initial_state=0.0,
):
    return phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.asarray([initial_state]),
        t0=t0,
        t1=t1,
        wiener_terms=(
            phx.solver.WienerTerm(
                "brownian",
                lambda t, state, args: jnp.ones((1, 1)),
                (1,),
                structure=structure,
            ),
        ),
        interpretation=interpretation,
    )


def _realization(
    seed,
    *,
    sample_shape=(),
    support=(0.0, 1.0),
    tolerance=1e-3,
    levy_area="brownian",
    label=None,
):
    return phx.stochastic.WienerRealization(
        jr.key(seed),
        (1,),
        support=support,
        sample_shape=sample_shape,
        tolerance=tolerance,
        levy_area=levy_area,
        label=label,
    )


def test_solve_diffrax_ode_is_accurate_differentiable_and_jittable():
    times = jnp.linspace(0.0, 1.0, 11)
    solution = phx.solver.solve_diffrax(_geometric_problem(), save_times=times)
    expected = 2.0 * jnp.exp(0.7 * times)

    def terminal(rate):
        solved = phx.solver.solve_diffrax(
            _geometric_problem(rate),
            save_times=jnp.asarray([1.0]),
        )
        return solved.states[-1, 0]

    compiled = jax.jit(terminal)(jnp.asarray(0.7))
    derivative = jax.grad(terminal)(jnp.asarray(0.7))

    assert solution.states.shape == (11, 1)
    assert solution.times.shape == (11,)
    assert solution.sample_shape == ()
    assert solution.solver_name == "Tsit5"
    assert bool(solution.successful)
    assert jnp.allclose(solution.states[:, 0], expected, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(compiled, 2.0 * jnp.exp(0.7), rtol=2e-5)
    assert jnp.allclose(derivative, 2.0 * jnp.exp(0.7), rtol=3e-5)


def test_dense_ode_evaluates_vector_times_and_remains_differentiable():
    query_times = jnp.asarray([[0.0, 0.25], [0.5, 1.0]])
    solution = phx.solver.solve_diffrax(
        _geometric_problem(),
        save_times=jnp.asarray([0.0, 1.0]),
        dense=True,
    )
    values = solution.evaluate(query_times)
    compiled = eqx.filter_jit(lambda solved, query: solved.evaluate(query))(
        solution, query_times
    )

    def interpolated_terminal(rate):
        solved = phx.solver.solve_diffrax(
            _geometric_problem(rate),
            save_times=jnp.asarray([0.0]),
            dense=True,
        )
        return solved.evaluate(jnp.asarray(1.0))[0]

    derivative = jax.grad(interpolated_terminal)(jnp.asarray(0.7))
    expected = 2.0 * jnp.exp(0.7 * query_times)

    assert solution.has_dense_interpolation
    assert values.shape == (2, 2, 1)
    assert jnp.allclose(values[..., 0], expected, rtol=2e-5, atol=2e-6)
    assert jnp.array_equal(compiled, values)
    assert jnp.allclose(derivative, 2.0 * jnp.exp(0.7), rtol=3e-5)


def test_dense_interpolation_is_opt_in_and_rejects_out_of_range_times():
    plain = phx.solver.solve_diffrax(
        _geometric_problem(),
        save_times=jnp.asarray([0.0, 1.0]),
    )
    dense = phx.solver.solve_diffrax(
        _geometric_problem(),
        save_times=jnp.asarray([0.0, 1.0]),
        dense=True,
    )

    assert not plain.has_dense_interpolation
    with pytest.raises(ValueError, match="no dense interpolation"):
        plain.evaluate(jnp.asarray(0.5))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="within every solution interval",
    ):
        dense.evaluate(jnp.asarray(1.1))


def test_solve_diffrax_sde_replays_realization_and_changes_with_key():
    problem = _brownian_problem()
    times = jnp.asarray([0.25, 0.5, 1.0])
    first_realization = _realization(3, label="first")
    first = phx.solver.solve_diffrax(
        problem,
        save_times=times,
        realization=first_realization,
        dt0=0.02,
    )
    replay = phx.solver.solve_diffrax(
        problem,
        save_times=times,
        realization=first_realization,
        dt0=0.02,
    )
    changed = phx.solver.solve_diffrax(
        problem,
        save_times=times,
        realization=_realization(4),
        dt0=0.02,
    )
    compiled = eqx.filter_jit(
        lambda value: phx.solver.solve_diffrax(
            problem,
            save_times=times,
            realization=value,
            dt0=0.02,
        )
    )(first_realization)

    assert first.solver_name == "Euler"
    assert first.realization is first_realization
    assert first.realization.label == "first"
    assert len(first.realization.realization_id) == 64
    assert jnp.array_equal(first.states, replay.states)
    assert not jnp.array_equal(first.states, changed.states)
    assert jnp.array_equal(first.states, compiled.states)
    assert jnp.array_equal(
        jr.key_data(first.realization.path_keys),
        jr.key_data(jr.fold_in(first_realization.root_key, 0)),
    )


def test_solve_diffrax_ensemble_has_process_axis_and_brownian_moments():
    realization = _realization(8, sample_shape=(256,))
    solution = phx.solver.solve_diffrax_ensemble(
        _brownian_problem(),
        save_times=jnp.asarray([1.0]),
        realization=realization,
        dt0=0.02,
    )
    terminal = solution.states[:, -1, 0]

    assert solution.states.shape == (256, 1, 1)
    assert solution.times.shape == (256, 1)
    assert solution.valid.shape == (256, 1)
    assert solution.sample_shape == (256,)
    assert solution.realization.path_keys.shape == (256,)
    assert jnp.all(solution.successful)
    assert abs(float(jnp.mean(terminal))) < 0.15
    assert jnp.allclose(jnp.var(terminal), 1.0, rtol=0.2, atol=0.1)


def test_dense_sde_ensemble_preserves_sample_query_and_state_axes():
    saved_times = jnp.asarray([0.0, 0.5, 1.0])
    solution = phx.solver.solve_diffrax_ensemble(
        _brownian_problem(),
        save_times=saved_times,
        realization=_realization(9, sample_shape=(4,)),
        dt0=0.01,
        dense=True,
    )
    query_times = jnp.asarray([[0.0, 0.5], [0.75, 1.0]])

    values = solution.evaluate(query_times)

    assert solution.has_dense_interpolation
    assert values.shape == (4, 2, 2, 1)
    assert jnp.all(jnp.isfinite(values))
    assert jnp.array_equal(solution.evaluate(saved_times), solution.states)


def test_ensemble_preserves_multidimensional_sample_shape():
    solution = phx.solver.solve_diffrax_ensemble(
        _brownian_problem(t1=0.1),
        save_times=jnp.asarray([0.1]),
        realization=_realization(10, sample_shape=(2, 3)),
        dt0=0.02,
    )

    assert solution.sample_shape == (2, 3)
    assert solution.states.shape == (2, 3, 1, 1)
    assert solution.times.shape == (2, 3, 1)
    assert solution.stats["num_steps"].shape == (2, 3)


def test_stratonovich_defaults_to_euler_heun_and_accepts_explicit_solver():
    problem = _brownian_problem(interpretation="stratonovich")
    realization = _realization(5)
    default = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([1.0]),
        realization=realization,
        dt0=0.02,
    )
    explicit = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([1.0]),
        realization=realization,
        solver=dfx.EulerHeun(),
        dt0=0.02,
    )

    assert default.solver_name == "EulerHeun"
    assert jnp.array_equal(default.states, explicit.states)


def test_global_realization_matches_direct_and_split_solves():
    realization = _realization(31, tolerance=1e-4)
    direct = phx.solver.solve_diffrax(
        _brownian_problem(),
        save_times=jnp.asarray([0.5, 1.0]),
        realization=realization,
        dt0=0.01,
    )
    left = phx.solver.solve_diffrax(
        _brownian_problem(t1=0.5),
        save_times=jnp.asarray([0.5]),
        realization=realization,
        dt0=0.01,
    )
    right = phx.solver.solve_diffrax(
        _brownian_problem(t0=0.5, initial_state=left.states[-1, 0]),
        save_times=jnp.asarray([1.0]),
        realization=realization,
        dt0=0.01,
    )

    split = jnp.concatenate((left.states, right.states), axis=0)
    assert jnp.allclose(direct.states, split, rtol=0.0, atol=2e-12)


def test_realization_paths_are_prefix_stable_when_batch_grows():
    problem = _brownian_problem(t1=0.1)
    small = phx.solver.solve_diffrax_ensemble(
        problem,
        save_times=jnp.asarray([0.1]),
        realization=_realization(32, sample_shape=(4,)),
        dt0=0.01,
    )
    large = phx.solver.solve_diffrax_ensemble(
        problem,
        save_times=jnp.asarray([0.1]),
        realization=_realization(32, sample_shape=(7,)),
        dt0=0.01,
    )

    assert jnp.array_equal(small.states, large.states[:4])
    assert jnp.array_equal(
        jr.key_data(small.realization.path_keys),
        jr.key_data(large.realization.path_keys[:4]),
    )


def test_antithetic_realization_pairs_terminal_values():
    realization = phx.stochastic.WienerRealization.antithetic(
        jr.key(33),
        (1,),
        support=(0.0, 1.0),
        num_pairs=8,
        tolerance=1e-3,
    )
    solution = phx.solver.solve_diffrax_ensemble(
        _brownian_problem(),
        save_times=jnp.asarray([1.0]),
        realization=realization,
        dt0=0.02,
    )
    terminal = solution.states[:, -1, 0].reshape((8, 2))

    assert jnp.allclose(jnp.sum(terminal, axis=1), 0.0, atol=2e-12)
    assert jnp.array_equal(
        jr.key_data(realization.path_keys[::2]),
        jr.key_data(realization.path_keys[1::2]),
    )


def test_multiple_wiener_terms_match_one_concatenated_term():
    drift = lambda t, state, args: jnp.zeros_like(state)
    split_problem = phx.solver.DifferentialProblem(
        drift,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=0.2,
        wiener_terms=(
            phx.solver.WienerTerm(
                "left",
                lambda t, state, args: jnp.asarray([[1.0]]),
                (1,),
                structure="additive",
            ),
            phx.solver.WienerTerm(
                "right",
                lambda t, state, args: jnp.asarray([[2.0]]),
                (1,),
                structure="additive",
            ),
        ),
    )
    combined_problem = phx.solver.DifferentialProblem(
        drift,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=0.2,
        wiener_terms=(
            phx.solver.WienerTerm(
                "combined",
                lambda t, state, args: jnp.asarray([[1.0, 2.0]]),
                (2,),
                structure="additive",
            ),
        ),
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(34),
        (2,),
        support=(0.0, 0.2),
        tolerance=1e-3,
    )

    split = phx.solver.solve_diffrax(
        split_problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        dt0=0.02,
    )
    combined = phx.solver.solve_diffrax(
        combined_problem,
        save_times=jnp.asarray([0.2]),
        realization=realization,
        dt0=0.02,
    )

    assert jnp.array_equal(split.states, combined.states)
    assert split.wiener_term_slices == {"left": (0, 1), "right": (1, 2)}


def test_solver_interpretation_and_levy_contracts_are_enforced():
    ito = _brownian_problem(interpretation="ito")
    stratonovich = _brownian_problem(interpretation="stratonovich")
    realization = _realization(35)

    with pytest.raises(ValueError, match="Itô problem"):
        phx.solver.solve_diffrax(
            ito,
            save_times=jnp.asarray([1.0]),
            realization=realization,
            solver=dfx.EulerHeun(),
            dt0=0.02,
        )
    with pytest.raises(ValueError, match="Stratonovich problem"):
        phx.solver.solve_diffrax(
            stratonovich,
            save_times=jnp.asarray([1.0]),
            realization=realization,
            solver=dfx.Euler(),
            dt0=0.02,
        )
    with pytest.raises(ValueError, match="explicitly marked"):
        phx.solver.solve_diffrax(
            ito,
            save_times=jnp.asarray([1.0]),
            realization=realization,
            solver=dfx.Tsit5(),
            dt0=0.02,
        )
    with pytest.raises(ValueError, match="requires AbstractSpaceTimeLevyArea"):
        phx.solver.solve_diffrax(
            _brownian_problem(structure="additive"),
            save_times=jnp.asarray([1.0]),
            realization=realization,
            solver=dfx.ShARK(),
            dt0=0.02,
        )
    higher_order = phx.solver.solve_diffrax(
        _brownian_problem(structure="additive"),
        save_times=jnp.asarray([1.0]),
        realization=_realization(35, levy_area="space_time"),
        solver=dfx.ShARK(),
        dt0=0.02,
    )
    assert jnp.all(jnp.isfinite(higher_order.states))


def test_additive_noise_can_use_either_interpretation_solver():
    problem = _brownian_problem(interpretation="ito", structure="additive")
    realization = _realization(36)
    ito = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([1.0]),
        realization=realization,
        solver=dfx.Euler(),
        dt0=0.02,
    )
    stratonovich = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([1.0]),
        realization=realization,
        solver=dfx.EulerHeun(),
        dt0=0.02,
    )

    assert jnp.array_equal(ito.states, stratonovich.states)


def test_diffrax_contract_rejects_invalid_problem_realization_and_save_configuration():
    deterministic = _geometric_problem()
    stochastic = _brownian_problem()
    scalar_realization = _realization(0)
    batch_realization = _realization(0, sample_shape=(2,))

    with pytest.raises(ValueError, match="requires t1 > t0"):
        phx.solver.DifferentialProblem(
            lambda t, state, args: state,
            jnp.asarray([1.0]),
            t0=1.0,
            t1=0.0,
        )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="strictly increasing"
    ):
        phx.solver.solve_diffrax(
            deterministic,
            save_times=jnp.asarray([0.5, 0.5]),
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="time interval"):
        phx.solver.solve_diffrax(deterministic, save_times=jnp.asarray([1.1]))
    with pytest.raises(TypeError, match="dense must be a bool"):
        phx.solver.solve_diffrax(
            deterministic,
            save_times=jnp.asarray([1.0]),
            dense=1,
        )
    with pytest.raises(ValueError, match="require a WienerRealization"):
        phx.solver.solve_diffrax(
            stochastic,
            save_times=jnp.asarray([1.0]),
            dt0=0.01,
        )
    with pytest.raises(ValueError, match="explicit dt0"):
        phx.solver.solve_diffrax(
            stochastic,
            save_times=jnp.asarray([1.0]),
            realization=scalar_realization,
        )
    with pytest.raises(ValueError, match="do not accept a WienerRealization"):
        phx.solver.solve_diffrax(
            deterministic,
            save_times=jnp.asarray([1.0]),
            realization=scalar_realization,
        )
    with pytest.raises(ValueError, match="requires a stochastic problem"):
        phx.solver.solve_diffrax_ensemble(
            deterministic,
            save_times=jnp.asarray([1.0]),
            realization=batch_realization,
            dt0=0.01,
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="strictly smaller"):
        phx.solver.solve_diffrax(
            stochastic,
            save_times=jnp.asarray([1.0]),
            realization=_realization(1, tolerance=0.01),
            dt0=0.01,
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="before the Wiener"):
        phx.solver.solve_diffrax(
            _brownian_problem(t0=0.0, t1=0.5),
            save_times=jnp.asarray([0.5]),
            realization=_realization(2, support=(0.1, 1.0)),
            dt0=0.01,
        )
    identified_problem = phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        wiener_terms=(
            phx.solver.WienerTerm(
                "identified",
                lambda t, state, args: jnp.ones((1, 1)),
                (1,),
                basis_id="basis-a",
            ),
        ),
    )
    with pytest.raises(ValueError, match="noise_id must match"):
        phx.solver.solve_diffrax(
            identified_problem,
            save_times=jnp.asarray([1.0]),
            realization=phx.stochastic.WienerRealization(
                jr.key(3),
                (1,),
                support=(0.0, 1.0),
                tolerance=1e-3,
                noise_id="basis-b",
            ),
            dt0=0.02,
        )


def test_wiener_term_validates_names_shapes_and_uniqueness():
    coefficient = lambda t, state, args: jnp.ones((1, 1))
    first = phx.solver.WienerTerm("same", coefficient, (1,))

    with pytest.raises(ValueError, match="non-empty"):
        phx.solver.WienerTerm("", coefficient, (1,))
    with pytest.raises(ValueError, match="coefficient must return shape"):
        phx.solver.DifferentialProblem(
            lambda t, state, args: state,
            jnp.asarray([1.0]),
            t0=0.0,
            t1=1.0,
            wiener_terms=(
                phx.solver.WienerTerm(
                    "wrong",
                    lambda t, state, args: jnp.ones((2,)),
                    (1,),
                ),
            ),
        )
    with pytest.raises(ValueError, match="names must be unique"):
        phx.solver.DifferentialProblem(
            lambda t, state, args: state,
            jnp.asarray([1.0]),
            t0=0.0,
            t1=1.0,
            wiener_terms=(first, first),
        )
