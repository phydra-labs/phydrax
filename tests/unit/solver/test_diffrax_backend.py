import warnings
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
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
    assert solution.realization is not None
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
    assert small.realization is not None
    assert large.realization is not None

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
    invalid_dense: Any = 1
    with pytest.raises(TypeError, match="dense must be a bool"):
        phx.solver.solve_diffrax(
            deterministic,
            save_times=jnp.asarray([1.0]),
            dense=invalid_dense,
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


def test_complex_ode_packing_matches_explicit_real_system_dense_and_gradient():
    initial = jnp.asarray([1.0 + 0.25j], dtype=jnp.complex128)
    save_times = jnp.linspace(0.0, 0.6, 7)

    def solve_complex(real_rate):
        rate = real_rate + 0.7j
        problem = phx.solver.DifferentialProblem(
            lambda t, state, args: args["rate"] * state + 0.2 * jnp.conj(state),
            initial,
            t0=0.0,
            t1=0.6,
            args={"rate": rate},
        )
        return phx.solver.solve_diffrax(
            problem,
            save_times=save_times,
            dense=True,
            rtol=1e-9,
            atol=1e-11,
        )

    def solve_real(real_rate):
        def drift(time, state, args):
            del time, args
            real, imag = state
            return jnp.stack(
                (
                    (real_rate + 0.2) * real - 0.7 * imag,
                    0.7 * real + (real_rate - 0.2) * imag,
                )
            )

        problem = phx.solver.DifferentialProblem(
            drift,
            jnp.stack((jnp.real(initial), jnp.imag(initial))),
            t0=0.0,
            t1=0.6,
        )
        return phx.solver.solve_diffrax(
            problem,
            save_times=save_times,
            dense=True,
            rtol=1e-9,
            atol=1e-11,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Complex dtype support in Diffrax.*",
        )
        complex_solution = solve_complex(jnp.asarray(-0.4))
    real_solution = solve_real(jnp.asarray(-0.4))
    expected = jax.lax.complex(
        real_solution.states[:, 0],
        real_solution.states[:, 1],
    )
    queries = jnp.asarray([0.15, 0.45])
    expected_dense = jax.lax.complex(
        real_solution.evaluate(queries)[:, 0],
        real_solution.evaluate(queries)[:, 1],
    )
    gradient = jax.grad(
        lambda rate: jnp.sum(jnp.abs(solve_complex(rate).states[-1]) ** 2)
    )(jnp.asarray(-0.4))
    reference_gradient = jax.grad(
        lambda rate: jnp.sum(jnp.abs(solve_real(rate).states[-1]) ** 2)
    )(jnp.asarray(-0.4))
    evidence = complex_solution.temporal_evidence

    assert evidence is not None
    assert evidence.state_coordinates is not None
    assert evidence.state_coordinates.source_shape == (1,)
    assert evidence.state_coordinates.coordinate_shape == (2, 1)
    assert evidence.state_coordinates.source_dtype == "complex128"
    assert evidence.state_coordinates.coordinate_dtype == "float64"
    assert complex_solution.states.shape == (7, 1)
    assert complex_solution.states.dtype == jnp.complex128
    assert jnp.allclose(complex_solution.states, expected, rtol=2e-8, atol=2e-9)
    assert jnp.allclose(
        complex_solution.evaluate(queries),
        expected_dense,
        rtol=2e-8,
        atol=2e-9,
    )
    assert jnp.allclose(gradient, reference_gradient, rtol=2e-7, atol=2e-8)


def test_complex_sde_packing_matches_real_system_pathwise_and_in_ensemble():
    complex_term = phx.solver.WienerTerm(
        "imaginary",
        lambda t, state, args: 1j * jnp.ones(state.shape + (1,)),
        (1,),
        structure="additive",
    )
    real_term = phx.solver.WienerTerm(
        "imaginary",
        lambda t, state, args: jnp.broadcast_to(
            jnp.asarray([[[0.0]], [[1.0]]]),
            state.shape + (1,),
        ),
        (1,),
        structure="additive",
    )
    complex_problem = phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.asarray([0.0 + 0.0j]),
        t0=0.0,
        t1=0.2,
        wiener_terms=(complex_term,),
    )
    real_problem = phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.zeros((2, 1)),
        t0=0.0,
        t1=0.2,
        wiener_terms=(real_term,),
    )
    scalar_realization = _realization(
        81,
        support=(0.0, 0.2),
        tolerance=1e-4,
    )
    save_times = jnp.asarray([0.05, 0.1, 0.2])

    complex_solution = phx.solver.solve_diffrax(
        complex_problem,
        save_times=save_times,
        realization=scalar_realization,
        solver=dfx.Euler(),
        dt0=0.01,
    )
    real_solution = phx.solver.solve_diffrax(
        real_problem,
        save_times=save_times,
        realization=scalar_realization,
        solver=dfx.Euler(),
        dt0=0.01,
    )
    expected = jax.lax.complex(
        real_solution.states[:, 0],
        real_solution.states[:, 1],
    )

    ensemble_realization = _realization(
        82,
        sample_shape=(2, 3),
        support=(0.0, 0.2),
        tolerance=1e-4,
    )
    complex_ensemble = phx.solver.solve_diffrax_ensemble(
        complex_problem,
        save_times=save_times,
        realization=ensemble_realization,
        solver=dfx.Euler(),
        dt0=0.01,
        dense=True,
    )
    real_ensemble = phx.solver.solve_diffrax_ensemble(
        real_problem,
        save_times=save_times,
        realization=ensemble_realization,
        solver=dfx.Euler(),
        dt0=0.01,
        dense=True,
    )
    expected_ensemble = jax.lax.complex(
        real_ensemble.states[..., 0, :],
        real_ensemble.states[..., 1, :],
    )
    queries = jnp.asarray([0.025, 0.175])
    real_dense = real_ensemble.evaluate(queries)
    expected_dense = jax.lax.complex(
        real_dense[..., 0, :],
        real_dense[..., 1, :],
    )

    assert jnp.array_equal(complex_solution.states, expected)
    assert complex_ensemble.states.shape == (2, 3, 3, 1)
    assert jnp.array_equal(complex_ensemble.states, expected_ensemble)
    assert jnp.array_equal(complex_ensemble.evaluate(queries), expected_dense)
    assert complex_ensemble.temporal_evidence.state_coordinates is not None
    assert (
        complex_ensemble.temporal_evidence.state_coordinates.norm_relation == "isometry"
    )


def test_complex_packing_wraps_events_and_split_dynamics():
    event = dfx.Event(
        lambda t, state, args, **kwargs: jnp.real(state[0]) - 0.5,
        root_finder=optx.Newton(rtol=1e-9, atol=1e-9),
    )
    problem = phx.solver.DifferentialProblem(
        lambda t, state, args: -state,
        jnp.asarray([1.0 + 0.0j]),
        t0=0.0,
        t1=2.0,
    )
    event_solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 0.6, 0.8]),
        event=event,
        dt0=0.05,
    )
    split = phx.solver.SplitDifferentialProblem(
        lambda t, state, args: 1j * state,
        lambda t, state, args: -9.0 * state,
        jnp.asarray([1.0 + 0.0j]),
        t0=0.0,
        t1=0.2,
    )
    split_solution = phx.solver.solve_diffrax(
        split,
        save_times=jnp.asarray([0.2]),
        rtol=1e-8,
        atol=1e-10,
    )

    assert event_solution.event_terminated
    assert jnp.isfinite(event_solution.states[:2]).all()
    assert jnp.isinf(event_solution.states[-1]).all()
    assert split_solution.temporal_evidence.state_coordinates is not None
    assert split_solution.temporal_evidence.equation_form == "additive-ode"
    assert jnp.allclose(
        split_solution.states[-1, 0],
        jnp.exp((-9.0 + 1j) * 0.2),
        rtol=3e-7,
        atol=3e-8,
    )


def test_complex_state_policy_precision_and_real_bypass_are_explicit():
    complex_problem = phx.solver.DifferentialProblem(
        lambda t, state, args: 1j * state,
        jnp.asarray([1.0 + 0.0j]),
        t0=0.0,
        t1=0.1,
    )
    with pytest.raises(ValueError, match="strategy"):
        phx.solver.DiffraxComplexStatePolicy("typo")
    with pytest.raises(ValueError, match="rejected"):
        phx.solver.solve_diffrax(
            complex_problem,
            save_times=jnp.asarray([0.1]),
            complex_state_policy=phx.solver.DiffraxComplexStatePolicy("reject"),
        )
    with pytest.raises(ValueError, match="complex output dtype"):
        phx.solver.solve_diffrax(
            complex_problem,
            save_times=jnp.asarray([0.1]),
            precision=phx.solver.TemporalPrecisionPolicy(output_dtype=jnp.float64),
        )
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    geometric_problem = phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.eye(2, dtype=jnp.complex128),
        t0=0.0,
        t1=0.1,
        state_geometry=geometry,
    )
    with pytest.raises(ValueError, match="Nontrivial state geometry"):
        phx.solver.solve_diffrax(
            geometric_problem,
            save_times=jnp.asarray([0.1]),
            dt0=0.01,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Complex dtype support in Diffrax.*",
        )
        native = phx.solver.solve_diffrax(
            complex_problem,
            save_times=jnp.asarray([0.1]),
            complex_state_policy=phx.solver.DiffraxComplexStatePolicy("native"),
        )
    assert native.temporal_evidence.state_coordinates is None

    real_problem = _geometric_problem()
    implicit_default = phx.solver.solve_diffrax(
        real_problem,
        save_times=jnp.asarray([1.0]),
    )
    explicit_default = phx.solver.solve_diffrax(
        real_problem,
        save_times=jnp.asarray([1.0]),
        complex_state_policy=phx.solver.DiffraxComplexStatePolicy(),
    )
    assert implicit_default.temporal_evidence.state_coordinates is None
    assert explicit_default.temporal_evidence.state_coordinates is None
    assert (
        implicit_default.temporal_evidence.configuration_id
        == explicit_default.temporal_evidence.configuration_id
    )


def test_diagonal_wiener_ensemble_preserves_distinct_initial_states():
    dimension = 2
    dense = phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.zeros((dimension,)),
        t0=0.0,
        t1=0.1,
        wiener_terms=(
            phx.solver.WienerTerm(
                "noise",
                lambda t, state, args: 0.3 * jnp.eye(dimension),
                (dimension,),
                structure="additive",
            ),
        ),
    )
    diagonal_term = phx.solver.WienerTerm(
        "noise",
        lambda t, state, args: jnp.full(state.shape, 0.3),
        (dimension,),
        structure="additive",
        representation="diagonal",
    )
    diagonal = phx.solver.DifferentialProblem(
        lambda t, state, args: jnp.zeros_like(state),
        jnp.zeros((dimension,)),
        t0=0.0,
        t1=0.1,
        wiener_terms=(diagonal_term,),
    )
    realization = phx.stochastic.WienerRealization.independent(
        jr.key(90),
        (dimension,),
        support=(0.0, 0.1),
        sample_shape=(3,),
        tolerance=1e-4,
    )
    initial = jnp.asarray([[0.0, 0.0], [1.0, -1.0], [3.0, 2.0]])
    dense_solution = phx.solver.solve_diffrax_ensemble(
        dense,
        save_times=jnp.asarray([0.1]),
        realization=realization,
        initial_states=initial,
        solver=dfx.Euler(),
        dt0=0.01,
    )
    diagonal_solution = phx.solver.solve_diffrax_ensemble(
        diagonal,
        save_times=jnp.asarray([0.1]),
        realization=realization,
        initial_states=initial,
        solver=dfx.Euler(),
        dt0=0.01,
    )

    assert jnp.array_equal(dense_solution.states, diagonal_solution.states)
    assert jnp.array_equal(
        diagonal_solution.states[:, 0] - initial,
        dense_solution.states[:, 0] - initial,
    )
    with pytest.raises(ValueError, match="no implicit dense matrix"):
        diagonal_term.coefficient_matrix(0.0, jnp.zeros((dimension,)))
    with pytest.raises(ValueError, match="initial_states must have shape"):
        phx.solver.solve_diffrax_ensemble(
            diagonal,
            save_times=jnp.asarray([0.1]),
            realization=realization,
            initial_states=jnp.zeros((2, dimension)),
            dt0=0.01,
        )


def test_prepared_real_coordinate_tree_keeps_pytree_callbacks_public_and_backend_real():
    initial = {
        "z": jnp.asarray([1.0 + 0.5j], dtype=jnp.complex128),
        "x": jnp.asarray([2.0], dtype=jnp.float64),
    }
    complex_map = phx.linalg.ComplexCartesianCoordinates(
        phx.linalg.ArraySpace((1,), dtype=jnp.complex128)
    )
    coordinates = phx.linalg.prepare_real_coordinate_tree(
        initial,
        {"z": complex_map, "x": None},
    )
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: {
            "z": jnp.conj(state["z"]),
            "x": -state["x"],
        },
        initial,
        t0=0.0,
        t1=0.2,
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 0.2]),
        state_coordinates=coordinates,
        dense=True,
    )
    dense = solution.evaluate(jnp.asarray([0.1]))

    assert solution.states["z"].dtype == jnp.complex128
    assert solution.states["x"].dtype == jnp.float64
    assert dense["z"].shape == (1, 1)
    assert solution.temporal_evidence.state_coordinates.evidence_id == (
        coordinates.evidence.evidence_id
    )


def test_real_coordinate_tree_preserves_noncomplex_argument_leaves():
    initial = {"x": jnp.asarray([0.0], dtype=jnp.float64)}
    coordinates = phx.linalg.prepare_real_coordinate_tree(initial, {"x": None})
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: {
            "x": jnp.where(args["enabled"], args["rate"], 0.0)
        },
        initial,
        t0=0.0,
        t1=0.2,
        args={
            "enabled": jnp.asarray(True),
            "rate": jnp.asarray([2.0], dtype=jnp.float64),
        },
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 0.2]),
        solver=dfx.Euler(),
        dt0=0.01,
        state_coordinates=coordinates,
    )

    assert jnp.allclose(solution.states["x"][-1], 0.4)
