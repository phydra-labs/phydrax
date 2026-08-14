import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


_GENERATOR = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])


def _rotation(angle):
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return jnp.asarray([[cosine, -sine], [sine, cosine]])


def _so_delay_problem(*, rate=0.7, delay=0.2, t1=0.8):
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)

    def history(time, parameter):
        return _rotation(parameter * time)

    def drift(time, state, memory, parameter):
        del time
        angular_rate = parameter * (1.0 + 0.15 * memory["past"][1, 0])
        return angular_rate * (state @ _GENERATOR)

    return phx.solver.DelayDifferentialProblem(
        drift,
        history,
        (phx.solver.ConstantDelay("past", delay),),
        t0=0.0,
        t1=t1,
        args=jnp.asarray(rate),
        state_geometry=geometry,
        problem_id="geometric-delay:so2",
    )


def _assert_so2(values, *, atol=2e-5):
    products = jnp.swapaxes(values, -1, -2) @ values
    assert jnp.allclose(products, jnp.eye(2), atol=atol)
    assert jnp.all(jnp.linalg.det(values) > 0.0)


def _geometric_solver(name, geometry):
    if name == "euler":
        return phx.solver.GeometricEuler(geometry)
    if name == "rkmk-midpoint":
        return phx.solver.RKMK(geometry, method="midpoint")
    if name == "rkmk-rk4":
        return phx.solver.RKMK(geometry, method="rk4")
    if name == "commutator-free":
        return phx.solver.CommutatorFreeSolver(geometry)
    if name == "deterministic-srkmk":
        return phx.solver.SRKMK(geometry)
    raise AssertionError(name)


@pytest.mark.parametrize(
    "solver_name",
    (
        "euler",
        "rkmk-midpoint",
        "rkmk-rk4",
        "commutator-free",
        "deterministic-srkmk",
    ),
)
def test_fixed_geometric_delay_solvers_preserve_so_and_dense_history(solver_name):
    problem = _so_delay_problem()
    solver = _geometric_solver(solver_name, problem.state_geometry)
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.linspace(0.0, 0.8, 9),
        solver=solver,
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.04,
        dense=True,
        max_steps=128,
    )

    off_grid = solution.evaluate(jnp.asarray([0.013, 0.177, 0.333, 0.619, 0.791]))
    _assert_so2(solution.states)
    _assert_so2(off_grid)
    assert jnp.all(jax.vmap(problem.state_geometry.contains)(off_grid))
    assert solution.solver_id == solver.solver_id
    assert solution.resolved_method == solver.resolved_method
    assert solution.metadata["state_geometry_id"] == problem.state_geometry_id
    assert solution.stats["controller_mode"] == "fixed"
    assert solution.stats["stage_abscissae"] == solver.stage_abscissae
    assert solution.interpolation.history.computed_history.interpolation_cls is (
        phx.solver.GeometricLocalInterpolation
    )


def test_geometric_stage_contract_is_static_explicit_and_causal_bound_is_closed():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    euler = phx.solver.GeometricEuler(geometry)
    midpoint = phx.solver.RKMK(geometry, method="midpoint")
    rk4 = phx.solver.RKMK(geometry, method="rk4")
    commutator_free = phx.solver.CommutatorFreeSolver(geometry)
    srkmk = phx.solver.SRKMK(geometry)

    assert euler.stage_abscissae == (0.0,)
    assert midpoint.stage_abscissae == (0.0, 0.5)
    assert rk4.stage_abscissae == (0.0, 0.5, 0.5, 1.0)
    assert commutator_free.stage_abscissae == (0.0, 1.0)
    assert srkmk.stage_abscissae == (0.0, 1.0)
    for solver in (euler, midpoint, rk4, commutator_free, srkmk):
        assert solver.causal_stage_extent > 0.0
        assert max(solver.stage_abscissae) <= solver.causal_stage_extent

    problem = _so_delay_problem(delay=0.2, t1=0.4)
    causal = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.4]),
        solver=phx.solver.RKMK(problem.state_geometry, method="midpoint"),
        dt0=0.4,
        initial_discontinuities=(),
        max_steps=8,
    )
    _assert_so2(causal.states)
    assert jnp.asarray(causal.stats["maximum_causal_step"]) == jnp.asarray(0.4)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="causal delay step bound",
    ):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.4]),
            solver=phx.solver.RKMK(problem.state_geometry, method="midpoint"),
            dt0=0.4001,
            initial_discontinuities=(),
            max_steps=8,
        )


def test_euclidean_matrix_geometric_delay_matches_euler_and_rkmk_converges():
    geometry = phx.metrix.EuclideanStateGeometry()
    base = jnp.asarray([[1.0, -0.3], [0.2, 1.7]])
    rate = jnp.asarray(2.0)
    delay = 0.5

    def history(time, parameter):
        return jnp.exp(parameter * time) * base

    def drift(time, state, memory, parameter):
        del time, state
        return parameter * jnp.exp(parameter * delay) * memory["past"]

    problem = phx.solver.DelayDifferentialProblem(
        drift,
        history,
        (phx.solver.ConstantDelay("past", delay),),
        t0=0.0,
        t1=0.4,
        args=rate,
        state_geometry=geometry,
    )
    geometric_euler = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.linspace(0.0, 0.4, 5),
        solver=phx.solver.GeometricEuler(geometry),
        dt0=0.1,
        max_steps=16,
    )
    ordinary_euler = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.linspace(0.0, 0.4, 5),
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.1,
        max_steps=16,
    )
    assert geometric_euler.states.shape == (5, 2, 2)
    assert jnp.allclose(geometric_euler.states, ordinary_euler.states)

    exact = jnp.exp(rate * 0.4) * base

    def error(dt):
        terminal = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.4]),
            solver=phx.solver.RKMK(geometry),
            dt0=dt,
            max_steps=16,
        ).states[0]
        return jnp.linalg.norm(terminal - exact)

    coarse_error = error(0.2)
    fine_error = error(0.1)
    assert fine_error < coarse_error
    assert coarse_error / fine_error > 8.0


def test_spd_geometric_delay_keeps_embedded_state_and_off_grid_history_positive():
    geometry = phx.metrix.SymmetricPositiveDefiniteStateGeometry(2)
    base = jnp.asarray([[2.0, 0.25], [0.25, 1.1]])

    def history(time, args):
        del time, args
        return base

    def drift(time, state, memory, args):
        del time, args
        return 0.04 * (state + memory["past"])

    problem = phx.solver.DelayDifferentialProblem(
        drift,
        history,
        (phx.solver.ConstantDelay("past", 0.25),),
        t0=0.0,
        t1=0.75,
        state_geometry=geometry,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.0, 0.25, 0.5, 0.75]),
        solver=phx.solver.RKMK(geometry),
        dt0=0.05,
        dense=True,
        max_steps=64,
    )
    dense = solution.evaluate(jnp.asarray([0.031, 0.213, 0.417, 0.699]))

    assert jnp.all(jnp.linalg.eigvalsh(solution.states) > 0.0)
    assert jnp.all(jnp.linalg.eigvalsh(dense) > 0.0)
    assert jnp.all(jax.vmap(geometry.contains)(dense))


def test_geometric_delay_is_jittable_vectorizable_and_differentiable():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)

    def terminal(rate):
        problem = _so_delay_problem(rate=rate, delay=0.2, t1=0.4)
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.4]),
            solver=phx.solver.RKMK(geometry),
            dt0=0.04,
            max_steps=32,
        ).states[0, 1, 0]

    rate = jnp.asarray(0.6)
    direct = terminal(rate)
    compiled = eqx.filter_jit(terminal)(rate)
    derivative = jax.grad(terminal)(rate)
    batched = jax.vmap(terminal)(jnp.asarray([0.4, 0.6, 0.8]))

    assert jnp.allclose(compiled, direct)
    assert jnp.isfinite(derivative)
    assert jnp.abs(derivative) > 0.05
    assert batched.shape == (3,)
    assert jnp.all(jnp.isfinite(batched))
    assert jnp.all(jnp.diff(batched) > 0.0)


def test_geometric_delay_rejects_solver_geometry_drift_history_and_controller_mismatch():
    problem = _so_delay_problem(t1=0.4)
    times = jnp.asarray([0.4])

    with pytest.raises(ValueError, match="nontrivial state_geometry"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            solver=dfx.Euler(),
            dt0=0.05,
        )
    with pytest.raises(ValueError, match="same state_geometry_id"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            solver=phx.solver.GeometricEuler(
                phx.metrix.SpecialOrthogonalStateGeometry(3)
            ),
            dt0=0.05,
        )

    no_geometry = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: jnp.ones((2, 2)),
        (phx.solver.ConstantDelay("past", 0.2),),
        t0=0.0,
        t1=0.4,
    )
    with pytest.raises(ValueError, match="declare state_geometry"):
        phx.solver.solve_diffrax_delay(
            no_geometry,
            save_times=times,
            solver=phx.solver.GeometricEuler(phx.metrix.EuclideanStateGeometry()),
            dt0=0.05,
        )
    with pytest.raises(ValueError, match="explicit fixed dt0"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            solver=phx.solver.RKMK(problem.state_geometry),
        )
    with pytest.raises(ValueError, match="ConstantStepSize"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            solver=phx.solver.RKMK(problem.state_geometry),
            stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6),
            dt0=0.05,
        )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="initial state is outside state_geometry",
    ):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: jnp.zeros_like(state),
            lambda time, args: jnp.ones((2, 2)),
            (phx.solver.ConstantDelay("past", 0.2),),
            t0=0.0,
            t1=0.4,
            state_geometry=problem.state_geometry,
        )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="tangent-compatible",
    ):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: jnp.ones_like(state),
            lambda time, args: jnp.eye(2),
            (phx.solver.ConstantDelay("past", 0.2),),
            t0=0.0,
            t1=0.4,
            state_geometry=problem.state_geometry,
        )

    def invalid_prehistory(time, args):
        del args
        return jnp.where(time >= 0.0, jnp.eye(2), jnp.ones((2, 2)))

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="delayed history value lies outside state_geometry",
    ):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: state @ _GENERATOR,
            invalid_prehistory,
            (phx.solver.ConstantDelay("past", 0.2),),
            t0=0.0,
            t1=0.4,
            state_geometry=problem.state_geometry,
        )


def test_non_euclidean_distributed_and_neutral_delays_require_geometry_maps():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    history = lambda time, args: jnp.eye(2)
    tangent_drift = lambda time, state, memory, args: state @ _GENERATOR

    distributed = phx.solver.DistributedDelay(
        "spread",
        lambda time, lag, state, args: jnp.asarray(1.0),
        (0.1, 0.2),
    )
    with pytest.raises(ValueError, match="explicit reducer"):
        phx.solver.DelayDifferentialProblem(
            tangent_drift,
            history,
            (distributed,),
            t0=0.0,
            t1=0.4,
            state_geometry=geometry,
        )

    neutral = phx.solver.DerivativeDelay(
        "velocity",
        phx.solver.ConstantDelay("point", 0.2),
    )
    with pytest.raises(ValueError, match="tangent transport"):
        phx.solver.DelayDifferentialProblem(
            tangent_drift,
            history,
            (neutral,),
            t0=0.0,
            t1=0.4,
            history_derivative=lambda time, args: jnp.zeros((2, 2)),
            state_geometry=geometry,
        )

    transported_neutral = phx.solver.DerivativeDelay(
        "transported-velocity",
        phx.solver.ConstantDelay("point", 0.2),
        transport=lambda delayed, current, derivative, args: current @ _GENERATOR,
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="not tangent at the delayed state",
    ):
        phx.solver.DelayDifferentialProblem(
            tangent_drift,
            history,
            (transported_neutral,),
            t0=0.0,
            t1=0.4,
            history_derivative=lambda time, args: jnp.ones((2, 2)),
            state_geometry=geometry,
        )

    euclidean = phx.metrix.EuclideanStateGeometry()
    euclidean_problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["velocity"],
        history,
        (neutral,),
        t0=0.0,
        t1=0.2,
        history_derivative=lambda time, args: jnp.zeros((2, 2)),
        state_geometry=euclidean,
    )
    solution = phx.solver.solve_diffrax_delay(
        euclidean_problem,
        save_times=jnp.asarray([0.2]),
        solver=phx.solver.GeometricEuler(euclidean),
        dt0=0.1,
        max_steps=8,
    )
    assert jnp.allclose(solution.states[0], jnp.eye(2))


def test_stratonovich_geometric_delay_preserves_manifold_and_replays_path():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    noise = phx.solver.DelayWienerTerm(
        "rotation",
        lambda time, state, memory, args: (
            0.2 * (1.0 + 0.1 * memory["past"][1, 0]) * (state @ _GENERATOR)
        )[..., None],
        (1,),
        structure="commutative",
        basis_id="basis:geometric-delay-so2",
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: 0.1 * (state @ _GENERATOR),
        lambda time, args: jnp.eye(2),
        (phx.solver.ConstantDelay("past", 0.1),),
        t0=0.0,
        t1=0.2,
        wiener_terms=(noise,),
        interpretation="stratonovich",
        state_geometry=geometry,
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(20),
        (1,),
        support=(0.0, 0.2),
        tolerance=1e-5,
        noise_id=problem.noise_id,
    )
    common = {
        "save_times": jnp.linspace(0.0, 0.2, 5),
        "realization": realization,
        "dt0": 0.025,
        "dense": True,
        "max_steps": 32,
    }

    solution = phx.solver.solve_diffrax_delay(problem, **common)
    replay = phx.solver.solve_diffrax_delay(
        problem,
        solver=phx.solver.SRKMK(geometry),
        **common,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=common["save_times"],
        realization=realization,
        solver=phx.solver.SRKMK(geometry),
        dt0=0.025,
        dense=True,
        max_steps_per_segment=3,
    )
    off_grid = solution.evaluate(jnp.asarray([0.037, 0.123, 0.191]))
    segmented_off_grid = segmented.evaluate(jnp.asarray([0.037, 0.123, 0.191]))

    _assert_so2(solution.states)
    _assert_so2(off_grid)
    assert jnp.array_equal(solution.states, replay.states)
    assert jnp.array_equal(solution.states, segmented.states)
    assert jnp.array_equal(off_grid, segmented_off_grid)
    assert segmented.stats["num_segments"] > 1
    assert solution.solver_name == "SRKMK"
    assert solution.solver_id == replay.solver_id
    assert solution.stats["continuous_extension"] == "srkmk-wiener-path"


def test_stratonovich_geometric_advanced_memory_replays_all_history_modes():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)

    def distributed_reducer(
        time,
        state,
        nodes,
        weights,
        kernels,
        delayed_values,
        args,
    ):
        del time, nodes, kernels, args
        local_values = jax.vmap(lambda point: geometry.inverse_retract(state, point))(
            delayed_values
        )
        local_mean = jnp.tensordot(
            weights,
            local_values,
            axes=((0,), (0,)),
        ) / jnp.sum(weights)
        return geometry.retract(state, local_mean)

    state_delay = phx.solver.StateDependentDelay(
        "past",
        lambda time, state, args: jnp.asarray(0.1),
        minimum_delay=0.1,
        maximum_delay=0.1,
    )
    distributed_delay = phx.solver.DistributedDelay(
        "spread",
        lambda time, lag, state, args: jnp.asarray(1.0),
        (0.08, 0.12),
        quadrature=phx.integration.GaussLegendreRule(4),
        reducer=distributed_reducer,
    )
    noise = phx.solver.DelayWienerTerm(
        "rotation",
        lambda time, state, memory, args: (
            0.2 * (1.0 + 0.05 * memory["spread"][1, 0]) * (state @ _GENERATOR)
        )[..., None],
        (1,),
        structure="commutative",
        basis_id="basis:geometric-advanced-delay",
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: (
            0.1 * (1.0 + 0.05 * memory["past"][1, 0]) * (state @ _GENERATOR)
        ),
        lambda time, args: jnp.eye(2),
        (state_delay, distributed_delay),
        t0=0.0,
        t1=0.24,
        wiener_terms=(noise,),
        interpretation="stratonovich",
        state_geometry=geometry,
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(9),
        (1,),
        support=(0.0, 0.24),
        tolerance=1e-5,
        noise_id=problem.noise_id,
    )
    common = {
        "save_times": jnp.linspace(0.0, 0.24, 7),
        "realization": realization,
        "dt0": 0.02,
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
        max_steps_per_segment=6,
        **common,
    )
    query = jnp.asarray([0.13, 0.19, 0.23])

    _assert_so2(full.states)
    _assert_so2(rolling.evaluate(query))
    _assert_so2(segmented.evaluate(query))
    assert jnp.array_equal(rolling.states, full.states)
    assert jnp.allclose(segmented.states, full.states, rtol=0.0, atol=2e-14)
    assert jnp.allclose(
        segmented.evaluate(query),
        full.evaluate(query),
        rtol=0.0,
        atol=2e-14,
    )
    assert segmented.stats["state_dependent_tracking"] == (
        "first-order-pathwise-untracked"
    )
    assert full.metadata["distributed_delay_quadrature"][0]["node_count"] == 4


def test_geometric_delay_rejects_stochastic_ito_geometry():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    noise = phx.solver.DelayWienerTerm(
        "rotation",
        lambda time, state, memory, args: (state @ _GENERATOR)[..., None],
        (1,),
        structure="commutative",
        basis_id="basis:geometric-delay-so2",
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: state @ _GENERATOR,
        lambda time, args: jnp.eye(2),
        (phx.solver.ConstantDelay("past", 0.2),),
        t0=0.0,
        t1=0.4,
        wiener_terms=(noise,),
        interpretation="ito",
        state_geometry=geometry,
    )
    with pytest.raises(ValueError, match="Itô geometry"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.4]),
            realization=phx.stochastic.WienerRealization(
                jr.key(21),
                (1,),
                support=(0.0, 0.4),
                noise_id=problem.noise_id,
            ),
            solver=phx.solver.SRKMK(geometry),
            dt0=0.05,
        )


def test_stratonovich_geometric_delay_rejects_normal_diffusion():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    noise = phx.solver.DelayWienerTerm(
        "normal",
        lambda time, state, memory, args: jnp.ones((2, 2, 1)),
        (1,),
        structure="commutative",
        basis_id="basis:non-tangent-so2",
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: jnp.eye(2),
        (phx.solver.ConstantDelay("past", 0.1),),
        t0=0.0,
        t1=0.1,
        wiener_terms=(noise,),
        interpretation="stratonovich",
        state_geometry=geometry,
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(22),
        (1,),
        support=(0.0, 0.1),
        noise_id=problem.noise_id,
    )

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="diffusion must be tangent-compatible",
    ):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.1]),
            realization=realization,
            dt0=0.05,
            max_steps=8,
        )


def test_commutator_free_tableau_rejects_noncausal_stage_abscissa():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        phx.solver.CommutatorFreeTableau(
            abscissae=(0.0, -0.5),
            stage_coefficients=((), (1.0,)),
            composition_coefficients=((0.5, 0.5),),
            order=2,
            tableau_id="tableau:invalid-negative-stage",
        )
