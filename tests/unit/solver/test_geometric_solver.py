#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
import pytest

import phydrax as phx


def _so_problem(*, rate=1.0, stochastic=False, interpretation="ito"):
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    generator = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    terms = ()
    if stochastic:
        terms = (
            phx.solver.WienerTerm(
                "rotation",
                lambda time, state, args: (state @ generator)[..., None],
                (1,),
                structure="commutative",
                basis_id="basis:so2-rotation",
            ),
        )
    return phx.solver.DifferentialProblem(
        lambda time, state, value: value * (1.0 + time) * (state @ generator),
        jnp.eye(2),
        t0=0.0,
        t1=1.0,
        args=jnp.asarray(rate),
        wiener_terms=terms,
        interpretation=interpretation,
        state_geometry=geometry,
    )


def _assert_so(values, *, atol=2e-8):
    products = jnp.swapaxes(values, -1, -2) @ values
    assert jnp.allclose(products, jnp.eye(values.shape[-1]), atol=atol)
    assert jnp.all(jnp.linalg.det(values) > 0.0)


def _spd_problem():
    geometry = phx.metrix.SymmetricPositiveDefiniteStateGeometry(2)
    forcing = jnp.array([[0.1, 0.04], [0.04, -0.05]])
    return phx.solver.DifferentialProblem(
        lambda time, state, args: forcing,
        jnp.array([[2.0, 0.3], [0.3, 1.0]]),
        t0=0.0,
        t1=1.0,
        state_geometry=geometry,
    )


def test_problem_validates_membership_and_rejects_ordinary_solver():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    with pytest.raises(Exception, match="outside state_geometry"):
        phx.solver.DifferentialProblem(
            lambda time, state, args: state,
            jnp.ones((2, 2)),
            t0=0.0,
            t1=1.0,
            state_geometry=geometry,
        )

    problem = _so_problem()
    with pytest.raises(ValueError, match="nontrivial state_geometry"):
        phx.solver.solve_diffrax(
            problem,
            save_times=jnp.asarray([1.0]),
            solver=dfx.Euler(),
            dt0=0.01,
        )
    with pytest.raises(ValueError, match="ConstantStepSize"):
        phx.solver.solve_diffrax(
            problem,
            save_times=jnp.asarray([1.0]),
            solver=phx.solver.RKMK(problem.state_geometry),
            stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6),
            dt0=0.01,
        )
    embedded = phx.metrix.EmbeddedStateGeometry(
        membership=lambda state: jnp.asarray(True),
        tangent_projection=lambda state, vector: vector,
        retraction=lambda state, tangent: state + tangent,
        geometry_id="state-geometry:test-embedded",
        retraction_method="test-addition",
    )
    with pytest.raises(ValueError, match="exact pullback"):
        phx.solver.RKMK(embedded)
    exact_embedded = phx.metrix.EmbeddedStateGeometry(
        membership=lambda state: jnp.asarray(True),
        tangent_projection=lambda state, vector: vector,
        retraction=lambda state, tangent: state + tangent,
        inverse_retraction=lambda state, point: point - state,
        retraction_pullback=lambda state, local, tangent: tangent,
        geometry_id="state-geometry:test-exact-embedded",
        retraction_method="test-addition",
    )
    assert exact_embedded.supports_exact_pullback
    assert isinstance(phx.solver.RKMK(exact_embedded), phx.solver.RKMK)


def test_euclidean_geometric_euler_agrees_with_ordinary_euler():
    geometry = phx.metrix.EuclideanStateGeometry()
    problem = phx.solver.DifferentialProblem(
        lambda time, state, rate: rate * state,
        jnp.array([2.0, -1.0]),
        t0=0.0,
        t1=0.5,
        args=jnp.asarray(0.3),
        state_geometry=geometry,
    )
    times = jnp.linspace(0.0, 0.5, 6)
    geometric = phx.solver.solve_diffrax(
        problem,
        save_times=times,
        solver=phx.solver.GeometricEuler(geometry),
        dt0=0.01,
    )
    ordinary = phx.solver.solve_diffrax(
        problem,
        save_times=times,
        solver=dfx.Euler(),
        dt0=0.01,
        stepsize_controller=dfx.ConstantStepSize(),
    )

    assert jnp.array_equal(geometric.states, ordinary.states)
    assert geometric.state_geometry_id == geometry.geometry_id
    assert ordinary.state_geometry_id == geometry.geometry_id
    assert geometric.solver_id == phx.solver.GeometricEuler(geometry).solver_id
    assert geometric.resolved_method == "euler:addition"
    assert ordinary.solver_id == "solver:diffrax:Euler"
    assert ordinary.resolved_method == "Euler"
    assert (
        phx.solver.solver_state_geometry(phx.solver.GeometricEuler(geometry)) is geometry
    )
    with pytest.raises(TypeError, match="geometric-solver contract"):
        phx.solver.solver_state_geometry(dfx.Euler())


def test_rkmk_so_dense_output_jit_gradient_and_convergence():
    times = jnp.asarray([0.0, 0.5, 1.0])

    def terminal(rate, dt):
        problem = _so_problem(rate=rate)
        solution = phx.solver.solve_diffrax(
            problem,
            save_times=times,
            solver=phx.solver.RKMK(problem.state_geometry),
            dt0=dt,
            dense=True,
        )
        return solution.states[-1], solution.evaluate(jnp.linspace(0.0, 1.0, 17))

    endpoint, dense = eqx.filter_jit(terminal)(jnp.asarray(0.7), 0.1)
    _assert_so(endpoint)
    _assert_so(dense)
    derivative = jax.grad(lambda rate: terminal(rate, 0.05)[0][1, 0])(jnp.asarray(0.7))
    assert jnp.allclose(
        derivative,
        1.5 * jnp.cos(1.5 * 0.7),
        rtol=2e-5,
        atol=2e-6,
    )

    geometry3 = phx.metrix.SpecialOrthogonalStateGeometry(3)
    first = jnp.array([[0.0, -0.7, 0.2], [0.7, 0.0, -0.3], [-0.2, 0.3, 0.0]])
    second = jnp.array([[0.0, 0.1, -0.4], [-0.1, 0.0, 0.6], [0.4, -0.6, 0.0]])
    field = lambda time, state, args: state @ (first + time * second)

    def noncommuting(dt):
        problem = phx.solver.DifferentialProblem(
            field,
            jnp.eye(3),
            t0=0.0,
            t1=1.0,
            state_geometry=geometry3,
        )
        return phx.solver.solve_diffrax(
            problem,
            save_times=jnp.asarray([1.0]),
            solver=phx.solver.RKMK(geometry3),
            dt0=dt,
        ).states[-1]

    reference_problem = phx.solver.DifferentialProblem(
        field,
        jnp.eye(3),
        t0=0.0,
        t1=1.0,
    )
    reference = phx.solver.solve_diffrax(
        reference_problem,
        save_times=jnp.asarray([1.0]),
        rtol=1e-11,
        atol=1e-13,
    ).states[-1]
    coarse_error = jnp.linalg.norm(noncommuting(0.2) - reference)
    fine_error = jnp.linalg.norm(noncommuting(0.1) - reference)
    assert coarse_error / fine_error > 8.0


def test_commutator_free_requires_shared_trivialization_and_spd_dense_rkmk():
    problem = _so_problem(rate=0.4)
    tableau = phx.solver.CommutatorFreeTableau(
        abscissae=(0.0, 1.0),
        stage_coefficients=((), (1.0,)),
        composition_coefficients=((0.5, 0.5),),
        order=2,
        tableau_id="tableau:test-cf2",
    )
    solver = phx.solver.CommutatorFreeSolver(
        problem.state_geometry,
        tableau=tableau,
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 0.5, 1.0]),
        solver=solver,
        dt0=0.05,
        dense=True,
    )
    _assert_so(solution.states)
    _assert_so(solution.evaluate(jnp.linspace(0.0, 1.0, 21)))
    assert solution.solver_id == solver.solver_id
    assert solution.resolved_method == solver.resolved_method
    assert (
        phx.solver.solver_state_geometry(solver).geometry_id == problem.state_geometry_id
    )

    spd_problem = _spd_problem()
    with pytest.raises(ValueError, match="shared-trivialization"):
        phx.solver.CommutatorFreeSolver(spd_problem.state_geometry)
    spd_solution = phx.solver.solve_diffrax(
        spd_problem,
        save_times=jnp.asarray([0.0, 0.5, 1.0]),
        solver=phx.solver.RKMK(spd_problem.state_geometry),
        dt0=0.05,
        dense=True,
    )
    spd_dense = spd_solution.evaluate(jnp.linspace(0.0, 1.0, 21))
    assert jnp.all(jnp.linalg.eigvalsh(spd_solution.states) > 0.0)
    assert jnp.all(jnp.linalg.eigvalsh(spd_dense) > 0.0)


def test_commutator_free_midpoint_has_second_order_on_noncommuting_so3_flow():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(3)
    first = jnp.array([[0.0, -0.7, 0.2], [0.7, 0.0, -0.3], [-0.2, 0.3, 0.0]])
    second = jnp.array([[0.0, 0.1, -0.4], [-0.1, 0.0, 0.6], [0.4, -0.6, 0.0]])

    def vector_field(time, state, args):
        del args
        return state @ (first + time * second)

    def terminal(step_size):
        problem = phx.solver.DifferentialProblem(
            vector_field,
            jnp.eye(3),
            t0=0.0,
            t1=1.0,
            state_geometry=geometry,
        )
        return phx.solver.solve_diffrax(
            problem,
            save_times=jnp.asarray([1.0]),
            solver=phx.solver.CommutatorFreeSolver(geometry),
            dt0=step_size,
        ).states[-1]

    reference = phx.solver.solve_diffrax(
        phx.solver.DifferentialProblem(
            vector_field,
            jnp.eye(3),
            t0=0.0,
            t1=1.0,
        ),
        save_times=jnp.asarray([1.0]),
        rtol=1e-11,
        atol=1e-13,
    ).states[-1]
    coarse = terminal(0.2)
    fine = terminal(0.1)

    _assert_so(jnp.stack((coarse, fine)))
    coarse_error = jnp.linalg.norm(coarse - reference)
    fine_error = jnp.linalg.norm(fine - reference)
    assert coarse_error / fine_error > 3.5


def test_geometric_event_uses_on_manifold_interpolation():
    problem = _so_problem()
    target = 0.4
    event_time = jnp.sqrt(1.0 + 2.0 * target) - 1.0
    event = dfx.Event(
        lambda t, y, args, **kwargs: y[1, 0] - jnp.sin(target),
        root_finder=optx.Newton(rtol=1e-8, atol=1e-8),
    )
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, event_time]),
        solver=phx.solver.RKMK(problem.state_geometry),
        dt0=0.1,
        event=event,
        dense=True,
    )

    assert bool(solution.event_mask)
    _assert_so(solution.states[solution.valid])
    _assert_so(solution.evaluate(jnp.asarray([0.2, event_time])))


def test_srkmk_spd_corrector_uses_base_local_retraction():
    geometry = phx.metrix.SymmetricPositiveDefiniteStateGeometry(2)
    base = jnp.array([[2.0, 0.35], [0.35, 1.1]])
    constant_field = jnp.array([[0.2, -0.08], [-0.08, 0.12]])
    control = dfx.LinearInterpolation(
        ts=jnp.asarray([0.0, 1.0]),
        ys=jnp.asarray([[0.0], [0.2]]),
    )
    terms = dfx.MultiTerm(
        dfx.ODETerm(lambda time, state, args: jnp.zeros_like(state)),
        dfx.ControlTerm(
            lambda time, state, args: constant_field[..., None],
            control,
        ),
    )
    solver = phx.solver.SRKMK(geometry)
    actual = solver.step(
        terms,
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        base,
        None,
        None,
        False,
    )[0]

    retraction = geometry.local_retraction(base)
    noise_increment = 0.2 * constant_field
    first = retraction.pullback(jnp.zeros_like(base), noise_increment)
    predictor = retraction.evaluate(first)
    corrected = retraction.pullback(
        first,
        geometry.project_tangent(predictor, noise_increment),
    )
    expected = retraction.evaluate(0.5 * (first + corrected))
    assert jnp.allclose(actual, expected)
    assert bool(geometry.contains(actual))


def test_srkmk_stratonovich_batch_preserves_so_and_rejects_ito():
    stratonovich = _so_problem(
        rate=0.0,
        stochastic=True,
        interpretation="stratonovich",
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(9),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(3,),
        tolerance=1e-3,
        noise_id=stratonovich.noise_id,
    )
    solution = phx.solver.solve_diffrax_ensemble(
        stratonovich,
        save_times=jnp.asarray([0.0, 0.5, 1.0]),
        realization=realization,
        solver=phx.solver.SRKMK(stratonovich.state_geometry),
        dt0=0.02,
        dense=True,
    )
    _assert_so(solution.states)
    _assert_so(solution.evaluate(jnp.asarray([0.1, 0.7])))
    assert solution.sample_shape == (3,)

    ito = _so_problem(rate=0.0, stochastic=True, interpretation="ito")
    with pytest.raises(ValueError, match="Itô geometry"):
        phx.solver.solve_diffrax(
            ito,
            save_times=jnp.asarray([1.0]),
            realization=phx.stochastic.WienerRealization(
                jr.key(10),
                (1,),
                support=(0.0, 1.0),
                noise_id=ito.noise_id,
            ),
            solver=phx.solver.SRKMK(ito.state_geometry),
            dt0=0.02,
        )
