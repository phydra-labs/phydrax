import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_recursive_lift_uses_davie_bracket_orientation_and_explicit_fields():
    left = jnp.asarray([[0.0, 1.0], [0.0, 0.0]])
    right = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    state = jnp.asarray([0.7, -0.4])
    basis = phx.stochastic.PrimitiveBasis(2, 2)

    def vector_fields(time, value, args):
        del time, args
        return jnp.stack((left @ value, right @ value), axis=-1)

    lifted = phx.solver.lift_rough_vector_fields(
        vector_fields, basis, jnp.asarray(0.0), state, None
    )
    expected_bracket = (right @ left - left @ right) @ state
    explicit = phx.solver.lift_rough_vector_fields(
        vector_fields,
        basis,
        jnp.asarray(0.0),
        state,
        None,
        explicit_fields=lambda time, value, args: lifted,
    )

    assert basis.words == ((0,), (1,), (0, 1))
    assert jnp.allclose(lifted[..., 2], expected_bracket)
    assert jnp.allclose(explicit, lifted)


def test_rough_solver_ids_are_stable_and_resolve_numerical_configuration():
    default = phx.solver.LogODE()
    repeated = phx.solver.LogODE()
    loose = phx.solver.LogODE(stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-6))
    stepped = phx.solver.LogODE(dt0=0.05)
    limited = phx.solver.LogODE(max_steps=32)
    inner_euler = phx.solver.LogODE(
        ode_solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.1,
    )
    linear_default = phx.solver.LinearLogODE((jnp.eye(2),))
    linear_repeated = phx.solver.LinearLogODE((2.0 * jnp.eye(2),))
    linear_short = phx.solver.LinearLogODE(
        (jnp.eye(2),),
        matrix_function_policy=phx.linalg.MatrixFunctionPolicy(
            "arnoldi", max_dimension=4
        ),
    )

    assert default.solver_id == repeated.solver_id
    assert default.solver_id != loose.solver_id
    assert default.solver_id != stepped.solver_id
    assert default.solver_id != limited.solver_id
    assert default.solver_id != inner_euler.solver_id
    assert linear_default.solver_id == linear_repeated.solver_id
    assert linear_default.solver_id != linear_short.solver_id
    assert (
        len(
            {
                phx.solver.RoughEuler().solver_id,
                phx.solver.Davie().solver_id,
                default.solver_id,
                linear_default.solver_id,
            }
        )
        == 4
    )


def test_general_and_linear_logode_agree_for_noncommuting_linear_system():
    left = jnp.asarray([[0.0, 1.0], [0.0, 0.0]])
    right = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    times = jnp.asarray([0.0, 0.4, 1.0])
    values = jnp.asarray([[0.0, 0.0], [0.7, 0.0], [0.7, -0.5]])
    control = phx.stochastic.LogSignatureControl.from_values(
        times, values, depth=3, coarse_indices=(0, 2)
    )

    def vector_fields(time, state, args):
        del time, args
        return jnp.stack((left @ state, right @ state), axis=-1)

    lifted_matrices = []
    for word, children in zip(
        control.primitive_basis.words, control.primitive_basis.children
    ):
        if children is None:
            lifted_matrices.append((left, right)[word[0]])
        else:
            left_index, right_index = children
            lifted_matrices.append(
                lifted_matrices[right_index] @ lifted_matrices[left_index]
                - lifted_matrices[left_index] @ lifted_matrices[right_index]
            )

    def explicit_fields(time, state, args):
        del time, args
        return jnp.stack(tuple(matrix @ state for matrix in lifted_matrices), axis=-1)

    problem = phx.solver.RoughDifferentialProblem(
        vector_fields,
        jnp.asarray([1.0, -0.2]),
        driver_dimension=2,
        geometry=phx.metrix.EuclideanStateGeometry(),
    )
    general = phx.solver.solve_rough_differential(
        problem, control, solver=phx.solver.LogODE()
    )
    explicit = phx.solver.solve_rough_differential(
        problem,
        control,
        solver=phx.solver.LogODE(explicit_fields=explicit_fields),
    )
    linear = phx.solver.solve_rough_differential(
        problem,
        control,
        solver=phx.solver.LinearLogODE(
            (left, right),
            matrix_function_policy=phx.linalg.MatrixFunctionPolicy(
                "arnoldi", max_dimension=2
            ),
        ),
    )

    assert jnp.allclose(general.states, linear.states, rtol=2e-7, atol=2e-8)
    assert jnp.allclose(general.states, explicit.states, rtol=2e-8, atol=2e-9)
    assert jnp.all(general.statuses == 0)
    assert jnp.all(linear.statuses == 0)
    assert general.state_geometry_id == "state-geometry:euclidean"
    assert general.metadata["state_geometry_id"] == general.state_geometry_id
    assert general.solver_name == "LogODE"
    assert explicit.solver_name == "LogODE"
    assert general.solver_id != explicit.solver_id
    assert linear.solver_name == "LinearLogODE"
    assert linear.solver_id == linear.solver.solver_id
    assert int(general.statistics["num_accepted_steps"][0]) > 0


def test_joint_time_channel_integrates_drift_and_time_dependent_fields():
    times = jnp.linspace(0.0, 1.0, 5)
    control = phx.stochastic.LogSignatureControl.from_values(
        times,
        jnp.zeros((5, 1)),
        depth=3,
        coarse_indices=(0, 2, 4),
        joint_time=True,
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: jnp.zeros(state.shape + (1,)),
        jnp.asarray([0.0]),
        driver_dimension=1,
        drift=lambda time, state, args: jnp.ones_like(state) * time,
        time_dependent=True,
    )
    solution = phx.solver.solve_rough_differential(
        problem,
        control,
        save_times=jnp.asarray([0.5, 1.0]),
        solver=phx.solver.LogODE(),
    )

    assert jnp.allclose(solution.states[:, 0], jnp.asarray([0.125, 0.5]), atol=2e-8)
    assert solution.successful


def test_joint_time_logode_batches_sample_paths():
    times = jnp.asarray([0.0, 0.5, 1.0])
    values = jnp.stack((times, -times), axis=0)[..., None]
    control = phx.stochastic.LogSignatureControl.from_values(
        times,
        values,
        depth=3,
        coarse_indices=(0, 1, 2),
        sample_shape=(2,),
        joint_time=True,
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: jnp.ones(state.shape + (1,)),
        jnp.asarray([0.0]),
        driver_dimension=1,
        drift=lambda time, state, args: jnp.ones_like(state),
    )
    solution = phx.solver.solve_rough_differential(
        problem,
        control,
        save_times=jnp.asarray([1.0]),
        solver=phx.solver.LogODE(),
    )

    assert solution.states.shape == (2, 1, 1)
    assert solution.statuses.shape == (2, 2)
    assert jnp.all(solution.successful)
    assert jnp.allclose(solution.states[:, 0, 0], jnp.asarray([2.0, 0.0]))


def test_linear_logode_rejects_time_dependent_problem():
    times = jnp.asarray([0.0, 1.0])
    control = phx.stochastic.LogSignatureControl.from_values(
        times,
        jnp.asarray([[0.0], [0.5]]),
        depth=2,
        joint_time=True,
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: (time * state)[..., None],
        jnp.asarray([1.0]),
        driver_dimension=1,
        drift=lambda time, state, args: 0.2 * state,
        time_dependent=True,
    )

    with pytest.raises(ValueError, match="autonomous explicit operators"):
        phx.solver.solve_rough_differential(
            problem,
            control,
            solver=phx.solver.LinearLogODE((jnp.asarray([[0.2]]), jnp.asarray([[0.5]]))),
        )


def test_logode_exposes_failed_inner_diffrax_status():
    control = phx.stochastic.LogSignatureControl.from_values(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [1.0]]),
        depth=2,
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: state[..., None],
        jnp.asarray([1.0]),
        driver_dimension=1,
    )
    solution = phx.solver.solve_rough_differential(
        problem,
        control,
        solver=phx.solver.LogODE(dt0=0.01, max_steps=1),
    )

    assert solution.statuses.shape == (1,)
    assert int(solution.statuses[0]) != 0
    assert not bool(solution.successful)
    assert int(solution.statistics["num_steps"][0]) == 1


def test_logode_local_retraction_preserves_special_orthogonal_state():
    times = jnp.linspace(0.0, 1.0, 5)
    angle = 0.7
    control = phx.stochastic.LogSignatureControl.from_values(
        times,
        (angle * times)[:, None],
        depth=2,
        coarse_indices=(0, 2, 4),
    )
    generator = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: jnp.stack((state @ generator,), axis=-1),
        jnp.eye(2),
        driver_dimension=1,
        geometry=geometry,
    )
    solution = phx.solver.solve_rough_differential(
        problem, control, solver=phx.solver.LogODE()
    )
    expected = jax.scipy.linalg.expm(angle * generator)

    assert solution.successful
    assert geometry.contains(solution.states)
    assert jnp.allclose(solution.states[-1], expected, atol=2e-9)
    assert jnp.allclose(
        jnp.swapaxes(solution.states, -1, -2) @ solution.states,
        jnp.eye(2),
        atol=2e-9,
    )


def test_logode_local_retraction_preserves_spd_state_and_refines():
    times = jnp.linspace(0.0, 1.0, 9)
    total_increment = 0.7
    values = (total_increment * times)[:, None]
    coarse_control = phx.stochastic.LogSignatureControl.from_values(
        times, values, depth=2, coarse_indices=(0, 8)
    )
    fine_control = phx.stochastic.LogSignatureControl.from_values(
        times, values, depth=2, coarse_indices=tuple(range(9))
    )
    generator = jnp.asarray([[0.2, 0.1], [0.1, -0.1]])
    initial = jnp.asarray([[2.0, 0.3], [0.3, 1.0]])
    geometry = phx.metrix.SymmetricPositiveDefiniteStateGeometry(2)
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: jnp.stack(
            (generator @ state + state @ generator.T,), axis=-1
        ),
        initial,
        driver_dimension=1,
        geometry=geometry,
    )
    coarse = phx.solver.solve_rough_differential(
        problem, coarse_control, solver=phx.solver.LogODE()
    )
    fine = phx.solver.solve_rough_differential(
        problem, fine_control, solver=phx.solver.LogODE()
    )
    flow = jax.scipy.linalg.expm(total_increment * generator)
    expected = flow @ initial @ flow.T

    assert coarse.successful
    assert fine.successful
    assert geometry.contains(coarse.states)
    assert geometry.contains(fine.states)
    assert jnp.all(jnp.linalg.eigvalsh(fine.states) > 0.0)
    assert jnp.linalg.norm(fine.states[-1] - expected) < jnp.linalg.norm(
        coarse.states[-1] - expected
    )


def test_depth_three_accepts_hurst_point_three_while_depth_two_rejects():
    process = phx.stochastic.FractionalGaussianProcess(0.3, 0.2)
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(903),
        jnp.linspace(0.0, 1.0, 9),
    )
    depth_two = phx.stochastic.LogSignatureControl.from_fractional_gaussian(
        realization, depth=2, coarse_indices=(0, 4, 8)
    )
    depth_three = phx.stochastic.LogSignatureControl.from_fractional_gaussian(
        realization, depth=3, coarse_indices=(0, 4, 8)
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: 0.2 * state[..., None],
        jnp.asarray([1.0]),
        driver_dimension=1,
    )

    with pytest.raises(ValueError, match="Control depth 2"):
        phx.solver.solve_rough_differential(
            problem, depth_two, solver=phx.solver.LogODE()
        )
    solution = phx.solver.solve_rough_differential(
        problem, depth_three, solver=phx.solver.LogODE()
    )

    assert solution.successful
    assert jnp.all(jnp.isfinite(solution.states))


def test_logode_is_jittable_batched_and_differentiable():
    times = jnp.linspace(0.0, 1.0, 5)
    values = jnp.stack((times, -times, 2.0 * times), axis=0)[..., None]
    control = phx.stochastic.LogSignatureControl.from_values(
        times,
        values,
        depth=2,
        coarse_indices=(0, 2, 4),
        sample_shape=(3,),
    )
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, rate: rate * state[..., None],
        jnp.asarray([1.0]),
        driver_dimension=1,
        args=jnp.asarray(0.5),
    )
    solver = phx.solver.LogODE()

    def terminals(rate):
        parameterized = eqx.tree_at(lambda value: value.args, problem, rate)
        return phx.solver.solve_rough_differential(
            parameterized,
            control,
            save_times=jnp.asarray([1.0]),
            solver=solver,
        ).states[..., 0, 0]

    compiled = jax.jit(terminals)
    actual = compiled(jnp.asarray(0.5))
    gradient = jax.jacrev(terminals)(jnp.asarray(0.5))
    expected = jnp.exp(jnp.asarray([0.5, -0.5, 1.0]))

    assert jnp.allclose(actual, expected, rtol=2e-8, atol=2e-9)
    assert jnp.allclose(
        gradient,
        jnp.asarray([1.0, -1.0, 2.0]) * expected,
        rtol=2e-7,
        atol=2e-8,
    )


def test_linear_logode_rejects_unconverged_matrix_function_intervals():
    control = phx.stochastic.LogSignatureControl.from_values(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [1.0]]),
        depth=2,
    )
    matrix = jnp.diag(jnp.asarray([1.0, 2.0]))
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: jnp.stack((matrix @ state,), axis=-1),
        jnp.asarray([1.0, 1.0]),
        driver_dimension=1,
    )
    solution = phx.solver.solve_rough_differential(
        problem,
        control,
        solver=phx.solver.LinearLogODE(
            (matrix,),
            matrix_function_policy=phx.linalg.MatrixFunctionPolicy(
                "arnoldi",
                max_dimension=1,
                error_tolerance=1e-12,
            ),
        ),
    )

    assert solution.statuses.shape == (1,)
    assert int(solution.statuses[0]) != 0
    assert not bool(solution.successful)
    assert int(solution.statistics["num_accepted_steps"][0]) == 0
    assert int(solution.statistics["num_rejected_steps"][0]) == 1
