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
        return jnp.stack(
            tuple(matrix @ state for matrix in lifted_matrices), axis=-1
        )

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
            matrix_function_policy=phx.solver.MatrixFunctionPolicy(
                "arnoldi", num_matvecs=2
            ),
        ),
    )

    assert jnp.allclose(general.states, linear.states, rtol=2e-7, atol=2e-8)
    assert jnp.allclose(general.states, explicit.states, rtol=2e-8, atol=2e-9)
    assert jnp.all(general.statuses == 0)
    assert jnp.all(linear.statuses == 0)
    assert general.metadata["geometry_id"] == "state-geometry:euclidean"
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
