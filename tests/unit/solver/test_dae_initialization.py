import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _semi_explicit_system():
    return phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: jnp.asarray(
            (
                state_rate[0] + parameter * state[0],
                state[1] - state[0] ** 2,
            )
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        state_scale=jnp.asarray((2.0, 3.0)),
        state_rate_scale=jnp.asarray((4.0, 5.0)),
        residual_scale=jnp.asarray((0.5, 2.0)),
        system_id="semi-explicit-initialization",
    )


def _strict_policy():
    return phx.solver.DAESolvePolicy(
        initialization_termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-11,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=20,
        )
    )


def test_dae_structure_broadcasts_roles_and_preserves_independent_scales():
    structure = phx.dynamics.DAEStructure(
        ("differential", "algebraic"),
        component_axis=-1,
    )
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, args: state_rate + state,
        state_shape=(3, 2),
        structure=structure,
        state_scale=2.0,
        state_rate_scale=jnp.asarray((4.0, 5.0)),
        residual_scale=10.0,
        system_id="spatial-roles",
    )
    expected_differential = jnp.asarray(((True, False), (True, False), (True, False)))

    assert jnp.array_equal(
        structure.differential_variable_mask(system.state_shape),
        expected_differential,
    )
    assert jnp.array_equal(
        structure.algebraic_equation_mask(system.state_shape),
        ~expected_differential,
    )
    assert jnp.array_equal(system.state_scale, jnp.full((3, 2), 2.0))
    assert jnp.array_equal(
        system.state_rate_scale,
        jnp.broadcast_to(jnp.asarray((4.0, 5.0)), (3, 2)),
    )
    assert jnp.array_equal(
        system.scaled_residual(0.0, jnp.ones((3, 2)), jnp.ones((3, 2))),
        jnp.full((3, 2), 0.2),
    )


def test_mass_matrix_constructor_preserves_raw_implicit_residual():
    mass = jnp.asarray(((1.0, 0.0), (0.0, 0.0)))
    system = phx.dynamics.DifferentialAlgebraicSystem.from_mass_matrix(
        mass,
        lambda time, state, args: jnp.asarray((-state[0], state[1] - state[0])),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        system_id="mass-matrix",
    )

    actual = system(0.0, jnp.asarray((2.0, 2.0)), jnp.asarray((-2.0, 7.0)))

    assert jnp.array_equal(actual, jnp.zeros(2))


def test_index_one_initialization_fixes_differential_state_and_algebraic_rate():
    system = _semi_explicit_system()
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((2.0, 0.0)),
        initial_state_rate=jnp.asarray((0.0, 17.0)),
        args=jnp.asarray(2.0),
        problem_id="index-one-initialization",
    )
    result = phx.solver.initialize_dae(problem, 0.0, policy=_strict_policy())

    assert result.valid
    assert result.status == int(phx.solver.DAEInitializationStatus.SUCCESS)
    assert jnp.allclose(result.state, jnp.asarray((2.0, 4.0)), atol=1e-10)
    assert jnp.allclose(result.state_rate, jnp.asarray((-4.0, 17.0)), atol=1e-10)
    assert jnp.array_equal(result.fixed_state_mask, jnp.asarray((True, False)))
    assert jnp.array_equal(result.fixed_rate_mask, jnp.asarray((False, True)))
    assert jnp.array_equal(result.rate_valid, jnp.asarray((True, False)))
    assert result.nonlinear_result is not None
    assert result.nonlinear_result.successful
    assert result.residual_norm <= result.residual_threshold


def test_consistent_initialization_remains_inside_parameter_gradient():
    system = _semi_explicit_system()
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((2.0, 0.0)),
        initial_state_rate=jnp.zeros(2),
        args=jnp.asarray(1.0),
        problem_id="initialization-gradient",
    )

    def differential_rate(parameter):
        result = phx.solver.initialize_dae(
            problem,
            0.0,
            policy=_strict_policy(),
            args=parameter,
        )
        return result.state_rate[0]

    value, gradient = jax.jit(jax.value_and_grad(differential_rate))(jnp.asarray(2.0))

    assert jnp.allclose(value, -4.0, atol=1e-10)
    assert jnp.allclose(gradient, -2.0, atol=1e-9)


def test_fixed_rate_and_check_only_modes_have_distinct_validity_contracts():
    system = _semi_explicit_system()
    fixed_rate_problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((9.0, -3.0)),
        initial_state_rate=jnp.asarray((-1.0, 8.0)),
        args=jnp.asarray(2.0),
        initialization=phx.solver.DAEInitializationSpec.fixed_rate_state(),
        problem_id="fixed-rate-initialization",
    )
    fixed_rate = phx.solver.initialize_dae(
        fixed_rate_problem,
        0.0,
        policy=_strict_policy(),
    )

    assert fixed_rate.valid
    assert jnp.allclose(fixed_rate.state, jnp.asarray((0.5, 0.25)), atol=1e-10)
    assert jnp.array_equal(fixed_rate.state_rate, jnp.asarray((-1.0, 8.0)))
    assert jnp.all(fixed_rate.rate_valid)

    check_problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0, 0.0)),
        initial_state_rate=jnp.zeros(2),
        args=jnp.asarray(1.0),
        initialization=phx.solver.DAEInitializationSpec.check_only(),
        problem_id="check-only-initialization",
    )
    checked = phx.solver.initialize_dae(check_problem, 0.0, policy=_strict_policy())

    assert not checked.valid
    assert checked.status == int(phx.solver.DAEInitializationStatus.RESIDUAL_TOO_LARGE)
    assert checked.nonlinear_result is None
    assert jnp.array_equal(checked.state, check_problem.initial_state)
    assert jnp.array_equal(checked.state_rate, check_problem.initial_state_rate)


def test_custom_initialization_requires_one_unknown_per_residual_scalar():
    system = _semi_explicit_system()
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0, 1.0)),
        initialization=phx.solver.DAEInitializationSpec.from_masks(
            jnp.asarray((True, True)),
            jnp.asarray((True, False)),
        ),
        problem_id="invalid-custom-initialization",
    )

    with pytest.raises(ValueError, match="exactly one free state/rate unknown"):
        phx.solver.initialize_dae(problem, 0.0)
