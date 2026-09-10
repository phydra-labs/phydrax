#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _problem(initial=1.0, batch=False):
    horizon = 3
    prefix = (2,) if batch else ()
    dynamics = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), prefix + (horizon, 1, 1))
    controls = jnp.broadcast_to(jnp.ones((horizon, 1, 1)), prefix + (horizon, 1, 1))
    initial_state = (
        jnp.asarray([[initial], [2.0 * initial]]) if batch else jnp.asarray([initial])
    )
    return phx.control.LinearQuadraticControlProblem(
        dynamics,
        controls,
        initial_state,
        jnp.broadcast_to(jnp.ones((horizon, 1, 1)), prefix + (horizon, 1, 1)),
        jnp.broadcast_to(jnp.ones((horizon, 1, 1)), prefix + (horizon, 1, 1)),
        jnp.broadcast_to(jnp.ones((1, 1)), prefix + (1, 1)),
        control_lower_bounds=jnp.broadcast_to(
            -2.0 * jnp.ones((horizon, 1)), prefix + (horizon, 1)
        ),
        control_upper_bounds=jnp.broadcast_to(
            2.0 * jnp.ones((horizon, 1)), prefix + (horizon, 1)
        ),
        problem_id="sparse-control",
    )


def test_structural_sparse_control_operators_match_dense_compilation():
    dense = phx.control.compile_linear_quadratic_control(_problem())
    sparse = phx.control.compile_linear_quadratic_control(
        _problem(),
        compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
    )

    assert sparse.representation == "sparse"
    assert isinstance(sparse.program, phx.optim.ConicProgram)
    assert sparse.program.quadratic_is_sparse
    assert sparse.program.constraint_is_sparse
    np.testing.assert_allclose(
        sparse.program.quadratic.as_dense(),
        dense.program.quadratic,
        atol=1e-12,
    )
    constraints = sparse.program.constraint_matrix.as_dense()
    equalities = sparse.constraint_layout.num_equalities
    np.testing.assert_allclose(
        constraints[:equalities],
        dense.program.equality_matrix,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        constraints[equalities:],
        dense.program.inequality_matrix[..., : dense.program.num_user_inequalities, :],
        atol=1e-12,
    )
    assert sparse.bound_layout.control_lower_slices


def test_sparse_control_compilation_preserves_shared_case_batches():
    compilation = phx.control.compile_linear_quadratic_control(
        _problem(batch=True),
        compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
    )

    quadratic = compilation.program.quadratic
    constraints = compilation.program.constraint_matrix
    assert quadratic.batch_shape == (2,)
    assert constraints.batch_shape == (2,)
    assert quadratic.sparse_storage().batch_shape == (2,)
    assert constraints.sparse_storage().batch_shape == (2,)


def test_sparse_prepared_control_refresh_and_solution_match_dense():
    compilation_policy = phx.control.LinearControlCompilationPolicy("sparse")
    prepared = phx.control.prepare_linear_quadratic_control(
        _problem(),
        compilation_policy=compilation_policy,
    )
    refreshed = phx.control.refresh_linear_quadratic_control(
        prepared,
        _problem(initial=2.0),
    )
    sparse_result = phx.control.solve_prepared_linear_quadratic_control(refreshed)
    dense_result = phx.control.solve_linear_quadratic_control(_problem(initial=2.0))

    assert refreshed.compilation.representation == "sparse"
    assert refreshed.prepared.numeric_version == 1
    np.testing.assert_allclose(sparse_result.controls, dense_result.controls, atol=2e-5)
    np.testing.assert_allclose(sparse_result.states, dense_result.states, atol=2e-5)
