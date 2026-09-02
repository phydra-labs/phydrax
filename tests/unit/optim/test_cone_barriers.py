#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_builtin_cone_barriers_have_finite_symmetric_hessians():
    cases = (
        (phx.optim.NonnegativeCone(2), jnp.asarray([1.0, 2.0])),
        (phx.optim.SecondOrderCone(3), jnp.asarray([2.0, 0.2, -0.1])),
        (phx.optim.RotatedSecondOrderCone(3), jnp.asarray([1.0, 1.0, 0.2])),
        (phx.optim.ExponentialCone(), jnp.asarray([0.0, 1.0, 2.0])),
        (phx.optim.PowerCone(0.4), jnp.asarray([1.5, 1.2, 0.2])),
        (
            phx.optim.PositiveSemidefiniteCone(2),
            phx.optim.PositiveSemidefiniteCone(2).pack(
                jnp.asarray([[2.0, 0.1], [0.1, 1.5]])
            ),
        ),
    )
    for cone, point in cases:
        oracle = phx.optim.cone_barrier_oracle(cone)
        hessian = oracle.hessian(point)
        assert jnp.isfinite(oracle.value(point))
        assert jnp.all(jnp.isfinite(oracle.gradient(point)))
        assert jnp.allclose(hessian, hessian.T, atol=1e-6)
        step = oracle.maximum_interior_step(point, -0.1 * point)
        assert step > 0.0


def test_batched_cone_differentials_are_independent_per_leading_index():
    psd = phx.optim.PositiveSemidefiniteCone(2)
    cases = (
        (phx.optim.SecondOrderCone(3), jnp.asarray([2.0, 0.2, -0.1])),
        (phx.optim.ExponentialCone(), jnp.asarray([0.0, 1.0, 2.0])),
        (psd, psd.pack(jnp.asarray([[2.0, 0.1], [0.1, 1.5]]))),
        (
            phx.optim.ProductCone(
                (phx.optim.SecondOrderCone(2), phx.optim.NonnegativeCone(2))
            ),
            jnp.asarray([2.0, 0.2, 1.2, 0.8]),
        ),
    )
    scales = jnp.asarray([[0.8, 1.0], [1.2, 1.5]])
    indices = ((0, 0), (0, 1), (1, 0), (1, 1))

    for cone, reference in cases:
        oracle = phx.optim.cone_barrier_oracle(cone)
        points = scales[..., None] * reference
        direction = jnp.linspace(-0.3, 0.2, cone.dimension)
        directions = jnp.broadcast_to(direction, points.shape)

        values = oracle.value(points)
        gradients = oracle.gradient(points)
        hessians = oracle.hessian(points)
        actions = oracle.hessian_action(points, directions)

        assert values.shape == points.shape[:-1]
        assert gradients.shape == points.shape
        assert hessians.shape == points.shape[:-1] + (
            cone.dimension,
            cone.dimension,
        )
        assert actions.shape == points.shape
        for index in indices:
            point = points[index]
            vector = directions[index]
            expected_hessian = oracle.hessian(point)
            assert jnp.allclose(values[index], oracle.value(point), atol=1e-6)
            assert jnp.allclose(gradients[index], oracle.gradient(point), atol=1e-6)
            assert jnp.allclose(hessians[index], expected_hessian, atol=1e-6)
            assert jnp.allclose(
                actions[index], oracle.hessian_action(point, vector), atol=1e-6
            )
            assert jnp.allclose(actions[index], expected_hessian @ vector, atol=1e-6)


def test_batched_product_cone_interior_steps_are_independent():
    cone = phx.optim.ProductCone(
        (phx.optim.NonnegativeCone(1), phx.optim.NonnegativeCone(1))
    )
    oracle = phx.optim.cone_barrier_oracle(cone)
    point = jnp.asarray([[1.0, 10.0], [10.0, 2.0], [4.0, 10.0]])
    direction = jnp.asarray([[-1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])

    step = oracle.maximum_interior_step(point, direction)

    assert step.shape == (3,)
    assert jnp.allclose(step, 0.995 * jnp.asarray([1.0, 2.0, 4.0]), atol=1e-5)


def test_native_homogeneous_predictor_corrector_recovers_qp_solution():
    program = phx.optim.ConicProgram(
        jnp.asarray([[1.0]]),
        jnp.asarray([-1.0]),
        jnp.asarray([[1.0]]),
        jnp.asarray([1.0]),
        phx.optim.ZeroCone(1),
    )
    result = phx.optim.solve_conic_program(
        program,
        policy=phx.optim.ConvexSolvePolicy(
            phx.optim.NativeHomogeneousConic(),
            termination=phx.optim.ConvexTermination(
                absolute=1e-7,
                relative=1e-7,
                maximum_steps=64,
            ),
        ),
    )
    assert result.successful
    assert jnp.allclose(result.primal, jnp.asarray([1.0]), atol=1e-6)
    assert result.provenance.backend == "phydrax"
    assert result.provenance.backend_version == "native-jax-hsd"
