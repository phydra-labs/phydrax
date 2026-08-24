#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._programming._cone_root import safeguarded_newton_bisection


def _mp_power_projection(value, exponent):
    with mp.workdps(100):
        x, y, z = (mp.mpf(str(component)) for component in value)
        alpha = mp.mpf(str(exponent))
        complement = 1 - alpha
        absolute_z = abs(z)

        def positive_root(source, weight, root, gap):
            product = weight * root * gap
            radical = mp.sqrt(source * source + 4 * product)
            if source >= 0:
                return (source + radical) / 2
            return 2 * product / (radical - source)

        def coordinates(transformed):
            root = absolute_z / (1 + mp.exp(-transformed))
            gap = absolute_z / (1 + mp.exp(transformed))
            projected_x = positive_root(x, alpha, root, gap)
            projected_y = positive_root(y, complement, root, gap)
            return root, projected_x, projected_y

        def function(transformed):
            root, projected_x, projected_y = coordinates(transformed)
            return (
                alpha * mp.log(projected_x)
                + complement * mp.log(projected_y)
                - mp.log(root)
            )

        lower = mp.mpf(-256)
        upper = mp.mpf(256)
        lower_value = function(lower)
        upper_value = function(upper)
        assert lower_value * upper_value < 0
        for _ in range(400):
            midpoint = (lower + upper) / 2
            midpoint_value = function(midpoint)
            if mp.sign(lower_value) == mp.sign(midpoint_value):
                lower = midpoint
                lower_value = midpoint_value
            else:
                upper = midpoint
        root, projected_x, projected_y = coordinates((lower + upper) / 2)
        geometric = projected_x**alpha * projected_y**complement
        return np.asarray(
            [float(projected_x), float(projected_y), float(mp.sign(z) * geometric)]
        )


def test_psd_scaled_upper_column_packing_is_frobenius_isometric():
    cone = phx.optim.PositiveSemidefiniteCone(3)
    matrix = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    other = jnp.asarray([[2.0, -1.0, 0.5], [-1.0, 3.0, 4.0], [0.5, 4.0, -2.0]])
    expected = jnp.asarray(
        [1.0, 2.0 * jnp.sqrt(2.0), 4.0, 3.0 * jnp.sqrt(2.0), 5.0 * jnp.sqrt(2.0), 6.0]
    )

    np.testing.assert_allclose(cone.pack(matrix), expected)
    np.testing.assert_allclose(cone.unpack(expected), matrix)
    np.testing.assert_allclose(
        jnp.vdot(cone.pack(matrix), cone.pack(other)),
        jnp.trace(matrix @ other),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(cone.pack(matrix)), jnp.linalg.norm(matrix)
    )
    with pytest.raises((ValueError, RuntimeError), match="symmetric"):
        jax.block_until_ready(cone.pack(matrix.at[0, 1].set(7.0)))


def test_psd_projection_and_frechet_derivative_handle_repeated_nonzero_spectrum():
    cone = phx.optim.PositiveSemidefiniteCone(3)
    matrix = jnp.diag(jnp.asarray([2.0, 2.0, -1.0]))
    direction = jnp.asarray([[0.3, -0.2, 0.4], [-0.2, 0.1, -0.5], [0.4, -0.5, 0.7]])
    packed = cone.pack(matrix)
    packed_direction = cone.pack(direction)
    projected = cone.project(packed)

    np.testing.assert_allclose(
        cone.unpack(projected), jnp.diag(jnp.asarray([2.0, 2.0, 0.0])), atol=1e-12
    )
    derivative = jax.jvp(cone.project, (packed,), (packed_direction,))[1]
    step = 1e-5
    finite_difference = (
        cone.project(packed + step * packed_direction)
        - cone.project(packed - step * packed_direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, atol=2e-6, rtol=2e-6)
    assert cone.dual_projection_smoothness_margin(packed) == 1.0
    boundary = cone.pack(jnp.diag(jnp.asarray([2.0, 0.0, -1.0])))
    assert cone.dual_projection_smoothness_margin(boundary) == 0.0


def test_exponential_projection_matches_primary_reference_values():
    cone = phx.optim.ExponentialCone()
    values = jnp.asarray(
        [
            [1.0, 2.0, 3.0],
            [0.14814832, 1.04294573, 0.67905585],
            [-0.78301134, 1.82790084, -1.05417044],
            [1.3282585, -0.43277314, 1.7468072],
            [0.50210027, 0.12314491, -1.77568921],
        ]
    )
    expected = jnp.asarray(
        [
            [0.8899428, 1.94041881, 3.06957226],
            [-0.02001571, 0.8709169, 0.85112944],
            [-1.17415616, 0.9567094, 0.280399],
            [0.53160512, 0.2804836, 1.86652094],
            [0.0, 0.0, 0.0],
        ]
    )
    projected = jax.jit(cone.project)(values)

    np.testing.assert_allclose(projected, expected, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(cone.project(projected), projected, atol=2e-7, rtol=2e-7)
    assert jnp.all(cone.contains(projected, tolerance=2e-7))


def test_exponential_and_power_cones_satisfy_moreau_and_regular_derivatives():
    cones = (phx.optim.ExponentialCone(), phx.optim.PowerCone(0.4))
    value = jnp.asarray([1.0, 2.0, 3.0])
    first = jnp.asarray([0.2, -0.1, 0.3])
    second = jnp.asarray([-0.4, 0.5, 0.1])

    for cone in cones:
        np.testing.assert_allclose(
            cone.project(value) - cone.project_dual(-value), value, atol=3e-7, rtol=3e-7
        )
        derivative_first = jax.jvp(cone.project, (value,), (first,))[1]
        derivative_second = jax.jvp(cone.project, (value,), (second,))[1]
        step = 1e-5
        finite_difference = (
            cone.project(value + step * first) - cone.project(value - step * first)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            derivative_first, finite_difference, atol=3e-6, rtol=3e-5
        )
        np.testing.assert_allclose(
            jnp.vdot(second, derivative_first),
            jnp.vdot(first, derivative_second),
            atol=3e-6,
            rtol=3e-5,
        )
        assert cone.dual_projection_smoothness_margin(value) > 0.0
        assert cone.dual_projection_smoothness_margin(jnp.zeros(3)) == 0.0


def test_power_cone_projection_is_idempotent_and_exponent_swap_equivariant():
    value = jnp.asarray([-1.0, 2.0, 1.0])
    for exponent in (0.01, 0.1, 0.5, 0.9, 0.99):
        cone = phx.optim.PowerCone(exponent)
        projected = cone.project(value)
        np.testing.assert_allclose(
            cone.project(projected), projected, atol=3e-7, rtol=3e-7
        )
        assert cone.contains(projected, tolerance=3e-7)
        swapped = phx.optim.PowerCone(1.0 - exponent).project(
            value[jnp.asarray([1, 0, 2])]
        )
        np.testing.assert_allclose(
            swapped,
            projected[jnp.asarray([1, 0, 2])],
            atol=3e-7,
            rtol=3e-7,
        )


def test_psd_float32_cross_sign_divided_difference_uses_regular_ratio():
    cone = phx.optim.PositiveSemidefiniteCone(2)
    matrix = jnp.diag(jnp.asarray([-2e-4, 1e-4], dtype=jnp.float32))
    direction = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    packed = cone.pack(matrix)
    packed_direction = cone.pack(direction)

    derivative = jax.jvp(cone.project, (packed,), (packed_direction,))[1]
    expected = cone.pack(
        jnp.asarray([[0.0, 1.0 / 3.0], [1.0 / 3.0, 0.0]], dtype=jnp.float32)
    )
    step = jnp.asarray(1e-5, dtype=jnp.float32)
    finite_difference = (
        cone.project(packed + step * packed_direction)
        - cone.project(packed - step * packed_direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(derivative, expected, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        atol=2e-4,
        rtol=2e-3,
    )
    assert cone.dual_projection_smoothness_margin(packed) > 0.0


def test_asymmetric_projectors_do_not_widen_mathematical_membership_regions():
    epsilon = jnp.finfo(jnp.float64).eps
    cases = (
        (
            phx.optim.ExponentialCone(),
            jnp.asarray([0.0, 1.0, 1.0 - 64.0 * epsilon]),
        ),
        (
            phx.optim.PowerCone(0.5),
            jnp.asarray([1.0, 1.0, 1.0 + 64.0 * epsilon]),
        ),
    )
    for cone, value in cases:
        projected = cone.project(value)
        assert not cone.contains(value, tolerance=0.0)
        assert not jnp.array_equal(projected, value)


def test_exponential_log_domain_membership_handles_tiny_scale_large_ratio():
    cone = phx.optim.ExponentialCone()
    tiny = jnp.finfo(jnp.float64).tiny
    primal = jnp.asarray([800.0 * tiny, tiny, 1e40])
    dual = jnp.asarray([-tiny, -800.0 * tiny, 3e39])

    for scale in (1e-8, 1.0, 1e8):
        scaled_primal = scale * primal
        scaled_dual = scale * dual
        assert cone.contains(scaled_primal, tolerance=0.0)
        assert cone.contains_dual(scaled_dual, tolerance=0.0)
        np.testing.assert_array_equal(cone.project(scaled_primal), scaled_primal)
        np.testing.assert_array_equal(cone.project_dual(scaled_dual), scaled_dual)

    outside = primal.at[2].set(1e30)
    assert not cone.contains(outside, tolerance=0.0)
    assert not jnp.array_equal(cone.project(outside), outside)


def test_power_float32_matches_high_precision_oracle_across_scales_and_exponents():
    base = np.asarray([-1.0, 2.0, 1.0])
    direction = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float32)
    scales = (1e-30, 1e-15, 1e-3, 1.0, 1e3, 1e15, 1e30)
    for exponent in (0.01, 0.1, 0.5, 0.9, 0.99):
        cone = phx.optim.PowerCone(exponent)
        project = jax.jit(cone.project)
        oracle = _mp_power_projection(base, exponent)
        base_value = jnp.asarray(base, dtype=jnp.float32)
        base_projected = project(base_value)
        reference_derivative = jax.jvp(
            project,
            (base_value,),
            (direction,),
        )[1]
        for scale in scales:
            value = jnp.asarray(scale * base, dtype=jnp.float32)
            projected = project(value)
            reference = jnp.asarray(scale * oracle, dtype=jnp.float64)
            denominator = jnp.maximum(
                jnp.max(jnp.abs(reference)),
                jnp.finfo(jnp.float64).tiny,
            )
            relative_error = (
                jnp.max(jnp.abs(projected.astype(jnp.float64) - reference)) / denominator
            )
            scaled_residual = cone.residual(projected.astype(jnp.float64)) / denominator
            derivative = jax.jvp(
                project,
                (value,),
                (direction,),
            )[1]

            assert jnp.all(jnp.isfinite(projected))
            assert relative_error <= 5e-5
            assert scaled_residual <= 5e-5
            np.testing.assert_allclose(
                projected / jnp.asarray(scale, dtype=jnp.float32),
                base_projected,
                atol=5e-5,
                rtol=5e-5,
            )
            np.testing.assert_allclose(
                derivative,
                reference_derivative,
                atol=5e-5,
                rtol=5e-5,
            )


def test_power_float32_preserves_mixed_magnitude_coordinates():
    cone = phx.optim.PowerCone(0.99)
    project = jax.jit(cone.project)
    direction = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float32)
    primal = jnp.asarray([1e30, 1e-30, 1e29], dtype=jnp.float32)

    assert cone.contains(primal, tolerance=0.0)
    assert jnp.array_equal(project(primal), primal)
    np.testing.assert_array_equal(
        jax.jvp(project, (primal,), (direction,))[1],
        direction,
    )

    exterior = primal.at[2].set(3e29)
    projected = project(exterior)
    oracle = _mp_power_projection(np.asarray(exterior, dtype=np.float64), cone.exponent)
    reference_derivative = jax.jvp(
        cone.project,
        (exterior.astype(jnp.float64),),
        (direction.astype(jnp.float64),),
    )[1]
    derivative = jax.jvp(project, (exterior,), (direction,))[1]

    assert not cone.contains(exterior, tolerance=0.0)
    assert jnp.all(jnp.isfinite(projected))
    np.testing.assert_allclose(
        projected.astype(jnp.float64),
        oracle,
        atol=0.0,
        rtol=5e-5,
    )
    np.testing.assert_allclose(
        derivative.astype(jnp.float64),
        reference_derivative,
        atol=jnp.finfo(jnp.float32).smallest_subnormal,
        rtol=5e-5,
    )


def test_power_float32_extreme_scale_smoothness_margin_is_homogeneous():
    cone = phx.optim.PowerCone(0.4)
    value = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    reference = cone.dual_projection_smoothness_margin(value)
    assert reference > 0.0

    for scale in (1e-30, 1e30):
        scaled = cone.dual_projection_smoothness_margin(
            jnp.asarray(scale, dtype=jnp.float32) * value
        )
        np.testing.assert_allclose(
            scaled / jnp.asarray(scale, dtype=jnp.float32),
            reference,
            atol=5e-6,
            rtol=5e-5,
        )


def test_safeguarded_root_requires_residual_and_bracket_certificates():
    dtype = jnp.float64
    absolute = jnp.asarray(1e-6, dtype=dtype)
    relative = jnp.asarray(0.0, dtype=dtype)
    width_only = safeguarded_newton_bisection(
        lambda value: (1e12 * value - 0.9, jnp.asarray(0.0, dtype=dtype)),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(1e-12, dtype=dtype),
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        maximum_steps=1,
    )
    assert width_only.bracket_width <= absolute
    assert width_only.residual > absolute
    assert not width_only.converged

    residual_only = safeguarded_newton_bisection(
        lambda value: (value + 1e-8, jnp.asarray(0.0, dtype=dtype)),
        jnp.asarray(-1.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        maximum_steps=1,
    )
    assert residual_only.residual <= absolute
    assert residual_only.bracket_width > absolute
    assert not residual_only.converged

    endpoint = safeguarded_newton_bisection(
        lambda value: (value - 1.0, jnp.asarray(1.0, dtype=dtype)),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(3.0, dtype=dtype),
        absolute_tolerance=absolute,
        relative_tolerance=relative,
        maximum_steps=1,
    )
    assert endpoint.converged
    assert endpoint.root == 1.0
    assert endpoint.residual == 0.0
    assert endpoint.bracket_width == 0.0


def test_advanced_cone_topology_validation_and_identity():
    with pytest.raises(ValueError, match="matrix_size"):
        phx.optim.PositiveSemidefiniteCone(0)
    for exponent in (0.0, 1.0, -0.2, jnp.nan, jnp.inf):
        with pytest.raises(ValueError, match="exponent"):
            phx.optim.PowerCone(exponent)

    assert (
        phx.optim.PositiveSemidefiniteCone(2).cone_id
        != phx.optim.PositiveSemidefiniteCone(3).cone_id
    )
    assert phx.optim.PowerCone(0.25).cone_id != phx.optim.PowerCone(0.75).cone_id
    capabilities = phx.optim.ClarabelInteriorPoint().capabilities
    assert capabilities.dense
    assert not capabilities.sparse
    assert not capabilities.matrix_free
    assert not capabilities.implicit_differentiation
    assert not capabilities.algorithmic_differentiation
