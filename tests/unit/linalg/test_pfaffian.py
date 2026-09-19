#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _skew(value):
    value = jnp.asarray(value)
    return value - jnp.swapaxes(value, -1, -2)


def test_real_complex_and_batched_values_obey_pfaffian_square_identity():
    real = jnp.asarray(
        [
            [0.0, 2.0, -1.0, 0.5],
            [-2.0, 0.0, 3.0, 1.5],
            [1.0, -3.0, 0.0, 4.0],
            [-0.5, -1.5, -4.0, 0.0],
        ]
    )
    expected = 2.0 * 4.0 - (-1.0) * 1.5 + 0.5 * 3.0
    result = phx.linalg.evaluate_pfaffian(
        real,
        phx.linalg.PfaffianPolicy(verify_determinant=True),
    )
    np.testing.assert_allclose(result.value, expected)
    np.testing.assert_allclose(result.value**2, jnp.linalg.det(real), rtol=1.0e-12)
    assert bool(result.successful)
    assert bool(result.determinant_identity_verified)
    assert result.determinant_identity_residual < 1.0e-12

    complex_matrix = real.astype(jnp.complex128).at[0, 1].set(2.0 + 1.5j)
    complex_matrix = complex_matrix.at[1, 0].set(-complex_matrix[0, 1])
    batch = jnp.stack((complex_matrix, (1.0 - 0.25j) * complex_matrix))
    batched = phx.linalg.evaluate_pfaffian(batch)
    np.testing.assert_allclose(
        batched.value * batched.value,
        jnp.linalg.det(batch),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(jnp.abs(batched.sign), 1.0)
    np.testing.assert_allclose(jnp.exp(batched.log_abs), jnp.abs(batched.value))
    assert bool(jnp.all(batched.successful))
    assert not bool(jnp.any(batched.determinant_identity_verified))
    assert bool(jnp.all(jnp.isnan(batched.determinant_identity_residual)))


def test_permutation_orientation_empty_and_odd_structural_values():
    upper = jnp.zeros((6, 6)).at[0, 1].set(2.0).at[2, 3].set(3.0).at[4, 5].set(5.0)
    matrix = upper - upper.T
    base = phx.linalg.evaluate_pfaffian(matrix)
    permutation = jnp.asarray([1, 0, 2, 3, 4, 5])
    permuted = matrix[permutation[:, None], permutation[None, :]]
    swapped = phx.linalg.evaluate_pfaffian(permuted)
    np.testing.assert_allclose(swapped.value, -base.value, rtol=1.0e-11)

    empty = phx.linalg.evaluate_pfaffian(jnp.zeros((0, 0)))
    assert empty.value == 1
    assert empty.sign == 1
    assert empty.log_abs == 0
    assert not bool(empty.singular)
    assert bool(empty.successful)

    odd = phx.linalg.evaluate_pfaffian(_skew(jnp.arange(9.0).reshape(3, 3)))
    assert odd.value == 0
    assert odd.sign == 0
    assert jnp.isneginf(odd.log_abs)
    assert bool(odd.singular)
    assert bool(odd.successful)
    assert not bool(odd.log_derivative_valid)


def test_singular_and_near_singular_evidence_guards_log_derivatives():
    singular = jnp.zeros((4, 4), dtype=jnp.float64)
    singular_result = phx.linalg.evaluate_pfaffian(singular)
    assert singular_result.value == 0
    assert jnp.isneginf(singular_result.log_abs)
    assert bool(singular_result.singular)
    assert bool(singular_result.successful)
    assert not bool(singular_result.log_derivative_valid)

    tangent = _skew(jnp.arange(16.0).reshape(4, 4))
    _, log_tangent = jax.jvp(
        lambda value: phx.linalg.evaluate_pfaffian(value).log_abs,
        (singular,),
        (tangent,),
    )
    assert jnp.isnan(log_tangent)

    zero_two = jnp.zeros((2, 2), dtype=jnp.float64)
    unit_direction = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])
    _, value_tangent = jax.jvp(
        lambda value: phx.linalg.evaluate_pfaffian(value).value,
        (zero_two,),
        (unit_direction,),
    )
    np.testing.assert_allclose(value_tangent, 1.0)

    paired_direction = jnp.zeros((4, 4), dtype=jnp.float64)
    paired_direction = paired_direction.at[0, 1].set(1.0)
    paired_direction = paired_direction.at[1, 0].set(-1.0)
    paired_direction = paired_direction.at[2, 3].set(1.0)
    paired_direction = paired_direction.at[3, 2].set(-1.0)
    second_derivative = jax.hessian(
        lambda scale: phx.linalg.evaluate_pfaffian(scale * paired_direction).value
    )(jnp.asarray(0.0))
    np.testing.assert_allclose(second_derivative, 2.0)
    assert bool(singular_result.value_derivative_valid)

    unsupported = phx.linalg.evaluate_pfaffian(jnp.zeros((6, 6), dtype=jnp.float64))
    assert not bool(unsupported.value_derivative_valid)

    near = jnp.asarray([[0.0, 1.0e-12], [-1.0e-12, 0.0]])
    guarded = phx.linalg.evaluate_pfaffian(
        near,
        phx.linalg.PfaffianPolicy(pivot_tolerance=1.0e-10),
    )
    resolved = phx.linalg.evaluate_pfaffian(
        near,
        phx.linalg.PfaffianPolicy(pivot_tolerance=1.0e-14),
    )
    assert bool(guarded.singular)
    assert not bool(resolved.singular)
    np.testing.assert_allclose(resolved.value, 1.0e-12)

    huge = jnp.asarray(
        [
            [0.0, 1.0e200, 0.0, 0.0],
            [-1.0e200, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0e200],
            [0.0, 0.0, -1.0e200, 0.0],
        ]
    )
    overflow = phx.linalg.evaluate_pfaffian(huge)
    assert jnp.isinf(overflow.value)
    assert not bool(overflow.value_finite)
    assert bool(overflow.successful)
    assert jnp.isfinite(overflow.log_abs)

    maximum = 0.75 * jnp.finfo(jnp.float64).max
    extreme_projection = jnp.asarray(
        [[0.0, maximum], [-maximum, 0.0]],
        dtype=jnp.float64,
    )
    projected = phx.linalg.evaluate_pfaffian(
        extreme_projection,
        phx.linalg.PfaffianPolicy(pivot_tolerance=0.0),
    )
    assert bool(projected.successful)
    assert not bool(projected.singular)
    assert projected.sign == 1.0
    assert jnp.isfinite(projected.log_abs)
    np.testing.assert_allclose(projected.log_abs, jnp.log(maximum))

    magnitude = jnp.asarray(1.0e200, dtype=jnp.float64)
    schur_growth = jnp.asarray(
        [
            [0.0, 1.0, magnitude, magnitude],
            [-1.0, 0.0, 0.4 * magnitude, 0.5 * magnitude],
            [-magnitude, -0.4 * magnitude, 0.0, 0.0],
            [-magnitude, -0.5 * magnitude, 0.0, 0.0],
        ]
    )
    grown = phx.linalg.evaluate_pfaffian(
        schur_growth,
        phx.linalg.PfaffianPolicy(pivot_tolerance=0.0),
    )
    assert bool(grown.successful)
    assert not bool(grown.singular)
    assert grown.sign == -1.0
    assert jnp.isfinite(grown.log_abs)
    np.testing.assert_allclose(
        grown.log_abs,
        jnp.log(0.1) + 2.0 * jnp.log(magnitude),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_jit_vmap_refresh_and_regular_jvp_use_the_public_lifecycle():
    matrix = _skew(
        jnp.asarray(
            [
                [0.0, 0.2, 1.1, -0.7],
                [0.5, 0.0, 0.4, 1.3],
                [-0.2, 0.8, 0.0, 0.9],
                [1.0, -0.6, 0.3, 0.0],
            ],
            dtype=jnp.float64,
        )
    )
    plan = phx.linalg.plan_pfaffian(matrix)
    prepared = phx.linalg.prepare_pfaffian(matrix, plan)
    first = phx.linalg.evaluate_pfaffian(prepared)
    refreshed = phx.linalg.refresh_pfaffian(prepared, 1.5 * matrix)
    second = phx.linalg.evaluate_pfaffian(refreshed)
    assert second.numeric_version == first.numeric_version + 1
    np.testing.assert_allclose(second.value, 1.5**2 * first.value, rtol=1.0e-12)

    compiled = jax.jit(lambda value: phx.linalg.evaluate_pfaffian(value, plan).value)
    np.testing.assert_allclose(compiled(matrix), first.value)
    stacked = jnp.stack((matrix, 2.0 * matrix, -0.5 * matrix))
    mapped = jax.vmap(compiled)(stacked)
    np.testing.assert_allclose(mapped, jnp.asarray([1.0, 4.0, 0.25]) * first.value)

    direction = _skew(jnp.cos(jnp.arange(16.0).reshape(4, 4)))
    _, derivative = jax.jvp(compiled, (matrix,), (direction,))
    expected = 0.5 * first.value * jnp.trace(jnp.linalg.solve(matrix, direction))
    np.testing.assert_allclose(derivative, expected, rtol=1.0e-10, atol=1.0e-12)
    assert bool(first.log_derivative_valid)


def test_resource_refusal_and_require_or_project_skew_policy():
    matrix = jnp.asarray([[0.2, 2.0], [-1.0, -0.1]])
    required = phx.linalg.evaluate_pfaffian(
        matrix,
        phx.linalg.PfaffianPolicy(
            skew_mode="require",
            antisymmetry_tolerance=1.0e-14,
        ),
    )
    assert not bool(required.successful)
    assert jnp.isnan(required.value)
    assert not bool(required.antisymmetric)

    projected = phx.linalg.evaluate_pfaffian(
        matrix,
        phx.linalg.PfaffianPolicy(
            skew_mode="project",
            antisymmetry_tolerance=1.0e-14,
        ),
    )
    expected = 0.5 * (matrix[0, 1] - matrix[1, 0])
    np.testing.assert_allclose(projected.value, expected)
    assert bool(projected.successful)
    assert not bool(projected.antisymmetric)

    nonfinite = phx.linalg.evaluate_pfaffian(
        jnp.asarray([[0.0, jnp.nan], [jnp.nan, 0.0]])
    )
    assert not bool(nonfinite.input_finite)
    assert not bool(nonfinite.successful)
    assert jnp.isnan(nonfinite.value)

    with pytest.raises(ValueError, match="resource rejection"):
        phx.linalg.plan_pfaffian(
            jnp.zeros((8, 8)),
            phx.linalg.PfaffianPolicy(max_dimension=4),
        )
    with pytest.raises(ValueError, match="persistent factors"):
        phx.linalg.plan_pfaffian(
            jnp.zeros((8, 8)),
            phx.linalg.PfaffianPolicy(max_storage_bytes=64),
        )

    with pytest.raises(TypeError, match="floating or complex"):
        phx.linalg.plan_pfaffian(jnp.zeros((4, 4), dtype=jnp.int32))
