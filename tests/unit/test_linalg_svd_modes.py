# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


la = phx.linalg
svd = la.svd


def _fit(
    matrix: Array, mode: svd.SVDDifferentiationMode, count: int = 2, /
) -> svd.SVDSolveResult:
    return svd.svd(
        svd.SVDProblem(la.DenseLinearOperator(matrix)),
        policy=svd.SVDSolvePolicy(count=count, differentiation=mode),
    )


def test_native_projector_repeated_cluster_has_nontrivial_fit_response() -> None:
    matrix = jnp.diag(jnp.asarray([3.0, 3.0, 1.0, 0.5], dtype=jnp.float64))
    direction = jnp.asarray(
        [
            [0.2, -0.4, 0.7, -0.3],
            [0.5, 0.1, -0.6, 0.8],
            [-0.2, 0.9, 0.3, -0.5],
            [0.4, -0.7, 0.6, 0.2],
        ],
        dtype=matrix.dtype,
    )
    vector = jnp.asarray([0.8, -0.3, 0.4, -0.6], dtype=matrix.dtype)
    contraction = jnp.asarray([-0.4, 0.6, 0.2, 0.7], dtype=matrix.dtype)

    def action(value: Array) -> Array:
        result = _fit(value, "projector")
        return svd.projector_action(
            result.right_coordinates, result.right_response, vector
        )

    result = _fit(matrix, "projector")
    assert bool(result.derivative_valid)
    assert np.min(result.diagnostics.isolation_gaps) < 0
    assert float(result.diagnostics.cutoff_gap) > 0
    _, actual = jax.jvp(action, (matrix,), (direction,))
    epsilon = 1e-5
    expected = (
        action(matrix + epsilon * direction) - action(matrix - epsilon * direction)
    ) / (2 * epsilon)
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-8)
    assert np.linalg.norm(expected) > 0.05
    reverse = jax.grad(lambda value: jnp.vdot(contraction, action(value)).real)(matrix)
    assert np.isclose(
        jnp.vdot(reverse, direction).real,
        jnp.vdot(contraction, expected).real,
        rtol=2e-6,
        atol=2e-8,
    )
    _, raw_basis_tangent = jax.jvp(
        lambda value: _fit(value, "projector").right_coordinates, (matrix,), (direction,)
    )
    assert np.array_equal(raw_basis_tangent, np.zeros_like(raw_basis_tangent))
    basis_result = _fit(matrix, "basis")
    assert not bool(basis_result.derivative_valid)
    assert int(basis_result.primal_status) == 0
    assert int(basis_result.status) == int(svd.SVDSolveStatus.DIFFERENTIATION_REJECTED)


@pytest.mark.parametrize("mode", ["none", "singular-values", "basis"])
def test_native_fit_mode_values_and_canonical_basis_derivatives(
    mode: svd.SVDDifferentiationMode,
) -> None:
    matrix = jnp.asarray(
        [[3.0, 0.3, -0.4], [0.2, 2.0, 0.6], [0.4, -0.1, 0.5], [-0.2, 0.3, 0.1]],
        dtype=jnp.float64,
    )
    direction = jnp.asarray(
        [[0.1, -0.2, 0.3], [-0.5, 0.1, 0.4], [0.2, 0.6, -0.1], [0.3, -0.1, 0.4]],
        dtype=matrix.dtype,
    )
    left_probe = jnp.asarray(
        [[0.4, -0.3], [-0.2, 0.7], [0.1, -0.6], [0.8, 0.2]], dtype=matrix.dtype
    )
    right_probe = jnp.asarray([[0.6, -0.1], [0.2, -0.8], [-0.4, 0.5]], dtype=matrix.dtype)

    def output(value: Array) -> Array:
        result = _fit(value, mode)
        return (
            jnp.vdot(
                jnp.asarray([0.7, -0.4], dtype=value.dtype), result.singular_values
            ).real
            + jnp.vdot(left_probe, result.left_coordinates).real
            + jnp.vdot(right_probe, result.right_coordinates).real
        )

    _, tangent = jax.jvp(output, (matrix,), (direction,))
    if mode == "none":
        assert float(tangent) == 0
    else:
        epsilon = 1e-5
        if mode == "singular-values":

            def selected_values(value: Array) -> Array:
                return jnp.vdot(
                    jnp.asarray([0.7, -0.4], dtype=value.dtype),
                    _fit(value, mode).singular_values,
                ).real

            expected = (
                selected_values(matrix + epsilon * direction)
                - selected_values(matrix - epsilon * direction)
            ) / (2 * epsilon)
        else:
            expected = (
                output(matrix + epsilon * direction)
                - output(matrix - epsilon * direction)
            ) / (2 * epsilon)
        assert np.isclose(tangent, expected, rtol=2e-6, atol=2e-8)
    result = _fit(matrix, mode)
    assert np.allclose(
        matrix @ result.right_coordinates,
        result.left_coordinates * result.singular_values,
        atol=1e-12,
    )
    assert np.array_equal(
        result.right_response.frame_correction, np.zeros_like(result.right_coordinates)
    )


def test_native_moving_source_target_metrics_reach_values_and_projector() -> None:
    matrix = jnp.asarray([[3.0, -0.5], [0.2, 1.5], [0.6, -0.2]], dtype=jnp.float64)
    source_weights = jnp.asarray([1.2, 2.0], dtype=matrix.dtype)
    target_weights = jnp.asarray([0.8, 1.3, 1.9], dtype=matrix.dtype)
    source_direction = jnp.asarray([0.3, -0.2], dtype=matrix.dtype)
    target_direction = jnp.asarray([-0.1, 0.4, 0.2], dtype=matrix.dtype)
    probe = jnp.asarray([0.7, -0.4], dtype=matrix.dtype)
    contraction = jnp.asarray([-0.2, 0.6], dtype=matrix.dtype)

    def action(source: Array, target: Array) -> Array:
        source_space = la.ArraySpace(
            (2,), dtype=matrix.dtype, pairing=la.DiagonalPairing(source)
        )
        target_space = la.ArraySpace(
            (3,), dtype=matrix.dtype, pairing=la.DiagonalPairing(target)
        )
        result = svd.svd(
            svd.SVDProblem(
                la.DenseLinearOperator(matrix, source=source_space, target=target_space)
            ),
            policy=svd.SVDSolvePolicy(count=1, differentiation="projector"),
        )
        return jnp.vdot(
            contraction,
            svd.projector_action(
                result.right_coordinates, result.right_response, source_space.riesz(probe)
            ),
        ).real

    _, tangent = jax.jvp(
        action, (source_weights, target_weights), (source_direction, target_direction)
    )
    epsilon = 1e-5
    expected = (
        action(
            source_weights + epsilon * source_direction,
            target_weights + epsilon * target_direction,
        )
        - action(
            source_weights - epsilon * source_direction,
            target_weights - epsilon * target_direction,
        )
    ) / (2 * epsilon)
    assert np.isclose(tangent, expected, rtol=2e-6, atol=2e-8)
    assert abs(float(expected)) > 1e-4


def test_full_square_zero_projector_is_identity_with_zero_spectral_tangent() -> None:
    matrix = jnp.zeros((3, 3), dtype=jnp.float64)
    probe = jnp.asarray([0.4, -0.8, 0.3], dtype=matrix.dtype)

    def action(value: Array) -> Array:
        result = _fit(value, "projector", 3)
        return svd.projector_action(
            result.right_coordinates, result.right_response, probe
        )

    result = _fit(matrix, "projector", 3)
    assert bool(result.derivative_valid)
    primal, tangent = jax.jvp(action, (matrix,), (jnp.ones_like(matrix),))
    assert np.allclose(primal, probe, atol=1e-12)
    assert np.array_equal(tangent, np.zeros_like(tangent))


def test_native_basis_refuses_nonunique_physical_pivot() -> None:
    root_half = jnp.sqrt(jnp.asarray(0.5, jnp.float64))
    right = jnp.asarray(
        [[root_half, -root_half], [root_half, root_half]], dtype=jnp.float64
    )
    matrix = jnp.diag(jnp.asarray([3.0, 1.0], dtype=right.dtype)) @ right.T
    result = _fit(matrix, "basis", 1)
    assert int(result.primal_status) == 0
    assert not bool(result.derivative_valid)
    assert np.allclose(result.diagnostics.pivot_gaps, 0, atol=1e-12)
