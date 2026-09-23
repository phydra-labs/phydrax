#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.linalg._substructuring import DeluxeScalingPlan, SubstructuredSPDSystem


la = phx.linalg


def test_complex_orthonormal_frame_normalizes_qr_phases():
    matrix = jnp.asarray(
        [
            [1.0 + 2.0j, 0.5 - 0.25j],
            [-0.75 + 0.5j, 2.0 + 1.0j],
            [1.5 - 0.5j, -1.0 + 0.75j],
        ],
        dtype=jnp.complex128,
    )

    result = la.orthonormal_frame(matrix)
    reduced = jnp.conj(result.tangents.T) @ matrix
    full_frame = jnp.concatenate((result.tangents, result.normal_basis), axis=-1)

    assert bool(result.successful)
    assert jnp.allclose(
        jnp.conj(result.tangents.T) @ result.tangents,
        jnp.eye(2),
        atol=1e-12,
    )
    assert jnp.all(jnp.real(jnp.diag(reduced)) >= 0.0)
    assert jnp.allclose(jnp.imag(jnp.diag(reduced)), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.linalg.det(full_frame), 1.0, atol=1e-12)


def test_banded_operator_rejects_dtype_mismatch_and_budgets_every_batch_factor():
    real_space = la.ArraySpace((3,), dtype=jnp.float64)
    complex_bands = jnp.ones((3, 3), dtype=jnp.complex128)
    with pytest.raises(TypeError, match="produces dtype"):
        la.BandedLinearOperator(
            complex_bands,
            lower_bandwidth=1,
            upper_bandwidth=1,
            space=real_space,
        )

    bands = jnp.asarray(
        [
            [[0.0, -1.0, -1.0], [4.0, 4.0, 4.0], [-1.0, -1.0, 0.0]],
            [[0.0, 0.5, 0.5], [3.0, 3.5, 4.0], [0.25, 0.25, 0.0]],
        ],
        dtype=jnp.float64,
    )
    operator = la.BandedLinearOperator(
        bands,
        lower_bandwidth=1,
        upper_bandwidth=1,
    )
    with pytest.raises(ValueError, match="factorization estimate"):
        la.plan(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.StructuredDirect(),
                resources=la.SolveResourcePolicy(
                    factorization_bytes=100,
                    workspace_bytes=4096,
                ),
            ),
        )


def test_deluxe_scaling_resolves_unsorted_local_to_global_maps():
    system = SubstructuredSPDSystem(
        (
            jnp.asarray([[3.0, 0.25], [0.25, 2.0]]),
            jnp.asarray([[4.0, -0.5], [-0.5, 2.5]]),
        ),
        (
            jnp.asarray([2, 0]),
            jnp.asarray([2, 1]),
        ),
    )

    plan = DeluxeScalingPlan(system)

    assert len(plan.interfaces) == 1
    interface = plan.interfaces[0]
    assert jnp.array_equal(interface.global_dof_ids, jnp.asarray([2]))
    assert jnp.array_equal(interface.left_local_indices, jnp.asarray([0]))
    assert jnp.array_equal(interface.right_local_indices, jnp.asarray([0]))
    assert interface.partition_unity_error < 1e-12


def test_complex_tridiagonal_lines_use_real_pivot_tolerances():
    lower = jnp.asarray([0.0, 1.0 - 0.5j, -0.25 + 0.2j])
    diagonal = jnp.asarray([3.0 + 0.5j, 4.0 - 0.25j, 2.5 + 0.75j])
    upper = jnp.asarray([0.5 + 0.25j, -1.0 + 0.1j, 0.0])
    rhs = jnp.asarray([1.0 + 0.5j, -2.0 + 1.0j, 0.25 - 0.75j])
    matrix = jnp.diag(diagonal) + jnp.diag(lower[1:], -1) + jnp.diag(upper[:-1], 1)

    result = la.solve_tridiagonal_lines(lower, diagonal, upper, rhs)

    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, rhs), atol=1e-12)
    with pytest.raises(ValueError, match="pivot_tolerance"):
        la.solve_tridiagonal_lines(
            lower,
            diagonal,
            upper,
            rhs,
            pivot_tolerance=-1.0,
        )
