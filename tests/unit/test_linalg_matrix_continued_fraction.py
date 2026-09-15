#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


la = phx.linalg


def _assemble_block_tridiagonal(diagonal, upper, lower):
    block_count, block_size, _ = diagonal.shape
    dimension = block_count * block_size
    matrix = jnp.zeros((dimension, dimension), dtype=diagonal.dtype)
    for index in range(block_count):
        start = index * block_size
        matrix = matrix.at[
            start : start + block_size,
            start : start + block_size,
        ].set(diagonal[index])
        if index < block_count - 1:
            next_start = start + block_size
            matrix = matrix.at[
                start : start + block_size,
                next_start : next_start + block_size,
            ].set(upper[index])
            matrix = matrix.at[
                next_start : next_start + block_size,
                start : start + block_size,
            ].set(lower[index])
    return matrix


def _leading_resolvent_blocks(matrix, shifts, block_size):
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    right_hand_side = identity[:, :block_size]
    return jax.vmap(
        lambda shift: jnp.linalg.solve(
            shift * identity - matrix,
            right_hand_side,
        )[:block_size]
    )(shifts)


def test_matrix_continued_fraction_matches_noncommuting_block_resolvent():
    diagonal = jnp.asarray(
        [
            [[1.0 + 0.1j, 0.2], [-0.3j, 2.0 - 0.1j]],
            [[2.5, -0.2j], [0.1 + 0.05j, 3.0]],
            [[3.5 - 0.1j, 0.15], [-0.25j, 4.0 + 0.2j]],
        ],
        dtype=jnp.complex128,
    )
    upper = jnp.asarray(
        [
            [[0.3, 0.1j], [-0.2, 0.25]],
            [[-0.1j, 0.2], [0.15, -0.3j]],
        ],
        dtype=jnp.complex128,
    )
    lower = jnp.asarray(
        [
            [[0.15, -0.05], [0.2j, 0.35]],
            [[0.25, 0.1j], [-0.2, 0.1]],
        ],
        dtype=jnp.complex128,
    )
    shifts = jnp.asarray([-0.5 + 0.4j, 2.0 + 0.4j, 5.0 + 0.4j])
    matrix = _assemble_block_tridiagonal(diagonal, upper, lower)

    eager = la.matrix_continued_fraction(
        diagonal,
        upper,
        shifts,
        lower_couplings=lower,
    )
    compiled = eqx.filter_jit(la.matrix_continued_fraction)(
        diagonal,
        upper,
        shifts,
        lower_couplings=lower,
    )
    vmapped = jax.vmap(
        lambda shift: (
            la.matrix_continued_fraction(
                diagonal,
                upper,
                shift,
                lower_couplings=lower,
            ).value
        )
    )(shifts)
    expected = _leading_resolvent_blocks(matrix, shifts, block_size=2)

    np.testing.assert_allclose(eager.value, expected, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(compiled.value, expected, rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(vmapped, expected, rtol=3e-11, atol=3e-11)
    assert bool(eager.all_successful)
    assert bool(jnp.all(eager.diagnostics.maximum_relative_inverse_residual < 1e-12))
    assert eager.provenance.coupling == "explicit-upper-lower"


def test_adjoint_coupling_default_preserves_matrix_resolvent_sign():
    diagonal = jnp.asarray(
        [
            [[1.0, 0.2j], [-0.2j, 1.5]],
            [[2.0, 0.1 - 0.05j], [0.1 + 0.05j, 2.5]],
            [[3.0, -0.15j], [0.15j, 3.5]],
        ],
        dtype=jnp.complex128,
    )
    upper = jnp.asarray(
        [
            [[0.3, 0.1j], [0.05, -0.2]],
            [[-0.1j, 0.25], [0.2, 0.15j]],
        ],
        dtype=jnp.complex128,
    )
    lower = jnp.conj(jnp.swapaxes(upper, -1, -2))
    shifts = jnp.asarray([0.5 + 0.25j, 2.25 + 0.25j, 4.0 + 0.25j])

    implicit = la.matrix_continued_fraction(diagonal, upper, shifts)
    explicit = la.matrix_continued_fraction(
        diagonal,
        upper,
        shifts,
        lower_couplings=lower,
    )
    hermitian_imaginary_part = (
        implicit.value - jnp.conj(jnp.swapaxes(implicit.value, -1, -2))
    ) / (2.0j)
    spectral_blocks = -hermitian_imaginary_part

    np.testing.assert_allclose(implicit.value, explicit.value, rtol=2e-12, atol=2e-12)
    assert implicit.value.shape == (3, 2, 2)
    assert bool(jnp.all(jnp.linalg.eigvalsh(spectral_blocks) >= -1e-12))
    assert implicit.provenance.coupling == "adjoint-paired-upper-lower"


def test_terminal_self_energy_matches_eliminated_block():
    diagonal = jnp.asarray(
        [
            [[1.0, 0.1], [0.1, 1.5]],
            [[2.0, -0.2j], [0.2j, 2.5]],
            [[3.0, 0.15], [0.15, 3.5]],
        ],
        dtype=jnp.complex128,
    )
    upper = jnp.asarray(
        [
            [[0.25, 0.1j], [-0.05, 0.2]],
            [[0.1, -0.15j], [0.2j, 0.3]],
        ],
        dtype=jnp.complex128,
    )
    lower = jnp.conj(jnp.swapaxes(upper, -1, -2))
    shifts = jnp.asarray([0.25 + 0.3j, 2.75 + 0.3j])
    identity = jnp.eye(2, dtype=jnp.complex128)
    tail_resolvent = jax.vmap(
        lambda shift: jnp.linalg.solve(shift * identity - diagonal[2], identity)
    )(shifts)
    self_energy = jax.vmap(lambda value: upper[1] @ value @ lower[1])(tail_resolvent)

    result = la.matrix_continued_fraction(
        diagonal[:2],
        upper[:1],
        shifts,
        lower_couplings=lower[:1],
        terminal_self_energy=self_energy,
    )
    full_matrix = _assemble_block_tridiagonal(diagonal, upper, lower)
    expected = _leading_resolvent_blocks(full_matrix, shifts, block_size=2)

    np.testing.assert_allclose(result.value, expected, rtol=3e-11, atol=3e-11)
    assert bool(result.all_successful)
    assert result.provenance.termination == "explicit-terminal-self-energy"


def test_matrix_continued_fraction_isolates_failures_and_validates_shapes():
    diagonal = jnp.asarray([[[1.0, 0.0], [0.0, 2.0]]])
    shifts = jnp.asarray([1.0 + 0.0j, 3.0 + 0.0j, jnp.nan + 0.0j])

    result = eqx.filter_jit(la.matrix_continued_fraction)(diagonal, None, shifts)

    np.testing.assert_array_equal(
        result.status,
        jnp.asarray(
            [
                int(la.MatrixContinuedFractionStatus.SINGULAR),
                int(la.MatrixContinuedFractionStatus.SUCCESS),
                int(la.MatrixContinuedFractionStatus.NONFINITE),
            ],
            dtype=jnp.int32,
        ),
    )
    assert bool(jnp.all(jnp.isnan(result.value[0])))
    np.testing.assert_allclose(result.value[1], jnp.diag(jnp.asarray([0.5, 1.0])))
    assert bool(jnp.all(jnp.isnan(result.value[2])))
    assert int(result.diagnostics.singular_level[0]) == 0
    assert int(result.diagnostics.nonfinite_level[2]) == -1

    with pytest.raises(ValueError, match="upper_couplings"):
        la.matrix_continued_fraction(
            jnp.zeros((2, 2, 2)),
            jnp.zeros((2, 2, 2)),
            shifts[:2],
        )
    with pytest.raises(ValueError, match="lower_couplings"):
        la.matrix_continued_fraction(
            jnp.zeros((2, 2, 2)),
            jnp.zeros((1, 2, 2)),
            shifts[:2],
            lower_couplings=jnp.zeros((2, 2, 2)),
        )
    with pytest.raises(ValueError, match="terminal_self_energy"):
        la.matrix_continued_fraction(
            diagonal,
            None,
            shifts[:2],
            terminal_self_energy=jnp.zeros((2, 2, 3)),
        )


def test_matrix_continued_fraction_shift_jvp_matches_dense_derivative():
    diagonal = jnp.asarray(
        [
            [[1.0, 0.2], [0.2, 1.5]],
            [[2.0, -0.1], [-0.1, 2.5]],
        ],
        dtype=jnp.float64,
    )
    upper = jnp.asarray([[[0.3, 0.1], [-0.2, 0.25]]], dtype=jnp.float64)
    lower = jnp.swapaxes(upper, -1, -2)
    matrix = _assemble_block_tridiagonal(diagonal, upper, lower)

    def evaluate(real_shift):
        return la.matrix_continued_fraction(
            diagonal,
            upper,
            real_shift + 0.35j,
            lower_couplings=lower,
        ).value

    real_shift = jnp.asarray(0.6, dtype=jnp.float64)
    _, tangent = jax.jvp(
        evaluate,
        (real_shift,),
        (jnp.asarray(1.0, dtype=real_shift.dtype),),
    )
    shift = real_shift + 0.35j
    resolvent = jnp.linalg.inv(
        shift * jnp.eye(matrix.shape[0], dtype=jnp.complex128) - matrix
    )
    expected = -(resolvent @ resolvent)[:2, :2]

    np.testing.assert_allclose(tangent, expected, rtol=3e-11, atol=3e-11)
