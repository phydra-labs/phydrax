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


def _self_adjoint_operator(matrix, operator_id):
    return la.DenseLinearOperator(
        matrix,
        properties=la.OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        ),
        operator_id=operator_id,
    )


def _projection(matrix, initial, max_dimension, operator_id):
    operator = _self_adjoint_operator(matrix, operator_id)
    projection = la.prepare_krylov_projection(
        operator,
        initial,
        la.KrylovProjectionPolicy("lanczos", max_dimension=max_dimension),
    )
    return operator, projection


def _dense_resolvent_form(matrix, initial, shifts):
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    return jax.vmap(
        lambda shift: jnp.vdot(
            initial,
            jnp.linalg.solve(shift * identity - matrix, initial),
        )
    )(shifts)


def test_lanczos_resolvent_form_matches_dense_and_transforms():
    matrix = jnp.asarray(
        [
            [2.0, 1.0 + 0.5j, 0.0],
            [1.0 - 0.5j, 3.0, -0.25j],
            [0.0, 0.25j, 1.5],
        ],
        dtype=jnp.complex128,
    )
    initial = jnp.asarray([1.0 + 0.2j, -0.5j, 0.75], dtype=jnp.complex128)
    _, projection = _projection(matrix, initial, 3, "resolvent-exact")
    shifts = jnp.asarray(
        [-0.5 + 0.3j, 1.25 + 0.3j, 4.5 + 0.3j],
        dtype=jnp.complex128,
    )

    eager = la.lanczos_resolvent_form(projection, shifts)
    compiled = eqx.filter_jit(la.lanczos_resolvent_form)(projection, shifts)
    vmapped = jax.vmap(lambda shift: la.lanczos_resolvent_form(projection, shift).value)(
        shifts
    )
    expected = _dense_resolvent_form(matrix, initial, shifts)

    np.testing.assert_allclose(eager.value, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(compiled.value, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(vmapped, expected, rtol=2e-11, atol=2e-11)
    assert bool(jnp.all(eager.successful))
    assert bool(eager.all_successful)
    assert bool(eager.diagnostics.projection_exact)
    assert bool(jnp.all(jnp.imag(eager.value) < 0.0))


def test_truncated_lanczos_resolvent_is_finite_without_false_success():
    diagonal = jnp.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float64)
    off_diagonal = jnp.asarray([0.4, -0.3, 0.5, 0.2], dtype=jnp.float64)
    matrix = jnp.diag(diagonal)
    matrix = matrix + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    initial = jnp.asarray([1.0, -0.3, 0.2, 0.5, -0.1], dtype=jnp.float64)
    _, projection = _projection(matrix, initial, 3, "resolvent-truncated")
    shifts = jnp.asarray([-0.5 + 0.4j, 3.0 + 0.4j], dtype=jnp.complex128)

    result = la.lanczos_resolvent_form(projection, shifts)
    dimension = int(projection.effective_dimension)
    projected = projection.projected_operator[:dimension, :dimension]
    norm_squared = jnp.vdot(initial, initial).real
    expected = norm_squared * jax.vmap(
        lambda shift: jnp.linalg.solve(
            shift * jnp.eye(dimension, dtype=shifts.dtype)
            - projected.astype(shifts.dtype),
            jnp.eye(dimension, dtype=shifts.dtype)[0],
        )[0]
    )(shifts)

    np.testing.assert_allclose(result.value, expected, rtol=2e-11, atol=2e-11)
    assert bool(jnp.all(result.diagnostics.finite))
    assert not bool(jnp.any(result.successful))
    assert bool(jnp.all(result.status == int(la.LanczosResolventStatus.TRUNCATED)))
    assert not bool(result.diagnostics.projection_exact)
    assert bool(jnp.all(result.diagnostics.indicator_available))
    assert bool(jnp.all(jnp.isfinite(result.diagnostics.truncation_indicator)))


def test_terminal_resolvent_closes_uniform_chain_without_success_claim():
    coupling = jnp.asarray(0.7, dtype=jnp.float64)
    matrix = jnp.diag(jnp.full((6,), coupling), 1)
    matrix = matrix + matrix.T
    initial = jnp.eye(7, dtype=jnp.float64)[0]
    _, projection = _projection(matrix, initial, 3, "resolvent-terminal")
    shift = jnp.asarray(0.4 + 0.6j, dtype=jnp.complex128)
    discriminant = jnp.sqrt(shift**2 - 4.0 * coupling**2)
    terminal = (shift - discriminant) / (2.0 * coupling**2)

    result = la.lanczos_resolvent_form(
        projection,
        shift,
        terminal_resolvent=terminal,
    )

    np.testing.assert_allclose(result.value, terminal, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        result.diagnostics.boundary_coupling,
        coupling,
        rtol=2e-11,
        atol=2e-11,
    )
    assert int(result.status) == int(la.LanczosResolventStatus.TRUNCATED)
    assert not bool(result.successful)
    assert not bool(result.diagnostics.indicator_available)
    assert result.provenance.termination == "explicit-terminal-resolvent"


def test_lanczos_resolvent_reports_lane_failures_and_rejects_wrong_projection():
    matrix = jnp.asarray([[2.0]], dtype=jnp.float64)
    initial = jnp.asarray([1.0], dtype=jnp.float64)
    operator, projection = _projection(matrix, initial, 1, "resolvent-statuses")
    shifts = jnp.asarray([2.0 + 0.0j, 2.0 + 1.0j, jnp.nan + 0.0j])

    result = eqx.filter_jit(la.lanczos_resolvent_form)(projection, shifts)

    np.testing.assert_array_equal(
        result.status,
        jnp.asarray(
            [
                int(la.LanczosResolventStatus.SINGULAR),
                int(la.LanczosResolventStatus.SUCCESS),
                int(la.LanczosResolventStatus.NONFINITE),
            ],
            dtype=jnp.int32,
        ),
    )
    assert bool(jnp.isnan(result.value[0]))
    np.testing.assert_allclose(result.value[1], -1.0j)
    assert bool(jnp.isnan(result.value[2]))

    arnoldi = la.prepare_krylov_projection(
        operator,
        initial,
        la.KrylovProjectionPolicy("arnoldi", max_dimension=1),
    )
    with pytest.raises(ValueError, match="require a Lanczos projection"):
        la.lanczos_resolvent_form(arnoldi, shifts)
    with pytest.raises(ValueError, match="exactly the shifts shape"):
        la.lanczos_resolvent_form(
            projection,
            shifts[:2],
            terminal_resolvent=jnp.zeros((3,)),
        )


def test_lanczos_resolvent_shift_jvp_matches_dense_derivative():
    matrix = jnp.asarray([[2.0, 0.4], [0.4, 1.0]], dtype=jnp.float64)
    initial = jnp.asarray([1.0, -0.25], dtype=jnp.float64)
    _, projection = _projection(matrix, initial, 2, "resolvent-jvp")

    def evaluate(real_shift):
        shift = real_shift + 0.35j
        return la.lanczos_resolvent_form(projection, shift).value

    real_shift = jnp.asarray(0.7, dtype=jnp.float64)
    _, tangent = jax.jvp(
        evaluate,
        (real_shift,),
        (jnp.asarray(1.0, dtype=real_shift.dtype),),
    )
    shift = real_shift + 0.35j
    shifted = shift * jnp.eye(2, dtype=jnp.complex128) - matrix
    expected = -jnp.vdot(
        initial,
        jnp.linalg.solve(shifted, jnp.linalg.solve(shifted, initial)),
    )

    np.testing.assert_allclose(tangent, expected, rtol=3e-11, atol=3e-11)
