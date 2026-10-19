from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.linalg import ArraySpace, RankPolicy
from phydrax.linalg._constraints import _OrthogonalConstraintFrame


def _rows(count: int) -> np.ndarray:
    return np.asarray(
        (
            (1.0, 0.5, -0.4),
            (-2.0, 1.0, 0.2),
            (0.3, -0.7, 1.1),
            (1.2, 0.8, 0.6),
            (-0.4, 1.4, -0.9),
            (0.7, 0.2, 1.3),
        ),
        dtype=np.float64,
    )[:, :count]


def _roundoff(rows: np.ndarray) -> float:
    # Dimension and condition-amplified rounding allowance for the finite
    # Householder/correction sequence and its differentiated normal equations.
    return float(
        256.0 * max(rows.shape) * np.finfo(np.float64).eps * np.linalg.cond(rows) ** 2
    )


def _frame(rows: Array) -> _OrthogonalConstraintFrame:
    return _OrthogonalConstraintFrame(
        ArraySpace((rows.shape[0],), dtype=rows.dtype),
        rows,
        RankPolicy(),
        constraint_id="numeric-aggregate-frame",
    )


def _lift_reference(
    rows: np.ndarray, right: np.ndarray, seed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gram = rows.T @ rows
    multiplier = np.linalg.solve(gram, right)
    lifted = rows @ multiplier
    right_gradient = np.linalg.solve(gram, rows.T @ seed)
    row_gradient = np.outer(seed - rows @ right_gradient, multiplier) - np.outer(
        lifted, right_gradient
    )
    return lifted, multiplier, row_gradient, right_gradient


@pytest.mark.parametrize("count", (2, 3), ids=("two-rows", "three-rows"))
def test_frame_lift_matches_numpy_minimum_norm_and_rank_evidence(count: int) -> None:
    rows = _rows(count)
    right = np.linspace(-0.7, 1.1, count, dtype=np.float64)
    frame = _frame(jnp.asarray(rows))
    lifted = jax.jit(lambda bank, rhs: _frame(bank).lift(rhs))(
        jnp.asarray(rows), jnp.asarray(right)
    )
    expected = np.linalg.lstsq(rows.T, right, rcond=None)[0]
    singular_values = np.linalg.svd(rows, compute_uv=False)
    _, triangular = np.linalg.qr(rows, mode="reduced")
    tolerance = _roundoff(rows)
    assert bool(frame.valid)
    assert int(frame.rank) == count
    np.testing.assert_array_equal(frame.rows, rows)
    np.testing.assert_allclose(lifted, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(rows.T @ lifted, right, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        frame.pivot_norms, np.abs(np.diag(triangular)), atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        frame.singular_values, singular_values, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        frame.rank_cutoff,
        max(rows.shape) * np.finfo(np.float64).eps * singular_values[0],
        atol=0.0,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        frame.condition_estimate,
        singular_values[0] / singular_values[-1],
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("count", (2, 3), ids=("two-rows", "three-rows"))
def test_frame_complement_and_lift_coordinate_transposes(count: int) -> None:
    rows = _rows(count)
    frame = _frame(jnp.asarray(rows))
    prolongation = frame.constraint.prolongation
    dimension = len(rows) - count
    basis = np.asarray(
        jax.vmap(prolongation.mv, in_axes=1, out_axes=1)(
            jnp.eye(dimension, dtype=jnp.float64)
        )
    )
    seed = np.linspace(-0.4, 0.9, len(rows), dtype=np.float64)
    coordinates = np.linspace(0.3, 1.2, dimension, dtype=np.float64)
    right = np.linspace(0.2, 0.8, count, dtype=np.float64)
    right_inverse = np.linalg.lstsq(rows.T, np.eye(count, dtype=np.float64), rcond=None)[
        0
    ]
    tolerance = _roundoff(rows)
    np.testing.assert_allclose(rows.T @ basis, 0.0, atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(
        basis.T @ basis, np.eye(dimension, dtype=np.float64), atol=tolerance, rtol=0.0
    )
    np.testing.assert_allclose(
        prolongation.transpose_mv(jnp.asarray(seed)),
        basis.T @ seed,
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        frame.constraint.reduce_vector(prolongation.mv(jnp.asarray(coordinates))),
        coordinates,
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        basis.T @ frame.lift(jnp.asarray(right)), 0.0, atol=tolerance, rtol=0.0
    )
    transposed = frame.transpose_lift(jnp.asarray(seed))
    np.testing.assert_allclose(
        transposed, right_inverse.T @ seed, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        seed @ frame.lift(jnp.asarray(right)),
        right @ transposed,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("count", (2, 3), ids=("two-rows", "three-rows"))
def test_preconstructed_frame_numerical_state_changes_under_one_compiled_action(
    count: int,
) -> None:
    rows = _rows(count)
    changed_rows = rows + 0.17 * np.sin(np.arange(rows.size).reshape(rows.shape) + 0.4)
    right = jnp.linspace(-0.2, 0.8, count, dtype=jnp.float64)
    coordinates = jnp.linspace(0.1, 0.7, len(rows) - count, dtype=jnp.float64)
    seed = jnp.linspace(-0.5, 0.9, len(rows), dtype=jnp.float64)

    @eqx.filter_jit
    def apply(
        frame: _OrthogonalConstraintFrame,
        rhs: Array,
        reduced: Array,
        value: Array,
    ) -> tuple[Array, Array, Array]:
        return (
            frame.lift(rhs),
            frame.constraint.prolongation.mv(reduced),
            frame.transpose_lift(value),
        )

    first = apply(_frame(jnp.asarray(rows)), right, coordinates, seed)
    changed = apply(_frame(jnp.asarray(changed_rows)), right, coordinates, seed)
    tolerance = _roundoff(changed_rows)
    expected = np.linalg.lstsq(changed_rows.T, np.asarray(right), rcond=None)[0]
    right_inverse = np.linalg.lstsq(
        changed_rows.T, np.eye(count, dtype=np.float64), rcond=None
    )[0]
    np.testing.assert_allclose(changed[0], expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(changed_rows.T @ changed[1], 0.0, atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(
        changed[2], right_inverse.T @ seed, atol=tolerance, rtol=tolerance
    )
    assert np.linalg.norm(np.asarray(changed[0] - first[0])) > 100.0 * tolerance
    assert np.linalg.norm(np.asarray(changed[1] - first[1])) > 100.0 * tolerance
    assert np.linalg.norm(np.asarray(changed[2] - first[2])) > 100.0 * tolerance


@pytest.mark.parametrize("count", (2, 3), ids=("two-rows", "three-rows"))
def test_frame_full_row_and_rhs_jvp_vjp_match_independent_normal_equations(
    count: int,
) -> None:
    rows = _rows(count)
    right = np.linspace(-0.4, 0.9, count, dtype=np.float64)
    row_direction = 0.2 * np.cos(np.arange(rows.size).reshape(rows.shape) + 0.3)
    right_direction = np.linspace(0.3, -0.2, count, dtype=np.float64)
    seed = np.linspace(-0.8, 0.6, len(rows), dtype=np.float64)

    def response(bank: Array, rhs: Array) -> Array:
        return _frame(bank).lift(rhs)

    @jax.jit
    def differentiate(
        bank: Array, rhs: Array, bank_direction: Array, rhs_direction: Array, value: Array
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        return (
            jax.jvp(response, (bank, rhs), (bank_direction, rhs_direction)),
            jax.vjp(response, bank, rhs)[1](value),
        )

    (lifted, tangent), (row_gradient, right_gradient) = differentiate(
        jnp.asarray(rows),
        jnp.asarray(right),
        jnp.asarray(row_direction),
        jnp.asarray(right_direction),
        jnp.asarray(seed),
    )
    expected, multiplier, expected_rows, expected_right = _lift_reference(
        rows, right, seed
    )
    gram_direction = row_direction.T @ rows + rows.T @ row_direction
    multiplier_direction = np.linalg.solve(
        rows.T @ rows, right_direction - gram_direction @ multiplier
    )
    expected_tangent = row_direction @ multiplier + rows @ multiplier_direction
    tolerance = _roundoff(rows)
    np.testing.assert_allclose(lifted, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(tangent, expected_tangent, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        row_gradient, expected_rows, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        right_gradient, expected_right, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        rows.T @ tangent + row_direction.T @ lifted,
        right_direction,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("count", (2, 3), ids=("two-rows", "three-rows"))
def test_frame_complement_row_derivatives_preserve_kernel_and_orthonormality(
    count: int,
) -> None:
    rows = jnp.asarray(_rows(count))
    direction = 0.1 * jnp.sin(jnp.arange(rows.size).reshape(rows.shape) + 0.2)
    coordinates = jnp.linspace(-0.6, 0.8, len(rows) - count, dtype=jnp.float64)
    seed = jnp.linspace(-0.4, 0.9, len(rows), dtype=jnp.float64)

    def response(bank: Array) -> Array:
        return _frame(bank).constraint.prolongation.mv(coordinates)

    (value, tangent), gradient = jax.jit(
        lambda bank, vector: (
            jax.jvp(response, (bank,), (vector,)),
            jax.vjp(response, bank)[1](seed)[0],
        )
    )(rows, direction)
    tolerance = _roundoff(np.asarray(rows))
    np.testing.assert_allclose(
        rows.T @ tangent + direction.T @ value, 0.0, atol=tolerance, rtol=0.0
    )
    np.testing.assert_allclose(value @ tangent, 0.0, atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(
        seed @ tangent, jnp.sum(gradient * direction), atol=tolerance, rtol=tolerance
    )
    # A centered difference balances O(h^2) truncation with O(eps/h) rounding.
    step = np.finfo(np.float64).eps ** (1.0 / 3.0)
    forward = response(rows + step * direction)
    backward = response(rows - step * direction)
    finite_difference = (forward - backward) / (2.0 * step)
    difference_bound = 32.0 * np.finfo(np.float64).eps ** (2.0 / 3.0)
    np.testing.assert_allclose(
        tangent, finite_difference, atol=difference_bound, rtol=difference_bound
    )


@pytest.mark.parametrize("count", (2, 3), ids=("two-rows", "three-rows"))
def test_dependent_frame_retains_rank_and_refuses_even_compatible_lifts(
    count: int,
) -> None:
    rows = _rows(count)
    rows[:, -1] = 2.0 * rows[:, 0]
    frame = _frame(jnp.asarray(rows))
    compatible = rows.T @ np.linspace(0.2, 0.8, len(rows), dtype=np.float64)
    assert not bool(frame.valid)
    assert int(frame.rank) == count - 1
    assert float(frame.singular_values[-1]) <= float(frame.rank_cutoff)
    assert np.all(np.isfinite(frame.singular_values))
    retained = np.asarray(frame.singular_values) > float(frame.rank_cutoff)
    np.testing.assert_allclose(
        frame.condition_estimate,
        frame.singular_values[0] / frame.singular_values[np.flatnonzero(retained)[-1]],
        atol=_roundoff(_rows(count)),
        rtol=_roundoff(_rows(count)),
    )
    assert np.all(np.isnan(frame.lift(jnp.asarray(compatible))))
    assert np.all(np.isnan(frame.lift(jnp.zeros((count,), dtype=jnp.float64))))
    assert np.all(
        np.isnan(frame.transpose_lift(jnp.ones((len(rows),), dtype=jnp.float64)))
    )


def test_zero_frame_retains_zero_rank_and_refuses_zero_lifts() -> None:
    frame = _frame(jnp.zeros((4, 2), dtype=jnp.float64))
    assert not bool(frame.valid)
    assert int(frame.rank) == 0
    np.testing.assert_array_equal(frame.singular_values, np.zeros((2,), dtype=np.float64))
    assert float(frame.rank_cutoff) == 0.0
    assert np.isinf(float(frame.condition_estimate))
    assert np.all(np.isnan(frame.lift(jnp.zeros((2,), dtype=jnp.float64))))
    assert np.all(np.isnan(frame.transpose_lift(jnp.zeros((4,), dtype=jnp.float64))))


@pytest.mark.parametrize(
    ("relative", "absolute"),
    ((1e-6, 0.0), (0.0, 1e-6)),
    ids=("relative-cutoff", "absolute-cutoff"),
)
def test_frame_uses_frozen_rank_policy_for_unaltered_source_rows(
    relative: float, absolute: float
) -> None:
    rows = jnp.asarray(((1.0, 0.0), (0.0, 1e-7), (0.0, 0.0)), dtype=jnp.float64)
    frame = _OrthogonalConstraintFrame(
        ArraySpace((3,), dtype=jnp.float64),
        rows,
        RankPolicy(relative_cutoff=relative, absolute_cutoff=absolute),
        constraint_id="source-row-rank-policy",
    )
    assert not bool(frame.valid)
    assert int(frame.rank) == 1
    np.testing.assert_array_equal(frame.rows, rows)
    # A diagonal two-mode spectrum requires only scaling and unscaling roundoff.
    roundoff = 8.0 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        frame.singular_values, (1.0, 1e-7), rtol=roundoff, atol=0.0
    )
    np.testing.assert_allclose(frame.rank_cutoff, 1e-6, rtol=roundoff, atol=0.0)
    assert np.all(np.isnan(frame.lift(jnp.ones((2,), dtype=jnp.float64))))


@pytest.mark.parametrize("nonfinite", (np.inf, np.nan), ids=("infinite-row", "nan-row"))
def test_nonfinite_frame_preserves_undefined_spectrum_and_lifts(nonfinite: float) -> None:
    rows = _rows(2)
    rows[0, 1] = nonfinite
    frame = _frame(jnp.asarray(rows))
    assert not bool(frame.valid)
    assert np.all(np.isnan(frame.singular_values))
    assert np.isnan(float(frame.rank_cutoff))
    assert np.all(np.isnan(frame.lift(jnp.ones((2,), dtype=jnp.float64))))
    assert np.all(
        np.isnan(frame.transpose_lift(jnp.ones((len(rows),), dtype=jnp.float64)))
    )


@pytest.mark.parametrize("count", (2, 3), ids=("two-dimensions", "three-dimensions"))
def test_full_rank_frame_has_native_empty_complement(count: int) -> None:
    rows = np.asarray(
        ((1.0, 0.3, -0.2), (-0.4, 1.2, 0.6), (0.7, -0.5, 1.1)), dtype=np.float64
    )[:count, :count]
    frame = _frame(jnp.asarray(rows))
    empty = jnp.empty((0,), dtype=jnp.float64)
    right = jnp.linspace(-0.2, 0.7, count, dtype=jnp.float64)
    seed = jnp.linspace(0.1, 0.8, count, dtype=jnp.float64)
    assert bool(frame.valid)
    assert int(frame.rank) == count
    assert frame.constraint.reduced_space.size == 0
    np.testing.assert_array_equal(
        frame.constraint.prolongation.mv(empty), np.zeros(count)
    )
    assert frame.constraint.prolongation.transpose_mv(seed).shape == (0,)
    assert frame.constraint.reduce_vector(seed).shape == (0,)
    tolerance = _roundoff(rows)
    np.testing.assert_allclose(
        frame.constraint.expand(empty, frame.lift(right)),
        np.linalg.solve(rows.T, right),
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_allclose(
        frame.transpose_lift(seed),
        np.linalg.solve(rows, seed),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("count", (2, 3), ids=("two-rows", "three-rows"))
@pytest.mark.parametrize("scale", (1e-200, 1e200), ids=("tiny-rows", "large-rows"))
def test_frame_extreme_row_scales_preserve_lift_transpose_and_full_ad(
    count: int, scale: float
) -> None:
    rows = _rows(count)
    right = np.linspace(-0.5, 0.8, count, dtype=np.float64)
    row_direction = 0.1 * np.sin(np.arange(rows.size).reshape(rows.shape) + 0.7)
    right_direction = np.linspace(0.3, -0.2, count, dtype=np.float64)
    seed = np.linspace(-0.6, 0.9, len(rows), dtype=np.float64)
    frame = _frame(scale * jnp.asarray(rows))
    tolerance = _roundoff(rows)
    assert bool(frame.valid)
    np.testing.assert_array_equal(frame.rows, scale * rows)
    np.testing.assert_allclose(
        frame.singular_values / scale,
        np.linalg.svd(rows, compute_uv=False),
        atol=tolerance,
        rtol=tolerance,
    )
    _, triangular = np.linalg.qr(rows, mode="reduced")
    np.testing.assert_allclose(
        frame.pivot_norms / scale,
        np.abs(np.diag(triangular)),
        atol=tolerance,
        rtol=tolerance,
    )
    right_inverse = np.linalg.lstsq(rows.T, np.eye(count, dtype=np.float64), rcond=None)[
        0
    ]
    np.testing.assert_allclose(
        frame.transpose_lift(scale * jnp.asarray(seed)),
        right_inverse.T @ seed,
        atol=tolerance,
        rtol=tolerance,
    )

    def response(bank: Array, rhs: Array) -> Array:
        return _frame(scale * bank).lift(scale * rhs)

    (lifted, tangent), (row_gradient, right_gradient) = jax.jit(
        lambda bank, rhs, dbank, drhs, value: (
            jax.jvp(response, (bank, rhs), (dbank, drhs)),
            jax.vjp(response, bank, rhs)[1](value),
        )
    )(
        jnp.asarray(rows),
        jnp.asarray(right),
        jnp.asarray(row_direction),
        jnp.asarray(right_direction),
        jnp.asarray(seed),
    )
    expected, multiplier, expected_rows, expected_right = _lift_reference(
        rows, right, seed
    )
    gram_direction = row_direction.T @ rows + rows.T @ row_direction
    expected_tangent = row_direction @ multiplier + rows @ np.linalg.solve(
        rows.T @ rows, right_direction - gram_direction @ multiplier
    )
    np.testing.assert_allclose(lifted, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(tangent, expected_tangent, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        row_gradient, expected_rows, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        right_gradient, expected_right, atol=tolerance, rtol=tolerance
    )
