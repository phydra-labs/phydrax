import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.tensor_train as tt


def test_tt_svd_exact_algebra_operator_and_bounded_dense_reconstruction():
    dense = jnp.arange(24, dtype=jnp.float32).reshape((2, 3, 4))
    decomposition = tt.tt_svd(dense, max_ranks=(2, 4), relative_tolerance=0.0)
    train = decomposition.tensor

    assert jnp.allclose(train.to_dense(max_entries=24), dense, atol=2e-5)
    assert jnp.allclose(train.entry((1, 2, 3)), dense[1, 2, 3], atol=2e-5)
    assert jnp.allclose(
        train.evaluate(jnp.asarray([[0, 1, 2], [1, 2, 3]])),
        jnp.asarray([dense[0, 1, 2], dense[1, 2, 3]]),
        atol=2e-5,
    )
    assert jnp.allclose((train + train).to_dense(max_entries=24), 2 * dense, atol=4e-5)
    assert jnp.allclose(
        train.hadamard(train).to_dense(max_entries=24), dense**2, atol=1e-3
    )
    assert jnp.allclose(train.inner(train), jnp.sum(dense**2), rtol=2e-5)

    factors = (
        jnp.asarray([[1.0, 2.0], [0.0, 1.0]]),
        jnp.asarray([[2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 2.0]]),
        jnp.eye(4),
    )
    operator = tt.kronecker_operator(factors)
    applied = operator.apply(train)
    matrix = operator.to_matrix(max_entries=24 * 24)
    assert jnp.allclose(
        applied.to_dense(max_entries=24).reshape((-1,)),
        matrix @ dense.reshape((-1,)),
        atol=1e-4,
    )
    composed = operator.compose(tt.identity_operator((2, 3, 4)))
    assert jnp.allclose(composed.to_matrix(max_entries=24 * 24), matrix, atol=1e-5)
    output_indices = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    input_indices = jnp.asarray([[0, 1, 3]], dtype=jnp.int32)
    operator_dense = operator.to_dense(max_entries=24 * 24)
    assert jnp.allclose(
        operator.evaluate(output_indices, input_indices),
        operator_dense[1, 2, 3, 0, 1, 3][None],
    )
    rounded_operator = operator.round(max_ranks=1, relative_tolerance=0.0)
    assert rounded_operator.evidence.exact
    assert jnp.allclose(rounded_operator.operator.to_matrix(max_entries=24 * 24), matrix)


def test_rounding_reports_rss_bound_including_every_discarded_mode():
    dense = jnp.asarray(
        np.random.default_rng(4).normal(size=(4, 5, 3)), dtype=jnp.float32
    )
    exact = tt.tt_svd(dense, max_ranks=(4, 3), relative_tolerance=0.0).tensor
    rounded = exact.round(max_ranks=(1, 1), relative_tolerance=0.0)
    reconstructed = rounded.tensor.to_dense(max_entries=60)
    measured = jnp.sqrt(jnp.sum(jnp.abs(dense - reconstructed) ** 2))
    rss = jnp.sqrt(jnp.sum(rounded.evidence.per_cut_discarded_frobenius**2))

    assert rounded.evidence.per_cut_discarded_frobenius.shape == (2,)
    assert jnp.allclose(rounded.evidence.frobenius_error_bound, rss)
    assert measured <= rounded.evidence.frobenius_error_bound + 2e-5
    assert not rounded.evidence.tolerance_met
    assert int(rounded.evidence.status) == int(
        tt.TTRoundingStatus.RANK_CAP_REACHED_BEFORE_TOLERANCE
    )


def test_relative_tolerance_masks_negligible_singular_directions_at_static_capacity():
    dense = jnp.diag(jnp.asarray([1.0, 1e-8], dtype=jnp.float32))
    compressed = tt.tt_svd(
        dense,
        max_ranks=2,
        relative_tolerance=1e-6,
    )
    assert compressed.tensor.ranks == (2,)
    assert jnp.array_equal(compressed.evidence.selected_ranks, jnp.asarray([1]))
    assert compressed.evidence.tolerance_met
    assert jnp.allclose(
        compressed.tensor.to_dense(max_entries=4),
        jnp.diag(jnp.asarray([1.0, 0.0], dtype=jnp.float32)),
        atol=1e-7,
    )

    exact = tt.tt_svd(dense, max_ranks=2, relative_tolerance=0.0).tensor
    rounded = tt.round_tensor_train(
        exact,
        max_ranks=2,
        relative_tolerance=1e-6,
    )
    assert rounded.tensor.ranks == (2,)
    assert jnp.array_equal(rounded.evidence.selected_ranks, jnp.asarray([1]))
    assert rounded.evidence.tolerance_met


def test_qtt_ordering_round_trip_evaluation_and_analytic_linear_quadrature():
    grid = tt.TensorizedGrid.uniform(((0.0, 1.0), (0.0, 1.0)), (4, 4), rule="trapezoid")
    blocked = tt.QuanticsLayout.binary((4, 4), ordering="blocked")
    interleaved = tt.QuanticsLayout.binary((4, 4), ordering="interleaved")
    points = jnp.asarray([[0, 0], [1, 0], [3, 3]], dtype=jnp.int32)

    assert jnp.array_equal(blocked.undigitize(blocked.digitize(points)), points)
    assert jnp.array_equal(interleaved.undigitize(interleaved.digitize(points)), points)
    assert not jnp.array_equal(blocked.digitize(points), interleaved.digitize(points))

    x, y = jnp.meshgrid(grid.axis_nodes[0], grid.axis_nodes[1], indexing="ij")
    dense = x + y
    qtt = tt.qtt_digitize(dense, interleaved, max_ranks=4, relative_tolerance=0.0).tensor
    assert jnp.allclose(
        tt.qtt_evaluate(qtt, interleaved, points),
        dense[points[:, 0], points[:, 1]],
        atol=2e-5,
    )
    assert jnp.allclose(
        tt.qtt_quadrature(qtt, interleaved, grid, max_entries=16),
        1.0,
        atol=2e-5,
    )

    function = tt.TensorFunction(
        lambda coordinates: coordinates[:, 0] + coordinates[:, 1],
        grid,
        vectorized=True,
        name="linear-sum",
    )
    assert jnp.allclose(function.quadrature(max_evaluations=16), 1.0, atol=2e-6)


def test_tensor_indices_reject_fractional_and_out_of_range_values():
    train = tt.TensorTrain((jnp.ones((1, 4, 1), dtype=jnp.float32),))
    with pytest.raises(TypeError, match="integer dtype"):
        train.evaluate(jnp.asarray([[1.5]], dtype=jnp.float32))

    grid = tt.TensorizedGrid.uniform(((0.0, 1.0),), (4,), rule="midpoint")
    with pytest.raises(Exception, match="out-of-range"):
        grid.coordinates(jnp.asarray([[4]], dtype=jnp.int32))

    layout = tt.QuanticsLayout.binary((4,))
    with pytest.raises(Exception, match="out-of-range"):
        layout.digitize(jnp.asarray([[4]], dtype=jnp.int32))
    with pytest.raises(Exception, match="out-of-range"):
        layout.undigitize(jnp.asarray([[0, 2]], dtype=jnp.int32))


def test_tt_svd_and_rounding_execute_under_jit_with_array_evidence():
    dense = jnp.arange(24, dtype=jnp.float32).reshape((2, 3, 4))
    compressed = jax.jit(
        lambda value: tt.tt_svd(
            value,
            max_ranks=(2, 4),
            relative_tolerance=0.0,
        )
    )(dense)
    rounded = jax.jit(
        lambda value: tt.round_tensor_train(
            value,
            max_ranks=(2, 4),
            relative_tolerance=0.0,
        )
    )(compressed.tensor)
    assert jnp.allclose(rounded.tensor.to_dense(max_entries=24), dense, atol=2e-5)
    assert rounded.evidence.tolerance_met
