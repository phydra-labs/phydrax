import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_support_preserves_shared_coordinates_and_disconnected_restarts():
    shared = phx.series.SeriesSupport(
        jnp.asarray([0.0, 1.0, 2.0]),
        node_valid=jnp.asarray([[True, True, True], [True, True, False]]),
        series_shape=(2,),
        series_axes=("case",),
        coordinate_name="time",
        coordinate_id="shared-time",
    )

    assert shared.coordinates.shape == (3,)
    assert shared.broadcast_coordinates().shape == (2, 3)
    np.testing.assert_array_equal(
        shared.edge_valid,
        np.asarray([[True, True], [True, False]]),
    )

    restarted = phx.series.SeriesSupport(
        jnp.asarray([0.0, 1.0, 2.0, 0.0, 1.0]),
        edge_valid=jnp.asarray([True, True, False, True]),
        coordinate_name="time",
        coordinate_id="restarted-time",
    )
    assert not bool(restarted.edge_valid[2])

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="increase strictly"):
        invalid = phx.series.SeriesSupport(
            jnp.asarray([0.0, 1.0, 0.0]),
            coordinate_name="time",
            coordinate_id="invalid-time",
        )
        jax.block_until_ready(invalid.edge_valid)


def test_sampled_series_supports_pytrees_component_masks_and_edge_values():
    support = phx.series.SeriesSupport(
        jnp.asarray([0.0, 1.0, 2.0]),
        coordinate_name="parameter",
        coordinate_id="parameter-grid",
    )
    values = {
        "continuous": jnp.asarray([[1.0, 2.0], [3.0, jnp.nan], [5.0, 6.0]]),
        "mode": jnp.asarray([0, 1, 1]),
    }
    value_valid = {
        "continuous": jnp.asarray([[True, True], [True, False], [True, True]]),
        "mode": jnp.ones((3,), dtype=bool),
    }
    series = phx.series.SampledSeries(
        support,
        values,
        value_valid=value_valid,
        series_id="mixed-node-series",
    )
    np.testing.assert_array_equal(series.sample_valid, [True, False, True])

    edge = phx.series.SampledSeries(
        support,
        jnp.asarray([[2.0], [4.0]]),
        alignment="edge",
        series_id="edge-series",
    )
    assert edge.sample_shape == (2,)
    np.testing.assert_allclose(edge.values_for(0), [[2.0], [4.0]])


def test_pair_view_is_lazy_and_never_crosses_disconnected_edges():
    support = phx.series.SeriesSupport(
        jnp.asarray([0.0, 1.0, 2.0, 0.0, 1.0]),
        edge_valid=jnp.asarray([True, True, False, True]),
        coordinate_name="time",
        coordinate_id="two-episodes",
    )
    series = phx.series.SampledSeries(
        support,
        jnp.arange(5.0)[:, None],
        series_id="states",
    )
    pairs = phx.series.SeriesPairView.from_lag(series, 2)

    np.testing.assert_array_equal(pairs.valid, [True, False, False])
    np.testing.assert_allclose(pairs.source_values[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(pairs.target_values[:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(pairs.coordinate_delta, [2.0, -1.0, -1.0])

    complete_support = phx.series.SeriesSupport(
        jnp.arange(3.0),
        coordinate_name="time",
        coordinate_id="partially-observed-time",
    )
    incomplete = phx.series.SampledSeries(
        complete_support,
        jnp.asarray([0.0, jnp.nan, 2.0]),
        value_valid=jnp.asarray([True, False, True]),
        series_id="partially-observed-values",
    )
    assert not bool(phx.series.SeriesPairView.from_lag(incomplete, 2).valid[0])


def test_reconstruction_handles_irregular_cases_bounds_and_breakpoints():
    times = jnp.asarray([[0.0, 1.0, 3.0, 99.0], [-1.0, 0.5, 2.0, 123.0]])
    valid = jnp.asarray([[True, True, True, False], [True, True, True, False]])
    values = jnp.asarray(
        [
            [[0.0], [10.0], [30.0], [jnp.nan]],
            [[0.0], [15.0], [30.0], [jnp.nan]],
        ]
    )
    support = phx.series.SeriesSupport(
        times,
        node_valid=valid,
        series_axes=("case",),
        coordinate_name="time",
        coordinate_id="irregular-time",
    )
    series = phx.series.SampledSeries(support, values, series_id="input")
    reconstruction = phx.series.SampledSeriesReconstruction(
        series,
        interpolation="linear",
        bounds="fill",
    )

    evaluation = eqx.filter_jit(reconstruction.evaluate)(
        jnp.asarray([2.0, 1.25]), jnp.asarray([0, 1])
    )
    np.testing.assert_array_equal(evaluation.support, [True, True])
    np.testing.assert_allclose(evaluation.values[:, 0], [20.0, 22.5])
    outside = reconstruction.evaluate(jnp.asarray([-0.1, 3.1]), 0)
    np.testing.assert_array_equal(outside.support, [False, False])
    np.testing.assert_allclose(outside.values[:, 0], [0.0, 30.0])

    points, mask = reconstruction.breakpoints(0.5, 2.5, 0)
    np.testing.assert_allclose(points, [0.0, 1.0, 3.0, 99.0])
    np.testing.assert_array_equal(mask, [False, True, False, False])

    integer_support = phx.series.SeriesSupport(
        jnp.arange(3),
        coordinate_name="iteration",
        coordinate_kind="discrete",
        coordinate_id="iterations",
    )
    integer_series = phx.series.SampledSeries(
        integer_support,
        jnp.arange(3.0),
        series_id="iterates",
    )
    integer_reconstruction = phx.series.SampledSeriesReconstruction(
        integer_series,
        interpolation="previous",
    )
    _, integer_mask = integer_reconstruction.breakpoints(0.5, 1.5)
    np.testing.assert_array_equal(integer_mask, [False, True, False])


def test_reconstruction_preserves_hold_ties_derivatives_and_gradients():
    support = phx.series.SeriesSupport(
        jnp.asarray([0.0, 1.0, 2.0, 3.0]),
        coordinate_name="time",
        coordinate_id="regular-time",
    )
    node_series = phx.series.SampledSeries(
        support,
        jnp.asarray([0.0, 1.0, 4.0, 9.0]),
        series_id="quadratic",
    )
    cubic = phx.series.SampledSeriesReconstruction(
        node_series,
        interpolation="cubic_hermite",
    )
    np.testing.assert_allclose(cubic.evaluate(1.5).values, 2.25)
    np.testing.assert_allclose(cubic.evaluate(1.5, derivative_order=1).values, 3.0)
    np.testing.assert_allclose(cubic.evaluate(1.5, derivative_order=2).values, 2.0)

    nearest = phx.series.SampledSeriesReconstruction(
        node_series,
        interpolation="nearest",
        nearest_tie_policy="round_even",
    )
    np.testing.assert_allclose(
        nearest.evaluate(jnp.asarray([0.5, 1.5])).values, [0.0, 4.0]
    )

    previous = phx.series.SampledSeriesReconstruction(
        node_series,
        interpolation="previous",
    )
    np.testing.assert_allclose(previous.evaluate(jnp.nextafter(1.0, 0.0)).values, 0.0)
    np.testing.assert_allclose(previous.evaluate(1.0).values, 1.0)

    edge_series = phx.series.SampledSeries(
        support,
        jnp.asarray([3.0, 5.0, 7.0]),
        alignment="edge",
        series_id="held",
    )
    left = phx.series.SampledSeriesReconstruction(
        edge_series,
        interpolation="interval_hold",
        node_side="left",
    )
    right = phx.series.SampledSeriesReconstruction(
        edge_series,
        interpolation="interval_hold",
        node_side="right",
    )
    np.testing.assert_allclose(left.evaluate(1.0).values, 3.0)
    np.testing.assert_allclose(right.evaluate(1.0).values, 5.0)

    def at_half(values):
        candidate = phx.series.SampledSeries(
            support,
            values,
            series_id="differentiable-linear",
        )
        linear = phx.series.SampledSeriesReconstruction(
            candidate,
            interpolation="linear",
        )
        return linear.evaluate(0.5).values

    np.testing.assert_allclose(
        jax.grad(at_half)(jnp.asarray([0.0, 2.0, 4.0, 6.0])),
        [0.5, 0.5, 0.0, 0.0],
    )


def test_reconstruction_rejects_disconnected_support():
    support = phx.series.SeriesSupport(
        jnp.asarray([0.0, 1.0, 0.0, 1.0]),
        edge_valid=jnp.asarray([True, False, True]),
        coordinate_name="time",
        coordinate_id="disconnected",
    )
    series = phx.series.SampledSeries(
        support,
        jnp.arange(4.0),
        series_id="disconnected-values",
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="connected valid prefix"
    ):
        reconstruction = phx.series.SampledSeriesReconstruction(
            series,
            interpolation="linear",
        )
        jax.block_until_ready(reconstruction.series.support.coordinates)
