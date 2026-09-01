#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.velocimetry.piv import (
    correlate_windows,
    CorrelationBatch,
    find_top_peaks,
    WindowBatch2D,
)


def test_chunked_extended_correlation_recovers_row_column_shift_and_mask():
    first_values = jr.normal(jr.key(4), (3, 8, 8))
    margin = (3, 3)
    expected = jnp.asarray([1.0, -2.0])
    second_values = jnp.zeros((3, 14, 14))
    second_values = second_values.at[:, 4:12, 1:9].set(first_values)
    first_mask = jnp.ones_like(first_values, dtype=bool).at[2].set(False)
    second_mask = jnp.ones_like(second_values, dtype=bool).at[2].set(False)
    first = WindowBatch2D(first_values, first_mask, jnp.zeros((3, 2)), (1, 3))
    second = WindowBatch2D(second_values, second_mask, jnp.zeros((3, 2)), (1, 3))

    correlation = correlate_windows(
        first,
        second,
        mode="extended",
        search_margin=margin,
        chunk_size=2,
        minimum_valid_fraction=0.8,
    )
    peaks = find_top_peaks(correlation, top_k=2, method="parabolic")

    assert jnp.array_equal(
        jnp.rint(peaks.offsets_rc[:2, 0]), jnp.broadcast_to(expected, (2, 2))
    )
    assert jnp.array_equal(peaks.valid[:, 0], jnp.asarray([True, True, False]))
    assert jnp.all(correlation.overlap[:2, 4, 1] >= 64.0 - 1e-4)


def test_linear_correlation_uses_overlap_mask_instead_of_wrapping():
    first_values = jr.normal(jr.key(5), (1, 8, 8))
    second_values = jnp.zeros_like(first_values)
    second_values = second_values.at[:, 1:, 2:].set(first_values[:, :-1, :-2])
    first_mask = jnp.ones_like(first_values, dtype=bool)
    second_mask = jnp.zeros_like(second_values, dtype=bool).at[:, 1:, 2:].set(True)
    first = WindowBatch2D(first_values, first_mask, jnp.zeros((1, 2)), (1, 1))
    second = WindowBatch2D(second_values, second_mask, jnp.zeros((1, 2)), (1, 1))

    correlation = correlate_windows(
        first,
        second,
        mode="linear",
        search_margin=(3, 3),
        chunk_size=1,
        minimum_valid_fraction=0.5,
    )
    peak = find_top_peaks(correlation, top_k=2, method="parabolic")

    assert jnp.array_equal(jnp.rint(peak.offsets_rc[0, 0]), jnp.asarray([1.0, 2.0]))
    assert correlation.overlap[0, 4, 5] == 42


def test_circular_correlation_preserves_positive_row_down_shift():
    first_values = jr.normal(jr.key(7), (1, 8, 8))
    second_values = jnp.roll(first_values, (2, -1), axis=(-2, -1))
    mask = jnp.ones_like(first_values, dtype=bool)
    first = WindowBatch2D(first_values, mask, jnp.zeros((1, 2)), (1, 1))
    second = WindowBatch2D(second_values, mask, jnp.zeros((1, 2)), (1, 1))

    correlation = correlate_windows(
        first,
        second,
        mode="circular",
        search_margin=(3, 3),
        chunk_size=1,
        minimum_valid_fraction=1.0,
    )
    peak = find_top_peaks(correlation, top_k=2, method="parabolic")

    assert jnp.array_equal(jnp.rint(peak.offsets_rc[0, 0]), jnp.asarray([2.0, -1.0]))


def test_top_k_ties_are_row_major_and_gaussian_fit_is_subpixel():
    rows, columns = jnp.meshgrid(jnp.arange(-2, 3), jnp.arange(-2, 3), indexing="ij")
    lags = jnp.stack((rows, columns), axis=-1)
    tied = CorrelationBatch(
        jnp.ones((1, 5, 5)),
        jnp.ones((1, 5, 5)),
        jnp.ones((1, 5, 5), dtype=bool),
        lags,
        "linear",
    )
    tied_peaks = find_top_peaks(tied, top_k=3, method="parabolic")

    assert jnp.array_equal(
        tied_peaks.offsets_rc[0],
        jnp.asarray([[-2.0, -2.0], [-2.0, -1.0], [-2.0, 0.0]]),
    )

    surface = jnp.exp(-((rows - 0.3) ** 2) / 2.0 - ((columns + 0.2) ** 2) / 3.0)
    smooth = CorrelationBatch(
        surface[None],
        jnp.ones((1, 5, 5)),
        jnp.ones((1, 5, 5), dtype=bool),
        lags,
        "linear",
    )
    fitted = find_top_peaks(smooth, top_k=2, method="gaussian")

    assert jnp.allclose(fitted.offsets_rc[0, 0], jnp.asarray([0.3, -0.2]), atol=1e-5)
    assert jnp.all(jnp.isfinite(fitted.covariance_rc[0, 0]))
