#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_interface_observables_scenario_1() -> None:
    coordinates = jnp.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (jnp.nan, jnp.nan),
        )
    )
    fraction = jnp.asarray((1.0, 1.0, 0.0, 0.0, jnp.nan))
    weights = jnp.asarray((0.25, 0.75, 0.5, 0.5, jnp.nan))
    mask = jnp.asarray((True, True, True, True, False))

    metrics = phx.geometry.phase_geometry_metrics(
        fraction,
        coordinates,
        weights,
        mask=mask,
    )

    np.testing.assert_allclose(metrics.measure, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(metrics.centroid, jnp.asarray((0.75, 0.0)), atol=1.0e-14)
    assert bool(metrics.centroid_defined)
    metrics = phx.geometry.phase_geometry_metrics(
        jnp.zeros((3,)),
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        jnp.ones((3,)),
    )

    assert metrics.measure == 0.0
    assert not bool(metrics.centroid_defined)
    assert jnp.isnan(metrics.centroid).all()
    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (jnp.nan, jnp.nan)))
    predicted = jnp.asarray(((0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (jnp.nan, jnp.nan)))
    mask = jnp.asarray((True, True, True, False))

    metrics = phx.geometry.interface_distance_metrics(
        predicted,
        reference,
        predicted_mask=mask,
        reference_mask=mask,
        chunk_size=2,
    )

    np.testing.assert_allclose(metrics.symmetric_mean_distance, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(metrics.hausdorff_distance, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(
        metrics.percentile_hausdorff_distance,
        1.0,
        atol=1.0e-14,
    )
    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    predicted = jnp.stack((reference, reference + jnp.asarray((0.0, 2.0))))

    metrics = phx.geometry.interface_distance_metrics(predicted, reference)

    assert metrics.symmetric_mean_distance.shape == (2,)
    np.testing.assert_allclose(
        metrics.symmetric_mean_distance,
        jnp.asarray((0.0, 2.0)),
        atol=1.0e-14,
    )
    coordinates = jnp.asarray(((0.0 + 1.0j, 0.0), (1.0, 0.0)))
    with pytest.raises(TypeError, match="coordinates must be real-valued"):
        phx.geometry.phase_geometry_metrics(
            jnp.ones((2,)),
            coordinates,
            jnp.ones((2,)),
        )
    with pytest.raises(TypeError, match="predicted_points must be real-valued"):
        phx.geometry.interface_distance_metrics(coordinates, jnp.real(coordinates))


def test_percentile_hausdorff_separates_one_spurious_point() -> None:
    reference = jnp.zeros((20, 2))
    predicted = jnp.concatenate((reference, jnp.asarray(((10.0, 0.0),))), axis=0)

    metrics = phx.geometry.interface_distance_metrics(
        predicted,
        reference,
        percentile=0.95,
        chunk_size=7,
    )

    assert metrics.hausdorff_distance == 10.0
    assert metrics.percentile_hausdorff_distance == 0.0
    assert 0.0 < metrics.symmetric_mean_distance < 1.0


@pytest.mark.parametrize(
    ("chunk_size", "target_chunk_size"),
    [(2, 3), (4, 2), (64, 64), (1, 1), (64, None)],
    ids=["both-tails", "target-tail", "oversized-tiles", "unit-tiles", "budget-default"],
)
def test_interface_distances_match_asymmetric_masked_broadcast_reference(
    chunk_size: int,
    target_chunk_size: int | None,
) -> None:
    predicted = np.asarray(
        [
            [[[9.0, 0.0], [0.0, 1.0], [4.0, 2.0], [12.0, -1.0], [np.nan, np.nan]]],
            [[[9.0, 1.0], [0.0, 3.0], [4.0, 0.0], [12.0, 2.0], [np.nan, np.nan]]],
        ],
        dtype=np.float64,
    )
    reference = np.asarray(
        [
            [
                [
                    [0.0, 0.0],
                    [2.0, 1.0],
                    [5.0, 2.0],
                    [16.0, 0.0],
                    [9.0, 0.0],
                    [7.0, 1.0],
                    [np.nan, np.nan],
                ],
                [
                    [0.0, 2.0],
                    [2.0, 3.0],
                    [5.0, 0.0],
                    [16.0, 2.0],
                    [9.0, 1.0],
                    [7.0, -1.0],
                    [np.nan, np.nan],
                ],
                [
                    [0.0, -1.0],
                    [2.0, 2.0],
                    [5.0, 1.0],
                    [16.0, -2.0],
                    [9.0, 0.0],
                    [7.0, 3.0],
                    [np.nan, np.nan],
                ],
            ]
        ],
        dtype=np.float64,
    )
    predicted_mask = np.asarray(
        [[[True, True, True, True, False]], [[True, False, True, True, False]]],
        dtype=np.bool_,
    )
    reference_mask = np.asarray(
        [
            [
                [True, False, True, True, True, True, False],
                [False, True, True, True, True, False, False],
                [True, True, False, True, True, True, False],
            ]
        ],
        dtype=np.bool_,
    )
    percentile = 0.63
    metrics = phx.geometry.interface_distance_metrics(
        predicted,
        reference,
        predicted_mask=predicted_mask,
        reference_mask=reference_mask,
        percentile=percentile,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )
    expected_mean = np.empty((2, 3), dtype=np.float64)
    expected_max = np.empty((2, 3), dtype=np.float64)
    expected_quantile = np.empty((2, 3), dtype=np.float64)
    # Independent Euclidean differences, not the kernel's quadratic identity.
    for row, column in np.ndindex(2, 3):
        source = predicted[row, 0, predicted_mask[row, 0]]
        target = reference[0, column, reference_mask[0, column]]
        distances = np.sqrt(np.sum((source[:, None] - target[None, :]) ** 2, axis=-1))
        forward = np.min(distances, axis=1)
        backward = np.min(distances, axis=0)
        expected_mean[row, column] = 0.5 * (np.mean(forward) + np.mean(backward))
        expected_max[row, column] = max(np.max(forward), np.max(backward))
        expected_quantile[row, column] = max(
            np.quantile(forward, percentile), np.quantile(backward, percentile)
        )
    assert metrics.symmetric_mean_distance.shape == (2, 3)
    assert metrics.hausdorff_distance.shape == (2, 3)
    assert metrics.percentile_hausdorff_distance.shape == (2, 3)
    np.testing.assert_allclose(metrics.symmetric_mean_distance, expected_mean, atol=1e-12)
    np.testing.assert_allclose(metrics.hausdorff_distance, expected_max, atol=1e-12)
    np.testing.assert_allclose(
        metrics.percentile_hausdorff_distance, expected_quantile, atol=1e-12
    )


@pytest.mark.parametrize("target_chunk_size", [0, -3])
def test_interface_distance_rejects_nonpositive_target_chunk(
    target_chunk_size: int,
) -> None:
    points = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="target_chunk_size must be positive"):
        phx.geometry.interface_distance_metrics(
            points, points, target_chunk_size=target_chunk_size
        )


def test_compiled_interface_distances_preserve_single_point_outputs() -> None:
    compiled = jax.jit(
        phx.geometry.interface_distance_metrics,
        static_argnames=("chunk_size", "target_chunk_size"),
    )
    metrics = compiled(
        np.asarray([[1.0, 2.0]], dtype=np.float64),
        np.asarray([[4.0, 6.0]], dtype=np.float64),
        chunk_size=3,
        target_chunk_size=None,
    )
    assert metrics.symmetric_mean_distance.shape == ()
    np.testing.assert_allclose(metrics.symmetric_mean_distance, 5.0, atol=1e-12)
    np.testing.assert_allclose(metrics.hausdorff_distance, 5.0, atol=1e-12)
    np.testing.assert_allclose(metrics.percentile_hausdorff_distance, 5.0, atol=1e-12)


def test_interface_distance_budget_default_preserves_empty_case_axis() -> None:
    metrics = phx.geometry.interface_distance_metrics(
        np.zeros((0, 3, 2), dtype=np.float64),
        np.zeros((1, 5, 2), dtype=np.float64),
    )
    assert metrics.symmetric_mean_distance.shape == (0,)
    assert metrics.hausdorff_distance.shape == (0,)
    assert metrics.percentile_hausdorff_distance.shape == (0,)
