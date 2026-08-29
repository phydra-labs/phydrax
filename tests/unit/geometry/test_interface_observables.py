#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_phase_geometry_uses_physical_weights_and_ignores_padding():
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


def test_zero_measure_phase_has_explicitly_undefined_centroid():
    metrics = phx.geometry.phase_geometry_metrics(
        jnp.zeros((3,)),
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        jnp.ones((3,)),
    )

    assert metrics.measure == 0.0
    assert not bool(metrics.centroid_defined)
    assert jnp.isnan(metrics.centroid).all()


def test_interface_distances_recover_uniform_translation_with_padding():
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


def test_percentile_hausdorff_separates_one_spurious_point():
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


def test_interface_distances_preserve_case_axes():
    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    predicted = jnp.stack((reference, reference + jnp.asarray((0.0, 2.0))))

    metrics = phx.geometry.interface_distance_metrics(predicted, reference)

    assert metrics.symmetric_mean_distance.shape == (2,)
    np.testing.assert_allclose(
        metrics.symmetric_mean_distance,
        jnp.asarray((0.0, 2.0)),
        atol=1.0e-14,
    )


def test_interface_observables_reject_complex_geometry():
    coordinates = jnp.asarray(((0.0 + 1.0j, 0.0), (1.0, 0.0)))
    with pytest.raises(TypeError, match="coordinates must be real-valued"):
        phx.geometry.phase_geometry_metrics(
            jnp.ones((2,)),
            coordinates,
            jnp.ones((2,)),
        )
    with pytest.raises(TypeError, match="predicted_points must be real-valued"):
        phx.geometry.interface_distance_metrics(coordinates, jnp.real(coordinates))
