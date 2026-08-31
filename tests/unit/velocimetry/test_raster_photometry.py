#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.velocimetry.imaging._photometry import (
    apply_photometry,
    PhotometricResponse,
)
from phydrax.velocimetry.imaging._raster import (
    GaussianRasterizer,
    RASTER_CLIPPED,
    RASTER_SUPPORT_OVERFLOW,
)
from phydrax.velocimetry.imaging._types import ImageGeometry2D


def test_noiseless_gaussian_raster_is_deterministic_and_flux_audited():
    geometry = ImageGeometry2D((21, 25))
    rasterizer = GaussianRasterizer(6, cutoff=3.0)
    row_column = jnp.asarray([[10.25, 12.5], [4.0, 6.0]])
    amplitude = jnp.asarray([7.0, 3.0])
    active = jnp.asarray([True, False])

    first = rasterizer.render(geometry, row_column, amplitude, 1.0, active)
    second = rasterizer.render(geometry, row_column, amplitude, 1.0, active)

    assert jnp.array_equal(first.image, second.image)
    assert jnp.allclose(jnp.sum(first.image), amplitude[0])
    assert jnp.allclose(first.evidence.deposited_flux, jnp.asarray([7.0, 0.0]))
    assert first.evidence.supported_count == 1
    assert first.successful


def test_gaussian_raster_exposes_border_and_support_overflow():
    geometry = ImageGeometry2D((12, 12))
    rasterizer = GaussianRasterizer(3, cutoff=3.0)
    result = rasterizer.render(
        geometry,
        jnp.asarray([[0.2, 0.2], [6.0, 6.0]]),
        jnp.ones((2,)),
        jnp.asarray([0.8, 2.0]),
        jnp.ones((2,), dtype=bool),
    )

    assert result.evidence.truncated.tolist() == [True, True]
    assert result.evidence.overflow.tolist() == [False, True]
    assert result.evidence.status.tolist() == [RASTER_CLIPPED, RASTER_SUPPORT_OVERFLOW]
    assert not result.successful


def test_fixed_topology_gaussian_raster_has_finite_coordinate_derivative():
    geometry = ImageGeometry2D((17, 17))
    rasterizer = GaussianRasterizer(5, cutoff=3.0)
    columns = jnp.arange(17, dtype=float)[None, :]

    def image_column_moment(column):
        result = rasterizer.render(
            geometry,
            jnp.asarray([[8.25, column]]),
            jnp.asarray([1.0]),
            jnp.asarray([1.0]),
            jnp.asarray([True]),
        )
        return jnp.sum(result.image * columns)

    derivative = jax.grad(image_column_moment)(jnp.asarray(8.25))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0


def test_photometry_separates_noiseless_response_noise_and_sensor_mask():
    ideal = jnp.asarray([[1.0, 3.0], [5.0, 7.0]])
    sensor_mask = jnp.asarray([[True, True], [False, True]])
    response = PhotometricResponse(
        gain=2.0,
        black_level=1.0,
        saturation_level=10.0,
    )
    result = apply_photometry(response, ideal, valid_mask=sensor_mask)

    assert jnp.array_equal(result.signal, jnp.asarray([[3.0, 7.0], [0.0, 10.0]]))
    assert jnp.all(result.evidence.noise == 0.0)
    assert result.evidence.saturated_count == 1
    assert result.successful

    stochastic = PhotometricResponse(read_noise_std=0.1)
    with pytest.raises(ValueError, match="PRNG key"):
        apply_photometry(stochastic, ideal)
    noisy = apply_photometry(stochastic, ideal, key=jr.PRNGKey(12))
    repeated = apply_photometry(stochastic, ideal, key=jr.PRNGKey(12))
    assert jnp.array_equal(noisy.signal, repeated.signal)
    assert jnp.any(noisy.evidence.noise != 0.0)

    shot = apply_photometry(
        PhotometricResponse(shot_noise=True),
        ideal,
        key=jr.PRNGKey(21),
    )
    assert shot.stochastic
    assert jnp.all(shot.signal >= 0.0)
