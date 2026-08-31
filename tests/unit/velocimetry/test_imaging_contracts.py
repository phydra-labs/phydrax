#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.velocimetry.imaging import (
    backward_warp,
    bilinear_sample,
    DenseDisplacementField2D,
    image_coordinates,
    ImageGeometry2D,
    ImagePair2D,
)


def test_image_contracts_preserve_row_column_components_and_masks():
    geometry = ImageGeometry2D(
        (3, 4), pixel_origin_rc=(2.0, 5.0), pixel_spacing_rc=(0.5, 0.25)
    )
    first = jnp.arange(12.0).reshape((3, 4))
    pair = ImagePair2D(first, first + 1.0, geometry, first_mask=first != 5.0, delta_t=0.2)
    positions = image_coordinates(geometry)
    field = DenseDisplacementField2D(
        positions,
        jnp.broadcast_to(jnp.asarray([1.0, -2.0]), positions.shape),
        jnp.ones((3, 4), dtype=bool),
        geometry_id=geometry.geometry_id,
    )

    assert geometry.coordinate_convention == "row-down-column-right"
    assert pair.first_mask[1, 1] == jnp.asarray(False)
    assert jnp.array_equal(field.displacement_rc[0, 0], jnp.asarray([1.0, -2.0]))


def test_native_bilinear_sample_is_strict_and_nonperiodic_at_borders():
    image = jnp.arange(9.0).reshape((3, 3))
    coordinates = jnp.asarray([[0.5, 0.5], [2.0, 2.0], [-0.1, 1.0], [1.0, 3.0]])
    sampled = jax.jit(bilinear_sample)(image, coordinates, fill_value=-7.0)

    assert jnp.allclose(sampled.values, jnp.asarray([2.0, 8.0, -7.0, -7.0]))
    assert jnp.array_equal(sampled.valid, jnp.asarray([True, True, False, False]))


def test_backward_warp_has_declared_row_down_column_right_sign():
    image = jnp.arange(25.0).reshape((5, 5))
    displacement = jnp.zeros((5, 5, 2)).at[..., 1].set(1.0)
    warped = backward_warp(image, displacement, fill_value=-1.0)

    assert jnp.array_equal(warped.values[:, 1:], image[:, :-1])
    assert jnp.array_equal(warped.valid[:, 0], jnp.zeros((5,), dtype=bool))
    assert jnp.array_equal(warped.values[:, 0], -jnp.ones((5,)))
