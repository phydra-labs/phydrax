#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.velocimetry.imaging import (
    DenseDisplacementField2D,
    image_coordinates,
    ImageGeometry2D,
    ImagePair2D,
)
from phydrax.velocimetry.piv import (
    AffinePixelMap2D,
    convert_to_physical,
    deform_image_pair,
    HomographyPixelMap2D,
    map_pixels_to_physical,
    PIVQuality2D,
    replace_invalid_vectors,
    validate_field,
)


def test_second_and_symmetric_deformation_align_known_translation():
    geometry = ImageGeometry2D((8, 8))
    first = jnp.arange(64.0).reshape((8, 8))
    second = jnp.zeros_like(first).at[:, 1:].set(first[:, :-1])
    pair = ImagePair2D(first, second, geometry)
    positions = image_coordinates(geometry)
    predictor = DenseDisplacementField2D(
        positions,
        jnp.broadcast_to(jnp.asarray([0.0, 1.0]), positions.shape),
        jnp.ones((8, 8), dtype=bool),
        geometry_id=geometry.geometry_id,
    )

    second_frame = deform_image_pair(pair, predictor, mode="second")
    symmetric = deform_image_pair(pair, predictor, mode="symmetric")

    assert jnp.allclose(
        second_frame.second[second_frame.second_mask], first[second_frame.second_mask]
    )
    assert jnp.allclose(
        symmetric.first[symmetric.first_mask & symmetric.second_mask],
        symmetric.second[symmetric.first_mask & symmetric.second_mask],
    )


def test_validation_evidence_and_replacement_do_not_mutate_raw_vectors():
    row, column = jnp.meshgrid(jnp.arange(3.0), jnp.arange(3.0), indexing="ij")
    positions = jnp.stack((row, column), axis=-1)
    vectors = jnp.broadcast_to(jnp.asarray([1.0, -1.0]), (3, 3, 2))
    vectors = vectors.at[1, 1].set(jnp.asarray([9.0, 9.0]))
    raw = DenseDisplacementField2D(
        positions,
        vectors,
        jnp.ones((3, 3), dtype=bool),
        geometry_id="geometry",
    )
    quality = PIVQuality2D(
        jnp.ones((3, 3)),
        0.5 * jnp.ones((3, 3)),
        2.0 * jnp.ones((3, 3)),
        4.0 * jnp.ones((3, 3)),
        jnp.ones((3, 3)),
    )

    validated, evidence = validate_field(
        raw,
        quality,
        maximum_displacement=(20.0, 20.0),
        minimum_correlation=0.0,
        minimum_peak_ratio=1.1,
        radius=1,
        minimum_neighbors=3,
        median_threshold=2.0,
        median_epsilon=0.1,
    )
    replaced, replacement = replace_invalid_vectors(
        validated,
        radius=1,
        iterations=1,
        minimum_neighbors=3,
    )

    assert not evidence.local_consistency_accepted[1, 1]
    assert not validated.valid[1, 1]
    assert replacement.replaced[1, 1]
    assert jnp.array_equal(replaced.displacement_rc[1, 1], jnp.asarray([1.0, -1.0]))
    assert jnp.array_equal(raw.displacement_rc[1, 1], jnp.asarray([9.0, 9.0]))


def test_affine_and_homography_use_right_handed_xy_endpoints_and_units():
    positions = jnp.asarray([[[2.0, 3.0]]])
    displacement = jnp.asarray([[[1.0, 2.0]]])
    field = DenseDisplacementField2D(
        positions,
        displacement,
        jnp.asarray([[True]]),
        geometry_id="geometry",
    )
    affine = AffinePixelMap2D(
        jnp.asarray([[2.0, 0.0, 10.0], [0.0, -3.0, 20.0]]),
        spatial_unit="mm",
    )
    physical = convert_to_physical(field, affine, delta_t=2.0, time_unit="s")

    assert jnp.allclose(physical.positions_xy[0, 0], jnp.asarray([16.0, 14.0]))
    assert jnp.allclose(physical.displacement_xy[0, 0], jnp.asarray([4.0, -3.0]))
    assert jnp.allclose(physical.velocity_xy[0, 0], jnp.asarray([2.0, -1.5]))
    assert physical.spatial_unit == "mm"
    assert physical.time_unit == "s"

    homography = HomographyPixelMap2D(
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 1.0]]),
        spatial_unit="m",
    )
    start, start_valid = map_pixels_to_physical(homography, positions)
    end, end_valid = map_pixels_to_physical(homography, positions + displacement)
    nonlinear = convert_to_physical(field, homography, delta_t=1.0, time_unit="s")

    assert start_valid[0, 0] & end_valid[0, 0]
    assert jnp.allclose(nonlinear.displacement_xy, end - start)
