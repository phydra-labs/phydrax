#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._numerics import (
    LEAST_SQUARES_RANK_DEFICIENT,
    LEAST_SQUARES_SUCCESS,
    solve_weighted_least_squares,
)


def test_weighted_least_squares_recovers_multioutput_and_raw_coordinates():
    x = jnp.linspace(-2.0, 3.0, 17)
    design = jnp.stack((jnp.ones_like(x), x, x**2), axis=-1)
    expected = jnp.asarray([[1.0, -2.0], [0.5, 3.0], [-0.25, 0.75]])
    target = design @ expected
    weights = jnp.linspace(0.2, 2.0, x.size)

    result = solve_weighted_least_squares(
        design,
        target,
        weights=weights,
        center=True,
        scale=True,
        min_samples=3,
    )

    assert bool(result.valid)
    assert int(result.status) == LEAST_SQUARES_SUCCESS
    np.testing.assert_allclose(
        design @ result.raw_coefficients + result.intercept,
        target,
        atol=2e-12,
    )
    np.testing.assert_allclose(result.prediction, target, atol=2e-12)
    assert float(result.normal_equation_error) < 1e-12


def test_masked_nonfinite_padding_is_inert():
    design = jnp.asarray([[1.0, 0.0], [1.0, 1.0], [jnp.nan, jnp.inf]])
    target = jnp.asarray([[2.0], [5.0], [jnp.nan]])
    mask = jnp.asarray([True, True, False])

    result = solve_weighted_least_squares(
        design,
        target,
        mask=mask,
        min_samples=2,
    )

    assert bool(result.valid)
    np.testing.assert_allclose(result.coefficients[:, 0], jnp.asarray([2.0, 3.0]))
    assert int(result.sample_count) == 2
    assert not bool(result.valid_rows[-1])


def test_complex_least_squares_uses_hermitian_geometry():
    design = jnp.asarray([[1.0 + 1.0j, 2.0], [2.0 - 1.0j, -1.0j], [0.5, 3.0 + 2.0j]])
    expected = jnp.asarray([[2.0 - 0.5j], [-1.0 + 3.0j]])
    target = design @ expected

    result = solve_weighted_least_squares(design, target, min_samples=2)

    assert bool(result.valid)
    np.testing.assert_allclose(result.coefficients, expected, atol=2e-12)


def test_unregularized_rank_deficiency_is_reported_without_repair():
    design = jnp.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    target = jnp.asarray([1.0, 2.0, 3.0])

    result = solve_weighted_least_squares(design, target, min_samples=2)

    assert not bool(result.valid)
    assert int(result.status) == LEAST_SQUARES_RANK_DEFICIENT
    assert int(result.rank) == 1
