#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.algebraic import (
    PolynomialScaling,
    SparsePolynomialSupport,
    SparsePolynomialSystem,
)


def test_coo_factory_canonicalizes_support_and_coefficients_together():
    first = SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("f", "g"),
        (1, 0, 1, 0),
        ((0, 0), (0, 1), (1, 1), (2, 0)),
        jnp.asarray((-1.0, 2.0j, 3.0, 1.0), dtype=jnp.complex64),
    )
    second = SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("f", "g"),
        (0, 1, 0, 1),
        ((2, 0), (1, 1), (0, 1), (0, 0)),
        jnp.asarray((1.0, 3.0, 2.0j, -1.0), dtype=jnp.complex64),
    )

    assert first.support.support_id == second.support.support_id
    assert first.system_id == second.system_id
    np.testing.assert_array_equal(first.support.equation_indices, (0, 0, 1, 1))
    np.testing.assert_array_equal(
        first.support.exponents,
        ((0, 1), (2, 0), (0, 0), (1, 1)),
    )
    points = jnp.asarray(((0.0 + 0.0j, 0.0 + 0.0j), (1.0 + 2.0j, -0.5j)))
    np.testing.assert_allclose(first.evaluate(points), second.evaluate(points))


def test_support_rejects_duplicate_coo_terms():
    with pytest.raises(ValueError, match="duplicate"):
        SparsePolynomialSupport(
            ("x",),
            ("f",),
            (0, 0),
            ((2,), (2,)),
        )


def test_batched_complex_evaluation_and_zero_safe_analytic_jacobian():
    system = SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("f", "g"),
        (0, 0, 1, 1),
        ((2, 0), (0, 1), (1, 1), (0, 0)),
        jnp.asarray((1.0, 2.0j, 3.0, -1.0), dtype=jnp.complex64),
    )
    points = jnp.asarray(
        ((0.0 + 0.0j, 0.0 + 0.0j), (1.0 + 2.0j, -0.5j)),
        dtype=jnp.complex64,
    )

    expected_values = jnp.stack(
        (
            points[:, 0] ** 2 + 2.0j * points[:, 1],
            3.0 * points[:, 0] * points[:, 1] - 1.0,
        ),
        axis=-1,
    )
    expected_jacobian = jnp.stack(
        (
            jnp.stack((2.0 * points[:, 0], jnp.full((2,), 2.0j)), axis=-1),
            jnp.stack((3.0 * points[:, 1], 3.0 * points[:, 0]), axis=-1),
        ),
        axis=-2,
    )
    np.testing.assert_allclose(
        system.evaluate(points), expected_values, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        system.jacobian(points), expected_jacobian, rtol=1e-6, atol=1e-6
    )
    assert bool(jnp.all(jnp.isfinite(system.jacobian(points))))


def test_scaling_round_trip_preserves_fixed_zero_coefficient_slots():
    system = SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("f", "g"),
        (0, 0, 1, 1),
        ((2, 0), (0, 1), (1, 1), (0, 0)),
        jnp.asarray((1.0, 0.0, 3.0, -1.0)),
    )
    scaling = PolynomialScaling(
        jnp.asarray((2.0, 3.0)),
        jnp.asarray((5.0, 7.0)),
    )
    physical = jnp.asarray(((1.25, -2.0), (-0.5, 0.75)))
    scaled_points = scaling.to_scaled_points(physical)
    scaled = scaling.scale_system(system)

    np.testing.assert_allclose(
        scaled.evaluate(scaled_points),
        scaling.to_scaled_residuals(system.evaluate(physical)),
    )
    restored = scaling.unscale_system(scaled)
    np.testing.assert_allclose(restored.coefficients, system.coefficients)
    assert restored.support.support_id == system.support.support_id
    assert restored.support.term_count == system.support.term_count
    assert int(jnp.count_nonzero(restored.coefficients == 0.0)) == 1
    np.testing.assert_allclose(
        scaling.to_physical_points(scaled_points),
        physical,
    )
