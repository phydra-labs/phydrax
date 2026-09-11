#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import spectral as spectral_api


def _explicit_synthesis(coefficients, displacements, kind):
    modal = jnp.asarray(coefficients)
    points = jnp.asarray(displacements)
    limit = modal.shape[0]
    center = limit - 1
    point_shape = points.shape[:-1]
    payload_shape = modal.shape[2:]
    output = jnp.zeros(
        point_shape + payload_shape,
        dtype=jnp.result_type(modal.dtype, points.dtype, 1j),
    )
    function = (
        phx.special.solid_harmonic_regular
        if kind == "regular"
        else phx.special.solid_harmonic_irregular
    )
    for degree in range(limit):
        for order in range(-degree, degree + 1):
            basis = function(degree, order, points)
            output = (
                output
                + basis.reshape(point_shape + (1,) * len(payload_shape))
                * modal[degree, center + order]
            )
    return output


def _complex_payload_coefficients(limit):
    width = 2 * limit - 1
    real = jnp.arange(limit * width * 6, dtype=jnp.float64).reshape((limit, width, 2, 3))
    coefficients = (0.013 * real - 0.4) + 1j * (0.2 - 0.007 * real)
    degrees = jnp.arange(limit)[:, None]
    orders = jnp.arange(-(limit - 1), limit)[None, :]
    valid = jnp.abs(orders) <= degrees
    coefficients = jnp.where(valid[..., None, None], coefficients, 0.0j)
    return coefficients.at[0, 0, 0, 0].set(jnp.nan + 1j * jnp.inf)


@pytest.mark.parametrize("kind", ["regular", "irregular"])
def test_complex_payload_synthesis_matches_explicit_mode_sum(kind):
    limit = 5
    coefficients = _complex_payload_coefficients(limit)
    points = jnp.asarray(
        [
            [[0.4, -0.2, 0.8], [-0.7, 0.5, 1.1]],
            [[1.2, 0.3, -0.6], [0.25, 0.9, 0.45]],
        ],
        dtype=jnp.float64,
    )
    prepared = spectral_api.SolidHarmonicPlan(limit, kind=kind, reality=False).prepare()
    actual = prepared.evaluate(coefficients, points)
    expected = _explicit_synthesis(coefficients, points, kind)

    assert actual.shape == points.shape[:-1] + coefficients.shape[2:]
    assert actual.dtype == jnp.complex128
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


def test_real_synthesis_uses_independent_half_and_signed_conjugacy():
    limit = 5
    center = limit - 1
    coefficients = jnp.zeros((limit, 2 * limit - 1, 2), dtype=jnp.complex128)
    coefficients = coefficients.at[0, center].set(jnp.asarray([0.7, -0.2]))
    coefficients = coefficients.at[2, center].set(jnp.asarray([-0.4, 0.9]))
    coefficients = coefficients.at[3, center + 1].set(
        jnp.asarray([0.3 - 0.5j, -0.1 + 0.8j])
    )
    coefficients = coefficients.at[4, center + 3].set(
        jnp.asarray([-0.2 + 0.6j, 0.45 - 0.25j])
    )

    canonical = coefficients
    for degree in range(limit):
        for order in range(1, degree + 1):
            canonical = canonical.at[degree, center - order].set(
                (-1) ** order * jnp.conj(canonical[degree, center + order])
            )
    contaminated = coefficients.at[3, center - 1].set(
        jnp.asarray([jnp.nan + 1j * jnp.inf, 17.0 - 4.0j])
    )
    points = jnp.asarray([[0.4, -0.3, 0.9], [-0.8, 0.6, 0.2], [1.1, 0.2, -0.5]])
    prepared = spectral_api.SolidHarmonicPlan(limit, reality=True).prepare()
    actual = prepared.evaluate(contaminated, points)
    expected = jnp.real(_explicit_synthesis(canonical, points, "regular"))

    assert actual.shape == (3, 2)
    assert not jnp.iscomplexobj(actual)
    assert jnp.all(jnp.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)


def test_regular_origin_and_irregular_singular_lanes_survive_fused_synthesis():
    limit = 4
    center = limit - 1
    coefficients = jnp.zeros((limit, 2 * limit - 1), dtype=jnp.complex128)
    coefficients = coefficients.at[0, center].set(1.3)
    coefficients = coefficients.at[2, center + 1].set(0.4 - 0.2j)
    coefficients = coefficients.at[3, center - 2].set(-0.1 + 0.7j)
    points = jnp.asarray([[0.0, 0.0, 0.0], [0.4, -0.3, 0.9], [-0.6, 0.1, 0.7]])

    regular = spectral_api.SolidHarmonicPlan(
        limit, kind="regular", reality=False
    ).prepare()
    regular_values = regular.evaluate(coefficients, points)
    np.testing.assert_allclose(
        regular_values[0],
        1.3 / np.sqrt(4.0 * np.pi),
        rtol=0.0,
        atol=0.0,
    )

    irregular = spectral_api.SolidHarmonicPlan(
        limit, kind="irregular", reality=False
    ).prepare()
    irregular_values = irregular.evaluate(coefficients, points)
    assert jnp.isnan(jnp.real(irregular_values[0]))
    assert jnp.isnan(jnp.imag(irregular_values[0]))
    assert jnp.all(jnp.isfinite(irregular_values[1:]))
    np.testing.assert_allclose(
        irregular_values[1:],
        _explicit_synthesis(coefficients, points[1:], "irregular"),
        rtol=3e-12,
        atol=3e-12,
    )


def test_fused_synthesis_is_jittable_and_has_explicit_sum_coordinate_jvp():
    limit = 4
    coefficients = _complex_payload_coefficients(limit)[..., 0, 0]
    points = jnp.asarray([[0.4, -0.2, 0.8], [-0.5, 0.7, 1.2]], dtype=jnp.float64)
    direction = jnp.asarray([[0.1, 0.3, -0.2], [-0.25, 0.15, 0.4]], dtype=jnp.float64)
    prepared = spectral_api.SolidHarmonicPlan(
        limit, kind="irregular", reality=False
    ).prepare()
    compiled = eqx.filter_jit(
        lambda operator, modal, locations: operator.evaluate(modal, locations)
    )(prepared, coefficients, points)
    expected = _explicit_synthesis(coefficients, points, "irregular")
    np.testing.assert_allclose(compiled, expected, rtol=3e-12, atol=3e-12)

    _, actual_jvp = jax.jvp(
        lambda locations: prepared.evaluate(coefficients, locations),
        (points,),
        (direction,),
    )
    _, expected_jvp = jax.jvp(
        lambda locations: _explicit_synthesis(coefficients, locations, "irregular"),
        (points,),
        (direction,),
    )
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2e-11, atol=2e-11)


def test_plan_dtype_validation_resources_and_provenance_contract():
    real = spectral_api.SolidHarmonicPlan(3, reality=True).prepare()
    complex_ = spectral_api.SolidHarmonicPlan(3, reality=False).prepare()
    coefficients = jnp.zeros((3, 5), dtype=jnp.complex64).at[0, 2].set(1.0)
    points = jnp.asarray([[0.2, -0.4, 0.7]], dtype=jnp.float32)

    assert real.evaluate(coefficients, points).dtype == jnp.float32
    assert complex_.evaluate(coefficients, points).dtype == jnp.complex64
    assert real.layout.coefficient_shape == (3, 5)
    assert real.prepared_id != complex_.prepared_id
    assert real.plan.plan_id != complex_.plan.plan_id
    resources = dict(real.preparation.resource_counts)
    assert resources["logical_modes"] == 9
    assert resources["padded_coefficients"] == 15
    assert resources["persistent_array_bytes"] == 0
    assert resources["point_mode_table_entries"] == 0

    with pytest.raises(TypeError):
        spectral_api.SolidHarmonicPlan(True)
    with pytest.raises(ValueError):
        spectral_api.SolidHarmonicPlan(0)
    with pytest.raises(ValueError):
        spectral_api.SolidHarmonicPlan(3, kind="exterior")
    with pytest.raises(ValueError, match="must begin with shape"):
        real.evaluate(jnp.ones((3, 4)), points)
    with pytest.raises(ValueError, match="dimension 3"):
        real.evaluate(coefficients, jnp.ones((2, 2)))
    with pytest.raises(TypeError):
        real.evaluate(coefficients, jnp.ones((2, 3), dtype=complex))
