import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.compressible_flow import (
    NormalShockReferencePlan,
    ObliqueShockReferencePlan,
    PrandtlMeyerReferencePlan,
)


def test_normal_shock_matches_calorically_perfect_reference():
    result = NormalShockReferencePlan(1.4).evaluate(2.0)
    assert bool(result.successful)
    np.testing.assert_allclose(result.density_ratio, 8.0 / 3.0, rtol=2.0e-7)
    np.testing.assert_allclose(result.pressure_ratio, 4.5, rtol=2.0e-7)
    np.testing.assert_allclose(result.temperature_ratio, 1.6875, rtol=2.0e-7)
    np.testing.assert_allclose(result.downstream_mach, jnp.sqrt(1.0 / 3.0), rtol=2.0e-7)


def test_oblique_shock_selects_weak_and_strong_branches():
    theta = np.deg2rad(10.0)
    weak = ObliqueShockReferencePlan(1.4, "weak").evaluate(2.0, theta)
    strong = ObliqueShockReferencePlan(1.4, "strong").evaluate(2.0, theta)

    assert bool(weak.successful)
    assert bool(strong.successful)
    np.testing.assert_allclose(np.rad2deg(float(weak.shock_angle)), 39.314, atol=2.0e-3)
    assert weak.shock_angle < strong.shock_angle
    assert weak.downstream_mach > 1.0
    assert strong.downstream_mach < 1.0
    assert abs(float(weak.residual)) < 1.0e-10


def test_oblique_shock_rejects_detached_configuration():
    with pytest.raises(ValueError, match="detached"):
        ObliqueShockReferencePlan(1.4).evaluate(2.0, np.deg2rad(30.0))


def test_prandtl_meyer_inversion_recovers_turning_angle():
    turning = np.deg2rad(10.0)
    result = PrandtlMeyerReferencePlan(1.4).evaluate(2.0, turning)

    assert bool(result.successful)
    np.testing.assert_allclose(result.downstream_mach, 2.384887, rtol=2.0e-5)
    np.testing.assert_allclose(
        result.downstream_angle - result.upstream_angle, turning, atol=2.0e-10
    )
    assert abs(float(result.residual)) < 1.0e-10
