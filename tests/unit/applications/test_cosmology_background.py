import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def test_flat_background_limits_and_parameter_gradient():
    matter = cosmology.FLRWBackground(2.0, 1.0)
    radiation = cosmology.FLRWBackground(3.0, 0.0, radiation_density=1.0)
    np.testing.assert_allclose(matter.hubble(0.25), 16.0, rtol=1e-12)
    np.testing.assert_allclose(radiation.hubble(0.25), 48.0, rtol=1e-12)
    np.testing.assert_allclose(matter.hubble(1.0), 2.0, rtol=1e-12)

    def hubble_at_half(matter_density):
        background = cosmology.FLRWBackground(
            2.0,
            matter_density,
            dark_energy_density=1.0 - matter_density,
        )
        return background.hubble(0.5)

    value, derivative = jax.value_and_grad(hubble_at_half)(jnp.asarray(0.3))
    epsilon = 1.0e-5
    finite_difference = (
        hubble_at_half(0.3 + epsilon) - hubble_at_half(0.3 - epsilon)
    ) / (2.0 * epsilon)
    assert jnp.isfinite(value)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5)


def test_curvature_and_cpl_expansion_match_closed_form():
    background = cosmology.FLRWBackground(
        70.0,
        0.3,
        curvature_density=0.05,
        dark_energy_w0=-0.9,
        dark_energy_wa=0.2,
    )
    scale = jnp.asarray(0.6)
    dark_scaling = scale ** (-3.0 * (1.0 - 0.9 + 0.2)) * jnp.exp(
        -3.0 * 0.2 * (1.0 - scale)
    )
    expected = 0.3 / scale**3 + 0.05 / scale**2 + 0.65 * dark_scaling
    np.testing.assert_allclose(background.expansion_squared(scale), expected, rtol=1e-12)
    assert background.equation_of_state(scale) == pytest.approx(-0.82)
    assert (
        background.realization.physical_state_form_id
        == background.physical_state.state_form_id
    )
    assert background.realization.parameter_names[-2:] == (
        "dark_energy_w0",
        "dark_energy_wa",
    )


def test_flrw_distance_plan_flat_milne_de_sitter_and_duality():
    plan = cosmology.FLRWDistancePlan(light_speed=1.0, order=96)
    redshift = jnp.asarray(1.0)
    milne = cosmology.FLRWBackground(1.0, 0.0, curvature_density=1.0)
    milne_result = plan.evaluate(milne, redshift)
    logarithm = np.log(2.0)
    np.testing.assert_allclose(
        milne_result.radial_comoving_distance, logarithm, rtol=2e-12
    )
    np.testing.assert_allclose(
        milne_result.transverse_comoving_distance, np.sinh(logarithm), rtol=2e-12
    )
    np.testing.assert_allclose(
        milne_result.luminosity_distance,
        4.0 * milne_result.angular_diameter_distance,
        rtol=1e-12,
    )

    de_sitter = cosmology.FLRWBackground(1.0, 0.0)
    result = plan.evaluate(de_sitter, redshift)
    np.testing.assert_allclose(result.radial_comoving_distance, 1.0, rtol=1e-12)
    np.testing.assert_allclose(result.transverse_comoving_distance, 1.0, rtol=1e-12)
    np.testing.assert_allclose(result.lookback_time, np.log(2.0), rtol=2e-12)

    def transverse(curvature):
        model = cosmology.FLRWBackground(1.0, 0.3, curvature_density=curvature)
        return plan.transverse_comoving_distance(model, 0.5)

    assert jnp.isfinite(jax.grad(transverse)(jnp.asarray(0.0)))


def test_background_and_flat_execution_reject_invalid_domains():
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="satisfying closure"):
        value = cosmology.FLRWBackground(1.0, -0.1)
        jax.block_until_ready(value.hubble_constant)
    background = cosmology.FLRWBackground(1.0, 0.3)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="Scale factor"):
        jax.block_until_ready(background.hubble(0.0))
    curved = cosmology.FLRWBackground(1.0, 0.3, curvature_density=0.01)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="zero spatial curvature"
    ):
        jax.block_until_ready(curved.require_flat(jnp.asarray(1.0)))


def test_growth_matches_einstein_de_sitter_and_supports_flat_cpl():
    nodes = jnp.geomspace(1.0e-2, 1.0, 24)
    background = cosmology.FLRWBackground(1.0, 1.0)
    plan = cosmology.FLRWGrowthPlan(nodes)
    history = eqx.filter_jit(plan.solve)(background)
    np.testing.assert_allclose(history.first_order_growth, nodes, rtol=2e-6)
    np.testing.assert_allclose(history.first_order_rate, 1.0, rtol=2e-6)
    np.testing.assert_allclose(
        history.second_order_growth,
        (3.0 / 7.0) * nodes**2,
        rtol=5e-6,
    )
    np.testing.assert_allclose(history.second_order_rate, 2.0, rtol=5e-6)

    cpl = cosmology.FLRWBackground(
        1.0,
        0.3,
        dark_energy_w0=-0.9,
        dark_energy_wa=0.1,
    )
    cpl_history = plan.solve(cpl)
    assert jnp.all(jnp.isfinite(cpl_history.first_order_growth))

    curved = cosmology.FLRWBackground(1.0, 0.3, curvature_density=0.01)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="zero spatial curvature"
    ):
        jax.block_until_ready(plan.solve(curved).first_order_growth)
