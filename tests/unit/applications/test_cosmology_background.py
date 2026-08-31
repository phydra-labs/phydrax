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


def test_background_rejects_invalid_dynamic_values():
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite flat"):
        value = cosmology.FLRWBackground(1.0, -0.1)
        jax.block_until_ready(value.hubble_constant)
    background = cosmology.FLRWBackground(1.0, 0.3)
    with pytest.raises(eqx.EquinoxRuntimeError, match="Scale factor"):
        jax.block_until_ready(background.hubble(0.0))


def test_growth_matches_einstein_de_sitter_and_is_jittable():
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
    expansion = plan.expansion_history(background)
    np.testing.assert_allclose(expansion.hubble(nodes), nodes ** (-1.5), rtol=1e-12)


def test_growth_plan_requires_increasing_nodes_ending_at_one():
    with pytest.raises(ValueError, match="end at a=1"):
        cosmology.FLRWGrowthPlan(jnp.asarray([0.1, 0.5]))
    with pytest.raises(ValueError, match="increasing"):
        cosmology.FLRWGrowthPlan(jnp.asarray([0.1, 0.1, 1.0]))
