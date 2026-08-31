import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def _provenance(scale):
    return cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_id="flat-flrw",
        numerical_policy_id="test-policy",
        scale_id=scale.scale_id,
        source_kind="external",
        differentiability="coordinate-only",
    )


def test_expansion_and_growth_products_interpolate_with_stable_shapes():
    scale = cosmology.CosmologyScaleContract("L", "M", "T")
    provenance = _provenance(scale)
    nodes = jnp.asarray([0.25, 0.5, 1.0])
    expansion = cosmology.ExpansionHistory(
        nodes, jnp.asarray([8.0, 4.0, 2.0]), scale, provenance
    )
    np.testing.assert_allclose(expansion.hubble(nodes), [8.0, 4.0, 2.0])
    assert expansion.hubble(0.75).shape == ()

    growth = cosmology.LagrangianGrowthHistory(
        nodes,
        nodes,
        jnp.ones_like(nodes),
        (3.0 / 7.0) * nodes**2,
        2.0 * jnp.ones_like(nodes),
        scale,
        provenance,
    )
    first, first_rate, second, second_rate = growth.evaluate(0.5)
    np.testing.assert_allclose(first, 0.5)
    np.testing.assert_allclose(first_rate, 1.0)
    np.testing.assert_allclose(second, 3.0 / 28.0)
    np.testing.assert_allclose(second_rate, 2.0)


def test_matter_power_table_bilinear_query_and_domain_contract():
    scale = cosmology.CosmologyScaleContract("Mpc", "mass", "time")
    provenance = _provenance(scale)
    scales = jnp.asarray([0.5, 1.0])
    wavenumbers = jnp.asarray([1.0, 2.0, 4.0])
    values = scales[:, None] * wavenumbers[None, :]
    table = cosmology.MatterPowerTable(scales, wavenumbers, values, scale, provenance)
    query = table.evaluate(jnp.asarray([1.0, 3.0]), 0.75)
    np.testing.assert_allclose(query, [0.75, 2.25], rtol=1e-12)
    compiled = eqx.filter_jit(table.evaluate)(jnp.asarray([1.5, 2.5]), 0.75)
    np.testing.assert_allclose(compiled, [1.125, 1.875], rtol=1e-12)

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="outside"):
        jax.block_until_ready(table.evaluate(jnp.asarray([0.5]), 0.75))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="outside"):
        jax.block_until_ready(table.evaluate(jnp.asarray([1.5]), 1.1))


def test_product_validation_rejects_invalid_values_and_scale_mismatch():
    scale = cosmology.CosmologyScaleContract("L", "M", "T")
    other = cosmology.CosmologyScaleContract("other-L", "M", "T")
    provenance = _provenance(scale)
    with pytest.raises(ValueError, match="scale and provenance"):
        cosmology.ExpansionHistory([0.5, 1.0], [2.0, 1.0], other, provenance)
    with pytest.raises(eqx.EquinoxRuntimeError, match="non-negative"):
        table = cosmology.MatterPowerTable(
            [0.5, 1.0],
            [1.0, 2.0],
            [[1.0, -1.0], [1.0, 1.0]],
            scale,
            provenance,
        )
        jax.block_until_ready(table.power_values)
