# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def test_empirical_cubature_preserves_physical_complex_feature_moments_and_mask():
    samples = jnp.linspace(-1.0, 1.0, 21)
    source = phx.integration.materialize(
        phx.integration.weighted(
            samples,
            jnp.zeros((21,)),
            normalized=False,
            target_mass=jnp.asarray(3.0),
        )
    )
    basis = jnp.stack((jnp.ones_like(samples), samples + 1j * samples**2), axis=1)
    compressed = phx.integration.empirical_cubature(
        source,
        basis,
        phx.integration.EmpiricalCubaturePlan(),
    )
    source_value = phx.integration.reduce(
        lambda value: jnp.stack((jnp.ones_like(value), value, value**2), axis=-1),
        source,
    )
    compressed_value = phx.integration.reduce(
        lambda value: jnp.stack((jnp.ones_like(value), value, value**2), axis=-1),
        compressed,
    )
    assert jnp.allclose(compressed_value.value.data, source_value.value.data, atol=1e-6)
    assert jnp.isclose(compressed.target.target_mass, 3.0)
    assert jnp.array_equal(
        compressed.batch.mask,
        compressed.batch.mask & jnp.isfinite(compressed.batch.log_weights),
    )
