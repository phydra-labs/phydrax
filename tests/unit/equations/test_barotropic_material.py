#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_tait_material_matches_reference_state_and_energy_identity():
    material = phx.equations.TaitBarotropicMaterial(
        997.0,
        20.0,
        exponent=7.0,
        background_pressure=13.0,
    )
    reference_density = jnp.asarray(material.reference_density)

    assert material.pressure(reference_density) == pytest.approx(13.0)
    assert material.sound_speed(reference_density) == pytest.approx(20.0)
    assert material.specific_internal_energy(reference_density) == pytest.approx(0.0)

    for density in (850.0, 997.0, 1150.0):
        density_ = jnp.asarray(density)
        energy_derivative = jax.grad(material.specific_internal_energy)(density_)
        pressure_identity = material.pressure(density_) / density_**2
        assert jnp.allclose(
            energy_derivative,
            pressure_identity,
            rtol=2e-12,
            atol=2e-14,
        )


def test_tait_material_admissibility_is_explicit_without_clipping():
    material = phx.equations.TaitBarotropicMaterial(
        1.0,
        2.0,
        density_floor=0.1,
    )
    density = jnp.asarray([1.0, 0.1, 0.09, jnp.nan])

    assert jnp.array_equal(
        material.admissible(density),
        jnp.asarray([True, True, False, False]),
    )
    assert material.pressure(jnp.asarray(0.09)) != material.pressure(jnp.asarray(0.1))


def test_tait_material_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="positive finite"):
        phx.equations.TaitBarotropicMaterial(0.0, 1.0)
    with pytest.raises(ValueError, match="exponent > 1"):
        phx.equations.TaitBarotropicMaterial(1.0, 1.0, exponent=1.0)
    with pytest.raises(ValueError, match="background pressure"):
        phx.equations.TaitBarotropicMaterial(
            1.0,
            1.0,
            background_pressure=jnp.nan,
        )
