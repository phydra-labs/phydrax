#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.equations import (
    IdealGasMaterial,
    NobleAbelStiffenedGasMaterial,
    StiffenedGasMaterial,
)


def _water() -> NobleAbelStiffenedGasMaterial:
    # Liquid-water NASG coefficients of Le Métayer & Saurel (2016).
    return NobleAbelStiffenedGasMaterial(
        1.19, 6.217e8, 6.61e-4, 3610.0, reference_energy=-1.177788e6
    )


def test_nasg_reduces_to_ideal_and_stiffened_gas() -> None:
    ideal = IdealGasMaterial(1.4, 287.0)
    limit = NobleAbelStiffenedGasMaterial(1.4, 0.0, 0.0, 287.0 / 0.4)
    density, pressure = jnp.asarray(1.2), jnp.asarray(1.0e5)
    pairs = (
        (limit.specific_internal_energy, ideal.specific_internal_energy),
        (limit.temperature, ideal.temperature),
        (limit.sound_speed, ideal.sound_speed),
        (limit.specific_enthalpy, ideal.specific_enthalpy),
    )
    for noble_abel, reference in pairs:
        assert float(noble_abel(density, pressure)) == pytest.approx(
            float(reference(density, pressure)), rel=1.0e-13
        )
    stiffened = StiffenedGasMaterial(4.4, 6.0e8, 1816.0, reference_energy=2.0e5)
    no_covolume = NobleAbelStiffenedGasMaterial(4.4, 6.0e8, 0.0, 1816.0, reference_energy=2.0e5)
    energy = jnp.asarray(3.0e5)
    assert float(no_covolume.pressure(jnp.asarray(1000.0), energy)) == pytest.approx(
        float(stiffened.pressure(jnp.asarray(1000.0), energy)), rel=1.0e-13
    )


def test_nasg_caloric_consistency_and_sound_speed() -> None:
    material = _water()
    density, pressure = jnp.asarray(1000.0), jnp.asarray(1.0e5)
    energy = material.specific_internal_energy(density, pressure)
    assert float(material.pressure(density, energy)) == pytest.approx(1.0e5, rel=1.0e-9)
    # c² = (∂p/∂ρ)_s = ∂p/∂ρ|_e + (p/ρ²) ∂p/∂e|_ρ.
    dp_drho = jax.grad(material.pressure, argnums=0)(density, energy)
    dp_de = jax.grad(material.pressure, argnums=1)(density, energy)
    isentropic = float(dp_drho + pressure / density**2 * dp_de)
    assert float(material.sound_speed(density, pressure)) ** 2 == pytest.approx(isentropic, rel=1.0e-10)
    enthalpy = material.specific_enthalpy(density, pressure)
    temperature = material.temperature(density, pressure)
    expected = 1.19 * 3610.0 * temperature + 6.61e-4 * pressure - 1.177788e6
    assert float(enthalpy) == pytest.approx(float(expected), rel=1.0e-12)
    assert float(material.specific_heat_cp(density, pressure)) == pytest.approx(1.19 * 3610.0)


def test_nasg_admissibility_requires_free_volume_and_positive_stiffened_pressure() -> None:
    material = _water()
    assert bool(material.admissible(jnp.asarray(1000.0), jnp.asarray(1.0e5)))
    assert not bool(material.admissible(jnp.asarray(1.0 / 6.0e-4), jnp.asarray(1.0e5)))
    assert not bool(material.admissible(jnp.asarray(1000.0), jnp.asarray(-7.0e8)))
    assert not bool(material.admissible(jnp.asarray(np.nan), jnp.asarray(1.0e5)))
    with pytest.raises(ValueError, match="Noble-Abel"):
        NobleAbelStiffenedGasMaterial(1.19, 6.217e8, -1.0e-4, 3610.0)
    with pytest.raises(ValueError, match="Noble-Abel"):
        NobleAbelStiffenedGasMaterial(1.0, 6.217e8, 6.61e-4, 3610.0)
