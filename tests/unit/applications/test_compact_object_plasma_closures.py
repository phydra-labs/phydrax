#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import RelativityScaleContract
from phydrax.applications.compact_objects._plasma_closures import (
    BoundedNonthermalParticleDistribution,
    PlasmaKineticRegime,
    TwoTemperatureElectronIonClosure,
    TwoTemperaturePlasmaState,
)
from phydrax.units import KILOGRAM


def _scale():
    return RelativityScaleContract.geometric(KILOGRAM)


def test_two_temperature_exchange_relaxes_exactly_without_creating_energy():
    closure = TwoTemperatureElectronIonClosure(
        _scale(),
        equilibration_time=2.0,
        minimum_temperature=1.0,
        maximum_temperature=1.0e9,
    )
    state = TwoTemperaturePlasmaState(jnp.asarray(100.0), jnp.asarray(400.0))

    result = jax.jit(closure.advance)(
        jnp.asarray(2.0e20),
        jnp.asarray(1.0e20),
        state,
        jnp.asarray(2.0),
    )

    assert float(result.candidate.electron_temperature) > 100.0
    assert float(result.candidate.ion_temperature) < 400.0
    np.testing.assert_allclose(
        result.electron_energy_exchange + result.ion_energy_exchange,
        0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(result.equilibrium_temperature, 200.0, rtol=1.0e-6)
    assert bool(result.converged)
    assert bool(result.qualified)
    assert bool(result.derivative_valid)


def test_two_temperature_closure_rejects_unrepresented_kinetic_regimes():
    with pytest.raises(NotImplementedError, match="velocity-space kinetic"):
        TwoTemperatureElectronIonClosure(
            _scale(),
            equilibration_time=1.0,
            minimum_temperature=1.0,
            maximum_temperature=1.0e6,
            regime=PlasmaKineticRegime.FULL_PHASE_SPACE,
        )


def test_bounded_nonthermal_record_has_content_identity_and_physical_moments():
    distribution = BoundedNonthermalParticleDistribution(
        _scale(),
        jnp.asarray((1.0, 2.0, 8.0)),
        jnp.asarray((3.0, 1.0)),
        particle_rest_energy=2.0,
        species="electron",
    )
    same = BoundedNonthermalParticleDistribution(
        _scale(),
        jnp.asarray((1.0, 2.0, 8.0)),
        jnp.asarray((3.0, 1.0)),
        particle_rest_energy=2.0,
        species="electron",
    )
    changed = BoundedNonthermalParticleDistribution(
        _scale(),
        jnp.asarray((1.0, 2.0, 8.0)),
        jnp.asarray((3.0, 2.0)),
        particle_rest_energy=2.0,
        species="electron",
    )

    moments = jax.jit(lambda value: value.moments())(distribution)

    np.testing.assert_allclose(moments.number_density, 4.0)
    assert float(moments.kinetic_energy_density) > 0.0
    assert float(moments.isotropic_pressure) > 0.0
    assert bool(moments.qualified)
    assert distribution.distribution_id == same.distribution_id
    assert distribution.distribution_id != changed.distribution_id


def test_nonthermal_record_rejects_unbounded_or_anisotropic_support():
    with pytest.raises(ValueError, match="invalid"):
        BoundedNonthermalParticleDistribution(
            _scale(),
            jnp.asarray((1.0, 2.0, jnp.inf)),
            jnp.ones(2),
            particle_rest_energy=1.0,
            species="electron",
        )
    with pytest.raises(NotImplementedError, match="anisotropic"):
        BoundedNonthermalParticleDistribution(
            _scale(),
            jnp.asarray((1.0, 2.0)),
            jnp.ones(1),
            particle_rest_energy=1.0,
            species="electron",
            regime=PlasmaKineticRegime.ANISOTROPIC_GYROTROPIC,
        )
