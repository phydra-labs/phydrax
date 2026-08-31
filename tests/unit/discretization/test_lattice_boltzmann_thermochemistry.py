#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

from phydrax.discretization.lattice_boltzmann import (
    BoussinesqCouplingPlan,
    D2Q9,
    LatticeBoltzmannPrecisionPolicy,
    SpeciesLatticeBoltzmannPlan,
    ThermalLatticeBoltzmannPlan,
)
from phydrax.discretization.lattice_boltzmann._species import (
    collide_species,
    species_equilibrium,
    species_raw_moments,
)
from phydrax.discretization.lattice_boltzmann._thermal import (
    boussinesq_force,
    collide_thermal,
    sensible_energy_from_temperature,
    temperature_from_sensible_energy,
    thermal_equilibrium,
    thermal_raw_moments,
)


def test_thermal_energy_distribution_round_trips_temperature_and_moments():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    plan = ThermalLatticeBoltzmannPlan(2.5, 0.4, reference_temperature=300.0)
    temperature = jnp.asarray([[310.0, 315.0], [305.0, 300.0]])
    velocity = jnp.broadcast_to(jnp.asarray((0.02, -0.01)), temperature.shape + (2,))
    energy = sensible_energy_from_temperature(temperature, plan)
    equilibrium = thermal_equilibrium(energy, velocity, lattice, precision)
    recovered, flux = thermal_raw_moments(equilibrium, lattice, precision)

    np.testing.assert_allclose(
        temperature_from_sensible_energy(energy, plan), temperature
    )
    np.testing.assert_allclose(recovered, energy, atol=1e-12)
    np.testing.assert_allclose(flux, energy[..., None] * velocity, atol=1e-12)
    collided = collide_thermal(
        equilibrium,
        velocity,
        jnp.zeros_like(energy),
        plan,
        lattice,
        precision,
        jnp.asarray(0.1),
        jnp.asarray(1.0),
    )
    assert collided.successful
    np.testing.assert_allclose(collided.populations, equilibrium, atol=1e-12)


def test_species_distributions_preserve_each_scalar_moment():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    plan = SpeciesLatticeBoltzmannPlan(jnp.asarray((0.1, 0.2)))
    concentration = jnp.asarray([[[1.0, 0.5], [0.8, 0.2]]])
    velocity = jnp.broadcast_to(jnp.asarray((0.01, 0.0)), concentration.shape[:-1] + (2,))
    equilibrium = species_equilibrium(concentration, velocity, lattice, precision)
    recovered, flux = species_raw_moments(equilibrium, lattice, precision)

    np.testing.assert_allclose(recovered, concentration, atol=1e-12)
    np.testing.assert_allclose(
        flux,
        oe.contract("...s,...d->...sd", concentration, velocity),
        atol=1e-12,
    )
    result = collide_species(
        equilibrium,
        velocity,
        jnp.zeros_like(concentration),
        plan,
        lattice,
        precision,
        jnp.asarray(0.1),
        jnp.asarray(1.0),
    )
    assert result.successful
    np.testing.assert_allclose(result.populations, equilibrium, atol=1e-12)
    assert plan.species_count == 2


def test_boussinesq_force_uses_declared_reference_temperature_and_gravity():
    plan = BoussinesqCouplingPlan(
        2.0,
        0.1,
        jnp.asarray((0.0, -9.81)),
        reference_temperature=300.0,
    )
    temperature = jnp.asarray((300.0, 310.0))
    force = boussinesq_force(temperature, plan)

    np.testing.assert_allclose(force[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(force[1], jnp.asarray((0.0, -19.62)), atol=1e-14)
