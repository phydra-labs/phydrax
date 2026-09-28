#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.electromagnetics import (
    ColdPlasmaDielectric,
    KineticDispersionProblem,
    KineticPlasmaDielectric,
    LossConeDistribution,
    PlasmaWaveMode,
    RelativisticWeakGrowthPlan,
)


scale = phx.ElectromagneticScaleContract.si()
charge = float(scale.elementary_charge)
electron_mass = float(scale.electron_mass)
permittivity = float(scale.vacuum_permittivity)
light = float(scale.speed_of_light)

# Landau-damped Langmuir branch at kλ_D = 0.3 … 0.5.
density = 1.0e18
temperature = 100.0 * charge
langmuir = KineticPlasmaDielectric(
    scale,
    densities=[density],
    charge_numbers=[-1.0],
    mass_ratios=[1.0],
    magnetic_field=[0.0, 0.0, 0.5],
    parallel_temperatures=[temperature],
    perpendicular_temperatures=[temperature],
)
omega_p = math.sqrt(density * charge * charge / (permittivity * electron_mass))
debye = math.sqrt(permittivity * temperature / (density * charge * charge))
landau = KineticDispersionProblem(langmuir, model="electrostatic").solve(
    jnp.linspace(0.3, 0.5, 5) / debye, jnp.asarray(0.0), jnp.asarray(1.16 * omega_p + 0j)
)

# Whistler anisotropy instability: T⊥/T∥ = 3 electrons, parallel propagation.
field = 1.0e-3
omega_c = charge * field / electron_mass
whistler_density = (4.0 * omega_c) ** 2 * permittivity * electron_mass / charge**2
hot = KineticPlasmaDielectric(
    scale,
    densities=[whistler_density],
    charge_numbers=[-1.0],
    mass_ratios=[1.0],
    magnetic_field=[0.0, 0.0, field],
    parallel_temperatures=[1.0e3 * charge],
    perpendicular_temperatures=[3.0e3 * charge],
)
whistler = KineticDispersionProblem(hot).solve(
    jnp.linspace(0.8, 1.2, 5) * 4.0 * omega_c / light,
    jnp.asarray(0.0),
    jnp.asarray((0.42 + 0.0005j) * omega_c),
)

# Electron-cyclotron maser growth of the O mode from a loss-cone driver.
background = ColdPlasmaDielectric(
    scale,
    densities=[1.0e15],
    charge_numbers=[-1.0],
    mass_ratios=[1.0],
    magnetic_field=[0.0, 0.0, 0.1],
)
maser = RelativisticWeakGrowthPlan(
    background, LossConeDistribution(0.05, index=1), density=1.0e13
).evaluate(
    jnp.asarray([0.997, 0.998, 0.999]) * charge * 0.1 / electron_mass,
    jnp.asarray(0.5 * math.pi),
    PlasmaWaveMode.ORDINARY,
)

print(
    {
        "landau_omega_over_omega_p": np.round(
            np.asarray(landau.frequencies) / omega_p, 4
        ).tolist(),
        "landau_status": np.asarray(landau.status).tolist(),
        "whistler_growth_over_omega_c": np.round(
            np.asarray(whistler.frequencies).imag / omega_c, 5
        ).tolist(),
        "whistler_status": np.asarray(whistler.status).tolist(),
        "maser_growth_per_s": np.asarray(maser.growth_rate).tolist(),
        "maser_status": np.asarray(maser.status).tolist(),
    }
)
