# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Mode-resolved thermal cyclotron emission in a dispersive magnetized plasma.

Evaluates the exact harmonic sum for a mildly relativistic Jüttner population,
checks Kirchhoff's law against the absorption computed from the distribution
gradient, and compares the continuous-harmonic route at a high harmonic.
"""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.electromagnetics import (
    ColdPlasmaDielectric,
    MagnetobremsstrahlungPlan,
    PlasmaWaveMode,
    ThermalJuttnerDistribution,
)


def main() -> None:
    scale = phx.ElectromagneticScaleContract.si()
    background = ColdPlasmaDielectric(
        scale,
        densities=[1.0e18],
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        magnetic_field=[0.0, 0.0, 1.0],
    )
    theta_e = 0.02
    plan = MagnetobremsstrahlungPlan(
        background,
        ThermalJuttnerDistribution(theta_e),
        emitter_density=1.0e15,
        maximum_harmonics=32,
    )
    gyrofrequency = float(plan.gyrofrequency)
    omega = jnp.asarray([2.2, 2.9, 3.6]) * gyrofrequency
    result = plan.evaluate(omega, 1.2)
    extraordinary = result.select(PlasmaWaveMode.EXTRAORDINARY, result.emission)
    ordinary = result.select(PlasmaWaveMode.ORDINARY, result.emission)
    print("harmonics", np.asarray(result.harmonic_range[..., 0, :]).tolist())
    print("x_mode_emission", np.asarray(extraordinary).tolist())
    print("o_to_x_ratio", np.asarray(ordinary / extraordinary).tolist())

    index = result.faraday.wave.refractive_index.real
    boltzmann = float(scale.relativity.boltzmann_constant)
    thermal_energy = (
        theta_e * float(scale.electron_mass) * float(scale.speed_of_light) ** 2
    )
    rayleigh_jeans = (
        thermal_energy
        * omega[:, None] ** 2
        / (8.0 * np.pi**3 * float(scale.speed_of_light) ** 2)
    )
    kirchhoff = result.emission / (index**2 * rayleigh_jeans * result.absorption)
    print("kirchhoff_ratio", np.asarray(kirchhoff).tolist())
    print("temperature_k", thermal_energy / boltzmann)

    hot = ThermalJuttnerDistribution(0.3, tail_tolerance=1.0e-10)
    high = 24.0 * gyrofrequency
    exact = MagnetobremsstrahlungPlan(
        background, hot, emitter_density=1.0e15, maximum_harmonics=512
    ).evaluate(high, 1.2)
    fast = MagnetobremsstrahlungPlan(
        background, hot, emitter_density=1.0e15, route="continuous-harmonic"
    ).evaluate(high, 1.2)
    print(
        "continuous_relative_error",
        np.asarray(fast.emission / exact.emission - 1.0).tolist(),
        "effective_harmonic",
        np.asarray(fast.effective_harmonic).tolist(),
    )


if __name__ == "__main__":
    main()
