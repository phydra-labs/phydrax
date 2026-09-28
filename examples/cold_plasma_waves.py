#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.electromagnetics import ColdPlasmaDielectric, PlasmaWaveMode


scale = phx.ElectromagneticScaleContract.si()
proton_mass_ratio = 1836.152673426
plasma = ColdPlasmaDielectric(
    scale,
    densities=[1.0e11, 1.0e11],
    charge_numbers=[-1.0, 1.0],
    mass_ratios=[1.0, proton_mass_ratio],
    magnetic_field=[0.0, 0.0, 5.0e-5],
)
electron_plasma = float(jnp.sqrt(plasma.plasma_frequency_squared[0]))
electron_cyclotron = abs(float(plasma.cyclotron_frequency[0]))

frequencies = plasma.characteristic_frequencies()
positive_cutoffs = {
    name: float(np.max(np.asarray(roots).real))
    for name, roots in (
        ("right", frequencies.right_cutoffs),
        ("left", frequencies.left_cutoffs),
        ("plasma", frequencies.plasma_cutoffs),
        ("upper_hybrid", frequencies.hybrid_resonances),
    )
}

whistler_frequency = 0.2 * electron_cyclotron
angles = jnp.linspace(0.0, 1.2, 7)
whistler = plasma.refractive_indices(whistler_frequency, angles)
whistler_index = whistler.select(PlasmaWaveMode.RIGHT, whistler.refractive_index)
cone = plasma.resonance_cone(whistler_frequency)

radio_frequency = 20.0 * electron_plasma
faraday = plasma.faraday_coefficients(radio_frequency, 0.0)

print(
    {
        "cutoffs_rad_per_s": positive_cutoffs,
        "whistler_refractive_index": np.asarray(whistler_index.real).round(3).tolist(),
        "whistler_status": np.asarray(whistler.status).tolist(),
        "resonance_cone_deg": float(jnp.degrees(cone.angle)),
        "faraday_rotation_rad_per_m": float(faraday.rotation.real),
    }
)
