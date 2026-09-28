#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import accelerator


scale = phx.ElectromagneticScaleContract.si()
light = float(scale.speed_of_light)
electron_charge = float(scale.elementary_charge)
electron_rest_energy = float(scale.electron_mass) * light**2
momentum = 1.0e9 * electron_charge / light

# A 2 GHz, Q = 5 cavity mode seen by a 1 mm rms Gaussian bunch of 1e10 electrons.
resonator = accelerator.ResonatorWake(1.0e4, 5.0, 2.0e9, kind="longitudinal", scale=scale)
plan = resonator.wake_function(np.linspace(0.0, 0.2, 4001))
rng = np.random.default_rng(0)
count = 4096
coordinates = np.zeros((count, 6))
coordinates[:, 4] = rng.normal(0.0, 1.0e-3, count)
bunch = accelerator.AcceleratorBunch(
    jnp.asarray(coordinates),
    jnp.full((count,), 1.0e10 / count),
    jnp.arange(count, dtype=jnp.int32),
    reference_rest_energy=electron_rest_energy,
    reference_momentum=momentum,
    reference_charge=electron_charge,
    bunch_id="gaussian",
)
kick = accelerator.apply_wake(plan, bunch, scale, bin_count=64)

# Gaussian-bunch loss factor from the impedance: k = (1/π) ∫ Re Z e^{-(ωσ/c)²} dω.
frequencies = np.linspace(0.0, 400.0e9, 400001)
impedance = np.asarray(resonator.impedance(frequencies)).real
omega = 2.0 * math.pi * frequencies
spectrum = np.exp(-((omega * 1.0e-3 / light) ** 2))
gaussian_loss_factor = float(np.trapezoid(impedance * spectrum, omega)) / math.pi

# Resistive wall of a 1 cm copper pipe, 10 m long.
wall = accelerator.ResistiveWallWake(1.0e-2, 5.8e7, 10.0, scale=scale)
wall_plan = wall.wake_function("longitudinal", np.linspace(0.0, 0.01, 11))

print(
    {
        "point_loss_factor_V_per_C": resonator.loss_factor,
        "gaussian_loss_factor_from_impedance_V_per_C": gaussian_loss_factor,
        "tracked_loss_factor_V_per_C": float(kick.loss_factor),
        "causality_defect": float(kick.causality_defect),
        "resistive_wall_s0_m": wall.characteristic_length,
        "resistive_wall_W0_V_per_C": float(wall_plan.wake_values[0]),
    }
)
