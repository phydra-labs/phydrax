# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cold-plasma rays: cutoff reflection on a density ramp and Faraday rotation.

Traces an ordinary-mode ray into a linear density ramp and compares its turning
point with ``x_t = L n₀² cos²θ₀``, then carries a linearly polarized wave along
``B₀`` through a density gradient with coupled Stokes transfer and compares the
rotation of the plane of polarization with the rotation-measure integral.
"""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.radiation_transport import PlasmaRayTransferPlan
from phydrax.electromagnetics import PlasmaWaveMode
from phydrax.optics.geometric import (
    ColdPlasmaHamiltonian,
    ColdPlasmaProfile,
    DispersionRayPlan,
)


def main() -> None:
    scale = phx.ElectromagneticScaleContract.si()
    coordinates = phx.SpatialCoordinateContract(phx.units.METER)
    charge = float(scale.elementary_charge)
    mass = float(scale.electron_mass)
    permittivity = float(scale.vacuum_permittivity)
    light = float(scale.speed_of_light)

    omega = 2.0 * np.pi * 1.0e9
    critical = permittivity * mass * omega**2 / charge**2
    ramp = ColdPlasmaProfile(
        scale,
        coordinates,
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        density=lambda x: jnp.stack([critical * (0.1 + x[0])]),
        magnetic_field=lambda x: jnp.stack([0.0 * x[0], 0.0 * x[0], 0.02 + 0.0 * x[0]]),
        profile_id="linear-ramp",
    )
    ordinary = ColdPlasmaHamiltonian(
        ramp, angular_frequency=omega, mode=PlasmaWaveMode.ORDINARY
    )
    angle = 0.5
    rays = (
        DispersionRayPlan(ordinary, 0.02, 130, hamiltonian_tolerance=1.0e-6)
        .prepare()
        .integrate(np.zeros((1, 3)), np.asarray(((np.cos(angle), np.sin(angle), 0.0),)))
    )
    if not bool(rays.evidence.successful):
        raise RuntimeError("Plasma ray evidence rejected the ordinary-mode ray.")
    print(
        "turning_point",
        float(np.max(rays.position_history[:, 0, 0])),
        "analytic",
        0.9 * np.cos(angle) ** 2,
    )

    omega = 2.0 * np.pi * 1.0e10
    gyro = 0.1
    base = 0.05 * permittivity * mass * omega**2 / charge**2
    field = gyro * mass * omega / charge
    along_field = ColdPlasmaProfile(
        scale,
        coordinates,
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        density=lambda x: jnp.stack([base * (1.0 + 2.0 * x[2])]),
        magnetic_field=lambda x: jnp.stack([0.0 * x[2], 0.0 * x[2], field + 0.0 * x[2]]),
        profile_id="faraday-ramp",
    )
    right = ColdPlasmaHamiltonian(
        along_field, angular_frequency=omega, mode=PlasmaWaveMode.RIGHT
    )
    count = 100
    rays = (
        DispersionRayPlan(right, 0.01, count, hamiltonian_tolerance=1.0e-6)
        .prepare()
        .integrate(np.zeros((1, 3)), np.asarray(((0.0, 0.0, 1.0),)))
    )
    path = right.sample_path(rays, (1.0, 0.0, 0.0))
    zeros = np.zeros((1, count, 2))
    transfer = PlasmaRayTransferPlan(
        path, coupling="strong", anisotropy_tolerance=0.1
    ).evaluate(zeros, zeros, np.asarray(((1.0, 1.0, 0.0, 0.0),)))
    stokes = np.asarray(transfer.stokes[0])
    end = float(np.sum(path.segment_lengths))
    z = np.linspace(0.0, end, 4001)
    plasma = 0.05 * (1.0 + 2.0 * z)
    integrand = (
        0.5
        * omega
        / light
        * (np.sqrt(1.0 - plasma / (1.0 + gyro)) - np.sqrt(1.0 - plasma / (1.0 - gyro)))
    )
    rotation = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z)))
    print(
        "faraday_rotation",
        float(transfer.faraday_rotation[0]),
        "integral",
        rotation,
        "plane_angle",
        0.5 * float(np.arctan2(stokes[2], stokes[1])),
    )
    print(
        "invariant_conserved",
        bool(np.allclose(transfer.invariant, transfer.incident_invariant)),
        "successful",
        bool(transfer.successful),
    )


if __name__ == "__main__":
    main()
