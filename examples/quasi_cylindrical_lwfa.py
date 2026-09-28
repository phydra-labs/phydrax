#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quasi-cylindrical laser-wakefield acceleration with an antenna-launched laser.

Plasma units: lengths in ``c/ω_p``, times in ``1/ω_p``, fields in ``m_e c ω_p/e``
(electrons have ``q/m = −1``, the electron density is one). A linearly polarized
``m = 1`` Gaussian pulse (``a₀ = 0.5``, ``k₀ = 5k_p``) is launched by a one-way
sheet antenna in vacuum, enters a cold plasma, and drives an ``m = 0`` wake. The
run prints the on-axis wake amplitude next to the one-dimensional linear estimate
``E_max = (√π/2)(a₀²/2) k_pσ exp(−k_p²σ²/4)`` (σ = L/√2 for a
``cos``-averaged ``a² ∝ exp(−2ξ²/L²)``); the finite waist lowers the on-axis
three-dimensional wake below it. The Gauss residual stays at roundoff.
"""

import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


spectral = phx.solver.maxwell.spectral
D = phx.discretization

K0 = 5.0
WAVELENGTH = 2.0 * np.pi / K0
AXIAL = 144
LENGTH = AXIAL * WAVELENGTH / 12.0  # 12 axial cells per laser wavelength
RADIUS, RADIAL, MODES = 6.0, 48, 2
A0, WAIST, DURATION = 0.5, 2.5, 1.0
ANTENNA, PEAK_TIME = 1.0, 2.5
PLASMA_LOWER, PLASMA_UPPER, PLASMA_RADIUS = 5.0, LENGTH - 0.2, 4.5
PER_CELL = (2, 2, 4)  # particles per cell along z, r, θ
FINAL_TIME = 10.0

grid = D.pic.QuasiCylindricalGrid(RADIUS, RADIAL, 0.0, LENGTH, AXIAL, MODES)

# Evenly spaced cold plasma with weights ∝ r (uniform density one).
spacing_z = grid.axial_spacing / PER_CELL[0]
spacing_r = grid.radial_spacing / PER_CELL[1]
z = np.arange(PLASMA_LOWER + 0.5 * spacing_z, PLASMA_UPPER, spacing_z)
r = np.arange(0.5 * spacing_r, PLASMA_RADIUS, spacing_r)
theta = 2.0 * np.pi / PER_CELL[2] * np.arange(PER_CELL[2])
zz, rr, tt = np.meshgrid(z, r, theta, indexing="ij")
tt = tt + 2.0 * np.pi / PER_CELL[2] * (
    (np.arange(z.size)[:, None, None] * 0.618 + np.arange(r.size)[None, :, None] * 0.414)
    % 1.0
)
positions = np.stack((rr * np.cos(tt), rr * np.sin(tt), zz), axis=-1).reshape(-1, 3)
weights = (rr * 2.0 * np.pi / PER_CELL[2] * spacing_r * spacing_z).reshape(-1)
count = weights.size

transfer = D.pic.AzimuthalTransferPlan(grid)
species, transfers = [], []
for offset, specific, name, mass in (
    (0, -1.0, "electrons", 1.0),
    (10**7, 1.0e-9, "ions", 1.0e9),  # immobile neutralizing ions
):
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + count),
        jnp.asarray(mass * weights),
        ambient_dimension=3,
    ).prepare()
    species.append(
        D.pic.PICSpeciesPlan(
            D.ParticlePopulationPlan(support),
            D.pic.PICChargeModelPlan(
                specific,
                name,
                minimum_charge_number=1,
                maximum_charge_number=1,
                initial_charge_number=1,
            ),
        )
    )
    transfers.append(transfer.prepare(support))

# Antenna: E_x envelope A0 k0 exp(−r²/w₀² − (t − t₀)²/L²) at focus, carrier k0.
samples = np.linspace(-RADIUS - 0.5, RADIUS + 0.5, 97)
times = np.linspace(0.0, 2.0 * PEAK_TIME, 121)
x, y = np.meshgrid(samples, samples, indexing="ij")
envelope = np.zeros((samples.size, samples.size, times.size, 2), complex)
envelope[..., 0] = (
    A0
    * K0
    * np.exp(-(x**2 + y**2) / WAIST**2)[:, :, None]
    * np.exp(-((times - PEAK_TIME) ** 2) / DURATION**2)
)
antenna = spectral.QuasiCylindricalAntennaPlan(
    ANTENNA, samples, samples, times, envelope, carrier_angular_frequency=K0
)

solver = spectral.QuasiCylindricalMaxwellPlan(
    grid, charge_conservation="spectral-correction", antennas=(antenna,)
).prepare(tuple(transfers))
pic = phx.solver.ElectromagneticPICPlan(solver, species=tuple(species))
step = 0.9 * float(solver.stable_step)
steps = int(round(FINAL_TIME / step))
state = pic.initialize((positions, positions), (np.zeros((count, 3)),) * 2, step)
advance = eqx.filter_jit(lambda value: pic.step_detailed(value, step))

start = time.perf_counter()
constraint = 0.0
for _ in range(steps):
    result = advance(state)
    if not bool(result.successful):
        raise RuntimeError("A quasi-cylindrical PIC step was rejected.")
    constraint = max(constraint, float(result.diagnostics.electric_constraint))
    state = result.accepted_state
elapsed = time.perf_counter() - start

on_axis = np.real(np.asarray(state.field.electric[0, 0, :, 2]))
laser = np.abs(np.asarray(state.field.electric[1, 0, :, 0]))
front = grid.axial_coordinates[int(np.argmax(laser))]
wake = (grid.axial_coordinates > PLASMA_LOWER + 0.5) & (grid.axial_coordinates < front)
sigma = DURATION / np.sqrt(2.0)
linear = 0.5 * np.sqrt(np.pi) * (A0**2 / 2.0) * sigma * np.exp(-(sigma**2) / 4.0)
print(f"particles per species       {count}")
print(f"steps                        {steps} (dt = {step:.4f}, {elapsed:.1f} s)")
print(f"laser centroid on axis       z = {front:.2f}")
print(f"peak on-axis wake |E_z|      {np.max(np.abs(on_axis[wake])):.4f}")
print(f"1-D linear wake estimate     {linear:.4f}")
print(f"max Gauss residual           {constraint:.2e}")
