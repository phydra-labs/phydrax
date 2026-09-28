#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Seeded soft-X-ray FEL amplifier with the time-independent averaged model.

A 5.1 GeV, 3 kA beam drives a 36 m planar undulator (λ_u = 3 cm, K = 3.5)
resonant at 1.07 nm. The one-dimensional run reproduces the cold high-gain
growth exp(√3 ρ k_u z) and saturates near ρ P_beam; a helical run with a
matched smooth-focused round beam uses the optics angular-spectrum field and
compares its fitted gain length with Ming Xie's formula. Every run reports
its particle↔field energy ledger.
"""

import math

import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.applications.accelerator import (
    fel,
    InsertionDeviceField,
    InsertionDevicePolarization,
)
from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import AngularSpectrumPlan, PlaneFieldSpace


scale = phx.ElectromagneticScaleContract.si()
light = float(scale.speed_of_light)
charge = float(scale.elementary_charge)
mass = float(scale.electron_mass)
period, gamma, current, beam_size = 0.03, 1.0e4, 3000.0, 30.0e-6


def peak_field(deflection: float) -> float:
    return deflection * 2.0 * math.pi * mass * light / (charge * period)


def undulator(
    deflection: float, polarization: InsertionDevicePolarization, **segment: float
) -> fel.FELUndulatorLattice:
    device = InsertionDeviceField(
        peak_field(deflection),
        period,
        1200,
        polarization=polarization,
        center=0.0,
        aperture=(1.0e-2, 1.0e-2),
    )
    return fel.FELUndulatorLattice(
        scale, (fel.FELUndulatorSegment(device, **segment),), step_length=0.05
    )


def summarize(label: str, result: fel.FELResult) -> None:
    scaling = result.gain.scaling
    rho = float(scaling.pierce_parameter[0])
    print(f"{label}:")
    print(f"  Pierce parameter ρ          {rho:.3e}")
    print(
        f"  gain length fit / 1-D       {float(result.gain.fitted_gain_length[0]):.3f} m"
        f" / {float(scaling.one_dimensional_gain_length[0]):.3f} m"
    )
    print(f"  Ming Xie gain length        {float(scaling.ming_xie_gain_length[0]):.3f} m")
    print(
        f"  saturation {float(result.gain.saturation_power[0]) / 1e9:.2f} GW at "
        f"{float(result.gain.saturation_position[0]):.1f} m "
        f"({float(result.gain.saturation_power[0]) / (rho * float(scaling.beam_power[0])):.2f} ρP_beam)"
    )
    print(f"  ledger relative defect      {float(result.ledger.relative_defect[0]):.1e}")
    print(
        f"  status                      {fel.FELStatus(int(result.evidence.status[0]))!r}"
    )


# One-dimensional cold beam: tiny emittance with large β keeps the 2πσ² area.
planar = undulator(3.5, "planar")
wavelength = planar.resonant_wavelength(gamma)
cold = fel.FELBeamSlices(
    [0.0],
    [current],
    [gamma],
    [0.0],
    [[1.0e-9, 1.0e-9]],
    [[beam_size**2 * gamma / 1.0e-9] * 2],
    [[0.0, 0.0]],
)
one_dimensional = fel.FELPlan(
    planar,
    wavelength,
    loading=fel.FELLoading(64, 6, shot_noise="quiet"),
    harmonics=(1, 3),
    seed=fel.FELSeed(1.0e3, waist=beam_size),
)
result = one_dimensional.solve(cold, jax.random.key(0))
summarize("one-dimensional, harmonics (1, 3)", result)
peak = jnp.argmax(result.power[0, :, 0])
print(
    "  third-harmonic / fundamental at saturation "
    f"{float(result.power[0, peak, 1] / result.power[0, peak, 0]):.2e}"
)

# Helical undulator with a matched round beam in smooth focusing (β = 9 m).
helical_deflection = 3.5 / math.sqrt(2.0)
beta = 9.0
natural = (helical_deflection * 2.0 * math.pi / period) ** 2 / (2.0 * gamma**2)
gradient = (1.0 / beta**2 - natural) * gamma * mass * light / charge
helical = undulator(helical_deflection, "helical", smooth_focusing_gradient=gradient)
emittance = beam_size**2 / beta * gamma
beam = fel.FELBeamSlices(
    [0.0], [current], [gamma], [1.0e-4], [[emittance] * 2], [[beta] * 2], [[0.0] * 2]
)
grid = TensorGridPlan(
    (UniformAxisSpec(48), UniformAxisSpec(48)), axis_names=("x", "y")
).prepare(jnp.asarray([[-240.0e-6, -240.0e-6], [240.0e-6, 240.0e-6]]))
three_dimensional = fel.FELPlan(
    helical,
    helical.resonant_wavelength(gamma),
    loading=fel.FELLoading(1024, 4, shot_noise="quiet"),
    transverse="angular-spectrum",
    field_space=PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window"),
    propagation=AngularSpectrumPlan(12),
    seed=fel.FELSeed(1.0e3, waist=40.0e-6),
)
summarize("angular-spectrum, helical", three_dimensional.solve(beam, jax.random.key(0)))
