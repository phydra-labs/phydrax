#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-dependent averaged FEL: SASE in an open window and an HGHG radiator.

A 3 kA, γ = 10⁴ beam in a helical undulator starts from shot noise (SASE); the
radiation slips ahead through head padding, and the run reports the exit
pulse energy, spike count, coherence time, spectral bandwidth, padding
evidence, and energy ledger. A second run prebunches the beam at the fourth
harmonic of a seed laser with a modulator and a chicane (HGHG) and compares
the entrance bunching with the Bessel formula.
"""

import math

import equinox as eqx
import jax
import numpy as np
from scipy import special

import phydrax as phx
from phydrax.applications.accelerator import (
    AcceleratorConvention,
    fel,
    InsertionDeviceField,
    SymplecticMapPlan,
)


scale = phx.ElectromagneticScaleContract.si()
light = float(scale.speed_of_light)
charge = float(scale.elementary_charge)
mass = float(scale.electron_mass)
period, gamma, current = 0.03, 1.0e4, 3000.0
deflection = 3.5 / math.sqrt(2.0)
beta = math.sqrt(2.0) * gamma / (deflection * 2.0 * math.pi / period)


def lattice(periods: int) -> fel.FELUndulatorLattice:
    device = InsertionDeviceField(
        deflection * 2.0 * math.pi * mass * light / (charge * period),
        period,
        periods,
        polarization="helical",
        center=0.0,
        aperture=(1e-2, 1e-2),
    )
    return fel.FELUndulatorLattice(
        scale, (fel.FELUndulatorSegment(device),), step_length=period
    )


def beam(count: int, spacing: float, spread: float) -> fel.FELBeamSlices:
    return fel.FELBeamSlices(
        np.arange(count) * spacing,
        np.full(count, current),
        np.full(count, gamma),
        np.full(count, spread),
        np.full((count, 2), 1.0e-7),
        np.full((count, 2), beta),
        np.zeros((count, 2)),
    )


# SASE: 256 slices one wavelength apart behind head padding covering the
# 600-wavelength slippage (one slot per one-period step).
undulator = lattice(600)
wavelength = undulator.resonant_wavelength(gamma)
sase = fel.FELTimeDependentPlan(
    fel.FELPlan(
        undulator,
        wavelength,
        loading=fel.FELLoading(4, 4, shot_noise="fawley"),
        slice_batch=512,
    ),
    slippage="commensurate",
    boundary="open",
    head_padding=640,
)
result = eqx.filter_jit(sase.solve)(beam(256, wavelength, 1.0e-4), jax.random.key(0))
spectrum = result.spectrum
scaling = result.scaling
print("SASE, open window with head padding")
print(f"  Pierce parameter            {float(scaling.pierce_parameter[0]):.3e}")
print(f"  exit pulse energy           {float(result.pulse_energy[-1, 0]):.3e} J")
print(
    f"  peak power / (ρ P_beam)     "
    f"{float(result.power[-1, :, 0].max() / (scaling.pierce_parameter[0] * scaling.beam_power[0])):.2f}"
)
print(f"  spikes                      {int(spectrum.spike_count[0])}")
print(f"  coherence time              {float(spectrum.coherence_time[0]):.3e} s")
print(
    f"  rms bandwidth / ω           "
    f"{float(spectrum.rms_bandwidth[0] / result.angular_frequencies[0]):.3e}"
)
print(f"  padding covers slippage     {bool(result.evidence.padding_sufficient)}")
print(f"  exit energy fraction        {float(result.evidence.exit_fraction):.2e}")
print(f"  ledger relative defect      {float(result.ledger.relative_defect):.2e}")
print(f"  status                      {fel.FELStatus(int(result.evidence.status))!r}")

# HGHG: modulator A = 4 σ_γ and a chicane with B = k_b R₅₆ σ_δ = 0.3 at h = 4.
harmonic, amplitude, strength, spread = 4, 4.0, 0.3, 1.0e-4
chicane = np.eye(6)
chicane[4, 5] = -strength / (2.0 * math.pi / (harmonic * wavelength) * spread)
hghg = fel.FELTimeDependentPlan(
    fel.FELPlan(
        lattice(2),
        wavelength,
        loading=fel.FELLoading(4096, 4 * harmonic, shot_noise="quiet"),
        slice_batch=8,
    ),
    slippage="commensurate",
    boundary="periodic",
    prebunching=fel.FELPrebunching(
        (
            fel.FELModulator(amplitude * spread * gamma),
            SymplecticMapPlan(
                chicane, np.zeros(6), AcceleratorConvention(), element_id="chicane"
            ),
        ),
        harmonic=harmonic,
        reference_lorentz_factor=gamma,
    ),
)
bunched = eqx.filter_jit(hghg.solve)(beam(8, wavelength, spread), jax.random.key(1))
expected = float(
    np.abs(special.jv(np.float64(harmonic), np.float64(harmonic * amplitude * strength)))
) * math.exp(-0.5 * (harmonic * strength) ** 2)
print("HGHG prebunching at the fourth harmonic")
print(
    f"  entrance bunching           {abs(complex(bunched.bunching[0, :, 0].mean())):.4f}"
)
print(f"  |J_h(hAB)| e^(-h²B²/2)      {expected:.4f}")
