#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Full-wave FEL in a Lorentz-boosted frame: seeded gain and prebunched emission.

A γ = 20, six-period planar undulator (K = 1) in PIC code units drives a
one-dimensional flat-top electron–positron beam (the neutral full-wave
counterpart of a beam without space charge) through a self-consistent PSATD
PIC run in the frame where the beam is on average at rest. The first run seeds
the beam with a lab plane wave and compares the steady small-signal gain with
the period-averaged `FELPlan`; the second prebunches the beam (bunching
``J₁(a)``) without a seed and compares the steady coherent field with the
Kroll–Morton–Rosenbluth value ``κ I L J₁(a)/(ε₀ c A γ)``. Both runs report the
boosted-frame resolution and the lab energy ledger.
"""

import math

import jax
import numpy as np
from scipy import special

from phydrax import ElectromagneticScaleContract
from phydrax.applications.accelerator import fel, InsertionDeviceField
from phydrax.discretization.pic import PIC_CODE_RELATIVITY
from phydrax.units import CHARGE, UnitDefinition


scale = ElectromagneticScaleContract.code_units(
    PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=1,
    boltzmann_constant=1,
    elementary_charge=1,
    electron_mass=1,
    vacuum_permittivity=1,
    constant_set_id="fel-full-wave-example",
)
deflection, period, periods, gamma = 1.0, 1.0, 6, 20.0
# The one-dimensional cross section: area 2πσ² of the averaged model's slice.
sigma = 0.01
area = 2.0 * math.pi * sigma**2
width = math.sqrt(area)
boost = gamma / math.sqrt(1.0 + 0.5 * deflection**2)

device = InsertionDeviceField(
    deflection * 2.0 * math.pi / period,
    period,
    periods,
    polarization="planar",
    center=0.0,
    aperture=(0.3, 0.09),
    ramp_periods=0.125,
)
lattice = fel.FELUndulatorLattice(
    scale, (fel.FELUndulatorSegment(device),), step_length=period / 16.0
)

# Seeded small-signal gain, detuned to the averaged model's gain maximum.
current, amplitude = 4.0e-3, 0.3
wavelength = lattice.resonant_wavelength(gamma) * 1.07
seeded = fel.FELFullWavePlan(
    lattice,
    wavelength,
    boost_lorentz_factor=boost,
    transverse_size=(width, width),
    cells_per_wavelength=24,
    steps_per_period=48,
    seed=fel.FELFullWaveSeed(amplitude),
)
beam = seeded.flat_top_beam(gamma, current, wavelengths=22, taper_wavelengths=3.0)
result = seeded.prepare(beam).run()
if result.seed_amplitude is None or result.steady_gain is None:
    raise RuntimeError("A seeded flat-top run reports its seed amplitude and gain.")
frame = result.evidence.frame
print(
    f"boost γ_b = {frame.boost_lorentz_factor:.3f} (resonant γ_z = "
    f"{frame.resonant_lorentz_factor:.3f}), {frame.cells_per_wavelength:.1f} cells "
    f"per λ′, grid {frame.grid_shape}, {frame.step_count} steps"
)
print(
    f"seeded run: status {int(result.evidence.status)}, seed amplitude "
    f"{float(result.seed_amplitude):.4f} (declared {amplitude})"
)

emittance = 1.0e-6
averaged = fel.FELPlan(
    lattice,
    wavelength,
    loading=fel.FELLoading(1, 16, shot_noise="quiet"),
    seed=fel.FELSeed(0.5 * amplitude**2 * area, waist=1.0),
).solve(
    fel.FELBeamSlices(
        [0.0],
        [current],
        [gamma],
        [0.0],
        [[emittance, emittance]],
        [[gamma * sigma**2 / emittance] * 2],
        [[0.0, 0.0]],
    ),
    jax.random.key(0),
)
power = np.asarray(averaged.power[0, :, 0])
print(
    f"small-signal gain: full wave {float(result.steady_gain):.4f}, "
    f"averaged FELPlan {power[-1] / power[0] - 1.0:.4f}"
)
ledger = result.ledger
print(
    f"lab ledger: beam {float(ledger.beam_energy_change):.3e}, field "
    f"{float(ledger.field_energy_change):.3e}, escaped "
    f"{float(ledger.escaped_energy):.3e}, injected "
    f"{float(ledger.injected_energy):.3e}, relative defect "
    f"{float(ledger.relative_defect):.2e}"
)

# Prebunched coherent emission against KMR: κ = a_w [JJ]/√2, [JJ] = J₀(ξ) − J₁(ξ).
modulation = 0.2
prebunched = fel.FELFullWavePlan(
    lattice,
    lattice.resonant_wavelength(gamma),
    boost_lorentz_factor=boost,
    transverse_size=(width, width),
    nci_energy_fraction=0.1,
)
coherent = prebunched.prepare(
    prebunched.flat_top_beam(
        gamma,
        current,
        wavelengths=22,
        taper_wavelengths=3.0,
        phase_modulation=modulation,
    )
).run()
if coherent.steady_amplitude is None:
    raise RuntimeError("A flat-top run reports its steady forward amplitude.")
xi = deflection**2 / (4.0 + 2.0 * deflection**2)
bessel = special.jv(np.asarray([0.0, 1.0, 1.0]), np.asarray([xi, xi, modulation]))
coupling = (deflection / math.sqrt(2.0)) * (bessel[0] - bessel[1]) / math.sqrt(2.0)
kmr = coupling * current * periods * period * bessel[2] / (area * gamma)
print(
    f"prebunched run: status {int(coherent.evidence.status)}, steady field "
    f"{float(coherent.steady_amplitude):.4e}, KMR {kmr:.4e}"
)
