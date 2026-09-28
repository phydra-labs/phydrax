#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""On-axis undulator radiation from a lab-time tracked electron.

A 100-γ electron crosses a ten-period planar undulator with K = 1 and matched
terminations. The tracked lanes feed the vacuum trajectory-radiation spectrum,
whose on-axis lines sit at the odd harmonics of 2γ²ω_u/(1 + K²/2) with
relative width ≈ 0.886/(nN).
"""

import math

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import accelerator
from phydrax.electromagnetics import RadiationObserverPlan, TrajectoryRadiationPlan


scale = phx.ElectromagneticScaleContract.si()
light = float(scale.speed_of_light)
charge = float(scale.elementary_charge)
mass = float(scale.electron_mass)
gamma, deflection, period, period_count = 100.0, 1.0, 0.02, 10
peak_field = deflection * 2.0 * math.pi * mass * light / (charge * period)

undulator = accelerator.InsertionDeviceField(
    peak_field,
    period,
    period_count,
    polarization="planar",
    center=0.0,
    aperture=(1.0e-3, 1.0e-3),
    ramp_periods=0.25,
)
beamline = accelerator.FieldMapBeamline(scale, (undulator,), element_ids=("undulator",))
half_length = 0.5 * period_count * period + 8.0 * undulator.ramp_width
time_step = period / (128.0 * light)
plan = accelerator.FieldMapTrackingPlan(
    beamline,
    entrance_plane=-half_length,
    exit_plane=half_length,
    time_step=time_step,
    step_count=math.ceil(2.02 * half_length / (light * time_step)) + 4,
)
bunch = accelerator.AcceleratorBunch(
    jnp.zeros((1, 6)),
    jnp.ones((1,)),
    jnp.zeros((1,), dtype=jnp.int32),
    reference_rest_energy=mass * light**2,
    reference_momentum=math.sqrt(gamma**2 - 1.0) * mass * light,
    reference_charge=-charge,
    bunch_id="electron",
)
tracked = accelerator.track_field_map(plan, bunch)

resonance = undulator.resonance(scale, gamma)
fundamental = resonance.fundamental_angular_frequency
frequencies = fundamental * np.linspace(0.5, 3.5, 3001)
observers = RadiationObserverPlan(np.array([[0.0, 0.0, 1.0]]), np.array([1.0, 0.0, 0.0]))
spectrum = (
    TrajectoryRadiationPlan(
        scale, observers, frequencies, coherence="coherent", route="segment-exact"
    )
    .prepare()
    .evaluate(tracked.trajectory)
)
energy = np.asarray(spectrum.spectral_energy[:, 0])


def line(harmonic: int) -> tuple[float, float]:
    window = np.abs(frequencies / fundamental - harmonic) < 0.5
    peak = int(np.argmax(np.where(window, energy, 0.0)))
    above = np.flatnonzero(window & (energy >= 0.5 * energy[peak]))
    width = (frequencies[above[-1]] - frequencies[above[0]]) / (harmonic * fundamental)
    return float(frequencies[peak] / (harmonic * fundamental)), float(
        harmonic * period_count * width
    )


print(
    {
        "deflection_parameter": resonance.deflection_parameter,
        "fundamental_angular_frequency": fundamental,
        "tracking_accepted": bool(tracked.evidence.accepted),
        "exit_offset_m": float(tracked.bunch.coordinates[0, 0]),
        "exit_angle": float(tracked.bunch.coordinates[0, 1]),
        "exit_zeta_m": float(tracked.bunch.coordinates[0, 4]),
        "radiation_resolved": bool(spectrum.evidence.resolved),
        "first_peak_over_prediction": line(1)[0],
        "third_peak_over_prediction": line(3)[0],
        "first_width_times_nN": line(1)[1],
        "third_width_times_nN": line(3)[1],
        "second_over_third_harmonic": float(
            energy[np.argmin(np.abs(frequencies - 2.0 * fundamental))]
            / energy[np.argmin(np.abs(frequencies - 3.0 * fundamental))]
        ),
    }
)
