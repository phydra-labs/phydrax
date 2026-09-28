#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.applications import accelerator


scale = phx.ElectromagneticScaleContract.si()
light = float(scale.speed_of_light)
electron_charge = float(scale.elementary_charge)
rest_energy = float(scale.electron_mass) * light**2
classical_radius = float(scale.classical_electron_radius)


def fodo_ring(
    rho: float, cells: int, length_scale: float, gradient: float, voltage: float
) -> accelerator.RingLattice:
    bend = rho * math.pi / cells
    quad = 0.3 * length_scale
    drift = 0.5 * length_scale
    return accelerator.RingLattice(
        [
            accelerator.RingElement(
                "quadrupole", "qf1", length=0.5 * quad, gradient=gradient
            ),
            accelerator.RingElement("drift", "d1", length=drift),
            accelerator.RingElement(
                "sector-bend", "b1", length=bend, curvature=1.0 / rho
            ),
            accelerator.RingElement("drift", "d2", length=drift),
            accelerator.RingElement("quadrupole", "qd", length=quad, gradient=-gradient),
            accelerator.RingElement("drift", "d3", length=drift),
            accelerator.RingElement(
                "sector-bend", "b2", length=bend, curvature=1.0 / rho
            ),
            accelerator.RingElement("drift", "d4", length=drift),
            accelerator.RingElement(
                "quadrupole", "qf2", length=0.5 * quad, gradient=gradient
            ),
            accelerator.RingElement("rf-cavity", "rf", voltage=voltage, harmonic=1),
        ],
        periodicity=cells,
    )


def radiation_plan(
    lattice: accelerator.RingLattice, gamma: float
) -> accelerator.RingRadiationPlan:
    return accelerator.RingRadiationPlan(
        lattice,
        scale,
        reference_rest_energy=rest_energy,
        reference_momentum=math.sqrt(gamma * gamma - 1.0) * rest_energy / light,
        reference_charge=-electron_charge,
    )


# A 3 GeV isomagnetic FODO ring: radiation integrals and equilibrium.
ring = radiation_plan(
    fodo_ring(10.0, 16, 1.0, 1.2, 2.0e5), 3.0e9 * electron_charge / rest_energy
)
integrals = ring.integrals
equilibrium = ring.equilibrium

# A compressed ring (γ = 300, 2 % energy loss per turn) that reaches its
# radiation equilibrium within a few hundred tracked turns.
gamma = 300.0
rho = (4.0 * math.pi / 3.0) * classical_radius * gamma**3 / 0.02
loss = 0.02 * gamma * rest_energy
fast = radiation_plan(
    fodo_ring(
        rho, 6, rho / 10.0, 0.85 / (rho / 10.0) ** 2, 2.0 * loss / electron_charge / 6
    ),
    gamma,
)
tracking = accelerator.RadiativeRingTrackingPlan(
    fast,
    250,
    model="stochastic",
    horizontal_aperture=1.0,
    vertical_aperture=1.0,
    slices_per_bend=1,
)
count = 512
bunch = accelerator.AcceleratorBunch(
    jnp.zeros((count, 6)),
    jnp.ones((count,)),
    jnp.arange(count, dtype=jnp.int32),
    reference_rest_energy=rest_energy,
    reference_momentum=fast.reference_momentum,
    reference_charge=-electron_charge,
    bunch_id="cold-start",
)
evidence = accelerator.track_radiative_ring(
    tracking, bunch, key=jax.random.key(0)
).evidence
tail = slice(-50, None)

print(
    {
        "I1_m": float(integrals.i1),
        "I2_per_m": float(integrals.i2),
        "I4_per_m": float(integrals.i4),
        "I5_per_m": float(integrals.i5),
        "U0_keV": float(equilibrium.energy_loss_per_turn) / electron_charge / 1e3,
        "damping_partition": [float(value) for value in equilibrium.damping_partition],
        "damping_times_ms": [1e3 * float(value) for value in equilibrium.damping_times],
        "emittance_nm": 1e9 * float(equilibrium.horizontal_emittance),
        "energy_spread": float(equilibrium.energy_spread),
        "tracked_status": int(evidence.status),
        "tracked_energy_spread_over_theory": float(
            jnp.mean(evidence.energy_spread[tail]) / fast.equilibrium.energy_spread
        ),
        "tracked_loss_over_U0": float(
            jnp.mean(evidence.radiated_energy[tail])
            / fast.equilibrium.energy_loss_per_turn
        ),
        "tracked_photons_per_turn": float(jnp.mean(evidence.photon_count[tail])),
    }
)
