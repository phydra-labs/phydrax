#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Galilean PSATD suppresses the numerical Cherenkov instability of drifting plasma.

A neutral cold plasma drifts along x at Lorentz factor 10 in a periodic box.
With the standard PSATD the high-|k| field energy grows at the Godfrey–Vay
linear rate; in Galilean coordinates comoving with the plasma it stays at the
noise level.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx


spectral = phx.solver.maxwell.spectral
D = phx.discretization

GAMMA = 10.0
SPACING = 0.3868
COUNTS = (24, 3, 12)
STEP = 0.45 * SPACING
STEPS = 70
SPEED = float(np.sqrt(1.0 - 1.0 / GAMMA**2))
PLASMA_FREQUENCY_SQUARED = 4.0 * GAMMA  # ω_p²/γ = 4 in units c = 1

grid = D.TensorGridPlan(
    tuple(D.UniformCellAxisSpec(n, periodic=True) for n in COUNTS),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * SPACING for n in COUNTS]]))
bridge = D.StructuredCochainBridge(grid)

ix, iy, iz = np.meshgrid(*(np.arange(n) for n in COUNTS), indexing="ij")
ions = np.stack(
    ((ix + 0.5) * SPACING, (iy + 0.5) * SPACING, (iz + 0.5) * SPACING), axis=-1
).reshape(-1, 3)
electrons = ions.copy()
electrons[:, (0, 2)] += np.random.default_rng(1).uniform(
    -0.05 * SPACING, 0.05 * SPACING, (ions.shape[0], 2)
)
count = ions.shape[0]
weight = PLASMA_FREQUENCY_SQUARED * float(np.prod(COUNTS)) * SPACING**3 / count

species, charged = [], []
for offset, specific, name, mass in (
    (0, -1.0, "electrons", weight),
    (10**6, 1.0 / 1836.0, "ions", 1836.0 * weight),
):
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + count), mass * jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    charged.append(
        D.ChargedParticlePlan(specific * mass * jnp.ones((count,)), name).prepare(support)
    )
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
transfer_plan = D.pic.PICParticleCochainTransferPlan(bridge, shape_order=2)
transfers = tuple(transfer_plan.prepare(value) for value in charged)
currents = tuple(D.pic.ChargeConservingCurrentPlan(value) for value in transfers)
velocity = np.zeros(ions.shape)
velocity[:, 0] = SPEED
times = STEP * (np.arange(STEPS) + 1.0)


def high_k_energy(options: dict[str, Any]) -> tuple[Any, np.ndarray]:
    solver = spectral.SpectralMaxwellPlan(bridge, **options).prepare(transfers, currents)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    monitor = spectral.SpectralNCIMonitorPlan(solver, high_fraction=0.3)
    state = eqx.filter_jit(
        lambda: pic.initialize((electrons, ions), (velocity, velocity), STEP)
    )()

    def body(value: Any, _: None) -> tuple[Any, tuple[Array, Array]]:
        result = pic.step_detailed(value, STEP)
        sample = monitor.sample(result.accepted_state.field)
        return result.accepted_state, (sample.high_energy, result.successful)

    _, (energy, successful) = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=STEPS)
    )(state)
    if not bool(jnp.all(successful)):
        raise RuntimeError("A PIC step was rejected.")
    return solver, np.asarray(energy)


standard_solver, standard = high_k_energy({})
_, galilean = high_k_energy(
    {
        "variant": "galilean",
        "galilean_velocity": (SPEED, 0.0, 0.0),
        "charge_conservation": "update-with-rho",
    }
)
reference = spectral.godfrey_vay_growth_rate(
    standard_solver,
    drift_axis=0,
    transverse_axis=2,
    drift_speed=SPEED,
    plasma_frequency=float(np.sqrt(PLASMA_FREQUENCY_SQUARED)),
    step_size=STEP,
)
fit = spectral.SpectralNCIMonitorPlan.fit
print(
    {
        "godfrey_vay_growth_rate": float(reference.maximum_growth_rate),
        "standard_growth_rate": float(fit(times, standard, start=4.5, stop=11.0).rate),
        "galilean_growth_rate": float(fit(times, galilean, start=4.5, stop=11.0).rate),
        "standard_energy_gain": float(standard[-1] / standard[0]),
        "galilean_energy_gain": float(galilean[-1] / galilean[0]),
    }
)
