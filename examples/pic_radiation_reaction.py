#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from fractions import Fraction

import jax.numpy as jnp

import phydrax as phx
from phydrax.discretization import pic
from phydrax.units import CHARGE, UnitDefinition


# Code units over the PIC scale: c = ε₀ = 1; one species with q/m = −1.
charge, mass = -0.1, 0.1
scale = phx.ElectromagneticScaleContract.code_units(
    pic.PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=Fraction(1, 10**8),
    boltzmann_constant=1,
    elementary_charge=1,
    electron_mass=1,
    vacuum_permittivity=1,
    constant_set_id="radiation-reaction-example",
)
reaction = pic.RadiationReactionPlan(
    "landau-lifshitz-reduced", scale, charge, mass, maximum_chi=1.0e-2, minimum_gamma=1.0
)


def species(sign: float, name: str, offset: int) -> pic.PICSpeciesPlan:
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(offset, offset + 2), jnp.ones((2,)), ambient_dimension=1
    ).prepare()
    return pic.PICSpeciesPlan(
        phx.discretization.ParticlePopulationPlan(support),
        pic.PICChargeModelPlan(
            sign,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )


grid = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(16, periodic=True),), axis_names=("x",)
).prepare(jnp.asarray([[0.0], [1.0]]))
run = phx.solver.ElectromagneticPICPlan(
    phx.solver.ReducedMaxwellPICFieldSolver(
        phx.solver.CompatibleMaxwell1DPlan(grid), pic.ReducedPICTransferPlan(grid)
    ),
    species=(species(-1.0, "electrons", 0), species(1.0, "ions", 10)),
    processes=(pic.RadiationReactionProcess(reaction, 0),),
    ownership="subgrid-reaction",
)

# Electrons at γ₀ = 50 gyrate in a uniform B_z = 1 (ω_B = 1); light macroparticles.
gamma0, dt, steps = 50.0, 0.01, 100
position = jnp.asarray([[0.2], [0.7]])
weight = jnp.full((2,), 1.0e-5 * mass)
zero = jnp.zeros((16,))
state = run.initialize(
    (position, position),
    (jnp.zeros((2, 3)).at[:, 0].set(math.sqrt(1.0 - 1.0 / gamma0**2)), jnp.zeros((2, 3))),
    dt,
    masses=(weight, weight),
    magnetic=(zero, zero, zero + 1.0),
)
radiated = 0.0
for _ in range(steps):
    result = run.step_detailed(state, dt)
    if not bool(result.successful):
        raise RuntimeError(f"step rejected: {int(result.diagnostics.rejection_reason)}")
    radiated += float(result.diagnostics.energy.radiated)
    state = result.accepted_state

proper = state.species[0].particles.proper_velocity
gamma = float(jnp.sqrt(1.0 + jnp.sum(proper[0] ** 2)))
kinetic = float(jnp.sum(weight * (gamma - 1.0)))
tau = charge**2 / (6.0 * math.pi * mass)
exact = 1.0 / math.tanh(tau * steps * dt + math.atanh(1.0 / gamma0))
(ledger,) = result.diagnostics.processes
separation = float(ledger.radiation.minimum_scale_separation) if ledger.radiation else 0.0
print(f"gamma after {steps} steps: {gamma:.4f} (planar LL cooling {exact:.4f})")
print(
    f"radiated {radiated:.6e} vs kinetic loss "
    f"{float(jnp.sum(weight)) * (gamma0 - 1.0) - kinetic:.6e}"
)
print(f"critical frequency / grid cutoff: {separation:.1f}")
