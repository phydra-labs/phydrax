#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


grid = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(32, periodic=True),),
    axis_names=("x",),
).prepare(jnp.asarray([[0.0], [1.0]]))
bridge = phx.discretization.StructuredCochainBridge(grid)

negative_particles = phx.discretization.ParticleSetPlan(
    jnp.arange(8), jnp.ones((8,)), ambient_dimension=1
).prepare()
positive_particles = phx.discretization.ParticleSetPlan(
    jnp.arange(100, 108), jnp.ones((8,)), ambient_dimension=1
).prepare()
negative = phx.discretization.ChargedParticlePlan(-jnp.ones((8,)), "electrons").prepare(
    negative_particles
)
positive = phx.discretization.ChargedParticlePlan(jnp.ones((8,)), "ions").prepare(
    positive_particles
)

transfer_plan = phx.discretization.pic.PICParticleCochainTransferPlan(bridge)
transfers = (transfer_plan.prepare(negative), transfer_plan.prepare(positive))
field = phx.solver.CochainElectrostaticPlan(
    bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
)
pic = phx.solver.ElectrostaticPICPlan(field, transfers)

base = (jnp.arange(8, dtype=float)[:, None] + 0.5) / 8.0
state = pic.initialize(
    (base + 0.002 * jnp.sin(2.0 * jnp.pi * base), base),
    (jnp.zeros((8, 1)), jnp.zeros((8, 1))),
)
result = pic.step_detailed(state, 1.0e-3)

print(
    {
        "successful": bool(result.successful),
        "charge_balance_defect": float(result.diagnostics.charge_balance_defect),
        "poisson_residual": float(result.diagnostics.poisson_residual),
        "energy_defect": float(result.diagnostics.energy.defect),
    }
)
