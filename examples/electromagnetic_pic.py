#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


grid = phx.discretization.TensorGridPlan(
    tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
bridge = phx.discretization.StructuredCochainBridge(grid)

particles = []
charged = []
for offset, charge, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(offset, offset + 4), jnp.ones((4,)), ambient_dimension=3
    ).prepare()
    particles.append(support)
    charged.append(
        phx.discretization.ChargedParticlePlan(
            charge * jnp.ones((4,)), name
        ).prepare(support)
    )

transfer_plan = phx.discretization.pic.PICParticleCochainTransferPlan(bridge)
transfers = tuple(transfer_plan.prepare(value) for value in charged)
currents = tuple(
    phx.discretization.pic.ChargeConservingCurrentPlan(value) for value in transfers
)
current_source = phx.solver.PICMaxwellCurrentSourcePlan()
maxwell = phx.solver.CompatibleMaxwellPlan(
    bridge, sources=(current_source,), plan_id="periodic-pic-maxwell"
).prepare()
electrostatic = phx.solver.CochainElectrostaticPlan(bridge, boundary="periodic")
pic = phx.solver.ElectromagneticPICPlan(
    maxwell, electrostatic, transfers, currents
)

position = jnp.asarray(
    [[0.20, 0.20, 0.20], [0.35, 0.45, 0.55], [0.60, 0.30, 0.70], [0.80, 0.75, 0.40]]
)
velocity = jnp.zeros((4, 3))
dt = 0.02 * maxwell.stable_dt
state = pic.initialize((position, position), (velocity, velocity), dt)
result = pic.step_detailed(state, dt)

print(
    {
        "successful": bool(result.successful),
        "continuity_defect": float(result.diagnostics.continuity_defect),
        "gauss_defect": float(result.diagnostics.electric_constraint),
        "magnetic_defect": float(result.diagnostics.magnetic_constraint),
    }
)
