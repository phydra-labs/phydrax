#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_electrostatic_pic_fixed_step_workflow_retains_constraints():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    transfers = []
    for offset, sign, name in ((0, -1.0, "negative"), (100, 1.0, "positive")):
        particles = phx.discretization.ParticleSetPlan(
            jnp.arange(offset, offset + 4), jnp.ones((4,)), ambient_dimension=1
        ).prepare()
        charged = phx.discretization.ChargedParticlePlan(
            sign * jnp.ones((4,)), name
        ).prepare(particles)
        transfers.append(
            phx.discretization.pic.PICParticleCochainTransferPlan(bridge).prepare(
                charged
            )
        )
    plan = phx.solver.ElectrostaticPICPlan(
        phx.solver.CochainElectrostaticPlan(bridge, boundary="periodic"),
        tuple(transfers),
    )
    position = jnp.asarray([[0.125], [0.375], [0.625], [0.875]])
    state = plan.initialize(
        (position + 0.002, position),
        (jnp.zeros((4, 1)), jnp.zeros((4, 1))),
    )
    method = phx.solver.ElectrostaticPICFixedStepMethod(plan)
    solution = phx.solver.solve_fixed_step(
        phx.solver.FixedStepProblem(
            method,
            state,
            t0=0.0,
            t1=0.002,
            step_size=0.001,
            discretization_bundle=plan.discretization_bundle,
            state_geometry=phx.metrix.EuclideanStateGeometry(),
        )
    )
    assert solution.successful
    assert jax.tree.leaves(solution.states)
