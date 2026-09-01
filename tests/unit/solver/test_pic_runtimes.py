#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _species(bridge, offset, sign, name, count=4):
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(offset, offset + count),
        jnp.ones((count,)),
        ambient_dimension=bridge.dimension,
    ).prepare()
    charged = phx.discretization.ChargedParticlePlan(
        sign * jnp.ones((count,)), name
    ).prepare(particles)
    transfer = phx.discretization.pic.PICParticleCochainTransferPlan(bridge).prepare(
        charged
    )
    return charged, transfer


def test_electrostatic_pic_step_is_atomic_and_constraint_aware():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    _, negative = _species(bridge, 0, -1.0, "negative")
    _, positive = _species(bridge, 100, 1.0, "positive")
    field = phx.solver.CochainElectrostaticPlan(
        bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
    )
    pic = phx.solver.ElectrostaticPICPlan(field, (negative, positive))
    position = jnp.asarray([[0.15], [0.35], [0.60], [0.85]])
    state = pic.initialize(
        (position + 0.002, position),
        (jnp.zeros((4, 1)), jnp.zeros((4, 1))),
    )
    result = pic.step_detailed(state, 1.0e-3)
    assert result.successful
    assert result.diagnostics.poisson_residual < 1.0e-8
    assert result.diagnostics.charge_balance_defect < 1.0e-12

    rejected = pic.step_detailed(state, 10.0)
    assert not rejected.successful
    np.testing.assert_array_equal(
        rejected.accepted_state.particles[0].position, state.particles[0].position
    )


def test_electromagnetic_pic_preserves_zero_current_constraints():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(3, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    _, negative = _species(bridge, 0, -1.0, "negative", count=2)
    _, positive = _species(bridge, 100, 1.0, "positive", count=2)
    transfers = (negative, positive)
    currents = tuple(
        phx.discretization.pic.ChargeConservingCurrentPlan(value) for value in transfers
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        plan_id="test-pic-maxwell",
    ).prepare()
    electrostatic = phx.solver.CochainElectrostaticPlan(
        bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
    )
    pic = phx.solver.ElectromagneticPICPlan(maxwell, electrostatic, transfers, currents)
    position = jnp.asarray([[0.25, 0.25, 0.25], [0.7, 0.6, 0.5]])
    velocity = jnp.zeros((2, 3))
    dt = 0.01 * maxwell.stable_dt
    state = pic.initialize((position, position), (velocity, velocity), dt)
    result = pic.step_detailed(state, dt)
    assert result.successful
    assert result.diagnostics.continuity_defect < 1.0e-10
    assert result.diagnostics.particle_maxwell_charge_defect < 1.0e-10
    assert result.diagnostics.electric_constraint < 1.0e-10
    assert result.diagnostics.magnetic_constraint < 1.0e-10
