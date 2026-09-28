#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_reduced_pic_and_ghost_fluid_workflows_share_fixed_shape_contracts() -> None:
    grid_1d = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=1
    ).prepare()
    field_plan = phx.solver.CompatibleMaxwell1DPlan(grid_1d)
    reduced = phx.solver.ElectromagneticPICPlan(
        phx.solver.ReducedMaxwellPICFieldSolver(
            field_plan, phx.discretization.pic.ReducedPICTransferPlan(grid_1d)
        ),
        species=tuple(
            phx.discretization.pic.PICSpeciesPlan(
                phx.discretization.ParticlePopulationPlan(support),
                phx.discretization.pic.PICChargeModelPlan(
                    sign,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
            for sign, name in ((-1.0, "electrons"), (1.0, "ions"))
        ),
    )
    # A periodic 1-D domain admits Gauss-consistent fields only when neutral.
    position = jnp.asarray([[0.2], [0.7]])
    state = reduced.initialize(
        (position + 0.01, position), (jnp.zeros((2, 3)), jnp.zeros((2, 3))), 1e-3
    )
    pic_result = reduced.step_detailed(state, 1e-3)
    assert pic_result.successful
    assert pic_result.accepted_state.species[0].particles.position.shape == (2, 1)

    grid_2d = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid_2d).prepare()
    mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    geometry = phx.discretization.flip.ParticleLevelSetPlan(grid_2d, 0.2).evaluate(
        jnp.asarray([[0.4, 0.4], [0.55, 0.4], [0.4, 0.55], [0.55, 0.55]]),
        jnp.ones((4,), dtype="bool"),
    )
    projection = phx.solver.MACGhostFluidProjectionPlan(
        phx.solver.MACFreeSurfaceProjectionPlan(
            mac,
            boundaries=phx.discretization.MACBoundaryPlan(mac).prepare(),
            tolerance=1e-7,
        )
    )
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    fluid_result = projection.project(velocity, geometry, 1e-3)
    assert fluid_result.successful
    assert fluid_result.pressure.shape == finite_volume.cell_shape
