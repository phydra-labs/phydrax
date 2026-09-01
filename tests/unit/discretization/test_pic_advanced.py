#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _population(capacity=4, dimension=3):
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity), jnp.ones((capacity,)), ambient_dimension=dimension
    ).prepare()
    plan = phx.discretization.ParticlePopulationPlan(particles)
    return particles, plan, plan.initialize()


def test_dynamic_charge_requires_compensating_charge():
    _, _, population = _population()
    ion = phx.discretization.pic.PICChargeModelPlan(
        1.0,
        "ions",
        minimum_charge_number=0,
        maximum_charge_number=3,
        initial_charge_number=0,
    )
    state = ion.initialize(population)
    failed = ion.transition(
        population,
        state,
        jnp.asarray([1, 0, 0, 0]),
        1,
    )
    assert not failed.successful
    accepted = ion.transition(
        population,
        state,
        jnp.asarray([1, 0, 0, 0]),
        1,
        compensating_charge=-1.0,
    )
    assert accepted.successful
    assert accepted.accepted_state.charge_number[0] == 1


def test_coulomb_and_background_collisions_report_correct_ledgers():
    _, _, population = _population()
    velocity = jnp.asarray(
        [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]
    )
    coulomb = phx.discretization.pic.collisions.CoulombCollisionPlan(
        1.0, maximum_probability=0.2
    ).collide(
        velocity,
        population.mass,
        population.active,
        population.incarnation,
        jr.key(3),
        0.1,
    )
    assert coulomb.successful
    assert coulomb.momentum_defect < 1e-12
    assert jnp.abs(coulomb.energy_defect) < 1e-12

    background = phx.discretization.pic.collisions.BackgroundMCCPlan(
        1.0, maximum_probability=0.2
    ).collide(
        velocity,
        population.mass,
        population.active,
        jr.key(4),
        0.1,
    )
    assert background.successful
    np.testing.assert_allclose(
        jnp.sum(population.mass[:, None] * background.accepted_velocity, axis=0)
        + background.background_momentum_source,
        jnp.sum(population.mass[:, None] * velocity, axis=0),
    )


def test_reduced_maxwell_and_current_projection_preserve_constraints():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    field_plan = phx.solver.CompatibleMaxwell1DPlan(grid)
    field = field_plan.initialize()
    current = (
        jnp.zeros((16,)),
        jnp.sin(2.0 * jnp.pi * jnp.arange(16) / 16) * 1e-3,
        jnp.zeros((16,)),
    )
    state, diagnostics = field_plan.step(field, current, 0.1 * field_plan.stable_dt)
    assert diagnostics.successful
    assert diagnostics.electric_constraint_linf < 1e-12
    assert jnp.all(jnp.isfinite(state.electric[1]))

    transfer = phx.discretization.pic.ReducedPICTransferPlan(grid)
    start = jnp.asarray([[0.2], [0.7]])
    end = start + jnp.asarray([[0.01], [-0.01]])
    result = transfer.current(
        start,
        end,
        jnp.asarray([-1.0, 1.0]),
        jnp.asarray([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        jnp.asarray([True, True]),
        0.02,
    )
    assert result.successful
    assert result.maximum_continuity_defect < 1e-9


def test_simplicial_locator_and_whitney_current_are_conservative():
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (phx.discretization.CellBlock("tri", "triangle", jnp.asarray(((0, 1, 2),))),),
    )
    locator = phx.discretization.PreparedSimplicialCellLocator(mesh)
    located = locator.locate(jnp.asarray(((0.2, 0.2), (0.6, 0.2))))
    assert located.successful.all()
    np.testing.assert_allclose(jnp.sum(located.barycentric, axis=1), 1.0)

    tetra_mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        (phx.discretization.CellBlock("tet", "tetrahedron", jnp.asarray(((0, 1, 2, 3),))),),
    )
    tetra_locator = phx.discretization.PreparedSimplicialCellLocator(tetra_mesh)
    current = phx.discretization.pic.UnstructuredWhitneyCurrentPlan(
        tetra_locator, maximum_segments=2
    ).deposit(
        jnp.asarray([[0.1, 0.1, 0.1]]),
        jnp.asarray([[0.2, 0.1, 0.1]]),
        jnp.asarray([1.0]),
        jnp.asarray([True]),
        0.1,
    )
    assert current.successful
    assert current.maximum_continuity_defect < 1e-9
