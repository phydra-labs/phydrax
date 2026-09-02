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


def test_reduced_cic_nonperiodic_boundaries_conserve_macrocharge():
    cases = (
        (
            (phx.discretization.UniformCellAxisSpec(8, periodic=False),),
            jnp.asarray(((0.0,), (0.01,), (0.99,), (1.0,))),
            jnp.asarray((1.0, -0.25, 2.0, 0.5)),
        ),
        (
            (
                phx.discretization.UniformCellAxisSpec(8, periodic=False),
                phx.discretization.UniformCellAxisSpec(8, periodic=False),
            ),
            jnp.asarray(((0.0, 0.0), (0.01, 0.4), (1.0, 0.25), (0.7, 0.99), (1.0, 1.0))),
            jnp.asarray((1.0, -0.25, 2.0, 0.5, 0.75)),
        ),
        (
            (
                phx.discretization.UniformCellAxisSpec(8, periodic=False),
                phx.discretization.UniformCellAxisSpec(8, periodic=True),
            ),
            jnp.asarray(((0.0, 0.0), (0.01, 0.4), (0.99, 0.8), (1.0, 1.0))),
            jnp.asarray((1.0, -0.25, 2.0, 0.5)),
        ),
    )
    for axes, positions, macrocharge in cases:
        dimension = len(axes)
        grid = phx.discretization.TensorGridPlan(
            axes, axis_names=("x", "y")[:dimension]
        ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))
        transfer = phx.discretization.pic.ReducedPICTransferPlan(grid)
        deposited = transfer.deposit(
            positions, macrocharge, jnp.ones_like(macrocharge, dtype=bool)
        )

        np.testing.assert_allclose(
            jnp.sum(deposited) * transfer.cell_volume,
            jnp.sum(macrocharge),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_simplicial_locator_and_whitney_current_are_conservative():
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (phx.discretization.CellBlock("tri", "triangle", jnp.asarray(((0, 1, 2),))),),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    locator = phx.discretization.PreparedSimplicialCellLocator(
        phx.discretization.fem.prepare_finite_element_cell_map(discretization, 0),
        discretization.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(1, 8, 3),
    )
    located = locator.locate(jnp.asarray(((0.2, 0.2), (0.6, 0.2))))
    assert located.successful.all()
    np.testing.assert_allclose(jnp.sum(located.barycentric, axis=1), 1.0)

    tetra_mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "tet", "tetrahedron", jnp.asarray(((0, 1, 2, 3),))
            ),
        ),
    )
    tetra_discretization = phx.discretization.FiniteElementPlan(
        tetra_mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("tetrahedron", 1)
        ),
    ).prepare()
    tetra_locator = phx.discretization.PreparedSimplicialCellLocator(
        phx.discretization.fem.prepare_finite_element_cell_map(tetra_discretization, 0),
        tetra_discretization.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(1, 8, 4),
    )
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


def test_simplicial_locator_and_dependent_ids_track_deformed_coordinates():
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (phx.discretization.CellBlock("tri", "triangle", jnp.asarray(((0, 1, 2),))),),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    cell_map = phx.discretization.fem.prepare_finite_element_cell_map(discretization, 0)
    policy = phx.discretization.SimplicialLocationPolicy(1, 8, 3)
    coordinates = discretization.default_runtime.coordinates
    deformed_coordinates = coordinates.at[1, 0].set(1.1).at[2, 1].set(0.9)
    locator = phx.discretization.PreparedSimplicialCellLocator(
        cell_map, coordinates, policy
    )
    deformed_locator = phx.discretization.PreparedSimplicialCellLocator(
        cell_map, deformed_coordinates, policy
    )

    assert locator.locator_id != deformed_locator.locator_id
    current_plan = phx.discretization.pic.UnstructuredWhitneyCurrentPlan(locator)
    deformed_current_plan = phx.discretization.pic.UnstructuredWhitneyCurrentPlan(
        deformed_locator
    )
    assert current_plan.plan_id != deformed_current_plan.plan_id

    charge_model = phx.discretization.pic.PICChargeModelPlan(
        1.0,
        "ions",
        minimum_charge_number=0,
        maximum_charge_number=1,
        initial_charge_number=1,
    )
    boundary = jnp.ones((3,), dtype=bool)
    electrostatic_plan = phx.discretization.pic.UnstructuredElectrostaticPICPlan(
        locator, charge_model, boundary
    )
    deformed_electrostatic_plan = phx.discretization.pic.UnstructuredElectrostaticPICPlan(
        deformed_locator, charge_model, boundary
    )
    assert electrostatic_plan.plan_id != deformed_electrostatic_plan.plan_id


def test_reduced_maxwell_cpml_memory_resets_and_rolls_back_atomically():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(32, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    impedance = phx.solver.maxwell.MaxwellBoundaryPlan("impedance", admittance=1.0)
    plan = phx.solver.CompatibleMaxwell1DPlan(
        grid,
        boundaries=((impedance, impedance),),
        pml=phx.solver.maxwell.MaxwellCPMLPlan(5),
    )
    x = jnp.arange(plan.count)
    pulse = jnp.exp(-(((x - 3.0) / 1.25) ** 2))
    zero = jnp.zeros_like(pulse)
    initial = plan.initialize(electric=(zero, pulse, zero))
    advanced, diagnostics = plan.step(
        initial,
        (zero, zero, zero),
        0.5 * plan.stable_dt,
    )

    assert diagnostics.successful
    assert advanced.pml_memory is not None
    memories = advanced.pml_memory.electric_memory + advanced.pml_memory.magnetic_memory
    assert any(bool(jnp.any(jnp.abs(value) > 0.0)) for value in memories)

    checkpoint = advanced
    rolled_back, rejected = plan.step(
        checkpoint,
        (zero, zero, zero),
        2.0 * plan.stable_dt,
    )
    assert not rejected.successful
    for restored, saved in zip(
        rolled_back.pml_memory.electric_memory + rolled_back.pml_memory.magnetic_memory,
        checkpoint.pml_memory.electric_memory + checkpoint.pml_memory.magnetic_memory,
        strict=True,
    ):
        np.testing.assert_array_equal(restored, saved)
    for restored, saved in zip(
        rolled_back.electric + rolled_back.magnetic,
        checkpoint.electric + checkpoint.magnetic,
        strict=True,
    ):
        np.testing.assert_array_equal(restored, saved)

    reset = plan.reset_pml(checkpoint)
    for value in reset.pml_memory.electric_memory + reset.pml_memory.magnetic_memory:
        np.testing.assert_array_equal(value, jnp.zeros_like(value))
    for reset_value, saved in zip(
        reset.electric + reset.magnetic,
        checkpoint.electric + checkpoint.magnetic,
        strict=True,
    ):
        np.testing.assert_array_equal(reset_value, saved)


def test_curved_quadratic_simplices_round_trip_through_fe_cell_map():
    cases = (
        (
            "triangle",
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
            jnp.asarray(((0.23, 0.31), (0.61, 0.12))),
        ),
        (
            "tetrahedron",
            jnp.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            ),
            jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
            jnp.asarray(((0.18, 0.21, 0.16), (0.42, 0.11, 0.09))),
        ),
    )
    for cell_kind, mesh_coordinates, cells, reference_points in cases:
        block_name = f"{cell_kind}s"
        mesh = phx.discretization.CellMesh(
            mesh_coordinates,
            (phx.discretization.CellBlock(block_name, cell_kind, cells),),
        )
        coordinate_element = phx.discretization.lagrange_element(cell_kind, 2)
        reference_nodes = coordinate_element.reference_nodes
        if cell_kind == "triangle":
            xi, eta = reference_nodes[:, 0], reference_nodes[:, 1]
            coordinate_values = jnp.stack(
                (
                    xi + 0.12 * xi * eta,
                    eta + 0.08 * xi * (1.0 - xi - eta),
                ),
                axis=-1,
            )
        else:
            xi, eta, zeta = (
                reference_nodes[:, 0],
                reference_nodes[:, 1],
                reference_nodes[:, 2],
            )
            coordinate_values = jnp.stack(
                (
                    xi + 0.08 * xi * eta,
                    eta + 0.06 * eta * zeta,
                    zeta + 0.05 * xi * zeta,
                ),
                axis=-1,
            )
        coordinate_spec = phx.discretization.FiniteElementCoordinateSpec(
            {block_name: coordinate_element},
            {
                block_name: jnp.arange(
                    coordinate_element.local_dof_count, dtype=jnp.int32
                )[None, :]
            },
            coordinate_values,
        )
        discretization = phx.discretization.FiniteElementPlan(
            mesh,
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element(cell_kind, 1)
            ),
            coordinate_spec=coordinate_spec,
        ).prepare()
        cell_map = phx.discretization.fem.prepare_finite_element_cell_map(
            discretization, 0
        )
        physical = cell_map.evaluate(
            coordinate_values,
            jnp.zeros((reference_points.shape[0],), dtype=jnp.int32),
            reference_points,
        ).physical_points
        locator = phx.discretization.PreparedSimplicialCellLocator(
            cell_map,
            coordinate_values,
            phx.discretization.SimplicialLocationPolicy(
                1,
                12,
                coordinate_element.local_dof_count + 1,
                residual_tolerance=1.0e-9,
            ),
        )
        located = locator.locate(physical)

        assert located.successful.all()
        np.testing.assert_array_equal(located.cell_ids, jnp.zeros_like(located.cell_ids))
        np.testing.assert_allclose(
            located.reference_coordinates,
            reference_points,
            rtol=1.0e-8,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            jnp.sum(located.barycentric, axis=-1),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_ale_flip_epoch_is_prepared_and_failed_steps_restore_checkpoint():
    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "triangles",
                "triangle",
                jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
            ),
        ),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    cell_map = phx.discretization.fem.prepare_finite_element_cell_map(discretization, 0)
    measure = phx.discretization.DiscreteMeasure(
        "ale-vertices",
        mesh.support.support_id,
        mesh.topology.entities(0).entity_set_id,
        jnp.ones((3,)),
    )
    target = phx.discretization.MeshSplatTarget(mesh, entity_dimension=0, measure=measure)
    position = jnp.asarray(((0.2, 0.2),))
    stable_id = jnp.asarray((17,), dtype=jnp.int64)
    splat = phx.discretization.SimplicialBarycentricSplatAssignment().prepare(
        target,
        position,
        jnp.asarray((True,)),
        stable_id,
    )
    particle_set = phx.discretization.ParticleSetPlan(
        stable_id, jnp.ones((1,)), ambient_dimension=2
    ).prepare()
    population = phx.discretization.ParticlePopulationPlan(particle_set).initialize(
        active_mask=jnp.asarray((True,)),
        masses=jnp.asarray((2.0,)),
    )
    epoch = phx.discretization.ParticleGridSplatEpoch(
        splat,
        population,
        position,
        jnp.zeros((3, 3)),
    )
    state = phx.discretization.flip.ALEFLIPState(
        epoch,
        phx.discretization.flip.FLIPParticleState(position, jnp.zeros_like(position)),
        discretization.default_runtime.coordinates,
        jnp.zeros((3,)),
        jnp.zeros((3,)),
    )
    plan = phx.discretization.flip.ALEFLIPPlan(splat, cell_map)
    prepared = phx.discretization.flip.prepare_ale_flip(plan, epoch)
    dt = 0.1
    target_velocity = jnp.broadcast_to(jnp.asarray((0.2, -0.1)), (3, 2))
    moved_coordinates = state.coordinates + dt * target_velocity
    step = phx.discretization.flip.advance_ale_flip(
        prepared,
        state,
        target_velocity,
        target_velocity,
        moved_coordinates,
        dt,
    )

    assert step.successful
    np.testing.assert_allclose(step.relative_particle_velocity, 0.0)
    np.testing.assert_allclose(
        step.accepted_state.particles.position,
        position + dt * jnp.asarray(((0.2, -0.1),)),
    )

    transitioned = phx.discretization.flip.transition_ale_flip_epoch(
        step.accepted_state,
        splat,
        population,
        step.accepted_state.particles.position,
        step.accepted_state.coordinates,
    )
    assert transitioned.epoch.epoch_number == epoch.epoch_number + 1
    np.testing.assert_allclose(
        transitioned.particles.velocity,
        step.accepted_state.particles.velocity,
    )

    stale = phx.discretization.flip.advance_ale_flip(
        prepared,
        transitioned,
        target_velocity,
        target_velocity,
        transitioned.coordinates + dt * target_velocity,
        dt,
    )
    assert not stale.successful
    np.testing.assert_array_equal(
        stale.accepted_state.epoch.epoch_number,
        transitioned.epoch.epoch_number,
    )
    np.testing.assert_array_equal(
        stale.accepted_state.particles.position,
        transitioned.particles.position,
    )
    np.testing.assert_array_equal(
        stale.accepted_state.coordinates,
        transitioned.coordinates,
    )

    fresh = phx.discretization.flip.prepare_ale_flip(plan, transitioned.epoch)
    resumed = phx.discretization.flip.advance_ale_flip(
        fresh,
        transitioned,
        target_velocity,
        target_velocity,
        transitioned.coordinates + dt * target_velocity,
        dt,
    )
    assert resumed.successful
