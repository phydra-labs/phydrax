import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _hierarchy():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(
                0, (4,), 4, halo_width=1, refinement_ratio=2
            ),
            phx.discretization.BlockLevelPlan(
                1, (2,), 16, halo_width=1, refinement_ratio=2
            ),
            phx.discretization.BlockLevelPlan(2, (2,), 32, halo_width=1),
        ),
    )
    prepared = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    initial = prepared.initial_topology()
    coarse_tags = jnp.zeros((4, 4), dtype="bool").at[1:3].set(True)
    empty_middle = jnp.zeros((16, 2), dtype="bool")
    middle = prepared.compile_topology(initial, (coarse_tags, empty_middle)).topology
    middle_tags = jnp.zeros((16, 2), dtype="bool").at[3, 1].set(True)
    compiled = prepared.compile_topology(middle, (coarse_tags, middle_tags))
    assert compiled.status.successful
    return prepared, compiled.topology


def _runtime(prepared, topology):
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: jnp.zeros_like(state),
        lambda left, right, axis, args: jnp.zeros(left.shape[:-1]),
        system_id="cosmology-block-amr-test-scalar",
    )
    finite_volume = phx.discretization.BlockAMRFiniteVolumePlan(
        prepared,
        system,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    return phx.solver.BlockAMRRuntimePlan(finite_volume).prepare(topology)


def _runtime_state(runtime, scale_factor=0.5):
    topology = runtime.dynamics.topology
    levels = tuple(
        phx.discretization.BlockLevelState(
            plan,
            metadata,
            jnp.where(
                metadata.active.reshape((plan.maximum_blocks, 1, 1)),
                jnp.ones((plan.maximum_blocks, *plan.block_shape, 1)),
                0.0,
            ),
        )
        for plan, metadata in zip(topology.plan.levels, topology.levels, strict=True)
    )
    hierarchy = phx.discretization.BlockHierarchyState(topology, levels)
    return runtime.initial_state(hierarchy, time=scale_factor)


def _gravity(topology):
    routing = cosmology.BlockAMRParticleRoutingPlan(topology)
    operator = phx.discretization.CompositeAMRDiffusionPlan(routing.layout).prepare(1.0)
    return routing, cosmology.BlockAMRGravityPlan(operator, routing)


def test_n_level_particle_routing_uses_canonical_topology_and_conserves_deposit():
    _, topology = _hierarchy()
    routing, _ = _gravity(topology)
    positions = jnp.asarray([[0.49], [0.40], [0.05]])
    assignment = routing.route(positions)

    np.testing.assert_array_equal(assignment.levels, [2, 1, 0])
    np.testing.assert_array_equal(
        assignment.block_ids,
        [
            topology.plan.block_id(2, (15,)),
            topology.plan.block_id(1, (6,)),
            topology.plan.block_id(0, (0,)),
        ],
    )
    assert assignment.epoch_id == topology.epoch.epoch_id
    assert assignment.topology_id == topology.topology_id
    assert assignment.partition_id == topology.partition_id
    assert bool(assignment.successful)

    deposited = routing.deposit_density(assignment, jnp.asarray([2.0, 3.0, 5.0]))
    assert bool(deposited.successful)
    np.testing.assert_allclose(deposited.source_mass, 10.0)
    np.testing.assert_allclose(deposited.deposited_mass, 10.0)
    np.testing.assert_allclose(deposited.balance_defect, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(routing.layout.integral(deposited.density), 10.0)

    level_field = tuple(
        jnp.broadcast_to(jnp.asarray(level + 1.0), mask.shape + (1,))
        for level, mask in enumerate(routing.layout.leaf_mask)
    )
    gathered = routing.gather(assignment, level_field)
    assert bool(gathered.successful)
    np.testing.assert_allclose(gathered.values[:, 0], [3.0, 2.0, 1.0])


def test_composite_gravity_uses_one_linalg_solve_and_conservative_interface_routes():
    _, topology = _hierarchy()
    routing, gravity = _gravity(topology)
    positions = jnp.asarray([[0.49], [0.40], [0.05]])
    result = gravity.particle_force(positions, jnp.asarray([2.0, 3.0, 5.0]))

    assert bool(result.successful)
    assert bool(result.gravity.solve_result.successful)
    assert result.gravity.solve_result.provenance.method == "projected-pcg"
    assert result.gravity.epoch_id == topology.epoch.epoch_id
    np.testing.assert_allclose(result.deposited.balance_defect, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(result.gravity.source_integral, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        result.gravity.interface_flux_conservation_defect, 0.0, atol=0.0
    )

    layout = routing.layout
    potential = layout.flatten_cells(result.gravity.potential)[:, 0]
    image = layout.flatten_cells(gravity.operator.mv(result.gravity.potential))[:, 0]
    source = layout.flatten_cells(result.gravity.source)[:, 0]
    leaf = np.asarray(layout.flat_leaf_mask)
    np.testing.assert_allclose(image[leaf], source[leaf], rtol=2.0e-8, atol=2.0e-9)
    routes = gravity.operator.plan.routes
    interface = np.asarray(routes.edge_level_jump)
    assert np.any(interface)
    interface_gradient = (potential[routes.edge_right] - potential[routes.edge_left]) / (
        routes.edge_left_distance + routes.edge_right_distance
    )
    assert np.all(np.isfinite(np.asarray(interface_gradient)[interface]))
    assert all(
        np.all(np.isfinite(np.asarray(value))) for value in result.gravity.acceleration
    )


def test_block_amr_epoch_commit_is_atomic_for_flux_gravity_and_routing_failure():
    prepared, topology = _hierarchy()
    runtime = _runtime(prepared, topology)
    previous_state = _runtime_state(runtime)
    advance = runtime.advance(previous_state, 0.1)
    assert bool(advance.accepted)

    routing, gravity_plan = _gravity(topology)
    force = gravity_plan.particle_force(
        jnp.asarray([[0.49], [0.40], [0.05]]),
        jnp.asarray([2.0, 3.0, 5.0]),
    )
    previous_particles = cosmology.CosmologicalParticleState(
        jnp.asarray([[0.49], [0.40], [0.05]]),
        jnp.asarray([[1.0], [2.0], [3.0]]),
        jnp.asarray(0.5),
    )
    candidate_particles = cosmology.CosmologicalParticleState(
        previous_particles.positions,
        previous_particles.canonical_momenta + 0.25,
        jnp.asarray(0.6),
    )
    epoch = cosmology.BlockAMREpochPlan(runtime, routing)
    accepted = epoch.commit(
        previous_state,
        advance,
        previous_particles,
        candidate_particles,
        force.gravity,
    )

    assert bool(accepted.successful)
    assert accepted.epoch_id == topology.epoch.epoch_id
    np.testing.assert_allclose(accepted.runtime_state.time, 0.6)
    np.testing.assert_allclose(accepted.particles.scale_factor, 0.6)
    np.testing.assert_allclose(
        accepted.particles.canonical_momenta,
        candidate_particles.canonical_momenta,
    )

    failed_gravity = eqx.tree_at(
        lambda value: value.successful,
        force.gravity,
        jnp.asarray(False),
    )
    rejected_gravity = epoch.commit(
        previous_state,
        advance,
        previous_particles,
        candidate_particles,
        failed_gravity,
    )
    bad_particles = cosmology.CosmologicalParticleState(
        candidate_particles.positions.at[0, 0].set(jnp.nan),
        candidate_particles.canonical_momenta,
        candidate_particles.scale_factor,
    )
    rejected_routing = epoch.commit(
        previous_state,
        advance,
        previous_particles,
        bad_particles,
        force.gravity,
    )
    rejected_runtime = runtime.advance(previous_state, jnp.asarray(jnp.nan))
    rejected_flux = epoch.commit(
        previous_state,
        rejected_runtime,
        previous_particles,
        candidate_particles,
        force.gravity,
    )
    previous_assignment = routing.route(previous_particles.positions)

    for rejected in (rejected_gravity, rejected_routing, rejected_flux):
        assert not bool(rejected.successful)
        np.testing.assert_array_equal(
            rejected.runtime_state.level_accepted_steps,
            previous_state.level_accepted_steps,
        )
        np.testing.assert_array_equal(
            rejected.particles.positions, previous_particles.positions
        )
        np.testing.assert_array_equal(
            rejected.particles.canonical_momenta,
            previous_particles.canonical_momenta,
        )
        np.testing.assert_array_equal(
            rejected.particles.scale_factor,
            previous_particles.scale_factor,
        )
        np.testing.assert_array_equal(
            rejected.assignment.block_ids,
            previous_assignment.block_ids,
        )
        np.testing.assert_array_equal(
            rejected.assignment.global_cell_indices,
            previous_assignment.global_cell_indices,
        )
        np.testing.assert_array_equal(
            rejected.assignment.active,
            previous_assignment.active,
        )
        for actual, expected in zip(
            rejected.runtime_state.hierarchy_state.levels,
            previous_state.hierarchy_state.levels,
            strict=True,
        ):
            np.testing.assert_array_equal(actual.values, expected.values)
