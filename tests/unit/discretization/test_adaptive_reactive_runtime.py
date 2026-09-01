#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _dem_epoch():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([10, 20]), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    material = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.2]]),
    )
    compiled = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "adaptive-runtime", material, gravity=jnp.zeros((2,))
        ),
        particles,
        phx.discretization.RigidSphereSetPlan(
            jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
        ),
        phx.discretization.SoftSphereDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
            )
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.99, 0.0]]),
        jnp.zeros((2, 2)),
    )
    return phx.discretization.initialize_particle_execution_epoch(
        compiled.dynamics, state
    )


def test_pair_identity_and_contact_history_survive_capacity_growth():
    epoch = _dem_epoch()
    old_identity = epoch.state.particle_history.pair_keys[0]
    old_overlap = epoch.state.particle_history.normal.previous_overlap[0]
    transition = phx.discretization.grow_particle_execution_epoch(
        epoch,
        phx.discretization.ParticleCapacityGrowthPolicy(
            minimum_increment=2, maximum_capacity=8
        ),
        phx.discretization.ParticleCapacityRequest(1),
        jnp.asarray(0.0),
    )
    assert transition.successful
    assert transition.accepted_epoch.dynamics.bodies.capacity == 4
    next_history = transition.accepted_epoch.state.particle_history
    matches = jnp.all(next_history.pair_keys == old_identity, axis=-1)
    index = jnp.argmax(matches.astype(jnp.int32))
    assert jnp.any(matches)
    assert jnp.isclose(next_history.normal.previous_overlap[index], old_overlap)
    assert jnp.abs(transition.mass_residual) < 1.0e-12
    assert jnp.linalg.norm(transition.momentum_residual) < 1.0e-12


def test_capacity_limit_rejects_without_changing_epoch():
    epoch = _dem_epoch()
    transition = phx.discretization.grow_particle_execution_epoch(
        epoch,
        phx.discretization.ParticleCapacityGrowthPolicy(
            minimum_increment=2, maximum_capacity=2
        ),
        phx.discretization.ParticleCapacityRequest(1),
        jnp.asarray(0.0),
    )
    assert not transition.successful
    assert transition.accepted_epoch.epoch_id == epoch.epoch_id
    assert transition.accepted_epoch.dynamics.bodies.capacity == 2


def test_triangle_wall_feature_identity_has_unique_shared_edge_owner():
    wall = phx.discretization.TriangleWallPlan(
        jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        jnp.asarray([[0, 1, 2], [1, 3, 2]]),
        jnp.asarray([0, 0]),
        triangle_ids=jnp.asarray([7, 3]),
    ).prepare()
    shared = set(np.asarray(wall.edge_ids[0]).tolist()) & set(
        np.asarray(wall.edge_ids[1]).tolist()
    )
    assert len(shared) == 1
    edge_id = next(iter(shared))
    assert wall.edge_owner_triangle_ids[edge_id] == 3


def _internal_epoch_state(epoch):
    particles = epoch.dynamics.bodies.particles
    batch = phx.discretization.ParticleInternalBatchPlan(
        jnp.arange(particles.capacity),
        phx.discretization.RadialShellMeshPlan(
            phx.discretization.ParticleInternalGeometry.SPHERE, 1
        ),
        1,
    ).prepare(particles)
    state = phx.discretization.initialize_particle_internal_batch(
        batch,
        jnp.ones((particles.capacity, 1)),
        jnp.ones((particles.capacity, 1, 1)),
        jnp.full((particles.capacity, 1), 0.2),
        jnp.ones((particles.capacity, 1)),
        jnp.full((particles.capacity,), 0.5),
    )
    return batch, state


def test_fixed_pool_insertion_grows_epoch_and_retires_identity_once():
    epoch = _dem_epoch()
    batch, internal = _internal_epoch_state(epoch)
    template = phx.discretization.ReactiveParticleTemplatePlan(
        0.1,
        1.0,
        0,
        jnp.zeros((2,)),
        jnp.zeros((1,)),
        jnp.asarray([1.0]),
        jnp.asarray([[1.0]]),
        jnp.asarray([0.2]),
        jnp.asarray([1.0]),
    )
    result = phx.discretization.insert_reactive_particles_with_growth(
        phx.discretization.ParticleInsertionPlan(
            jnp.asarray([2.0, -0.5]), jnp.asarray([3.0, 0.5]), 1
        ),
        phx.discretization.ReactiveParticleTemplateDistributionPlan(
            (template,), jnp.asarray([1.0])
        ),
        epoch,
        batch,
        internal,
        jnp.asarray([1.0]),
        jr.key(0),
        jnp.asarray(0.0),
        phx.discretization.ParticleCapacityGrowthPolicy(
            minimum_increment=2, maximum_capacity=8
        ),
    )
    assert result.successful
    assert result.transition is not None
    assert result.epoch.dynamics.bodies.capacity == 4
    assert jnp.sum(result.epoch.ever_occupied) == 3
    inserted = result.insertion.owner_slots[0]
    assert result.epoch.state.body_properties.active[inserted]


def test_fragmentation_grows_epoch_and_conserves_inventory():
    epoch = _dem_epoch()
    batch, internal = _internal_epoch_state(epoch)
    result = phx.discretization.fragment_particle_with_growth(
        phx.discretization.ThermochemicalFragmentationPlan(2),
        epoch,
        batch,
        internal,
        jnp.asarray(0),
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([0.25, 0.25]),
        jnp.asarray([True, True]),
        jnp.asarray([1.0]),
        jnp.asarray(0.0),
        phx.discretization.ParticleCapacityGrowthPolicy(
            minimum_increment=2, maximum_capacity=8
        ),
    )
    assert result.successful
    assert result.transition is not None
    assert result.epoch.retired[0]
    assert jnp.abs(result.fragmentation.mass_residual) < 1.0e-12
    assert jnp.abs(result.fragmentation.energy_residual) < 1.0e-12


def test_segmented_epoch_execution_records_growth_and_routes():
    epoch = _dem_epoch()

    def step(current, index):
        return current.dynamics.step_detailed(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * 1.0e-5),
            current.state,
            jnp.asarray(1.0e-5),
            None,
        )

    trajectory = phx.solver.advance_particle_epoch_segments(
        epoch,
        step,
        (1, 1),
        growth_policy=phx.discretization.ParticleCapacityGrowthPolicy(
            minimum_increment=2, maximum_capacity=8
        ),
        growth_requests=(phx.discretization.ParticleCapacityRequest(1),),
    )
    assert trajectory.successful
    assert len(trajectory.segments) == 2
    assert len(trajectory.transitions) == 1
    assert trajectory.final_epoch.dynamics.bodies.capacity == 4


def test_unstructured_internal_mesh_measures_and_transport_are_conservative():
    mesh = phx.discretization.UnstructuredParticleInternalMeshPlan(
        jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        tetrahedra=jnp.asarray([[0, 1, 2, 3]]),
    )
    metrics = mesh.prepare().metrics(jnp.asarray([2.0]))
    assert jnp.isclose(metrics.cell_measures[0, 0], 4.0 / 3.0)
    assert metrics.successful

    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    batch = phx.discretization.ParticleInternalBatchPlan(
        jnp.asarray([0]), mesh, 1
    ).prepare(particles)
    schema = phx.equations.ChemicalSpeciesSchema(
        ("A",),
        (phx.equations.ChemicalPhaseKind.SOLID,),
        jnp.asarray([1.0]),
        ("X",),
        jnp.asarray([[1]]),
        jnp.zeros_like(jnp.asarray([1.0]), dtype=jnp.int32),
    )
    thermo = phx.equations.ParticleThermodynamicMaterialPlan(
        phx.equations.PolynomialSpeciesThermodynamicsPlan(
            schema, jnp.asarray([10.0]), jnp.asarray([0.0])
        )
    )
    material = phx.equations.ParticleThermochemicalMaterialBundle(
        thermo,
        phx.equations.ParticleTransportMaterialPlan(
            schema, jnp.asarray([1.0]), jnp.asarray([0.0])
        ),
    )
    species = jnp.ones((1, 1, 1))
    state = phx.discretization.initialize_particle_internal_batch(
        batch,
        thermo.energy_from_temperature(jnp.asarray([[300.0]]), species),
        species,
        jnp.asarray([[0.2]]),
        jnp.ones((1, 1)),
        jnp.asarray([1.0]),
    )
    boundary = phx.equations.ParticleTransportBoundary(
        jnp.asarray([400.0]),
        jnp.zeros((1, 1)),
        jnp.asarray([1.0]),
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1)),
    )
    evaluation = phx.equations.evaluate_particle_transport(
        batch, state, material, boundary
    )
    assert evaluation.successful
    assert jnp.abs(evaluation.internal_energy_residual) < 1.0e-12


def test_superquadric_wall_sphere_limit_and_action_reaction():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    shapes = phx.discretization.SuperquadricSetPlan(
        jnp.asarray([[0.5, 0.5, 0.5]]),
        jnp.asarray([2.0]),
        jnp.asarray([2.0]),
        jnp.asarray([0]),
    )
    wall = phx.discretization.TriangleWallPlan(
        jnp.asarray([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]),
        jnp.asarray([[0, 1, 2]]),
        jnp.asarray([0]),
    )
    material = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.2]]),
    )
    dynamics = phx.discretization.SuperquadricDEMPlan(
        shapes,
        phx.discretization.SuperquadricContactPlan(),
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        ),
        walls=(wall,),
        wall_geometry=phx.discretization.SuperquadricTriangleContactPlan(),
    ).prepare(
        particles,
        material,
        phx.discretization.DenseParticleNeighborhoodPlan(0),
    )
    state = dynamics.initialize_state(
        jnp.asarray([[0.0, 0.0, 0.4]]),
        jnp.zeros((1, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        jnp.zeros((1, 3)),
    )
    evaluation = dynamics.evaluate(state, jnp.asarray(1.0e-4))
    response = evaluation.walls[0]
    assert evaluation.successful
    assert jnp.isclose(response.geometry.geometry.gap[0], -0.1, atol=1.0e-8)
    assert (
        jnp.linalg.norm(
            jnp.sum(response.particle_load.force, axis=0) + response.reaction_force
        )
        < 1.0e-12
    )
    assert response.geometry.feature_tie_margin[0] > 0.0
    observables = phx.discretization.evaluate_wall_facet_observables(
        dynamics.walls[0], response.geometry, response.contact
    )
    wear = phx.discretization.FinnieWearPlan(
        jnp.asarray([[1.0e-6]]), jnp.asarray([[1.0e6]])
    ).evaluate(dynamics.walls[0], response.geometry, response.contact)
    assert observables.successful
    assert wear.successful


def test_ellipsoid_plane_witness_recovers_analytic_axis_gap():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([4]), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    shape_plan = phx.discretization.SuperquadricSetPlan(
        jnp.asarray([[0.6, 0.4, 0.3]]),
        jnp.asarray([2.0]),
        jnp.asarray([2.0]),
        jnp.asarray([0]),
    )
    shapes = shape_plan.prepare(particles)
    bodies = shape_plan.rigid_body_plan(particles).prepare(particles)
    kinematics = bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.25]]),
        jnp.zeros((1, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        jnp.zeros((1, 3)),
    )
    wall = phx.discretization.TriangleWallPlan(
        jnp.asarray([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]),
        jnp.asarray([[0, 1, 2]]),
        jnp.asarray([0]),
    ).prepare()
    result = phx.discretization.superquadric_triangle_contact_geometry(
        phx.discretization.SuperquadricTriangleContactPlan(),
        shapes,
        kinematics,
        wall,
    )
    assert result.geometry.valid[0]
    assert jnp.isclose(result.geometry.gap[0], -0.05, atol=1.0e-8)
    assert result.witness_residual[0] < 1.0e-10
