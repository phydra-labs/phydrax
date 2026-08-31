#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _dem(dimension=3, *, contact=None, particle_count=2, maximum_pairs=None):
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count),
        jnp.ones((particle_count,)),
        ambient_dimension=dimension,
    ).prepare()
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
        rolling_friction=jnp.asarray([[0.1]]),
    )
    selected = (
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        )
        if contact is None
        else contact
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "reactive-foundation", materials, gravity=jnp.zeros((dimension,))
    )
    compiled = phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        phx.discretization.RigidSphereSetPlan(
            jnp.full((particle_count,), 0.5),
            jnp.zeros((particle_count,), dtype=jnp.int32),
        ),
        phx.discretization.SoftSphereDEMMethodPlan(selected),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            particle_count * (particle_count - 1) // 2
            if maximum_pairs is None
            else maximum_pairs
        ),
    )
    return compiled, particles


def test_compositional_contact_history_cohesion_and_rotational_energy_are_valid():
    cohesion = phx.discretization.CompositeDEMCohesionPlan(
        (
            phx.discretization.DMTContactCohesionPlan(0.05, 0.1),
            phx.discretization.LinearCapillaryBridgePlan(0.07, 0.0, 1.0e-9, 0.1),
            phx.discretization.NearContactLubricationPlan(1.0e-3, 0.1, 1.0e-5),
        )
    )
    contact = phx.discretization.DEMContactModelPlan(
        phx.discretization.HertzNormalContactPlan(),
        cohesion=cohesion,
        rotational=phx.discretization.ElasticRollingTorsionalResistancePlan(
            100.0,
            50.0,
            rolling_damping=1.0,
            torsional_damping=1.0,
            torsional_friction=0.05,
        ),
    )
    compiled, _ = _dem(contact=contact)
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]),
    )
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(1.0e-4), None
    )
    history = evaluation.particle_contact.next_history
    assert evaluation.successful
    assert len(history.cohesion.components) == 3
    assert history.normal.maximum_overlap.shape == (1,)
    assert history.tangential.displacement.shape == (1, 3)
    assert history.rotational.rolling_displacement.shape == (1, 3)
    assert jnp.max(jnp.abs(evaluation.particle_contact.bridge_volume_residual)) < 1.0e-14
    assert evaluation.particle_contact.rotational_dissipated_work[0] >= 0.0
    assert jnp.allclose(
        evaluation.particle_contact.rotational_torque_left,
        -evaluation.particle_contact.rotational_torque_right,
    )


def test_superquadric_geometry_contact_and_rigid_step_recover_sphere_limit():
    geometry = phx.geometry.Superquadric((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)).compile()
    points = jnp.asarray([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    curvature = geometry.contact_curvature(points)
    assert jnp.allclose(geometry.boundary_field(points), 0.0)
    assert jnp.allclose(curvature.principal_curvatures, 2.0, rtol=2.0e-3)
    assert jnp.all(curvature.valid)

    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    shapes = phx.discretization.SuperquadricSetPlan(
        jnp.full((2, 3), 0.5),
        jnp.full((2,), 2.0),
        jnp.full((2,), 2.0),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    material = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    plan = phx.discretization.SuperquadricDEMPlan(
        shapes,
        phx.discretization.SuperquadricContactPlan(),
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        ),
    )
    dynamics = plan.prepare(
        particles, material, phx.discretization.DenseParticleNeighborhoodPlan(1)
    )
    state = dynamics.initialize_state(
        jnp.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
    )
    evaluation = dynamics.evaluate(state, jnp.asarray(1.0e-4))
    step = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-4))
    assert evaluation.successful
    assert jnp.allclose(evaluation.geometry.gap, -0.1, atol=1.0e-8)
    assert jnp.allclose(evaluation.geometry.effective_radius, 0.25, rtol=2.0e-3)
    assert jnp.allclose(evaluation.load.force[0], -evaluation.load.force[1])
    assert step.successful


def test_multicontact_correction_is_nonlocal_convergent_and_optional():
    contact = phx.discretization.DEMContactModelPlan(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1, 2]), jnp.ones((3,)), ambient_dimension=2
    ).prepare()
    material = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    compiled = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "multicontact", material, gravity=jnp.zeros((2,))
        ),
        particles,
        phx.discretization.RigidSphereSetPlan(
            jnp.full((3,), 0.5), jnp.zeros((3,), dtype=jnp.int32)
        ),
        phx.discretization.SoftSphereDEMMethodPlan(
            contact,
            multicontact=phx.discretization.ElasticHalfSpaceMulticontactPlan(
                iterations=8, convergence_tolerance=1.0e-4
            ),
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(3),
    )
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [-0.9, 0.0], [0.9, 0.0]]),
        jnp.zeros((3, 2)),
    )
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(1.0e-4), None
    )
    assert evaluation.successful
    assert evaluation.multicontact.successful
    assert jnp.max(evaluation.multicontact.gap_correction) > 0.0
    assert evaluation.multicontact.residual < 1.0e-4


def test_radial_mesh_morphology_and_fixed_pool_insertion_preserve_inventory():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    batch_plan = phx.discretization.ParticleInternalBatchPlan(
        jnp.asarray([0, 1]),
        phx.discretization.RadialShellMeshPlan(
            phx.discretization.ParticleInternalGeometry.SPHERE, 2
        ),
        1,
    )
    batch = batch_plan.prepare(particles)
    metrics = batch.mesh.metrics(jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(
        jnp.sum(metrics.cell_measures, axis=1),
        (4.0 / 3.0) * jnp.pi * jnp.asarray([1.0, 8.0]),
    )
    state = phx.discretization.initialize_particle_internal_batch(
        batch,
        jnp.ones((2, 2)),
        jnp.ones((2, 2, 1)),
        jnp.zeros((2, 2)),
        jnp.ones((2, 2)),
        jnp.asarray([1.0, 2.0]),
    )
    conversion = phx.discretization.initialize_particle_conversion_state((state,))
    morphology = phx.discretization.DensityPorosityMorphologyPlan(
        (jnp.asarray([1.0]),), neighborhood_skin=0.1
    )
    evaluation = morphology.evaluate((batch,), conversion, (jnp.asarray([1.0]),))
    assert evaluation.successful
    assert jnp.isclose(evaluation.body_properties.masses[0], 2.0)
    assert jnp.abs(evaluation.mass_residual) < 1.0e-12

    region = phx.discretization.ParticleRegionPlan(
        jnp.asarray([-1.0, -1.0, -1.0]), jnp.asarray([1.0, 1.0, 1.0])
    )
    residence = region.initialize_residence(jnp.zeros((2, 3)))
    residence = region.update_residence(residence, jnp.zeros((2, 3)), jnp.asarray(0.5))
    assert jnp.allclose(residence.residence_time, 0.5)
    flow = phx.discretization.MassFlowSurfacePlan(
        jnp.zeros((3,)), jnp.asarray([1.0, 0.0, 0.0])
    )
    assert jnp.isclose(
        flow.crossed_mass(
            jnp.asarray([[-1.0, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0]]),
            jnp.asarray([2.0]),
            jnp.asarray([True]),
        ),
        2.0,
    )

    compiled, _ = _dem(dimension=3, particle_count=2)
    dem_state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
    )
    dem_state = eqx.tree_at(
        lambda value: value.body_properties.active,
        dem_state,
        jnp.asarray([True, False]),
    )
    internal_state = eqx.tree_at(
        lambda value: value.active,
        state,
        jnp.asarray([True, False]),
    )
    template = phx.discretization.ReactiveParticleTemplatePlan(
        0.1,
        1.0,
        0,
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([[0.5], [0.5]]),
        jnp.full((2,), 0.2),
        jnp.ones((2,)),
        outer_scale=0.1,
    )
    distribution = phx.discretization.ReactiveParticleTemplateDistributionPlan(
        (template,), jnp.asarray([1.0])
    )
    insertion = phx.discretization.insert_reactive_particles(
        phx.discretization.ParticleInsertionPlan(
            jnp.asarray([2.0, -0.2, -0.2]),
            jnp.asarray([3.0, 0.2, 0.2]),
            1,
            maximum_attempts=8,
        ),
        distribution,
        compiled.dynamics,
        dem_state,
        batch,
        internal_state,
        jnp.asarray([1.0]),
        jr.key(0),
        jnp.asarray(0.0),
    )
    assert insertion.successful
    assert insertion.owner_slots[0] == 1
    assert insertion.accepted_dem_state.body_properties.active[1]
    assert insertion.accepted_internal_state.internal_energy[1, 0] < 0.0


def test_wall_observables_and_wear_close_force_and_volume_channels():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.asarray([1.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5]), jnp.asarray([0])
    ).prepare(particles)
    kinematics = bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.4]]),
        jnp.asarray([[1.0, 0.0, -1.0]]),
        jnp.zeros((1, 3)),
    )
    wall = phx.discretization.TriangleWallPlan(
        jnp.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.asarray([[0, 1, 2]]),
        jnp.asarray([0]),
    ).prepare()
    geometry = phx.discretization.sphere_triangle_contact_geometry(
        bodies, kinematics, wall
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    contact_model = phx.discretization.DEMContactModelPlan(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
    ).prepare(materials, 3)
    history = contact_model.empty_history(1, jnp.float64)
    context = phx.discretization.DEMContactEvaluationContext(
        geometry.geometry.contact_keys,
        geometry.geometry.valid,
        jnp.asarray([False]),
        bodies.inverse_masses,
        jnp.zeros((1,)),
        bodies.radii,
        bodies.radii,
        bodies.material_ids,
        geometry.wall_material,
        jnp.asarray(1.0e-4),
        jnp.asarray(0, dtype=jnp.int32),
    )
    contact = contact_model.evaluate(
        geometry.geometry.as_contact_batch(), history, context
    )
    observables = phx.discretization.evaluate_wall_facet_observables(
        wall, geometry, contact
    )
    wear = phx.discretization.FinnieWearPlan(
        jnp.asarray([[1.0e-6]]), jnp.asarray([[1.0e6]])
    )
    wear_step = wear.step(
        wall, geometry, contact, wear.initialize(wall), jnp.asarray(0.1)
    )
    assert observables.successful
    assert jnp.linalg.norm(observables.force_residual) < 1.0e-12
    assert wear_step.successful
    assert jnp.all(wear_step.evaluation.wear_rate >= 0.0)
