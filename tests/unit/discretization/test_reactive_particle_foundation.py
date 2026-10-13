#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.typing import NDArray

import phydrax as phx


def _dem(
    dimension: Any = 3,
    *,
    contact: Any = None,
    particle_count: Any = 2,
    maximum_pairs: Any = None,
) -> Any:
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
        phx.discretization.CompliantDEMMethodPlan(selected),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            particle_count * (particle_count - 1) // 2
            if maximum_pairs is None
            else maximum_pairs
        ),
    )
    return compiled, particles


def test_reactive_particle_foundation_scenario_1() -> None:
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
    compiled = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "superquadric-foundation", material, gravity=jnp.zeros((3,))
        ),
        particles,
        shapes,
        phx.discretization.CompliantDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
            ),
            maximum_overlap_fraction=0.25,
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )
    dynamics = compiled.dynamics
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.zeros((2, 3)),
        orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    evaluation = dynamics.evaluate(jnp.asarray(0.0), state, jnp.asarray(1.0e-4), None)
    step = dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-4),
        None,
    )
    assert evaluation.successful
    assert jnp.allclose(evaluation.geometry.geometry.gap, -0.1, atol=1.0e-8)
    assert jnp.allclose(evaluation.geometry.geometry.effective_radius, 0.25, rtol=2.0e-3)
    assert jnp.allclose(evaluation.loads.total.force[0], -evaluation.loads.total.force[1])
    assert step.successful
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
        phx.discretization.CompliantDEMMethodPlan(
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
    # ty: ignore[unresolved-attribute]
    assert evaluation.multicontact.successful
    # ty: ignore[unresolved-attribute]
    assert jnp.max(evaluation.multicontact.gap_correction) > 0.0
    # ty: ignore[unresolved-attribute]
    assert evaluation.multicontact.residual < 1.0e-4


def test_reactive_particle_foundation_scenario_2() -> None:
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
        lambda value: value.body_properties.population.active,
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
    mesh = phx.discretization.UnstructuredParticleInternalMeshPlan(
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        tetrahedra=jnp.asarray([[0, 1, 2, 3]]),
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    batch = phx.discretization.ParticleInternalBatchPlan(
        jnp.asarray([0]), mesh, 1
    ).prepare(particles)
    state = phx.discretization.initialize_particle_internal_batch(
        batch,
        jnp.ones((1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.asarray([[0.2]]),
        jnp.ones((1, 1)),
        jnp.asarray([1.0]),
    )
    conversion = phx.discretization.initialize_particle_conversion_state((state,))
    morphology = phx.discretization.DensityPorosityMorphologyPlan(
        (jnp.asarray([1.0]),), neighborhood_skin=0.1
    )

    with pytest.raises(TypeError, match="radial shell meshes"):
        morphology.evaluate((batch,), conversion, (jnp.asarray([1.0]),))
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.asarray([1.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5]), jnp.asarray([0])
    ).prepare(particles)
    kinematics = bodies.bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.4]]),
        jnp.asarray([[1.0, 0.0, -1.0]]),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
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


def _multicontact_reference(
    plan: phx.discretization.ElasticHalfSpaceMulticontactPlan,
    left: NDArray[np.int32],
    right: NDArray[np.int32],
    points: NDArray[np.floating],
    normals: NDArray[np.floating],
    force: NDArray[np.floating],
    material_ids: NDArray[np.int32],
    materials: phx.equations.DEMMaterialTable,
    valid: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], float]:
    """Independent endpoint loops, with left endpoints preceding right endpoints."""
    count = len(left)
    owners = [*left.tolist(), *right.tolist()]
    young = np.asarray(materials.young_modulus)
    poisson = np.asarray(materials.poisson_ratio)
    deformation = np.zeros((2 * count,), dtype=np.float64)
    separation = math.inf
    for target, owner in enumerate(owners):
        edge = target % count
        target_normal = normals[edge] * (-1.0 if target < count else 1.0)
        for source, source_owner in enumerate(owners):
            source_edge = source % count
            if (
                owner != source_owner
                or edge == source_edge
                or not valid[edge]
                or not valid[source_edge]
            ):
                continue
            source_normal = normals[source_edge] * (-1.0 if source < count else 1.0)
            distance = math.sqrt(
                math.fsum(
                    (float(a) - float(b)) ** 2
                    for a, b in zip(points[edge], points[source_edge], strict=True)
                )
            )
            alignment = 0.5 * (1.0 + abs(float(np.dot(target_normal, source_normal))))
            material = material_ids[owner]
            deformation[target] += (
                plan.geometric_prefactor
                * (1.0 - float(poisson[material]) ** 2)
                / (
                    math.pi
                    * float(young[material])
                    * max(distance, plan.distance_regularization)
                )
                * alignment
                * float(force[source_edge])
            )
            separation = min(separation, distance)
    return deformation, separation


@pytest.mark.parametrize(
    "count", [1, 7, 129], ids=["isolated", "parallel", "ragged-tiles"]
)
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_multicontact_endpoint_order_and_brute_force_reference(
    count: int, dimension: int, dtype: type[np.floating]
) -> None:
    indices = np.arange(count, dtype=np.int32)
    left = indices % 5
    right = (indices + 1) % 5
    points = np.random.default_rng(81).normal(size=(count, dimension)).astype(dtype)
    # Coincident distinct contacts exercise regularization, not self exclusion.
    points[: min(3, count)] = points[0]
    normals = np.random.default_rng(82).normal(size=(count, dimension)).astype(dtype)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    force = (1.0 + indices % 11).astype(dtype)
    valid = indices % 7 != 6
    material_ids = np.asarray([0, 1, 0, 1, 0], dtype=np.int32)
    materials = phx.equations.DEMMaterialTable(
        np.asarray([2.0e5, 3.0e5], dtype=dtype),
        np.asarray([0.2, 0.3], dtype=dtype),
        np.full((2, 2), 0.8, dtype=dtype),
        np.full((2, 2), 0.4, dtype=dtype),
    )
    plan = phx.discretization.ElasticHalfSpaceMulticontactPlan(
        geometric_prefactor=1.3, distance_regularization=0.01
    )
    expected, separation = _multicontact_reference(
        plan, left, right, points, normals, force, material_ids, materials, valid
    )
    result = eqx.filter_jit(plan.evaluate)(
        left, right, points, normals, force, material_ids, materials, valid
    )
    tolerance = 3.0e-6 if dtype == np.float32 else 3.0e-13
    np.testing.assert_allclose(result.endpoint_deformation, expected, rtol=tolerance)
    np.testing.assert_allclose(
        result.gap_correction, expected[:count] + expected[count:], rtol=tolerance
    )
    np.testing.assert_allclose(result.regularity_margin, separation, rtol=tolerance)
    assert result.successful
    assert float(result.residual) == 0.0
    assert result.plan_id == plan.plan_id


def test_multicontact_masks_nonfinite_force_and_failure_status() -> None:
    plan = phx.discretization.ElasticHalfSpaceMulticontactPlan()
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    left = jnp.asarray([0, 0, 0], dtype=jnp.int32)
    right = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    points = jnp.asarray([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    normals = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    forces = jnp.asarray([1.0, 2.0, jnp.nan])
    material_ids = jnp.zeros((4,), dtype=jnp.int32)
    masked = plan.evaluate(
        left,
        right,
        points,
        normals,
        forces,
        material_ids,
        materials,
        jnp.asarray([True, True, False]),
    )
    unmasked = plan.evaluate(
        left,
        right,
        points,
        normals,
        forces,
        material_ids,
        materials,
        jnp.asarray([True, True, True]),
    )
    assert masked.successful
    np.testing.assert_array_equal(np.asarray(masked.endpoint_deformation)[[2, 5]], 0.0)
    assert not unmasked.successful
    assert np.isnan(np.asarray(unmasked.gap_correction)[:2]).all()


def test_multicontact_tiled_force_directional_derivative() -> None:
    count = 129
    indices = jnp.arange(count, dtype=jnp.int32)
    left = indices % 5
    right = (indices + 1) % 5
    coordinate = indices.astype(jnp.float64)
    points = jnp.stack((coordinate * 0.1, jnp.sin(coordinate)), axis=1)
    normals = jnp.tile(jnp.asarray([[1.0, 0.0]], dtype=jnp.float64), (count, 1))
    force = jnp.ones((count,), dtype=jnp.float64)
    direction = 0.1 + indices.astype(jnp.float64) / count
    material_ids = jnp.zeros((5,), dtype=jnp.int32)
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5], dtype=jnp.float64),
        jnp.asarray([0.25], dtype=jnp.float64),
        jnp.asarray([[0.8]], dtype=jnp.float64),
        jnp.asarray([[0.4]], dtype=jnp.float64),
    )
    valid = jnp.ones((count,), dtype=jnp.bool_)
    plan = phx.discretization.ElasticHalfSpaceMulticontactPlan()

    def deformation(load: jax.Array) -> jax.Array:
        return plan.evaluate(
            left, right, points, normals, load, material_ids, materials, valid
        ).endpoint_deformation

    _, tangent = jax.jit(lambda load, delta: jax.jvp(deformation, (load,), (delta,)))(
        force, direction
    )
    expected, _ = _multicontact_reference(
        plan,
        np.asarray(left),
        np.asarray(right),
        np.asarray(points),
        np.asarray(normals),
        np.asarray(direction),
        np.asarray(material_ids),
        materials,
        np.asarray(valid),
    )
    np.testing.assert_allclose(tangent, expected, rtol=3.0e-13)


@pytest.mark.parametrize("argument", [0, 1, 2, 3, 4, 5])
def test_multicontact_pair_shape_error_contract(argument: int) -> None:
    plan = phx.discretization.ElasticHalfSpaceMulticontactPlan()
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    pair = jnp.asarray([0, 1], dtype=jnp.int32)
    points = jnp.zeros((2, 2), dtype=jnp.float64)
    force = jnp.ones((2,), dtype=jnp.float64)
    valid = jnp.ones((2,), dtype=jnp.bool_)
    arguments = [pair, pair, points, points, force, valid]
    arguments[argument] = jnp.zeros((3,), dtype=jnp.float64)
    with pytest.raises(
        ValueError, match="Multicontact pair data has inconsistent shapes"
    ):
        plan.evaluate(
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
            arguments[4],
            pair,
            materials,
            arguments[5],
        )


def test_multicontact_empty_pair_reduction_error_contract() -> None:
    plan = phx.discretization.ElasticHalfSpaceMulticontactPlan()
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    indices = jnp.zeros((0,), dtype=jnp.int32)
    points = jnp.zeros((0, 2), dtype=jnp.float64)
    with pytest.raises(
        ValueError,
        match="zero-size array to reduction operation min which has no identity",
    ):
        plan.evaluate(
            indices,
            indices,
            points,
            points,
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((1,), dtype=jnp.int32),
            materials,
            jnp.zeros((0,), dtype=jnp.bool_),
        )
