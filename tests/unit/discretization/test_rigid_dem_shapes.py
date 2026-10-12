#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer contracts for the common compliant DEM shape compiler and stepper."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import AbstractRigidContactShapePlan, DEMRuntimeState
from phydrax.equations import CompiledDiscreteElementProblem


def _shapes(kind: str) -> AbstractRigidContactShapePlan:
    material = jnp.zeros((2,), dtype=jnp.int32)
    if kind == "sphere":
        return phx.discretization.RigidSphereSetPlan(jnp.full((2,), 0.5), material)
    if kind == "clump":
        template = phx.discretization.SphereClumpTemplatePlan(
            jnp.zeros((1, 3)),
            jnp.asarray([0.5]),
            jnp.asarray([1.0]),
            jnp.zeros((1,), dtype=jnp.int32),
        )
        return phx.discretization.RigidSphereClumpSetPlan((template,), material, material)
    if kind == "superquadric":
        return phx.discretization.SuperquadricSetPlan(
            jnp.full((2, 3), 0.5),
            jnp.full((2,), 2.0),
            jnp.full((2,), 2.0),
            material,
        )
    if kind == "convex":
        template = phx.discretization.ConvexShapePlan(
            jnp.asarray(
                [[0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, -0.5]]
            ),
            jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32),
            0,
        )
        return phx.discretization.ConvexShapeSetPlan((template,), material, material)
    raise ValueError(f"Unknown test shape {kind!r}")


def _compile(
    shapes: AbstractRigidContactShapePlan,
    *,
    maximum_contact_routes: int | None = None,
    neighborhood: phx.discretization.AbstractParticleNeighborhoodPlan | None = None,
) -> CompiledDiscreteElementProblem:
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([17, 41], dtype=jnp.int64),
        jnp.ones((2,), dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.4]]),
    )
    return phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "shared-shape-test",
            materials,
            gravity=jnp.zeros((3,)),
        ),
        particles,
        shapes,
        phx.discretization.CompliantDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.LinearSpringDashpotNormalPlan(100.0),
                tangential=phx.discretization.CundallStrackTangentialPlan(20.0),
            ),
            maximum_overlap_fraction=0.5,
            maximum_contact_routes=maximum_contact_routes,
        ),
        neighborhood=(
            phx.discretization.DenseParticleNeighborhoodPlan(1)
            if neighborhood is None
            else neighborhood
        ),
    )


def _state(compiled: CompiledDiscreteElementProblem) -> DEMRuntimeState:
    return compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        jnp.zeros((2, 3)),
        orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
    )


@pytest.mark.parametrize(
    "kind",
    ["sphere", "clump", "superquadric", "convex"],
    ids=["sphere", "clump", "superquadric", "convex"],
)
@pytest.mark.strict_jax
def test_four_shapes_share_compilation_evaluation_and_accepted_step(kind: str) -> None:
    compiled = _compile(_shapes(kind))
    state = _state(compiled)
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(1.0e-4), None
    )
    assert compiled.dynamics.shapes.shape_kind == kind
    assert evaluation.successful
    assert not evaluation.geometry.overflow
    np.testing.assert_allclose(
        np.sum(evaluation.loads.total.force, axis=0), 0.0, atol=1.0e-12
    )
    geometry = evaluation.geometry.geometry
    np.testing.assert_allclose(geometry.normal_gradient, geometry.normal)
    np.testing.assert_array_equal(
        geometry.surface_area_weight, np.ones_like(geometry.gap)
    )
    assert state.body_properties.inertia_com.shape == (2, 3, 3)
    np.testing.assert_allclose(state.body_properties.shape_scale, 1.0)
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-4),
        None,
    )
    assert detail.successful
    assert detail.accepted_state.energy.accepted_steps == 1
    np.testing.assert_allclose(
        np.linalg.norm(detail.accepted_state.kinematics.orientation, axis=-1),
        1.0,
        atol=1.0e-12,
    )


def test_one_centered_pebble_matches_sphere_force_inertia_and_step() -> None:
    sphere = _compile(_shapes("sphere"))
    clump = _compile(_shapes("clump"))
    sphere_state = _state(sphere)
    clump_state = _state(clump)
    np.testing.assert_allclose(
        clump_state.body_properties.inertia_com,
        sphere_state.body_properties.inertia_com,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        clump_state.loads.total.force, sphere_state.loads.total.force
    )
    np.testing.assert_allclose(
        clump_state.loads.total.torque, sphere_state.loads.total.torque
    )
    sphere_step = sphere.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        sphere_state,
        jnp.asarray(1.0e-4),
        None,
    )
    clump_step = clump.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        clump_state,
        jnp.asarray(1.0e-4),
        None,
    )
    assert sphere_step.successful and clump_step.successful
    for expected, actual in zip(
        jax.tree.leaves(sphere_step.accepted_state.kinematics),
        jax.tree.leaves(clump_step.accepted_state.kinematics),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)


def test_offset_clump_contact_torque_matches_common_point_moment_and_inertia() -> None:
    template = phx.discretization.SphereClumpTemplatePlan(
        jnp.asarray([[0.0, -0.2, 0.0], [0.0, 0.2, 0.0]]),
        jnp.asarray([0.25, 0.25]),
        jnp.asarray([0.5, 0.5]),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    shapes = phx.discretization.RigidSphereClumpSetPlan(
        (template,),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    compiled = _compile(shapes)
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [0.48, 0.3, 0.0]]),
        jnp.zeros((2, 3)),
        orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
    )
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(0.0), None
    )
    assert evaluation.successful
    geometry = evaluation.geometry.geometry
    force = np.asarray(evaluation.particle_contact.pair_force)
    valid = np.asarray(geometry.valid)
    left_torque = np.cross(np.asarray(geometry.left_owner_arm), force)
    right_torque = np.cross(np.asarray(geometry.right_owner_arm), -force)
    expected = np.stack(
        (np.sum(left_torque[valid], axis=0), np.sum(right_torque[valid], axis=0))
    )
    np.testing.assert_allclose(evaluation.loads.total.torque, expected, atol=1.0e-12)
    assert np.linalg.norm(expected) > 0.0
    # Parallel-axis inertia: two half-mass spheres at ±0.2 along y.
    np.testing.assert_allclose(
        state.body_properties.inertia_com[0], np.diag([0.065, 0.025, 0.065]), atol=1.0e-14
    )
    world_moment = np.sum(
        np.cross(
            np.asarray(state.kinematics.position),
            np.asarray(evaluation.loads.total.force),
        )
        + expected,
        axis=0,
    )
    np.testing.assert_allclose(world_moment, 0.0, atol=1.0e-12)
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-6),
        None,
    )
    assert detail.successful
    expected_angular = expected / np.asarray([0.065, 0.025, 0.065]) * 1.0e-6
    np.testing.assert_allclose(
        detail.accepted_state.kinematics.angular_velocity,
        expected_angular,
        rtol=1.0e-3,
        atol=1.0e-12,
    )


def test_contact_route_overflow_refuses_candidate_without_partial_commit() -> None:
    template = phx.discretization.SphereClumpTemplatePlan(
        jnp.asarray([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]),
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([0.5, 0.5]),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    box = phx.discretization.ParticleBox(
        jnp.asarray([-2.0, -2.0, -2.0]),
        jnp.asarray([2.0, 2.0, 2.0]),
        periodic_axes=(False, False, False),
    )
    crowded = _compile(
        phx.discretization.RigidSphereClumpSetPlan(
            (template,),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((2,), dtype=jnp.int32),
        ),
        maximum_contact_routes=1,
        neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(1.2, 2, 1, box),
    )
    # No broad-phase pair is initially present, so the undersized route buffer
    # admits this state; the trial drift creates the overflowing owner pair.
    state = crowded.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        jnp.asarray([[250.0, 0.0, 0.0], [-250.0, 0.0, 0.0]]),
        orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
    )
    detail = crowded.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-3),
        None,
    )
    assert detail.evaluation.geometry.overflow
    assert not detail.successful
    for accepted, committed in zip(
        jax.tree.leaves(detail.accepted_state), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(accepted, committed)


def test_clump_history_survives_verlet_rebuild_with_stable_component_keys() -> None:
    box = phx.discretization.ParticleBox(
        jnp.asarray([-2.0, -2.0, -2.0]),
        jnp.asarray([2.0, 2.0, 2.0]),
        periodic_axes=(False, False, False),
    )
    neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(
        phx.discretization.CellListParticleNeighborhoodPlan(1.2, 2, 1, box),
        1.0,
        0.2,
    )
    compiled = _compile(_shapes("clump"), neighborhood=neighborhood)
    state = _state(compiled)
    seeded = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-4),
        None,
    )
    assert seeded.successful
    state = seeded.accepted_state
    assert (
        np.linalg.norm(np.asarray(state.particle_history.tangential.displacement)) > 0.0
    )
    original = compiled.dynamics.evaluate(jnp.asarray(0.0), state, jnp.asarray(0.0), None)
    staged = eqx.tree_at(
        lambda value: value.kinematics.position,
        state,
        state.kinematics.position + jnp.asarray([0.0, 0.25, 0.0]),
    )
    rebuilt = compiled.dynamics.evaluate(jnp.asarray(0.0), staged, jnp.asarray(0.0), None)
    assert rebuilt.successful
    assert rebuilt.neighborhood_cache is not None
    assert rebuilt.neighborhood_cache.rebuilt
    np.testing.assert_array_equal(
        rebuilt.particle_contact.next_history.pair_keys,
        original.particle_contact.next_history.pair_keys,
    )
    np.testing.assert_allclose(
        rebuilt.particle_contact.next_history.tangential.displacement,
        original.particle_contact.next_history.tangential.displacement,
    )


@pytest.mark.parametrize(
    "kind", ["clump", "superquadric", "convex"], ids=["clump", "superquadric", "convex"]
)
def test_anisotropic_shape_initialization_requires_explicit_orientation(
    kind: str,
) -> None:
    compiled = _compile(_shapes(kind))
    with pytest.raises(ValueError, match="orientation"):
        compiled.initialize_state(
            0.0,
            jnp.asarray([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
            jnp.zeros((2, 3)),
        )


def test_sharp_convex_hertz_curvature_is_refused_at_preparation() -> None:
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([17, 41], dtype=jnp.int64),
        jnp.ones((2,), dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.0]]),
    )
    with pytest.raises(ValueError, match="curvature"):
        phx.equations.compile_discrete_element_problem(
            phx.equations.DiscreteElementProblemIR(
                "sharp-convex-hertz",
                materials,
                gravity=jnp.zeros((3,)),
            ),
            particles,
            _shapes("convex"),
            phx.discretization.CompliantDEMMethodPlan(
                phx.discretization.DEMContactModelPlan(
                    phx.discretization.HertzNormalContactPlan()
                ),
            ),
            neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
        )


@pytest.mark.strict_jax
def test_superquadric_state_chart_roundtrip_and_differential_duality() -> None:
    compiled = _compile(_shapes("superquadric"))
    state = _state(compiled)
    geometry = compiled.dynamics.state_geometry

    def role(angular: Array) -> DEMRuntimeState:
        ambient = jax.tree.map(
            lambda leaf: (
                jnp.full_like(leaf, 1.0e-4)
                if eqx.is_inexact_array(leaf)
                else jnp.zeros_like(leaf)
            ),
            state,
        )
        ambient = eqx.tree_at(
            lambda value: value.kinematics.orientation, ambient, angular
        )
        return geometry.project_tangent(state, ambient)

    local = role(jnp.asarray([[0.0, 0.05, -0.03, 0.02], [0.0, -0.02, 0.04, 0.01]]))
    direction = role(jnp.asarray([[0.0, 0.04, 0.02, -0.01], [0.0, 0.03, -0.01, 0.02]]))
    cotangent = role(jnp.asarray([[0.0, -0.01, 0.03, 0.08], [0.0, 0.02, 0.05, -0.04]]))
    assert local.kinematics.orientation.shape == (2, 3)
    point = geometry.retract(state, local)
    recovered = geometry.inverse_retract(state, point)
    assert point.kinematics.orientation.shape == (2, 4)
    np.testing.assert_allclose(
        np.linalg.norm(point.kinematics.orientation, axis=-1), 1.0, atol=1.0e-12
    )
    for expected, actual in zip(
        jax.tree.leaves(local), jax.tree.leaves(recovered), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1.0e-10)
    np.testing.assert_array_equal(
        point.particle_history.pair_keys, state.particle_history.pair_keys
    )
    pushed = geometry.retraction_jvp(state, local, direction)
    inverse_pushed = geometry.retraction_inverse_jvp(state, point, pushed)
    pulled = geometry.retraction_vjp(state, local, cotangent)
    assert pushed.kinematics.orientation.shape == (2, 3)
    for expected, actual in zip(
        jax.tree.leaves(direction), jax.tree.leaves(inverse_pushed), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1.0e-10)
    left_pair = sum(
        (
            jnp.vdot(covector, tangent)
            for covector, tangent in zip(
                jax.tree.leaves(cotangent), jax.tree.leaves(pushed), strict=True
            )
            if eqx.is_inexact_array(covector)
        ),
        start=jnp.asarray(0.0),
    )
    right_pair = sum(
        (
            jnp.vdot(covector, tangent)
            for covector, tangent in zip(
                jax.tree.leaves(pulled), jax.tree.leaves(direction), strict=True
            )
            if eqx.is_inexact_array(covector)
        ),
        start=jnp.asarray(0.0),
    )
    np.testing.assert_allclose(left_pair, right_pair, atol=1.0e-10)


def test_shared_dynamics_refuses_body_properties_from_different_shape_identity() -> None:
    sphere = _compile(_shapes("sphere"))
    clump = _compile(_shapes("clump"))
    state = _state(sphere)
    foreign = _state(clump).body_properties
    assert foreign.shape_id != state.body_properties.shape_id
    with pytest.raises(ValueError, match="shape"):
        sphere.dynamics.apply_body_properties(
            jnp.asarray(0.0),
            state,
            foreign,
            jnp.asarray(True),
        )


def test_linear_convex_contact_marks_inapplicable_curvature_and_unlimited_routes() -> (
    None
):
    compiled = _compile(_shapes("convex"))
    state = _state(compiled)
    diagnostics = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(1.0e-4), None
    ).diagnostics
    certificate = phx.discretization.dem_local_validity_certificate(
        diagnostics, phx.discretization.DEMSensitivityPolicy()
    )
    assert certificate.forward_successful
    assert np.isinf(certificate.curvature_margin)
    assert np.isinf(certificate.contact_capacity_margin)


@pytest.mark.parametrize("target", ["force", "torque"], ids=["force", "torque"])
def test_common_body_load_refuses_missing_owner_payload(target: str) -> None:
    bodies = _compile(_shapes("sphere")).dynamics.bodies
    force = jnp.zeros((2, 3), dtype=jnp.float64)
    torque = jnp.zeros((2, 3), dtype=jnp.float64)
    if target == "force":
        force = force[:1]
    else:
        torque = torque[:1]
    with pytest.raises(ValueError, match="load shape"):
        bodies.load(force, torque)


@pytest.mark.parametrize(
    "kind", ["clump", "superquadric", "convex"], ids=["clump", "superquadric", "convex"]
)
def test_nonspherical_periodic_geometry_is_refused_before_state_publication(
    kind: str,
) -> None:
    box = phx.discretization.ParticleBox(
        jnp.full((3,), -2.0, dtype=jnp.float64),
        jnp.full((3,), 2.0, dtype=jnp.float64),
        periodic_axes=(True, True, True),
    )
    with pytest.raises(ValueError, match="Periodic.*sphere"):
        _compile(
            _shapes(kind),
            neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1, box=box),
        )
