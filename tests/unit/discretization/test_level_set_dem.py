# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.particle._dem_contact import (
    LevelSetPenaltyNormalPlan,
    LevelSetTangentialContactPlan,
)
from phydrax.discretization.particle._level_set import (
    LevelSetRigidShapeSetPlan,
    LevelSetShapeTemplatePlan,
)
from phydrax.equations import CompiledDiscreteElementProblem


def _template(
    depth: int = 3, polar_nodes: int = 16, *, exact: bool = True
) -> LevelSetShapeTemplatePlan:
    resolution = 2**depth
    coordinates = np.stack(
        np.meshgrid(*([np.arange(resolution)] * 3), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid = phx.discretization.SparseVoxelGridPlan(
        phx.discretization.MortonAddressPlan((-1.0,) * 3, (1.0,) * 3, depth),
        brick_size=2,
        brick_capacity=(resolution // 2) ** 3,
    ).prepare(coordinates)
    geometry = phx.geometry.Sphere(
        (0.0,) * 3, 0.5, feature_id="level-set-sphere"
    ).compile()
    samples = phx.geometry.VoxelGeometrySamplingPlan(
        grid,
        enclosure=phx.geometry.ExactSDFEnclosureCertificate(geometry.field_certificate),
        enclosure_geometry=geometry,
    ).sample_level_set(geometry)
    mu, w = np.polynomial.legendre.leggauss(polar_nodes)
    angle = 2 * np.pi * np.arange(8) / 8
    nodes = 0.5 * np.stack(
        (
            np.broadcast_to(mu[:, None], (polar_nodes, 8)),
            np.sqrt(1 - mu[:, None] ** 2) * np.cos(angle)[None, :],
            np.sqrt(1 - mu[:, None] ** 2) * np.sin(angle)[None, :],
        ),
        axis=-1,
    ).reshape(-1, 3)
    weights = np.broadcast_to(
        (0.25 * w * 2 * np.pi / 8)[:, None], (polar_nodes, 8)
    ).reshape(-1)
    if exact:
        return LevelSetShapeTemplatePlan(
            samples,
            np.arange(nodes.shape[0], dtype=np.int64),
            nodes,
            weights,
            0,
            exact_volume=4 * np.pi * 0.5**3 / 3,
            exact_center_of_mass=np.zeros(3),
            exact_inertia_per_mass=0.1 * np.eye(3),
        )
    return LevelSetShapeTemplatePlan(
        samples,
        np.arange(nodes.shape[0], dtype=np.int64),
        nodes,
        weights,
        0,
        volume_tolerance=5.0,
    )


def _compile(
    template: LevelSetShapeTemplatePlan,
    *,
    capacity: int | None = None,
    verlet: bool = False,
) -> CompiledDiscreteElementProblem:
    nodes = template.surface_node_ids.size
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([17, 41], dtype=jnp.int64),
        jnp.ones(2, dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    shapes = LevelSetRigidShapeSetPlan(
        (template,),
        np.zeros(2, dtype=np.int32),
        np.zeros(2, dtype=np.int32),
        maximum_contact_routes=2 * nodes if capacity is None else capacity,
        maximum_candidate_slots=2 * nodes,
    )
    material = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.4]]),
    )
    method = phx.discretization.CompliantDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            LevelSetPenaltyNormalPlan(100.0),
            tangential=LevelSetTangentialContactPlan(20.0),
        ),
        maximum_overlap_fraction=0.8,
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1)
    if verlet:
        neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(
            neighborhood, 2.0 * float(template.bounding_radius), 0.2
        )
    return phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "level-set-test", material, gravity=jnp.zeros(3)
        ),
        particles,
        shapes,
        method,
        neighborhood=neighborhood,
    )


def _state(
    compiled: CompiledDiscreteElementProblem,
) -> phx.discretization.DEMRuntimeState:
    return compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]]),
        orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
    )


def _sphere_surface_force(distance: float, radius: float, stiffness: float) -> float:
    # Independent exact surface integral of k(R-r) n_x over the penetrating cap.
    lower = distance - radius
    a = distance**2 - radius**2

    def primitive(r: float) -> float:
        return radius * a * r - a * r * r / 2 + radius * r**3 / 3 - r**4 / 4

    return (
        np.pi * stiffness * radius / distance**2 * (primitive(radius) - primitive(lower))
    )


@pytest.mark.strict_jax
def test_sphere_field_force_converges_to_exact_surface_penalty() -> None:
    exact = _sphere_surface_force(0.8, 0.5, 100.0)
    errors = []
    # Refine beyond the measured 32³ field error (2.3%) to test the 2% criterion.
    for depth, nodes in ((4, 32), (5, 64), (6, 128)):
        compiled = _compile(_template(depth, nodes))
        force = float(np.asarray(_state(compiled).loads.total.force)[1, 0])
        errors.append(abs(force - exact))
    rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    print(f"sphere-level-set force errors={errors}, h-refinement rates={rates.tolist()}")
    assert errors[2] < errors[1] < errors[0]
    assert np.min(rates) > 0.8
    assert errors[-1] / exact < 0.02


def test_symmetric_half_area_energy_is_single_counted() -> None:
    template = _template()
    compiled = _compile(template)
    state = _state(compiled)
    result = compiled.dynamics.evaluate(jnp.asarray(0.0), state, jnp.asarray(0.0), None)
    assert result.successful
    geometry = result.geometry.geometry
    full_area = np.asarray(template.physical_surface_area_weights)[
        np.asarray(geometry.left_feature)
    ]
    np.testing.assert_allclose(
        geometry.surface_area_weight, 0.5 * full_area, rtol=1.0e-14
    )
    full_one_sided = (
        0.5
        * 100.0
        * np.asarray(geometry.overlap) ** 2
        * full_area
        * np.asarray(geometry.valid)
    )
    np.testing.assert_allclose(
        result.particle_contact.elastic_energy, 0.5 * full_one_sided, rtol=1.0e-13
    )
    np.testing.assert_allclose(
        np.sum(result.particle_contact.elastic_energy),
        0.5 * np.sum(full_one_sided),
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(np.sum(result.loads.total.force, axis=0), 0.0, atol=1e-13)
    torque = np.sum(
        np.cross(
            np.asarray(state.kinematics.position), np.asarray(result.loads.total.force)
        )
        + np.asarray(result.loads.total.torque),
        axis=0,
    )
    np.testing.assert_allclose(torque, 0.0, atol=1e-13)


def test_node_history_survives_verlet_rebuild() -> None:
    compiled = _compile(_template(), verlet=True)
    state = _state(compiled)
    first = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-4),
        None,
    )
    assert first.successful
    committed = first.accepted_state
    reused = compiled.dynamics.evaluate(
        jnp.asarray(1.0e-4), committed, jnp.asarray(0.0), None
    )
    cache = committed.neighborhood_cache
    if cache is None:
        raise ValueError("Verlet test requires its prepared cache.")
    forced_cache = eqx.tree_at(
        lambda c: c.reference_position, cache, cache.reference_position + 1.0
    )
    forced = eqx.tree_at(lambda s: s.neighborhood_cache, committed, forced_cache)
    rebuilt = compiled.dynamics.evaluate(
        jnp.asarray(1.0e-4), forced, jnp.asarray(0.0), None
    )
    assert reused.successful and rebuilt.successful
    np.testing.assert_array_equal(
        reused.particle_contact.next_history.pair_keys,
        rebuilt.particle_contact.next_history.pair_keys,
    )
    np.testing.assert_allclose(
        reused.particle_contact.next_history.tangential.displacement,
        rebuilt.particle_contact.next_history.tangential.displacement,
        atol=1.0e-15,
    )
    assert (
        np.max(
            np.abs(
                np.asarray(rebuilt.particle_contact.next_history.tangential.displacement)
            )
        )
        > 0.0
    )


def test_route_overflow_refuses_atomic_step() -> None:
    compiled = _compile(_template(), capacity=1)
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        jnp.asarray([[200.0, 0.0, 0.0], [-200.0, 0.0, 0.0]]),
        orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
    )
    result = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-3),
        None,
    )
    assert not result.successful
    assert result.evaluation.geometry.overflow
    for before, after in zip(
        jax.tree.leaves(state), jax.tree.leaves(result.accepted_state), strict=True
    ):
        np.testing.assert_array_equal(after, before)


def test_voxel_mass_properties_carry_error_evidence() -> None:
    template = _template(exact=False)
    assert template.evidence.successful
    assert not template.evidence.exact_properties
    assert template.evidence.volume_error_bound > 0.0
    assert template.evidence.boundary_cells > 0
    np.testing.assert_allclose(template.center_of_mass, 0.0, atol=1.0e-14)
    assert np.min(np.linalg.eigvalsh(np.asarray(template.inertia_per_mass))) > 0.0


def test_surface_routes_refuse_point_force_law() -> None:
    normal = LevelSetPenaltyNormalPlan(100.0)
    with pytest.raises(ValueError, match="per-area tangential"):
        phx.discretization.DEMContactModelPlan(
            normal, tangential=phx.discretization.CundallStrackTangentialPlan(10.0)
        )


def test_certified_level_set_query_encloses_source_sphere_distance() -> None:
    compiled = _compile(_template())
    state = _state(compiled)
    points = jnp.asarray([[0.31, 0.12, 0.07], [0.53, 0.15, 0.1]], dtype=jnp.float64)
    query = compiled.dynamics.shapes.evaluate_signed_field_query(
        state.kinematics, jnp.zeros(2, dtype=jnp.int32), points
    )
    exact = np.sqrt(np.sum(np.asarray(points) ** 2, axis=1)) - 0.5
    assert query.successful
    assert np.all(np.asarray(query.evaluation_error) > 0.0)
    assert np.all(
        np.abs(np.asarray(query.value) - exact) <= np.asarray(query.evaluation_error)
    )


def test_level_set_packing_uses_shared_shape_narrowphase() -> None:
    compiled = _compile(_template())
    packing = phx.discretization.RandomSequentialPackingPlan(
        jnp.full(3, -5.0, dtype=jnp.float64),
        jnp.full(3, 5.0, dtype=jnp.float64),
        8,
        maximum_candidate_evaluations=16,
    )
    result = packing.initialize(compiled.dynamics.shapes, jax.random.key(19))
    assert result.successful and result.complete
    assert np.all(np.asarray(result.accepted_mask))
    distance = np.linalg.norm(
        np.diff(np.asarray(result.accepted_kinematics.position), axis=0)
    )
    assert distance >= 1.0


def test_raw_gradient_force_matches_discrete_energy_derivative() -> None:
    compiled = _compile(_template())
    state = _state(compiled)

    def energy(offset: float) -> float:
        moved = eqx.tree_at(
            lambda s: s.kinematics.position,
            state,
            state.kinematics.position.at[1, 0].add(offset),
        )
        evaluated = compiled.dynamics.evaluate(
            jnp.asarray(0.0), moved, jnp.asarray(0.0), None
        )
        assert evaluated.successful
        return float(jnp.sum(evaluated.particle_contact.elastic_energy))

    epsilon = 1.0e-6
    derivative = (energy(epsilon) - energy(-epsilon)) / (2.0 * epsilon)
    np.testing.assert_allclose(
        state.loads.total.force[1, 0], -derivative, rtol=1.0e-7, atol=1.0e-10
    )


def test_level_set_verlet_rebuild_accounts_for_rotation_without_translation() -> None:
    compiled = _compile(_template(), verlet=True)
    state = _state(compiled)
    cache = state.neighborhood_cache
    if cache is None or not isinstance(
        compiled.dynamics.neighborhood,
        phx.discretization.PreparedVerletParticleNeighborhood,
    ):
        raise ValueError("Rotation test requires its prepared Verlet cache.")
    quarter_turn = jnp.asarray(
        [[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]] * 2, dtype=jnp.float64
    )
    updated = compiled.dynamics.neighborhood.update(
        state.kinematics.position,
        cache,
        orientation=quarter_turn,
        radii=compiled.dynamics.shapes.bounding_radii,
    )
    assert updated.successful
    assert updated.rebuilt
    assert updated.maximum_reference_displacement > 0.0


@pytest.mark.parametrize(
    "invalid_pose", ["negative-owner", "missing-owner", "non-unit-quaternion"]
)
def test_shared_level_set_query_refuses_invalid_pose(invalid_pose: str) -> None:
    compiled = _compile(_template())
    state = _state(compiled)
    indices = jnp.asarray([0], dtype=jnp.int32)
    kinematics = state.kinematics
    if invalid_pose == "negative-owner":
        indices = jnp.asarray([-1], dtype=jnp.int32)
    elif invalid_pose == "missing-owner":
        indices = jnp.asarray([2], dtype=jnp.int32)
    else:
        kinematics = eqx.tree_at(
            lambda k: k.orientation, kinematics, 2.0 * kinematics.orientation
        )
    shapes = compiled.dynamics.shapes
    if not isinstance(shapes, phx.discretization.PreparedLevelSetRigidShapeSet):
        raise ValueError("Pose refusal test requires its prepared level-set shape.")
    points = jnp.asarray([[0.35, 0.11, 0.07]], dtype=jnp.float64)
    query = phx.discretization.evaluate_level_set_point_query(
        shapes, kinematics, indices, points
    )
    assert not query.successful
    assert not np.any(np.asarray(query.valid))
    signed = shapes.evaluate_signed_field_query(kinematics, indices, points)
    assert not signed.successful
    assert np.all(np.isnan(np.asarray(signed.value)))
    assert np.all(np.isinf(np.asarray(signed.evaluation_error)))
