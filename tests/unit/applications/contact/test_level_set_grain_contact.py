#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.applications.contact import prepare_mpm_contact_participant
from phydrax.applications.contact._level_set_grain import (
    MPMLevelSetGrainContactParticipant,
)
from phydrax.discretization.particle._dem_contact import LevelSetPenaltyNormalPlan
from phydrax.discretization.particle._level_set import (
    LevelSetRigidShapeSetPlan,
    LevelSetShapeTemplatePlan,
    PreparedLevelSetRigidShapeSet,
)
from phydrax.discretization.particle._rigid_body import RigidBodyKinematics


pytestmark = pytest.mark.strict_jax


@pytest.fixture
def grains() -> PreparedLevelSetRigidShapeSet:
    address = phx.discretization.MortonAddressPlan((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), 4)
    coordinates = np.stack(
        np.meshgrid(np.arange(16), np.arange(16), np.arange(16), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    grid = phx.discretization.SparseVoxelGridPlan(
        address, brick_size=2, brick_capacity=512
    ).prepare(coordinates)
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.5, feature_id="grain").compile()
    samples = phx.geometry.VoxelGeometrySamplingPlan(grid).sample_level_set(geometry)
    template = LevelSetShapeTemplatePlan(
        samples,
        np.asarray([100], dtype=np.int64),
        np.asarray([[0.5, 0.0, 0.0]]),
        np.asarray([np.pi]),
        0,
        exact_volume=np.pi / 6.0,
        exact_center_of_mass=np.zeros(3),
        exact_inertia_per_mass=np.eye(3) * 0.1,
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([3], dtype=jnp.int64),
        jnp.asarray([2.0], dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    return LevelSetRigidShapeSetPlan(
        (template,),
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        maximum_contact_routes=8,
        maximum_candidate_slots=8,
    ).prepare(particles)


def _pose(grains: PreparedLevelSetRigidShapeSet) -> RigidBodyKinematics:
    return grains.bodies.kinematics(
        jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
    )


def _participant(
    grains: PreparedLevelSetRigidShapeSet,
    positions: Array,
    *,
    capacity: int = 8,
    radius: float = 0.0,
) -> MPMLevelSetGrainContactParticipant:
    count = positions.shape[0]
    source = phx.linalg.ArraySpace((count, 3), dtype=np.float64)
    plan = phx.discretization.CollisionSurfacePlan(
        jnp.arange(count, dtype=jnp.int64) + 30,
        ambient_dimension=3,
        edges=jnp.empty((0, 2), dtype=jnp.int32),
        allow_isolated_vertices=True,
    )
    mpm = prepare_mpm_contact_participant(
        plan,
        positions,
        phx.discretization.selection_collision_operator(
            source, jnp.arange(count, dtype=jnp.int32)
        ),
    )
    return MPMLevelSetGrainContactParticipant(
        mpm,
        grains,
        LevelSetPenaltyNormalPlan(200.0, damping_per_area=3.0),
        jnp.asarray([0.2, 0.3, 0.0][:count], dtype=jnp.float64),
        jnp.full((count,), radius, dtype=jnp.float64),
        maximum_contact_routes=capacity,
    )


def test_mpm_block_grain_balances_and_duality(
    grains: PreparedLevelSetRigidShapeSet,
) -> None:
    positions = jnp.asarray([[1.4, 2.1, 3.05], [1.4, 1.9, 3.05]], dtype=jnp.float64)
    participant = _participant(grains, positions)
    state = jnp.zeros_like(positions)
    rates = jnp.asarray([[0.1, 0.2, -0.1], [-0.2, 0.1, 0.3]], dtype=jnp.float64)
    result = eqx.filter_jit(participant.evaluate)(
        state, rates, jnp.ones((2,), dtype=jnp.float64), _pose(grains)
    )
    assert bool(result.successful)
    assert bool(result.participant_duality.valid)
    assert int(result.candidate_count) == 2
    assert int(result.active_count) == 2
    assert float(result.potential) > 0.0
    np.testing.assert_allclose(result.force_defect, 0.0, atol=1e-13)
    np.testing.assert_allclose(result.moment_defect, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        result.mpm_generalized_effort, result.mpm_surface_force, atol=1e-13
    )
    pose = _pose(grains)
    np.testing.assert_allclose(
        jnp.sum(result.mpm_surface_force, axis=0)
        + jnp.sum(result.grain_load.force, axis=0),
        0.0,
        atol=1e-13,
    )
    net_moment = jnp.sum(
        jnp.cross(positions, result.mpm_surface_force), axis=0
    ) + jnp.sum(
        result.grain_load.torque + jnp.cross(pose.position, result.grain_load.force),
        axis=0,
    )
    np.testing.assert_allclose(net_moment, 0.0, atol=1e-13)


def test_physical_area_raw_gradient_and_radius(
    grains: PreparedLevelSetRigidShapeSet,
) -> None:
    positions = jnp.asarray([[1.55, 2.0, 3.0], [1.55, 2.0, 3.0]], dtype=jnp.float64)
    participant = _participant(grains, positions, radius=0.15)
    result = participant.evaluate(
        jnp.zeros_like(positions),
        jnp.zeros_like(positions),
        jnp.ones((2,), dtype=jnp.float64),
        _pose(grains),
    )
    assert bool(result.successful)
    overlap = -result.gap[:2]
    expected = (
        200.0
        * overlap[:, None]
        * result.geometry.raw_gradient[:2]
        * jnp.asarray([0.2, 0.3], dtype=jnp.float64)[:, None]
    )
    np.testing.assert_allclose(result.mpm_surface_force, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        result.potential,
        jnp.sum(100.0 * overlap**2 * jnp.asarray([0.2, 0.3], dtype=jnp.float64)),
        rtol=1e-13,
    )
    np.testing.assert_allclose(result.moment_defect, 0.0, atol=1e-13)


def test_nonfinite_mass_refuses(grains: PreparedLevelSetRigidShapeSet) -> None:
    positions = jnp.asarray([[1.4, 2.1, 3.05]], dtype=jnp.float64)
    participant = _participant(grains, positions)
    result = participant.evaluate(
        jnp.zeros_like(positions),
        jnp.zeros_like(positions),
        jnp.asarray([jnp.nan], dtype=jnp.float64),
        _pose(grains),
    )
    assert not bool(result.successful)
    assert int(result.status) == 3
    np.testing.assert_array_equal(result.mpm_generalized_effort, 0.0)
    np.testing.assert_array_equal(result.grain_load.force, 0.0)


def test_overflow_refuses_entire_candidate(grains: PreparedLevelSetRigidShapeSet) -> None:
    positions = jnp.asarray([[1.4, 2.1, 3.05], [1.4, 1.9, 3.05]], dtype=jnp.float64)
    participant = _participant(grains, positions, capacity=1)
    result = participant.evaluate(
        jnp.zeros_like(positions),
        jnp.zeros_like(positions),
        jnp.ones((2,), dtype=jnp.float64),
        _pose(grains),
    )
    assert bool(result.overflow)
    assert not bool(result.successful)
    assert int(result.candidate_count) == 2
    assert int(result.status) == 1
    np.testing.assert_array_equal(result.mpm_generalized_effort, 0.0)
    np.testing.assert_array_equal(result.grain_load.force, 0.0)
    np.testing.assert_array_equal(result.grain_load.torque, 0.0)
    assert float(result.potential) == 0.0


@pytest.mark.parametrize(
    "point,radius",
    [((3.1, 2.0, 3.0), 3.0), ((1.0, 2.0, 3.0), 0.0)],
    ids=["incomplete-stencil", "zero-gradient"],
)
def test_unsupported_query_refuses(
    grains: PreparedLevelSetRigidShapeSet,
    point: tuple[float, float, float],
    radius: float,
) -> None:
    positions = jnp.asarray([point], dtype=jnp.float64)
    participant = _participant(grains, positions, radius=radius)
    result = participant.evaluate(
        jnp.zeros_like(positions),
        jnp.zeros_like(positions),
        jnp.ones((1,), dtype=jnp.float64),
        _pose(grains),
    )
    assert int(result.candidate_count) == 1
    assert not bool(result.query_valid)
    assert not bool(result.successful)
    assert int(result.status) == 2
    np.testing.assert_array_equal(result.mpm_surface_force, 0.0)
    np.testing.assert_array_equal(result.grain_load.force, 0.0)


def test_inactive_and_noncandidate_lanes_are_safe(
    grains: PreparedLevelSetRigidShapeSet,
) -> None:
    positions = jnp.asarray(
        [[50.0, 50.0, 50.0], [1.4, 2.1, 3.05], [50.0, 50.0, 50.0]], dtype=jnp.float64
    )
    participant = _participant(grains, positions)
    result = participant.evaluate(
        jnp.zeros_like(positions),
        jnp.zeros_like(positions),
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
        _pose(grains),
    )
    assert bool(result.successful)
    assert int(result.candidate_count) == 0
    assert int(result.active_count) == 0
    assert bool(result.query_valid)
    np.testing.assert_array_equal(result.mpm_surface_force, 0.0)
    np.testing.assert_array_equal(result.grain_load.torque, 0.0)


@pytest.mark.parametrize(
    "capacity", [8, 48], ids=["streamed-overflow", "complete-padded"]
)
def test_streamed_candidate_capacity_is_atomic(
    grains: PreparedLevelSetRigidShapeSet, capacity: int
) -> None:
    count = 40
    positions = jnp.broadcast_to(
        jnp.asarray([1.4, 2.1, 3.05], dtype=jnp.float64), (count, 3)
    )
    source = phx.linalg.ArraySpace((count, 3), dtype=np.float64)
    plan = phx.discretization.CollisionSurfacePlan(
        jnp.arange(count, dtype=jnp.int64) + 30,
        ambient_dimension=3,
        edges=jnp.empty((0, 2), dtype=jnp.int32),
        allow_isolated_vertices=True,
    )
    mpm = prepare_mpm_contact_participant(
        plan,
        positions,
        phx.discretization.selection_collision_operator(
            source, jnp.arange(count, dtype=jnp.int32)
        ),
    )
    participant = MPMLevelSetGrainContactParticipant(
        mpm,
        grains,
        LevelSetPenaltyNormalPlan(200.0),
        jnp.full((count,), 0.2, dtype=jnp.float64),
        jnp.zeros((count,), dtype=jnp.float64),
        maximum_contact_routes=capacity,
    )
    result = eqx.filter_jit(participant.evaluate)(
        jnp.zeros_like(positions),
        jnp.zeros_like(positions),
        jnp.ones((count,), dtype=jnp.float64),
        _pose(grains),
    )
    assert result.candidate_count == count
    assert bool(result.successful) == (capacity >= count)
    assert bool(result.overflow) == (capacity < count)
    np.testing.assert_array_equal(
        result.mpm_vertex_indices[: min(capacity, count)], np.arange(min(capacity, count))
    )
    if capacity < count:
        np.testing.assert_array_equal(result.mpm_surface_force, 0.0)
        np.testing.assert_array_equal(result.grain_load.force, 0.0)
        np.testing.assert_array_equal(result.route_force, 0.0)
    else:
        np.testing.assert_allclose(result.force_defect, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(result.moment_defect, 0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    "speed", (-100.0, 1.0, 100.0), ids=("closing", "rebound", "clamped")
)
def test_normal_damping_is_non_tensile_and_energy_consistent(
    grains: PreparedLevelSetRigidShapeSet, speed: float
) -> None:
    positions = jnp.asarray([[1.4, 2.0, 3.0]], dtype=jnp.float64)
    participant = _participant(grains, positions)
    rates = jnp.asarray([[speed, 0.0, 0.0]], dtype=jnp.float64)
    result = eqx.filter_jit(participant.evaluate)(
        jnp.zeros_like(positions), rates, jnp.ones(1, dtype=jnp.float64), _pose(grains)
    )
    assert result.successful
    assert result.active_count == 1
    normal = result.geometry.unit_normal[0]
    elastic_force = 0.2 * 200.0 * (-result.gap[0]) * result.geometry.raw_gradient[0]
    normal_rate = jnp.sum(rates[0] * normal)
    expected = jnp.maximum(jnp.sum(elastic_force * normal) - 0.2 * 3.0 * normal_rate, 0.0)
    np.testing.assert_allclose(result.mpm_surface_force[0], expected * normal, atol=1e-13)
    assert jnp.sum(result.mpm_surface_force[0] * normal) >= 0.0
    np.testing.assert_allclose(
        result.dissipated_power,
        jnp.sum((elastic_force - result.mpm_surface_force[0]) * rates[0]),
        rtol=1e-13,
    )
    assert result.dissipated_power >= 0.0
    if speed == 100.0:
        np.testing.assert_array_equal(result.mpm_surface_force, 0.0)
        np.testing.assert_array_equal(result.grain_load.force, 0.0)
        assert result.potential > 0.0
