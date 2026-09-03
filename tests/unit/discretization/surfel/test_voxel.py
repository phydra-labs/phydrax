from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _dense_grid():
    address = phx.discretization.MortonAddressPlan((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 3)
    coordinates = np.stack(
        np.meshgrid(np.arange(8), np.arange(8), np.arange(8), indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))
    return phx.discretization.SparseVoxelGridPlan(
        address, brick_size=2, brick_capacity=64
    ).prepare(coordinates)


def _plane_geometry(normals=None):
    positions = jnp.asarray(((0.0, 0.0, 0.0),))
    prepared = phx.discretization.SurfelSetPlan(
        jnp.asarray((0,)), positions, jnp.asarray((1.0,))
    ).prepare()
    normal = jnp.asarray(((0.0, 0.0, 1.0),)) if normals is None else normals
    axes = jnp.asarray(([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],))
    certificate = phx.discretization.SurfelGeometryCertificate(
        source_geometry_id="plane",
        source_kind="analytic-plane",
        position_accuracy=phx.discretization.SurfelAccuracy.EXACT,
        normal_accuracy=phx.discretization.SurfelAccuracy.EXACT,
        orientation_scope=phx.discretization.SurfelOrientationScope.GLOBAL,
        coverage_scope=phx.discretization.SurfelCoverageScope.SAMPLED,
    )
    return phx.discretization.SurfelGeometryPlan(prepared).materialize(
        positions, normal, axes, certificate=certificate
    )


def test_surfel_voxel_projection_recovers_local_plane_and_attributes() -> None:
    grid = _dense_grid()
    geometry = _plane_geometry()
    plan = phx.discretization.SurfelVoxelProjectionPlan(
        grid,
        geometry,
        maximum_voxels_per_surfel=256,
        route_capacity=256,
        normal_distance_support=0.3,
        route_padding=0.1,
    )
    prepared = plan.prepare(geometry)
    result = prepared.project(geometry, attributes=jnp.asarray(((4.0, -2.0),)))
    assert bool(result.successful)
    assert int(result.evidence.supported_voxels) > 0
    centers = grid.voxel_centers()
    np.testing.assert_allclose(
        result.implicit_value[result.supported],
        centers[..., 2][result.supported],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.normal[result.supported],
        jnp.broadcast_to(
            jnp.asarray((0.0, 0.0, 1.0)),
            result.normal[result.supported].shape,
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.attributes[result.supported],
        jnp.broadcast_to(
            jnp.asarray((4.0, -2.0)),
            result.attributes[result.supported].shape,
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_surfel_voxel_routes_fail_closed_on_candidate_capacity() -> None:
    grid = _dense_grid()
    geometry = _plane_geometry()
    prepared = phx.discretization.SurfelVoxelProjectionPlan(
        grid,
        geometry,
        maximum_voxels_per_surfel=1,
        route_capacity=1,
        normal_distance_support=0.3,
    ).prepare(geometry)
    assert not bool(prepared.evidence.successful)
    assert bool(prepared.evidence.candidate_overflow)
    assert int(prepared.evidence.maximum_candidates_per_surfel) > 1


def test_surfel_voxel_projection_rejects_invalid_confidence() -> None:
    grid = _dense_grid()
    geometry = _plane_geometry()
    routes = phx.discretization.SurfelVoxelProjectionPlan(
        grid,
        geometry,
        maximum_voxels_per_surfel=128,
        route_capacity=128,
        normal_distance_support=0.3,
    ).prepare(geometry)
    result = routes.project(geometry, confidence=jnp.asarray((jnp.nan,)))
    assert not bool(result.successful)
    assert int(result.evidence.invalid_confidence_surfels) == 1


def test_surfel_voxel_projection_rejects_opposing_surface_layers() -> None:
    grid = _dense_grid()
    positions = jnp.zeros((2, 3))
    prepared = phx.discretization.SurfelSetPlan(
        jnp.asarray((0, 1)), positions, jnp.ones((2,))
    ).prepare()
    normals = jnp.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))
    axes = jnp.tile(
        jnp.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))[None, ...],
        (2, 1, 1),
    )
    certificate = phx.discretization.SurfelGeometryCertificate(
        source_geometry_id="opposing-planes",
        source_kind="analytic",
        orientation_scope=phx.discretization.SurfelOrientationScope.LOCAL,
    )
    geometry = phx.discretization.SurfelGeometryPlan(prepared).materialize(
        positions, normals, axes, certificate=certificate
    )
    routes = phx.discretization.SurfelVoxelProjectionPlan(
        grid,
        geometry,
        maximum_voxels_per_surfel=128,
        route_capacity=256,
        normal_distance_support=0.3,
        minimum_normal_coherence=0.5,
    ).prepare(geometry)
    result = routes.project(geometry)
    assert int(result.evidence.conflicting_voxels) > 0
    assert not bool(jnp.any(result.supported & result.conflicting))


def test_surfel_voxel_projection_jits_and_has_fixed_route_gradient() -> None:
    grid = _dense_grid()
    geometry = _plane_geometry()
    prepared = phx.discretization.SurfelVoxelProjectionPlan(
        grid,
        geometry,
        maximum_voxels_per_surfel=256,
        route_capacity=256,
        normal_distance_support=0.3,
        route_padding=0.2,
    ).prepare(geometry)
    project = eqx.filter_jit(prepared.project)
    reference = project(geometry)
    assert bool(reference.successful)
    fixed_support = reference.supported

    def objective(offset):
        moved = eqx.tree_at(
            lambda value: value.position,
            geometry,
            geometry.position.at[0, 2].set(offset),
        )
        value = prepared.project(moved).implicit_value
        return jnp.sum(jnp.where(fixed_support, value, 0.0))

    gradient = jax.grad(objective)(jnp.asarray(0.0))
    np.testing.assert_allclose(
        gradient,
        -jnp.sum(fixed_support),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
