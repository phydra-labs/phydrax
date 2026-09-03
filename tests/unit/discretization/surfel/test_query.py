from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _indexed_geometry():
    positions = jnp.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (1.2, 0.0, 0.0)))
    ids = jnp.asarray((10, 11, 12), dtype=jnp.int64)
    prepared = phx.discretization.SurfelSetPlan(ids, positions, jnp.ones((3,))).prepare()
    normal = jnp.tile(jnp.asarray((0.0, 0.0, 1.0)), (3, 1))
    axes = jnp.tile(
        jnp.asarray(((0.5, 0.0), (0.0, 0.5), (0.0, 0.0)))[None, ...],
        (3, 1, 1),
    )
    certificate = phx.discretization.SurfelGeometryCertificate(
        source_geometry_id="ray-query-surfels",
        source_kind="analytic",
        orientation_scope=phx.discretization.SurfelOrientationScope.GLOBAL,
        one_sided=True,
    )
    geometry = phx.discretization.SurfelGeometryPlan(prepared).materialize(
        positions, normal, axes, certificate=certificate
    )
    hierarchy = phx.discretization.MortonPointHierarchyPlan(
        phx.discretization.MortonAddressPlan((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), 4),
        3,
        target_leaf_occupancy=1,
    ).build(positions, stable_ids=ids)
    bounds = phx.discretization.MortonPrimitiveBoundsPlan(hierarchy, 3).refit(
        positions - geometry.footprint_half_width,
        positions + geometry.footprint_half_width,
    )
    return geometry, hierarchy, bounds


def test_morton_primitive_bounds_contain_items_and_children() -> None:
    geometry, hierarchy, bounds = _indexed_geometry()
    assert bool(bounds.evidence.successful)
    active_items = geometry.active_mask
    assert bool(
        jnp.all(bounds.item_lower[active_items] <= geometry.position[active_items])
    )
    assert bool(
        jnp.all(bounds.item_upper[active_items] >= geometry.position[active_items])
    )
    children = hierarchy.node_children
    valid_child = children >= 0
    safe_child = jnp.maximum(children, 0)
    contains_lower = bounds.node_lower[:, None, :] <= bounds.node_lower[safe_child]
    contains_upper = bounds.node_upper[:, None, :] >= bounds.node_upper[safe_child]
    assert bool(jnp.all(~valid_child[..., None] | (contains_lower & contains_upper)))


def test_surfel_ray_query_orders_hits_and_reports_overflow() -> None:
    geometry, _, bounds = _indexed_geometry()
    origin = jnp.asarray(((0.0, 0.0, 2.0),))
    direction = jnp.asarray(((0.0, 0.0, -1.0),))
    query = phx.discretization.SurfelRayQueryPlan(
        bounds, geometry, maximum_hits_per_ray=3
    ).query(origin, direction)
    assert bool(query.evidence.successful[0])
    assert int(query.evidence.hit_count[0]) == 2
    np.testing.assert_array_equal(query.surfel_ids[0, :2], [10, 11])
    np.testing.assert_allclose(query.distance[0, :2], [2.0, 3.0])
    overflow = phx.discretization.SurfelRayQueryPlan(
        bounds, geometry, maximum_hits_per_ray=1
    ).query(origin, direction)
    assert bool(overflow.evidence.hit_overflow[0])
    assert not bool(overflow.evidence.successful[0])
    assert int(overflow.surfel_ids[0, 0]) == 10


def test_surfel_ray_query_respects_one_sided_and_parallel_geometry() -> None:
    geometry, _, bounds = _indexed_geometry()
    plan = phx.discretization.SurfelRayQueryPlan(bounds, geometry, maximum_hits_per_ray=3)
    backface = plan.query(
        jnp.asarray(((0.0, 0.0, -2.0),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
    )
    assert bool(backface.evidence.successful[0])
    assert int(backface.evidence.hit_count[0]) == 0
    parallel = plan.query(
        jnp.asarray(((0.0, 0.0, 0.25),)),
        jnp.asarray(((1.0, 0.0, 0.0),)),
    )
    assert bool(parallel.evidence.successful[0])
    assert int(parallel.evidence.hit_count[0]) == 0


def test_surfel_ray_query_jits_and_has_fixed_route_gradient() -> None:
    geometry, _, bounds = _indexed_geometry()
    plan = phx.discretization.SurfelRayQueryPlan(bounds, geometry, maximum_hits_per_ray=2)
    direction = jnp.asarray(((0.0, 0.0, -1.0),))
    query = eqx.filter_jit(plan.query)
    result = query(jnp.asarray(((0.0, 0.0, 2.0),)), direction)
    assert bool(result.evidence.successful[0])

    def distance(origin_z):
        hits = plan.query(jnp.asarray(((0.0, 0.0, origin_z),)), direction)
        return hits.distance[0, 0]

    gradient = jax.grad(distance)(jnp.asarray(2.0))
    np.testing.assert_allclose(gradient, 1.0)
