# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.contact._ccd import (
    CCDStatus,
    CertifiedAABBCCDPlan,
    collision_free_step_limit,
    InclusionCCDPlan,
)
from phydrax.discretization.contact._search import (
    DenseContactSearchPlan,
    SweepAndPruneContactSearchPlan,
)
from phydrax.discretization.contact._stencils import (
    ContactStencilBatch,
    ContactStencilKind,
)
from phydrax.discretization.contact._surface import (
    CollisionSurfacePlan,
    PreparedCollisionScene,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from phydrax.linalg import ArraySpace


def _scene() -> PreparedCollisionScene:
    space = ArraySpace((4, 3), dtype=np.float64)
    topology = CollisionSurfacePlan(
        np.arange(4, dtype=np.int64),
        ambient_dimension=3,
        body_ids=np.array([0, 1, 1, 1], dtype=np.int32),
        allow_isolated_vertices=True,
    )
    surface = PreparedCollisionSurface(
        topology,
        jnp.zeros((4, 3), dtype=jnp.float64),
        selection_collision_operator(space, np.arange(4, dtype=np.int32)),
    )
    return PreparedCollisionScene((surface,))


@pytest.mark.parametrize("certified", [False, True], ids=["inclusion", "rounding-aabb"])
@pytest.mark.parametrize(
    "kind, vertices",
    [
        (ContactStencilKind.FACE_VERTEX, [(0.0, 0.0, 1.0)] + [(0.0, 0.0, 0.0)] * 3),
        (
            ContactStencilKind.FACE_VERTEX,
            [(0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        ),
        (ContactStencilKind.EDGE_VERTEX, [(0.0, 0.0, 1.0)] + [(0.0, 0.0, 0.0)] * 3),
        (ContactStencilKind.EDGE_EDGE, [(0.0, 0.0, 1.0)] * 2 + [(0.0, 0.0, 0.0)] * 2),
    ],
    ids=["collapsed-face", "collinear-face", "collapsed-edge", "collapsed-edge-pair"],
)
def test_degenerate_routes_cannot_certify_collision_free(
    certified: bool,
    kind: ContactStencilKind,
    vertices: list[tuple[float, float, float]],
) -> None:
    scene = _scene()
    start = jnp.asarray(vertices, dtype=jnp.float64)
    epoch = DenseContactSearchPlan(
        edge_vertex_capacity=1,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=2.0,
    ).build(scene, start)
    indices = (
        [[0, 1, 2, -1]] if kind == ContactStencilKind.EDGE_VERTEX else [[0, 1, 2, 3]]
    )
    batch = ContactStencilBatch(
        kind,
        np.array(indices, dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        capacity=1,
    )
    epoch = eqx.tree_at(lambda value: value.edge_vertex, epoch, batch)
    plan = CertifiedAABBCCDPlan() if certified else InclusionCCDPlan()
    result = eqx.filter_jit(collision_free_step_limit)(plan, scene, epoch, start, start)
    assert int(result.query_count) == 1
    assert int(result.initial_violations) == 1
    assert int(result.status) == int(CCDStatus.INITIAL_DISTANCE_VIOLATION)
    assert not bool(result.successful)
    assert not bool(result.guarantee.successful)
    assert float(result.step_size) == 0.0


@pytest.mark.parametrize("distance", [0.0, 1.0], ids=["coincident", "separated"])
def test_point_point_routes_retain_true_separation(distance: float) -> None:
    scene = _scene()
    start = jnp.zeros((4, 3), dtype=jnp.float64).at[0, 2].set(distance)
    epoch = DenseContactSearchPlan(
        edge_vertex_capacity=1,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=2.0,
    ).build(scene, start)
    batch = ContactStencilBatch(
        ContactStencilKind.VERTEX_VERTEX,
        np.array([[0, 1, -1, -1]], dtype=np.int32),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        capacity=1,
    )
    epoch = eqx.tree_at(lambda value: value.edge_vertex, epoch, batch)
    result = collision_free_step_limit(InclusionCCDPlan(), scene, epoch, start, start)
    assert bool(result.successful) == (distance > 0.0)
    assert float(result.step_size) == (1.0 if distance > 0.0 else 0.0)


def test_isolated_vertex_sweep_returns_complete_empty_fixed_capacity_epoch() -> None:
    scene = _scene()
    positions = jnp.zeros((4, 3), dtype=jnp.float64)
    search = SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=2,
        edge_edge_capacity=3,
        face_vertex_capacity=4,
        activation_distance=2.0,
    )
    epoch = search.build(scene, positions)
    assert bool(epoch.successful)
    assert bool(epoch.complete)
    assert int(epoch.candidate_count) == 0
    for batch, capacity in zip(epoch.active_batches, (2, 3, 4), strict=True):
        assert batch.capacity == capacity
        assert batch.vertex_indices.shape == (capacity, 4)
        assert int(batch.actual_count) == 0
        assert int(batch.overflow_count) == 0
        assert not bool(batch.overflow)
        assert not bool(jnp.any(batch.valid))
