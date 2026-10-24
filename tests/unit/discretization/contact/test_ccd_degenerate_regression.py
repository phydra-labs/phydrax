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
from phydrax.discretization.contact._stencils import ContactPrimitiveProfile
from phydrax.discretization.contact._surface import (
    CollisionSurfacePlan,
    PreparedCollisionScene,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from phydrax.linalg import ArraySpace


def _scene(
    vertex_count: int,
    *,
    edges: list[tuple[int, int]] | None = None,
    faces: list[tuple[int, int, int]] | None = None,
) -> PreparedCollisionScene:
    space = ArraySpace((vertex_count, 3), dtype=np.float64)
    topology = CollisionSurfacePlan(
        np.arange(vertex_count, dtype=np.int64),
        ambient_dimension=3,
        edges=None if edges is None else np.asarray(edges, dtype=np.int32),
        faces=None if faces is None else np.asarray(faces, dtype=np.int32),
        body_ids=0,
        allow_isolated_vertices=True,
    )
    # A valid nondegenerate rest surface: vertices on the twisted cubic
    # (i, i^2, i^3) are distinct and no three are collinear. Degeneracy is
    # introduced only by the current positions passed to search and CCD.
    index = np.arange(vertex_count, dtype=np.float64)
    rest = np.stack((index, index**2, index**3), axis=-1)
    surface = PreparedCollisionSurface(
        topology,
        jnp.asarray(rest),
        selection_collision_operator(space, np.arange(vertex_count, dtype=np.int32)),
    )
    return PreparedCollisionScene((surface,))


type _Case = tuple[
    list[tuple[int, int]] | None,
    list[tuple[int, int, int]] | None,
    list[tuple[float, float, float]],
    ContactPrimitiveProfile,
]

_DEGENERATE_CASES: dict[str, _Case] = {
    "collapsed-face": (
        None,
        [(1, 2, 3)],
        [(0.0, 0.0, 1.0)] + [(0.0, 0.0, 0.0)] * 3,
        ContactPrimitiveProfile.MESH,
    ),
    "collinear-face": (
        None,
        [(1, 2, 3)],
        [(0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        ContactPrimitiveProfile.MESH,
    ),
    "collapsed-edge": (
        [(1, 2)],
        None,
        [(0.0, 0.0, 1.0)] + [(0.0, 0.0, 0.0)] * 2,
        ContactPrimitiveProfile.MESH,
    ),
    "collapsed-edge-pair": (
        [(0, 1), (2, 3)],
        None,
        [(0.0, 0.0, 1.0)] * 2 + [(0.0, 0.0, 0.0)] * 2,
        ContactPrimitiveProfile.SPHERE_CAPSULE,
    ),
}


@pytest.mark.parametrize("certified", [False, True], ids=["inclusion", "rounding-aabb"])
@pytest.mark.parametrize("case", sorted(_DEGENERATE_CASES))
def test_searched_degenerate_routes_cannot_certify_collision_free(
    certified: bool,
    case: str,
) -> None:
    edges, faces, vertices, profile = _DEGENERATE_CASES[case]
    start = jnp.asarray(vertices, dtype=jnp.float64)
    scene = _scene(start.shape[0], edges=edges, faces=faces)
    epoch = DenseContactSearchPlan(
        edge_vertex_capacity=4,
        edge_edge_capacity=4,
        face_vertex_capacity=4,
        activation_distance=2.0,
        primitive_profile=profile,
    ).build(scene, start)
    assert bool(epoch.successful)
    assert int(epoch.candidate_count) >= 1
    plan = CertifiedAABBCCDPlan() if certified else InclusionCCDPlan()
    result = eqx.filter_jit(collision_free_step_limit)(plan, scene, epoch, start, start)
    assert int(result.initial_violations) >= 1
    assert int(result.status) == int(CCDStatus.INITIAL_DISTANCE_VIOLATION)
    assert not bool(result.successful)
    assert not bool(result.guarantee.successful)
    assert float(result.step_size) == 0.0


@pytest.mark.parametrize("distance", [0.0, 1.0], ids=["coincident", "separated"])
def test_sphere_routes_retain_true_separation(distance: float) -> None:
    scene = _scene(2)
    start = jnp.zeros((2, 3), dtype=jnp.float64).at[0, 2].set(distance)
    epoch = DenseContactSearchPlan(
        edge_vertex_capacity=0,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        vertex_vertex_capacity=1,
        activation_distance=2.0,
        primitive_profile=ContactPrimitiveProfile.SPHERE_CAPSULE,
    ).build(scene, start)
    assert bool(epoch.successful)
    assert int(epoch.vertex_vertex.actual_count) == 1
    result = collision_free_step_limit(InclusionCCDPlan(), scene, epoch, start, start)
    assert bool(result.successful) == (distance > 0.0)
    assert float(result.step_size) == (1.0 if distance > 0.0 else 0.0)


def test_isolated_vertex_sweep_returns_complete_empty_fixed_capacity_epoch() -> None:
    scene = _scene(4)
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
