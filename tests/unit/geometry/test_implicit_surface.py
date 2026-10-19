#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.units import METER


def _grid(count: int = 9) -> phx.discretization.PreparedTensorGrid:
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))


def test_implicit_surface_scenario_1() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="sphere",
    ).compile(coordinate_unit=METER)
    policy = phx.geometry.ImplicitSurfacePolicy(
        projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
        maximum_intersection_pairs=500_000,
    )
    plan = phx.geometry.discover_implicit_surface(
        geometry,
        _grid(),
        policy=policy,
        source_id="sphere-surface",
    )
    base = plan.realize(geometry.state)
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))
    state = geometry.state.replace_at(radius_index, jnp.asarray(0.76))
    moved = eqx.filter_jit(plan.realize)(state)

    assert bool(base.accepted)
    assert bool(moved.accepted)
    assert moved.evidence.topology_id == base.evidence.topology_id
    assert jnp.array_equal(moved.faces, base.faces)
    assert not jnp.allclose(moved.vertices, base.vertices)
    mesh = moved.to_triangle_mesh()
    assert mesh.topology.watertight
    assert mesh.topology.num_face_components == 1
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.7,
        feature_id="sphere",
    ).compile(coordinate_unit=METER)

    with pytest.raises(ValueError, match="ambiguous zero"):
        phx.geometry.discover_implicit_surface(
            geometry,
            _grid(),
            source_id="ambiguous",
        )


def test_dual_surface_derivative_and_refresh_status_are_explicit() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="sphere",
    ).compile(coordinate_unit=METER)
    plan = phx.geometry.discover_implicit_surface(
        geometry,
        _grid(),
        policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=500_000,
        ),
        source_id="sphere-surface",
    )
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))

    def vertex_sum(radius: Array) -> Array:
        state = geometry.state.replace_at(radius_index, radius)
        return jnp.sum(plan.realize(state).proposed_vertices)

    derivative = jax.grad(vertex_sum)(jnp.asarray(0.75))
    expired = plan.realize(geometry.state.replace_at(radius_index, jnp.asarray(1.2)))

    assert jnp.isfinite(derivative)
    assert derivative != 0.0
    assert not bool(expired.accepted)
    assert bool(expired.refresh_required)
    assert jnp.array_equal(expired.vertices, plan.base_vertices)


def _boxes(count: int) -> Array:
    edges = jnp.linspace(-1.4, 1.4, count + 1)
    lower = jnp.stack(
        jnp.meshgrid(edges[:-1], edges[:-1], edges[:-1], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    return jnp.stack((lower, lower + (edges[1] - edges[0])), axis=1)


_DOMAIN = jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]])


def test_established_cover_is_bound_to_source_state_and_domain() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.75, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    cover = phx.geometry.establish_implicit_cover(
        geometry, _boxes(4), domain=_DOMAIN, source_id="sphere"
    )
    state = phx.geometry.implicit_state_id(geometry)
    radius_index = geometry.schema.index(phx.geometry.ParameterId("sphere", "radius"))
    moved = eqx.tree_at(
        lambda value: value.state,
        geometry,
        geometry.state.replace_at(radius_index, jnp.asarray(0.8)),
    )

    assert cover.bound_origin == "lipschitz_enclosure"
    assert cover.established
    assert cover.complete
    cover.require_bound("sphere", state)
    with pytest.raises(ValueError, match="CertifiedImplicitCover.binding"):
        cover.require_bound("sphere", phx.geometry.implicit_state_id(moved))
    # A Lipschitz bound encloses values but not gradients: no regularity claim.
    topology = phx.geometry.CertifiedImplicitTopology(
        cover, _triangle_topology(), premise="regular_value"
    )
    assert not topology.certified
    assert topology.unresolved_box_count == int(jnp.sum(cover.intersects))


def _triangle_topology() -> phx.discretization.CellComplexTopology:
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), jnp.asarray([[0, 1, 2]])
    ).topology


def _user_cover(
    boxes: Array,
    *,
    directions: ArrayLike | None = None,
    directional_lower: ArrayLike | None = None,
    directional_upper: ArrayLike | None = None,
) -> phx.geometry.CertifiedImplicitCover:
    count = boxes.shape[0]
    return phx.geometry.CertifiedImplicitCover(
        boxes,
        -jnp.ones((count,)),
        jnp.ones((count,)),
        jnp.tile(jnp.asarray([[0.5, -1.0, -1.0]]), (count, 1)),
        jnp.tile(jnp.asarray([[1.0, 1.0, 1.0]]), (count, 1)),
        domain=_DOMAIN,
        source_id="field",
        state_id="state",
        directions=directions,
        directional_lower=directional_lower,
        directional_upper=directional_upper,
    )


@pytest.mark.parametrize(
    ("boxes", "complete"),
    [
        pytest.param(_boxes(2), True, id="tiling"),
        pytest.param(_boxes(2)[:-1], False, id="gap"),
        pytest.param(jnp.concatenate((_boxes(2), _boxes(2)[:1])), False, id="overlap"),
    ],
)
def test_cover_completeness_is_decided_exactly(boxes: Array, complete: bool) -> None:
    cover = _user_cover(boxes)
    topology = phx.geometry.CertifiedImplicitTopology(
        cover, _triangle_topology(), premise="regular_value"
    )

    assert cover.bound_origin == "user_supplied"
    assert cover.complete == complete
    assert topology.certified == complete
    assert not topology.established


def test_topology_premises_are_checked_per_box() -> None:
    boxes = _boxes(2)
    count = boxes.shape[0]
    directions = jnp.asarray([[1.0, 1.0, 0.0]])
    cover = _user_cover(
        boxes,
        directions=directions,
        directional_lower=jnp.full((count, 1), 0.25),
        directional_upper=jnp.full((count, 1), 2.0),
    )

    def certified(premise: phx.geometry.ImplicitTopologyPremise) -> bool:
        return phx.geometry.CertifiedImplicitTopology(
            cover, _triangle_topology(), premise=premise
        ).certified

    # The x-gradient excludes zero, but the gradient hull admits orthogonal
    # normals, so small normal variation fails while regularity holds.
    assert certified("regular_value")
    assert not certified("small_normal_variation")
    assert certified("directional_monotone")
    with pytest.raises(ValueError, match="CertifiedImplicitCover.directional_bounds"):
        _user_cover(boxes, directions=directions)


def test_implicit_state_identity_refuses_wrong_geometry_kind() -> None:
    with pytest.raises(TypeError, match="geometry"):
        phx.geometry.implicit_state_id(None)  # ty: ignore[invalid-argument-type]


def test_implicit_topology_refuses_wrong_cover_kind() -> None:
    with pytest.raises(TypeError, match="cover"):
        phx.geometry.CertifiedImplicitTopology(
            None,  # ty: ignore[invalid-argument-type]
            _triangle_topology(),
            premise="regular_value",
        )


def test_stored_implicit_cover_preserves_established_origin_and_identity() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.75, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    cover = phx.geometry.establish_implicit_cover(
        geometry, _boxes(2), domain=_DOMAIN, source_id="sphere"
    )

    restored = phx.geometry.CertifiedImplicitCover.from_stored_fields(cover)

    assert restored.bound_origin == cover.bound_origin == "lipschitz_enclosure"
    assert restored.established
    assert restored.cover_id == cover.cover_id
    assert restored.source_id == cover.source_id
    assert restored.state_id == cover.state_id
    assert restored.complete == cover.complete
    assert restored.overlap_count == cover.overlap_count
    assert restored.uncovered_volume == cover.uncovered_volume
    assert restored.certified == cover.certified
    for stored, rebuilt in zip(
        jax.tree_util.tree_leaves(cover),
        jax.tree_util.tree_leaves(restored),
        strict=True,
    ):
        np.testing.assert_array_equal(stored, rebuilt)


def test_stored_implicit_cover_rejects_reversed_retained_bounds() -> None:
    cover = _user_cover(_boxes(2))
    corrupted = eqx.tree_at(
        lambda value: value.value_lower, cover, cover.value_upper + 1.0
    )

    with pytest.raises(ValueError, match=r"CertifiedImplicitCover\.bounds"):
        phx.geometry.CertifiedImplicitCover.from_stored_fields(corrupted)
