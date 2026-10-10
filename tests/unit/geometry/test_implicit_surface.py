#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _grid(count: Any = 9) -> Any:
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))


def test_implicit_surface_scenario_1() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0),
        0.75,
        feature_id="sphere",
    ).compile()
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
    ).compile()

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
    ).compile()
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

    def vertex_sum(radius: Any) -> Any:
        state = geometry.state.replace_at(radius_index, radius)
        return jnp.sum(plan.realize(state).proposed_vertices)

    derivative = jax.grad(vertex_sum)(jnp.asarray(0.75))
    expired = plan.realize(geometry.state.replace_at(radius_index, jnp.asarray(1.2)))

    assert jnp.isfinite(derivative)
    assert derivative != 0.0
    assert not bool(expired.accepted)
    assert bool(expired.refresh_required)
    assert jnp.array_equal(expired.vertices, plan.base_vertices)


def _boxes(count: int) -> Any:
    edges = jnp.linspace(-1.4, 1.4, count + 1)
    lower = jnp.stack(
        jnp.meshgrid(edges[:-1], edges[:-1], edges[:-1], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    return jnp.stack((lower, lower + (edges[1] - edges[0])), axis=1)


_DOMAIN = jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]])


def test_established_cover_is_bound_to_source_state_and_domain() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.75, feature_id="sphere").compile()
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
    with pytest.raises(ValueError, match="stale"):
        cover.require_bound("sphere", phx.geometry.implicit_state_id(moved))
    # A Lipschitz bound encloses values but not gradients: no regularity claim.
    topology = phx.geometry.CertifiedImplicitTopology(
        cover, _triangle_topology(), premise="regular_value"
    )
    assert not topology.certified
    assert topology.unresolved_box_count == int(jnp.sum(cover.intersects))


def _triangle_topology() -> Any:
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), jnp.asarray([[0, 1, 2]])
    ).topology


def _user_cover(boxes: Any, **directional: Any) -> Any:
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
        **directional,
    )


@pytest.mark.parametrize(
    ("boxes", "complete"),
    [
        pytest.param(_boxes(2), True, id="tiling"),
        pytest.param(_boxes(2)[:-1], False, id="gap"),
        pytest.param(jnp.concatenate((_boxes(2), _boxes(2)[:1])), False, id="overlap"),
    ],
)
def test_cover_completeness_is_decided_exactly(boxes: Any, complete: bool) -> None:
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

    def certified(premise: Any) -> bool:
        return phx.geometry.CertifiedImplicitTopology(
            cover, _triangle_topology(), premise=premise
        ).certified

    # The x-gradient excludes zero, but the gradient hull admits orthogonal
    # normals, so small normal variation fails while regularity holds.
    assert certified("regular_value")
    assert not certified("small_normal_variation")
    assert certified("directional_monotone")
    with pytest.raises(ValueError, match="go together"):
        _user_cover(boxes, directions=directions)
