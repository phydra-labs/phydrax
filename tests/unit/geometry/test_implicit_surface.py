#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _grid(count=9):
    return phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))


def test_dual_surface_has_static_watertight_topology_and_dynamic_coordinates():
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


def test_dual_surface_derivative_and_refresh_status_are_explicit():
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

    def vertex_sum(radius):
        state = geometry.state.replace_at(radius_index, radius)
        return jnp.sum(plan.realize(state).proposed_vertices)

    derivative = jax.grad(vertex_sum)(jnp.asarray(0.75))
    expired = plan.realize(geometry.state.replace_at(radius_index, jnp.asarray(1.2)))

    assert jnp.isfinite(derivative)
    assert derivative != 0.0
    assert not bool(expired.accepted)
    assert bool(expired.refresh_required)
    assert jnp.array_equal(expired.vertices, plan.base_vertices)


def test_ambiguous_lattice_zero_fails_closed():
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
