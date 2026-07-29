#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest
import trimesh

import phydrax as phx


def test_geometry_sampling_preserves_interior_and_boundary_measures():
    geometry = phx.domain.Square(center=(0.0, 0.0), side=2.0)

    interior = phx.nn.function_samples_from_geometry(
        geometry,
        12,
        component="interior",
        key=jr.key(0),
    )
    boundary = phx.nn.function_samples_from_geometry(
        geometry,
        10,
        component="boundary",
        key=jr.key(1),
    )

    assert interior.coordinates.shape == (12, 2)
    assert boundary.coordinates.shape == (10, 2)
    assert jnp.sum(interior.quadrature()) == pytest.approx(float(geometry.volume))
    assert jnp.sum(boundary.quadrature()) == pytest.approx(
        float(geometry.boundary_measure_value)
    )


def test_native_2d_mesh_builds_graph_and_simplicial_operator_topologies():
    geometry = phx.domain.Square(center=(0.0, 0.0), side=1.0)
    values = jnp.sum(geometry.mesh_vertices[:, :2], axis=-1)

    graph_samples = phx.nn.function_samples_from_mesh(
        geometry,
        values=values,
        topology_kind="graph",
    )
    simplicial_samples = phx.nn.function_samples_from_mesh(
        geometry,
        values=values,
        topology_kind="simplicial",
    )

    assert graph_samples.coordinates.shape[-1] == 2
    assert graph_samples.topology.kind == "graph"
    assert simplicial_samples.topology.kind == "simplicial"
    assert jnp.sum(graph_samples.quadrature()) == pytest.approx(float(geometry.area))
    assert jnp.sum(simplicial_samples.quadrature()) == pytest.approx(
        float(geometry.area)
    )
    assert jnp.array_equal(
        simplicial_samples.topology.sample_entities,
        jnp.arange(values.shape[0]),
    )


def test_native_3d_mesh_uses_surface_vertex_measure():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    geometry = phx.domain.Geometry3DFromCAD(mesh, recenter=False)

    samples = phx.nn.function_samples_from_mesh(geometry)

    assert samples.coordinates.shape == (geometry.mesh_vertices.shape[0], 3)
    assert jnp.sum(samples.quadrature()) == pytest.approx(
        float(geometry.boundary_measure_value)
    )
    assert samples.topology.site == "vertex"


def test_batched_point_cloud_adapter_compacts_masked_graph_nodes():
    coordinates = jnp.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
        ]
    )
    mask = jnp.asarray(
        [[True, True, True, False], [True, True, True, True]]
    )
    values = jnp.arange(8.0).reshape((2, 4))

    samples = phx.nn.function_samples_from_point_cloud(
        coordinates,
        values=values,
        mask=mask,
        quadrature_weights=jnp.ones((2, 4)),
        k=2,
    )
    graph = phx.nn.operator_graph_from_samples(samples, case_shape=(2,))

    assert samples.geometry_case_shape == (2,)
    assert samples.topology.case_shape == (2,)
    assert jnp.array_equal(samples.topology.graph.n_node, jnp.asarray([3, 4]))
    assert samples.topology.sample_entities[0, -1] == -1
    assert graph.nodes["features"].shape[0] == 7
    assert jnp.array_equal(
        graph.nodes["features"],
        jnp.concatenate((values[0, :3], values[1])),
    )
