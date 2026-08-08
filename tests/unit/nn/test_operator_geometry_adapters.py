#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest
import trimesh

import phydrax as phx


def test_geometry_sampling_preserves_interior_and_boundary_measures():
    geometry = phx.domain.GeometryDomain(phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile())

    interior = phx.nn.operator.function_samples_from_geometry(
        geometry,
        12,
        component="interior",
        key=jr.key(0),
    )
    boundary = phx.nn.operator.function_samples_from_geometry(
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


def test_canonical_triangle_mesh_builds_graph_and_simplicial_operator_topologies():
    mesh = phx.geometry.TriangleMesh(
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.asarray([[0, 1, 2], [0, 2, 3]]),
    )
    values = jnp.sum(mesh.vertices[:, :2], axis=-1)

    graph_samples = phx.nn.operator.function_samples_from_mesh(
        mesh,
        values=values,
        topology_kind="graph",
    )
    simplicial_samples = phx.nn.operator.function_samples_from_mesh(
        mesh,
        values=values,
        topology_kind="simplicial",
    )

    assert graph_samples.coordinates.shape[-1] == 3
    assert graph_samples.topology.kind == "graph"
    assert simplicial_samples.topology.kind == "simplicial"
    assert jnp.sum(graph_samples.quadrature()) == pytest.approx(float(mesh.measure))
    assert jnp.sum(simplicial_samples.quadrature()) == pytest.approx(float(mesh.measure))
    assert jnp.array_equal(
        simplicial_samples.topology.sample_entities,
        jnp.arange(values.shape[0]),
    )


def test_mesh_region_uses_surface_vertex_measure():
    host_mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    region = phx.geometry.mesh_region_from_source(host_mesh, recenter=False)

    samples = phx.nn.operator.function_samples_from_mesh(region)

    assert samples.coordinates.shape == (region.vertices.shape[0], 3)
    assert jnp.sum(samples.quadrature()) == pytest.approx(float(host_mesh.area))
    assert samples.topology.site == "vertex"


def test_batched_point_cloud_adapter_compacts_masked_graph_nodes():
    coordinates = jnp.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
        ]
    )
    mask = jnp.asarray([[True, True, True, False], [True, True, True, True]])
    values = jnp.arange(8.0).reshape((2, 4))

    samples = phx.nn.operator.function_samples_from_point_cloud(
        coordinates,
        values=values,
        mask=mask,
        quadrature_weights=jnp.ones((2, 4)),
        k=2,
    )
    graph = phx.nn.operator.operator_graph_from_samples(samples, case_shape=(2,))

    assert samples.geometry_case_shape == (2,)
    assert samples.topology.case_shape == (2,)
    assert jnp.array_equal(samples.topology.graph.n_node, jnp.asarray([3, 4]))
    assert samples.topology.sample_entities[0, -1] == -1
    assert graph.nodes["features"].shape[0] == 7
    assert jnp.array_equal(
        graph.nodes["features"],
        jnp.concatenate((values[0, :3], values[1])),
    )
