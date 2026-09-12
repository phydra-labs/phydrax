#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _triangle():
    return phx.discretization.polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)


def _su2_links(space):
    coordinates = jnp.asarray(
        [[0.21, -0.08, 0.13], [-0.17, 0.09, 0.04], [0.06, 0.14, -0.11]]
    )
    return jax.vmap(lambda value: space.group.exp(space.group.hat(value)))(coordinates)


def test_cell_boundary_paths_are_ordered_and_incidence_certified():
    topology = _triangle()
    boundaries = phx.discretization.prepare_cell_boundary_paths(topology)
    paths = boundaries.paths
    tails, heads = phx.discretization.oriented_edge_endpoints(topology)
    length = int(jnp.sum(paths.valid[0]))

    current = int(paths.start_vertices[0])
    for edge, sign in zip(
        paths.edge_indices[0, :length],
        paths.orientations[0, :length],
        strict=True,
    ):
        edge_index = int(edge)
        start = int(tails[edge_index] if sign > 0 else heads[edge_index])
        end = int(heads[edge_index] if sign > 0 else tails[edge_index])
        assert start == current
        current = end
    assert current == int(paths.start_vertices[0])


def test_matrix_holonomy_is_covariant_and_closed_trace_invariant():
    topology = _triangle()
    boundary = phx.discretization.prepare_cell_boundary_paths(topology)
    space = phx.graph.MatrixGaugeLinkSpace(
        topology,
        phx.metrix.SpecialUnitaryGroup(2),
    )
    links = _su2_links(space)
    vertex_coordinates = jnp.asarray(
        [[0.12, 0.02, -0.03], [-0.07, 0.11, 0.04], [0.03, -0.09, 0.08]]
    )
    vertices = jax.vmap(lambda value: space.group.exp(space.group.hat(value)))(
        vertex_coordinates
    )
    transformed = phx.graph.gauge_transform_links(space, links, vertices)
    trace = phx.graph.closed_path_trace(space, links, boundary.paths)
    transformed_trace = phx.graph.closed_path_trace(space, transformed, boundary.paths)

    assert space.contains(links)
    assert space.contains(transformed)
    assert jnp.allclose(trace, transformed_trace, atol=1e-10)

    length = int(jnp.sum(boundary.paths.valid[0]))
    open_paths = phx.discretization.OrientedEdgePathPlan(
        topology,
        boundary.paths.edge_indices[:, : length - 1],
        boundary.paths.orientations[:, : length - 1],
        path_names=("open",),
    )
    original_open = phx.graph.path_holonomy(space, links, open_paths)[0]
    transformed_open = phx.graph.path_holonomy(space, transformed, open_paths)[0]
    expected = space.group.compose(
        space.group.compose(
            vertices[open_paths.start_vertices[0]],
            original_open,
        ),
        space.group.inverse(vertices[open_paths.end_vertices[0]]),
    )
    assert jnp.allclose(transformed_open, expected, atol=1e-10)


def test_reverse_path_holonomy_is_group_inverse():
    topology = _triangle()
    boundary = phx.discretization.prepare_cell_boundary_paths(topology)
    reverse = phx.discretization.reverse_oriented_paths(boundary.paths)
    space = phx.graph.MatrixGaugeLinkSpace(
        topology,
        phx.metrix.SpecialUnitaryGroup(2),
    )
    links = _su2_links(space)
    forward_value = phx.graph.path_holonomy(space, links, boundary.paths)
    reverse_value = phx.graph.path_holonomy(space, links, reverse)

    assert jnp.allclose(reverse_value, space.group.inverse(forward_value), atol=1e-10)


def test_path_plan_rejects_discontinuity_and_invalid_padding():
    topology = _triangle()
    boundary = phx.discretization.prepare_cell_boundary_paths(topology).paths
    edges = jnp.asarray(boundary.edge_indices)
    signs = jnp.asarray(boundary.orientations)
    bad_valid = jnp.asarray([[True, False, True]])

    with pytest.raises(ValueError, match="contiguous prefix"):
        phx.discretization.OrientedEdgePathPlan(
            topology,
            edges,
            signs,
            valid=bad_valid,
        )
    with pytest.raises(ValueError, match="share a vertex"):
        phx.discretization.OrientedEdgePathPlan(
            topology,
            edges.at[0, 1].set(edges[0, 0]),
            signs,
        )


def test_nonabelian_path_order_changes_holonomy():
    topology = _triangle()
    boundary = phx.discretization.prepare_cell_boundary_paths(topology).paths
    space = phx.graph.MatrixGaugeLinkSpace(
        topology,
        phx.metrix.SpecialUnitaryGroup(2),
    )
    links = _su2_links(space)
    ordered = phx.graph.path_holonomy(space, links, boundary)[0]
    edge_order = boundary.edge_indices[:, ::-1]
    sign_order = boundary.orientations[:, ::-1]
    reordered = jnp.eye(2, dtype=links.dtype)
    for edge, sign in zip(edge_order[0], sign_order[0], strict=True):
        factor = links[edge] if sign > 0 else space.group.inverse(links[edge])
        reordered = reordered @ factor

    assert not jnp.allclose(ordered, reordered)
