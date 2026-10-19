#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


def _triangle() -> phx.discretization.CellComplexTopology:
    return phx.discretization.polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)


def _su2_links(space: phx.graph.MatrixGaugeLinkSpace) -> Array:
    coordinates = jnp.asarray(
        [[0.21, -0.08, 0.13], [-0.17, 0.09, 0.04], [0.06, 0.14, -0.11]]
    )
    return jax.vmap(lambda value: space.group.exp(space.group.hat(value)))(coordinates)


def test_matrix_gauge_scenario_1() -> None:
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
    original_open = phx.discretization.ordered_path_transport(open_paths, links)[0]
    transformed_open = phx.discretization.ordered_path_transport(open_paths, transformed)[
        0
    ]
    expected = space.group.compose(
        space.group.compose(
            vertices[open_paths.start_vertices[0]],
            original_open,
        ),
        space.group.inverse(vertices[open_paths.end_vertices[0]]),
    )
    assert jnp.allclose(transformed_open, expected, atol=1e-10)
    topology = _triangle()
    boundary = phx.discretization.prepare_cell_boundary_paths(topology)
    reverse = phx.discretization.reverse_oriented_paths(boundary.paths)
    space = phx.graph.MatrixGaugeLinkSpace(
        topology,
        phx.metrix.SpecialUnitaryGroup(2),
    )
    links = _su2_links(space)
    forward_value = phx.discretization.ordered_path_transport(boundary.paths, links)
    reverse_value = phx.discretization.ordered_path_transport(reverse, links)

    assert jnp.allclose(reverse_value, space.group.inverse(forward_value), atol=1e-10)


def test_matrix_gauge_scenario_2() -> None:
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
    topology = _triangle()
    boundary = phx.discretization.prepare_cell_boundary_paths(topology).paths
    space = phx.graph.MatrixGaugeLinkSpace(
        topology,
        phx.metrix.SpecialUnitaryGroup(2),
    )
    links = _su2_links(space)
    ordered = phx.discretization.ordered_path_transport(boundary, links)[0]
    edge_order = boundary.edge_indices[:, ::-1]
    sign_order = boundary.orientations[:, ::-1]
    reordered = jnp.eye(2, dtype=links.dtype)
    for edge, sign in zip(edge_order[0], sign_order[0], strict=True):
        factor = links[edge] if sign > 0 else space.group.inverse(links[edge])
        reordered = reordered @ factor

    assert not jnp.allclose(ordered, reordered)


@pytest.mark.parametrize("dimension", (2, 5), ids=("small-native", "native-lu"))
def test_special_unitary_membership_refuses_unit_determinant_violation(
    dimension: int,
) -> None:
    topology = _triangle()
    links = (
        jnp.broadcast_to(
            jnp.eye(dimension, dtype=jnp.complex128), (3, dimension, dimension)
        )
        .at[:, 0, 0]
        .set(-1.0)
    )
    unitary = phx.graph.MatrixGaugeLinkSpace(topology, phx.metrix.UnitaryGroup(dimension))
    special = phx.graph.MatrixGaugeLinkSpace(
        topology, phx.metrix.SpecialUnitaryGroup(dimension)
    )
    assert unitary.contains(links)
    assert not special.contains(links)


@pytest.mark.parametrize(
    ("ulps", "accepted"),
    ((4, True), (128, False)),
    ids=("below-membership-floor", "above-membership-floor"),
)
def test_special_unitary_membership_preserves_dtype_tolerance_floor(
    ulps: int,
    accepted: bool,
) -> None:
    # The declared tolerance is below one dtype ulp, so 64*eps controls admission.
    declared_tolerance = float(np.finfo(np.float64).eps / 1024)
    space = phx.graph.MatrixGaugeLinkSpace(
        _triangle(), phx.metrix.SpecialUnitaryGroup(2, tolerance=declared_tolerance)
    )
    # A diagonal phase residual of ulps*eps brackets the preserved 64*eps
    # membership floor; its Gram residual is only O(eps**2).
    residual = ulps * jnp.finfo(jnp.float64).eps
    links = space.identity().at[:, 0, 0].set(1.0 + 1j * residual)
    assert bool(space.contains(links)) is accepted


def test_native_determinant_retains_singular_jvp() -> None:
    matrix = jnp.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)
    tangent = jnp.asarray([[0.0, 0.0], [0.0, 3.0]], dtype=jnp.float64)
    value, derivative = jax.jvp(phx.linalg.determinant, (matrix,), (tangent,))
    assert value == 0.0
    assert derivative == 6.0


def test_native_determinant_retains_singular_holomorphic_gradient() -> None:
    matrix = jnp.asarray([[1.0 + 2.0j, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
    gradient = jax.grad(phx.linalg.determinant, holomorphic=True)(matrix)
    np.testing.assert_array_equal(
        gradient, np.asarray([[0.0, 0.0], [0.0, 1.0 + 2.0j]], dtype=np.complex128)
    )
