#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization import (
    oriented_edge_endpoints,
    polygonal_cell_complex,
    prepare_cell_boundary_paths,
    TensorTopology,
)
from phydrax.discretization._lattice_boundary import LatticeBoundaryPhasePlan
from phydrax.graph import gauge_transform_links, MatrixGaugeLinkSpace, path_holonomy
from phydrax.graph._gauge_transport import GaugeCovariantShiftPlan, GaugeStaplePlan
from phydrax.metrix import SpecialUnitaryGroup
from phydrax.metrix._gauge_representation import FundamentalGaugeRepresentation


def _cycle_shift_plan():
    topology = polygonal_cell_complex(None, jnp.asarray([[0, 1, 2, 3]]), 4)
    space = MatrixGaugeLinkSpace(topology, SpecialUnitaryGroup(2))
    tails, heads = (np.asarray(value) for value in oriented_edge_endpoints(topology))
    forward_sites = np.asarray([[1], [2], [3], [0]], dtype=np.int32)
    forward_edges = np.empty((4, 1), dtype=np.int32)
    orientations = np.empty((4, 1), dtype=np.int32)
    for site in range(4):
        neighbor = int(forward_sites[site, 0])
        positive = np.flatnonzero((tails == site) & (heads == neighbor))
        negative = np.flatnonzero((heads == site) & (tails == neighbor))
        if positive.size:
            forward_edges[site, 0] = int(positive[0])
            orientations[site, 0] = 1
        else:
            forward_edges[site, 0] = int(negative[0])
            orientations[site, 0] = -1
    boundary = LatticeBoundaryPhasePlan(
        TensorTopology(("x",), (4,), periodic=(True,)),
        jnp.asarray([-1.0 + 0.0j]),
    )
    representation = FundamentalGaugeRepresentation(space.group)
    return GaugeCovariantShiftPlan(
        space,
        representation,
        forward_sites,
        forward_edges,
        orientations,
        boundary,
    )


def test_covariant_forward_and_backward_shifts_transform_at_destination_site():
    plan = _cycle_shift_plan()
    group = plan.link_space.group
    link_coordinates = 0.05 * jnp.arange(12.0).reshape((4, 3))
    links = jax.vmap(lambda value: group.exp(group.hat(value)))(link_coordinates)
    field = jnp.asarray([[1.0, 0.2j], [0.3, 0.7], [-0.2j, 0.4], [0.1, -0.5j]])
    gauge_coordinates = 0.03 * (jnp.arange(12.0).reshape((4, 3)) - 5.0)
    vertices = jax.vmap(lambda value: group.exp(group.hat(value)))(gauge_coordinates)
    transformed_links = gauge_transform_links(plan.link_space, links, vertices)
    transformed_field = plan.representation.apply(vertices, field)

    forward = plan.forward(links, field)
    backward = plan.backward(links, field)
    assert jnp.allclose(
        plan.forward(transformed_links, transformed_field),
        plan.representation.apply(vertices[:, None], forward),
        atol=2e-6,
    )
    assert jnp.allclose(
        plan.backward(transformed_links, transformed_field),
        plan.representation.apply(vertices[:, None], backward),
        atol=2e-6,
    )
    assert jnp.allclose(
        jax.jit(lambda link_values, matter: plan.forward(link_values, matter))(
            links, field
        ),
        forward,
    )


def test_staple_is_the_ordered_complement_of_each_triangle_link():
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    boundaries = prepare_cell_boundary_paths(topology)
    space = MatrixGaugeLinkSpace(topology, SpecialUnitaryGroup(2))
    coordinates = jnp.asarray(
        [[0.1, -0.03, 0.04], [-0.07, 0.02, 0.05], [0.03, 0.08, -0.02]]
    )
    links = jax.vmap(lambda value: space.group.exp(space.group.hat(value)))(coordinates)
    plan = GaugeStaplePlan(space, boundaries)
    loop = path_holonomy(space, links, boundaries.paths)[0]

    assert plan.staples(links).shape == links.shape
    for edge in range(space.num_edges):
        reconstructed_real_trace = jnp.real(
            jnp.trace(links[edge] @ plan.staple(links, edge))
        )
        assert jnp.allclose(
            reconstructed_real_trace, jnp.real(jnp.trace(loop)), atol=1e-6
        )
