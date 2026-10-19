#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from tests._support.facet_graphs import finite_element, finite_volume


TRIANGLES = ((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4))

# Retain the float64 route-parity tolerance from this test at d2dfcb941.
ROUTE_PARITY_TOLERANCE = 1e-13


def test_facet_adjacency_contracts() -> None:
    volume = finite_volume(np.asarray(TRIANGLES, dtype=np.int32))
    fv = phx.graph.facet_adjacency(volume)
    fe = phx.graph.facet_adjacency(finite_element(volume.mesh))
    graph = phx.graph.GraphIR.from_edge_relation(fv.relation, nodes=volume.cell_centers)
    view = graph.edge_relation()

    assert fv.topology_id == fe.topology_id
    assert fv.relation.capacity == volume.owner_cells.shape[0]
    assert bool(jnp.all(fe.relation.valid))
    np.testing.assert_array_equal(fv.relation.valid, volume.neighbor_cells >= 0)
    np.testing.assert_array_equal(view.source_indices, fv.relation.source_indices)
    np.testing.assert_array_equal(view.target_indices, fv.relation.target_indices)
    np.testing.assert_array_equal(view.valid, fv.relation.valid)
    np.testing.assert_array_equal(graph.edge_mask, fv.relation.valid)

    flipped = finite_volume(
        np.asarray(((0, 1, 3), (1, 4, 3), (1, 2, 5), (1, 5, 4)), dtype=np.int32)
    )
    assert phx.graph.facet_adjacency(flipped).topology_id != fv.topology_id
    np.testing.assert_array_equal(volume.triangles, np.asarray(TRIANGLES, dtype=np.int32))


@pytest.mark.parametrize(
    ("facets", "owners", "neighbors", "error", "field"),
    (
        ([0, 1], [0, 1], [1, 1], ValueError, "neighbor_cells"),
        ([0, 1], [0, 1], [1, -2], ValueError, "neighbor_cells"),
        ([0, 0], [0, 1], [1, -1], ValueError, "facet_ids"),
        ([0, 1], [0, 3], [1, -1], ValueError, "owner_cells"),
        ([0.0, 1.0], [0, 1], [1, -1], TypeError, "facet_ids"),
    ),
    ids=(
        "self-route",
        "invalid-sentinel",
        "repeated-facet",
        "invalid-owner",
        "real-facet",
    ),
)
def test_facet_adjacency_refuses_invalid_routes(
    facets: list[int] | list[float],
    owners: list[int],
    neighbors: list[int],
    error: type[Exception],
    field: str,
) -> None:
    with pytest.raises(error, match=field):
        phx.graph.FacetAdjacency(
            np.asarray(facets),
            np.asarray(owners),
            np.asarray(neighbors),
            cell_count=2,
            cell_entity_set_id="cells",
            facet_entity_set_id="facets",
        )


def test_facet_adjacency_refuses_unsupported_discretization() -> None:
    with pytest.raises(TypeError, match="discretization"):
        # ty: ignore[invalid-argument-type]
        phx.graph.facet_adjacency(object())


@pytest.mark.parametrize(
    ("cell_count", "error"),
    ((-1, ValueError), (True, TypeError)),
    ids=("negative-count", "boolean-count"),
)
def test_facet_adjacency_refuses_invalid_cell_count(
    cell_count: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="FacetAdjacency.cell_count"):
        phx.graph.FacetAdjacency(
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([-1], dtype=np.int32),
            cell_count=cell_count,
            cell_entity_set_id="cells",
            facet_entity_set_id="facets",
        )


def test_physical_residual_and_mesh_graph_net_share_inert_boundary_routes() -> None:
    volume = finite_volume(np.asarray(TRIANGLES, dtype=np.int32))
    adjacency = phx.graph.facet_adjacency(volume)
    owners = np.asarray(volume.owner_cells)
    neighbors = np.asarray(volume.neighbor_cells)
    boundary = neighbors < 0
    flux = np.random.default_rng(0).normal(size=owners.shape)
    perturbed = np.where(boundary, flux + 100.0, flux)

    expected = np.zeros((volume.cell_count,))
    for face in np.flatnonzero(~boundary):
        expected[owners[face]] -= flux[face]
        expected[neighbors[face]] += flux[face]

    divergence = phx.graph.GraphFiniteVolumeDivergence(
        flux_key="flux", output_key="residual", normalize_by_volume=False
    )
    face_features = jnp.concatenate(
        [volume.face_centers, volume.face_measures[:, None]], axis=-1
    )
    model = phx.graph.MeshGraphNet(
        node_in_size=2,
        edge_in_size=3,
        node_out_size=1,
        latent_size=4,
        processor_steps=2,
        key=jr.key(0),
    )

    def residual(values: Any) -> Any:
        graph = phx.graph.GraphIR.from_edge_relation(
            adjacency.relation,
            nodes={"centers": volume.cell_centers},
            edges={"flux": jnp.asarray(values)},
        )
        return divergence(graph).nodes["residual"]

    def prediction(edges: Any) -> Any:
        graph = phx.graph.GraphIR.from_edge_relation(
            adjacency.relation, nodes=volume.cell_centers, edges=edges
        )
        return model(graph).nodes

    np.testing.assert_allclose(
        residual(flux), expected, rtol=ROUTE_PARITY_TOLERANCE, atol=ROUTE_PARITY_TOLERANCE
    )
    np.testing.assert_allclose(
        residual(perturbed),
        expected,
        rtol=ROUTE_PARITY_TOLERANCE,
        atol=ROUTE_PARITY_TOLERANCE,
    )
    boundary_rows = jnp.asarray(boundary)[:, None]
    np.testing.assert_allclose(
        prediction(jnp.where(boundary_rows, face_features + 50.0, face_features)),
        prediction(face_features),
        rtol=ROUTE_PARITY_TOLERANCE,
        atol=ROUTE_PARITY_TOLERANCE,
    )
