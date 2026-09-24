#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


VERTICES = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)


def _finite_volume(triangles):
    return phx.discretization.UnstructuredFiniteVolumePlan(
        VERTICES, triangles=np.asarray(triangles, dtype=np.int32)
    ).prepare()


def _finite_element(mesh):
    element = phx.discretization.lagrange_element("triangle", 1)
    return phx.discretization.FiniteElementPlan(
        mesh, (phx.discretization.FiniteElementFieldSpec("u", element),)
    ).prepare()


TRIANGLES = ((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4))


def test_facet_adjacency_identity_is_shared_by_finite_volume_finite_element_and_graph():
    finite_volume = _finite_volume(TRIANGLES)
    fv = phx.graph.facet_adjacency(finite_volume)
    fe = phx.graph.facet_adjacency(_finite_element(finite_volume.mesh))
    graph = phx.graph.GraphIR.from_edge_relation(
        fv.relation, nodes=finite_volume.cell_centers
    )
    view = graph.edge_relation()

    assert fv.topology_id == fe.topology_id
    assert fv.relation.capacity == finite_volume.owner_cells.shape[0]
    assert bool(jnp.all(fe.relation.valid))
    np.testing.assert_array_equal(fv.relation.valid, finite_volume.neighbor_cells >= 0)
    np.testing.assert_array_equal(view.source_indices, fv.relation.source_indices)
    np.testing.assert_array_equal(view.target_indices, fv.relation.target_indices)
    np.testing.assert_array_equal(view.valid, fv.relation.valid)
    np.testing.assert_array_equal(graph.edge_mask, fv.relation.valid)

    flipped = _finite_volume(((0, 1, 3), (1, 4, 3), (1, 2, 5), (1, 5, 4)))
    assert phx.graph.facet_adjacency(flipped).topology_id != fv.topology_id


def test_physical_residual_and_mesh_graph_net_share_inert_boundary_routes():
    finite_volume = _finite_volume(TRIANGLES)
    adjacency = phx.graph.facet_adjacency(finite_volume)
    owners = np.asarray(finite_volume.owner_cells)
    neighbors = np.asarray(finite_volume.neighbor_cells)
    boundary = neighbors < 0
    flux = np.random.default_rng(0).normal(size=owners.shape)
    perturbed = np.where(boundary, flux + 100.0, flux)

    expected = np.zeros((finite_volume.cell_count,))
    for face in np.flatnonzero(~boundary):
        expected[owners[face]] -= flux[face]
        expected[neighbors[face]] += flux[face]

    divergence = phx.graph.GraphFiniteVolumeDivergence(
        flux_key="flux", output_key="residual", normalize_by_volume=False
    )
    face_features = jnp.concatenate(
        [finite_volume.face_centers, finite_volume.face_measures[:, None]], axis=-1
    )
    model = phx.graph.MeshGraphNet(
        node_in_size=2,
        edge_in_size=3,
        node_out_size=1,
        latent_size=4,
        processor_steps=2,
        key=jr.key(0),
    )

    def residual(values):
        graph = phx.graph.GraphIR.from_edge_relation(
            adjacency.relation,
            nodes={"centers": finite_volume.cell_centers},
            edges={"flux": jnp.asarray(values)},
        )
        return divergence(graph).nodes["residual"]

    def prediction(edges):
        graph = phx.graph.GraphIR.from_edge_relation(
            adjacency.relation, nodes=finite_volume.cell_centers, edges=edges
        )
        return model(graph).nodes

    np.testing.assert_allclose(residual(flux), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(residual(perturbed), expected, rtol=1e-13, atol=1e-13)
    boundary_rows = jnp.asarray(boundary)[:, None]
    np.testing.assert_allclose(
        prediction(jnp.where(boundary_rows, face_features + 50.0, face_features)),
        prediction(face_features),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    ("facets", "owners", "neighbors", "error", "match"),
    [
        ([0, 1], [0, 1], [1, 1], ValueError, "itself"),
        ([0, 1], [0, 1], [1, -2], ValueError, "sentinel"),
        ([0, 0], [0, 1], [1, -1], ValueError, "unique"),
        ([0, 1], [0, 3], [1, -1], ValueError, "Owner"),
        ([0.0, 1.0], [0, 1], [1, -1], TypeError, "integer"),
    ],
)
def test_facet_adjacency_rejects_inconsistent_routes(
    facets, owners, neighbors, error, match
):
    with pytest.raises(error, match=match):
        phx.graph.FacetAdjacency(
            np.asarray(facets),
            np.asarray(owners),
            np.asarray(neighbors),
            cell_count=2,
            cell_entity_set_id="cells",
            facet_entity_set_id="facets",
        )


def test_facet_adjacency_requires_a_mesh_discretization():
    with pytest.raises(TypeError, match="facet_adjacency"):
        phx.graph.facet_adjacency(object())
