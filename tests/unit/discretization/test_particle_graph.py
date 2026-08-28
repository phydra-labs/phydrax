#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _compiled_pair():
    count = 8
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), spacing),
        ambient_dimension=1,
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    method = phx.discretization.BarotropicSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 * spacing,
    )
    problem = phx.equations.BarotropicFluidProblemIR(
        "graph-fluid",
        phx.equations.TaitBarotropicMaterial(1.0, 1.0),
    )
    dense = phx.equations.compile_barotropic_sph_problem(
        problem,
        particles,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2,
            box=box,
        ),
    )
    cell = phx.equations.compile_barotropic_sph_problem(
        problem,
        particles,
        method,
        neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
            2.5 * spacing,
            4,
            24,
            box,
        ),
    )
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    position = position + 0.01 * spacing * jnp.sin(2.0 * jnp.pi * position)
    return dense, cell, position


def _edge_ids(graph):
    mask = np.asarray(graph.edge_mask, dtype=bool)
    left = np.asarray(graph.edges["left_particle_id"])[mask, 0]
    right = np.asarray(graph.edges["right_particle_id"])[mask, 0]
    return set(zip(left.tolist(), right.tolist(), strict=True))


def test_dense_and_cell_physical_graphs_have_identical_undirected_edges():
    dense, cell, position = _compiled_pair()
    dense_graph = dense.dynamics.graph_view(position, directed=False)
    cell_graph = cell.dynamics.graph_view(position, directed=False)

    assert _edge_ids(dense_graph) == _edge_ids(cell_graph)
    assert int(jnp.sum(dense_graph.edge_mask)) == int(jnp.sum(cell_graph.edge_mask))
    assert jnp.array_equal(
        cell_graph.nodes["particle_id"][:, 0], jnp.arange(position.shape[0])
    )
    assert jnp.allclose(cell_graph.nodes["position"], position)
    relation = cell_graph.edge_relation(node_count=position.shape[0])
    assert jnp.array_equal(relation.source_indices, cell_graph.senders)
    assert jnp.array_equal(relation.target_indices, cell_graph.receivers)
    assert jnp.array_equal(relation.valid, cell_graph.edge_mask)


def test_directed_particle_graph_duplicates_and_reverses_every_route():
    _, cell, position = _compiled_pair()
    graph = cell.dynamics.graph_view(position, directed=True)
    half = graph.senders.shape[0] // 2

    assert graph.n_edge[0] == 2 * cell.dynamics.neighborhood.pair_capacity
    assert jnp.array_equal(graph.senders[:half], graph.receivers[half:])
    assert jnp.array_equal(graph.receivers[:half], graph.senders[half:])
    assert jnp.array_equal(graph.edge_mask[:half], graph.edge_mask[half:])
    assert jnp.allclose(
        graph.edges["displacement"][:half],
        -graph.edges["displacement"][half:],
    )
    assert jnp.allclose(graph.edges["distance"][:half], graph.edges["distance"][half:])


def test_graph_view_refuses_an_overflowed_relation():
    count = 4
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.CellListParticleNeighborhoodPlan(
        0.4,
        1,
        2,
        box,
    ).prepare(particles)
    position = jnp.asarray([[0.10], [0.11], [0.12], [0.13]])
    state = prepared.build(position)

    assert not state.successful
    with pytest.raises(Exception, match="neighborhood construction failed"):
        phx.discretization.particle_graph_view(
            particles,
            state,
            position,
        ).nodes["position"].block_until_ready()
