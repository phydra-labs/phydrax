#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class _IncomingWeightedSource:
    def __call__(self, graph):
        nodes = graph.nodes if graph.nodes.ndim == 1 else graph.nodes[:, 0]
        messages = graph.edges * nodes[graph.senders]
        nodes = phx.graph.segment_sum(messages, graph.receivers, graph.num_nodes)
        return graph.replace(nodes=nodes, validate=False)


def _graphs() -> tuple[phx.graph.GraphIR, phx.graph.GraphIR]:
    graph0 = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0]]),
        edges=jnp.array([[10.0]]),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        globals=jnp.array([[0.0]]),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[2.0], [4.0], [8.0]]),
        edges=jnp.array([[20.0], [30.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        globals=jnp.array([[1.0]]),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    return graph0, graph1


def test_graph_dataset_domain_materializes_batched_node_entities():
    domain = phx.domain.GraphDatasetDomain(_graphs(), measure="count")
    batch = domain.points_from_indices(
        jnp.array([0, 1], dtype=jnp.int32),
        component=phx.domain.Nodes(),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    assert isinstance(batch, phx.domain.GraphBatch)
    assert batch.graph.num_nodes == 5
    assert batch.graph.num_edges == 3
    assert jnp.allclose(batch["graph"].data[:, 0], jnp.array([0.0, 1.0, 2.0, 4.0, 8.0]))
    assert jnp.allclose(
        batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data,
        jnp.arange(5, dtype=jnp.int32),
    )
    assert jnp.allclose(
        batch[phx.domain.graph.GRAPH_DATASET_INDEX_KEY].data,
        jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32),
    )
    assert jnp.allclose(
        domain.component({"graph": phx.domain.Nodes()}).mass.value,
        5.0,
    )


def test_graph_dataset_domain_applies_local_node_sets_per_graph():
    domain = phx.domain.GraphDatasetDomain(_graphs(), measure="count")
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.BoundaryNodes([1]),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    assert jnp.allclose(batch["graph"].data[:, 0], jnp.array([1.0, 4.0]))
    assert jnp.allclose(
        batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data,
        jnp.array([1, 3], dtype=jnp.int32),
    )
    assert jnp.allclose(
        domain.component({"graph": phx.domain.BoundaryNodes([1])}).mass.value,
        2.0,
    )


def test_graph_dataset_domain_graph_gradient_on_local_edge_set():
    domain = phx.domain.GraphDatasetDomain(_graphs())
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.EdgeSet([0]),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    @domain.Function("graph")
    def u(node):
        return node[0]

    grad = phx.operators.graph_gradient(u)
    assert jnp.allclose(grad(batch).data, jnp.array([1.0, 2.0]))


def test_graph_dataset_domain_samples_through_residual_penalty():
    domain = phx.domain.GraphDatasetDomain(_graphs())
    component = domain.component({"graph": phx.domain.EdgeSet([0])})
    structure = phx.domain.SampleLayout((("graph",),))

    @domain.Function("graph")
    def u(node):
        del node
        return 2.0

    condition = phx.conditions.Residual("u", component, phx.operators.graph_gradient)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(2, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    assert term.loss({"u": u}, key=jr.key(0)) < 1e-12


def test_graph_dataset_domain_graph_model_restricts_to_node_set():
    domain = phx.domain.GraphDatasetDomain(_graphs())
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.BoundaryNodes([1]),
        structure=phx.domain.SampleLayout((("graph",),)),
    )
    model = phx.graph.GraphMapFeatures(embed_node_fn=lambda nodes: nodes + 1.0)
    u = domain.GraphModel(model)

    assert jnp.allclose(u(batch).data[:, 0], jnp.array([2.0, 5.0]))


def test_graph_dataset_domain_graph_model_accepts_edge_input_fn():
    domain = phx.domain.GraphDatasetDomain(_graphs())
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.BoundaryNodes([1]),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    @domain.Function("graph")
    def u(node):
        return node[0]

    @domain.Function("graph")
    def k(edge):
        return 0.0 * edge[0] + 2.0

    model = domain.GraphModel(
        _IncomingWeightedSource(),
        input_fn=u,
        edge_input_fn=k,
    )

    assert jnp.allclose(model(batch).data, jnp.array([0.0, 4.0]))


def test_graph_dataset_domain_layout_packs_graph_but_exposes_real_entities():
    base = phx.domain.GraphDatasetDomain(_graphs())
    layout = base.layout_for_batch_size(2, multiple=2)
    domain = base.with_layout(layout)
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.Nodes(),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    assert domain.layout is not None
    assert batch.graph.nodes.shape == (6, 1)
    assert batch.graph.senders.shape == (4,)
    assert batch.graph.n_node.shape == (2,)
    assert jnp.allclose(
        batch.graph.node_mask, jnp.array([True, True, True, True, True, False])
    )
    assert jnp.allclose(batch.graph.edge_mask, jnp.array([True, True, True, False]))
    assert jnp.allclose(batch["graph"].data[:, 0], jnp.array([0.0, 1.0, 2.0, 4.0, 8.0]))
    assert jnp.allclose(
        batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data,
        jnp.arange(5, dtype=jnp.int32),
    )


def test_graph_dataset_domain_layout_preserves_graph_operator_results():
    base = phx.domain.GraphDatasetDomain(_graphs())
    domain = base.with_layout(base.layout_for_batch_size(2, multiple=2))
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.EdgeSet([0]),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    @domain.Function("graph")
    def u(node):
        return node[0]

    assert batch.graph.edge_mask.shape == (4,)
    assert jnp.allclose(
        phx.operators.graph_gradient(u)(batch).data, jnp.array([1.0, 2.0])
    )


def test_graph_dataset_domain_layout_preserves_graph_model_results():
    base = phx.domain.GraphDatasetDomain(_graphs())
    domain = base.with_layout(base.layout_for_batch_size(2, multiple=2))
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.BoundaryNodes([1]),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    class AddValidNodeMask:
        def __call__(self, graph):
            assert graph.node_mask is not None
            assert graph.nodes.shape == (6, 1)
            nodes = graph.nodes + graph.node_mask.astype(float)[:, None]
            return graph.replace(nodes=nodes, validate=False)

    u = domain.GraphModel(AddValidNodeMask())
    assert jnp.allclose(u(batch).data[:, 0], jnp.array([2.0, 5.0]))


def test_graph_dataset_domain_layout_rejects_oversized_sample():
    layout = phx.graph.LayoutPlan(max_nodes=4, max_edges=4, max_graphs=2)
    domain = phx.domain.GraphDatasetDomain(_graphs(), layout=layout)

    try:
        domain.points_from_indices(
            [0, 1],
            component=phx.domain.Nodes(),
            structure=phx.domain.SampleLayout((("graph",),)),
        )
    except ValueError as exc:
        assert "max_nodes" in str(exc)
    else:
        raise AssertionError("Expected layout packing to reject oversized graph batch.")
