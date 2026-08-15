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
        edges=jnp.array([[1.0]]),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[2.0], [4.0], [8.0]]),
        edges=jnp.array([[1.0], [1.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    return graph0, graph1


def _domain() -> phx.domain.GraphTrajectoryDatasetDomain:
    return phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )


def test_graph_trajectory_points_from_case_time_repeats_time_over_nodes():
    domain = _domain()
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
        time_indices=jnp.array([1, 2], dtype=jnp.int32),
    )

    assert isinstance(batch, phx.domain.GraphBatch)
    assert batch.graph.num_nodes == 5
    assert jnp.allclose(batch["graph"].data[:, 0], jnp.array([0.0, 1.0, 2.0, 4.0, 8.0]))
    assert jnp.allclose(
        jnp.asarray(batch["t"].data), jnp.array([0.5, 0.5, 1.0, 1.0, 1.0])
    )
    assert jnp.allclose(
        jnp.asarray(batch[phx.domain.graph.GRAPH_TRAJECTORY_TIME_INDEX_KEY].data),
        jnp.array([1, 1, 2, 2, 2], dtype=jnp.int32),
    )


def test_graph_trajectory_domain_function_evaluates_graph_and_time():
    domain = _domain()
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )

    @domain.Function("graph", "t")
    def u(node, t):
        return node[0] + t

    assert jnp.allclose(jnp.asarray(u(batch).data), jnp.array([0.5, 1.5, 3.0, 5.0, 9.0]))


def test_graph_trajectory_gradient_remaps_time_from_edges_to_nodes():
    domain = _domain()
    component = domain.component(
        {"graph": phx.domain.EdgeSet([0]), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )

    @domain.Function("graph", "t")
    def u(node, t):
        return node[0] + t

    grad = phx.operators.graph_gradient(u)
    assert jnp.allclose(jnp.asarray(grad(batch).data), jnp.array([1.0, 2.0]))


def test_graph_trajectory_residual_penalty_samples_fixed_start_edges():
    domain = _domain()
    component = domain.component(
        {"graph": phx.domain.EdgeSet([0]), "t": phx.domain.FixedStart()}
    )
    structure = phx.domain.SampleLayout((("graph", "t"),))

    @domain.Function("graph", "t")
    def u(node, t):
        del node, t
        return 2.0

    condition = phx.conditions.Residual("u", component, phx.operators.graph_gradient)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(2, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    assert term.loss({"u": u}, key=jr.key(0)) < 1e-12


def test_graph_trajectory_graph_model_input_fn_uses_time_on_full_node_view():
    domain = _domain()
    component = domain.component(
        {"graph": phx.domain.BoundaryNodes([1]), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )

    @domain.Function("graph", "t")
    def input_fn(node, t):
        return node[0] + t

    model = phx.graph.GraphMapFeatures(embed_node_fn=lambda nodes: nodes)
    u = domain.GraphModel(model, input_fn=input_fn)

    assert jnp.allclose(jnp.asarray(u(batch).data), jnp.array([1.5, 5.0]))


def test_graph_trajectory_graph_model_edge_input_fn_uses_time_on_full_edge_view():
    domain = _domain()
    component = domain.component(
        {"graph": phx.domain.BoundaryNodes([1]), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )

    @domain.Function("graph", "t")
    def u(node, t):
        return node[0] + t

    @domain.Function("graph", "t")
    def k(edge, t):
        return edge[0] + t

    model = domain.GraphModel(
        _IncomingWeightedSource(),
        input_fn=u,
        edge_input_fn=k,
    )

    assert jnp.allclose(jnp.asarray(model(batch).data), jnp.array([0.75, 6.0]))


def test_graph_trajectory_layout_packs_topology_but_exposes_real_time_rows():
    base = _domain()
    domain = base.with_layout(base.layout_for_batch_size(2, multiple=2))
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )

    assert batch.graph.node_mask is not None
    assert batch.graph.nodes.shape == (6, 1)
    assert batch["graph"].data.shape == (5, 1)
    assert jnp.allclose(
        jnp.asarray(batch["t"].data), jnp.array([0.5, 0.5, 1.0, 1.0, 1.0])
    )
