#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


class _EdgeGlobalInputModel:
    def __call__(self, graph):
        nodes = dict(graph.nodes)
        edges = dict(graph.edges)
        scale = graph.globals["scale"][0]
        messages = edges["k"] * nodes["u"][graph.senders]
        aggregated = phx.graph.segment_sum(messages, graph.receivers, graph.num_nodes)
        nodes["out"] = aggregated + scale
        return graph.replace(nodes=nodes, validate=False)


def _graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[1.0], [2.0], [3.0]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )


def _mapping_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes={"base": jnp.array([[0.0], [1.0], [3.0]])},
        edges={"base": jnp.array([[2.0], [3.0]])},
        globals={"case": jnp.array([[5.0]])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def _batch(domain):
    component = domain.component({"graph": phx.domain.Nodes()})
    return component.sample(
        phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("graph",),)))
    )


def _boundary_batch(domain):
    component = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    return component.sample(
        phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),)))
    )


def test_graph_model_wrapper_returns_node_field():
    domain = phx.domain.GraphDomain(_graph())
    batch = _batch(domain)
    model = phx.graph.GraphMapFeatures(embed_node_fn=lambda nodes: nodes + 2.0)
    wrapped = phx.domain.DomainFunction(
        domain=domain,
        deps=("graph",),
        func=phx.domain.graph.GraphModel(model),
    )

    out = wrapped(batch)
    assert jnp.allclose(jnp.asarray(out.data), jnp.array([[3.0], [4.0], [5.0]]))


def test_graph_model_wrapper_restricts_output_to_node_set():
    domain = phx.domain.GraphDomain(_graph())
    batch = _boundary_batch(domain)
    model = phx.graph.GraphMapFeatures(embed_node_fn=lambda nodes: nodes + 2.0)
    wrapped = phx.domain.DomainFunction(
        domain=domain,
        deps=("graph",),
        func=phx.domain.graph.GraphModel(model),
    )

    out = wrapped(batch)
    assert jnp.allclose(jnp.asarray(out.data), jnp.array([[3.0], [5.0]]))


def test_graph_model_wrapper_input_fn_uses_full_node_view_for_node_sets():
    domain = phx.domain.GraphDomain(_graph())
    batch = _boundary_batch(domain)

    @domain.Function("graph")
    def input_fn(node):
        return 10.0 * node[0]

    model = phx.graph.GraphMapFeatures(embed_node_fn=lambda nodes: nodes)
    wrapped = phx.domain.DomainFunction(
        domain=domain,
        deps=("graph",),
        func=phx.domain.graph.GraphModel(model, input_fn=input_fn),
    )

    out = wrapped(batch)
    assert jnp.allclose(jnp.asarray(out.data), jnp.array([10.0, 30.0]))


def test_graph_model_wrapper_installs_edge_and_global_input_functions():
    domain = phx.domain.GraphDomain(_mapping_graph())
    batch = _boundary_batch(domain)

    @domain.Function("graph")
    def u(node):
        return node["base"][0]

    @domain.Function("graph")
    def k(edge):
        return edge["base"][0]

    @domain.Function("graph")
    def scale(case):
        return case["case"][0]

    out = domain.GraphModel(
        _EdgeGlobalInputModel(),
        input_fn=u,
        input_key="u",
        edge_input_fn=k,
        edge_input_key="k",
        global_input_fn=scale,
        global_input_key="scale",
        output_key="out",
    )(batch)

    assert jnp.allclose(jnp.asarray(out.data), jnp.array([5.0, 8.0]))


def test_graph_domain_graph_model_convenience():
    domain = phx.domain.GraphDomain(_graph())
    batch = _batch(domain)
    model = phx.graph.GraphMapFeatures(embed_node_fn=lambda nodes: 2.0 * nodes)
    u = domain.GraphModel(model)

    out = u(batch)
    assert jnp.allclose(jnp.asarray(out.data), jnp.array([[2.0], [4.0], [6.0]]))
