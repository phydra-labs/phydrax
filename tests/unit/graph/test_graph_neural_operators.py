#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _line_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]),
        edges=jnp.array([[2.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


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


def test_graph_kernel_integral_aggregates_weighted_source_nodes():
    kernel = phx.graph.GraphKernelIntegral(
        lambda edges, sent, recv, globals_: edges[:, 0],
    )
    out = kernel(_line_graph())

    assert jnp.allclose(out.nodes[:, 0], jnp.array([0.0, 0.0, 3.0]))


def test_graph_kernel_integral_normalizes_by_receiver_degree():
    graph = phx.graph.GraphIR(
        nodes=jnp.array([[1.0], [2.0], [4.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([2, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    kernel = phx.graph.GraphKernelIntegral(normalize=True)
    out = kernel(graph)

    assert jnp.allclose(out.nodes[:, 0], jnp.array([0.0, 0.0, 1.5]))


def test_graph_diffusion_computes_weighted_incidence_laplacian():
    diffusion = phx.graph.GraphDiffusion(
        lambda edges, sent, recv, globals_: edges[:, 0],
    )
    out = diffusion(_line_graph())

    assert jnp.allclose(out.nodes[:, 0], jnp.array([-2.0, -4.0, 6.0]))


def test_repeated_graph_processor_applies_block_multiple_times():
    block = phx.graph.GraphMapFeatures(embed_node_fn=lambda nodes: nodes + 1.0)
    processor = phx.graph.RepeatedGraphProcessor(block, steps=3)
    out = processor(_line_graph())

    assert jnp.allclose(out.nodes[:, 0], jnp.array([3.0, 4.0, 6.0]))


def test_graph_neural_operator_preserves_padding_entries():
    base = phx.domain.GraphDatasetDomain(_graphs())
    domain = base.with_layout(base.layout_for_batch_size(2, multiple=2))
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.Nodes(),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    out = phx.graph.GraphDiffusion()(batch.graph)

    assert out.node_mask is not None
    assert out.nodes.shape == (6, 1)
    assert jnp.allclose(out.nodes[:5, 0], jnp.array([-1.0, 1.0, -2.0, -2.0, 4.0]))
    assert jnp.allclose(out.nodes[5, 0], 0.0)


def test_graph_kernel_integral_wraps_as_domain_graph_model():
    domain = phx.domain.GraphDomain(_line_graph())
    component = domain.component({"graph": phx.domain.Nodes()})
    batch = component.sample(phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("graph",),))))

    @domain.Function("graph")
    def u(node):
        return node[0]

    model = phx.graph.GraphKernelIntegral(lambda edges, sent, recv, globals_: edges[:, 0])
    integral = domain.GraphModel(model, input_fn=u)

    assert jnp.allclose(integral(batch).data, jnp.array([0.0, 0.0, 3.0]))


def test_graph_diffusion_penalty_zero_for_constant_graph_time_field():
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.FixedStart()}
    )
    structure = phx.domain.SampleLayout((("graph", "t"),))

    @domain.Function("graph", "t")
    def u(node, t):
        del node, t
        return 2.0

    def residual(f):
        return domain.GraphModel(phx.graph.GraphDiffusion(), input_fn=f)

    condition = phx.conditions.Residual("u", component, residual)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(2, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    assert term.loss({"u": u}, key=jr.key(0)) < 1e-12
