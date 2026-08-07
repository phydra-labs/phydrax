#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _two_node_graph(nodes=None) -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[1.0], [3.0]]) if nodes is None else nodes,
        edges={"weight": jnp.array([1.0, 1.0])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 0], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def test_graph_laplacian_apply_matches_two_node_stencil():
    graph = _two_node_graph()

    out = phx.graph.graph_laplacian_apply(
        graph,
        graph.nodes,
        weight_key="weight",
        normalization="none",
    )

    assert jnp.allclose(out[:, 0], jnp.array([-2.0, 2.0]))


def test_graph_laplacian_operator_supports_mapping_keys():
    graph = _two_node_graph(
        nodes={
            "position": jnp.array([[0.0], [1.0]]),
            "u": jnp.array([1.0, 3.0]),
        }
    )

    out = phx.graph.GraphLaplacianOperator(
        weight_key="weight",
        input_key="u",
        output_key="lap_u",
        normalization="none",
    )(graph)

    assert jnp.allclose(out.nodes["lap_u"], jnp.array([-2.0, 2.0]))
    assert "position" in out.nodes
    assert "u" in out.nodes


def test_graph_polynomial_filter_linear_laplacian_term():
    graph = _two_node_graph()
    filt = phx.graph.GraphPolynomialFilter(
        jnp.array([0.0, 1.0]),
        weight_key="weight",
        normalization="none",
    )

    out = filt(graph)

    assert jnp.allclose(out.nodes[:, 0], jnp.array([-2.0, 2.0]))


def test_graph_polynomial_filter_feature_mixing_coefficients():
    graph = _two_node_graph()
    coeffs = jnp.array([[[1.0, 2.0]]])

    out = phx.graph.GraphPolynomialFilter(coeffs)(graph)

    assert out.nodes.shape == (2, 2)
    assert jnp.allclose(out.nodes, jnp.array([[1.0, 2.0], [3.0, 6.0]]))


def test_graph_chebyshev_filter_identity_and_first_scaled_term():
    graph = _two_node_graph()
    identity = phx.graph.GraphChebyshevFilter(
        jnp.array([1.0]),
        weight_key="weight",
    )
    first = phx.graph.GraphChebyshevFilter(
        jnp.array([0.0, 1.0]),
        weight_key="weight",
        lambda_max=2.0,
    )

    assert jnp.allclose(identity(graph).nodes, graph.nodes)
    assert jnp.allclose(first(graph).nodes[:, 0], jnp.array([-3.0, -1.0]))


def test_graph_spectral_filter_wraps_as_domain_graph_model():
    graph = _two_node_graph()
    domain = phx.domain.GraphDomain(graph)
    component = domain.component({"graph": phx.domain.Nodes()})
    batch = component.sample(phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),))))

    @domain.Function("graph")
    def u(node):
        return node[0]

    filt = domain.GraphModel(
        phx.graph.GraphPolynomialFilter(jnp.array([1.0])),
        input_fn=u,
    )

    assert jnp.allclose(filt(batch).data, jnp.array([1.0, 3.0]))


def test_graph_spectral_filter_preserves_padding_entries():
    graph0 = _two_node_graph()
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        edges={"weight": jnp.array([1.0, 1.0])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    base = phx.domain.GraphDatasetDomain((graph0, graph1))
    domain = base.with_layout(base.layout_for_batch_size(2, multiple=2))
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.Nodes(),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    out = phx.graph.GraphPolynomialFilter(
        jnp.array([0.0, 1.0]),
        weight_key="weight",
        normalization="none",
    )(batch.graph)

    assert out.node_mask is not None
    assert out.nodes.shape == (6, 1)
    assert jnp.allclose(out.nodes[:5, 0], jnp.array([-2.0, 2.0, 0.0, 1.0, 1.0]))
    assert jnp.allclose(out.nodes[5, 0], 0.0)
