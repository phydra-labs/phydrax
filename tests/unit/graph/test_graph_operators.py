from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx


def _line_graph(*, weighted: bool = False) -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.asarray([[0.0], [1.0], [3.0]]),
        edges=jnp.asarray([[2.0], [3.0]]) if weighted else None,
        senders=jnp.asarray([0, 1], dtype=jnp.int32),
        receivers=jnp.asarray([1, 2], dtype=jnp.int32),
        n_node=jnp.asarray([3], dtype=jnp.int32),
        n_edge=jnp.asarray([2], dtype=jnp.int32),
    )


def _node_batch(domain: phx.domain.GraphDomain) -> tuple[Any, Any]:
    component = domain.component({"graph": phx.domain.Nodes()})
    layout = phx.domain.SampleLayout((("graph",),))
    return component, component.sample(phx.domain.PointSampling(3, layout=layout))


def _edge_batch(domain: phx.domain.GraphDomain) -> tuple[Any, Any]:
    component = domain.component({"graph": phx.domain.Edges()})
    layout = phx.domain.SampleLayout((("graph",),))
    return component, component.sample(phx.domain.PointSampling(2, layout=layout))


def test_graph_degree_supports_full_and_restricted_node_sets() -> None:
    domain = phx.domain.GraphDomain(_line_graph())
    _, full = _node_batch(domain)
    # ty: ignore[invalid-argument-type]
    boundary_component = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    boundary = boundary_component.sample(
        phx.domain.PointSampling(
            2,
            layout=phx.domain.SampleLayout((("graph",),)),
        )
    )
    for mode, expected_full, expected_boundary in (
        ("in", jnp.asarray([0.0, 1.0, 1.0]), jnp.asarray([0.0, 1.0])),
        ("out", jnp.asarray([1.0, 1.0, 0.0]), jnp.asarray([1.0, 0.0])),
    ):
        degree = phx.operators.graph_degree(domain, mode=mode)
        assert jnp.allclose(jnp.asarray(degree(full).data), expected_full), mode
        assert jnp.allclose(jnp.asarray(degree(boundary).data), expected_boundary), mode


def test_unweighted_graph_operators_match_exact_line_graph_references() -> None:
    domain = phx.domain.GraphDomain(_line_graph())
    _, nodes = _node_batch(domain)
    _, edges = _edge_batch(domain)

    @domain.Function("graph")
    def field(node: jax.Array) -> jax.Array:
        return node[0]

    cases = (
        (
            "neighbor",
            phx.operators.neighbor_aggregate(field),
            nodes,
            jnp.asarray([0.0, 0.0, 1.0]),
        ),
        (
            "laplacian",
            phx.operators.graph_laplacian(field),
            nodes,
            jnp.asarray([0.0, 1.0, 2.0]),
        ),
        (
            "gradient",
            phx.operators.graph_gradient(field),
            edges,
            jnp.asarray([1.0, 2.0]),
        ),
    )
    for case_id, operator, batch, expected in cases:
        assert jnp.allclose(jnp.asarray(operator(batch).data), expected), case_id

    # ty: ignore[invalid-argument-type]
    edge_component = domain.component({"graph": phx.domain.InterfaceEdges([1])})
    restricted = edge_component.sample(
        phx.domain.PointSampling(
            1,
            layout=phx.domain.SampleLayout((("graph",),)),
        )
    )
    assert jnp.allclose(
        jnp.asarray(phx.operators.graph_gradient(field)(restricted).data),
        jnp.asarray([2.0]),
    )


def test_weighted_gradient_and_divergence_support_full_and_restricted_sets() -> None:
    domain = phx.domain.GraphDomain(_line_graph(weighted=True))
    _, edges = _edge_batch(domain)
    _, nodes = _node_batch(domain)

    @domain.Function("graph")
    def field(node: jax.Array) -> jax.Array:
        return node[0]

    @domain.Function("graph")
    def edge_value(edge: jax.Array) -> jax.Array:
        return edge[0]

    weighted_gradient = phx.operators.graph_gradient(field, weight=edge_value)
    divergence = phx.operators.graph_divergence(edge_value)
    assert jnp.allclose(
        jnp.asarray(weighted_gradient(edges).data),
        jnp.asarray([2.0, 6.0]),
    )
    assert jnp.allclose(
        jnp.asarray(divergence(nodes).data),
        jnp.asarray([-2.0, -1.0, 3.0]),
    )

    # ty: ignore[invalid-argument-type]
    boundary_component = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    boundary = boundary_component.sample(
        phx.domain.PointSampling(
            2,
            layout=phx.domain.SampleLayout((("graph",),)),
        )
    )
    assert jnp.allclose(
        jnp.asarray(divergence(boundary).data),
        jnp.asarray([-2.0, 3.0]),
    )


def test_incidence_laplacian_is_divergence_of_gradient() -> None:
    domain = phx.domain.GraphDomain(_line_graph())
    _, batch = _node_batch(domain)

    @domain.Function("graph")
    def field(node: jax.Array) -> jax.Array:
        return node[0]

    laplacian = phx.operators.graph_incidence_laplacian(field)
    composed = phx.operators.graph_divergence(phx.operators.graph_gradient(field))
    expected = jnp.asarray([-1.0, -1.0, 2.0])
    assert jnp.allclose(jnp.asarray(laplacian(batch).data), expected)
    assert jnp.allclose(jnp.asarray(composed(batch).data), expected)


def test_graph_derivative_constraints_vanish_for_constant_fields() -> None:
    domain = phx.domain.GraphDomain(_line_graph())

    @domain.Function("graph")
    def constant(node: jax.Array) -> float:
        del node
        return 2.0

    # ty: ignore[invalid-argument-type]
    boundary = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    cases = (
        (boundary, phx.operators.graph_incidence_laplacian, 2),
        (
            domain.component({"graph": phx.domain.Nodes()}),
            phx.operators.graph_laplacian,
            3,
        ),
        (
            domain.component({"graph": phx.domain.Edges()}),
            phx.operators.graph_gradient,
            2,
        ),
    )
    for component, operator, count in cases:
        condition = phx.conditions.Residual("u", component, operator)
        source = phx.integration.per_step(
            phx.integration.mean_over(component),
            phx.domain.PointSampling(
                count,
                layout=phx.domain.SampleLayout((("graph",),)),
            ),
        )
        assert phx.terms.ResidualPenalty(condition, source).loss({"u": constant}) < 1e-12
