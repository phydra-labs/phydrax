#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _line_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def _weighted_line_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]),
        edges=jnp.array([[2.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def _node_batch(domain):
    component = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.SampleLayout((("graph",),))
    return component, component.sample(phx.domain.PointSampling(3, layout=structure))


def _edge_batch(domain):
    component = domain.component({"graph": phx.domain.Edges()})
    structure = phx.domain.SampleLayout((("graph",),))
    return component, component.sample(phx.domain.PointSampling(2, layout=structure))


def test_graph_degree_operator():
    domain = phx.domain.GraphDomain(_line_graph())
    _component, batch = _node_batch(domain)

    deg_in = phx.operators.graph_degree(domain, mode="in")
    deg_out = phx.operators.graph_degree(domain, mode="out")

    assert jnp.allclose(deg_in(batch).data, jnp.array([0.0, 1.0, 1.0]))
    assert jnp.allclose(deg_out(batch).data, jnp.array([1.0, 1.0, 0.0]))


def test_graph_degree_operator_restricts_to_node_set():
    domain = phx.domain.GraphDomain(_line_graph())
    component = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    batch = component.sample(phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),))))

    deg_in = phx.operators.graph_degree(domain, mode="in")
    deg_out = phx.operators.graph_degree(domain, mode="out")

    assert jnp.allclose(deg_in(batch).data, jnp.array([0.0, 1.0]))
    assert jnp.allclose(deg_out(batch).data, jnp.array([1.0, 0.0]))


def test_neighbor_aggregate_operator():
    domain = phx.domain.GraphDomain(_line_graph())
    _component, batch = _node_batch(domain)

    @domain.Function("graph")
    def u(node):
        return node[0]

    agg = phx.operators.neighbor_aggregate(u)
    assert jnp.allclose(agg(batch).data, jnp.array([0.0, 0.0, 1.0]))


def test_graph_laplacian_operator():
    domain = phx.domain.GraphDomain(_line_graph())
    _component, batch = _node_batch(domain)

    @domain.Function("graph")
    def u(node):
        return node[0]

    lap = phx.operators.graph_laplacian(u)
    assert jnp.allclose(lap(batch).data, jnp.array([0.0, 1.0, 2.0]))


def test_graph_gradient_operator_on_edges():
    domain = phx.domain.GraphDomain(_line_graph())
    _component, batch = _edge_batch(domain)

    @domain.Function("graph")
    def u(node):
        return node[0]

    grad = phx.operators.graph_gradient(u)
    assert jnp.allclose(grad(batch).data, jnp.array([1.0, 2.0]))


def test_graph_gradient_operator_restricts_to_edge_set():
    domain = phx.domain.GraphDomain(_line_graph())
    component = domain.component({"graph": phx.domain.InterfaceEdges([1])})
    batch = component.sample(phx.domain.PointSampling(1, layout=phx.domain.SampleLayout((("graph",),))))

    @domain.Function("graph")
    def u(node):
        return node[0]

    grad = phx.operators.graph_gradient(u)
    assert jnp.allclose(grad(batch).data, jnp.array([2.0]))


def test_graph_gradient_supports_edge_weights():
    domain = phx.domain.GraphDomain(_weighted_line_graph())
    _component, batch = _edge_batch(domain)

    @domain.Function("graph")
    def u(node):
        return node[0]

    @domain.Function("graph")
    def weight(edge):
        return edge[0]

    grad = phx.operators.graph_gradient(u, weight=weight)
    assert jnp.allclose(grad(batch).data, jnp.array([2.0, 6.0]))


def test_graph_divergence_operator_on_nodes():
    domain = phx.domain.GraphDomain(_weighted_line_graph())
    _component, batch = _node_batch(domain)

    @domain.Function("graph")
    def flux(edge):
        return edge[0]

    div = phx.operators.graph_divergence(flux)
    assert jnp.allclose(div(batch).data, jnp.array([-2.0, -1.0, 3.0]))


def test_graph_divergence_operator_restricts_to_node_set():
    domain = phx.domain.GraphDomain(_weighted_line_graph())
    component = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    batch = component.sample(phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),))))

    @domain.Function("graph")
    def flux(edge):
        return edge[0]

    div = phx.operators.graph_divergence(flux)
    assert jnp.allclose(div(batch).data, jnp.array([-2.0, 3.0]))


def test_graph_incidence_laplacian_is_divergence_of_gradient():
    domain = phx.domain.GraphDomain(_line_graph())
    _component, batch = _node_batch(domain)

    @domain.Function("graph")
    def u(node):
        return node[0]

    lap = phx.operators.graph_incidence_laplacian(u)
    div_grad = phx.operators.graph_divergence(phx.operators.graph_gradient(u))
    expected = jnp.array([-1.0, -1.0, 2.0])

    assert jnp.allclose(lap(batch).data, expected)
    assert jnp.allclose(div_grad(batch).data, expected)


def test_graph_incidence_laplacian_constraint_on_boundary_nodes():
    domain = phx.domain.GraphDomain(_line_graph())
    component = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    structure = phx.domain.SampleLayout((("graph",),))

    @domain.Function("graph")
    def u(node):
        del node
        return 2.0

    constraint = phx.constraints.FunctionalConstraint.from_operator(component=component,
    operator=phx.operators.graph_incidence_laplacian,
    constraint_vars="u", sampling=phx.domain.PointSampling(2, layout=structure), )

    assert constraint.loss({"u": u}) < 1e-12


def test_graph_laplacian_constraint_zero_for_constant_field():
    domain = phx.domain.GraphDomain(_line_graph())
    component = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.SampleLayout((("graph",),))

    @domain.Function("graph")
    def u(node):
        del node
        return 2.0

    constraint = phx.constraints.FunctionalConstraint.from_operator(component=component,
    operator=phx.operators.graph_laplacian,
    constraint_vars="u", sampling=phx.domain.PointSampling(3, layout=structure), )

    assert constraint.loss({"u": u}) < 1e-12


def test_graph_gradient_constraint_zero_for_constant_field_on_edges():
    domain = phx.domain.GraphDomain(_line_graph())
    component = domain.component({"graph": phx.domain.Edges()})
    structure = phx.domain.SampleLayout((("graph",),))

    @domain.Function("graph")
    def u(node):
        del node
        return 2.0

    constraint = phx.constraints.FunctionalConstraint.from_operator(component=component,
    operator=phx.operators.graph_gradient,
    constraint_vars="u", sampling=phx.domain.PointSampling(2, layout=structure), )

    assert constraint.loss({"u": u}) < 1e-12
