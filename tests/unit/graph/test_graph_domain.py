#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _make_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        edges=jnp.array([[0.5], [1.5], [2.5]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        globals=jnp.array([[7.0]]),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )


def test_graph_domain_samples_node_batch():
    domain = phx.domain.GraphDomain(_make_graph())
    component = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.ProductStructure((("graph",),))

    batch = component.sample(3, structure=structure, key=jr.key(0))
    axis = batch.structure.axis_for("graph")

    assert isinstance(batch, phx.domain.GraphBatch)
    assert axis is not None
    assert batch["graph"].dims == (axis, None)
    assert batch["graph"].data.shape == (3, 1)


def test_graph_domain_function_evaluates_over_nodes():
    domain = phx.domain.GraphDomain(_make_graph())
    component = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.ProductStructure((("graph",),))
    batch = component.sample(3, structure=structure, key=jr.key(1))

    @domain.Function("graph")
    def u(node):
        return node[0] + 1.0

    out = u(batch)
    axis = batch.structure.axis_for("graph")
    assert out.dims == (axis,)
    assert jnp.allclose(out.data, jnp.array([1.0, 2.0, 3.0]))


def test_graph_domain_samples_explicit_node_sets():
    domain = phx.domain.GraphDomain(_make_graph(), measure="count")
    component = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    structure = phx.domain.ProductStructure((("graph",),))

    batch = component.sample(2, structure=structure, key=jr.key(1))

    assert batch.component_kind == "nodes"
    assert jnp.allclose(batch["graph"].data, jnp.array([[0.0], [2.0]]))
    assert jnp.allclose(
        batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data,
        jnp.array([0, 2], dtype=jnp.int32),
    )
    assert jnp.allclose(component.measure(), 2.0)


def test_graph_domain_samples_explicit_edge_sets():
    domain = phx.domain.GraphDomain(_make_graph(), measure="count")
    component = domain.component({"graph": phx.domain.InterfaceEdges([2, 0])})
    structure = phx.domain.ProductStructure((("graph",),))

    batch = component.sample(2, structure=structure, key=jr.key(1))

    assert batch.component_kind == "edges"
    assert jnp.allclose(batch["graph"].data, jnp.array([[2.5], [0.5]]))
    assert jnp.allclose(
        batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data,
        jnp.array([2, 0], dtype=jnp.int32),
    )
    assert jnp.allclose(component.measure(), 2.0)


def test_graph_domain_integral_measure_modes():
    graph = _make_graph()
    structure = phx.domain.ProductStructure((("graph",),))

    probability_domain = phx.domain.GraphDomain(graph, measure="probability")
    probability_component = probability_domain.component({"graph": phx.domain.Nodes()})
    probability_batch = probability_component.sample(3, structure=structure)
    prob_integral = phx.operators.integral(
        1.0, probability_batch, component=probability_component
    )
    assert jnp.allclose(prob_integral.data, 1.0)

    count_domain = phx.domain.GraphDomain(graph, measure="count")
    count_component = count_domain.component({"graph": phx.domain.Nodes()})
    count_batch = count_component.sample(3, structure=structure)
    count_integral = phx.operators.integral(1.0, count_batch, component=count_component)
    assert jnp.allclose(count_integral.data, 3.0)


def test_graph_domain_constraint_zero_residual():
    domain = phx.domain.GraphDomain(_make_graph())
    structure = phx.domain.ProductStructure((("graph",),))

    @domain.Function("graph")
    def u(node):
        del node
        return 0.0

    constraint = phx.constraints.FunctionalConstraint.from_operator(
        component=domain.component({"graph": phx.domain.Nodes()}),
        operator=lambda f: f,
        constraint_vars="u",
        num_points=3,
        structure=structure,
    )

    loss = constraint.loss({"u": u}, key=jr.key(2))
    assert loss < 1e-12
