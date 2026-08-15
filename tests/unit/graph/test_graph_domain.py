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
    structure = phx.domain.SampleLayout((("graph",),))

    batch = component.sample(phx.domain.PointSampling(3, layout=structure), key=jr.key(0))
    axis = batch.structure.axis_for("graph")

    assert isinstance(batch, phx.domain.GraphBatch)
    assert axis is not None
    assert batch["graph"].dims == (axis, None)
    assert batch["graph"].data.shape == (3, 1)


def test_graph_domain_function_evaluates_over_nodes():
    domain = phx.domain.GraphDomain(_make_graph())
    component = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.SampleLayout((("graph",),))
    batch = component.sample(phx.domain.PointSampling(3, layout=structure), key=jr.key(1))

    @domain.Function("graph")
    def u(node):
        return node[0] + 1.0

    out = u(batch)
    axis = batch.structure.axis_for("graph")
    assert out.dims == (axis,)
    assert jnp.allclose(jnp.asarray(out.data), jnp.array([1.0, 2.0, 3.0]))


def test_graph_domain_samples_explicit_node_sets():
    domain = phx.domain.GraphDomain(_make_graph(), measure="count")
    component = domain.component({"graph": phx.domain.BoundaryNodes([0, 2])})
    structure = phx.domain.SampleLayout((("graph",),))

    batch = component.sample(phx.domain.PointSampling(2, layout=structure), key=jr.key(1))

    assert batch.component_kind == "nodes"
    assert jnp.allclose(jnp.asarray(batch["graph"].data), jnp.array([[0.0], [2.0]]))
    assert jnp.allclose(
        jnp.asarray(batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data),
        jnp.array([0, 2], dtype=jnp.int32),
    )
    assert jnp.allclose(component.mass.value, 2.0)


def test_graph_domain_samples_explicit_edge_sets():
    domain = phx.domain.GraphDomain(_make_graph(), measure="count")
    component = domain.component({"graph": phx.domain.InterfaceEdges([2, 0])})
    structure = phx.domain.SampleLayout((("graph",),))

    batch = component.sample(phx.domain.PointSampling(2, layout=structure), key=jr.key(1))

    assert batch.component_kind == "edges"
    assert jnp.allclose(jnp.asarray(batch["graph"].data), jnp.array([[2.5], [0.5]]))
    assert jnp.allclose(
        jnp.asarray(batch[phx.domain.graph.GRAPH_ENTITY_INDEX_KEY].data),
        jnp.array([2, 0], dtype=jnp.int32),
    )
    assert jnp.allclose(component.mass.value, 2.0)


def test_graph_domain_integral_measure_modes():
    graph = _make_graph()
    structure = phx.domain.SampleLayout((("graph",),))

    probability_domain = phx.domain.GraphDomain(graph, measure="probability")
    probability_component = probability_domain.component({"graph": phx.domain.Nodes()})
    probability_batch = probability_component.sample(
        phx.domain.PointSampling(3, layout=structure)
    )
    probability_realization = phx.integration.from_samples(
        phx.integration.over(probability_component), probability_batch
    )
    prob_integral = phx.operators.integral(1.0, probability_realization)
    assert jnp.allclose(jnp.asarray(prob_integral.data), 1.0)

    count_domain = phx.domain.GraphDomain(graph, measure="count")
    count_component = count_domain.component({"graph": phx.domain.Nodes()})
    count_batch = count_component.sample(phx.domain.PointSampling(3, layout=structure))
    count_realization = phx.integration.from_samples(
        phx.integration.over(count_component), count_batch
    )
    count_integral = phx.operators.integral(1.0, count_realization)
    assert jnp.allclose(jnp.asarray(count_integral.data), 3.0)


def test_graph_domain_residual_penalty_is_zero():
    domain = phx.domain.GraphDomain(_make_graph())
    component = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.SampleLayout((("graph",),))

    @domain.Function("graph")
    def u(node):
        del node
        return 0.0

    condition = phx.conditions.Residual("u", component, lambda f: f)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(3, layout=structure),
    )
    term = phx.terms.ResidualPenalty(condition, source)

    loss = term.loss({"u": u}, key=jr.key(2))
    assert loss < 1e-12
