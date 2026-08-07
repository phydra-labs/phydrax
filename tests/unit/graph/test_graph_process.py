#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _graph(nodes=None) -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]) if nodes is None else nodes,
        edges=jnp.array([[1.0], [1.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


class ConstantNodeRate:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, graph):
        return graph.replace(nodes=jnp.full_like(graph.nodes, self.value), validate=False)


class LinearNodeRate:
    def __call__(self, graph):
        return graph.replace(nodes=graph.nodes, validate=False)


class ZeroNodeRate:
    def __call__(self, graph):
        return graph.replace(nodes=jnp.zeros_like(graph.nodes), validate=False)


def _trajectory_domain() -> phx.domain.GraphTrajectoryDatasetDomain:
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
    return phx.domain.GraphTrajectoryDatasetDomain(
        (graph0, graph1),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )


def test_euler_graph_stepper_advances_node_state():
    stepper = phx.graph.EulerGraphStepper(ConstantNodeRate(2.0), dt=0.25)
    out = stepper(_graph())

    assert jnp.allclose(out.nodes, jnp.array([[0.5], [1.5], [3.5]]))
    assert jnp.allclose(out.senders, jnp.array([0, 1], dtype=jnp.int32))


def test_rk4_graph_stepper_matches_exponential_for_linear_rate():
    stepper = phx.graph.RK4GraphStepper(LinearNodeRate(), dt=0.1)
    out = stepper(_graph(nodes=jnp.ones((3, 1))))
    expected_scale = 1.0 + 0.1 + 0.1**2 / 2.0 + 0.1**3 / 6.0 + 0.1**4 / 24.0

    assert jnp.allclose(out.nodes, expected_scale * jnp.ones((3, 1)))


def test_autoregressive_rollout_stacks_node_features():
    graph = _graph(nodes=jnp.zeros((3, 1)))
    stepper = phx.graph.EulerGraphStepper(ConstantNodeRate(1.0), dt=1.0)

    nodes = phx.graph.rollout_features(stepper, graph, steps=3, feature="nodes")

    assert nodes.shape == (4, 3, 1)
    assert jnp.allclose(nodes[:, 0, 0], jnp.array([0.0, 1.0, 2.0, 3.0]))


def test_rollout_feature_loss_zero_for_matching_targets():
    graph = _graph(nodes=jnp.zeros((3, 1)))
    stepper = phx.graph.EulerGraphStepper(ConstantNodeRate(1.0), dt=1.0)
    target = phx.graph.rollout_features(stepper, graph, steps=3, feature="nodes")

    assert phx.graph.rollout_feature_loss(stepper, graph, target, feature="nodes") < 1e-12


def test_graph_rollout_model_wraps_as_domain_function():
    domain = phx.domain.GraphDomain(_graph())
    batch = domain.sample_component(
        phx.domain.Nodes(),
        3,
        structure=phx.domain.SampleLayout((("graph",),)),
    )
    stepper = phx.graph.EulerGraphStepper(ConstantNodeRate(1.0), dt=1.0)
    rollout_fn = domain.GraphRolloutModel(stepper, steps=2, feature="nodes")

    out = rollout_fn(batch)

    assert out.data.shape == (3, 3, 1)
    assert jnp.allclose(out.data[0, :, 0], jnp.array([0.0, 1.0, 2.0]))
    assert jnp.allclose(out.data[2, :, 0], jnp.array([3.0, 4.0, 5.0]))


def test_graph_rollout_model_participates_in_functional_constraint():
    domain = phx.domain.GraphDomain(_graph())
    nodes = domain.component({"graph": phx.domain.Nodes()})
    structure = phx.domain.SampleLayout((("graph",),))
    stepper = phx.graph.EulerGraphStepper(ConstantNodeRate(1.0), dt=1.0)

    @domain.Function("graph")
    def u(node):
        return node[0]

    @domain.Function("graph")
    def target(node):
        return node[0] + jnp.arange(3.0)

    def residual(pred):
        return pred - target

    constraint = phx.constraints.FunctionalConstraint.from_operator(component=nodes,
    operator=residual,
    constraint_vars="pred", sampling=phx.domain.PointSampling(3, layout=structure), )
    rollout_fn = domain.GraphRolloutModel(stepper, steps=2, input_fn=u)

    assert constraint.loss({"pred": rollout_fn}) < 1e-12


def test_graph_rollout_model_respects_node_subsets():
    domain = phx.domain.GraphDomain(_graph())
    subset = domain.component(
        {"graph": phx.domain.NodeSet(jnp.array([0, 2], dtype=jnp.int32))}
    )
    batch = subset.sample(phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),))))
    stepper = phx.graph.EulerGraphStepper(ConstantNodeRate(1.0), dt=1.0)
    rollout_fn = domain.GraphRolloutModel(stepper, steps=1)

    out = rollout_fn(batch)

    assert out.data.shape == (2, 2, 1)
    assert jnp.allclose(out.data[:, :, 0], jnp.array([[0.0, 1.0], [3.0, 4.0]]))


def test_process_stepper_preserves_padding_entries():
    graph0 = _graph(nodes=jnp.zeros((3, 1)))
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[2.0], [4.0]]),
        edges=jnp.array([[1.0]]),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDatasetDomain((graph0, graph1))
    domain = domain.with_layout(domain.layout_for_batch_size(2, multiple=2))
    batch = domain.points_from_indices(
        [0, 1],
        component=phx.domain.Nodes(),
        structure=phx.domain.SampleLayout((("graph",),)),
    )

    stepper = phx.graph.EulerGraphStepper(ConstantNodeRate(1.0), dt=1.0)
    out = stepper(batch.graph)

    assert out.node_mask is not None
    assert out.nodes.shape == (6, 1)
    assert jnp.allclose(out.nodes[:5, 0], jnp.array([1.0, 1.0, 1.0, 3.0, 5.0]))
    assert jnp.allclose(out.nodes[5, 0], 0.0)


def test_graph_process_stepper_integrates_with_graph_trajectory_constraint():
    domain = _trajectory_domain()
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.FixedStart()}
    )
    structure = phx.domain.SampleLayout((("graph", "t"),))
    stepper = phx.graph.EulerGraphStepper(ZeroNodeRate(), dt=0.5)

    @domain.Function("graph", "t")
    def u(node, t):
        return node[0] + 0.0 * t

    def residual(f):
        return domain.GraphModel(stepper, input_fn=f) - f

    constraint = phx.constraints.FunctionalConstraint.from_operator(component=component,
    operator=residual,
    constraint_vars="u", sampling=phx.domain.PointSampling(2, layout=structure), )

    assert constraint.loss({"u": u}, key=jr.key(0)) < 1e-12
