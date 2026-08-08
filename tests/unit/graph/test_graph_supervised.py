#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


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


def _node_targets():
    return (
        jnp.array([10.0, 11.0]),
        jnp.array([22.0, 24.0, 28.0]),
    )


def _linear_node_targets():
    return tuple(10.0 + 2.0 * graph.nodes[:, 0] for graph in _graphs())


def test_graph_target_aligns_repeated_cases_and_node_sets():
    domain = phx.domain.GraphDatasetDomain(_graphs())
    batch = domain.points_from_indices(
        [1, 0, 1],
        component=phx.domain.BoundaryNodes([1]),
        structure=phx.domain.SampleLayout((("graph",),)),
    )
    target = phx.terms.GraphTarget(
        domain,
        _node_targets(),
        component_kind="nodes",
    )

    assert jnp.allclose(
        batch[phx.domain.graph.GRAPH_ENTITY_OFFSET_KEY].data,
        jnp.array([0, 3, 5], dtype=jnp.int32),
    )
    assert jnp.allclose(target(batch).data, jnp.array([24.0, 11.0, 24.0]))


def test_graph_supervised_constraint_zero_for_matching_node_function():
    domain = phx.domain.GraphDatasetDomain(_graphs())
    component = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def u(node):
        return 10.0 + 2.0 * node[0]

    constraint = phx.terms.GraphSupervisedTerm(
        "u",
        component,
        _linear_node_targets(),
        sampling=phx.domain.PointSampling(8, design="uniform"),
    )

    assert constraint.loss({"u": u}, key=jr.key(0)) < 1e-12
    assert jnp.allclose(
        constraint.data_metrics({"u": u}, key=jr.key(0))["data_accuracy"],
        1.0,
    )


def test_graph_trajectory_signal_matches_nearest_observations():
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )
    values = []
    for graph, length in zip(domain.graphs, domain.lengths.tolist(), strict=True):
        times = domain.start + domain.dt * jnp.arange(int(length))
        values.append(graph.nodes[:, 0][None, :] + 2.0 * times[:, None])
    signal = phx.terms.GraphTrajectorySignal(domain, tuple(values))
    component = domain.component({"graph": phx.domain.Nodes(), "t": phx.domain.Interior()})
    batch = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
        time_indices=jnp.array([1, 2], dtype=jnp.int32),
    )

    assert jnp.allclose(signal(batch).data, jnp.array([1.0, 2.0, 4.0, 6.0, 10.0]))


def test_graph_trajectory_signal_linearly_interpolates_time():
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )
    values = []
    for graph, length in zip(domain.graphs, domain.lengths.tolist(), strict=True):
        times = domain.start + domain.dt * jnp.arange(int(length))
        values.append(graph.nodes[:, 0][None, :] + 2.0 * times[:, None])
    signal = phx.terms.GraphTrajectorySignal(
        domain,
        tuple(values),
        interpolation="linear",
    )
    component = domain.component({"graph": phx.domain.BoundaryNodes([1]), "t": phx.domain.Interior()})
    batch = domain.points_from_case_time(
        [0, 1],
        [0.25, 0.75],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )

    assert jnp.allclose(signal(batch).data, jnp.array([1.5, 5.5]))


def test_graph_trajectory_supervised_constraint_zero_for_matching_function():
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(),
        jnp.array([3, 5], dtype=jnp.int32),
        dt=0.5,
    )
    values = []
    for graph, length in zip(domain.graphs, domain.lengths.tolist(), strict=True):
        times = domain.start + domain.dt * jnp.arange(int(length))
        values.append(graph.nodes[:, 0][None, :] + 2.0 * times[:, None])
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.FixedStart()}
    )

    @domain.Function("graph", "t")
    def u(node, t):
        return node[0] + 2.0 * t

    constraint = phx.terms.GraphTrajectorySupervisedTerm(
        "u",
        component,
        tuple(values),
        sampling=phx.domain.PointSampling(4),
    )

    assert constraint.loss({"u": u}, key=jr.key(1)) < 1e-12
    assert jnp.allclose(
        constraint.data_metrics({"u": u}, key=jr.key(1))["data_relative_l2_error"],
        0.0,
    )
