import importlib.util
from typing import Any

import jax.numpy as jnp
import pytest

import phydrax.graph as vx
from phydrax.graph.compat import jraph as jraph_compat


def _make_graph() -> vx.GraphIR:
    return vx.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [2.0]]),
        edges=jnp.array([[1.0], [1.5], [2.0]]),
        senders=jnp.array([0, 1, 2], dtype=jnp.int32),
        receivers=jnp.array([1, 2, 0], dtype=jnp.int32),
        globals=jnp.array([[3.0]]),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([3], dtype=jnp.int32),
    )


def test_graph_scenario_1() -> None:
    graph = _make_graph()
    assert graph.num_graphs == 1
    assert graph.num_nodes == 3
    assert graph.num_edges == 3
    assert graph.edge_index.shape == (2, 3)
    for senders, receivers, n_node, n_edge, message in [
        ([1], [0], [1, 1], [1, 0], r"Graph 0.*sender 1.*\[0, 1\)"),
        ([0], [1], [1, 1], [1, 0], r"Graph 0.*receiver 1.*\[0, 1\)"),
        ([0], [1], [1, 1], [0, 1], r"Graph 1.*sender 0.*\[1, 2\)"),
        ([0], [0], [0, 1], [1, 0], r"Graph 0.*sender 0.*\[0, 0\)"),
    ]:
        node_count = sum(n_node)
        edge_count = sum(n_edge)
        with pytest.raises(ValueError, match=message):
            vx.GraphIR(
                nodes=jnp.zeros((node_count, 1)),
                edges=jnp.zeros((edge_count, 1)),
                senders=jnp.asarray(senders, dtype=jnp.int32),
                receivers=jnp.asarray(receivers, dtype=jnp.int32),
                n_node=jnp.asarray(n_node, dtype=jnp.int32),
                n_edge=jnp.asarray(n_edge, dtype=jnp.int32),
            )
    for n_node, n_edge, senders, receivers in [
        ([1, 1], [1, 1], [0, 1], [0, 1]),
        ([1, 1], [0, 1], [1], [1]),
        ([0, 1], [0, 1], [0], [0]),
        ([0, 0], [0, 0], [], []),
    ]:
        graph = vx.GraphIR(
            nodes=jnp.zeros((sum(n_node), 1)),
            edges=jnp.zeros((sum(n_edge), 1)),
            senders=jnp.asarray(senders, dtype=jnp.int32),
            receivers=jnp.asarray(receivers, dtype=jnp.int32),
            n_node=jnp.asarray(n_node, dtype=jnp.int32),
            n_edge=jnp.asarray(n_edge, dtype=jnp.int32),
        )
        graph.validate(strict=False)
    graph = vx.GraphIR(
        nodes=jnp.zeros((2, 1)),
        edges=jnp.zeros((1, 1)),
        senders=jnp.asarray([1]),
        receivers=jnp.asarray([1]),
        n_node=jnp.asarray([1, 1]),
        n_edge=jnp.asarray([1, 0]),
        validate=False,
    )

    with pytest.raises(ValueError, match=r"Graph 0.*node interval \[0, 1\)"):
        graph.validate(strict=False)
    graph = _make_graph()
    counts = vx.graph_counts(graph)
    assert counts == {"n_graph": 1, "n_node": 3, "n_edge": 3}
    g1 = _make_graph()
    g2 = _make_graph().replace(nodes=jnp.array([[4.0], [5.0], [6.0]]), validate=True)

    batched = vx.batch_graphs((g1, g2))
    pieces = vx.unbatch_graph(batched)
    assert pieces[0].senders is not None
    assert g1.senders is not None
    assert pieces[1].receivers is not None
    assert g2.receivers is not None

    assert len(pieces) == 2
    assert pieces[0].num_nodes == g1.num_nodes
    assert pieces[1].num_edges == g2.num_edges
    assert jnp.array_equal(pieces[0].senders, g1.senders)
    assert jnp.array_equal(pieces[1].receivers, g2.receivers)


def test_missing_jraph_errors_provide_phydrax_install_guidance(monkeypatch: Any) -> None:
    graph = _make_graph()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)

    for call in (graph.as_jraph_tuple, jraph_compat.require_jraph):
        with pytest.raises(ImportError) as error:
            call()
        message = str(error.value)
        assert "pip install jraph" in message
        assert "phydrax" in message
        assert "vertax" not in message
        assert "[compat]" not in message
