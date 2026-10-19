import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.graph as vx


def _make_data(offset: float) -> vx.Data:
    return vx.Data(
        x=jnp.array([[offset], [offset + 1.0]]),
        edge_index=jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        edge_attr=jnp.array([[1.0], [2.0]]),
        y=jnp.array([offset]),
    )


def test_data_batch_scenario_1() -> None:
    data = _make_data(0.0)
    graph = data.to_graph_ir()
    assert graph.num_nodes == 2
    assert graph.num_edges == 2
    assert graph.nodes.shape == (2, 1)
    d1 = _make_data(0.0)
    d2 = _make_data(10.0)

    batch = vx.Batch.from_data_list([d1, d2])
    assert batch.x is not None
    assert batch.edge_index is not None
    assert batch.num_graphs == 2
    assert batch.x.shape == (4, 1)
    assert batch.edge_index.shape == (2, 4)

    recovered = batch.to_data_list()
    assert recovered[0].x is not None
    assert recovered[1].edge_index is not None
    assert len(recovered) == 2
    assert recovered[0].x.shape == (2, 1)
    assert recovered[1].edge_index.shape == (2, 2)
    first = _make_data(0.0)
    second = _make_data(10.0)
    first = vx.Data(
        x=first.x,
        edge_index=first.edge_index,
        edge_attr=first.edge_attr,
        y=jnp.asarray([[1.0, 2.0]]),
    )
    second = vx.Data(
        x=second.x,
        edge_index=second.edge_index,
        edge_attr=second.edge_attr,
        y=jnp.asarray([[3.0, 4.0]]),
    )

    batch = vx.Batch.from_data_list((first, second))

    # ty: ignore[invalid-argument-type]
    assert jnp.array_equal(batch.y, jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    recovered = batch.to_data_list()
    # ty: ignore[invalid-argument-type]
    assert jnp.array_equal(recovered[0].y, first.y)
    # ty: ignore[invalid-argument-type]
    assert jnp.array_equal(recovered[1].y, second.y)


def test_declared_node_count_keeps_trailing_isolated_nodes() -> None:
    data = vx.Data(
        edge_index=jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32),
        num_nodes=4,
    )
    graph = data.to_graph_ir()
    assert data.num_nodes == 4
    assert graph.num_nodes == 4
    recovered = vx.Data.from_graph_ir(graph)
    assert recovered.x is None
    assert recovered.pos is None
    assert recovered.num_nodes == 4
    np.testing.assert_array_equal(recovered.edge_index, data.edge_index)


def test_featureless_batch_round_trip_preserves_isolated_nodes() -> None:
    first = vx.Data(
        edge_index=jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32), num_nodes=4
    )
    second = vx.Data(num_nodes=3)
    batch = vx.Batch.from_data_list((first, second))
    assert batch.num_nodes == 7
    np.testing.assert_array_equal(batch.ptr, (0, 4, 7))
    recovered = batch.to_data_list()
    assert [data.num_nodes for data in recovered] == [4, 3]
    assert [data.num_edges for data in recovered] == [2, 0]


@pytest.mark.parametrize(
    "edges",
    (
        None,
        jnp.empty((2, 0), dtype=jnp.int32),
        jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32),
    ),
    ids=("no-edges", "empty-edges", "connected-edges"),
)
def test_featureless_data_refuses_omitted_node_count(edges: jax.Array | None) -> None:
    with pytest.raises(ValueError, match="Data.num_nodes"):
        vx.Data(edge_index=edges)


def test_declared_node_count_is_available_during_tracing() -> None:
    def count(edges: jax.Array) -> int:
        return vx.Data(edge_index=edges, num_nodes=4).num_nodes

    compiled = jax.jit(count)
    assert compiled(jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32)) == 4


@pytest.mark.parametrize(
    ("nodes", "x", "pos"),
    (
        ("x", jnp.zeros((3, 2), dtype=jnp.float64), None),
        ("pos", None, jnp.zeros((3, 2), dtype=jnp.float64)),
    ),
    ids=("features", "positions"),
)
def test_declared_node_count_refuses_mismatched_node_rows(
    nodes: str,
    x: jax.Array | None,
    pos: jax.Array | None,
) -> None:
    with pytest.raises(ValueError, match=f"Data.{nodes}"):
        vx.Data(x=x, pos=pos, num_nodes=4)


def test_empty_batch_declares_zero_nodes() -> None:
    assert vx.Batch.from_data_list(()).num_nodes == 0
