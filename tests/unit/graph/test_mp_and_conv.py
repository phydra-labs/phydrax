import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax.graph as vx


class AddMessagePassing(vx.MessagePassing):
    def message(self, x_j, x_i=None, edge_attr=None):
        del edge_attr
        if x_i is None:
            return x_j
        return x_j + x_i


def test_message_passing_runs():
    mp = AddMessagePassing(aggr="add", flow="source_to_target")
    x = jnp.array([[1.0], [2.0], [3.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)

    out = mp(x, edge_index)
    assert out.shape == (3, 1)


def test_message_passing_jit_runs():
    mp = AddMessagePassing(aggr="add", flow="source_to_target")
    x = jnp.array([[1.0], [2.0], [3.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)

    out = jax.jit(mp)(x, edge_index)
    assert out.shape == (3, 1)


def test_gcn_conv_shape():
    x = jnp.array([[1.0], [2.0], [3.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    conv = vx.GCNConv(in_features=1, out_features=4, key=jax.random.key(0))

    out = conv(x, edge_index)
    assert out.shape == (3, 4)


def test_gcn_conv_jit_runs():
    x = jnp.array([[1.0], [2.0], [3.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    conv = vx.GCNConv(in_features=1, out_features=4, key=jax.random.key(10))

    out = eqx.filter_jit(conv)(x, edge_index)
    assert out.shape == (3, 4)


def test_sage_conv_shape():
    x = jnp.array([[1.0], [2.0], [3.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    conv = vx.SAGEConv(in_features=1, out_features=3, key=jax.random.key(1))

    out = conv(x, edge_index)
    assert out.shape == (3, 3)


def test_sage_conv_jit_runs():
    x = jnp.array([[1.0], [2.0], [3.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    conv = vx.SAGEConv(in_features=1, out_features=3, key=jax.random.key(11))

    out = eqx.filter_jit(conv)(x, edge_index)
    assert out.shape == (3, 3)


def test_gin_conv_shape():
    x = jnp.array([[1.0], [2.0], [3.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    mlp = eqx.nn.Linear(1, 2, key=jax.random.key(2))
    conv = vx.GINConv(mlp, eps=0.1)

    out = conv(x, edge_index)
    assert out.shape == (3, 2)


def test_gin_conv_jit_runs():
    x = jnp.array([[1.0], [2.0], [3.0]])
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    mlp = eqx.nn.Linear(1, 2, key=jax.random.key(12))
    conv = vx.GINConv(mlp, eps=0.1)

    out = eqx.filter_jit(conv)(x, edge_index)
    assert out.shape == (3, 2)
