from typing import cast, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax.graph as vx


class _GraphConvolution(Protocol):
    def __call__(
        self,
        features: jax.Array,
        edge_index: jax.Array,
        /,
    ) -> jax.Array: ...


def _add_message(
    source: jax.Array,
    target: jax.Array | None = None,
    edge_attributes: jax.Array | None = None,
) -> jax.Array:
    del edge_attributes
    return source if target is None else source + target


def test_mp_and_conv_scenario_1() -> None:
    passing = vx.MessagePassing(
        aggr="add",
        flow="source_to_target",
        message=_add_message,
    )
    features = jnp.asarray([[1.0], [2.0], [3.0]])
    edges = jnp.asarray([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    eager = passing(features, edges)
    compiled = jax.jit(passing)(features, edges)
    assert eager.shape == (3, 1)
    assert jnp.array_equal(compiled, eager)
    source = jnp.asarray([[10.0], [20.0]])
    target = jnp.asarray([[1.0], [2.0], [3.0]])
    edges = jnp.asarray([[0, 1, 0], [2, 0, 1]], dtype=jnp.int32)
    passing = vx.MessagePassing(
        aggr="add",
        flow="target_to_source",
        message=lambda x_j, x_i, edge_attr: 10.0 * x_j + x_i,
    )
    result = passing((source, target), edges)
    assert jnp.array_equal(result, jnp.asarray([[70.0], [30.0]]))
    features = jnp.asarray([[1.0], [2.0], [3.0]])
    edges = jnp.asarray([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)
    cases: tuple[tuple[str, _GraphConvolution, tuple[int, int]], ...] = (
        (
            "gcn",
            vx.GCNConv(in_features=1, out_features=4, key=jax.random.key(0)),
            (3, 4),
        ),
        (
            "sage",
            vx.SAGEConv(in_features=1, out_features=3, key=jax.random.key(1)),
            (3, 3),
        ),
        (
            "gin",
            vx.GINConv(eqx.nn.Linear(1, 2, key=jax.random.key(2)), eps=0.1),
            (3, 2),
        ),
    )
    for case_id, convolution, expected_shape in cases:
        eager = convolution(features, edges)
        compiled = eqx.filter_jit(cast(_GraphConvolution, convolution))(
            features,
            edges,
        )
        assert eager.shape == expected_shape, case_id
        assert jnp.allclose(compiled, eager), case_id
