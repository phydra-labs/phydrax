from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

from phydrax.nn.layers import AttentionExecution, Linear, MeasureAwareAttention


def _layer(execution: AttentionExecution) -> MeasureAwareAttention:
    layer = MeasureAwareAttention(
        source_channels=4,
        query_channels=4,
        out_channels=4,
        num_heads=2,
        head_dim=2,
        execution=execution,
        block_size=3,
    )
    output = Linear(in_size=4, out_size=4, rwf=False, use_bias=False)
    output = eqx.tree_at(
        lambda value: value.weight, output, jnp.eye(4, dtype=jnp.float64)
    )
    return eqx.tree_at(lambda value: value.output, layer, output)


def _inputs() -> tuple[Array, Array, Array, Array, Array]:
    query = jnp.sin(jnp.arange(24, dtype=jnp.float64)).reshape(2, 3, 2, 2)
    key = jnp.cos(jnp.arange(56, dtype=jnp.float64) * 0.7).reshape(2, 7, 2, 2)
    value = jnp.sin(jnp.arange(56, dtype=jnp.float64) * 0.3).reshape(2, 7, 2, 2)
    weights = jnp.asarray([0.2, 0.0, 1.5, 0.3, 2.0, 0.1, 0.7], dtype=jnp.float64)
    mask = jnp.asarray([[True, True, True, False, True, True, True], [False] * 7])
    return query, key, value, weights, mask


@pytest.mark.parametrize(
    "execution",
    ["auto", "blockwise", "dense"],
    ids=["auto-float64", "explicit-blocks", "explicit-dense"],
)
def test_attention_matches_independent_measure_softmax(
    execution: AttentionExecution,
) -> None:
    query, key, value, weights, mask = _inputs()
    expected = np.zeros((2, 3, 2, 2), dtype=np.float64)
    for head in range(2):
        logits = (
            np.asarray(query[0, :, head]) @ np.asarray(key[0, :, head]).T / np.sqrt(2.0)
        )
        mass = np.asarray(weights) * np.asarray(mask[0])
        exponential = np.exp(logits - logits.max(axis=-1, keepdims=True)) * mass[None, :]
        expected[0, :, head] = (
            exponential / exponential.sum(axis=-1, keepdims=True)
        ) @ np.asarray(value[0, :, head])
    actual = _layer(execution).attend_projected(
        query, key, value, weights, source_mask=mask
    )
    np.testing.assert_allclose(actual, expected.reshape(2, 3, 4), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "execution", ["auto", "blockwise"], ids=["auto-float64", "explicit-blocks"]
)
def test_attention_value_and_gradients_match_explicit_dense(
    execution: AttentionExecution,
) -> None:
    query, key, value, weights, mask = _inputs()
    dense = _layer("dense")
    bounded = _layer(execution)

    def objective(layer: MeasureAwareAttention, q: Array, k: Array, v: Array) -> Array:
        result = layer.attend_projected(q, k, v, weights, source_mask=mask)
        return jnp.sum(result**2)

    reference = jax.value_and_grad(
        lambda q, k, v: objective(dense, q, k, v), argnums=(0, 1, 2)
    )(query, key, value)
    actual = jax.value_and_grad(
        lambda q, k, v: objective(bounded, q, k, v), argnums=(0, 1, 2)
    )(query, key, value)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True
    ):
        assert jnp.all(jnp.isfinite(left))
        np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)


def test_attention_auto_float32_preserves_fused_values() -> None:
    query, key, value, weights, mask = _inputs()
    query, key, value, weights = (
        array.astype(jnp.float32) for array in (query, key, value, weights)
    )
    auto = _layer("auto").attend_projected(query, key, value, weights, source_mask=mask)
    reference = _layer("dense").attend_projected(
        query, key, value, weights, source_mask=mask
    )
    np.testing.assert_allclose(auto, reference, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    "dtype", [jnp.float16, jnp.bfloat16], ids=["float16", "bfloat16"]
)
def test_attention_auto_low_precision_matches_dense(dtype: DTypeLike) -> None:
    query, key, value, weights, mask = _inputs()
    query, key, value, weights = (
        array.astype(dtype) for array in (query, key, value, weights)
    )
    actual = _layer("auto").attend_projected(query, key, value, weights, source_mask=mask)
    expected = _layer("dense").attend_projected(
        query, key, value, weights, source_mask=mask
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=5e-3)


def test_attention_refuses_nonpositive_block_capacity() -> None:
    with pytest.raises(ValueError, match="block_size must be positive"):
        MeasureAwareAttention(
            source_channels=4,
            query_channels=4,
            out_channels=4,
            num_heads=2,
            head_dim=2,
            block_size=0,
        )
