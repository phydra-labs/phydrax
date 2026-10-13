from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

from phydrax.nn.layers import AttentionExecution, Linear, MeasureAwareAttention


def _layer(
    execution: AttentionExecution,
    *,
    block_size: int | None = None,
    attention_workspace_bytes: int = 8 * 1024 * 1024,
) -> MeasureAwareAttention:
    layer = MeasureAwareAttention(
        source_channels=4,
        query_channels=4,
        out_channels=4,
        num_heads=2,
        head_dim=2,
        execution=execution,
        block_size=block_size,
        attention_workspace_bytes=attention_workspace_bytes,
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
    actual = _layer(execution, attention_workspace_bytes=256).attend_projected(
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
    bounded = _layer(
        execution,
        block_size=3 if execution == "blockwise" else None,
        attention_workspace_bytes=256,
    )

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
    auto = _layer("auto", attention_workspace_bytes=1).attend_projected(
        query, key, value, weights, source_mask=mask
    )
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
    actual = _layer("auto", attention_workspace_bytes=1).attend_projected(
        query, key, value, weights, source_mask=mask
    )
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


@pytest.mark.parametrize(
    "execution,block_size,budget",
    [
        ("auto", None, 8 * 1024 * 1024),
        ("auto", None, 672),
        ("blockwise", None, 672),
        ("auto", 7, 1),
        ("dense", 1, 1),
    ],
    ids=[
        "default-budget",
        "auto-full-fit",
        "blockwise-full-fit",
        "explicit-full-tile",
        "dense-override",
    ],
)
def test_attention_full_fit_is_exactly_dense(
    execution: AttentionExecution, block_size: int | None, budget: int
) -> None:
    query, key, value, weights, mask = _inputs()
    expected = _layer("dense").attend_projected(
        query, key, value, weights, source_mask=mask
    )
    actual = _layer(
        execution, block_size=block_size, attention_workspace_bytes=budget
    ).attend_projected(query, key, value, weights, source_mask=mask)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "execution,block_size,budget",
    [
        ("auto", None, 256),
        ("blockwise", None, 256),
        ("auto", 3, 8 * 1024 * 1024),
        ("blockwise", 3, 1),
        ("auto", None, 1),
    ],
    ids=[
        "auto-over-budget",
        "blockwise-derived-tile",
        "explicit-tile-overrides-full-fit",
        "explicit-tile-overrides-small-budget",
        "one-column-floor",
    ],
)
def test_attention_budgeted_tiles_preserve_masks_and_measure(
    execution: AttentionExecution, block_size: int | None, budget: int
) -> None:
    query, key, value, _, mask = _inputs()
    weights = jnp.asarray([0.2, -1.0, 1.5, jnp.nan, 2.0, jnp.inf, 0.7], dtype=jnp.float64)
    query_mask = jnp.asarray([[True, False, True], [True, True, True]])
    expected = _layer("dense").attend_projected(
        query, key, value, weights, source_mask=mask, query_mask=query_mask
    )
    actual = _layer(
        execution, block_size=block_size, attention_workspace_bytes=budget
    ).attend_projected(
        query, key, value, weights, source_mask=mask, query_mask=query_mask
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("budget", [0, -1], ids=["zero", "negative"])
def test_attention_refuses_nonpositive_workspace_budget(budget: int) -> None:
    with pytest.raises(ValueError, match="attention_workspace_bytes must be positive"):
        _layer("auto", attention_workspace_bytes=budget)


@pytest.mark.parametrize("budget", [0.5, float("nan")], ids=["fractional", "nan"])
def test_attention_refuses_noninteger_workspace_budget(budget: float) -> None:
    with pytest.raises(TypeError, match="attention_workspace_bytes must be an integer"):
        _layer(
            "auto",
            attention_workspace_bytes=budget,  # ty: ignore[invalid-argument-type]
        )


def test_attention_refuses_boolean_workspace_budget() -> None:
    with pytest.raises(TypeError, match="attention_workspace_bytes must be an integer"):
        _layer("auto", attention_workspace_bytes=True)


def test_attention_explicit_float32_tile_preserves_dense_values() -> None:
    query, key, value, weights, mask = _inputs()
    query, key, value, weights = (
        array.astype(jnp.float32) for array in (query, key, value, weights)
    )
    actual = _layer("auto", block_size=3).attend_projected(
        query, key, value, weights, source_mask=mask
    )
    expected = _layer("dense").attend_projected(
        query, key, value, weights, source_mask=mask
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
