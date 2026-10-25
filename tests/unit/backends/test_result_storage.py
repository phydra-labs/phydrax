"""Owned result storage keeps values, projections, derivatives and lifetimes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.backends._result_storage import evaluate_with_result_storage
from phydrax.execution import default_device


pytestmark = pytest.mark.strict_jax

_DTYPES = pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))


def _devices() -> tuple[jax.Device, ...]:
    return (jax.devices("cpu")[0],) + tuple(
        device for device in jax.devices() if device.platform == "gpu"
    )


_DEVICES = pytest.mark.parametrize(
    "device", _devices(), ids=lambda device: f"{device.platform}-{device.id}"
)


def _query(position: Array, velocity: Array, gains: Array) -> dict[str, Any]:
    acceleration = gains[:, None] * jnp.sin(position) - velocity * velocity
    energy = jnp.sum(acceleration * acceleration)
    return {
        "acceleration": acceleration,
        "repeated": acceleration,
        "energy": energy,
        "success": jnp.all(jnp.isfinite(acceleration))[None],
        "count": jnp.sum(acceleration > 0, dtype=jnp.int32),
        "position": position,
        "layout": jnp.arange(position.shape[0], dtype=jnp.int32),
    }


def _owned(position: Array, velocity: Array, gains: Array) -> dict[str, Any]:
    return evaluate_with_result_storage(_query, position, velocity, gains)


def _inputs(seed: int, dtype: Any) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(size=(5, 3)), dtype),
        jnp.asarray(rng.normal(size=(5, 3)), dtype),
        jnp.asarray(rng.normal(size=(5,)), dtype),
    )


def _assert_same(actual: Any, expected: Any) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for value, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if isinstance(reference, str):
            assert value == reference
            continue
        assert jnp.result_type(value) == jnp.result_type(reference)
        np.testing.assert_array_equal(np.asarray(value), np.asarray(reference))


def _assert_close(actual: Any, expected: Any) -> None:
    """Derivative trees: same structure and dtypes, values to a few ulps."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for value, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        dtype = jnp.result_type(reference)
        assert jnp.result_type(value) == dtype
        resolution = float(jnp.finfo(dtype).eps) * 16
        np.testing.assert_allclose(
            np.asarray(value), np.asarray(reference), rtol=resolution, atol=resolution
        )


def _snapshot(tree: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf if isinstance(leaf, str) else np.array(leaf, copy=True), tree
    )


@_DTYPES
@_DEVICES
def test_results_match_direct_evaluation_on_every_device(
    dtype: Any, device: jax.Device
) -> None:
    with default_device(device):
        operands = jax.device_put(_inputs(1, dtype), device)
        owned = jax.jit(_owned)(*operands)
        direct = jax.jit(_query)(*operands)
        _assert_same(owned, direct)
        assert owned["acceleration"].devices() == {device}


@_DEVICES
def test_static_leaves_and_concrete_operands_reach_the_computation_unchanged(
    device: jax.Device,
) -> None:
    table = np.arange(4, dtype=np.int32)

    def compute(value: Array, table: np.ndarray, label: str) -> dict[str, Any]:
        # Concrete operands stay concrete: host indexing would fail on a tracer.
        offset = int(table[-1])
        return {"value": value * offset, "table": table, "label": label}

    value = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)
    with default_device(device):
        moved = jax.device_put(value, device)
        result = eqx.filter_jit(
            lambda x: evaluate_with_result_storage(compute, x, table, "tag")
        )(moved)
        assert result["label"] == "tag"
        np.testing.assert_array_equal(result["table"], table)
        np.testing.assert_array_equal(result["value"], jax.jit(lambda x: x * 3)(moved))


@_DTYPES
@_DEVICES
def test_retained_results_survive_later_executions(
    dtype: Any, device: jax.Device
) -> None:
    with default_device(device):
        first, second, third = (
            jax.device_put(_inputs(seed, dtype), device) for seed in (3, 4, 5)
        )
        compiled = jax.jit(_owned).lower(*first).compile()
        retained = compiled(*first)
        expected = _snapshot(retained)
        later = compiled(*second)
        latest = compiled(*third)
        _assert_same(retained, expected)
        _assert_same(later, jax.jit(_query)(*second))
        _assert_same(latest, jax.jit(_query)(*third))
        _assert_same(retained, _snapshot(jax.jit(_query)(*first)))


def _two_queries(position: Array, velocity: Array, gains: Array) -> tuple[Any, Any]:
    first = _owned(position, velocity, gains)
    second = _owned(first["acceleration"], velocity, gains * gains)
    return first, second


def _direct_two_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Any, Any]:
    first = _query(position, velocity, gains)
    return first, _query(first["acceleration"], velocity, gains * gains)


def _rollout(position: Array, velocity: Array, gains: Array) -> tuple[Array, Array]:
    def step(state: Array, _: None) -> tuple[Array, tuple[Array, Array]]:
        result = _owned(state, velocity, gains)
        return state + result["acceleration"], (result["acceleration"], result["energy"])

    _, history = jax.lax.scan(step, position, None, length=3)
    return history


def _direct_rollout(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array]:
    def step(state: Array, _: None) -> tuple[Array, tuple[Array, Array]]:
        result = _query(state, velocity, gains)
        return state + result["acceleration"], (result["acceleration"], result["energy"])

    _, history = jax.lax.scan(step, position, None, length=3)
    return history


def _query_variant(position: Array, velocity: Array, gains: Array) -> dict[str, Any]:
    """A different computation with the same inputs and result types as `_query`."""
    return _query(jnp.cos(position), velocity, gains)


def _distinct_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Any, Any, Any]:
    return (
        evaluate_with_result_storage(_query, position, velocity, gains),
        evaluate_with_result_storage(_query_variant, position, velocity, gains),
        evaluate_with_result_storage(_query, position, velocity, gains),
    )


def _direct_distinct_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Any, Any, Any]:
    return (
        _query(position, velocity, gains),
        _query_variant(position, velocity, gains),
        _query(position, velocity, gains),
    )


def _split(first: Array, second: Array) -> dict[str, Array]:
    return {"sum": first + second, "difference": first - second}


def _projected_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array]:
    # Same computation and inputs, different retained same-typed results.
    del gains
    return (
        evaluate_with_result_storage(_split, position, velocity)["sum"],
        evaluate_with_result_storage(_split, position, velocity)["difference"],
    )


def _direct_projected_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array]:
    del gains
    return _split(position, velocity)["sum"], _split(position, velocity)["difference"]


def _weighted(first: Array, second: Array, third: Array) -> Array:
    return first + 2.0 * second + 3.0 * third


def _aliased_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array]:
    # Same program and deduplicated inputs, different argument roles.
    del gains
    return (
        evaluate_with_result_storage(_weighted, position, position, velocity),
        evaluate_with_result_storage(_weighted, position, velocity, velocity),
    )


def _direct_aliased_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array]:
    del gains
    return (
        _weighted(position, position, velocity),
        _weighted(position, velocity, velocity),
    )


def _real_sine(value: Array) -> Array:
    return jnp.sin(jnp.real(value))


def _complex_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array]:
    # Supported results of distinct complex inputs.
    del gains
    return (
        evaluate_with_result_storage(_real_sine, jax.lax.complex(position, velocity)),
        evaluate_with_result_storage(_real_sine, jax.lax.complex(velocity, position)),
    )


def _direct_complex_queries(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array]:
    del gains
    return (
        _real_sine(jax.lax.complex(position, velocity)),
        _real_sine(jax.lax.complex(velocity, position)),
    )


def _cumulative(value: Array) -> Array:
    return jnp.cumsum(value)


def _owned_cumulative(value: Array) -> Array:
    return evaluate_with_result_storage(_cumulative, value)


def _layouts(
    cumulative: Callable[[Array], Array], position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array, Array, Array]:
    # Same computation over the same physical nonsymmetric arrays, read by rows
    # or columns, and by the two nested axis orders. Built without elementwise
    # arithmetic so fusion cannot change the compared values.
    del gains
    square = position @ velocity.T
    cube = jnp.stack([square, square.T, square, square.T, square], axis=-1)
    return (
        jax.vmap(cumulative, in_axes=0)(square),
        jax.vmap(cumulative, in_axes=1)(square),
        jax.vmap(jax.vmap(cumulative, in_axes=0), in_axes=1)(cube),
        jax.vmap(jax.vmap(cumulative, in_axes=1), in_axes=0)(cube),
    )


def _batched_layouts(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array, Array, Array]:
    return _layouts(_owned_cumulative, position, velocity, gains)


def _direct_batched_layouts(
    position: Array, velocity: Array, gains: Array
) -> tuple[Array, Array, Array, Array]:
    return _layouts(_cumulative, position, velocity, gains)


@_DTYPES
@_DEVICES
@pytest.mark.parametrize(
    ("owned", "direct"),
    (
        (_two_queries, _direct_two_queries),
        (_rollout, _direct_rollout),
        (_distinct_queries, _direct_distinct_queries),
        (_projected_queries, _direct_projected_queries),
        (_aliased_queries, _direct_aliased_queries),
        (_complex_queries, _direct_complex_queries),
        (_batched_layouts, _direct_batched_layouts),
    ),
    ids=(
        "chained",
        "scan",
        "distinct-same-input",
        "different-projections",
        "aliased-arguments",
        "complex-dependencies",
        "batched-layouts",
    ),
)
def test_composed_queries_keep_retained_results(
    dtype: Any,
    device: jax.Device,
    owned: Callable[[Array, Array, Array], Any],
    direct: Callable[[Array, Array, Array], Any],
) -> None:
    with default_device(device):
        first, second = (jax.device_put(_inputs(seed, dtype), device) for seed in (6, 7))
        compiled = jax.jit(owned).lower(*first).compile()
        retained = compiled(*first)
        expected = _snapshot(retained)
        later = compiled(*second)
        _assert_same(retained, expected)
        _assert_same(retained, _snapshot(jax.jit(direct)(*first)))
        _assert_same(later, jax.jit(direct)(*second))


@_DEVICES
@pytest.mark.parametrize(
    "transform", (jax.grad, jax.hessian), ids=("gradient", "hessian")
)
def test_closed_over_tracers_stay_dynamic_and_differentiable(
    device: jax.Device, transform: Callable[..., Callable[..., Any]]
) -> None:
    position, velocity, gains = _inputs(8, jnp.float64)

    def owned(weight: Array, position: Array) -> Array:
        result = evaluate_with_result_storage(
            lambda q: _query(q, velocity * weight, gains), position
        )
        return result["energy"] + jnp.sum(result["acceleration"])

    def direct(weight: Array, position: Array) -> Array:
        result = _query(position, velocity * weight, gains)
        return result["energy"] + jnp.sum(result["acceleration"])

    with default_device(device):
        weight = jax.device_put(jnp.asarray(0.7, jnp.float64), device)
        moved = jax.device_put(position, device)
        velocity, gains = jax.device_put((velocity, gains), device)
        actual = jax.jit(transform(owned, argnums=(0, 1)))(weight, moved)
        expected = jax.jit(transform(direct, argnums=(0, 1)))(weight, moved)
        _assert_close(actual, expected)
        later = jax.device_put(jnp.asarray(1.3, jnp.float64), device)
        _assert_same(jax.jit(owned)(later, moved), jax.jit(direct)(later, moved))


@_DTYPES
@_DEVICES
def test_forward_derivatives_match(dtype: Any, device: jax.Device) -> None:
    operands = _inputs(9, dtype)
    tangents = _inputs(10, dtype)

    def owned(*values: Array) -> Array:
        return _owned(*values)["acceleration"]

    def direct(*values: Array) -> Array:
        return _query(*values)["acceleration"]

    with default_device(device):
        primals = jax.device_put(operands, device)
        directions = jax.device_put(tangents, device)
        actual = jax.jit(lambda p, t: jax.jvp(owned, p, t))(primals, directions)
        expected = jax.jit(lambda p, t: jax.jvp(direct, p, t))(primals, directions)
        _assert_close(actual, expected)


@_DTYPES
@_DEVICES
def test_linear_transpose_matches_reverse_cumulative_sum(
    dtype: Any, device: jax.Device
) -> None:
    with default_device(device):
        primal = _inputs(9, dtype)[2]
        cotangent = primal[::-1]

        def linear(value: Array) -> Array:
            return evaluate_with_result_storage(lambda x: 2.0 * jnp.cumsum(x), value)

        (actual,) = jax.jit(jax.linear_transpose(linear, primal))(cotangent)
        expected = 2.0 * jnp.cumsum(cotangent[::-1])[::-1]
        _assert_close(actual, expected)


def _accelerate(position: Array, gains: Array, velocity: Array) -> dict[str, Array]:
    return {
        "acceleration": _query(position, velocity, gains)["acceleration"],
        "gains": 2.0 * gains,
    }


@_DTYPES
@_DEVICES
@pytest.mark.parametrize("nested", (False, True), ids=("shared-leaf", "nested-worlds"))
def test_batching_preserves_each_result_batching(
    dtype: Any, device: jax.Device, nested: bool
) -> None:
    positions = jnp.stack([_inputs(seed, dtype)[0] for seed in (11, 12, 13)])
    _, velocity, gains = _inputs(14, dtype)

    def owned(position: Array, gains: Array) -> dict[str, Any]:
        return evaluate_with_result_storage(
            lambda q, g: _accelerate(q, g, velocity), position, gains
        )

    def direct(position: Array, gains: Array) -> dict[str, Any]:
        return _accelerate(position, gains, velocity)

    with default_device(device):
        mapped, shared, velocity = jax.device_put((positions, gains, velocity), device)
        if nested:
            scaled = jnp.stack([shared, 3.0 * shared])
            actual, expected = (
                jax.jit(
                    jax.vmap(jax.vmap(function, in_axes=(0, None)), in_axes=(None, 0))
                )(mapped, scaled)
                for function in (owned, direct)
            )
        else:
            out_axes = {"acceleration": 0, "gains": None}
            actual, expected = (
                jax.jit(jax.vmap(function, in_axes=(0, None), out_axes=out_axes))(
                    mapped, shared
                )
                for function in (owned, direct)
            )
        _assert_same(actual, expected)
