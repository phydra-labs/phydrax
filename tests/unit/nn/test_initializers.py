#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.layers import Linear


@pytest.mark.parametrize(
    ("initializer", "scale", "mode"),
    (
        ("lecun_normal", 1.0, "fan_in"),
        ("lecun_uniform", 1.0, "fan_in"),
        ("he_normal", 2.0, "fan_in"),
        ("he_uniform", 2.0, "fan_in"),
        ("glorot_normal", 1.0, "fan_avg"),
        ("glorot_uniform", 1.0, "fan_avg"),
    ),
)
def test_named_initializers_match_target_variance(
    initializer: Any, scale: Any, mode: Any
) -> None:
    in_size = 192
    out_size = 640
    layer = Linear(
        in_size=in_size,
        out_size=out_size,
        initializer=initializer,
        rwf=False,
        use_bias=False,
        key=jr.key(0),
    )
    denominator = in_size if mode == "fan_in" else (in_size + out_size) / 2
    target_variance = scale / denominator

    # ty: ignore[invalid-argument-type]
    assert jnp.isclose(jnp.var(layer.weight), target_variance, rtol=0.03)
    # ty: ignore[invalid-argument-type]
    assert jnp.abs(jnp.mean(layer.weight)) < 0.02 * jnp.sqrt(target_variance)


@pytest.mark.parametrize(
    ("canonical", "alias"),
    (
        ("he_normal", "kaiming_normal"),
        ("he_uniform", "kaiming_uniform"),
        ("glorot_normal", "xavier_normal"),
        ("glorot_uniform", "xavier_uniform"),
    ),
)
def test_initializer_aliases_are_exact(canonical: Any, alias: Any) -> None:
    def weight(initializer: Any) -> Any:
        return Linear(
            in_size=17,
            out_size=29,
            initializer=initializer,
            rwf=False,
            use_bias=False,
            key=jr.key(1),
        ).weight

    assert jnp.array_equal(weight(canonical), weight(alias))


@pytest.mark.parametrize(("in_size", "out_size"), ((3, 8), (8, 3)))
def test_orthogonal_initializer_handles_rectangular_weights(
    in_size: Any, out_size: Any
) -> None:
    def weight() -> Any:
        return Linear(
            in_size=in_size,
            out_size=out_size,
            initializer="orthogonal",
            rwf=False,
            use_bias=False,
            key=jr.key(2),
        ).weight

    first = weight()
    second = weight()
    gram = first.T @ first if out_size >= in_size else first @ first.T

    assert first.shape == (out_size, in_size)
    assert jnp.all(jnp.isfinite(first))
    assert jnp.array_equal(first, second)
    assert jnp.allclose(gram, jnp.eye(min(in_size, out_size)), atol=1e-5)


def test_default_initializer_remains_glorot_normal() -> None:
    kwargs = {
        "in_size": 11,
        "out_size": 7,
        "rwf": False,
        "use_bias": False,
        "key": jr.key(3),
    }

    # ty: ignore[invalid-argument-type]
    default = Linear(**kwargs)
    # ty: ignore[invalid-argument-type]
    explicit = Linear(initializer="glorot_normal", **kwargs)

    # ty: ignore[invalid-argument-type]
    assert jnp.array_equal(default.weight, explicit.weight)
