#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.layers import Linear


def test_initializers_scenario_1() -> None:
    for initializer, scale, mode in (
        ("lecun_normal", 1.0, "fan_in"),
        ("lecun_uniform", 1.0, "fan_in"),
        ("he_normal", 2.0, "fan_in"),
        ("he_uniform", 2.0, "fan_in"),
        ("glorot_normal", 1.0, "fan_avg"),
        ("glorot_uniform", 1.0, "fan_avg"),
    ):
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


def test_initializer_aliases_are_exact() -> None:
    for canonical, alias in (
        ("he_normal", "kaiming_normal"),
        ("he_uniform", "kaiming_uniform"),
        ("glorot_normal", "xavier_normal"),
        ("glorot_uniform", "xavier_uniform"),
    ):

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


def test_orthogonal_initializer_handles_rectangular_weights() -> None:
    for in_size, out_size in ((3, 8), (8, 3)):

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
