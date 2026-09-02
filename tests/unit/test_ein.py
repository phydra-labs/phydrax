#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import jax
import jax.numpy as jnp
import opt_einsum as oe
import pytest

import phydrax as phx


def test_contract_boundary_is_exact_and_jittable():
    assert phx.ein.__all__ == ["contract", "rearrange", "reduce", "repeat"]
    assert phx.ein.contract is oe.contract

    values = jnp.arange(6.0).reshape(2, 3)
    weight = jnp.arange(12.0).reshape(3, 4)
    apply = jax.jit(
        lambda x, matrix: phx.ein.contract(
            "...i,ij->...j",
            x,
            matrix,
            backend="jax",
        )
    )

    result = apply(values, weight)
    expected = values @ weight
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert jnp.array_equal(result, expected)


def test_rearrange_regroups_reorders_singletons_and_ellipsis():
    values = jnp.arange(2 * 12 * 5, dtype=jnp.int32).reshape(2, 12, 5)
    result = phx.ein.rearrange(
        values,
        "batch (height patch) channel -> batch height channel patch",
        patch=4,
    )
    expected = values.reshape(2, 3, 4, 5).transpose(0, 1, 3, 2)
    assert result.dtype == values.dtype
    assert jnp.array_equal(result, expected)

    restored = phx.ein.rearrange(
        result,
        "batch height channel patch -> batch (height patch) channel",
    )
    assert jnp.array_equal(restored, values)

    singleton = phx.ein.rearrange(
        jnp.arange(6).reshape(2, 3),
        "batch channel -> 1 channel batch 1",
    )
    assert singleton.shape == (1, 3, 2, 1)
    assert jnp.array_equal(singleton[0, :, :, 0], jnp.arange(6).reshape(2, 3).T)

    ellipsis = jnp.arange(24).reshape(2, 3, 4)
    collapsed = phx.ein.rearrange(ellipsis, "... channel -> channel (...)")
    assert jnp.array_equal(collapsed, ellipsis.transpose(2, 0, 1).reshape(4, 6))

    scalar = jnp.asarray(3.0)
    grouped_empty_ellipsis = phx.ein.rearrange(scalar, "... -> (...)")
    assert grouped_empty_ellipsis.shape == (1,)
    assert grouped_empty_ellipsis[0] == scalar

    empty = jnp.empty((0,))
    factored_empty = phx.ein.rearrange(
        empty,
        "(row column) -> row column",
        row=2,
    )
    assert factored_empty.shape == (2, 0)


@pytest.mark.parametrize(
    ("name", "function", "values"),
    [
        ("sum", jnp.sum, jnp.arange(24).reshape(2, 3, 4)),
        ("mean", jnp.mean, jnp.arange(24.0).reshape(2, 3, 4)),
        ("prod", jnp.prod, jnp.arange(1, 25).reshape(2, 3, 4)),
        ("min", jnp.min, jnp.arange(24).reshape(2, 3, 4)),
        ("max", jnp.max, jnp.arange(24).reshape(2, 3, 4)),
        ("all", jnp.all, jnp.asarray([True, False] * 12).reshape(2, 3, 4)),
        ("any", jnp.any, jnp.asarray([True, False] * 12).reshape(2, 3, 4)),
    ],
)
def test_reduce_matches_jax_primitives(
    name: str,
    function: Callable,
    values: jax.Array,
):
    result = phx.ein.reduce(
        values,
        "batch time channel -> batch channel",
        name,
    )
    expected = function(values, axis=1)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert jnp.array_equal(result, expected)


def test_reduce_preserves_jax_zero_axis_semantics():
    numeric = jnp.empty((2, 0, 3), dtype=jnp.int32)
    boolean = jnp.empty((2, 0, 3), dtype=jnp.bool_)

    for name, function, values in (
        ("sum", jnp.sum, numeric),
        ("prod", jnp.prod, numeric),
        ("all", jnp.all, boolean),
        ("any", jnp.any, boolean),
    ):
        result = phx.ein.reduce(
            values,
            "batch time channel -> batch channel",
            name,
        )
        expected = function(values, axis=1)
        assert result.dtype == expected.dtype
        assert jnp.array_equal(result, expected)

    floating = numeric.astype(jnp.float64)
    mean = phx.ein.reduce(
        floating,
        "batch time channel -> batch channel",
        "mean",
    )
    expected_mean = jnp.mean(floating, axis=1)
    assert mean.dtype == expected_mean.dtype
    assert jnp.allclose(mean, expected_mean, equal_nan=True)

    for name in ("min", "max"):
        with pytest.raises(ValueError, match="zero-size array"):
            phx.ein.reduce(
                numeric,
                "batch time channel -> batch channel",
                name,
            )


def test_repeat_broadcasts_new_axes_without_tiling_semantics():
    values = jnp.arange(6).reshape(2, 3)
    repeated = phx.ein.repeat(
        values,
        "batch channel -> batch replica channel",
        replica=4,
    )
    expected = jnp.broadcast_to(values[:, None, :], (2, 4, 3))
    assert jnp.array_equal(repeated, expected)

    grouped = phx.ein.repeat(
        values,
        "batch channel -> batch (replica channel)",
        replica=4,
    )
    assert jnp.array_equal(grouped, expected.reshape(2, 12))

    scalar = phx.ein.repeat(jnp.asarray(2.0), "-> replica", replica=3)
    assert jnp.array_equal(scalar, jnp.full((3,), 2.0))

    empty = phx.ein.repeat(
        jnp.empty((0, 3)),
        "batch channel -> batch replica channel",
        replica=2,
    )
    assert empty.shape == (0, 2, 3)


def test_transforms_compose_under_jit_grad_jvp_and_vmap():
    weight = jnp.arange(12.0).reshape(3, 4) / 7.0

    def objective(values, matrix):
        heads = phx.ein.rearrange(
            values,
            "batch (head channel) -> batch head channel",
            head=2,
        )
        replicated = phx.ein.repeat(
            heads,
            "batch head channel -> batch replica head channel",
            replica=2,
        )
        projected = phx.ein.contract(
            "brhc,cf->brhf",
            replicated,
            matrix,
            backend="jax",
        )
        return phx.ein.reduce(
            projected,
            "batch replica head feature ->",
            "sum",
        )

    def reference(values, matrix):
        heads = values.reshape(values.shape[0], 2, 3)
        replicated = jnp.broadcast_to(
            heads[:, None, :, :],
            (values.shape[0], 2, 2, 3),
        )
        return jnp.sum(replicated @ matrix)

    values = jnp.arange(18.0).reshape(3, 6) / 5.0
    assert jnp.allclose(objective(values, weight), reference(values, weight))
    assert jnp.allclose(
        jax.jit(objective)(values, weight),
        jax.jit(reference)(values, weight),
    )
    assert jnp.allclose(
        jax.grad(objective)(values, weight),
        jax.grad(reference)(values, weight),
    )

    tangent = jnp.ones_like(values)
    _, transformed_tangent = jax.jvp(
        lambda x: objective(x, weight),
        (values,),
        (tangent,),
    )
    _, reference_tangent = jax.jvp(
        lambda x: reference(x, weight),
        (values,),
        (tangent,),
    )
    assert jnp.allclose(transformed_tangent, reference_tangent)

    ensemble = jnp.stack([values, values + 1.0])
    transformed_batch = jax.vmap(lambda x: objective(x, weight))(ensemble)
    reference_batch = jax.vmap(lambda x: reference(x, weight))(ensemble)
    assert jnp.allclose(transformed_batch, reference_batch)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: phx.ein.rearrange(jnp.ones((2,)), "axis"), "exactly one"),
        (
            lambda: phx.ein.rearrange(jnp.ones((2,)), "((axis)) -> axis"),
            "nested groups",
        ),
        (
            lambda: phx.ein.rearrange(jnp.ones((2,)), "2 -> 1"),
            "only anonymous literal",
        ),
        (
            lambda: phx.ein.rearrange(jnp.ones((2, 2)), "axis axis -> axis"),
            "appears more than once",
        ),
        (
            lambda: phx.ein.rearrange(jnp.ones((2, 2)), "axis -> axis"),
            "input rank",
        ),
        (
            lambda: phx.ein.rearrange(jnp.ones((2,)), "axis -> axis", other=2),
            "unused axis",
        ),
        (
            lambda: phx.ein.rearrange(
                jnp.ones((6,)),
                "(row column) -> row column",
            ),
            "multiple unresolved factors",
        ),
        (
            lambda: phx.ein.rearrange(
                jnp.ones((5,)),
                "(row column) -> row column",
                row=2,
            ),
            "not divisible",
        ),
        (
            lambda: phx.ein.rearrange(jnp.ones((2,)), "axis -> ... axis"),
            "requires an input ellipsis",
        ),
        (
            lambda: phx.ein.rearrange(jnp.ones((2, 3)), "row column -> row"),
            "cannot remove",
        ),
        (
            lambda: phx.ein.reduce(
                jnp.ones((2,)),
                "axis -> axis other",
                "sum",
            ),
            "cannot add",
        ),
        (
            lambda: phx.ein.repeat(jnp.ones((2, 3)), "row column -> row"),
            "cannot remove",
        ),
        (
            lambda: phx.ein.repeat(jnp.ones((2,)), "axis -> copy axis"),
            "requires a size",
        ),
        (
            lambda: phx.ein.reduce(jnp.asarray(1.0), "... ->", "sum"),
            "must remove at least one axis",
        ),
        (
            lambda: phx.ein.repeat(jnp.ones((2,)), "axis -> axis"),
            "must add at least one named axis",
        ),
        (
            lambda: phx.ein.reduce(jnp.ones((2,)), "axis ->", "median"),
            "unsupported reduction",
        ),
        (
            lambda: phx.ein.repeat(
                jnp.ones((2,)),
                "axis -> copy axis",
                copy=True,
            ),
            "static integer",
        ),
        (
            lambda: phx.ein.repeat(
                jnp.ones((2,)),
                "axis -> copy axis",
                copy=0,
            ),
            "must be positive",
        ),
    ],
)
def test_invalid_patterns_and_shapes_fail_with_context(call: Callable, message: str):
    with pytest.raises((TypeError, ValueError), match=message) as error:
        call()
    assert "^" in str(error.value)


def test_dynamic_axis_sizes_are_rejected_during_tracing():
    transform = jax.jit(
        lambda values, copies: phx.ein.repeat(
            values,
            "axis -> copy axis",
            copy=copies,
        )
    )
    with pytest.raises(ValueError, match="static integer"):
        transform(jnp.ones((2,)), jnp.asarray(2))
