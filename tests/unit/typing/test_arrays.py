from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import export

import phydrax.typing as pt


class NodeDim(pt.Dim):
    pass


class ComponentDim(pt.Dim, minimum=1):
    pass


class BatchDims(pt.VariadicDim):
    pass


def test_exact_dtype_forms_accept_only_their_dtype() -> None:
    values = jnp.zeros((3,), dtype=jnp.float64)

    assert pt.parse(values, pt.Float64[NodeDim], "values") is values
    with pytest.raises(TypeError):
        pt.parse(values.astype(jnp.float32), pt.Float64[NodeDim], "values")
    with pytest.raises(TypeError):
        pt.parse(jnp.zeros((3,), dtype=jnp.int32), pt.Float64[NodeDim], "values")


@pytest.mark.parametrize(
    ("form", "dtype", "accepted"),
    [
        (pt.Float, jnp.bfloat16, True),
        (pt.Float, jnp.float16, True),
        (pt.Float, jnp.complex64, False),
        (pt.Inexact, jnp.complex128, True),
        (pt.Integer, jnp.uint8, True),
        (pt.Integer, jnp.bool_, False),
        (pt.Complex, jnp.complex64, True),
        (pt.Bool, jnp.bool_, True),
        (pt.Shaped, jnp.int8, True),
    ],
)
def test_category_forms_follow_the_jax_dtype_hierarchy(
    form: Any, dtype: Any, accepted: Any
) -> None:
    value = jnp.zeros((2,), dtype=dtype)
    if accepted:
        assert pt.parse(value, form[NodeDim], "value") is value
    else:
        with pytest.raises(TypeError):
            pt.parse(value, form[NodeDim], "value")


def test_shape_terms_bind_fixed_named_repeated_and_anonymous_extents() -> None:
    square = jnp.eye(3)
    pt.parse(square, pt.Float64[NodeDim, NodeDim], "square")
    pt.parse(square, pt.Float64[Literal[3], pt.AnyDim], "square")
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((3, 2)), pt.Float64[NodeDim, NodeDim], "rectangle")
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((2, 3)), pt.Float64[Literal[3], pt.AnyDim], "rows")
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((3,)), pt.Float64[NodeDim, NodeDim], "rank")


def test_scalar_and_any_shape_are_standalone_shapes() -> None:
    pt.parse(jnp.asarray(1.0), pt.Float64[pt.Scalar], "scalar")
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((1,)), pt.Float64[pt.Scalar], "scalar")
    pt.parse(jnp.zeros((2, 3, 4)), pt.Float64[pt.AnyShape], "any")
    pt.parse(jnp.asarray(1.0), pt.Float64[pt.AnyShape], "any")


def test_minimum_extent_is_a_value_contract() -> None:
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((0,)), pt.Float64[ComponentDim], "components")


def test_broadcast_extents_accept_one_or_the_bound_extent() -> None:
    scope = pt.Scope()
    pt.parse(jnp.zeros((1,)), pt.Float64[pt.Broadcast[NodeDim]], "unit", scope=scope)
    with pytest.raises(ValueError):
        scope.size(NodeDim)
    pt.parse(jnp.zeros((4,)), pt.Float64[pt.Broadcast[NodeDim]], "wide", scope=scope)
    assert scope.size(NodeDim) == 4
    pt.parse(jnp.zeros((1,)), pt.Float64[pt.Broadcast[NodeDim]], "unit", scope=scope)
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((3,)), pt.Float64[pt.Broadcast[NodeDim]], "other", scope=scope)


def test_variadic_groups_bind_zero_or_more_extents() -> None:
    scope = pt.Scope()
    pt.parse(jnp.zeros((2, 5, 3)), pt.Float64[BatchDims, NodeDim], "batched", scope=scope)
    assert scope.shape(BatchDims) == (2, 5)
    pt.parse(jnp.zeros((2, 5)), pt.Float64[BatchDims], "same", scope=scope)
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((2,)), pt.Float64[BatchDims], "different", scope=scope)
    pt.parse(jnp.zeros((3,)), pt.Float64[BatchDims, NodeDim], "unbatched")


def test_backends_are_distinct_kinds() -> None:
    with pytest.raises(TypeError):
        pt.parse(np.zeros((3,)), pt.Float64[NodeDim], "host")
    with pytest.raises(TypeError):
        pt.parse(jnp.zeros((3,)), pt.HostFloat64[NodeDim], "device")
    with pytest.raises(TypeError):
        pt.parse(1.0, pt.Float64[pt.Scalar], "python")
    pt.parse(np.zeros((3,)), pt.HostFloat64[NodeDim], "host")


def test_typed_keys_are_the_only_prng_keys() -> None:
    key = jax.random.key(0)

    assert pt.parse(key, pt.PRNGKey, "key") is key
    with pytest.raises(TypeError):
        pt.parse(jax.random.PRNGKey(0), pt.PRNGKey, "legacy")
    with pytest.raises(ValueError):
        pt.parse(jax.random.split(key, 2), pt.PRNGKey, "keys")


def test_checks_run_at_trace_time_on_tracers_without_adding_operations() -> None:
    def body(values: Any) -> Any:
        pt.parse(values, pt.Float64[NodeDim, ComponentDim], "values")
        return values

    jaxpr = jax.make_jaxpr(body)(jnp.zeros((2, 3)))
    assert not jaxpr.jaxpr.eqns
    jax.jit(body)(jnp.zeros((2, 3)))
    jax.grad(lambda values: jnp.sum(body(values)))(jnp.ones((2, 3)))
    jax.eval_shape(body, jax.ShapeDtypeStruct((2, 3), jnp.float64))
    with pytest.raises(ValueError):
        jax.jit(body)(jnp.zeros((2, 0)))


def test_vmap_checks_the_per_example_shape() -> None:
    def body(values: Any) -> Any:
        pt.parse(values, pt.Float64[ComponentDim], "example")
        return values

    jax.vmap(body)(jnp.zeros((5, 3)))
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((5, 3)), pt.Float64[ComponentDim], "stacked")


def test_scan_bodies_are_checked_once_during_tracing() -> None:
    def step(carry: Any, row: Any) -> Any:
        pt.parse(row, pt.Float64[ComponentDim], "row")
        return carry + jnp.sum(row), row

    total, _ = jax.lax.scan(step, jnp.asarray(0.0), jnp.ones((4, 3)))
    assert total == 12.0


def test_symbolic_extents_are_refused() -> None:
    (n,) = export.symbolic_shape("n")

    def body(values: Any) -> Any:
        pt.parse(values, pt.Float64[NodeDim], "symbolic")
        return values

    with pytest.raises(TypeError):
        jax.eval_shape(body, jax.ShapeDtypeStruct((n,), jnp.float64))
