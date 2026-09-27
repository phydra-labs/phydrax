from __future__ import annotations

import enum
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from jax import export

import phydrax.typing as pt
from tests._support.strategies import finite_arrays


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


class NodeDim(pt.Dim):
    pass


class ComponentDim(pt.Dim, minimum=1):
    pass


class BatchDims(pt.VariadicDim):
    pass


Basis: TypeAlias = Literal["nodal", "modal"]
type ModeLiteral = Literal["direct", "iterative"]


class Mode(enum.Enum):
    DENSE = "dense"
    SPARSE = "sparse"


class Flag(enum.IntEnum):
    ON = 1


def test_forms_scenario_1() -> None:
    assert ComponentDim.minimum == 1
    assert BatchDims.minimum == 0
    with pytest.raises(TypeError):
        ComponentDim()
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        class FractionalDim(pt.Dim, minimum=1.5):
            pass

    with pytest.raises(ValueError):

        class NegativeDim(pt.Dim, minimum=-1):
            pass

    for token in (pt.AnyDim, pt.AnyShape, pt.Scalar):
        with pytest.raises(TypeError):
            token()
        with pytest.raises(TypeError):
            type("Extended", (token,), {})

    invalid_forms: tuple[Any, ...] = (
        pt.Float64,
        pt.Size,
        pt.Identifiers,
        # ty: ignore[invalid-type-form]
        pt.Float64[3],
        pt.Float64[Literal[-1]],
        pt.Float64[Literal[1, 2]],
        pt.Float64[BatchDims, BatchDims],
        pt.Float64[pt.AnyShape, ComponentDim],
        pt.Float64[pt.Scalar, ComponentDim],
        pt.Float64[pt.Broadcast[BatchDims]],
        pt.Size[BatchDims],
        pt.Like[pt.Float64[ComponentDim]],
        list[pt.Float64[ComponentDim]],
        tuple[pt.Float64[ComponentDim], ...],
        pt.Float64[ComponentDim] | int,
    )
    for form in invalid_forms:
        with pytest.raises(TypeError):
            pt.parse(None, form, "value")

    class components(pt.Dim):
        pass

    assert pt.parse(3, pt.Size[components], "count") == 3
    assert pt.parse("direct", ModeLiteral, "mode") == "direct"
    with pytest.raises(ValueError, match="mode"):
        pt.parse("invalid", ModeLiteral, "mode")
    exact = jnp.zeros((3,), dtype=jnp.float64)
    assert pt.parse(exact, pt.Float64[NodeDim], "values") is exact
    for rejected in (
        exact.astype(jnp.float32),
        jnp.zeros((3,), dtype=jnp.int32),
    ):
        with pytest.raises(TypeError):
            pt.parse(rejected, pt.Float64[NodeDim], "values")

    cases: tuple[tuple[Any, npt.DTypeLike, bool], ...] = (
        (pt.Float, jnp.bfloat16, True),
        (pt.Float, jnp.float16, True),
        (pt.Float, jnp.complex64, False),
        (pt.Inexact, jnp.complex128, True),
        (pt.Integer, jnp.uint8, True),
        (pt.Integer, jnp.bool_, False),
        (pt.Complex, jnp.complex64, True),
        (pt.Bool, jnp.bool_, True),
        (pt.Shaped, jnp.int8, True),
    )
    for form, dtype, accepted in cases:
        value = jnp.zeros((2,), dtype=dtype)
        if accepted:
            assert pt.parse(value, form[NodeDim], "value") is value
        else:
            with pytest.raises(TypeError):
                pt.parse(value, form[NodeDim], "value")
    square = jnp.eye(3)
    pt.parse(square, pt.Float64[NodeDim, NodeDim], "square")
    pt.parse(square, pt.Float64[Literal[3], pt.AnyDim], "square")
    for rejected, form in (
        (jnp.zeros((3, 2)), pt.Float64[NodeDim, NodeDim]),
        (jnp.zeros((2, 3)), pt.Float64[Literal[3], pt.AnyDim]),
        (jnp.zeros((3,)), pt.Float64[NodeDim, NodeDim]),
        (jnp.zeros((0,)), pt.Float64[ComponentDim]),
    ):
        with pytest.raises(ValueError):
            pt.parse(rejected, form, "shape")

    pt.parse(jnp.asarray(1.0), pt.Float64[pt.Scalar], "scalar")
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((1,)), pt.Float64[pt.Scalar], "scalar")
    pt.parse(jnp.zeros((2, 3, 4)), pt.Float64[pt.AnyShape], "any")
    pt.parse(jnp.asarray(1.0), pt.Float64[pt.AnyShape], "any")

    broadcast_scope = pt.Scope()
    pt.parse(
        jnp.zeros((1,)),
        pt.Float64[pt.Broadcast[NodeDim]],
        "unit",
        scope=broadcast_scope,
    )
    with pytest.raises(ValueError):
        broadcast_scope.size(NodeDim)
    pt.parse(
        jnp.zeros((4,)),
        pt.Float64[pt.Broadcast[NodeDim]],
        "wide",
        scope=broadcast_scope,
    )
    pt.parse(
        jnp.zeros((1,)),
        pt.Float64[pt.Broadcast[NodeDim]],
        "unit",
        scope=broadcast_scope,
    )
    assert broadcast_scope.size(NodeDim) == 4
    with pytest.raises(ValueError):
        pt.parse(
            jnp.zeros((3,)),
            pt.Float64[pt.Broadcast[NodeDim]],
            "other",
            scope=broadcast_scope,
        )

    variadic_scope = pt.Scope()
    pt.parse(
        jnp.zeros((2, 5, 3)),
        pt.Float64[BatchDims, NodeDim],
        "batched",
        scope=variadic_scope,
    )
    assert variadic_scope.shape(BatchDims) == (2, 5)
    pt.parse(jnp.zeros((2, 5)), pt.Float64[BatchDims], "same", scope=variadic_scope)
    with pytest.raises(ValueError):
        pt.parse(
            jnp.zeros((2,)), pt.Float64[BatchDims], "different", scope=variadic_scope
        )
    pt.parse(jnp.zeros((3,)), pt.Float64[BatchDims, NodeDim], "unbatched")


@given(finite_arrays(dtype=np.float64, shapes=((), (1,), (2, 3))))
def test_exact_float_form_accepts_finite_host_generated_values(
    values: npt.NDArray[np.generic],
) -> None:
    device = jnp.asarray(values)
    assert pt.parse(device, pt.Float64[pt.AnyShape], "values") is device


def test_forms_scenario_2() -> None:
    refusals: tuple[tuple[Callable[[], object], type[Exception]], ...] = (
        (lambda: pt.parse(np.zeros((3,)), pt.Float64[NodeDim], "host"), TypeError),
        (lambda: pt.parse(jnp.zeros((3,)), pt.HostFloat64[NodeDim], "device"), TypeError),
        (lambda: pt.parse(1.0, pt.Float64[pt.Scalar], "python"), TypeError),
    )
    for operation, error in refusals:
        with pytest.raises(error):
            operation()
    host = np.zeros((3,))
    assert pt.parse(host, pt.HostFloat64[NodeDim], "host") is host

    key = jax.random.key(0)
    assert pt.parse(key, pt.PRNGKey, "key") is key
    with pytest.raises(TypeError):
        pt.parse(jax.random.PRNGKey(0), pt.PRNGKey, "legacy")
    with pytest.raises(ValueError):
        pt.parse(jax.random.split(key, 2), pt.PRNGKey, "keys")
    assert pt.parse(3, pt.Size[ComponentDim], "count") == 3
    for wrong_kind in (True, np.int64(3), Flag.ON, 3.0):
        with pytest.raises(TypeError):
            pt.parse(wrong_kind, pt.Size[ComponentDim], "count")
    with pytest.raises(ValueError):
        pt.parse(0, pt.Size[ComponentDim], "count")

    assert pt.parse("h2o", pt.Identifier, "name") == "h2o"
    with pytest.raises(TypeError):
        pt.parse(b"h2o", pt.Identifier, "name")
    for invalid in ("", " h2o", "h2o\n"):
        with pytest.raises(ValueError):
            pt.parse(invalid, pt.Identifier, "name")

    scope = pt.Scope()
    names = ("h2", "o2")
    assert pt.parse(names, pt.Identifiers[ComponentDim], "names", scope=scope) is names
    assert scope.size(ComponentDim) == 2
    for invalid in (["h2", "o2"], ("h2", 2)):
        with pytest.raises(TypeError):
            pt.parse(invalid, pt.Identifiers[ComponentDim], "names")
    for invalid in (("h2", "h2"), ()):
        with pytest.raises(ValueError):
            pt.parse(invalid, pt.Identifiers[ComponentDim], "names")
    parsed = pt.parse(np.str_("nodal"), Basis, "basis")
    assert parsed == "nodal" and type(parsed) is str
    assert pt.parse("modal", Basis, "basis") == "modal"
    with pytest.raises(ValueError):
        pt.parse("spectral", Basis, "basis")
    with pytest.raises(TypeError):
        pt.parse(np.asarray(["nodal"]), Basis, "basis")

    assert type(pt.parse(True, Literal[1, True], "flag")) is bool
    integer = pt.parse(True, Literal[1], "flag")
    assert integer == 1 and type(integer) is int
    assert pt.parse(Mode.DENSE, Literal["dense"] | Mode, "mode") is Mode.DENSE
    assert pt.parse(Mode.SPARSE, Mode, "mode") is Mode.SPARSE
    with pytest.raises(TypeError):
        pt.parse("sparse", Mode, "mode")

    assert pt.parse(None, pt.Identifier | None, "parent") is None
    pair = pt.parse((2, "h2"), tuple[pt.Size[ComponentDim], pt.Identifier], "pair")
    assert pair == (2, "h2")
    canonical = pt.parse(
        (np.str_("modal"), 2),
        tuple[Basis, pt.Size[ComponentDim]],
        "pair",
    )
    assert type(canonical[0]) is str
    with pytest.raises(ValueError):
        pt.parse((2,), tuple[pt.Size[ComponentDim], pt.Identifier], "pair")


def test_contract_checks_trace_without_numerical_operations() -> None:
    def matrix_body(values: jax.Array) -> jax.Array:
        pt.parse(values, pt.Float64[NodeDim, ComponentDim], "values")
        return values

    jaxpr = jax.make_jaxpr(matrix_body)(jnp.zeros((2, 3)))
    assert not jaxpr.jaxpr.eqns
    jax.jit(matrix_body)(jnp.zeros((2, 3)))
    jax.grad(lambda values: jnp.sum(matrix_body(values)))(jnp.ones((2, 3)))
    jax.eval_shape(matrix_body, jax.ShapeDtypeStruct((2, 3), jnp.float64))
    with pytest.raises(ValueError):
        jax.jit(matrix_body)(jnp.zeros((2, 0)))

    def vector_body(values: jax.Array) -> jax.Array:
        pt.parse(values, pt.Float64[ComponentDim], "example")
        return values

    jax.vmap(vector_body)(jnp.zeros((5, 3)))
    with pytest.raises(ValueError):
        pt.parse(jnp.zeros((5, 3)), pt.Float64[ComponentDim], "stacked")

    def step(carry: jax.Array, row: jax.Array) -> tuple[jax.Array, jax.Array]:
        pt.parse(row, pt.Float64[ComponentDim], "row")
        return carry + jnp.sum(row), row

    total, _ = jax.lax.scan(step, jnp.asarray(0.0), jnp.ones((4, 3)))
    assert total == 12.0

    (symbolic_size,) = export.symbolic_shape("n")
    with pytest.raises(TypeError):
        jax.eval_shape(
            vector_body,
            jax.ShapeDtypeStruct((symbolic_size,), jnp.float64),
        )
