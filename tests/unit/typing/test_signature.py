from __future__ import annotations

import gc
import sys
import threading
import weakref
from collections.abc import Callable
from types import FunctionType, ModuleType
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.typing as pt
from phydrax import StrictModule


pytestmark = pytest.mark.strict_jax


class NodeDim(pt.Dim):
    pass


class AbstractShape(StrictModule):
    pass


class Square(AbstractShape):
    side: float = eqx.field(static=True, default=1.0)


class Circle(AbstractShape):
    radius: float = eqx.field(static=True, default=1.0)


class Label(StrictModule):
    text: str = eqx.field(static=True, default="label")


@pt.checked
def describe(
    shape: AbstractShape,
    hint: Label | None = None,
    *,
    mode: Literal["area", "perimeter"] = "area",
    count: int = 1,
) -> tuple[AbstractShape, Label | None, str, int]:
    return shape, hint, mode, count


@pt.checked
def either(item: Square | Label) -> Square | Label:
    return item


@pt.checked
def weighted(
    values: pt.Float64[NodeDim],
    weights: pt.Float64[NodeDim],
    reduce: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    return reduce(values * weights)


@pt.checked
def collect(*shapes: AbstractShape, **labels: Label) -> int:
    return len(shapes) + len(labels)


@pt.checked
def lenient(
    value: Square | int, items: tuple[Square, ...], data: pt.Like[pt.Float64[NodeDim]]
) -> object:
    return value


@pytest.mark.parametrize(
    ("shape", "hint"),
    [
        pytest.param(Square(), None, id="abstract-base-accepts-subclass"),
        pytest.param(Circle(), Label(), id="optional-present"),
    ],
)
def test_nominal_arguments_are_forwarded_unchanged(
    shape: AbstractShape, hint: Label | None
) -> None:
    positional = describe(shape, hint)
    keyword = describe(shape, hint=hint)

    assert positional[0] is shape and positional[1] is hint
    assert keyword[0] is shape and keyword[1] is hint


@pytest.mark.parametrize(
    ("arguments", "keywords", "argument"),
    [
        pytest.param((Label(),), {}, "shape", id="wrong-positional-kind"),
        pytest.param((Square(), Square()), {}, "hint", id="wrong-optional-kind"),
        pytest.param((Square(),), {"hint": "label"}, "hint", id="wrong-keyword-kind"),
        pytest.param((), {"shape": None}, "shape", id="none-not-declared"),
    ],
)
def test_wrong_nominal_kind_raises_type_error_naming_the_argument(
    arguments: tuple[object, ...], keywords: dict[str, object], argument: str
) -> None:
    with pytest.raises(TypeError, match=rf"describe argument '{argument}'"):
        describe(*arguments, **keywords)  # ty: ignore[invalid-argument-type]


def test_union_accepts_each_alternative_and_refuses_other_kinds() -> None:
    square, label = Square(), Label()

    assert either(square) is square
    assert either(label) is label
    with pytest.raises(TypeError, match="either argument 'item'"):
        either(Circle())  # ty: ignore[invalid-argument-type]


def test_selector_scalar_and_conversion_annotations_stay_with_their_owner() -> None:
    # Literal selectors are parsed and canonicalized by the owner; plain scalars
    # keep the owner's admission (NumPy integers, for example) and conversion
    # inputs are converted by the owner.
    _, _, mode, count = describe(
        Square(),
        mode=np.str_("perimeter"),  # ty: ignore[invalid-argument-type]
        count=np.int64(3),  # ty: ignore[invalid-argument-type]
    )

    assert type(mode) is np.str_
    assert type(count) is np.int64
    assert lenient(3, (Square(),), [1.0, 2.0]) == 3


def test_static_only_union_member_does_not_partially_enforce_the_union() -> None:
    unchecked = "not checked"
    assert lenient(unchecked, (), np.ones(2)) is unchecked  # ty: ignore[invalid-argument-type]


def test_shared_dimension_binds_across_arguments_within_one_call() -> None:
    three = jnp.ones(3, dtype=jnp.float64)

    assert float(weighted(three, three, jnp.sum)) == 3.0
    with pytest.raises(ValueError, match="NodeDim=4 conflicts with NodeDim=3"):
        weighted(three, jnp.ones(4, dtype=jnp.float64), jnp.sum)


def test_dimension_bindings_do_not_leak_between_calls() -> None:
    three, four = jnp.ones(3, dtype=jnp.float64), jnp.ones(4, dtype=jnp.float64)

    assert float(weighted(three, three, jnp.sum)) == 3.0
    assert float(weighted(four, four, jnp.sum)) == 4.0


@pytest.mark.parametrize(
    ("values", "reduce", "error", "argument"),
    [
        pytest.param(
            np.ones(3), jnp.sum, TypeError, "values", id="host-array-not-converted"
        ),
        pytest.param(
            jnp.ones(3, dtype=jnp.float32), jnp.sum, TypeError, "values", id="wrong-dtype"
        ),
        pytest.param(
            jnp.ones((3, 1), dtype=jnp.float64),
            jnp.sum,
            ValueError,
            "values",
            id="wrong-rank",
        ),
        pytest.param(
            jnp.ones(3, dtype=jnp.float64), 3, TypeError, "reduce", id="not-callable"
        ),
    ],
)
def test_structural_argument_violations_have_contract_categories(
    values: object, reduce: object, error: type[Exception], argument: str
) -> None:
    weights = jnp.ones(3, dtype=jnp.float64)

    with pytest.raises(error, match=rf"weighted argument '{argument}'"):
        weighted(values, weights, reduce)  # ty: ignore[invalid-argument-type]


def test_variadic_annotations_describe_each_member() -> None:
    assert collect(Square(), Circle(), first=Label()) == 3
    with pytest.raises(TypeError, match=r"collect argument 'shapes'\[1\]"):
        collect(Square(), Label())  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"collect argument 'labels'\['first'\]"):
        collect(first=Square())  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("arguments", "keywords", "message"),
    [
        pytest.param((Label(), None, None), {}, "positional", id="too-many-positional"),
        pytest.param((Label(),), {"shape": Square()}, "multiple values", id="duplicate"),
        pytest.param(
            (Label(),), {"unknown": 1}, "unexpected keyword", id="unknown-keyword"
        ),
        pytest.param((), {"hint": Square()}, "missing", id="missing-required"),
    ],
)
def test_binding_errors_take_precedence_over_contract_errors(
    arguments: tuple[object, ...], keywords: dict[str, object], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        describe(*arguments, **keywords)  # ty: ignore[invalid-argument-type]


def test_refused_call_never_enters_the_body() -> None:
    entered: list[object] = []

    @pt.checked
    def record(shape: AbstractShape) -> None:
        entered.append(shape)

    with pytest.raises(TypeError):
        record(Label())  # ty: ignore[invalid-argument-type]

    assert entered == []


def test_body_exceptions_and_results_propagate_unchanged() -> None:
    sentinel = object()

    @pt.checked
    def fail(shape: AbstractShape) -> object:
        raise LookupError(sentinel)

    @pt.checked
    def identity(shape: AbstractShape, value: object) -> object:
        return value

    with pytest.raises(LookupError) as caught:
        fail(Square())
    assert caught.value.args == (sentinel,)
    assert identity(Square(), sentinel) is sentinel


class Container(StrictModule):
    shape: AbstractShape
    values: jax.Array

    @pt.checked
    def __init__(
        self, shape: AbstractShape, values: pt.Like[pt.Float64[NodeDim]]
    ) -> None:
        self.shape = shape
        self.values = jnp.asarray(values, dtype=jnp.float64)

    def __check_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("values must be a vector")

    @pt.checked
    def replace(self, shape: AbstractShape) -> Container:
        return Container(shape, self.values)

    @classmethod
    @pt.checked
    def of(cls, shape: AbstractShape) -> Container:
        return cls(shape, [0.0])

    @staticmethod
    @pt.checked
    def measure(shape: AbstractShape) -> float:
        return 1.0

    @pt.checked
    def link(self, other: Container) -> Container:
        return other


def test_checked_constructor_converts_inputs_then_runs_owner_validation() -> None:
    container = Container(Square(), [1.0, 2.0])

    assert container.values.dtype == jnp.float64
    with pytest.raises(TypeError, match=r"Container.__init__ argument 'shape'"):
        Container(Label(), [1.0])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="values must be a vector"):
        Container(Square(), [[1.0]])


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda c: c.replace(Label()), id="instance-method"),
        pytest.param(
            lambda c: Container.of(Label()),  # ty: ignore[invalid-argument-type]
            id="classmethod",
        ),
        pytest.param(
            lambda c: Container.measure(Label()),  # ty: ignore[invalid-argument-type]
            id="staticmethod",
        ),
        pytest.param(lambda c: c.link(Square()), id="forward-reference-to-owner"),
    ],
)
def test_method_descriptors_check_their_own_arguments(
    call: Callable[[Container], object],
) -> None:
    container = Container(Square(), [1.0])

    with pytest.raises(TypeError, match="argument"):
        call(container)


def test_methods_bind_and_return_through_descriptors() -> None:
    container = Container(Square(), [1.0])
    circle = Circle()

    assert container.replace(circle).shape is circle
    assert Container.of(circle).shape is circle
    assert Container.measure(circle) == 1.0
    assert container.link(container) is container


def test_unresolvable_annotation_fails_on_every_call_with_its_argument() -> None:
    @pt.checked
    def broken(value: UndefinedName) -> None:  # noqa: F821  # ty: ignore[unresolved-reference]
        pass

    for _ in range(2):
        with pytest.raises(TypeError, match=r"broken argument 'value'.*does not resolve"):
            broken(1)


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param("list[pt.Float64[NodeDim]]", id="vocabulary-in-container"),
        pytest.param("pt.Float64[NodeDim] | int", id="vocabulary-with-static-only"),
    ],
)
def test_vocabulary_in_unsupported_placement_is_refused(annotation: str) -> None:
    def function(value: object) -> None:
        pass

    function.__annotations__ = {"value": annotation, "return": "None"}
    guarded = pt.checked(function)

    with pytest.raises(TypeError, match="function argument 'value'"):
        guarded(jnp.ones(2, dtype=jnp.float64))


def _dynamic_boundary_references() -> tuple[
    weakref.ReferenceType[type], weakref.ReferenceType[FunctionType]
]:
    module = ModuleType("runtime_contract_ephemeral")
    sys.modules[module.__name__] = module
    try:
        exec(
            "from phydrax.typing import checked\n"
            "class Node:\n"
            "    @checked\n"
            "    def pair(self, other: 'Node') -> 'Node':\n"
            "        return other\n",
            vars(module),
        )
        owner = vars(module)["Node"]
        if not isinstance(owner, type):
            raise TypeError("The dynamic fixture owner must be a class.")
        boundary = vars(owner)["pair"]
        if not isinstance(boundary, FunctionType):
            raise TypeError("The dynamic fixture boundary must be a function.")
        instance = owner()
        assert boundary(instance, instance) is instance
        return weakref.ref(owner), weakref.ref(boundary)
    finally:
        sys.modules.pop(module.__name__)


def test_checked_cache_releases_dynamic_owner_and_boundary() -> None:
    owner, boundary = _dynamic_boundary_references()

    gc.collect()

    assert owner() is None
    assert boundary() is None


def test_first_calls_compile_one_plan_concurrently() -> None:
    @pt.checked
    def concurrent(shape: AbstractShape, values: pt.Float64[NodeDim]) -> int:
        return values.shape[0]

    barrier = threading.Barrier(8)
    results: list[int] = []
    errors: list[BaseException] = []

    def worker(size: int) -> None:
        barrier.wait()
        try:
            results.append(concurrent(Square(), jnp.ones(size, dtype=jnp.float64)))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(size,)) for size in range(1, 9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(results) == list(range(1, 9))


def test_compiled_callable_checks_while_tracing() -> None:
    @jax.jit
    @pt.checked
    def scale(values: pt.Float64[NodeDim], shape: Square) -> jax.Array:
        return shape.side * values

    values = jnp.arange(3.0, dtype=jnp.float64)

    assert jnp.array_equal(scale(values, Square(side=2.0)), 2.0 * values)
    with pytest.raises(TypeError, match="scale argument 'shape'"):
        scale(values, Circle())


def test_custom_derivative_primal_keeps_its_rule() -> None:
    @jax.custom_jvp
    @pt.checked
    def soft(values: jax.Array, shape: AbstractShape) -> jax.Array:
        return jnp.sin(values)

    @soft.defjvp
    def soft_jvp(
        primals: tuple[jax.Array, AbstractShape], tangents: tuple[jax.Array, object]
    ) -> tuple[jax.Array, jax.Array]:
        values, _ = primals
        return jnp.sin(values), 2.0 * tangents[0]

    values = jnp.arange(3.0, dtype=jnp.float64)
    tangent = jnp.ones(3, dtype=jnp.float64)

    primal_out, tangent_out = jax.jvp(lambda v: soft(v, Square()), (values,), (tangent,))
    assert jnp.array_equal(primal_out, jnp.sin(values))
    assert jnp.array_equal(tangent_out, 2.0 * tangent)
    with pytest.raises(TypeError, match="soft argument 'shape'"):
        soft(values, Label())


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(staticmethod(len), id="descriptor"),
        pytest.param(describe, id="already-checked"),
    ],
)
def test_checked_refuses_non_functions_and_repeated_decoration(target: object) -> None:
    with pytest.raises(TypeError, match="checked"):
        pt.checked(target)  # ty: ignore[invalid-argument-type]
