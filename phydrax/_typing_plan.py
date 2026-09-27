#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Private structural contract compiler, binding scope, and checker.

The closed grammar compiled here is documented in `phydrax.typing`. Checks read
only host metadata (object kind, rank, extents, dtype, Python values of static
metadata): they add no JAX operations, synchronize nothing, and never mutate the
checked value.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import types
import weakref
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import (
    ForwardRef,
    get_args,
    get_origin,
    Literal,
    NoReturn,
    TypeAlias,
    TypeAliasType,
    Union,
)

import jax
import numpy as np
from typing_extensions import evaluate_forward_ref, get_annotations

from ._dtype_names import dtype_matches, DTypeRule
from ._typing_forms import (
    AnyDim,
    AnyShape,
    Backend,
    Broadcast,
    Dim,
    Identifier,
    Identifiers,
    Like,
    PRNGKey,
    Scalar,
    Size,
    TENSOR_FORMS,
    VariadicDim,
)
from ._validation import is_canonical_identifier


# --- contract representation -------------------------------------------------


@dataclass(frozen=True, slots=True)
class FixedExtent:
    size: int


@dataclass(frozen=True, slots=True)
class DimExtent:
    dim: type[Dim]


@dataclass(frozen=True, slots=True)
class AnyExtent:
    pass


@dataclass(frozen=True, slots=True)
class BroadcastExtent:
    dim: type[Dim]


@dataclass(frozen=True, slots=True)
class VariadicExtents:
    """A group of extents; `dim=None` is the anonymous group of `AnyShape`."""

    dim: type[VariadicDim] | None


ShapeTerm: TypeAlias = (
    FixedExtent | DimExtent | AnyExtent | BroadcastExtent | VariadicExtents
)


@dataclass(frozen=True, slots=True)
class TensorContract:
    """Backend-neutral dtype rule and ordered shape terms."""

    dtype: DTypeRule | None
    shape: tuple[ShapeTerm, ...]
    label: str


@dataclass(frozen=True, slots=True)
class ArrayContract:
    backend: Backend
    tensor: TensorContract


@dataclass(frozen=True, slots=True)
class KeyContract:
    pass


@dataclass(frozen=True, slots=True)
class SizeContract:
    dim: type[Dim]


@dataclass(frozen=True, slots=True)
class IdentifierContract:
    pass


@dataclass(frozen=True, slots=True)
class IdentifiersContract:
    dim: type[Dim]


@dataclass(frozen=True, slots=True)
class LiteralContract:
    values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class EnumContract:
    enum: type[Enum]


@dataclass(frozen=True, slots=True)
class OptionalContract:
    inner: Contract


@dataclass(frozen=True, slots=True)
class UnionContract:
    alternatives: tuple[Contract, ...]


@dataclass(frozen=True, slots=True)
class FixedTupleContract:
    items: tuple[Contract, ...]


Contract: TypeAlias = (
    ArrayContract
    | KeyContract
    | SizeContract
    | IdentifierContract
    | IdentifiersContract
    | LiteralContract
    | EnumContract
    | OptionalContract
    | UnionContract
    | FixedTupleContract
)


@dataclass(frozen=True, slots=True)
class FieldPlan:
    name: str
    contract: Contract


@dataclass(frozen=True, slots=True)
class ClassPlan:
    owner: str
    fields: tuple[FieldPlan, ...]


@dataclass(frozen=True, slots=True)
class Violation:
    """One failed check; `kind` selects TypeError (`"type"`) or ValueError."""

    kind: Literal["type", "value"]
    message: str


# --- compilation ----------------------------------------------------------------

_SCALAR_ALIASES: frozenset[object] = frozenset({PRNGKey, Identifier})
_SIZED_ALIASES: frozenset[object] = frozenset({Size, Identifiers})


def _is_dim(value: object, /) -> bool:
    return isinstance(value, type) and issubclass(value, Dim)


def _is_vocabulary(value: object, /) -> bool:
    if isinstance(value, TypeAliasType):
        return (
            value in TENSOR_FORMS
            or value in _SCALAR_ALIASES
            or value in _SIZED_ALIASES
            or value is Like
        )
    if isinstance(value, type):
        return issubclass(value, (Dim, AnyDim, AnyShape, Scalar, Broadcast))
    return False


def contains_vocabulary(form: object, /) -> bool:
    """Return whether `form` mentions any Phydrax typing vocabulary."""
    if _is_vocabulary(form) or _is_vocabulary(get_origin(form)):
        return True
    return any(contains_vocabulary(argument) for argument in get_args(form))


def _form_error(form: object, reason: str, /) -> NoReturn:
    raise TypeError(f"Unsupported Phydrax typing form {form!r}: {reason}.")


def _dimension(form: object, argument: object, /) -> type[Dim]:
    if (
        isinstance(argument, type)
        and issubclass(argument, Dim)
        and not issubclass(argument, VariadicDim)
    ):
        return argument
    _form_error(form, "the size argument must be a non-variadic Dim subclass")


def _shape_term(form: object, argument: object, /) -> ShapeTerm:
    if get_origin(argument) is Literal:
        (extent,) = get_args(argument) if len(get_args(argument)) == 1 else (None,)
        if type(extent) is not int or extent < 0:
            _form_error(form, "a fixed extent is Literal[n] with one nonnegative int n")
        return FixedExtent(extent)
    if get_origin(argument) is Broadcast:
        return BroadcastExtent(_dimension(form, get_args(argument)[0]))
    if argument is AnyDim:
        return AnyExtent()
    if isinstance(argument, type) and issubclass(argument, VariadicDim):
        return VariadicExtents(argument)
    if isinstance(argument, type) and issubclass(argument, Dim):
        return DimExtent(argument)
    _form_error(form, f"{argument!r} is not a shape term")


def _tensor(
    form: object, alias: TypeAliasType, arguments: tuple[object, ...], /
) -> ArrayContract:
    registration = TENSOR_FORMS[alias]
    if arguments in ((Scalar,), (AnyShape,)):
        shape: tuple[ShapeTerm, ...] = (
            () if arguments == (Scalar,) else (VariadicExtents(None),)
        )
    elif not arguments or Scalar in arguments or AnyShape in arguments:
        _form_error(form, "Scalar and AnyShape must be the only shape argument")
    else:
        shape = tuple(_shape_term(form, argument) for argument in arguments)
        if sum(isinstance(term, VariadicExtents) for term in shape) > 1:
            _form_error(form, "a tensor contract allows one variadic group")
    label = f"{registration.name}[{', '.join(_argument_label(argument) for argument in arguments)}]"
    return ArrayContract(
        registration.backend, TensorContract(registration.dtype, shape, label)
    )


def _argument_label(argument: object, /) -> str:
    if get_origin(argument) is Literal:
        return str(get_args(argument)[0])
    if get_origin(argument) is Broadcast:
        return f"Broadcast[{_argument_label(get_args(argument)[0])}]"
    return argument.__name__ if isinstance(argument, type) else repr(argument)


def _union(form: object, arguments: tuple[object, ...], /) -> Contract | None:
    optional = type(None) in arguments
    compiled = [
        compile_form(argument) for argument in arguments if argument is not type(None)
    ]
    if all(contract is None for contract in compiled):
        return None
    if any(contract is None for contract in compiled):
        _form_error(form, "a union cannot mix contract forms with static-only forms")
    alternatives = tuple(contract for contract in compiled if contract is not None)
    inner = alternatives[0] if len(alternatives) == 1 else UnionContract(alternatives)
    return OptionalContract(inner) if optional else inner


def _fixed_tuple(form: object, arguments: tuple[object, ...], /) -> Contract | None:
    if not arguments or arguments[-1] is Ellipsis or arguments == ((),):
        if contains_vocabulary(form):
            _form_error(form, "contracts inside variable-length tuples are unsupported")
        return None
    compiled = [compile_form(argument) for argument in arguments]
    if all(contract is None for contract in compiled):
        return None
    if any(contract is None for contract in compiled):
        _form_error(form, "a fixed tuple cannot mix contract and static-only items")
    return FixedTupleContract(
        tuple(contract for contract in compiled if contract is not None)
    )


def compile_form(form: object, /) -> Contract | None:
    """Compile one annotation; `None` means an ordinary static-only annotation."""
    origin = get_origin(form)
    arguments = get_args(form)
    if isinstance(form, TypeAliasType) and not _is_vocabulary(form):
        return compile_form(form.__value__)
    if isinstance(origin, TypeAliasType) and origin in TENSOR_FORMS:
        return _tensor(form, origin, arguments)
    if origin is Size:
        return SizeContract(_dimension(form, arguments[0]))
    if origin is Identifiers:
        return IdentifiersContract(_dimension(form, arguments[0]))
    if form is PRNGKey:
        return KeyContract()
    if form is Identifier:
        return IdentifierContract()
    if form is Like or origin is Like:
        _form_error(form, "Like describes conversion inputs, not contracts")
    if isinstance(form, TypeAliasType) and _is_vocabulary(form):
        _form_error(form, "this form requires its shape or size arguments")
    if origin is Literal:
        return LiteralContract(arguments)
    if isinstance(form, type) and issubclass(form, Enum):
        return EnumContract(form)
    if origin is Union or origin is types.UnionType:
        return _union(form, arguments)
    if origin is tuple:
        return _fixed_tuple(form, arguments)
    if contains_vocabulary(form):
        _form_error(form, "Phydrax contracts are unsupported in this placement")
    return None


@lru_cache(maxsize=4096)
def _compiled_parse_form(form: Hashable, /) -> Contract:
    contract = compile_form(form)
    if contract is None:
        _form_error(form, "parse accepts only Phydrax contracts, Literal, and Enum forms")
    return contract


def parse_contract(form: object, /) -> Contract:
    """Compile a form accepted by `phydrax.typing.parse`, caching the result."""
    if not isinstance(form, Hashable):
        _form_error(form, "forms must be hashable")
    return _compiled_parse_form(form)


# --- class plans ------------------------------------------------------------------

_CLASS_PLANS: weakref.WeakKeyDictionary[type, ClassPlan] = weakref.WeakKeyDictionary()
_CLASS_PLAN_LOCK = threading.Lock()


def field_annotation(cls: type, name: str, /) -> object:
    for owner in cls.__mro__:
        annotations = get_annotations(owner)
        if name not in annotations:
            continue
        annotation = annotations[name]
        if not isinstance(annotation, str):
            return annotation
        module = sys.modules[owner.__module__]
        try:
            return evaluate_forward_ref(
                ForwardRef(annotation, module=owner.__module__),
                owner=owner,
                globals=vars(module),
                locals=dict(vars(owner)),
            )
        except NameError as error:
            # Resolution failures are re-raised with the owning class and field.
            raise TypeError(
                f"{cls.__qualname__}.{name}: annotation {annotation!r} does not resolve "
                f"at runtime ({error}); contract fields cannot use TYPE_CHECKING-only names."
            ) from error
    raise TypeError(f"{cls.__qualname__}.{name} has no annotation.")


def _compile_class(cls: type, /) -> ClassPlan:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__qualname__} is not a dataclass-based module.")
    fields: list[FieldPlan] = []
    for field in dataclasses.fields(cls):
        annotation = field_annotation(cls, field.name)
        try:
            contract = compile_form(annotation)
        except TypeError as error:
            raise TypeError(f"{cls.__qualname__}.{field.name}: {error}") from error
        if contract is not None:
            fields.append(FieldPlan(field.name, contract))
    if not fields:
        raise TypeError(f"{cls.__qualname__} declares no Phydrax contract fields.")
    return ClassPlan(cls.__qualname__, tuple(fields))


def class_plan(cls: type, /) -> ClassPlan:
    """Return the compiled plan of `cls`, compiling it once per class."""
    plan = _CLASS_PLANS.get(cls)
    if plan is not None:
        return plan
    with _CLASS_PLAN_LOCK:
        plan = _CLASS_PLANS.get(cls)
        if plan is None:
            plan = _compile_class(cls)
            _CLASS_PLANS[cls] = plan
    return plan


# --- binding scope ----------------------------------------------------------------

_Binding: TypeAlias = tuple[int | tuple[int, ...], str]


class Scope:
    """Dimension bindings shared by the checks of one validation operation.

    A scope maps each bound dimension to its extent (or extents, for a variadic
    group) and to the field that established it. Failed union alternatives roll
    their bindings back.
    """

    __slots__ = ("_bindings", "_log")

    def __init__(self) -> None:
        self._bindings: dict[type[Dim], _Binding] = {}
        self._log: list[type[Dim]] = []

    def size(self, dim: type[Dim], /) -> int:
        """Return the extent bound to one ordinary dimension."""
        value = self._lookup(dim)
        if isinstance(value, tuple):
            raise TypeError(f"{dim.__name__} is variadic; use shape().")
        return value

    def shape(self, dims: type[VariadicDim], /) -> tuple[int, ...]:
        """Return the extents bound to one variadic dimension group."""
        value = self._lookup(dims)
        if not isinstance(value, tuple):
            raise TypeError(f"{dims.__name__} is not variadic; use size().")
        return value

    def _lookup(self, dim: type[Dim], /) -> int | tuple[int, ...]:
        if not _is_dim(dim):
            raise TypeError("Scope lookups take a Dim subclass.")
        if dim not in self._bindings:
            raise ValueError(f"{dim.__name__} is not bound in this scope.")
        return self._bindings[dim][0]

    def _mark(self) -> int:
        return len(self._log)

    def _rollback(self, mark: int, /) -> None:
        while len(self._log) > mark:
            del self._bindings[self._log.pop()]

    def _bind(
        self, dim: type[Dim], value: int | tuple[int, ...], origin: str, /
    ) -> Violation | None:
        bound = self._bindings.get(dim)
        if bound is None:
            self._bindings[dim] = (value, origin)
            self._log.append(dim)
            return None
        if bound[0] == value:
            return None
        return Violation(
            "value",
            f"{origin}: {dim.__name__}={value} conflicts with {dim.__name__}={bound[0]} "
            f"bound by {bound[1]}",
        )


# --- checks -----------------------------------------------------------------------

_Outcome: TypeAlias = tuple[Violation | None, object]


def _type_violation(path: str, expected: str, value: object, /) -> _Outcome:
    return Violation(
        "type", f"{path}: expected {expected}; got {type(value).__name__}."
    ), value


def _describe_array(dtype: object, shape: tuple[object, ...], /) -> str:
    return f"{dtype}[{', '.join(str(extent) for extent in shape)}]"


def _check_terms(
    terms: tuple[ShapeTerm, ...], shape: tuple[int, ...], scope: Scope, path: str, /
) -> Violation | None:
    variadic = [
        index for index, term in enumerate(terms) if isinstance(term, VariadicExtents)
    ]
    if not variadic:
        if len(shape) != len(terms):
            return Violation(
                "value", f"{path}: expected rank {len(terms)}; got rank {len(shape)}."
            )
        pairs = list(zip(terms, shape, strict=True))
    else:
        index = variadic[0]
        tail = len(terms) - index - 1
        if len(shape) < len(terms) - 1:
            return Violation(
                "value",
                f"{path}: expected rank at least {len(terms) - 1}; got rank {len(shape)}.",
            )
        group = terms[index]
        middle = shape[index : len(shape) - tail]
        if isinstance(group, VariadicExtents) and group.dim is not None:
            if len(middle) < group.dim.minimum:
                return Violation(
                    "value",
                    f"{path}: {group.dim.__name__} needs at least {group.dim.minimum} extents.",
                )
            violation = scope._bind(group.dim, tuple(middle), path)
            if violation is not None:
                return violation
        pairs = [
            *zip(terms[:index], shape[:index]),
            *zip(terms[index + 1 :], shape[len(shape) - tail :]),
        ]
    for term, extent in pairs:
        violation = _check_term(term, extent, scope, path)
        if violation is not None:
            return violation
    return None


def _check_term(
    term: ShapeTerm, extent: int, scope: Scope, path: str, /
) -> Violation | None:
    match term:
        case FixedExtent(size=size):
            if extent != size:
                return Violation(
                    "value", f"{path}: expected extent {size}; got {extent}."
                )
            return None
        case AnyExtent():
            return None
        case BroadcastExtent(dim=dim):
            return None if extent == 1 else _bind_extent(dim, extent, scope, path)
        case DimExtent(dim=dim):
            return _bind_extent(dim, extent, scope, path)
        case VariadicExtents():
            raise AssertionError("variadic groups are matched before extents")


def _bind_extent(
    dim: type[Dim], extent: int, scope: Scope, path: str, /
) -> Violation | None:
    if extent < dim.minimum:
        return Violation(
            "value",
            f"{path}: {dim.__name__} must be at least {dim.minimum}; got {extent}.",
        )
    return scope._bind(dim, extent, path)


def _check_array(
    contract: ArrayContract, value: object, scope: Scope, path: str, /
) -> _Outcome:
    tensor = contract.tensor
    if contract.backend == "jax":
        if not isinstance(value, jax.Array):
            return _type_violation(path, f"JAX array {tensor.label}", value)
    elif not isinstance(value, np.ndarray):
        return _type_violation(path, f"NumPy array {tensor.label}", value)
    shape = tuple(value.shape)
    if not all(type(extent) is int for extent in shape):
        return Violation(
            "type", f"{path}: symbolic extents are unsupported; got {shape}."
        ), value
    violation = _check_terms(tensor.shape, shape, scope, path)
    if (
        violation is None
        and tensor.dtype is not None
        and not dtype_matches(tensor.dtype, value.dtype)
    ):
        violation = Violation(
            "type",
            f"{path}: expected {tensor.label} with dtype {tensor.dtype.label}; "
            f"got {_describe_array(value.dtype, shape)}.",
        )
    return violation, value


def _check_key(value: object, path: str, /) -> _Outcome:
    if not isinstance(value, jax.Array):
        return _type_violation(path, "a typed JAX PRNG key", value)
    if not jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        return Violation(
            "type",
            f"{path}: expected a typed PRNG key from jax.random.key; got dtype {value.dtype}.",
        ), value
    if value.shape != ():
        return Violation(
            "value", f"{path}: expected one scalar key; got shape {value.shape}."
        ), value
    return None, value


def _check_size(
    contract: SizeContract, value: object, scope: Scope, path: str, /
) -> _Outcome:
    if type(value) is not int:
        return _type_violation(path, f"int Size[{contract.dim.__name__}]", value)
    return _bind_extent(contract.dim, value, scope, path), value


def _check_identifiers(
    contract: IdentifiersContract, value: object, scope: Scope, path: str, /
) -> _Outcome:
    if not isinstance(value, tuple):
        return _type_violation(path, f"tuple Identifiers[{contract.dim.__name__}]", value)
    for item in value:
        if not isinstance(item, str):
            return _type_violation(f"{path} item", "str identifier", item)
        if not is_canonical_identifier(item):
            return Violation(
                "value", f"{path}: {item!r} is not a canonical identifier."
            ), value
    if len(set(value)) != len(value):
        return Violation("value", f"{path}: identifiers must be unique."), value
    return _bind_extent(contract.dim, len(value), scope, path), value


def _check_literal(
    contract: LiteralContract, value: object, path: str, canonicalize: bool, /
) -> _Outcome:
    for literal in contract.values:
        if type(value) is type(literal) and value == literal:
            return None, literal
    if canonicalize and not isinstance(
        value, (np.ndarray, jax.Array, tuple, list, dict, set, frozenset)
    ):
        for literal in contract.values:
            if value == literal:
                return None, literal
    kinds = {type(literal) for literal in contract.values}
    kind: Literal["type", "value"] = "value" if type(value) in kinds else "type"
    return Violation(
        kind, f"{path}: expected one of {contract.values!r}; got {value!r}."
    ), value


def _check_union(
    alternatives: tuple[Contract, ...],
    value: object,
    scope: Scope,
    path: str,
    canonicalize: bool,
    /,
) -> _Outcome:
    failures: list[Violation] = []
    for alternative in alternatives:
        mark = scope._mark()
        violation, result = check(
            alternative, value, scope, path, canonicalize=canonicalize
        )
        if violation is None:
            return None, result
        scope._rollback(mark)
        failures.append(violation)
    kind: Literal["type", "value"] = (
        "type" if all(item.kind == "type" for item in failures) else "value"
    )
    return Violation(kind, " | ".join(item.message for item in failures)), value


def _check_tuple(
    contract: FixedTupleContract,
    value: object,
    scope: Scope,
    path: str,
    canonicalize: bool,
    /,
) -> _Outcome:
    if not isinstance(value, tuple):
        return _type_violation(path, f"tuple of {len(contract.items)} items", value)
    if len(value) != len(contract.items):
        return Violation(
            "value", f"{path}: expected {len(contract.items)} items; got {len(value)}."
        ), value
    results: list[object] = []
    for index, (item_contract, item) in enumerate(
        zip(contract.items, value, strict=True)
    ):
        violation, result = check(
            item_contract, item, scope, f"{path}[{index}]", canonicalize=canonicalize
        )
        if violation is not None:
            return violation, value
        results.append(result)
    same = all(result is item for result, item in zip(results, value, strict=True))
    return None, value if same else tuple(results)


def check(
    contract: Contract, value: object, scope: Scope, path: str, /, *, canonicalize: bool
) -> _Outcome:
    """Check `value` against `contract`, returning a violation and the accepted value.

    With `canonicalize`, Literal alternatives accept equal values and return the
    declared literal; without it, the exact declared runtime type is required.
    """
    match contract:
        case ArrayContract():
            return _check_array(contract, value, scope, path)
        case KeyContract():
            return _check_key(value, path)
        case SizeContract():
            return _check_size(contract, value, scope, path)
        case IdentifierContract():
            if not isinstance(value, str):
                return _type_violation(path, "str identifier", value)
            if not is_canonical_identifier(value):
                return Violation(
                    "value", f"{path}: {value!r} is not a canonical identifier."
                ), value
            return None, value
        case IdentifiersContract():
            return _check_identifiers(contract, value, scope, path)
        case LiteralContract():
            return _check_literal(contract, value, path, canonicalize)
        case EnumContract(enum=enum):
            if not isinstance(value, enum):
                return _type_violation(path, f"{enum.__name__} member", value)
            return None, value
        case OptionalContract(inner=inner):
            return (
                (None, value)
                if value is None
                else check(inner, value, scope, path, canonicalize=canonicalize)
            )
        case UnionContract(alternatives=alternatives):
            return _check_union(alternatives, value, scope, path, canonicalize)
        case FixedTupleContract():
            return _check_tuple(contract, value, scope, path, canonicalize)


def raise_violation(violation: Violation, /) -> NoReturn:
    """Raise the exception category selected by one violation."""
    match violation.kind:
        case "type":
            raise TypeError(violation.message)
        case "value":
            raise ValueError(violation.message)


def validate_instance(instance: object, /) -> None:
    """Check every contract field of one module in declaration order."""
    plan = class_plan(type(instance))
    scope = Scope()
    for field in plan.fields:
        violation, _ = check(
            field.contract,
            getattr(instance, field.name),
            scope,
            f"{plan.owner}.{field.name}",
            canonicalize=False,
        )
        if violation is not None:
            raise_violation(violation)


def validate_constructed(instance: object, /) -> None:
    """Check one freshly constructed opted-in module; called by the strict metaclass."""
    validate_instance(instance)


def validate_tree(root: object, /) -> None:
    """Check every opted-in strict module reachable from `root`.

    Traversal follows dataclass fields (static and dynamic), tuples, lists, and
    mappings, visits each object once (cycle-safe), and never descends into
    arrays, callables, or other objects.
    """
    # Lazy: the strict metaclass imports this module.
    from ._strict import Strict

    seen: set[int] = set()
    stack: list[object] = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, type):
            continue
        if dataclasses.is_dataclass(value):
            if isinstance(value, Strict) and type(value)._strict_contract_:
                validate_instance(value)
            children = [getattr(value, field.name) for field in dataclasses.fields(value)]
        elif isinstance(value, tuple | list):
            children = list(value)
        elif isinstance(value, Mapping):
            children = list(value.values())
        else:
            continue
        stack.extend(reversed(children))
