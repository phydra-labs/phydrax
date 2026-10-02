#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Private signature plans behind `phydrax.typing.checked`.

A plan is compiled lazily, once per decorated function, from its resolved
parameter annotations through `compile_input_form`. Each call reads the supported
arguments in declaration order against one fresh dimension `Scope` and forwards
the original arguments unchanged. Python's own binding rules stay authoritative:
a call that cannot bind raises its binding error before any contract error.
"""

from __future__ import annotations

import functools
import inspect
import sys
import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import FunctionType
from typing import assert_never, ForwardRef, Literal, NoReturn

from typing_extensions import evaluate_forward_ref, get_annotations

from ._typing_plan import (
    check,
    compile_input_form,
    Contract,
    NominalContract,
    OptionalContract,
    raise_violation,
    Scope,
    UnionContract,
    Violation,
)


type _SlotKind = Literal["single", "variadic-positional", "variadic-keyword"]
_EMPTY = inspect.Parameter.empty


@dataclass(frozen=True, slots=True)
class _Slot:
    """One checked parameter, where a call supplies it, and its fast acceptance.

    `accepts` holds the classes of a purely nominal contract (optional values
    included), so the common accepted call costs one `isinstance`; any other
    outcome runs the complete checker, which owns every violation.
    """

    name: str
    kind: _SlotKind
    contract: Contract
    path: str
    position: int | None
    keyword: bool
    default: object
    accepts: tuple[type, ...] | None


@dataclass(frozen=True, slots=True)
class _SignaturePlan:
    signature: inspect.Signature
    slots: tuple[_Slot, ...]
    positional_count: int
    named_keywords: frozenset[str]


def _accepted_classes(contract: Contract, /) -> tuple[type, ...] | None:
    match contract:
        case NominalContract(cls=cls):
            return (cls,)
        case OptionalContract(inner=inner):
            classes = _accepted_classes(inner)
            return None if classes is None else (*classes, type(None))
        case UnionContract(alternatives=alternatives):
            accepted: list[type] = []
            for alternative in alternatives:
                classes = _accepted_classes(alternative)
                if classes is None:
                    return None
                accepted.extend(classes)
            return tuple(accepted)
        case _:
            return None


def _owner_class(function: FunctionType, /) -> type | None:
    """Return the class defining `function` when it is reachable from its module."""
    *outer, _ = function.__qualname__.split(".")
    if "<locals>" in outer:
        return None
    namespace = vars(sys.modules[function.__module__])
    owner: type | None = None
    for name in outer:
        candidate = namespace.get(name)
        if not isinstance(candidate, type):
            return None
        owner = candidate
        namespace = vars(candidate)
    return owner


def _local_names(function: FunctionType, owner: type | None, /) -> dict[str, object]:
    """Return the class namespace and type parameters visible to annotations."""
    names: dict[str, object] = {} if owner is None else dict(vars(owner))
    type_params = function.__type_params__ + (
        () if owner is None else owner.__type_params__
    )
    names.update((parameter.__name__, parameter) for parameter in type_params)
    return names


def _resolve(
    function: FunctionType,
    name: str,
    annotation: object,
    owner: type | None,
    local_names: dict[str, object],
    /,
) -> object:
    if not isinstance(annotation, str):
        return annotation
    try:
        return evaluate_forward_ref(
            ForwardRef(annotation, module=function.__module__),
            owner=owner,
            globals=function.__globals__,
            locals=local_names,
        )
    except NameError as error:
        # Re-raised with the owning function and parameter: a checked signature
        # cannot depend on TYPE_CHECKING-only names.
        raise TypeError(
            f"{function.__qualname__} argument {name!r}: annotation {annotation!r} "
            f"does not resolve at runtime ({error})."
        ) from error


def _slot(
    function: FunctionType,
    parameter: inspect.Parameter,
    position: int,
    contract: Contract,
    /,
) -> _Slot:
    kind: _SlotKind
    match parameter.kind:
        case inspect.Parameter.POSITIONAL_ONLY:
            kind, index, keyword = "single", position, False
        case inspect.Parameter.POSITIONAL_OR_KEYWORD:
            kind, index, keyword = "single", position, True
        case inspect.Parameter.KEYWORD_ONLY:
            kind, index, keyword = "single", None, True
        case inspect.Parameter.VAR_POSITIONAL:
            kind, index, keyword = "variadic-positional", None, False
        case inspect.Parameter.VAR_KEYWORD:
            kind, index, keyword = "variadic-keyword", None, False
        case _:
            raise TypeError(f"Unsupported parameter kind {parameter.kind!r}.")
    return _Slot(
        parameter.name,
        kind,
        contract,
        f"{function.__qualname__} argument {parameter.name!r}",
        index,
        keyword,
        parameter.default,
        _accepted_classes(contract),
    )


def compile_signature(function: FunctionType, /) -> _SignaturePlan:
    """Compile the input contracts of one function's parameters."""
    signature = inspect.signature(function, follow_wrapped=False)
    annotations = get_annotations(function)
    owner = _owner_class(function)
    local_names = _local_names(function, owner)
    slots: list[_Slot] = []
    positional_count = 0
    for position, parameter in enumerate(signature.parameters.values()):
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
        if parameter.name not in annotations:
            continue
        form = _resolve(
            function, parameter.name, annotations[parameter.name], owner, local_names
        )
        try:
            contract = compile_input_form(form)
        except TypeError as error:
            raise TypeError(
                f"{function.__qualname__} argument {parameter.name!r}: {error}"
            ) from error
        if contract is not None:
            slots.append(_slot(function, parameter, position, contract))
    named_keywords = frozenset(
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    return _SignaturePlan(signature, tuple(slots), positional_count, named_keywords)


def _check_variadic(
    plan: _SignaturePlan,
    slot: _Slot,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    scope: Scope,
    /,
) -> Violation | None:
    items: list[tuple[int | str, object]]
    match slot.kind:
        case "variadic-positional":
            items = list(enumerate(args[plan.positional_count :]))
        case "variadic-keyword":
            items = [
                (key, value)
                for key, value in kwargs.items()
                if key not in plan.named_keywords
            ]
        case "single":
            raise TypeError(f"{slot.path} is not variadic.")
        case _:
            assert_never(slot.kind)
    for label, value in items:
        if slot.accepts is not None and isinstance(value, slot.accepts):
            continue
        violation = check(
            slot.contract, value, scope, f"{slot.path}[{label!r}]", canonicalize=False
        )[0]
        if violation is not None:
            return violation
    return None


def check_call(
    plan: _SignaturePlan, args: tuple[object, ...], kwargs: Mapping[str, object], /
) -> None:
    """Check one call's supported arguments in declaration order."""
    # Created on first need: purely nominal calls never bind a dimension.
    scope: Scope | None = None
    for slot in plan.slots:
        if slot.kind != "single":
            scope = Scope() if scope is None else scope
            violation = _check_variadic(plan, slot, args, kwargs, scope)
        else:
            position = slot.position
            if position is not None and position < len(args):
                value = args[position]
            elif slot.keyword and slot.name in kwargs:
                value = kwargs[slot.name]
            elif slot.default is not _EMPTY:
                value = slot.default
            else:
                # Missing: the call itself raises Python's binding error.
                continue
            if slot.accepts is not None and isinstance(value, slot.accepts):
                continue
            scope = Scope() if scope is None else scope
            violation = check(slot.contract, value, scope, slot.path, canonicalize=False)[
                0
            ]
        if violation is not None:
            _refuse(plan, violation, args, kwargs)


def _refuse(
    plan: _SignaturePlan,
    violation: Violation,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    /,
) -> NoReturn:
    # A call that cannot bind reports its binding error, never a contract error
    # read from a misplaced argument.
    plan.signature.bind(*args, **kwargs)
    raise_violation(violation)


class _LazyPlan:
    """Compile a signature plan on first use; failures are not cached."""

    __slots__ = ("_function", "_lock", "_plan", "__weakref__")

    def __init__(self, function: FunctionType, /) -> None:
        self._function = function
        self._lock = threading.Lock()
        self._plan: _SignaturePlan | None = None

    def get(self) -> _SignaturePlan:
        plan = self._plan
        if plan is not None:
            return plan
        with self._lock:
            if self._plan is None:
                self._plan = compile_signature(self._function)
            return self._plan


# A plan can reach its wrapper through a nominal owner or function globals.
# Weak values keep that cycle owned by the wrapper, not by the registry.
_CHECKED: weakref.WeakKeyDictionary[
    Callable[..., object], weakref.ReferenceType[_LazyPlan]
] = weakref.WeakKeyDictionary()


def checked[**P, R](function: Callable[P, R], /) -> Callable[P, R]:
    """Check a function's annotated arguments against their input contracts.

    Apply it directly to a Python function (closest to the `def`, beneath
    `staticmethod`, `classmethod`, `property`, and JAX transformations). The
    plan compiles on first call, so annotations may name classes defined later
    in the module, but they must resolve at runtime. Checked inputs are nominal
    runtime classes (subclasses included), callables, Phydrax tensor and metadata
    forms, and optional values, unions, and fixed tuples of them; all other
    annotations are static-only and remain with the owning validation.

    Arguments are checked in declaration order, omitted defaults included, and
    dimensions bind in one fresh `Scope` per call. Values are forwarded
    unchanged: nothing is converted, canonicalized, transferred, or traced, and
    the return value is not checked. Wrong kinds raise `TypeError`; wrong ranks,
    extents, and dimension bindings raise `ValueError`. A call that cannot bind
    raises Python's binding `TypeError` first.
    """
    if not isinstance(function, FunctionType):
        raise TypeError(
            f"checked decorates Python functions; got {type(function).__qualname__}. "
            "Apply it beneath staticmethod, classmethod, property, and transformations."
        )
    if function in _CHECKED:
        raise TypeError(f"{function.__qualname__} is already checked.")
    plan = _LazyPlan(function)

    @functools.wraps(function)
    def checked_call(*args: P.args, **kwargs: P.kwargs) -> R:
        check_call(plan._plan or plan.get(), args, kwargs)
        return function(*args, **kwargs)

    _CHECKED[checked_call] = weakref.ref(plan)
    return checked_call


def checked_plan(function: Callable[..., object], /) -> _SignaturePlan | None:
    """Return the compiled plan of a `checked` function, or `None` if unchecked."""
    reference = _CHECKED.get(function)
    if reference is None:
        return None
    plan = reference()
    if plan is None:
        raise RuntimeError("A live checked boundary has lost its signature plan.")
    return plan.get()
