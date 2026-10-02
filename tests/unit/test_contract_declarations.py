"""Repository-wide structural contract declarations."""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib
import inspect
import pkgutil
import sys
from types import FunctionType

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax import _typing_plan, _typing_signature, StrictModule
from phydrax._strict import Strict


_VOCABULARY_NAMES = frozenset(
    name
    for name in phx.typing.__all__
    if name not in ("Scope", "as_array", "as_host_array", "checked", "parse", "validate")
)


@functools.cache
def _strict_dataclasses() -> tuple[type, ...]:
    for info in pkgutil.walk_packages(phx.__path__, "phydrax."):
        importlib.import_module(info.name)
    classes: list[type] = []
    stack: list[type] = [StrictModule]
    seen: set[type] = set()
    while stack:
        for child in stack.pop().__subclasses__():
            if child in seen:
                continue
            seen.add(child)
            stack.append(child)
            if child.__module__.startswith("phydrax") and dataclasses.is_dataclass(child):
                classes.append(child)
    return tuple(sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__)))


def _class_functions(cls: type, /) -> list[object]:
    functions: list[object] = []
    for value in vars(cls).values():
        match value:
            case classmethod() | staticmethod():
                functions.append(value.__func__)
            case property():
                functions.extend((value.fget, value.fset, value.fdel))
            case type() if value.__qualname__.startswith(f"{cls.__qualname__}."):
                functions.extend(_class_functions(value))
            case _:
                functions.append(value)
    return functions


@functools.cache
def _checked_boundaries() -> tuple[FunctionType, ...]:
    _strict_dataclasses()
    boundaries: dict[FunctionType, None] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "phydrax" and not name.startswith("phydrax."):
            continue
        for value in vars(module).values():
            owned_class = isinstance(value, type) and value.__module__ == name
            for function in _class_functions(value) if owned_class else [value]:
                if (
                    isinstance(function, FunctionType)
                    and _typing_signature.checked_plan(function) is not None
                ):
                    boundaries[function] = None
    return tuple(boundaries)


def _mentions_vocabulary(annotation: str, /) -> bool:
    names = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(ast.parse(annotation, mode="eval"))
        if isinstance(node, ast.Name | ast.Attribute)
    }
    return bool(names & _VOCABULARY_NAMES)


def _contract_fields(cls: type, /) -> tuple[str, ...]:
    names: list[str] = []
    # ty: ignore[invalid-argument-type]
    for field in dataclasses.fields(cls):
        owner = next(
            base for base in cls.__mro__ if field.name in inspect.get_annotations(base)
        )
        annotation = inspect.get_annotations(owner)[field.name]
        if isinstance(annotation, str):
            if not _mentions_vocabulary(annotation):
                continue
            annotation = _typing_plan.field_annotation(cls, field.name)
        if _typing_plan.contains_vocabulary(annotation):
            names.append(field.name)
    return tuple(names)


def test_contract_declarations_scenario_1() -> None:
    undeclared = [
        f"{cls.__module__}.{cls.__qualname__}: {fields}"
        for cls in _strict_dataclasses()
        # ty: ignore[unresolved-attribute]
        if not cls._strict_contract_ and (fields := _contract_fields(cls))
    ]
    assert undeclared == []
    # ty: ignore[unresolved-attribute]
    opted = [cls for cls in _strict_dataclasses() if cls._strict_contract_]
    assert opted
    for cls in opted:
        assert _typing_plan.class_plan(cls).fields

    class Holder(StrictModule):
        values: jnp.ndarray

        def scale(self, factor: phx.typing.Float64[phx.typing.Scalar], /) -> None:
            del factor

    assert not Holder._strict_contract_
    assert _contract_fields(Holder) == ()

    class Unresolved(StrictModule):
        __strict_contract__ = True

        # ty: ignore[unresolved-reference]
        values: phx.typing.Float64[MissingDim]  # noqa: F821

    with pytest.raises(TypeError, match="Unresolved.values"):
        Unresolved(jnp.zeros((1,)))


def test_contract_declarations_scenario_2() -> None:
    with pytest.raises(TypeError):

        class Withdrawn(StrictModule):
            __strict_contract__ = False

            values: jnp.ndarray

    with pytest.raises(TypeError):

        class PlainStrict(Strict):
            __strict_contract__ = True


def test_every_checked_boundary_resolves_and_enforces_an_input_contract() -> None:
    boundaries = _checked_boundaries()
    assert boundaries
    plans = {
        f"{function.__module__}.{function.__qualname__}": _typing_signature.checked_plan(
            function
        )
        for function in boundaries
    }
    ineffective = [name for name, plan in plans.items() if plan is None or not plan.slots]
    assert ineffective == []
