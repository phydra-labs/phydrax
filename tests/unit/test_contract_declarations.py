"""Repository-wide structural contract declarations."""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib
import inspect
import pkgutil

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax import _typing_plan, StrictModule
from phydrax._strict import Strict


_VOCABULARY_NAMES = frozenset(
    name
    for name in phx.typing.__all__
    if name not in ("Scope", "as_array", "as_host_array", "parse", "validate")
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


def _mentions_vocabulary(annotation: str, /) -> bool:
    names = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(ast.parse(annotation, mode="eval"))
        if isinstance(node, ast.Name | ast.Attribute)
    }
    return bool(names & _VOCABULARY_NAMES)


def _contract_fields(cls: type, /) -> tuple[str, ...]:
    names: list[str] = []
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


def test_every_contract_field_belongs_to_an_opted_in_module():
    undeclared = [
        f"{cls.__module__}.{cls.__qualname__}: {fields}"
        for cls in _strict_dataclasses()
        if not cls._strict_contract_ and (fields := _contract_fields(cls))
    ]
    assert undeclared == []


def test_every_opted_in_module_compiles():
    opted = [cls for cls in _strict_dataclasses() if cls._strict_contract_]
    assert opted
    for cls in opted:
        assert _typing_plan.class_plan(cls).fields


def test_ordinary_function_annotations_do_not_opt_a_class_in():
    class Holder(StrictModule):
        values: jnp.ndarray

        def scale(self, factor: phx.typing.Float64[phx.typing.Scalar], /) -> None:
            del factor

    assert not Holder._strict_contract_
    assert _contract_fields(Holder) == ()


def test_unresolved_contract_annotations_fail_with_class_and_field_context():
    class Unresolved(StrictModule):
        __strict_contract__ = True

        values: phx.typing.Float64[MissingDim]  # noqa: F821

    with pytest.raises(TypeError, match="Unresolved.values"):
        Unresolved(jnp.zeros((1,)))


def test_opt_in_may_only_be_declared_true():
    with pytest.raises(TypeError):

        class Withdrawn(StrictModule):
            __strict_contract__ = False

            values: jnp.ndarray


def test_classes_that_are_not_dataclass_modules_cannot_opt_in():
    with pytest.raises(TypeError):

        class PlainStrict(Strict):
            __strict_contract__ = True
