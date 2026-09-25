#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Repository invariants for Equinox module fields and abstract attributes."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import pkgutil
import re
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._strict import StrictModule


_STRINGIFIED_ABSTRACT = re.compile(r"\s*(?:eqx\.|equinox\.)?Abstract(?:Class)?Var\[")
_CLASS_BODY_CALLABLES = (property, classmethod, staticmethod, types.FunctionType)


def _subclasses(root: type, /) -> set[type]:
    seen: set[type] = set()
    stack = [root]
    while stack:
        for child in stack.pop().__subclasses__():
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


@functools.cache
def _modules() -> tuple[type, ...]:
    for info in pkgutil.walk_packages(phx.__path__, "phydrax."):
        try:
            importlib.import_module(info.name)
        except ImportError:
            # Optional-dependency modules cannot define built-in classes here.
            continue
    return tuple(
        sorted(
            (
                cls
                for cls in _subclasses(eqx.Module)
                if cls.__module__.startswith("phydrax")
            ),
            key=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
        )
    )


def _name(cls: type, /) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _annotation_owner(cls: type, name: str, /) -> type:
    return next(k for k in cls.__mro__ if name in inspect.get_annotations(k))


def _attribute_owner(cls: type, name: str, /) -> type | None:
    return next((k for k in cls.__mro__ if name in k.__dict__), None)


def test_no_module_field_is_shadowed_by_a_class_attribute():
    """A field shadowed by a property or method is never stored on instances.

    Equinox then flattens a missing value, and unflattening cannot restore it, so
    flatten -> unflatten -> flatten changes the tree structure.
    """
    phantom = []
    for cls in _modules():
        if not dataclasses.is_dataclass(cls):
            continue
        for field in dataclasses.fields(cls):
            declared = _annotation_owner(cls, field.name)
            owner = _attribute_owner(cls, field.name)
            if owner is None:
                continue
            value = owner.__dict__[field.name]
            shadowed_below = owner is not declared and declared in owner.__mro__
            defined_in_body = isinstance(value, _CLASS_BODY_CALLABLES)
            if shadowed_below or defined_in_body:
                phantom.append(f"{_name(cls)}.{field.name} ({_name(owner)})")
    assert not phantom, "\n".join(phantom)


def test_abstract_var_annotations_are_never_dataclass_fields():
    """Stringified `eqx.AbstractVar[...]` is silently concrete under equinox."""
    stringified = []
    for cls in _modules():
        for name, annotation in inspect.get_annotations(cls).items():
            if isinstance(annotation, str) and _STRINGIFIED_ABSTRACT.match(annotation):
                stringified.append(f"{_name(cls)}.{name}")
    assert not stringified, "\n".join(stringified)


def test_concrete_modules_implement_every_abstract_var():
    unresolved = []
    for cls in _modules():
        if cls.__name__.startswith(("Abstract", "_Abstract")):
            continue
        if getattr(cls, "_strict_is_abstract_", False):
            continue
        missing = set(getattr(cls, "__abstractvars__", ())) | set(
            getattr(cls, "__abstractclassvars__", ())
        )
        if missing:
            unresolved.append(f"{_name(cls)}: {sorted(missing)}")
    assert not unresolved, "\n".join(unresolved)


class AbstractIdentified(StrictModule):
    identifier: eqx.AbstractVar[str]


class FieldIdentified(AbstractIdentified):
    identifier: str


class PropertyIdentified(AbstractIdentified):
    @property
    def identifier(self) -> str:
        return "property"


def test_future_annotation_abstract_var_is_abstract_and_satisfiable():
    assert dataclasses.fields(AbstractIdentified) == ()
    assert AbstractIdentified.__abstractvars__ == frozenset({"identifier"})
    with pytest.raises(TypeError):
        AbstractIdentified()

    assert [field.name for field in dataclasses.fields(FieldIdentified)] == ["identifier"]
    assert FieldIdentified("field").identifier == "field"
    assert dataclasses.fields(PropertyIdentified) == ()
    assert PropertyIdentified().identifier == "property"


def _round_trip_modules():
    return (
        phx.domain.Interval1d(0.0, 1.0),
        phx.domain.HyperRectangle([0.0, -1.0], [1.0, 1.0]),
        phx.domain.GeometryDomain(
            phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
        ),
        phx.optim.ReducedAdjoint(),
        PropertyIdentified(),
    )


@pytest.mark.parametrize(
    "module", _round_trip_modules(), ids=lambda module: type(module).__name__
)
def test_flatten_unflatten_round_trip_keeps_tree_structure(module):
    leaves, treedef = jax.tree_util.tree_flatten(module)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jax.tree_util.tree_structure(rebuilt) == treedef
    assert eqx.tree_equal(rebuilt, module)
    # Non-field instance state is flattened by equinox as wrapper metadata.
    assert set(vars(rebuilt)) == set(vars(module))


class ScaledCall(StrictModule):
    scale: jax.Array

    def __call__(self, x):
        return self.scale * x


def test_strict_module_callable_composes_with_filter_jit_of_filter_vmap():
    module = ScaledCall(jnp.asarray(2.0))
    result = eqx.filter_jit(eqx.filter_vmap(module))(jnp.arange(3.0))
    assert jnp.array_equal(result, jnp.asarray([0.0, 2.0, 4.0]))


def test_strict_module_is_immutable_after_construction():
    module = ScaledCall(jnp.asarray(2.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        module.scale = jnp.asarray(3.0)
    with pytest.raises(AttributeError, match="Cannot delete"):
        del module.scale
    assert module.scale == 2.0
