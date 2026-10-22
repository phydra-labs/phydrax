#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from functools import update_wrapper
from inspect import signature as callable_signature
from typing import Callable, cast, overload, Protocol, TypeVar

import jax
import numpy as np

from ..._strict import StrictModule
from ...typing import checked
from .._coordinate_budget import CoordinateEnclosureBudget


_Module = TypeVar("_Module", bound=StrictModule, contravariant=True)


@dataclass(frozen=True, slots=True)
class _RestoredNodeSignature:
    structure: jax.tree_util.PyTreeDef
    identities: tuple[object, ...]
    immutable: bool


@dataclass(frozen=True, slots=True)
class _AuthenticatedNode:
    node: StrictModule
    signature: _RestoredNodeSignature
    budget: CoordinateEnclosureBudget | None


@dataclass
class RestorationValidationContext:
    """Authenticate immutable nodes for one explicitly owned closure replay."""

    # Strong references prevent object-ID reuse during this one closure replay.
    authenticated: dict[int, _AuthenticatedNode] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)
    budget: CoordinateEnclosureBudget | None = None


class _RestorationFunction(Protocol[_Module]):
    def __call__(
        self,
        node: _Module,
        /,
        *,
        validation_context: RestorationValidationContext,
    ) -> None: ...


class _BoundRestorationValidator(Protocol):
    def __call__(
        self,
        *,
        validation_context: RestorationValidationContext | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None: ...


class _RestorationValidator(Protocol[_Module]):
    def __call__(
        self,
        node: _Module,
        /,
        *,
        validation_context: RestorationValidationContext | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None: ...

    @overload
    def __get__(
        self, node: None, owner: type[_Module] | None = None, /
    ) -> _RestorationValidator[_Module]: ...

    @overload
    def __get__(
        self, node: _Module, owner: type[_Module] | None = None, /
    ) -> _BoundRestorationValidator: ...


def _signature(node: StrictModule) -> _RestoredNodeSignature:
    leaves, structure = jax.tree_util.tree_flatten(node)
    immutable = True
    identities: list[object] = []
    for value in leaves:
        match value:
            case jax.Array():
                identities.append((id(value), value.shape, str(value.dtype)))
            case np.ndarray():
                identities.append(
                    (id(value), value.shape, value.dtype.str, bool(value.flags.writeable))
                )
                immutable = immutable and not value.flags.writeable
            case _:
                identities.append(value)
    return _RestoredNodeSignature(structure, tuple(identities), immutable)


def authenticate_restored_node(
    function: _RestorationFunction[_Module],
) -> _RestorationValidator[_Module]:
    """Memoize completed validation within the caller's closure replay context."""

    @checked
    def validate(
        node: _Module,
        /,
        *,
        validation_context: RestorationValidationContext | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        context = (
            RestorationValidationContext()
            if validation_context is None
            else validation_context
        )
        ledger = context.budget if budget is None else budget
        signature = _signature(node)
        prior = context.authenticated.get(id(node))
        if (
            signature.immutable
            and prior is not None
            and prior.node is node
            and prior.signature.structure == signature.structure
            and prior.signature.identities == signature.identities
            and prior.budget is ledger
        ):
            return
        if id(node) in context.active:
            raise ValueError(
                "finite_element.restoration_graph must be acyclic; got a recursive source node"
            )
        context.active.add(id(node))
        previous = context.budget
        context.budget = ledger
        try:
            if "budget" in callable_signature(function).parameters:
                cast(Callable[..., None], function)(
                    node, validation_context=context, budget=ledger
                )
            else:
                function(node, validation_context=context)
            if signature.immutable:
                context.authenticated[id(node)] = _AuthenticatedNode(
                    node, signature, ledger
                )
        finally:
            context.budget = previous
            context.active.remove(id(node))

    update_wrapper(validate, function)
    return validate
