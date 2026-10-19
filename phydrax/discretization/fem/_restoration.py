#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from functools import update_wrapper
from typing import Protocol, TypeVar

import jax
import numpy as np

from ..._strict import StrictModule
from ...typing import checked


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


@dataclass
class RestorationValidationContext:
    """Authenticate immutable nodes for one explicitly owned closure replay."""

    # Strong references prevent object-ID reuse during this one closure replay.
    authenticated: dict[int, _AuthenticatedNode] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)


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
    ) -> None: ...


class _RestorationValidator(Protocol[_Module]):
    def __call__(
        self,
        node: _Module,
        /,
        *,
        validation_context: RestorationValidationContext | None = None,
    ) -> None: ...

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
    ) -> None:
        context = (
            RestorationValidationContext()
            if validation_context is None
            else validation_context
        )
        signature = _signature(node)
        prior = context.authenticated.get(id(node))
        if (
            signature.immutable
            and prior is not None
            and prior.node is node
            and prior.signature.structure == signature.structure
            and prior.signature.identities == signature.identities
        ):
            return
        if id(node) in context.active:
            raise ValueError(
                "finite_element.restoration_graph must be acyclic; got a recursive source node"
            )
        context.active.add(id(node))
        try:
            function(node, validation_context=context)
            if signature.immutable:
                context.authenticated[id(node)] = _AuthenticatedNode(node, signature)
        finally:
            context.active.remove(id(node))

    update_wrapper(validate, function)
    return validate
