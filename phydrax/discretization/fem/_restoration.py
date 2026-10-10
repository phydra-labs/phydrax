#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar

import jax
import numpy as np

from ..._strict import StrictModule


_Module = TypeVar("_Module", bound=StrictModule)


@dataclass
class _ValidationContext:
    # Strong references prevent object-ID reuse during this one closure replay.
    authenticated: dict[int, tuple[StrictModule, object, tuple[object, ...]]] = field(
        default_factory=dict
    )
    active: set[int] = field(default_factory=set)


_context: ContextVar[_ValidationContext | None] = ContextVar(
    "finite_element_restoration", default=None
)


@contextmanager
def finite_element_restoration_validation_scope() -> Iterator[None]:
    """Authenticate shared immutable nodes once within one full closure replay."""
    if _context.get() is not None:
        yield
        return
    token = _context.set(_ValidationContext())
    try:
        yield
    finally:
        _context.reset(token)


def _signature(node: StrictModule) -> tuple[object, tuple[object, ...], bool]:
    leaves, structure = jax.tree_util.tree_flatten(node)
    immutable = True
    identities: list[object] = []
    for value in leaves:
        if isinstance(value, jax.Array):
            identities.append((id(value), value.shape, str(value.dtype)))
        elif isinstance(value, np.ndarray):
            identities.append(
                (id(value), value.shape, value.dtype.str, bool(value.flags.writeable))
            )
            # Read-only host metadata is immutable within this closure replay and
            # may authenticate once just like a JAX array. Writable scientific
            # banks still undergo every owning byte replay.
            immutable = immutable and not value.flags.writeable
        else:
            identities.append(value)
    # The exact PyTree definition includes every declared static field and
    # scientific/source ID. Immutable JAX and read-only host leaf identity binds
    # the already authenticated bytes; no numerical payload is cached.
    return structure, tuple(identities), immutable


def authenticate_restored_node(
    function: Callable[[_Module], None],
) -> Callable[[_Module], None]:
    """Memoize only completed exact owning validation of one immutable DAG node."""

    @wraps(function)
    def validate(node: _Module) -> None:
        with finite_element_restoration_validation_scope():
            context = _context.get()
            if context is None:
                raise RuntimeError("Restoration authentication context is unavailable.")
            structure, identities, immutable = _signature(node)
            prior = context.authenticated.get(id(node))
            if (
                immutable
                and prior is not None
                and prior[0] is node
                and prior[1] == structure
                and prior[2] == identities
            ):
                return
            if id(node) in context.active:
                raise ValueError(
                    "Finite-element source restoration contains a cyclic owning graph."
                )
            context.active.add(id(node))
            try:
                function(node)
                if immutable:
                    context.authenticated[id(node)] = (node, structure, identities)
            finally:
                context.active.remove(id(node))

    return validate
