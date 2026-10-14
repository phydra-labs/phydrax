#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deferred package re-exports.

A package facade maps each re-exported name to its owner module. Owners load on
first access, so importing one owner never imports the whole package and its
import cycles, and importing the package never initializes a JAX backend.

`load_all` imports every owner of every facade loaded so far, until no new facade
appears. It reproduces the complete package import graph for consumers that
must observe every import-time registration, such as cold artifact restore.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import Any, Final

from ._jax_config import process_numerics


# Owner specifications are relative module names; `module:` marks a submodule
# that is itself the exported value.
_SUBMODULE: Final = "module:"
_FACADES: Final[dict[str, Mapping[str, str]]] = {}
_LOCK: Final = threading.Lock()


def load_module(name: str, package: str | None = None, /) -> ModuleType:
    """Import one deferred module under the process numerical settings.

    A module first loaded inside a scoped `phydrax.config.context` would
    otherwise fix its import-time constants under that scope for the process.
    """
    with process_numerics():
        return import_module(name, package)


def deferred_exports(
    package: str,
    namespace: dict[str, object],
    exports: Mapping[str, str],
    /,
    *,
    fallback: Callable[[str], object] | None = None,
    listing: Callable[[], list[str]] | None = None,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Return the module `__getattr__` and `__dir__` of one deferring facade.

    `fallback` resolves names the facade defers by its own rules; `listing`
    contributes its own names to `dir()`. Like every package facade, the
    returned `__getattr__` is a dynamic boundary: statically, re-exported names
    are typed by the facade's `TYPE_CHECKING` imports.
    """
    owners = MappingProxyType(dict(exports))
    with _LOCK:
        _FACADES[package] = owners

    def __getattr__(name: str) -> Any:
        owner = owners.get(name)
        if owner is None:
            if fallback is None:
                raise AttributeError(f"module {package!r} has no attribute {name!r}")
            with process_numerics():
                return fallback(name)
        # One scope covers the owner import and the lookup on it: an owner that is
        # itself a lazy facade loads its own owners during that lookup.
        with process_numerics():
            if owner.startswith(_SUBMODULE):
                value: object = import_module(owner.removeprefix(_SUBMODULE), package)
            else:
                value = getattr(import_module(owner, package), name)
            namespace[name] = value
        return value

    def __dir__() -> list[str]:
        names = set(namespace) if listing is None else set(listing())
        return sorted(names | owners.keys())

    return __getattr__, __dir__


def load_all() -> None:
    """Import every deferred owner of every loaded facade, to a fixed point."""
    loaded: set[str] = set()
    while True:
        with _LOCK:
            pending = sorted(
                (package, owners)
                for package, owners in _FACADES.items()
                if package not in loaded
            )
        if not pending:
            return
        for package, owners in pending:
            loaded.add(package)
            for owner in sorted(set(owners.values())):
                load_module(owner.removeprefix(_SUBMODULE), package)


__all__ = ["deferred_exports", "load_all", "load_module"]
