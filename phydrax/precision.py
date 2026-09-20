#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical precision formats, evidence, bounded rewriting, and selection."""

from importlib import import_module

from ._precision import __all__ as _format_all
from ._precision_rewrite import __all__ as _rewrite_all


_FACADE_EXPORT_MODULES = ("._precision", "._precision_rewrite")


def __getattr__(name: str):
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = list(_format_all)
__all__ += [name for name in _rewrite_all if name not in __all__]
