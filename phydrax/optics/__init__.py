#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable geometric, wave, transport, and guided-coupling optics."""

from importlib import import_module

from . import (
    beamlets as beamlets,
    geometric as geometric,
    materials as materials,
    sbs as sbs,
    transport as transport,
    wave as wave,
)
from .beamlets import __all__ as _beamlets_all
from .geometric import __all__ as _geometric_all
from .materials import __all__ as _materials_all
from .sbs import __all__ as _sbs_all
from .transport import __all__ as _transport_all
from .wave import __all__ as _wave_all


_FACADE_EXPORT_MODULES = (
    ".beamlets",
    ".geometric",
    ".materials",
    ".sbs",
    ".transport",
    ".wave",
)


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


__all__ = [
    "beamlets",
    "geometric",
    "materials",
    "sbs",
    "transport",
    "wave",
]
__all__ += [
    name
    for name in (
        *_beamlets_all,
        *_geometric_all,
        *_materials_all,
        *_sbs_all,
        *_transport_all,
        *_wave_all,
    )
    if name not in __all__
]
